# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online map, A* plan, and single-waypoint selection.

The map is empty at the start of every episode. The waypoint handed to the
high-level policy is one point on the current A* path, at most five maze cells
of arc length ahead, and at least one cell unless the goal itself is closer.
"""

from __future__ import annotations

import torch

from go2_lidar_maze.constants import CELL_PITCH_M, MAX_ASTAR_CELLS, MIN_ASTAR_CELLS
from go2_lidar_maze.nav_core.occupancy import OccupancyGrid
from go2_lidar_maze.nav_core.path_utils import cumulative_length, fillet_polyline, point_at_distance
from go2_lidar_maze.nav_core.planner import astar_grid, path_length_m

_PLAN_RES_M = 0.25
_INFLATE_CELLS = 1


def select_waypoint_on_path(
    path_xy: torch.Tensor,
    *,
    cell_m: float = CELL_PITCH_M,
    min_cells: int = MIN_ASTAR_CELLS,
    max_cells: int = MAX_ASTAR_CELLS,
) -> torch.Tensor:
    """Pick the waypoint on ``path_xy`` whose arc length is in ``[min, max]`` cells.

    If the whole path is shorter than ``min_cells``, the last point (the goal)
    is returned. That is the only case below one cell.

    Args:
        path_xy: Polyline from the robot to the goal ``(K, 2)`` [m].
        cell_m: Length of one maze cell [m].
        min_cells: Minimum A* length in cells.
        max_cells: Maximum A* length in cells.

    Returns:
        Waypoint ``(2,)`` [m].
    """
    if path_xy.shape[0] == 0:
        raise ValueError("select_waypoint_on_path received an empty path.")
    total = float(cumulative_length(path_xy)[-1])
    lo = min_cells * cell_m
    hi = max_cells * cell_m
    if total <= lo:
        return path_xy[-1]
    return point_at_distance(path_xy, min(total, hi))


class NavigationStack:
    """Per-environment occupancy map and A* waypoint selector."""

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.grid = OccupancyGrid(num_envs=num_envs, device=self.device, resolution=0.2, size=320)
        self._goal_xy = torch.zeros((num_envs, 2), device=self.device)
        self._age = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._have_plan = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: torch.Tensor, robot_xy: torch.Tensor, goal_xy: torch.Tensor) -> None:
        """Wipe the map and anchor it halfway between the robot and the goal."""
        anchor = 0.5 * (robot_xy + goal_xy)
        span = torch.linalg.norm(goal_xy - robot_xy, dim=-1)
        # Coarsen only when the goal plus a detour would fall off a 0.2 m grid.
        needed = span * 1.35 + 8.0
        res = (needed / float(self.grid.size)).clamp(min=0.2, max=0.4)
        self.grid.reset(env_ids, anchor, resolution=res)
        self._goal_xy[env_ids] = goal_xy
        self._age[env_ids] = 0
        self._have_plan[env_ids] = False

    @torch.no_grad()
    def update(self, ray_hits_w: torch.Tensor, sensor_pos_w: torch.Tensor) -> None:
        """Fold the latest LiDAR scan into the map."""
        self.grid.update(ray_hits_w, sensor_pos_w)
        self._age += 1

    def plan_waypoint(
        self,
        env_index: int,
        robot_xy: torch.Tensor,
        *,
        force: bool = False,
        replan_every: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return ``(waypoint_xy, smoothed_path_xy, astar_length_m)`` for one env.

        Replanning runs when ``force`` is set, on the first call, or every
        ``replan_every`` control steps (2 s at 10 Hz).
        """
        due = force or (not bool(self._have_plan[env_index])) or int(self._age[env_index]) >= replan_every
        if not due and hasattr(self, "_cached") and env_index in self._cached:
            return self._cached[env_index]
        waypoint, path, length = self._plan_one(env_index, robot_xy)
        if not hasattr(self, "_cached"):
            self._cached: dict[int, tuple[torch.Tensor, torch.Tensor, float]] = {}
        self._cached[env_index] = (waypoint, path, length)
        self._have_plan[env_index] = True
        self._age[env_index] = 0
        return waypoint, path, length

    def waypoint_invalid(self, env_index: int, robot_xy: torch.Tensor, waypoint_xy: torch.Tensor) -> bool:
        """True when ``waypoint_xy`` sits on a wall or its A* length exceeds 5 cells.

        A missing route also counts as invalid: newly seen walls have boxed the
        point in, so the caller should issue a fresh waypoint.
        """
        occupied, unknown, res, origin, fine_half, factor, plan_res = self._planning_grid(env_index)
        shape = occupied.shape
        cell = _world_to_coarse_cell(waypoint_xy, origin, res, fine_half, factor, shape)
        if bool(occupied[cell]):
            return True
        start = _world_to_coarse_cell(robot_xy, origin, res, fine_half, factor, shape)
        path = astar_grid(occupied, start, cell, unknown=unknown, resolution=plan_res)
        if path is None:
            return True
        return path_length_m(path, plan_res) > MAX_ASTAR_CELLS * CELL_PITCH_M

    def invalidate(self, env_index: int) -> None:
        """Force the next ``plan_waypoint`` call to search again."""
        self._have_plan[env_index] = False

    def needs_replan(self, env_index: int, replan_every: int = 20) -> bool:
        """True when the map has been updated for ``replan_every`` control steps."""
        return int(self._age[env_index]) >= replan_every

    def _plan_one(
        self, env_index: int, robot_xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        occupied, unknown, res, origin, fine_half, factor, plan_res = self._planning_grid(env_index)
        start = _world_to_coarse_cell(robot_xy, origin, res, fine_half, factor, occupied.shape)
        goal = _world_to_coarse_cell(self._goal_xy[env_index], origin, res, fine_half, factor, occupied.shape)
        path_cells = astar_grid(
            occupied,
            start,
            goal,
            unknown=unknown,
            resolution=plan_res,
        )
        if path_cells is None:
            goal_xy = self._goal_xy[env_index]
            path_xy = torch.stack((robot_xy, goal_xy), dim=0)
            return goal_xy, path_xy, float(torch.linalg.norm(goal_xy - robot_xy))
        path_xy = torch.stack(
            [
                _coarse_cell_to_world(cell, origin, res, fine_half, factor, robot_xy.device)
                for cell in path_cells
            ],
            dim=0,
        )
        path_xy = fillet_polyline(path_xy, radius=0.5)
        length = path_length_m(path_cells, plan_res)
        waypoint = select_waypoint_on_path(path_xy)
        return waypoint, path_xy, length

    def _planning_grid(self, env_index: int):
        occupied_fine = self.grid.obstacle_mask()[env_index]
        unknown_fine = ~(self.grid.seen_free_mask()[env_index] | occupied_fine)
        res = float(self.grid.res[env_index])
        occupied, unknown, factor = _coarsen(occupied_fine, unknown_fine, res, _PLAN_RES_M)
        occupied = _inflate(occupied, _INFLATE_CELLS)
        return (
            occupied,
            unknown,
            res,
            self.grid.origin[env_index],
            self.grid.size // 2,
            factor,
            res * factor,
        )


def _world_to_coarse_cell(
    xy: torch.Tensor,
    origin: torch.Tensor,
    fine_res: float,
    fine_half: int,
    factor: int,
    shape: tuple[int, int],
) -> tuple[int, int]:
    """Map world XY to a coarsened cell. Row is x, matching the occupancy grid."""
    rel = (xy - origin) / fine_res
    fine_row = int(torch.floor(rel[0]).item()) + fine_half
    fine_col = int(torch.floor(rel[1]).item()) + fine_half
    row = fine_row // factor
    col = fine_col // factor
    row = max(0, min(shape[0] - 1, row))
    col = max(0, min(shape[1] - 1, col))
    return row, col


def _coarse_cell_to_world(
    cell: tuple[int, int],
    origin: torch.Tensor,
    fine_res: float,
    fine_half: int,
    factor: int,
    device: torch.device,
) -> torch.Tensor:
    """World XY of the centre of a coarsened cell [m]."""
    fine = torch.tensor(
        [cell[0] * factor + (factor - 1) / 2.0, cell[1] * factor + (factor - 1) / 2.0],
        device=device,
        dtype=torch.float32,
    )
    return (fine - fine_half + 0.5) * fine_res + origin


def _coarsen(
    occupied: torch.Tensor, unknown: torch.Tensor, src_res: float, dst_res: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Max-pool the fine grid. Returns the integer pool factor (1 keeps the grid)."""
    if dst_res <= src_res * 1.01:
        return occupied, unknown, 1
    factor = max(1, int(round(dst_res / src_res)))
    edge = occupied.shape[0] - (occupied.shape[0] % factor)
    occ = occupied[:edge, :edge].reshape(edge // factor, factor, edge // factor, factor).any(dim=(1, 3))
    unk = unknown[:edge, :edge].reshape(edge // factor, factor, edge // factor, factor).all(dim=(1, 3))
    return occ, unk, factor


def _inflate(occupied: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return occupied
    kernel = 2 * radius + 1
    pooled = torch.nn.functional.max_pool2d(
        occupied.float().unsqueeze(0).unsqueeze(0),
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )
    return pooled.squeeze(0).squeeze(0) > 0.5
