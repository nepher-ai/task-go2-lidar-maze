# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online map and a single waypoint on the confirmed corridor graph.

The map is empty at the start of every episode. LiDAR hits are walls and the
beams in front of them are free. A doorway counts as open only when that
opening is mapped free, so the route cannot cross a wall the scan has not
already disproved. The waypoint is a corridor-cell centre, one to five cells
ahead, which is the contract the high-level policy was trained on.
"""

from __future__ import annotations

import math
from collections import deque

import torch

from go2_lidar_maze.constants import (
    CELL_CENTER_ORIGIN_M,
    CELL_PITCH_M,
    GRID_SIZE,
    MAX_ASTAR_CELLS,
    MIN_ASTAR_CELLS,
)
from go2_lidar_maze.nav_core.occupancy import OccupancyGrid
from go2_lidar_maze.nav_core.path_utils import cumulative_length, fillet_polyline, point_at_distance, points_at_arc_length

_PLAN_RES_M = 0.25
_INFLATE_CELLS = 1
# 480 cells at 0.25 m is 120 m, enough for detours off the start-goal chord
# of the 50 m maze. A grid sized only to that chord leaves the far side unmapped.
_GRID_CELLS = 480
_GRID_RES_M = 0.25
_MIN_COVERAGE_M = 120.0
# Across the opening, inside the 2 m corridor, and through the 0.5 m wall slab.
# The scan hits the near face, not the middle of the slab, so the slab has to be sampled.
_DOOR_ACROSS_M = (-0.6, -0.3, 0.0, 0.3, 0.6)
_DOOR_THROUGH_M = (-0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45)
# Short of the wall face (1.0 m from the cell centre) so a look does not collide.
_PEEK_M = 0.75
# Five cells is the training maximum, but a 12 m carrot makes this policy
# sprint and clip the corridor. Two cells stays inside the speeds it survived.
_ONLINE_MAX_CELLS = 2
_UNKNOWN = 0
_FREE = 1
_WALL = 2
# Online routes are the robot plus at most two cell centres.
_MAX_ROUTE_POINTS = 8
# Sample spacing is one grid cell, and a two-cell route is under 8 m.
_ROUTE_CHECK_SAMPLES = 48


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
    """Per-environment occupancy map and waypoint selector."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        cell_origin_xy: torch.Tensor | None = None,
        cell_pitch: torch.Tensor | None = None,
        grid_n: torch.Tensor | None = None,
    ) -> None:
        """Build one occupancy map per environment.

        ``cell_origin_xy`` is the world XY [m] of cell ``(0, 0)``, shape ``(N, 2)``.
        ``cell_pitch`` is the corridor pitch [m], shape ``(N,)``. ``grid_n`` is the
        maze edge in cells, shape ``(N,)``. Omitted tensors use the 20×20 training
        maze (origin 1.5 m, pitch 2.5 m).
        """
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cell_origin_xy = _per_env_origin(num_envs, self.device, cell_origin_xy)
        self.cell_pitch = _per_env_pitch(num_envs, self.device, cell_pitch)
        self.grid_n = _per_env_n(num_envs, self.device, grid_n)
        self.grid = OccupancyGrid(
            num_envs=num_envs, device=self.device, resolution=_GRID_RES_M, size=_GRID_CELLS
        )
        self._goal_xy = torch.zeros((num_envs, 2), device=self.device)
        self._age = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._have_plan = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._cached: dict[int, tuple[torch.Tensor, torch.Tensor, float]] = {}
        self._route_xy = torch.zeros((num_envs, _MAX_ROUTE_POINTS, 2), device=self.device)
        self._route_n = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Unknown doors the robot has already stood in front of. Cleared on reset.
        self._rejected: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}

    def reset(self, env_ids: torch.Tensor, robot_xy: torch.Tensor, goal_xy: torch.Tensor) -> None:
        """Wipe the map and anchor it halfway between the robot and the goal."""
        anchor = 0.5 * (robot_xy + goal_xy)
        span = torch.linalg.norm(goal_xy - robot_xy, dim=-1)
        # The route wanders well off the start-goal chord. Cover that, not just the chord.
        needed = torch.maximum(span * 2.0 + 24.0, torch.full_like(span, _MIN_COVERAGE_M))
        res = (needed / float(self.grid.size)).clamp(min=_GRID_RES_M, max=0.35)
        self.grid.reset(env_ids, anchor, resolution=res)
        self._goal_xy[env_ids] = goal_xy
        self._age[env_ids] = 0
        self._have_plan[env_ids] = False
        self._route_n[env_ids] = 0
        for index in env_ids.tolist():
            self._cached.pop(int(index), None)
            self._rejected.pop(int(index), None)

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
        hold_xy: torch.Tensor | None = None,
        replan_every: int = 10,
        heading_xy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return ``(waypoint_xy, smoothed_path_xy, route_length_m)`` for one env.

        Replanning runs when ``force`` is set, on the first call, or every
        ``replan_every`` control steps. ``hold_xy`` is accepted for callers
        that still pass it; the issued point is always the current corridor target.
        """
        due = force or (not bool(self._have_plan[env_index])) or int(self._age[env_index]) >= replan_every
        if not due and env_index in self._cached:
            return self._cached[env_index]
        del hold_xy
        waypoint, path, length = self._plan_one(env_index, robot_xy, heading_xy)
        self._cached[env_index] = (waypoint, path, length)
        self._have_plan[env_index] = True
        self._age[env_index] = 0
        return waypoint, path, length

    def waypoint_invalid(self, env_index: int, robot_xy: torch.Tensor, waypoint_xy: torch.Tensor) -> bool:
        """True when the waypoint or the committed route has entered a wall.

        Samples next to the robot are ignored: inflation often covers the cell
        the body is standing in, and that must not count as a blocked route.
        """
        robots = robot_xy.new_zeros(self.num_envs, 2)
        waypoints = waypoint_xy.new_zeros(self.num_envs, 2)
        robots[env_index] = robot_xy
        waypoints[env_index] = waypoint_xy
        return bool(self.blocked_routes(robots, waypoints)[env_index])

    def blocked_routes(self, robot_xy: torch.Tensor, waypoint_xy: torch.Tensor) -> torch.Tensor:
        """Per-environment mask: the waypoint or the committed route sits on a mapped wall.

        ``robot_xy`` and ``waypoint_xy`` are ``(N, 2)`` [m]. Samples within
        0.45 m of the robot are ignored. An environment with no route is blocked.
        """
        walls = self.grid.counts >= self.grid.hit_threshold
        on_wall = self._walls_at(walls, waypoint_xy)
        missing = self._route_n <= 0
        spacing = self.grid.res.clamp(min=0.2)
        total = cumulative_length(self._route_xy)[:, -1]
        short = total < 1e-3
        steps = (total / spacing).floor().to(torch.long) + 1
        steps = steps.clamp(min=2, max=_ROUTE_CHECK_SAMPLES)
        slot = torch.arange(_ROUTE_CHECK_SAMPLES, device=self.device)
        frac = slot.view(1, -1) / (steps - 1).clamp(min=1).unsqueeze(1)
        dist = frac * total.unsqueeze(1)
        points = points_at_arc_length(self._route_xy, dist)
        near = torch.linalg.norm(points - robot_xy.unsqueeze(1), dim=-1) < 0.45
        keep = slot.view(1, -1) < steps.unsqueeze(1)
        hit = self._walls_at(walls, points) & keep & ~near & ~short.unsqueeze(1)
        return on_wall | missing | hit.any(dim=-1)

    def _walls_at(self, walls: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        """Wall mask sampled at world XY ``(N, 2)`` or ``(N, M, 2)`` [m]."""
        extra = xy.dim() - 2
        origin = self.grid.origin.view(self.num_envs, *([1] * extra), 2)
        res = self.grid.res.clamp(min=1e-6).view(self.num_envs, *([1] * extra), 1)
        rel = (xy - origin) / res
        cell = torch.floor(rel).to(torch.long) + self.grid._half
        cell = cell.clamp(0, self.grid.size - 1)
        env = torch.arange(self.num_envs, device=xy.device)
        if extra == 0:
            return walls[env, cell[:, 0], cell[:, 1]]
        env = env.view(-1, 1).expand(-1, xy.shape[1])
        return walls[env, cell[..., 0], cell[..., 1]]

    def invalidate(self, env_index: int) -> None:
        """Force the next ``plan_waypoint`` call to search again."""
        self._have_plan[env_index] = False

    def needs_replan(self, env_index: int, replan_every: int = 10) -> bool:
        """True when the map has been updated for ``replan_every`` control steps."""
        return int(self._age[env_index]) >= replan_every

    def _plan_one(
        self,
        env_index: int,
        robot_xy: torch.Tensor,
        heading_xy: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        rejected = self._rejected.setdefault(env_index, set())
        raw = _corridor_polyline(
            self.grid,
            env_index,
            robot_xy,
            self._goal_xy[env_index],
            rejected,
            heading_xy,
            origin_xy=self.cell_origin_xy[env_index],
            pitch=float(self.cell_pitch[env_index]),
            n=int(self.grid_n[env_index]),
        )
        self._store_route(env_index, raw)
        # Fillet matches privileged training. Endpoints stay on the cell centres.
        path_xy = fillet_polyline(raw, radius=0.5)
        waypoint = raw[-1]
        length = float(cumulative_length(path_xy)[-1])
        return waypoint, path_xy, length

    def _store_route(self, env_index: int, raw: torch.Tensor) -> None:
        """Keep the committed polyline for the batched wall check."""
        count = min(int(raw.shape[0]), _MAX_ROUTE_POINTS)
        self._route_xy[env_index] = raw[-1]
        self._route_xy[env_index, :count] = raw[:count]
        self._route_n[env_index] = count

    def _planning_grid(self, env_index: int):
        occupied_fine = self.grid.obstacle_mask()[env_index]
        unknown_fine = ~(self.grid.seen_free_mask()[env_index] | occupied_fine)
        res = float(self.grid.res[env_index])
        raw, unknown, factor = _coarsen(occupied_fine, unknown_fine, res, _PLAN_RES_M)
        occupied = _inflate(raw, _INFLATE_CELLS)
        return (
            occupied,
            unknown,
            raw,
            res,
            self.grid.origin[env_index],
            self.grid.size // 2,
            factor,
            res * factor,
        )


def _per_env_origin(num_envs: int, device: torch.device, value: torch.Tensor | None) -> torch.Tensor:
    """World XY [m] of cell ``(0, 0)`` for each environment, shape ``(N, 2)``."""
    if value is None:
        return torch.full((num_envs, 2), CELL_CENTER_ORIGIN_M, dtype=torch.float32, device=device)
    origin = value.to(device=device, dtype=torch.float32)
    if origin.shape != (num_envs, 2):
        raise ValueError(f"cell_origin_xy must be {(num_envs, 2)}, got {tuple(origin.shape)}.")
    return origin


def _per_env_pitch(num_envs: int, device: torch.device, value: torch.Tensor | None) -> torch.Tensor:
    """Corridor pitch [m] per environment, shape ``(N,)``."""
    if value is None:
        return torch.full((num_envs,), CELL_PITCH_M, dtype=torch.float32, device=device)
    pitch = value.to(device=device, dtype=torch.float32).reshape(-1)
    if pitch.shape != (num_envs,):
        raise ValueError(f"cell_pitch must be {(num_envs,)}, got {tuple(pitch.shape)}.")
    return pitch


def _per_env_n(num_envs: int, device: torch.device, value: torch.Tensor | None) -> torch.Tensor:
    """Maze edge length in cells per environment, shape ``(N,)``."""
    if value is None:
        return torch.full((num_envs,), GRID_SIZE, dtype=torch.long, device=device)
    n = value.to(device=device, dtype=torch.long).reshape(-1)
    if n.shape != (num_envs,):
        raise ValueError(f"grid_n must be {(num_envs,)}, got {tuple(n.shape)}.")
    return n


def _corridor_polyline(
    grid: OccupancyGrid,
    env_index: int,
    robot_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    rejected: set[tuple[tuple[int, int], tuple[int, int]]],
    heading_xy: torch.Tensor | None = None,
    *,
    origin_xy: torch.Tensor,
    pitch: float,
    n: int,
) -> torch.Tensor:
    """Cell-centre polyline from the robot toward the goal, at most two cells.

    Edges exist only where the doorway is mapped free. The target is the goal
    when that graph reaches it, otherwise the frontier whose known distance
    plus octile distance to the goal is smallest. A farther frontier wins a tie,
    so a hallway is followed to its end instead of stopping at the entrance.
    """
    horizontal, vertical = _door_states(grid, env_index, origin_xy, pitch, n)
    h_state = horizontal.tolist()
    v_state = vertical.tolist()
    # An empty map has no evidence either way. Probing toward the goal here
    # aims through the nearest wall. Wait until a scan has classified a door.
    if not any(value for row in h_state for value in row) and not any(value for row in v_state for value in row):
        return robot_xy.unsqueeze(0)
    start = _maze_cell(robot_xy, origin_xy, pitch, n)
    goal = _maze_cell(goal_xy, origin_xy, pitch, n)
    dist = [[-1] * n for _ in range(n)]
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    dist[start[0]][start[1]] = 0
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        row, col = queue.popleft()
        for nxt in _open_neighbors(row, col, h_state, v_state, n):
            if dist[nxt[0]][nxt[1]] >= 0:
                continue
            dist[nxt[0]][nxt[1]] = dist[row][col] + 1
            parent[nxt] = (row, col)
            queue.append(nxt)
    if dist[goal[0]][goal[1]] >= 0:
        target = goal
    else:
        # An unknown door on the cell we are standing in must not outrank a
        # corridor we can already walk. That stall is how the robot freezes.
        target = _best_frontier(
            dist, h_state, v_state, goal, parent, start, heading_xy, n, exclude=start
        )
        if target is None:
            return _peek_unknown(robot_xy, start, h_state, v_state, goal, rejected, origin_xy, pitch, n)
    cells = _reconstruct(parent, target, start, n)[: _ONLINE_MAX_CELLS + 1]
    # A two-cell step through a corner is a diagonal the body cannot take.
    # Stop on the corner cell; the next issue continues down the new hall.
    if len(cells) >= 3:
        first = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
        second = (cells[2][0] - cells[1][0], cells[2][1] - cells[1][1])
        if first != second:
            cells = cells[:2]
    centres = [_cell_center(row, col, robot_xy.device, origin_xy, pitch) for row, col in cells[1:]]
    if not centres:
        return robot_xy.unsqueeze(0)
    return torch.stack((robot_xy, *centres), dim=0)


def _door_states(
    grid: OccupancyGrid,
    env_index: int,
    origin_xy: torch.Tensor,
    pitch: float,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Doorway labels, ``0`` unknown, ``1`` open, ``2`` wall.

    ``horizontal[row, col]`` separates cells ``(row, col)`` and ``(row, col + 1)``.
    ``vertical[row, col]`` separates ``(row, col)`` and ``(row + 1, col)``.
    ``origin_xy`` is the world XY [m] of cell ``(0, 0)`` and ``pitch`` is the
    corridor pitch [m].
    """
    device = grid.counts.device
    ox = float(origin_xy[0])
    oy = float(origin_xy[1])
    rows = torch.arange(n, device=device)
    cols = torch.arange(n - 1, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    x = ox + pitch * (cc.to(torch.float32) + 0.5)
    y = oy + pitch * rr.to(torch.float32)
    horizontal = _classify_opening(grid, env_index, x, y, along_x=False)
    rows = torch.arange(n - 1, device=device)
    cols = torch.arange(n, device=device)
    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    x = ox + pitch * cc.to(torch.float32)
    y = oy + pitch * (rr.to(torch.float32) + 0.5)
    vertical = _classify_opening(grid, env_index, x, y, along_x=True)
    return horizontal, vertical


def _classify_opening(
    grid: OccupancyGrid,
    env_index: int,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    along_x: bool,
) -> torch.Tensor:
    """Label each doorway from samples across its opening."""
    across = torch.tensor(_DOOR_ACROSS_M, device=x.device, dtype=x.dtype)
    through = torch.tensor(_DOOR_THROUGH_M, device=x.device, dtype=x.dtype)
    span = (across.numel(), through.numel(), *x.shape)
    across_off = across.view(-1, 1, 1, 1)
    through_off = through.view(1, -1, 1, 1)
    # Dim 0 runs across the opening. Dim 1 runs through the wall slab.
    if along_x:
        xs = (x.view(1, 1, *x.shape) + across_off).expand(span)
        ys = (y.view(1, 1, *y.shape) + through_off).expand(span)
    else:
        xs = (x.view(1, 1, *x.shape) + through_off).expand(span)
        ys = (y.view(1, 1, *y.shape) + across_off).expand(span)
    classes = _point_class(grid, env_index, torch.stack((xs, ys), dim=-1))
    # One corner hit must not seal a 2 m opening. A real wall fills the slab.
    wall_count = classes.eq(_WALL).sum(dim=(0, 1))
    is_wall = wall_count >= 4
    free_count = classes.eq(_FREE).sum(dim=(0, 1))
    center_free = classes[len(_DOOR_ACROSS_M) // 2, len(_DOOR_THROUGH_M) // 2].eq(_FREE)
    state = torch.zeros(x.shape, dtype=torch.long, device=x.device)
    state = torch.where(center_free & ~is_wall & (free_count >= 3), torch.ones_like(state), state)
    return torch.where(is_wall, torch.full_like(state, _WALL), state)


def _point_class(grid: OccupancyGrid, env_index: int, pts: torch.Tensor) -> torch.Tensor:
    """``0`` unknown, ``1`` free, ``2`` wall for world points ``(..., 2)`` [m]."""
    origin = grid.origin[env_index]
    res = grid.res[env_index].clamp(min=1e-6)
    rel = (pts - origin) / res
    inside = rel.abs().amax(dim=-1) < float(grid._half)
    cell = torch.floor(rel).to(torch.long) + grid._half
    cell = cell.clamp(0, grid.size - 1)
    walls = grid.counts[env_index, cell[..., 0], cell[..., 1]] >= grid.hit_threshold
    free = grid.free_counts[env_index, cell[..., 0], cell[..., 1]] > 0
    classes = torch.zeros(pts.shape[:-1], dtype=torch.long, device=pts.device)
    classes = torch.where(free, torch.ones_like(classes), classes)
    classes = torch.where(walls, torch.full_like(classes, _WALL), classes)
    return torch.where(inside, classes, torch.zeros_like(classes))


def _maze_cell(xy: torch.Tensor, origin_xy: torch.Tensor, pitch: float, n: int) -> tuple[int, int]:
    """Nearest maze cell ``(row, col)`` of a world point. Row follows ``y``."""
    col = int(math.floor((float(xy[0]) - float(origin_xy[0])) / pitch + 0.5))
    row = int(math.floor((float(xy[1]) - float(origin_xy[1])) / pitch + 0.5))
    return (max(0, min(n - 1, row)), max(0, min(n - 1, col)))


def _cell_center(
    row: int, col: int, device: torch.device, origin_xy: torch.Tensor, pitch: float
) -> torch.Tensor:
    """World XY [m] of corridor cell ``(row, col)``."""
    return torch.tensor(
        (
            float(origin_xy[0]) + pitch * col,
            float(origin_xy[1]) + pitch * row,
        ),
        device=device,
        dtype=torch.float32,
    )


def _open_neighbors(
    row: int,
    col: int,
    horizontal: list[list[int]],
    vertical: list[list[int]],
    n: int,
) -> list[tuple[int, int]]:
    """Cells joined to ``(row, col)`` by an open doorway."""
    found: list[tuple[int, int]] = []
    if col + 1 < n and horizontal[row][col] == _FREE:
        found.append((row, col + 1))
    if col - 1 >= 0 and horizontal[row][col - 1] == _FREE:
        found.append((row, col - 1))
    if row + 1 < n and vertical[row][col] == _FREE:
        found.append((row + 1, col))
    if row - 1 >= 0 and vertical[row - 1][col] == _FREE:
        found.append((row - 1, col))
    return found


def _unknown_neighbors(
    row: int,
    col: int,
    horizontal: list[list[int]],
    vertical: list[list[int]],
    n: int,
) -> list[tuple[int, int]]:
    """In-grid neighbours whose doorway has not been classified."""
    found: list[tuple[int, int]] = []
    if col + 1 < n and horizontal[row][col] == _UNKNOWN:
        found.append((row, col + 1))
    if col - 1 >= 0 and horizontal[row][col - 1] == _UNKNOWN:
        found.append((row, col - 1))
    if row + 1 < n and vertical[row][col] == _UNKNOWN:
        found.append((row + 1, col))
    if row - 1 >= 0 and vertical[row - 1][col] == _UNKNOWN:
        found.append((row - 1, col))
    return found


def _best_frontier(
    dist: list[list[int]],
    horizontal: list[list[int]],
    vertical: list[list[int]],
    goal: tuple[int, int],
    parent: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    heading_xy: torch.Tensor | None,
    n: int,
    exclude: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Reached cell that minimises known steps plus octile steps to the goal.

    ``exclude`` is the robot's cell. An unknown door there is not a place we
    can walk yet, so it must not beat a frontier further along an open hall.
    A first step behind the robot costs four extra cells, so a side pocket
    does not win while the corridor ahead is still open.
    """
    best: tuple[int, int] | None = None
    best_key: tuple[float, int] | None = None
    for row in range(n):
        for col in range(n):
            if dist[row][col] < 0 or (row, col) == exclude:
                continue
            if not _unknown_neighbors(row, col, horizontal, vertical, n):
                continue
            score = dist[row][col] + _octile_cells(row, col, goal[0], goal[1])
            if _reverses(parent, (row, col), start, heading_xy, n):
                score += 4.0
            key = (score, -dist[row][col])
            if best_key is None or key < best_key:
                best_key = key
                best = (row, col)
    return best


def _reverses(
    parent: dict[tuple[int, int], tuple[int, int]],
    target: tuple[int, int],
    start: tuple[int, int],
    heading_xy: torch.Tensor | None,
    n: int,
) -> bool:
    """True when the first step from ``start`` toward ``target`` is behind the robot."""
    if heading_xy is None:
        return False
    cell = target
    guard = 0
    while parent.get(cell) != start and guard < n * n:
        nxt = parent.get(cell)
        if nxt is None:
            return False
        cell = nxt
        guard += 1
    if parent.get(cell) != start:
        return False
    drow = cell[0] - start[0]
    dcol = cell[1] - start[1]
    return dcol * float(heading_xy[0]) + drow * float(heading_xy[1]) < -0.3


def _peek_unknown(
    robot_xy: torch.Tensor,
    start: tuple[int, int],
    horizontal: list[list[int]],
    vertical: list[list[int]],
    goal: tuple[int, int],
    rejected: set[tuple[tuple[int, int], tuple[int, int]]],
    origin_xy: torch.Tensor,
    pitch: float,
    n: int,
) -> torch.Tensor:
    """Step toward an unknown door, short of the wall, to classify it.

    A door the robot is already standing in front of is remembered as rejected
    so the next call looks down a different opening.
    """
    candidates = [
        cell
        for cell in _unknown_neighbors(start[0], start[1], horizontal, vertical, n)
        if (start, cell) not in rejected
    ]
    while candidates:
        best = min(candidates, key=lambda cell: _octile_cells(cell[0], cell[1], goal[0], goal[1]))
        here = _cell_center(start[0], start[1], robot_xy.device, origin_xy, pitch)
        there = _cell_center(best[0], best[1], robot_xy.device, origin_xy, pitch)
        delta = there - here
        length = torch.linalg.norm(delta)
        if float(length) < 1e-3:
            rejected.add((start, best))
            candidates.remove(best)
            continue
        peek = here + delta / length * _PEEK_M
        if float(torch.linalg.norm(robot_xy - peek)) < 0.5:
            rejected.add((start, best))
            candidates.remove(best)
            continue
        return torch.stack((robot_xy, peek), dim=0)
    return robot_xy.unsqueeze(0)


def _reconstruct(
    parent: dict[tuple[int, int], tuple[int, int]],
    target: tuple[int, int],
    start: tuple[int, int],
    n: int,
) -> list[tuple[int, int]]:
    """Cells from ``start`` to ``target``, inclusive."""
    cells = [target]
    guard = 0
    while cells[-1] != start and guard < n * n:
        cells.append(parent[cells[-1]])
        guard += 1
    cells.reverse()
    return cells


def _octile_cells(row: int, col: int, goal_row: int, goal_col: int) -> float:
    """Octile distance in maze cells."""
    dr = abs(row - goal_row)
    dc = abs(col - goal_col)
    diag = min(dr, dc)
    return diag * math.sqrt(2.0) + (dr + dc - 2 * diag)


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
