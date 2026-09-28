# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One body-frame waypoint, issued from a privileged path or from the online planner.

The policy only observes the waypoint. The smoothed A* path is kept for the
training reward. A waypoint is held until the robot is within the reach radius,
then the next one is issued without resetting the episode.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm
from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg

from go2_lidar_maze.constants import (
    CELL_CENTER_ORIGIN_M,
    CELL_PITCH_M,
    GRID_SIZE,
    MAX_ASTAR_CELLS,
    REACH_RADIUS_M,
    SPAWN_JITTER_M,
)
from go2_lidar_maze.nav_core.path_utils import fillet_polyline, resample_polyline
from go2_lidar_maze.nav_core.stack import NavigationStack

_MAX_POINTS = 64


class WaypointCommand(CommandTerm):
    """Hold a single waypoint and the path that justifies it."""

    cfg: WaypointCommandCfg

    def __init__(self, cfg: WaypointCommandCfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.waypoint_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.path_xy = torch.zeros(self.num_envs, _MAX_POINTS, 2, device=self.device)
        self.s_bar = torch.zeros(self.num_envs, device=self.device)
        self.path_length = torch.zeros(self.num_envs, device=self.device)
        self.time_on_waypoint = torch.zeros(self.num_envs, device=self.device)
        self.time_budget = torch.full((self.num_envs,), 10.0, device=self.device)
        self.is_final = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bonus_paid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cross_track = torch.zeros(self.num_envs, device=self.device)
        self.reached_count = torch.zeros(self.num_envs, device=self.device)
        self.issued_count = torch.zeros(self.num_envs, device=self.device)
        self.k_max = 1
        self._layout = None
        self._stack: NavigationStack | None = None
        if cfg.source == "online":
            self._stack = NavigationStack(self.num_envs, self.device)
        self.metrics["waypoint_success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Waypoint in the robot body frame, divided by ``5 * cell`` and clipped to [-1, 1]."""
        return self.waypoint_b()

    def waypoint_b(self) -> torch.Tensor:
        """Body-frame waypoint ``(N, 2)``, scaled by the 5-cell lookahead [m]."""
        pos = self.robot.data.root_pos_w.torch[:, :2]
        quat = self.robot.data.root_quat_w.torch
        delta = self.waypoint_w - pos
        yaw = _yaw(quat)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        bx = cos_y * delta[:, 0] + sin_y * delta[:, 1]
        by = -sin_y * delta[:, 0] + cos_y * delta[:, 1]
        scale = MAX_ASTAR_CELLS * CELL_PITCH_M
        return torch.stack((bx, by), dim=-1).clamp(-scale, scale) / scale

    def distance_to_waypoint(self) -> torch.Tensor:
        """Planar distance from the robot to the active waypoint [m]."""
        pos = self.robot.data.root_pos_w.torch[:, :2]
        return torch.linalg.norm(self.waypoint_w - pos, dim=-1)

    def _update_metrics(self):
        issued = self.issued_count.clamp(min=1.0)
        self.metrics["waypoint_success"][:] = self.reached_count / issued

    def _resample_command(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        self.reached_count[ids] = 0.0
        self.issued_count[ids] = 0.0
        if self._stack is not None:
            robot_xy = self.robot.data.root_pos_w.torch[ids, :2]
            self._stack.reset(ids, robot_xy, self._env.goal_xy[ids])
        self._issue(ids)

    def _update_command(self):
        self.time_on_waypoint += self._env.step_dt
        if self.cfg.source == "online":
            self._update_online()
        reached = self.distance_to_waypoint() < REACH_RADIUS_M
        newly = reached & ~self.bonus_paid
        self.reached_count += newly.float()
        self.bonus_paid |= reached
        # Chain a new waypoint. The reward term reads ``bonus_paid`` on this
        # same step, before the next observation, so the bonus is not lost.
        ids = newly.nonzero(as_tuple=False).flatten()
        if len(ids) > 0 and not self.is_final[ids].all():
            # Do not chain past the maze goal. Those envs are terminating.
            chain = ids[~self.is_final[ids]]
            if len(chain) > 0:
                self._issue(chain)

    def _update_online(self):
        sensor = self._env.scene.sensors["lidar"]
        self._stack.update(sensor.data.ray_hits_w.torch, sensor.data.pos_w.torch)
        robot_xy = self.robot.data.root_pos_w.torch[:, :2]
        for index in range(self.num_envs):
            if bool(self.bonus_paid[index]):
                continue
            if self._stack.waypoint_invalid(index, robot_xy[index], self.waypoint_w[index]):
                self._stack.invalidate(index)
                self._issue([index])
                continue
            # Rebuild the path every 2 s. The waypoint timer resets only when
            # the selected point actually moves, so a stable waypoint cannot
            # be re-issued for another bonus.
            if self._stack.needs_replan(index):
                self._refresh_online(index, robot_xy[index])

    def _refresh_online(self, index: int, robot_xy: torch.Tensor):
        waypoint, path, _length = self._stack.plan_waypoint(index, robot_xy, force=False)
        moved = float(torch.linalg.norm(waypoint - self.waypoint_w[index])) > 0.25
        self.waypoint_w[index] = waypoint
        stored = resample_polyline(path, spacing=0.5, max_points=_MAX_POINTS)
        self._write_path(index, stored)
        goal_dist = torch.linalg.norm(waypoint - self._env.goal_xy[index])
        self.is_final[index] = bool(goal_dist < 0.3)
        if moved:
            self.s_bar[index] = 0.0
            self.time_on_waypoint[index] = 0.0
            self.bonus_paid[index] = False
            length = self.path_length[index].clamp(min=CELL_PITCH_M)
            self.time_budget[index] = (length / 0.5 + 5.0).clamp(10.0, 30.0)

    def _issue(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        if self.cfg.source == "privileged":
            self._issue_privileged(ids)
        else:
            self._issue_online(ids)
        self.s_bar[ids] = 0.0
        self.time_on_waypoint[ids] = 0.0
        self.bonus_paid[ids] = False
        self.issued_count[ids] += 1.0
        length = self.path_length[ids].clamp(min=CELL_PITCH_M)
        self.time_budget[ids] = (length / 0.5 + 5.0).clamp(10.0, 30.0)

    def _issue_privileged(self, ids: torch.Tensor):
        layout = self._layout_or_build()
        robot_xy = self.robot.data.root_pos_w.torch[:, :2]
        for index in ids.tolist():
            start = _nearest_cell(robot_xy[index])
            target, path = _sample_pair(layout, start, self.k_max, int(torch.randint(0, 10_000, (1,)).item()))
            cells_xy = [_cell_xy(cell) for cell in path]
            points = torch.tensor([robot_xy[index].tolist(), *cells_xy[1:]], device=self.device, dtype=torch.float32)
            jitter = (torch.rand(2, device=self.device) - 0.5) * (2.0 * SPAWN_JITTER_M)
            points[-1] = points[-1] + jitter
            self._store_path(index, points, is_final=False)

    def _issue_online(self, ids: torch.Tensor):
        robot_xy = self.robot.data.root_pos_w.torch[:, :2]
        goal_xy = self._env.goal_xy
        for index in ids.tolist():
            waypoint, path, _length = self._stack.plan_waypoint(index, robot_xy[index], force=True)
            self.waypoint_w[index] = waypoint
            stored = resample_polyline(path, spacing=0.5, max_points=_MAX_POINTS)
            self._write_path(index, stored)
            goal_dist = torch.linalg.norm(waypoint - goal_xy[index])
            self.is_final[index] = bool(goal_dist < 0.3)

    def _store_path(self, index: int, points: torch.Tensor, is_final: bool):
        smoothed = fillet_polyline(points, radius=0.5)
        stored = resample_polyline(smoothed, spacing=0.5, max_points=_MAX_POINTS)
        self.waypoint_w[index] = stored[-1]
        self._write_path(index, stored)
        self.is_final[index] = is_final

    def _write_path(self, index: int, stored: torch.Tensor):
        self.path_xy[index].zero_()
        count = stored.shape[0]
        self.path_xy[index, :count] = stored
        if count < _MAX_POINTS:
            self.path_xy[index, count:] = stored[-1]
        self.path_length[index] = torch.linalg.norm(stored[1:] - stored[:-1], dim=-1).sum()

    def _layout_or_build(self):
        if self._layout is None:
            from go2_lidar_maze.tasks.hl_maze.maze_layout import MazeLayout

            self._layout = MazeLayout.from_usd(self.cfg.maze_usd)
        return self._layout


@configclass
class WaypointCommandCfg(CommandTermCfg):
    """Configuration for :class:`WaypointCommand`."""

    class_type: type = WaypointCommand
    asset_name: str = "robot"
    source: str = "privileged"
    """``privileged`` during training, ``online`` at deployment."""
    maze_usd: str = ""
    """Wall USD used to build the privileged graph. Ignored when ``source`` is online."""
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)


def _yaw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Yaw [rad] from an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _nearest_cell(xy: torch.Tensor) -> int:
    col = int(torch.round((xy[0] - CELL_CENTER_ORIGIN_M) / CELL_PITCH_M).item())
    row = int(torch.round((xy[1] - CELL_CENTER_ORIGIN_M) / CELL_PITCH_M).item())
    col = max(0, min(GRID_SIZE - 1, col))
    row = max(0, min(GRID_SIZE - 1, row))
    return row * GRID_SIZE + col


def _cell_xy(index: int) -> tuple[float, float]:
    row, col = divmod(index, GRID_SIZE)
    return (
        CELL_CENTER_ORIGIN_M + CELL_PITCH_M * col,
        CELL_CENTER_ORIGIN_M + CELL_PITCH_M * row,
    )


def _sample_pair(layout, start: int, k_max: int, choice: int):
    """Sample an A* length in ``[1, k_max]`` that this cell can actually reach."""
    order = list(range(1, k_max + 1))
    # Rotate the preference so ``choice`` picks the distance, then fall back.
    shift = choice % len(order)
    order = order[shift:] + order[:shift]
    for k in order:
        if layout.has_distance(start, k):
            return layout.sample(start, k, choice)
    raise RuntimeError(f"Cell {start} has no neighbour within {k_max} cells. The maze graph is disconnected.")
