# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One body-frame waypoint, issued from a privileged path or from the online planner.

The policy observes the waypoint and a few points further along the path. The
smoothed A* path is kept for the training reward and for those points. A
privileged episode ends when the robot enters the reach radius; the reset then
draws a new pose and a new waypoint. Online play holds the waypoint until the
radius, then issues the next one without resetting.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

from go2_lidar_maze.constants import (
    CELL_CENTER_ORIGIN_M,
    CELL_PITCH_M,
    GRID_SIZE,
    MAX_ASTAR_CELLS,
    REACH_RADIUS_M,
    SPAWN_JITTER_M,
)
from go2_lidar_maze.nav_core.path_utils import (
    cumulative_length,
    fillet_polyline,
    points_at_arc_length,
    resample_polyline,
)
from go2_lidar_maze.nav_core.stack import NavigationStack
from go2_lidar_maze.tasks.hl_maze.lidar_frame import ray_origin_w

_MAX_POINTS = 64
# Online paths run from the robot to the maze goal. Keeping only the first 15 m
# leaves the 12.5 m waypoint and the 4 m look-ahead on a 0.5 m sample grid.
_ONLINE_PREFIX_M = 15.0
# Base-frame box around this robot [m]. The trunk visual is x [-0.128, 0.332],
# y ±0.097, z [-0.097, 0.089]; the box also covers the hips and the feet.
# Returns inside it are the body, not a maze wall.
_SELF_HIT_MIN = (-0.20, -0.17, -0.50)
_SELF_HIT_MAX = (0.36, 0.17, 0.095)


def _without_self_hits(hits: torch.Tensor, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor) -> torch.Tensor:
    """Copy of ``hits`` with returns on this robot replaced by infinity.

    ``hits`` is ``(N, B, 3)`` [m]. The root pose is the base link. A point inside
    ``_SELF_HIT_MIN`` / ``_SELF_HIT_MAX`` in that frame is the body or a leg, so the
    occupancy grid must not mark it as a wall.
    """
    quat = root_quat_w.unsqueeze(1).expand(-1, hits.shape[1], -1)
    local = quat_apply_inverse(quat, hits - root_pos_w.unsqueeze(1))
    low = torch.tensor(_SELF_HIT_MIN, device=hits.device, dtype=hits.dtype)
    high = torch.tensor(_SELF_HIT_MAX, device=hits.device, dtype=hits.dtype)
    inside = torch.isfinite(local).all(dim=-1) & (local >= low).all(dim=-1) & (local <= high).all(dim=-1)
    return hits.masked_fill(inside.unsqueeze(-1), float("inf"))


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
        self.k_issued = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Outcome of the episode that just ended. Written before the resample
        # clears the counters, so an evaluator can read it after ``step``.
        self.last_k = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.last_reached = torch.zeros(self.num_envs, device=self.device)
        self.last_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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

    def _resample_command(self, env_ids: Sequence[int] | slice):
        # A full reset passes slice(None). The planner reset walks integer indices.
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        self._remember_outcome(ids)
        self.reached_count[ids] = 0.0
        self.issued_count[ids] = 0.0
        if self._stack is not None:
            robot_xy = self.robot.data.root_pos_w.torch[ids, :2]
            self._stack.reset(ids, robot_xy, self._env.goal_xy[ids])
        self._issue(ids)

    def _update_command(self):
        self.time_on_waypoint += self._env.step_dt
        if self.cfg.source != "online":
            # Privileged arrival is a termination. The reset samples the next waypoint.
            return
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
                # A waypoint already inside the reach radius would chain again
                # on the next step. Latch the arrival so that only happens once.
                still_there = self.distance_to_waypoint()[chain] < REACH_RADIUS_M
                self.bonus_paid[chain] |= still_there

    def _update_online(self):
        sensor = self._env.scene.sensors["lidar"]
        hits = _without_self_hits(
            sensor.data.ray_hits_w.torch,
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
        )
        self._stack.update(hits, ray_origin_w(sensor))
        robot_xy = self.robot.data.root_pos_w.torch[:, :2]
        headings = _headings(self)
        # One batched wall test. A route that stays blocked used to rebuild the
        # distance field for every robot on every control step, and the episode
        # never reached its summary print. Rebuild on the first scan, then at
        # the field interval.
        blocked = self._stack.blocked_routes(robot_xy, self.waypoint_w)
        due = self._stack.field_due()
        refresh = ~self._stack.field_ready | (blocked & due)
        latched = self.bonus_paid & (self._stack._age >= 10)
        repair = (~self.bonus_paid) & blocked & due
        self._stack.refresh_fields(refresh, robot_xy, headings)
        need = latched | repair
        if not bool(need.any()):
            return
        self._stack.plan_mask(need, robot_xy, headings)
        waypoint = self._stack.planned_xy
        waiting = torch.linalg.norm(waypoint - robot_xy, dim=-1) < REACH_RADIUS_M
        # The scan has not opened a route yet. Burning the waypoint budget
        # here ends the episode before the map exists.
        skip = need & latched & waiting
        apply = need & ~skip
        moved = torch.linalg.norm(waypoint - self.waypoint_w, dim=-1)
        # A wall that only grazes the route replans to the same point.
        # Resetting the timer there makes the waypoint thrash.
        reset = apply & (latched | (moved > 0.3))
        self.waypoint_w = torch.where(apply.unsqueeze(-1), waypoint, self.waypoint_w)
        self._write_online_paths(apply.nonzero(as_tuple=False).flatten())
        self.s_bar = torch.where(reset, torch.zeros_like(self.s_bar), self.s_bar)
        self.time_on_waypoint = torch.where(reset, torch.zeros_like(self.time_on_waypoint), self.time_on_waypoint)
        self.bonus_paid = self.bonus_paid & ~reset
        self.issued_count = self.issued_count + reset.to(self.issued_count.dtype)
        goal_dist = torch.linalg.norm(self.waypoint_w - self._env.goal_xy, dim=-1)
        self.is_final = torch.where(reset, goal_dist < 0.3, self.is_final)
        length = self.path_length.clamp(min=CELL_PITCH_M)
        budget = (length / 0.5 + 5.0).clamp(10.0, 30.0)
        self.time_budget = torch.where(reset, budget, self.time_budget)
        hold = (apply & waiting) | skip
        self.time_on_waypoint = torch.where(hold, torch.zeros_like(self.time_on_waypoint), self.time_on_waypoint)

    def _issue(self, env_ids: Sequence[int]):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten()
        if ids.numel() == 0:
            return
        self.k_issued[ids] = 0
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
            choice = int(torch.randint(0, 10_000, (1,)).item())
            _, path, length_k = _sample_pair(layout, start, self.k_max, choice, self.cfg.frontier_fraction)
            self.k_issued[index] = length_k
            cells_xy = [_cell_xy(cell) for cell in path]
            points = torch.tensor([robot_xy[index].tolist(), *cells_xy[1:]], device=self.device, dtype=torch.float32)
            jitter = (torch.rand(2, device=self.device) - 0.5) * (2.0 * SPAWN_JITTER_M)
            points[-1] = points[-1] + jitter
            self._store_path(index, points, is_final=False)

    def _issue_online(self, ids: torch.Tensor):
        robot_xy = self.robot.data.root_pos_w.torch[:, :2]
        goal_xy = self._env.goal_xy
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self._stack.field_ready[ids] = False
        self._stack.plan_mask(mask, robot_xy, _headings(self))
        self.waypoint_w[ids] = self._stack.planned_xy[ids]
        self._write_online_paths(ids)
        goal_dist = torch.linalg.norm(self.waypoint_w[ids] - goal_xy[ids], dim=-1)
        self.is_final[ids] = goal_dist < 0.3

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

    def _remember_outcome(self, ids: torch.Tensor) -> None:
        """Copy the ending state before this resample clears it."""
        self.last_k[ids] = self.k_issued[ids]
        self.last_reached[ids] = self.reached_count[ids]
        terms = self._env.termination_manager
        self.last_collision[ids] = terms.get_term("collision")[ids]
        timed_out = terms.get_term("time_out")[ids] | terms.get_term("waypoint_timeout")[ids]
        self.last_timeout[ids] = timed_out

    def _write_online_paths(self, ids: torch.Tensor) -> None:
        """Resample the first 15 m of each issued path onto the 64-point buffer.

        ``ids`` selects environments. The tail past each path's own sample count
        repeats that path's endpoint, matching :func:`resample_polyline`.
        """
        if ids.numel() == 0:
            return
        smooth = self._stack._smooth_xy[ids]
        total = cumulative_length(smooth)[:, -1]
        capped = torch.minimum(total, total.new_full(total.shape, _ONLINE_PREFIX_M))
        count = (capped / 0.5).to(torch.long) + 1
        count = count.clamp(min=2, max=_MAX_POINTS)
        slot = torch.arange(_MAX_POINTS, device=self.device)
        frac = slot.unsqueeze(0) / (count - 1).unsqueeze(1).to(torch.float32)
        dist = frac * capped.unsqueeze(1)
        dist = torch.where(slot.unsqueeze(0) < count.unsqueeze(1), dist, capped.unsqueeze(1))
        self.path_xy[ids] = points_at_arc_length(smooth, dist)
        steps = self.path_xy[ids, 1:] - self.path_xy[ids, :-1]
        self.path_length[ids] = torch.linalg.norm(steps, dim=-1).sum(dim=-1)

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
    frontier_fraction: float = 0.5
    """Probability of drawing an A* length of ``k_max``.

    The remaining probability is spread uniformly over lengths ``1`` to
    ``k_max - 1``. At ``k_max = 1`` every waypoint is one cell, and at
    ``k_max = 5`` half of the waypoints are the five-cell length used at
    deployment.
    """
    maze_usd: str = ""
    """Wall USD used to build the privileged graph. Ignored when ``source`` is online."""
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)


def _yaw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Yaw [rad] from an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _headings(command: WaypointCommand) -> torch.Tensor:
    """Planar heading of every robot, as a unit vector in world XY. Shape ``(N, 2)``."""
    yaw = _yaw(command.robot.data.root_quat_w.torch)
    return torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)


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


def _sample_pair(layout, start: int, k_max: int, choice: int, frontier_fraction: float = 0.5):
    """Sample an A* length in ``[1, k_max]`` that this cell can actually reach.

    ``k_max`` is tried first with probability ``frontier_fraction``. Otherwise
    a shorter length is tried first. A length the cell cannot reach falls
    through to the next candidate.

    Returns:
        ``(target, path, k)`` with ``k`` the length that was issued.
    """
    if k_max <= 1:
        order = [1]
    else:
        shorter = list(range(1, k_max))
        shift = choice % len(shorter)
        shorter = shorter[shift:] + shorter[:shift]
        draw = (choice % 10_000) / 10_000.0
        if draw < frontier_fraction:
            order = [k_max, *shorter]
        else:
            order = [*shorter, k_max]
    for k in order:
        if layout.has_distance(start, k):
            target, path = layout.sample(start, k, choice)
            return target, path, k
    raise RuntimeError(f"Cell {start} has no neighbour within {k_max} cells. The maze graph is disconnected.")
