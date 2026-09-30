# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online map and a waypoint on space the LiDAR has cleared.

The map is empty at the start of every episode. LiDAR hits are walls and the
beams in front of them are free. A distance field on that grid reaches the
goal without a maze lattice: inflated walls block, unseen cells cost more
than cells the scan has cleared, and the waypoint is the seen prefix of the
route, at most 5 m and about 1 m short of the first sharp corner. A point
that sits against a mapped wall is pulled back along that line.
"""

from __future__ import annotations

import math

import torch

from go2_lidar_maze.constants import CELL_PITCH_M, MAX_ASTAR_CELLS, MIN_ASTAR_CELLS
from go2_lidar_maze.nav_core.occupancy import OccupancyGrid
from go2_lidar_maze.nav_core.path_utils import (
    cumulative_length,
    fillet_polyline,
    point_at_distance,
    points_at_arc_length,
)
from go2_lidar_maze.nav_core.planner import (
    UNREACHABLE,
    _clearance_penalty,
    distance_field_batch,
    trace_route_batch,
)

_PLAN_RES_M = 0.25
_INFLATE_CELLS = 1
# 480 cells at 0.25 m is 120 m, enough for detours off the start-goal chord
# of the 50 m maze. A grid sized only to that chord leaves the far side unmapped.
_GRID_CELLS = 480
_GRID_RES_M = 0.25
_MIN_COVERAGE_M = 120.0
# Two training cells. A longer carrot makes this policy sprint into walls.
_LOOKAHEAD_M = 5.0
# Trace past the lookahead so the corner cut can see the next segment.
_TRACE_M = 8.0
# Grid chamfers are about 45 deg. A corridor corner is about 90 deg.
_TURN_RAD = math.radians(50.0)
_TURN_MIN_M = 1.0
# Stop this far before the corner vertex [m]. The vertex itself is the inside
# of the turn, and this policy drives into that wall.
_CORNER_BACKOFF_M = 1.0
_MIN_CARROT_M = 1.5
# Pull the endpoint back while it is this close to a mapped wall [m].
_WALL_CLEAR_M = 0.75
# A mapped corridor wins unless the unscanned shortcut is several times shorter.
_UNKNOWN_COST = 3.0
# One-time extra cost [m] for a first step behind the robot.
_REVERSE_PENALTY_M = 4.0
_CLEARANCE_RADIUS = 2
_CLEARANCE_WEIGHT = 1.0
_CROP_MARGIN_M = 6.0
# Window around the robot-goal chord [m]. It does not grow with the mapped
# region: a late scan covers ~120 m, and solving that whole grid is the step cost.
_MIN_CROP_M = 48.0
_MAX_FIELD_CELLS = 160
_FIELD_EVERY = 5
# Sweeps between convergence checks. The check is one scalar, so it can be
# fine enough that the robot-cell stop does not run a full extra window.
_FIELD_CHECK_EVERY = 8
_MAX_ROUTE_POINTS = 96
# Fillet samples can outgrow the wall-check buffer. The observation path keeps them.
_MAX_SMOOTH = 768
_ROUTE_CHECK_SAMPLES = 48
_SIMPLIFY_M = 0.5
_FOOTPRINT_M = 0.45


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

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        """Build one occupancy map per environment.

        The map is anchored from the robot and the goal at reset. It does not
        store a maze lattice.
        """
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.grid = OccupancyGrid(num_envs=num_envs, device=self.device, resolution=_GRID_RES_M, size=_GRID_CELLS)
        self._goal_xy = torch.zeros((num_envs, 2), device=self.device)
        self._age = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._have_plan = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._cached: dict[int, tuple[torch.Tensor, torch.Tensor, float]] = {}
        self._route_xy = torch.zeros((num_envs, _MAX_ROUTE_POINTS, 2), device=self.device)
        self._route_n = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._smooth_xy = torch.zeros((num_envs, _MAX_SMOOTH, 2), device=self.device)
        self._smooth_n = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.field_age = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.field_ready = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.planned_xy = torch.zeros((num_envs, 2), device=self.device)
        # Distance fields live in one buffer. ``_field_h`` / ``_field_w`` mark the
        # live crop; cells outside it stay unreachable.
        self._dist = torch.full((num_envs, _MAX_FIELD_CELLS, _MAX_FIELD_CELLS), UNREACHABLE, device=self.device)
        self._seen = torch.zeros((num_envs, _MAX_FIELD_CELLS, _MAX_FIELD_CELLS), dtype=torch.bool, device=self.device)
        self._field_h = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._field_w = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._field_r0 = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._field_c0 = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._field_factor = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self._field_res = torch.ones(num_envs, device=self.device)
        self._field_origin = torch.zeros((num_envs, 2), device=self.device)
        self._field_coarse_res = torch.ones(num_envs, device=self.device)
        self._paths: list[torch.Tensor | None] = [None] * num_envs

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
        self._smooth_n[env_ids] = 0
        self.field_age[env_ids] = 0
        self.field_ready[env_ids] = False
        self._field_h[env_ids] = 0
        self._field_w[env_ids] = 0
        for index in env_ids.tolist():
            self._cached.pop(int(index), None)
            self._paths[int(index)] = None

    @torch.no_grad()
    def update(self, ray_hits_w: torch.Tensor, sensor_pos_w: torch.Tensor) -> None:
        """Fold the latest LiDAR scan into the map."""
        self.grid.update(ray_hits_w, sensor_pos_w)
        self._age += 1
        self.field_age += 1

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
        that still pass it; the issued point lies on seen-free space toward the goal.
        """
        due = force or (not bool(self._have_plan[env_index])) or int(self._age[env_index]) >= replan_every
        if not due and env_index in self._cached:
            return self._cached[env_index]
        del hold_xy
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[env_index] = True
        robots = torch.zeros((self.num_envs, 2), device=self.device)
        robots[env_index] = robot_xy
        headings = None
        if heading_xy is not None:
            headings = torch.zeros((self.num_envs, 2), device=self.device)
            headings[env_index] = heading_xy
        if force:
            self.field_ready[env_index] = False
        self.plan_mask(mask, robots, headings)
        return self._cached[env_index]

    def path_of(self, index: int) -> torch.Tensor:
        """Last path issued for ``index``, shape ``(K, 2)`` [m]."""
        path = self._paths[index]
        if path is None:
            return self.planned_xy[index].unsqueeze(0)
        return path

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
        on_wall = self._counts_at(waypoint_xy) >= self.grid.hit_threshold
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
        near = torch.linalg.norm(points - robot_xy.unsqueeze(1), dim=-1) < _FOOTPRINT_M
        keep = slot.view(1, -1) < steps.unsqueeze(1)
        hit = (self._counts_at(points) >= self.grid.hit_threshold) & keep & ~near & ~short.unsqueeze(1)
        return on_wall | missing | hit.any(dim=-1)

    def _counts_at(self, xy: torch.Tensor) -> torch.Tensor:
        """Occupancy counts sampled at world XY ``(N, 2)`` or ``(N, M, 2)`` [m]."""
        extra = xy.dim() - 2
        origin = self.grid.origin.view(self.num_envs, *([1] * extra), 2)
        res = self.grid.res.clamp(min=1e-6).view(self.num_envs, *([1] * extra), 1)
        rel = (xy - origin) / res
        cell = torch.floor(rel).to(torch.long) + self.grid._half
        cell = cell.clamp(0, self.grid.size - 1)
        env = torch.arange(self.num_envs, device=xy.device)
        if extra == 0:
            return self.grid.counts[env, cell[:, 0], cell[:, 1]]
        env = env.view(-1, 1).expand(-1, xy.shape[1])
        return self.grid.counts[env, cell[..., 0], cell[..., 1]]

    def invalidate(self, env_index: int) -> None:
        """Force the next ``plan_waypoint`` call to search again."""
        self._have_plan[env_index] = False

    def needs_replan(self, env_index: int, replan_every: int = 10) -> bool:
        """True when the map has been updated for ``replan_every`` control steps."""
        return int(self._age[env_index]) >= replan_every

    def field_due(self) -> torch.Tensor:
        """True where the distance field is missing or older than the refresh interval."""
        return ~self.field_ready | (self.field_age >= _FIELD_EVERY)

    @torch.no_grad()
    def refresh_fields(
        self,
        mask: torch.Tensor,
        robot_xy: torch.Tensor,
        heading_xy: torch.Tensor | None = None,
    ) -> None:
        """Recompute cached distance fields for environments in ``mask``.

        ``robot_xy`` is ``(N, 2)`` [m]. ``heading_xy`` is an optional planar
        heading ``(N, 2)``. An environment with no free cells keeps an empty
        field and stays not ready, so the next scan is picked up immediately.
        """
        if not bool(mask.any()):
            return
        ids = mask.nonzero(as_tuple=False).flatten()
        known_lo, known_hi, has_free = self.grid.known_bounds(ids)
        empty = ids[~has_free]
        if empty.numel() > 0:
            self.field_ready[empty] = False
            self._field_h[empty] = 0
            self._field_w[empty] = 0
        live = ids[has_free]
        if live.numel() > 0:
            heading = None if heading_xy is None else heading_xy[live]
            self._refresh_live(live, robot_xy[live], heading, known_lo[has_free], known_hi[has_free])
        self.field_age[mask] = 0

    @torch.no_grad()
    def plan_mask(
        self,
        mask: torch.Tensor,
        robot_xy: torch.Tensor,
        heading_xy: torch.Tensor | None = None,
    ) -> None:
        """Issue waypoints for ``mask`` from the cached fields.

        A field is rebuilt when it is missing, older than ``_FIELD_EVERY``
        scans, or no longer contains the robot. Results land in
        :attr:`planned_xy` and :meth:`path_of`.
        """
        if not bool(mask.any()):
            return
        cover = self._covers(robot_xy)
        stale = mask & (~self.field_ready | (self.field_age >= _FIELD_EVERY) | ~cover)
        self.refresh_fields(stale, robot_xy, heading_xy)
        self._trace_mask(mask, robot_xy)
        self._age[mask] = 0
        self._have_plan[mask] = True

    def _refresh_live(
        self,
        ids: torch.Tensor,
        robot_xy: torch.Tensor,
        heading_xy: torch.Tensor | None,
        known_lo: torch.Tensor,
        known_hi: torch.Tensor,
    ) -> None:
        """Build and solve one distance field for every live environment in ``ids``.

        ``known_lo`` and ``known_hi`` are inclusive cell bounds ``(K, 2)`` of
        mapped walls and free space. ``known_hi < 0`` means that axis is empty.
        """
        res = self.grid.res[ids]
        origin = self.grid.origin[ids]
        goal_xy = self._goal_xy[ids]
        r0, c0, r1, c1, factor, robot_rc, goal_rc = _crop_bounds(
            robot_xy, goal_xy, origin, res, self.grid._half, self.grid.size, known_lo, known_hi
        )
        groups = []
        for value in factor.unique().tolist():
            pick = factor == int(value)
            groups.append(
                self._build_group(
                    ids[pick],
                    r0[pick],
                    c0[pick],
                    (r1 - r0)[pick],
                    (c1 - c0)[pick],
                    int(value),
                    robot_rc[pick],
                    goal_rc[pick],
                    res[pick],
                    origin[pick],
                    robot_xy[pick],
                    None if heading_xy is None else heading_xy[pick],
                )
            )
        self._solve_and_store(groups)

    def _build_group(
        self,
        ids: torch.Tensor,
        r0: torch.Tensor,
        c0: torch.Tensor,
        height: torch.Tensor,
        width: torch.Tensor,
        factor: int,
        robot_rc: torch.Tensor,
        goal_rc: torch.Tensor,
        res: torch.Tensor,
        origin: torch.Tensor,
        robot_xy: torch.Tensor,
        heading_xy: torch.Tensor | None,
    ) -> dict:
        """Coarse cost crop for every env in ``ids``. ``factor`` is shared.

        Cells past an environment's own crop stay free while walls are inflated
        and the clearance penalty is measured, then become walls. That matches
        a crop solved on its own, whose border is the end of the tensor.
        """
        count = int(ids.shape[0])
        hc = torch.div(height + factor - 1, factor, rounding_mode="floor")
        wc = torch.div(width + factor - 1, factor, rounding_mode="floor")
        h_coarse = int(hc.max())
        w_coarse = int(wc.max())
        hpad = h_coarse * factor
        wpad = w_coarse * factor
        device = self.device
        rows = torch.arange(hpad, device=device)
        cols = torch.arange(wpad, device=device)
        rr = r0.view(count, 1, 1) + rows.view(1, -1, 1)
        cc = c0.view(count, 1, 1) + cols.view(1, 1, -1)
        in_crop = (rows.view(1, -1, 1) < height.view(count, 1, 1)) & (cols.view(1, 1, -1) < width.view(count, 1, 1))
        in_pad = (rows.view(1, -1, 1) < (hc * factor).view(count, 1, 1)) & (
            cols.view(1, 1, -1) < (wc * factor).view(count, 1, 1)
        )
        in_pad = in_pad & ~in_crop
        env = ids.view(count, 1, 1).expand(count, hpad, wpad)
        rr_c = rr.clamp(0, self.grid.size - 1)
        cc_c = cc.clamp(0, self.grid.size - 1)
        counts = self.grid.counts[env, rr_c, cc_c]
        free = self.grid.free_counts[env, rr_c, cc_c]
        hit = counts >= self.grid.hit_threshold
        occupied = (hit & in_crop) | in_pad
        unseen = ((free <= 0) & ~hit & in_crop) | in_pad
        occupied = occupied.view(count, h_coarse, factor, w_coarse, factor).any(dim=(2, 4))
        unseen = unseen.view(count, h_coarse, factor, w_coarse, factor).all(dim=(2, 4))
        valid = (torch.arange(h_coarse, device=device).view(1, -1, 1) < hc.view(count, 1, 1)) & (
            torch.arange(w_coarse, device=device).view(1, 1, -1) < wc.view(count, 1, 1)
        )
        local_r = torch.div(robot_rc[:, 0] - r0, factor, rounding_mode="floor")
        local_c = torch.div(robot_rc[:, 1] - c0, factor, rounding_mode="floor")
        goal_r = torch.div(goal_rc[:, 0] - r0, factor, rounding_mode="floor")
        goal_c = torch.div(goal_rc[:, 1] - c0, factor, rounding_mode="floor")
        hc_safe = hc.clamp(min=1)
        wc_safe = wc.clamp(min=1)
        local_r = torch.minimum(local_r.clamp(min=0), hc_safe - 1)
        local_c = torch.minimum(local_c.clamp(min=0), wc_safe - 1)
        goal_r = torch.minimum(goal_r.clamp(min=0), hc_safe - 1)
        goal_c = torch.minimum(goal_c.clamp(min=0), wc_safe - 1)
        coarse_res = res * float(factor)
        inflated = _inflate(occupied, _INFLATE_CELLS)
        foot = _footprint_batch(h_coarse, w_coarse, local_r, local_c, coarse_res)
        inflated = inflated & ~(foot & ~occupied)
        inflated = torch.where(valid, inflated, torch.zeros_like(inflated))
        penalty = _clearance_penalty(inflated, _CLEARANCE_RADIUS, _CLEARANCE_WEIGHT)
        step = 1.0 + penalty
        step = torch.where(unseen & ~occupied & valid, step * _UNKNOWN_COST, step)
        step = _reverse_cost_batch(step, robot_xy, heading_xy, origin, res, self.grid._half, r0, c0, factor, coarse_res)
        step = torch.where(valid, step, torch.ones_like(step))
        inflated = torch.where(valid, inflated, torch.ones_like(inflated))
        seen = ((~unseen & ~occupied) | (foot & ~occupied)) & valid
        return {
            "ids": ids,
            "occupied": inflated,
            "step": step,
            "seen": seen,
            "goal": torch.stack((goal_r, goal_c), dim=-1),
            "robot": torch.stack((local_r, local_c), dim=-1),
            "coarse_res": coarse_res,
            "hc": hc,
            "wc": wc,
            "r0": r0,
            "c0": c0,
            "factor": factor,
            "res": res,
            "origin": origin,
        }

    def _solve_and_store(self, groups: list[dict]) -> None:
        """One field over every group, so the early-stop cost share matches a joint batch."""
        max_h = max(int(group["occupied"].shape[1]) for group in groups)
        max_w = max(int(group["occupied"].shape[2]) for group in groups)
        occupied_parts = []
        step_parts = []
        goal_parts = []
        robot_parts = []
        res_parts = []
        for group in groups:
            count, height, width = group["occupied"].shape
            occupied = group["occupied"].new_ones((count, max_h, max_w))
            step = group["step"].new_ones((count, max_h, max_w))
            occupied[:, :height, :width] = group["occupied"]
            step[:, :height, :width] = group["step"]
            occupied_parts.append(occupied)
            step_parts.append(step)
            goal_parts.append(group["goal"])
            robot_parts.append(group["robot"])
            res_parts.append(group["coarse_res"])
        dist = distance_field_batch(
            torch.cat(occupied_parts, dim=0),
            torch.cat(goal_parts, dim=0),
            torch.cat(res_parts, dim=0),
            step_cost=torch.cat(step_parts, dim=0),
            robot=torch.cat(robot_parts, dim=0),
            max_iters=min(1024, (max_h + max_w) * 2),
            check_every=_FIELD_CHECK_EVERY,
        )
        cursor = 0
        for group in groups:
            count, height, width = group["occupied"].shape
            self._write_field(group, dist[cursor : cursor + count, :height, :width])
            cursor += count

    def _write_field(self, group: dict, dist: torch.Tensor) -> None:
        ids = group["ids"]
        height, width = dist.shape[-2], dist.shape[-1]
        self._dist[ids] = UNREACHABLE
        self._seen[ids] = False
        self._dist[ids, :height, :width] = dist
        self._seen[ids, :height, :width] = group["seen"]
        self._field_h[ids] = group["hc"]
        self._field_w[ids] = group["wc"]
        self._field_r0[ids] = group["r0"]
        self._field_c0[ids] = group["c0"]
        self._field_factor[ids] = group["factor"]
        self._field_res[ids] = group["res"]
        self._field_origin[ids] = group["origin"]
        self._field_coarse_res[ids] = group["coarse_res"]
        self.field_ready[ids] = True

    def _covers(self, robot_xy: torch.Tensor) -> torch.Tensor:
        """True where the cached field contains ``robot_xy`` ``(N, 2)`` [m]."""
        row, col = self._local_ids(robot_xy, None, clamp=False)
        inside = (row >= 0) & (col >= 0) & (row < self._field_h) & (col < self._field_w)
        return self.field_ready & inside

    def _local_ids(
        self, xy: torch.Tensor, ids: torch.Tensor | None, *, clamp: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coarse ``(row, col)`` of world XY. ``ids`` None uses every environment."""
        if ids is None:
            res = self._field_res
            origin = self._field_origin
            r0, c0 = self._field_r0, self._field_c0
            factor = self._field_factor
            height, width = self._field_h, self._field_w
        else:
            res = self._field_res[ids]
            origin = self._field_origin[ids]
            r0, c0 = self._field_r0[ids], self._field_c0[ids]
            factor = self._field_factor[ids]
            height, width = self._field_h[ids], self._field_w[ids]
        rel = (xy - origin) / res.clamp(min=1e-6).unsqueeze(-1)
        cell = torch.floor(rel).to(torch.long) + self.grid._half
        cell = cell.clamp(0, self.grid.size - 1)
        factor = factor.clamp(min=1)
        local_r = torch.div(cell[:, 0] - r0, factor, rounding_mode="floor")
        local_c = torch.div(cell[:, 1] - c0, factor, rounding_mode="floor")
        if clamp:
            local_r = torch.minimum(local_r.clamp(min=0), height.clamp(min=1) - 1)
            local_c = torch.minimum(local_c.clamp(min=0), width.clamp(min=1) - 1)
        return local_r, local_c

    def _trace_mask(self, mask: torch.Tensor, robot_xy: torch.Tensor) -> None:
        ids = mask.nonzero(as_tuple=False).flatten()
        ready = self.field_ready[ids]
        waiting = ids[~ready]
        if waiting.numel() > 0:
            self._store_paths(waiting, robot_xy[waiting], robot_xy[waiting].unsqueeze(1), None)
        live = ids[ready]
        if live.numel() == 0:
            return
        height = int(self._field_h[live].max())
        width = int(self._field_w[live].max())
        dist = self._dist[live, :height, :width]
        row, col = self._local_ids(robot_xy[live], live, clamp=True)
        starts = torch.stack((row, col), dim=-1)
        cells, counts = trace_route_batch(
            dist,
            starts,
            resolution=self._field_coarse_res[live],
            max_length_m=_TRACE_M,
            max_steps=64,
        )
        self._commit_batch(live, robot_xy[live], cells, counts)

    def _commit_batch(
        self,
        ids: torch.Tensor,
        robot_xy: torch.Tensor,
        cells: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """Turn traced cells into waypoints. Polyline edits run on one CPU copy."""
        kept = _seen_counts(cells, self._seen[ids], self._field_h[ids], self._field_w[ids], counts)
        world = _cells_to_world_batch(
            cells,
            self._field_r0[ids],
            self._field_c0[ids],
            self._field_factor[ids],
            self._field_res[ids],
            self._field_origin[ids],
            self.grid._half,
        )
        world = world.clone()
        world[:, 0] = robot_xy
        walls, win_r0, win_c0 = self._wall_windows(ids, world, kept)
        meta = torch.cat(
            (
                self.grid.res[ids].unsqueeze(1),
                self.grid.origin[ids],
                win_r0.unsqueeze(1).to(torch.float32),
                win_c0.unsqueeze(1).to(torch.float32),
                kept.unsqueeze(1).to(torch.float32),
            ),
            dim=1,
        )
        world_cpu, walls_cpu, meta_cpu = _to_cpu(world, walls, meta)
        count = int(ids.shape[0])
        planned = torch.zeros((count, 2), dtype=torch.float32)
        smooth = torch.zeros((count, _MAX_SMOOTH, 2), dtype=torch.float32)
        routes = torch.zeros((count, _MAX_ROUTE_POINTS, 2), dtype=torch.float32)
        smooth_n = torch.ones(count, dtype=torch.long)
        route_n = torch.ones(count, dtype=torch.long)
        lengths = torch.zeros(count, dtype=torch.float32)
        half = int(self.grid._half)
        size = int(self.grid.size)
        for slot in range(count):
            row = meta_cpu[slot]
            res = float(row[0])
            origin = (float(row[1]), float(row[2]))
            win_row = int(row[3])
            win_col = int(row[4])
            keep = int(row[5])
            robot = world_cpu[slot, 0]
            if keep <= 1:
                planned[slot] = robot
                _fill_polyline(smooth, slot, robot)
                _fill_polyline(routes, slot, robot)
                continue
            path = world_cpu[slot, :keep].clone()
            path[0] = robot
            path = _shape_route(path, walls_cpu[slot], win_row, win_col, res, origin, half, size)
            if path.shape[0] < 2 or float(torch.linalg.norm(path[-1] - path[0])) < 0.05:
                planned[slot] = robot
                _fill_polyline(smooth, slot, robot)
                _fill_polyline(routes, slot, robot)
                continue
            stored = fillet_polyline(path, radius=0.5) if path.shape[0] >= 3 else path
            if stored.shape[0] > _MAX_SMOOTH:
                stored = torch.cat((stored[: _MAX_SMOOTH - 1], stored[-1:]), dim=0)
            planned[slot] = path[-1]
            taken = int(stored.shape[0])
            _fill_polyline(smooth, slot, stored[-1])
            smooth[slot, :taken] = stored
            smooth_n[slot] = taken
            route_count = min(taken, _MAX_ROUTE_POINTS)
            _fill_polyline(routes, slot, stored[-1])
            routes[slot, :route_count] = stored[:route_count]
            route_n[slot] = route_count
            if taken >= 2:
                lengths[slot] = float(cumulative_length(stored)[-1])
        packed = torch.cat(
            (
                planned,
                lengths.unsqueeze(1),
                smooth_n.unsqueeze(1).to(torch.float32),
                route_n.unsqueeze(1).to(torch.float32),
                smooth.reshape(count, -1),
                routes.reshape(count, -1),
            ),
            dim=1,
        ).to(self.device)
        self._scatter_commit(ids, packed, smooth_n, lengths)

    def _scatter_commit(
        self,
        ids: torch.Tensor,
        packed: torch.Tensor,
        smooth_n_cpu: torch.Tensor,
        lengths_cpu: torch.Tensor,
    ) -> None:
        count = int(ids.shape[0])
        planned = packed[:, :2]
        smooth_n = packed[:, 3].to(torch.long)
        route_n = packed[:, 4].to(torch.long)
        cursor = 5
        smooth = packed[:, cursor : cursor + _MAX_SMOOTH * 2].reshape(count, _MAX_SMOOTH, 2)
        cursor += _MAX_SMOOTH * 2
        routes = packed[:, cursor : cursor + _MAX_ROUTE_POINTS * 2].reshape(count, _MAX_ROUTE_POINTS, 2)
        self.planned_xy[ids] = planned
        self._smooth_xy[ids] = smooth
        self._smooth_n[ids] = smooth_n
        self._route_xy[ids] = routes
        self._route_n[ids] = route_n
        id_list = ids.tolist()
        length_list = lengths_cpu.tolist()
        for slot, index in enumerate(id_list):
            taken = int(smooth_n_cpu[slot])
            path = smooth[slot, :taken].clone()
            self._paths[int(index)] = path
            self._cached[int(index)] = (planned[slot], path, float(length_list[slot]))

    def _store_paths(
        self,
        ids: torch.Tensor,
        waypoint: torch.Tensor,
        points: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> None:
        """Store a one-point wait, or a polyline already shaped ``(K, T, 2)``."""
        if points.dim() == 2:
            points = points.unsqueeze(1)
        count = int(ids.shape[0])
        taken = torch.ones(count, dtype=torch.long, device=self.device)
        point = waypoint.unsqueeze(1)
        self.planned_xy[ids] = waypoint
        self._smooth_xy[ids] = point.expand(-1, _MAX_SMOOTH, -1)
        self._smooth_n[ids] = taken
        self._route_xy[ids] = point.expand(-1, _MAX_ROUTE_POINTS, -1)
        self._route_n[ids] = taken
        if lengths is None:
            length_list = [0.0] * count
        else:
            length_list = lengths.detach().tolist()
        for slot, index in enumerate(ids.tolist()):
            path = waypoint[slot].unsqueeze(0)
            self._paths[int(index)] = path
            self._cached[int(index)] = (waypoint[slot], path, float(length_list[slot]))

    def _wall_windows(
        self, ids: torch.Tensor, world: torch.Tensor, kept: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Wall mask around each route, large enough for the pull-back search."""
        res = self.grid.res[ids]
        origin = self.grid.origin[ids]
        steps = torch.arange(world.shape[1], device=self.device).view(1, -1)
        valid = steps < kept.clamp(min=1).unsqueeze(1)
        picked = torch.where(valid.unsqueeze(-1), world, world[:, :1, :])
        rad = torch.ceil(_WALL_CLEAR_M / res.clamp(min=1e-6)).to(torch.long) + 1
        margin = (rad.to(torch.float32) + 2.0) * res
        lo = picked.amin(dim=1) - margin.unsqueeze(1)
        hi = picked.amax(dim=1) + margin.unsqueeze(1)
        half = self.grid._half
        size = self.grid.size

        def to_cell(xy: torch.Tensor) -> torch.Tensor:
            rel = (xy - origin) / res.clamp(min=1e-6).unsqueeze(1)
            return torch.floor(rel).to(torch.long) + half

        clo = to_cell(lo)
        chi = to_cell(hi)
        r0 = clo[:, 0].clamp(0, size - 1)
        c0 = clo[:, 1].clamp(0, size - 1)
        r1 = torch.maximum((chi[:, 0] + 1).clamp(max=size), r0 + 1)
        c1 = torch.maximum((chi[:, 1] + 1).clamp(max=size), c0 + 1)
        height = int((r1 - r0).max())
        width = int((c1 - c0).max())
        count = int(ids.shape[0])
        rows = torch.arange(height, device=self.device)
        cols = torch.arange(width, device=self.device)
        rr = (r0.view(count, 1, 1) + rows.view(1, -1, 1)).clamp(max=size - 1)
        cc = (c0.view(count, 1, 1) + cols.view(1, 1, -1)).clamp(max=size - 1)
        inside = (rows.view(1, -1, 1) < (r1 - r0).view(count, 1, 1)) & (
            cols.view(1, 1, -1) < (c1 - c0).view(count, 1, 1)
        )
        env = ids.view(count, 1, 1).expand(count, height, width)
        counts = self.grid.counts[env, rr, cc]
        walls = (counts >= self.grid.hit_threshold) & inside
        return walls, r0, c0


def _crop_bounds(
    robot_xy: torch.Tensor,
    goal_xy: torch.Tensor,
    origin: torch.Tensor,
    res: torch.Tensor,
    half: int,
    size: int,
    known_lo: torch.Tensor,
    known_hi: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Fine-grid crop around the mapped cells, the robot, the goal, and that chord.

    ``known_lo`` / ``known_hi`` are inclusive ``(K, 2)`` bounds. A negative
    ``known_hi`` means nothing is mapped on that axis, and the robot cell is
    used instead.
    """
    robot_rc = _cell_batch(robot_xy, origin, res, half, size)
    goal_rc = _cell_batch(goal_xy, origin, res, half, size)
    margin = torch.ceil(_CROP_MARGIN_M / res.clamp(min=1e-6)).to(torch.long)
    mid = 0.5 * (robot_xy + goal_xy)
    half_span = _MIN_CROP_M * 0.5
    cover_lo = _cell_batch(mid - half_span, origin, res, half, size)
    cover_hi = _cell_batch(mid + half_span, origin, res, half, size)
    mapped = known_hi[:, 0] >= 0
    box_r0 = torch.where(mapped, known_lo[:, 0], robot_rc[:, 0])
    box_c0 = torch.where(known_hi[:, 1] >= 0, known_lo[:, 1], robot_rc[:, 1])
    box_r1 = torch.where(mapped, known_hi[:, 0] + 1, robot_rc[:, 0] + 1)
    box_c1 = torch.where(known_hi[:, 1] >= 0, known_hi[:, 1] + 1, robot_rc[:, 1] + 1)
    r0 = torch.minimum(torch.minimum(box_r0, robot_rc[:, 0]), torch.minimum(goal_rc[:, 0], cover_lo[:, 0])) - margin
    c0 = torch.minimum(torch.minimum(box_c0, robot_rc[:, 1]), torch.minimum(goal_rc[:, 1], cover_lo[:, 1])) - margin
    r0 = r0.clamp(0, size - 1)
    c0 = c0.clamp(0, size - 1)
    r1 = torch.maximum(torch.maximum(box_r1, robot_rc[:, 0] + 1), torch.maximum(goal_rc[:, 0] + 1, cover_hi[:, 0] + 1))
    c1 = torch.maximum(torch.maximum(box_c1, robot_rc[:, 1] + 1), torch.maximum(goal_rc[:, 1] + 1, cover_hi[:, 1] + 1))
    r1 = r1 + margin
    c1 = c1 + margin
    r1 = r1.clamp(max=size)
    c1 = c1.clamp(max=size)
    r1 = torch.maximum(r1, r0 + 1)
    c1 = torch.maximum(c1, c0 + 1)
    height = r1 - r0
    width = c1 - c0
    base = torch.round(_PLAN_RES_M / res.clamp(min=1e-6)).to(torch.long).clamp(min=1)
    side = torch.maximum(height, width)
    need = torch.div(side + _MAX_FIELD_CELLS - 1, _MAX_FIELD_CELLS, rounding_mode="floor").clamp(min=1)
    factor = torch.maximum(base, need)
    return r0, c0, r1, c1, factor, robot_rc, goal_rc


def _cell_batch(xy: torch.Tensor, origin: torch.Tensor, res: torch.Tensor, half: int, size: int) -> torch.Tensor:
    """World XY ``(K, 2)`` [m] to a clamped ``(row, col)``."""
    rel = (xy - origin) / res.clamp(min=1e-6).unsqueeze(-1)
    cell = torch.floor(rel).to(torch.long) + half
    return cell.clamp(0, size - 1)


def _footprint_batch(
    height: int,
    width: int,
    robot_r: torch.Tensor,
    robot_c: torch.Tensor,
    coarse_res: torch.Tensor,
) -> torch.Tensor:
    """True for coarse cells within ``_FOOTPRINT_M`` of each robot."""
    device = robot_r.device
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    dist2 = (rows.view(1, -1, 1) - robot_r.view(-1, 1, 1)) ** 2 + (cols.view(1, 1, -1) - robot_c.view(-1, 1, 1)) ** 2
    limit = (_FOOTPRINT_M / coarse_res.clamp(min=1e-6)) ** 2
    return dist2 <= limit.view(-1, 1, 1)


def _reverse_cost_batch(
    step: torch.Tensor,
    robot_xy: torch.Tensor,
    heading_xy: torch.Tensor | None,
    origin: torch.Tensor,
    res: torch.Tensor,
    half: int,
    r0: torch.Tensor,
    c0: torch.Tensor,
    factor: int,
    coarse_res: torch.Tensor,
) -> torch.Tensor:
    """Add the reverse penalty once, on the ring of cells just behind each robot."""
    if heading_xy is None:
        return step
    norm = torch.linalg.norm(heading_xy, dim=-1)
    good = norm > 1e-6
    heading = heading_xy / norm.clamp(min=1e-6).unsqueeze(-1)
    count, height, width = step.shape
    rows = torch.arange(height, device=step.device)
    cols = torch.arange(width, device=step.device)
    half_f = factor // 2
    fine_r = r0.view(count, 1, 1) + rows.view(1, -1, 1) * factor + half_f
    fine_c = c0.view(count, 1, 1) + cols.view(1, 1, -1) * factor + half_f
    scale = res.view(count, 1, 1)
    xs = (fine_r.to(torch.float32) - half + 0.5) * scale + origin[:, 0].view(count, 1, 1)
    ys = (fine_c.to(torch.float32) - half + 0.5) * scale + origin[:, 1].view(count, 1, 1)
    rel_x = xs - robot_xy[:, 0].view(count, 1, 1)
    rel_y = ys - robot_xy[:, 1].view(count, 1, 1)
    dist = torch.sqrt(rel_x * rel_x + rel_y * rel_y).clamp(min=1e-6)
    forward = rel_x * heading[:, 0].view(count, 1, 1) + rel_y * heading[:, 1].view(count, 1, 1)
    ring = (forward < -0.3 * dist) & (dist > 0.3) & (dist < 1.5) & good.view(count, 1, 1)
    bump = _REVERSE_PENALTY_M / coarse_res.clamp(min=1e-6)
    return step + ring.to(step.dtype) * bump.view(count, 1, 1)


def _seen_counts(
    cells: torch.Tensor,
    seen: torch.Tensor,
    height: torch.Tensor,
    width: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """How many traced cells stay inside the seen prefix. The start is always kept."""
    row = cells[..., 0]
    col = cells[..., 1]
    steps = torch.arange(cells.shape[1], device=cells.device).view(1, -1)
    real = steps < counts.unsqueeze(1)
    inside = (row >= 0) & (col >= 0) & (row < height.unsqueeze(1)) & (col < width.unsqueeze(1)) & real
    safe_r = row.clamp(0, seen.shape[1] - 1)
    safe_c = col.clamp(0, seen.shape[2] - 1)
    env = torch.arange(seen.shape[0], device=cells.device).view(-1, 1).expand_as(safe_r)
    good = inside & seen[env, safe_r, safe_c]
    good = good.clone()
    good[:, 0] = inside[:, 0]
    return torch.cumprod(good.to(torch.int32), dim=1).sum(dim=1)


def _cells_to_world_batch(
    local: torch.Tensor,
    r0: torch.Tensor,
    c0: torch.Tensor,
    factor: torch.Tensor,
    res: torch.Tensor,
    origin: torch.Tensor,
    half: int,
) -> torch.Tensor:
    """World XY [m] of coarse cell centres. ``local`` is ``(K, T, 2)``."""
    factor_f = factor.to(torch.float32).view(-1, 1)
    mid = torch.div(factor, 2, rounding_mode="floor").to(torch.float32).view(-1, 1)
    fine_r = r0.to(torch.float32).view(-1, 1) + local[..., 0].to(torch.float32) * factor_f + mid
    fine_c = c0.to(torch.float32).view(-1, 1) + local[..., 1].to(torch.float32) * factor_f + mid
    xs = (fine_r - half + 0.5) * res.view(-1, 1) + origin[:, 0].view(-1, 1)
    ys = (fine_c - half + 0.5) * res.view(-1, 1) + origin[:, 1].view(-1, 1)
    return torch.stack((xs, ys), dim=-1)


def _to_cpu(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Copy to the host and synchronize once."""
    copied = tuple(tensor.detach().to("cpu", non_blocking=tensor.is_cuda) for tensor in tensors)
    if tensors and tensors[0].is_cuda:
        torch.cuda.synchronize()
    return copied


def _shape_route(
    path: torch.Tensor,
    walls: torch.Tensor,
    win_r0: int,
    win_c0: int,
    res: float,
    origin: tuple[float, float],
    half: int,
    size: int,
) -> torch.Tensor:
    """Simplify, cut the first corner, and pull the end off mapped walls. CPU only."""
    path = _drop_backtrack(path)
    path = _simplify_polyline(path, _SIMPLIFY_M)
    path, turned = _cut_turn(path, _TURN_MIN_M, _TURN_RAD)
    if turned:
        path = _shorten_end(path, _CORNER_BACKOFF_M, _MIN_CARROT_M)
    path = _cap_length(path, _LOOKAHEAD_M)
    return _pull_off_cpu(path, walls, win_r0, win_c0, res, origin, half, size)


def _pull_off_cpu(
    world: torch.Tensor,
    walls: torch.Tensor,
    win_r0: int,
    win_c0: int,
    res: float,
    origin: tuple[float, float],
    half: int,
    size: int,
) -> torch.Tensor:
    """Step the endpoint back while it is within ``_WALL_CLEAR_M`` of a wall."""
    if world.shape[0] < 2:
        return world
    length = float(cumulative_length(world)[-1])
    for _ in range(32):
        if (
            length <= _MIN_CARROT_M
            or _wall_distance_cpu(world[-1], walls, win_r0, win_c0, res, origin, half, size) >= _WALL_CLEAR_M
        ):
            break
        nxt = _cap_length(world, max(length - 0.25, _MIN_CARROT_M))
        nxt_len = float(cumulative_length(nxt)[-1])
        if nxt_len >= length - 1.0e-3:
            break
        world = nxt
        length = nxt_len
    return world


def _wall_distance_cpu(
    xy: torch.Tensor,
    walls: torch.Tensor,
    win_r0: int,
    win_c0: int,
    res: float,
    origin: tuple[float, float],
    half: int,
    size: int,
) -> float:
    """Distance [m] from ``xy`` to the nearest wall centre in the copied window."""
    point = _xy(xy)
    row, col = _cell_of(point, origin, res, half, size)
    rad = int(math.ceil(_WALL_CLEAR_M / max(res, 1e-6))) + 1
    r0, r1 = max(0, row - rad), min(size, row + rad + 1)
    c0, c1 = max(0, col - rad), min(size, col + rad + 1)
    lr0, lr1 = max(0, r0 - win_r0), min(int(walls.shape[0]), r1 - win_r0)
    lc0, lc1 = max(0, c0 - win_c0), min(int(walls.shape[1]), c1 - win_c0)
    if lr1 <= lr0 or lc1 <= lc0:
        return 1.0e6
    window = walls[lr0:lr1, lc0:lc1]
    if not bool(window.any()):
        return 1.0e6
    rr, cc = torch.where(window)
    centers_x = (win_r0 + lr0 + rr.to(torch.float32) - half + 0.5) * res + origin[0]
    centers_y = (win_c0 + lc0 + cc.to(torch.float32) - half + 0.5) * res + origin[1]
    delta_x = centers_x - point[0]
    delta_y = centers_y - point[1]
    return float(torch.sqrt(delta_x * delta_x + delta_y * delta_y).min())


def _fill_polyline(buffer: torch.Tensor, slot: int, point: torch.Tensor) -> None:
    """Repeat ``point`` along one polyline row."""
    buffer[slot] = point.reshape(1, 2).expand(buffer.shape[1], 2)


def _xy(point: torch.Tensor | tuple[float, float]) -> tuple[float, float]:
    """Two Python floats. A GPU tensor is read once."""
    if torch.is_tensor(point):
        values = point.detach().reshape(-1).tolist()
        return (float(values[0]), float(values[1]))
    return (float(point[0]), float(point[1]))


def _cell_of(xy: tuple[float, float], origin: tuple[float, float], res: float, half: int, size: int) -> tuple[int, int]:
    """World XY [m] to a clamped ``(row, col)``."""
    row = int(math.floor((xy[0] - origin[0]) / res)) + half
    col = int(math.floor((xy[1] - origin[1]) / res)) + half
    last = size - 1
    return (max(0, min(last, row)), max(0, min(last, col)))


def _drop_backtrack(points: torch.Tensor) -> torch.Tensor:
    """Drop a vertex that sits on the previous one."""
    if points.shape[0] < 2:
        return points
    keep = [0]
    for index in range(1, points.shape[0]):
        if float(torch.linalg.norm(points[index] - points[keep[-1]])) > 1e-3:
            keep.append(index)
    return points[keep]


def _simplify_polyline(points: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Ramer-Douglas-Peucker. Drops cell-scale jogs, keeps corridor corners."""
    count = int(points.shape[0])
    if count < 3:
        return points
    keep = torch.zeros(count, dtype=torch.bool, device=points.device)
    keep[0] = True
    keep[-1] = True
    stack = [(0, count - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        segment = points[end] - points[start]
        length = torch.linalg.norm(segment).clamp(min=1e-8)
        rel = points[start + 1 : end] - points[start]
        cross = (rel[:, 0] * segment[1] - rel[:, 1] * segment[0]).abs() / length
        offset = int(torch.argmax(cross).item())
        if float(cross[offset]) > epsilon:
            chosen = start + 1 + offset
            keep[chosen] = True
            stack.append((start, chosen))
            stack.append((chosen, end))
    return points[keep]


def _cut_turn(points: torch.Tensor, min_m: float, turn_rad: float) -> tuple[torch.Tensor, bool]:
    """End the prefix at the first sharp corner beyond ``min_m`` [m].

    The bool is true when a corner was cut, so the caller can stop short of
    that vertex.
    """
    if points.shape[0] < 3:
        return points, False
    cum = cumulative_length(points)
    limit = math.cos(turn_rad)
    for index in range(1, points.shape[0] - 1):
        if float(cum[index]) < min_m:
            continue
        incoming = points[index] - points[index - 1]
        outgoing = points[index + 1] - points[index]
        length_in = float(torch.linalg.norm(incoming))
        length_out = float(torch.linalg.norm(outgoing))
        if length_in < 1e-4 or length_out < 1e-4:
            continue
        cosine = float(torch.dot(incoming, outgoing) / (length_in * length_out))
        if cosine < limit:
            return points[: index + 1], True
    return points, False


def _shorten_end(points: torch.Tensor, backoff_m: float, min_m: float) -> torch.Tensor:
    """Drop the last ``backoff_m`` [m] unless that would leave less than ``min_m``."""
    if points.shape[0] < 2:
        return points
    length = float(cumulative_length(points)[-1])
    target = length - backoff_m
    if target < min_m:
        return points
    return _cap_length(points, target)


def _cap_length(points: torch.Tensor, max_m: float) -> torch.Tensor:
    """Cut ``points`` at arc length ``max_m`` [m]."""
    if points.shape[0] < 2:
        return points
    cum = cumulative_length(points)
    if float(cum[-1]) <= max_m:
        return points
    end = point_at_distance(points, max_m)
    before = int((cum < max_m - 1e-4).sum().item())
    before = max(1, min(before, points.shape[0]))
    kept = points[:before]
    if float(torch.linalg.norm(kept[-1] - end)) > 1e-3:
        kept = torch.cat((kept, end.unsqueeze(0)), dim=0)
    return kept


def _inflate(occupied: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a wall mask. A leading batch dimension is kept; a single grid is not."""
    if radius <= 0:
        return occupied
    kernel = 2 * radius + 1
    single = occupied.dim() == 2
    data = occupied.float()
    if single:
        data = data.unsqueeze(0)
    pooled = torch.nn.functional.max_pool2d(data.unsqueeze(1), kernel_size=kernel, stride=1, padding=radius)
    out = pooled[:, 0] > 0.5
    if single:
        return out[0]
    return out
