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
from go2_lidar_maze.nav_core.path_utils import cumulative_length, fillet_polyline, point_at_distance, points_at_arc_length
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
# Close start and goal cells can still need a tour of the tile. Cover that,
# not only the chord, or the field treats the detour as a wall.
_MIN_CROP_M = 48.0
_MAX_FIELD_CELLS = 160
_FIELD_EVERY = 5
_MAX_ROUTE_POINTS = 96
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
        self.grid = OccupancyGrid(
            num_envs=num_envs, device=self.device, resolution=_GRID_RES_M, size=_GRID_CELLS
        )
        self._goal_xy = torch.zeros((num_envs, 2), device=self.device)
        self._age = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._have_plan = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self._cached: dict[int, tuple[torch.Tensor, torch.Tensor, float]] = {}
        self._route_xy = torch.zeros((num_envs, _MAX_ROUTE_POINTS, 2), device=self.device)
        self._route_n = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.field_age = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.field_ready = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.planned_xy = torch.zeros((num_envs, 2), device=self.device)
        self._fields: list[dict | None] = [None] * num_envs
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
        self.field_age[env_ids] = 0
        self.field_ready[env_ids] = False
        for index in env_ids.tolist():
            self._cached.pop(int(index), None)
            self._fields[int(index)] = None
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
        near = torch.linalg.norm(points - robot_xy.unsqueeze(1), dim=-1) < _FOOTPRINT_M
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
        ids = [int(index) for index in mask.nonzero(as_tuple=False).flatten().tolist()]
        crops: list[tuple[int, dict]] = []
        for index in ids:
            if int(self.grid.free_counts[index].sum()) <= 0:
                self._fields[index] = None
                self.field_ready[index] = False
                continue
            heading = None if heading_xy is None else heading_xy[index]
            crops.append((index, self._build_crop(index, robot_xy[index], heading)))
        if crops:
            self._solve_crops(crops)
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
        cover = torch.ones_like(mask)
        for index in mask.nonzero(as_tuple=False).flatten().tolist():
            cover[int(index)] = self._field_covers(int(index), robot_xy[int(index)])
        stale = mask & (~self.field_ready | (self.field_age >= _FIELD_EVERY) | ~cover)
        self.refresh_fields(stale, robot_xy, heading_xy)
        self._trace_mask(mask, robot_xy)
        self._age[mask] = 0
        self._have_plan[mask] = True

    def _solve_crops(self, crops: list[tuple[int, dict]]) -> None:
        """One batched field for every crop, then slice the padding back off."""
        max_h = max(int(crop["occupied"].shape[0]) for _index, crop in crops)
        max_w = max(int(crop["occupied"].shape[1]) for _index, crop in crops)
        batch = len(crops)
        device = self.device
        occupied = torch.ones((batch, max_h, max_w), dtype=torch.bool, device=device)
        step = torch.ones((batch, max_h, max_w), dtype=torch.float32, device=device)
        goal = torch.zeros((batch, 2), dtype=torch.long, device=device)
        robot = torch.zeros((batch, 2), dtype=torch.long, device=device)
        res = torch.zeros(batch, dtype=torch.float32, device=device)
        for slot, (_index, crop) in enumerate(crops):
            height, width = crop["occupied"].shape
            occupied[slot, :height, :width] = crop["occupied"]
            step[slot, :height, :width] = crop["step"]
            goal[slot] = crop["goal"]
            robot[slot] = crop["robot"]
            res[slot] = crop["coarse_res"]
        # A wave crosses the crop in about H + W sweeps. Two quiet checks end it.
        dist = distance_field_batch(
            occupied,
            goal,
            res,
            step_cost=step,
            robot=robot,
            max_iters=min(1024, (max_h + max_w) * 2),
            check_every=8,
        )
        for slot, (index, crop) in enumerate(crops):
            height, width = crop["occupied"].shape
            self._fields[index] = {
                "dist": dist[slot, :height, :width].contiguous(),
                "seen": crop["seen"],
                "r0": crop["r0"],
                "c0": crop["c0"],
                "factor": crop["factor"],
                "res": crop["res"],
                "origin": crop["origin"],
                "half": crop["half"],
            }
            self.field_ready[index] = True

    def _trace_mask(self, mask: torch.Tensor, robot_xy: torch.Tensor) -> None:
        real: list[int] = []
        for index in mask.nonzero(as_tuple=False).flatten().tolist():
            index = int(index)
            if self._fields[index] is None:
                self._set_wait(index, robot_xy[index])
            else:
                real.append(index)
        if not real:
            return
        dists = [self._fields[index]["dist"] for index in real]
        max_h = max(int(dist.shape[0]) for dist in dists)
        max_w = max(int(dist.shape[1]) for dist in dists)
        batch = len(real)
        padded = torch.full((batch, max_h, max_w), UNREACHABLE, device=self.device)
        starts = torch.zeros((batch, 2), dtype=torch.long, device=self.device)
        res = torch.zeros(batch, dtype=torch.float32, device=self.device)
        for slot, index in enumerate(real):
            field = self._fields[index]
            dist = field["dist"]
            height, width = dist.shape
            padded[slot, :height, :width] = dist
            row, col = self._local_cell(field, robot_xy[index], clamp=True)
            starts[slot, 0] = row
            starts[slot, 1] = col
            res[slot] = float(field["res"]) * int(field["factor"])
        cells, counts = trace_route_batch(
            padded, starts, resolution=res, max_length_m=_TRACE_M, max_steps=64
        )
        for slot, index in enumerate(real):
            count = int(counts[slot].item())
            self._commit_route(index, robot_xy[index], cells[slot, :count])

    def _build_crop(
        self,
        index: int,
        robot_xy: torch.Tensor,
        heading_xy: torch.Tensor | None,
    ) -> dict:
        """Cost crop around the known map, the robot, and the goal."""
        raw = self.grid.counts[index] >= self.grid.hit_threshold
        unknown = (self.grid.free_counts[index] <= 0) & ~raw
        res = float(self.grid.res[index])
        origin = self.grid.origin[index]
        half = self.grid.size // 2
        known = raw | (self.grid.free_counts[index] > 0)
        robot_rc = self._world_cell(robot_xy, origin, res, half)
        goal_rc = self._world_cell(self._goal_xy[index], origin, res, half)
        margin = int(math.ceil(_CROP_MARGIN_M / max(res, 1e-6)))
        if bool(known.any()):
            rows = known.any(dim=1).nonzero().flatten()
            cols = known.any(dim=0).nonzero().flatten()
            r0, r1 = int(rows[0]), int(rows[-1]) + 1
            c0, c1 = int(cols[0]), int(cols[-1]) + 1
        else:
            r0, r1, c0, c1 = robot_rc[0], robot_rc[0] + 1, robot_rc[1], robot_rc[1] + 1
        mid = 0.5 * (robot_xy + self._goal_xy[index])
        half_span = torch.tensor([_MIN_CROP_M * 0.5, _MIN_CROP_M * 0.5], device=robot_xy.device)
        cover_lo = self._world_cell(mid - half_span, origin, res, half)
        cover_hi = self._world_cell(mid + half_span, origin, res, half)
        r0 = max(0, min(r0, robot_rc[0], goal_rc[0], cover_lo[0]) - margin)
        c0 = max(0, min(c0, robot_rc[1], goal_rc[1], cover_lo[1]) - margin)
        r1 = min(self.grid.size, max(r1, robot_rc[0] + 1, goal_rc[0] + 1, cover_hi[0] + 1) + margin)
        c1 = min(self.grid.size, max(c1, robot_rc[1] + 1, goal_rc[1] + 1, cover_hi[1] + 1) + margin)
        factor = _pool_factor(r1 - r0, c1 - c0, res)
        occupied = raw[r0:r1, c0:c1]
        unseen = unknown[r0:r1, c0:c1]
        if factor > 1:
            occupied = _pad_to_factor(occupied, factor, True)
            unseen = _pad_to_factor(unseen, factor, True)
            occupied, unseen, factor = _coarsen(occupied, unseen, res, res * factor)
        coarse_res = res * factor
        inflated = _inflate(occupied, _INFLATE_CELLS)
        robot_local = ((robot_rc[0] - r0) // factor, (robot_rc[1] - c0) // factor)
        goal_local = ((goal_rc[0] - r0) // factor, (goal_rc[1] - c0) // factor)
        height, width = inflated.shape
        robot_local = (min(height - 1, max(0, robot_local[0])), min(width - 1, max(0, robot_local[1])))
        goal_local = (min(height - 1, max(0, goal_local[0])), min(width - 1, max(0, goal_local[1])))
        foot = _footprint_mask(inflated.shape, robot_local, coarse_res, inflated.device)
        # Inflation must not trap the body. A real wall under the footprint stays blocked.
        inflated = inflated & ~(foot & ~occupied)
        penalty = _clearance_penalty(inflated, _CLEARANCE_RADIUS, _CLEARANCE_WEIGHT)
        step = 1.0 + penalty
        step = torch.where(unseen & ~occupied, step * _UNKNOWN_COST, step)
        step = _reverse_cost(step, robot_xy, heading_xy, origin, res, half, r0, c0, factor, coarse_res)
        seen = (~unseen & ~occupied) | (foot & ~occupied)
        return {
            "occupied": inflated,
            "step": step,
            "seen": seen,
            "goal": torch.tensor(goal_local, dtype=torch.long, device=self.device),
            "robot": torch.tensor(robot_local, dtype=torch.long, device=self.device),
            "coarse_res": coarse_res,
            "r0": r0,
            "c0": c0,
            "factor": int(factor),
            "res": res,
            "origin": origin,
            "half": half,
        }

    def _commit_route(self, index: int, robot_xy: torch.Tensor, local: torch.Tensor) -> None:
        """Clip the traced cells to seen space, then to the lookahead and the first corner."""
        field = self._fields[index]
        if field is None or local.shape[0] == 0:
            self._set_wait(index, robot_xy)
            return
        kept = _seen_prefix(local, field["seen"])
        if kept.shape[0] <= 1:
            self._set_wait(index, robot_xy)
            return
        world = _cells_to_world(field, kept)
        world = world.clone()
        world[0] = robot_xy.reshape(2)
        world = _drop_backtrack(world)
        world = _simplify_polyline(world, _SIMPLIFY_M)
        world, turned = _cut_turn(world, _TURN_MIN_M, _TURN_RAD)
        if turned:
            world = _shorten_end(world, _CORNER_BACKOFF_M, _MIN_CARROT_M)
        world = _cap_length(world, _LOOKAHEAD_M)
        world = self._pull_off_walls(index, world)
        if world.shape[0] < 2 or float(torch.linalg.norm(world[-1] - world[0])) < 0.05:
            self._set_wait(index, robot_xy)
            return
        stored = fillet_polyline(world, radius=0.5) if world.shape[0] >= 3 else world
        self._remember(index, world[-1], stored)

    def _remember(self, index: int, waypoint: torch.Tensor, stored: torch.Tensor) -> None:
        self._store_route(index, stored)
        self.planned_xy[index] = waypoint
        self._paths[index] = stored
        length = float(cumulative_length(stored)[-1]) if stored.shape[0] >= 2 else 0.0
        self._cached[index] = (waypoint, stored, length)

    def _pull_off_walls(self, index: int, world: torch.Tensor) -> torch.Tensor:
        """Step the endpoint back while it is within ``_WALL_CLEAR_M`` of a wall.

        Each pass must shorten the polyline. A cap that does not move the end
        stops the walk, so a carrot lying against a wall cannot spin here.
        """
        if world.shape[0] < 2:
            return world
        length = float(cumulative_length(world)[-1])
        for _ in range(32):
            if length <= _MIN_CARROT_M or self._wall_distance(index, world[-1]) >= _WALL_CLEAR_M:
                break
            nxt = _cap_length(world, max(length - 0.25, _MIN_CARROT_M))
            nxt_len = float(cumulative_length(nxt)[-1])
            if nxt_len >= length - 1.0e-3:
                break
            world = nxt
            length = nxt_len
        return world

    def _wall_distance(self, index: int, xy: torch.Tensor) -> float:
        """Distance [m] from ``xy`` to the nearest mapped wall cell centre."""
        res = float(self.grid.res[index])
        origin = self.grid.origin[index]
        half = int(self.grid._half)
        size = int(self.grid.size)
        rel = (xy - origin) / res
        row = int(math.floor(float(rel[0]))) + half
        col = int(math.floor(float(rel[1]))) + half
        rad = int(math.ceil(_WALL_CLEAR_M / max(res, 1e-6))) + 1
        r0, r1 = max(0, row - rad), min(size, row + rad + 1)
        c0, c1 = max(0, col - rad), min(size, col + rad + 1)
        window = self.grid.counts[index, r0:r1, c0:c1]
        walls = window >= self.grid.hit_threshold
        if not bool(walls.any()):
            return 1.0e6
        rr, cc = torch.where(walls)
        centers_x = (r0 + rr.to(torch.float32) - half + 0.5) * res + float(origin[0])
        centers_y = (c0 + cc.to(torch.float32) - half + 0.5) * res + float(origin[1])
        delta_x = centers_x - float(xy[0])
        delta_y = centers_y - float(xy[1])
        return float(torch.sqrt(delta_x * delta_x + delta_y * delta_y).min())

    def _set_wait(self, index: int, robot_xy: torch.Tensor) -> None:
        """No seen route yet. Hold the waypoint on the robot."""
        point = robot_xy.reshape(2)
        self._remember(index, point, point.unsqueeze(0))

    def _field_covers(self, index: int, robot_xy: torch.Tensor) -> bool:
        field = self._fields[index]
        if field is None:
            return False
        row, col = self._local_cell(field, robot_xy, clamp=False)
        height, width = field["dist"].shape
        return 0 <= row < height and 0 <= col < width

    def _local_cell(self, field: dict, xy: torch.Tensor, *, clamp: bool) -> tuple[int, int]:
        row, col = self._world_cell(xy, field["origin"], float(field["res"]), int(field["half"]))
        local = ((row - int(field["r0"])) // int(field["factor"]), (col - int(field["c0"])) // int(field["factor"]))
        if not clamp:
            return local
        height, width = field["dist"].shape
        return (min(height - 1, max(0, local[0])), min(width - 1, max(0, local[1])))

    def _world_cell(
        self, xy: torch.Tensor, origin: torch.Tensor, res: float, half: int
    ) -> tuple[int, int]:
        rel = (xy - origin) / res
        row = int(math.floor(float(rel[0]))) + half
        col = int(math.floor(float(rel[1]))) + half
        last = self.grid.size - 1
        return (max(0, min(last, row)), max(0, min(last, col)))

    def _store_route(self, env_index: int, raw: torch.Tensor) -> None:
        """Keep the committed polyline for the batched wall check."""
        count = min(int(raw.shape[0]), _MAX_ROUTE_POINTS)
        self._route_xy[env_index] = raw[-1]
        self._route_xy[env_index, :count] = raw[:count]
        self._route_n[env_index] = count


def _pool_factor(height: int, width: int, res: float) -> int:
    """Coarsen so the field stays near ``_PLAN_RES_M`` and under ``_MAX_FIELD_CELLS``."""
    factor = max(1, int(round(_PLAN_RES_M / max(res, 1e-6))))
    while max(height, width) / factor > _MAX_FIELD_CELLS:
        factor += 1
    return factor


def _pad_to_factor(mask: torch.Tensor, factor: int, fill: bool) -> torch.Tensor:
    """Pad the bottom and right edges so both sides divide by ``factor``."""
    height, width = mask.shape
    padded_h = (height + factor - 1) // factor * factor
    padded_w = (width + factor - 1) // factor * factor
    if padded_h == height and padded_w == width:
        return mask
    out = torch.full((padded_h, padded_w), fill, dtype=mask.dtype, device=mask.device)
    out[:height, :width] = mask
    return out


def _footprint_mask(
    shape: tuple[int, int], robot: tuple[int, int], res: float, device: torch.device
) -> torch.Tensor:
    """True for cells within ``_FOOTPRINT_M`` of ``robot``."""
    height, width = shape
    reach = max(1, int(math.ceil(_FOOTPRINT_M / max(res, 1e-6))))
    r0, r1 = max(0, robot[0] - reach), min(height, robot[0] + reach + 1)
    c0, c1 = max(0, robot[1] - reach), min(width, robot[1] + reach + 1)
    rows = torch.arange(r0, r1, device=device)
    cols = torch.arange(c0, c1, device=device)
    dist2 = (rows.view(-1, 1) - robot[0]) ** 2 + (cols.view(1, -1) - robot[1]) ** 2
    window = dist2 <= (_FOOTPRINT_M / max(res, 1e-6)) ** 2
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    mask[r0:r1, c0:c1] = window
    return mask


def _reverse_cost(
    step: torch.Tensor,
    robot_xy: torch.Tensor,
    heading_xy: torch.Tensor | None,
    origin: torch.Tensor,
    res: float,
    half: int,
    r0: int,
    c0: int,
    factor: int,
    coarse_res: float,
) -> torch.Tensor:
    """Add the reverse penalty once, on the ring of cells just behind the robot."""
    if heading_xy is None:
        return step
    heading = heading_xy.to(dtype=torch.float32)
    norm = torch.linalg.norm(heading)
    if float(norm) < 1e-6:
        return step
    heading = heading / norm
    height, width = step.shape
    rows = torch.arange(height, device=step.device)
    cols = torch.arange(width, device=step.device)
    fine_r = r0 + rows * factor + factor // 2
    fine_c = c0 + cols * factor + factor // 2
    xs = (fine_r.to(torch.float32) - half + 0.5) * res + float(origin[0])
    ys = (fine_c.to(torch.float32) - half + 0.5) * res + float(origin[1])
    rel_x = xs.view(-1, 1) - float(robot_xy[0])
    rel_y = ys.view(1, -1) - float(robot_xy[1])
    dist = torch.sqrt(rel_x**2 + rel_y**2).clamp(min=1e-6)
    forward = rel_x * float(heading[0]) + rel_y * float(heading[1])
    ring = (forward < -0.3 * dist) & (dist > 0.3) & (dist < 1.5)
    return step + ring.to(step.dtype) * (_REVERSE_PENALTY_M / max(coarse_res, 1e-6))


def _seen_prefix(local: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    """Cells from the start through the last seen cell. The start is always kept."""
    height, width = seen.shape
    keep = 1
    for index in range(local.shape[0]):
        row = int(local[index, 0])
        col = int(local[index, 1])
        if row < 0 or col < 0 or row >= height or col >= width:
            break
        if index > 0 and not bool(seen[row, col]):
            break
        keep = index + 1
    return local[:keep]


def _cells_to_world(field: dict, local: torch.Tensor) -> torch.Tensor:
    """World XY [m] of coarse cell centres."""
    factor = int(field["factor"])
    res = float(field["res"])
    half = int(field["half"])
    origin = field["origin"]
    fine_r = int(field["r0"]) + local[:, 0].to(torch.float32) * factor + factor // 2
    fine_c = int(field["c0"]) + local[:, 1].to(torch.float32) * factor + factor // 2
    xy = torch.stack(
        (
            (fine_r - half + 0.5) * res + float(origin[0]),
            (fine_c - half + 0.5) * res + float(origin[1]),
        ),
        dim=-1,
    )
    return xy


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


def _coarsen(
    occupied: torch.Tensor, unknown: torch.Tensor, src_res: float, dst_res: float
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Max-pool the fine grid. Returns the integer pool factor (1 keeps the grid)."""
    if dst_res <= src_res * 1.01:
        return occupied, unknown, 1
    factor = max(1, int(round(dst_res / src_res)))
    edge_h = occupied.shape[0] - (occupied.shape[0] % factor)
    edge_w = occupied.shape[1] - (occupied.shape[1] % factor)
    occ = occupied[:edge_h, :edge_w].reshape(edge_h // factor, factor, edge_w // factor, factor).any(dim=(1, 3))
    unk = unknown[:edge_h, :edge_w].reshape(edge_h // factor, factor, edge_w // factor, factor).all(dim=(1, 3))
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
