# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distance field, corner blocking, and corridor-graph waypoints."""

from __future__ import annotations

import math

import torch
from go2_lidar_maze.constants import CELL_CENTER_ORIGIN_M, CELL_PITCH_M
from go2_lidar_maze.nav_core.path_utils import commit_prefix
from go2_lidar_maze.nav_core.planner import UNREACHABLE, astar_grid, distance_field, trace_route
from go2_lidar_maze.nav_core.occupancy import OccupancyGrid
from go2_lidar_maze.nav_core.stack import NavigationStack


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_astar_grid_runs_on_the_occupancy_device():
    """Clearance cost must be built on the same device as the wall mask."""
    device = _device()
    occupied = torch.zeros((12, 12), dtype=torch.bool, device=device)
    occupied[5, 2:10] = True
    path = astar_grid(occupied, (8, 5), (2, 5), resolution=0.25)
    assert path is not None
    assert path[0] == (8, 5)
    assert path[-1] == (2, 5)
    assert all(not bool(occupied[cell]) for cell in path)


def test_distance_field_matches_octile_on_open_ground():
    """Three diagonal steps and one cardinal step from (0, 0) to (3, 4)."""
    occupied = torch.zeros((6, 6), dtype=torch.bool)
    dist = distance_field(occupied, (0, 0), resolution=1.0, robot=(3, 4), max_iters=16, settle_iters=4)
    expected = 3.0 * math.sqrt(2.0) + 1.0
    assert abs(float(dist[3, 4]) - expected) < 1e-3


def test_distance_field_goes_around_a_wall():
    """A wall across the middle leaves column 0 as the only gap."""
    occupied = torch.zeros((7, 7), dtype=torch.bool)
    occupied[3, 1:] = True
    dist = distance_field(occupied, (1, 3), resolution=1.0, robot=(5, 3), max_iters=40, settle_iters=8)
    assert float(dist[5, 3]) < UNREACHABLE * 0.5
    route = trace_route(dist, (5, 3), resolution=1.0, max_length_m=30.0, max_steps=30)
    assert all(not bool(occupied[cell]) for cell in route)
    assert any(col == 0 for _row, col in route)


def test_distance_field_does_not_cut_a_blocked_corner():
    """The diagonal from (1, 0) to (0, 1) clips the wall at (1, 1)."""
    occupied = torch.zeros((4, 4), dtype=torch.bool)
    occupied[1, 1] = True
    dist = distance_field(occupied, (0, 2), resolution=1.0, robot=(2, 0), max_iters=20, settle_iters=6)
    route = trace_route(dist, (2, 0), resolution=1.0, max_length_m=20.0, max_steps=20)
    steps = list(zip(route, route[1:]))
    assert ((1, 0), (0, 1)) not in steps
    assert ((0, 1), (1, 0)) not in steps
    assert float(dist[2, 0]) < UNREACHABLE * 0.5


def test_commit_prefix_stops_at_the_visible_frontier():
    """Eight metres of seen corridor are kept, and the unknown tail is not."""
    points = torch.tensor([[float(i), 0.0] for i in range(11)], dtype=torch.float32)
    seen = torch.zeros(11, dtype=torch.bool)
    seen[:5] = True
    blocked = torch.zeros(11, dtype=torch.bool)
    prefix, last_seen = commit_prefix(points, seen, blocked, probe_m=2.5, max_m=15.0)
    assert last_seen == 4
    assert prefix.shape[0] == 5
    assert float(prefix[-1, 0]) == 4.0


def test_commit_prefix_probes_until_one_cell_when_nothing_is_seen():
    """An empty map still yields a one-cell target, and no further."""
    points = torch.tensor([[float(i), 0.0] for i in range(11)], dtype=torch.float32)
    seen = torch.zeros(11, dtype=torch.bool)
    blocked = torch.zeros(11, dtype=torch.bool)
    prefix, last_seen = commit_prefix(points, seen, blocked, probe_m=2.5, max_m=15.0)
    assert last_seen == 0
    assert float(prefix[-1, 0]) == 2.0


def _mark_door(stack: NavigationStack, row: int, col: int, drow: int, dcol: int, *, wall: bool) -> None:
    """Paint the doorway samples the planner reads. ``(drow, dcol)`` is a unit step."""
    device = stack.grid.counts.device
    if dcol == 1:
        x = CELL_CENTER_ORIGIN_M + CELL_PITCH_M * (col + 0.5)
        y = CELL_CENTER_ORIGIN_M + CELL_PITCH_M * row
        offsets = torch.linspace(-0.7, 0.7, 15, device=device)
        pts = torch.stack((torch.full_like(offsets, x), y + offsets), dim=-1)
    else:
        x = CELL_CENTER_ORIGIN_M + CELL_PITCH_M * col
        y = CELL_CENTER_ORIGIN_M + CELL_PITCH_M * (row + 0.5)
        offsets = torch.linspace(-0.7, 0.7, 15, device=device)
        pts = torch.stack((x + offsets, torch.full_like(offsets, y)), dim=-1)
    # Nudge off cell boundaries so ``floor`` cannot skip the painted cell.
    shifts = torch.tensor([[dx, dy] for dx in (-0.05, 0.05) for dy in (-0.05, 0.05)], device=device)
    pts = (pts.unsqueeze(1) + shifts).reshape(1, -1, 2)
    cells = stack.grid.world_to_cell(pts)[0]
    if wall:
        stack.grid.counts[0, cells[:, 0], cells[:, 1]] = 4
        stack.grid.free_counts[0, cells[:, 0], cells[:, 1]] = 0
    else:
        stack.grid.free_counts[0, cells[:, 0], cells[:, 1]] = 4
        stack.grid.counts[0, cells[:, 0], cells[:, 1]] = 0


def test_waypoint_follows_open_doors_not_a_seen_shortcut():
    """A free cell toward the goal is ignored unless its doorway is open.

    The east hall is mapped. The cell due north is painted free, and the door
    in front of it is a wall, matching a LiDAR glimpse through some other gap.
    """
    device = _device()
    stack = NavigationStack(1, device)
    robot = torch.tensor([[CELL_CENTER_ORIGIN_M, CELL_CENTER_ORIGIN_M]], dtype=torch.float32, device=device)
    goal = torch.tensor([[CELL_CENTER_ORIGIN_M, CELL_CENTER_ORIGIN_M + 8.0 * CELL_PITCH_M]], device=device)
    stack.reset(torch.tensor([0], device=device), robot, goal)
    _mark_door(stack, 0, 0, 0, 1, wall=False)
    _mark_door(stack, 0, 1, 0, 1, wall=False)
    _mark_door(stack, 0, 0, 1, 0, wall=True)
    _mark_door(stack, 0, 1, 1, 0, wall=True)
    _mark_door(stack, 0, 2, 1, 0, wall=True)
    # Glimpse of the cell two rows north. Its door is not open.
    glimpse = stack.grid.world_to_cell(
        torch.tensor(
            [[[CELL_CENTER_ORIGIN_M, CELL_CENTER_ORIGIN_M + 2.0 * CELL_PITCH_M]]],
            device=device,
        )
    )[0, 0]
    stack.grid.free_counts[0, glimpse[0], glimpse[1]] = 4
    waypoint, _path, _length = stack.plan_waypoint(0, robot[0], force=True)
    assert abs(float(waypoint[0]) - (CELL_CENTER_ORIGIN_M + 2.0 * CELL_PITCH_M)) < 0.2
    assert abs(float(waypoint[1]) - CELL_CENTER_ORIGIN_M) < 0.2


def test_unknown_door_does_not_freeze_a_walkable_hall():
    """An unscanned door on the robot's cell loses to a hall that is already open."""
    device = _device()
    stack = NavigationStack(1, device)
    x = CELL_CENTER_ORIGIN_M + CELL_PITCH_M
    y = CELL_CENTER_ORIGIN_M + CELL_PITCH_M
    robot = torch.tensor([[x, y]], dtype=torch.float32, device=device)
    goal = torch.tensor([[x, y + 8.0 * CELL_PITCH_M]], device=device)
    stack.reset(torch.tensor([0], device=device), robot, goal)
    # North of the robot stays unknown. The only mapped route runs west, then north.
    _mark_door(stack, 1, 1, 0, 1, wall=True)
    _mark_door(stack, 0, 1, 1, 0, wall=True)
    _mark_door(stack, 1, 0, 0, 1, wall=False)
    _mark_door(stack, 1, 0, 1, 0, wall=False)
    _mark_door(stack, 0, 0, 1, 0, wall=True)
    _mark_door(stack, 2, 0, 0, 1, wall=True)
    waypoint, _path, _length = stack.plan_waypoint(0, robot[0], force=True)
    # The route turns, so the waypoint is the corner, not the cell past it.
    assert abs(float(waypoint[0]) - CELL_CENTER_ORIGIN_M) < 0.2
    assert abs(float(waypoint[1]) - y) < 0.2


def test_reverse_pocket_loses_to_the_corridor_ahead():
    """A side pocket behind the robot does not beat the open hall in front."""
    device = _device()
    stack = NavigationStack(1, device)
    x = CELL_CENTER_ORIGIN_M + CELL_PITCH_M
    y = CELL_CENTER_ORIGIN_M + CELL_PITCH_M
    robot = torch.tensor([[x, y]], dtype=torch.float32, device=device)
    goal = torch.tensor([[x, y + 8.0 * CELL_PITCH_M]], device=device)
    stack.reset(torch.tensor([0], device=device), robot, goal)
    # Facing +x. West then north is the geometrically better pocket.
    _mark_door(stack, 1, 1, 1, 0, wall=True)
    _mark_door(stack, 0, 1, 1, 0, wall=True)
    _mark_door(stack, 1, 0, 0, 1, wall=False)
    _mark_door(stack, 1, 0, 1, 0, wall=False)
    _mark_door(stack, 0, 0, 1, 0, wall=True)
    _mark_door(stack, 1, 1, 0, 1, wall=False)
    _mark_door(stack, 0, 2, 1, 0, wall=True)
    heading = torch.tensor([1.0, 0.0], device=device)
    waypoint, _path, _length = stack.plan_waypoint(0, robot[0], force=True, heading_xy=heading)
    assert abs(float(waypoint[0]) - (x + CELL_PITCH_M)) < 0.2
    assert abs(float(waypoint[1]) - y) < 0.2


def test_waypoint_follows_the_hall_away_from_the_goal():
    """The only open continuation runs west, so the waypoint does too."""
    device = _device()
    stack = NavigationStack(1, device)
    # Cell (1, 1). Goal is due north. The north and east doors are walls.
    x = CELL_CENTER_ORIGIN_M + CELL_PITCH_M
    y = CELL_CENTER_ORIGIN_M + CELL_PITCH_M
    robot = torch.tensor([[x, y]], dtype=torch.float32, device=device)
    goal = torch.tensor([[x, y + 8.0 * CELL_PITCH_M]], device=device)
    stack.reset(torch.tensor([0], device=device), robot, goal)
    _mark_door(stack, 1, 1, 1, 0, wall=True)
    _mark_door(stack, 1, 1, 0, 1, wall=True)
    _mark_door(stack, 0, 1, 1, 0, wall=True)
    _mark_door(stack, 1, 0, 0, 1, wall=False)
    _mark_door(stack, 1, 0, 1, 0, wall=False)
    _mark_door(stack, 0, 0, 1, 0, wall=True)
    _mark_door(stack, 2, 0, 0, 1, wall=True)
    waypoint, _path, _length = stack.plan_waypoint(0, robot[0], force=True)
    # The route turns, so the waypoint is the corner, not the cell past it.
    assert abs(float(waypoint[0]) - CELL_CENTER_ORIGIN_M) < 0.2
    assert abs(float(waypoint[1]) - y) < 0.2


def test_waypoint_uses_the_tile_cell_grid():
    """A tile offset and a 3.5 m pitch still put the waypoint on a cell centre."""
    device = _device()
    origin = torch.tensor([[40.0, 0.0]], dtype=torch.float32, device=device)
    pitch = torch.tensor([3.5], dtype=torch.float32, device=device)
    stack = NavigationStack(
        1,
        device,
        cell_origin_xy=origin,
        cell_pitch=pitch,
        grid_n=torch.tensor([8], dtype=torch.long, device=device),
    )
    robot = origin.clone()
    goal = origin + torch.tensor([[8.0 * 3.5, 0.0]], device=device)
    stack.reset(torch.tensor([0], device=device), robot, goal)
    _mark_door_at(stack, 0, 0, 0, 1, wall=False, origin=origin[0], pitch=3.5)
    _mark_door_at(stack, 0, 1, 0, 1, wall=False, origin=origin[0], pitch=3.5)
    _mark_door_at(stack, 0, 0, 1, 0, wall=True, origin=origin[0], pitch=3.5)
    _mark_door_at(stack, 0, 1, 1, 0, wall=True, origin=origin[0], pitch=3.5)
    _mark_door_at(stack, 0, 2, 1, 0, wall=True, origin=origin[0], pitch=3.5)
    waypoint, _path, _length = stack.plan_waypoint(0, robot[0], force=True)
    assert abs(float(waypoint[0]) - (40.0 + 2.0 * 3.5)) < 0.2
    assert abs(float(waypoint[1]) - 0.0) < 0.2


def _mark_door_at(
    stack: NavigationStack,
    row: int,
    col: int,
    drow: int,
    dcol: int,
    *,
    wall: bool,
    origin: torch.Tensor,
    pitch: float,
) -> None:
    """Paint one doorway on ``stack`` using that env's cell origin and pitch."""
    device = stack.grid.counts.device
    if dcol == 1:
        x = float(origin[0]) + pitch * (col + 0.5)
        y = float(origin[1]) + pitch * row
        offsets = torch.linspace(-0.7, 0.7, 15, device=device)
        pts = torch.stack((torch.full_like(offsets, x), y + offsets), dim=-1)
    else:
        x = float(origin[0]) + pitch * col
        y = float(origin[1]) + pitch * (row + 0.5)
        offsets = torch.linspace(-0.7, 0.7, 15, device=device)
        pts = torch.stack((x + offsets, torch.full_like(offsets, y)), dim=-1)
    shifts = torch.tensor([[dx, dy] for dx in (-0.05, 0.05) for dy in (-0.05, 0.05)], device=device)
    pts = (pts.unsqueeze(1) + shifts).reshape(1, -1, 2)
    cells = stack.grid.world_to_cell(pts)[0]
    if wall:
        stack.grid.counts[0, cells[:, 0], cells[:, 1]] = 4
        stack.grid.free_counts[0, cells[:, 0], cells[:, 1]] = 0
    else:
        stack.grid.free_counts[0, cells[:, 0], cells[:, 1]] = 4
        stack.grid.counts[0, cells[:, 0], cells[:, 1]] = 0


def test_occupancy_counts_only_valid_hits():
    """Two returns in one cell add two, and a miss does not touch the other environment."""
    grid = OccupancyGrid(2, "cpu", resolution=0.25, size=32)
    grid.reset(torch.tensor([0, 1]), torch.zeros(2, 2))
    hits = torch.full((2, 3, 3), float("inf"))
    hits[0, 0] = torch.tensor([1.0, 0.0, 0.5])
    hits[0, 1] = torch.tensor([1.0, 0.0, 0.5])
    sensor = torch.zeros(2, 3)
    sensor[:, 2] = 0.5
    grid.update(hits, sensor, free_samples=4)
    assert int(grid.counts[0, 20, 16]) == 2
    assert int(grid.counts[1].sum()) == 0
    assert int(grid.free_counts[0].sum()) > 0


def test_blocked_route_ignores_the_cell_under_the_robot():
    """A wall on the route counts, and a wall within 0.45 m of the robot does not."""
    stack = NavigationStack(1, "cpu")
    robot = torch.tensor([[10.0, 10.0]])
    stack.reset(torch.tensor([0]), robot, robot + torch.tensor([[5.0, 0.0]]))
    far = robot[0] + torch.tensor([2.0, 0.0])
    stack._route_n[0] = 2
    stack._route_xy[0, 0] = robot[0]
    stack._route_xy[0, 1:] = far
    cell = stack.grid.world_to_cell(far.view(1, 2))[0]
    stack.grid.counts[0, int(cell[0]), int(cell[1])] = 3
    assert bool(stack.blocked_routes(robot, far.view(1, 2))[0])

    stack.grid.counts.zero_()
    assert not bool(stack.blocked_routes(robot, far.view(1, 2))[0])

    near = robot[0] + torch.tensor([0.2, 0.0])
    near_cell = stack.grid.world_to_cell(near.view(1, 2))[0]
    stack.grid.counts[0, int(near_cell[0]), int(near_cell[1])] = 3
    assert not bool(stack.blocked_routes(robot, far.view(1, 2))[0])


def test_commit_prefix_stops_before_a_wall():
    """A blocked vertex is not part of the prefix."""
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    seen = torch.tensor([False, False, False, False])
    blocked = torch.tensor([False, False, True, False])
    prefix, _last_seen = commit_prefix(points, seen, blocked, probe_m=2.5, max_m=15.0)
    assert prefix.shape[0] == 2
    assert float(prefix[-1, 0]) == 1.0
