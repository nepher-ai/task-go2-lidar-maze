# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Distance field, corner blocking, and occupancy-grid waypoints."""

from __future__ import annotations

import math

import torch
from go2_lidar_maze.nav_core.occupancy import OccupancyGrid
from go2_lidar_maze.nav_core.path_utils import commit_prefix
from go2_lidar_maze.nav_core.planner import (
    UNREACHABLE,
    astar_grid,
    distance_field,
    distance_field_batch,
    trace_route,
    trace_route_batch,
)
from go2_lidar_maze.nav_core.stack import NavigationStack, _crop_bounds


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


def _paint_segment(
    stack: NavigationStack,
    start: tuple[float, float],
    end: tuple[float, float],
    radius: float,
    *,
    wall: bool,
) -> None:
    """Stamp a disk of free space or wall along a world segment."""
    grid = stack.grid
    res = float(grid.res[0])
    origin = grid.origin[0]
    half = grid.size // 2
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    steps = max(1, int(length / 0.15))
    span = int(math.ceil(radius / res)) + 1
    for index in range(steps + 1):
        t = index / steps
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        rel = torch.tensor([[x, y]], dtype=torch.float32, device=grid.counts.device)
        cell = grid.world_to_cell(rel)[0]
        r0, c0 = int(cell[0]), int(cell[1])
        for dr in range(-span, span + 1):
            for dc in range(-span, span + 1):
                row, col = r0 + dr, c0 + dc
                if row < 0 or col < 0 or row >= grid.size or col >= grid.size:
                    continue
                wx = (row - half + 0.5) * res + float(origin[0])
                wy = (col - half + 0.5) * res + float(origin[1])
                if math.hypot(wx - x, wy - y) > radius:
                    continue
                if wall:
                    grid.counts[0, row, col] = grid.hit_threshold
                    grid.free_counts[0, row, col] = 0
                elif int(grid.counts[0, row, col]) < grid.hit_threshold:
                    grid.free_counts[0, row, col] = 1


def _plan(robot: torch.Tensor, goal: torch.Tensor, segments: list[tuple], heading: torch.Tensor | None = None):
    device = robot.device
    stack = NavigationStack(1, device)
    stack.reset(torch.tensor([0], device=device), robot, goal)
    for start, end, radius, wall in segments:
        _paint_segment(stack, start, end, radius, wall=wall)
    waypoint, path, length = stack.plan_waypoint(0, robot[0], force=True, heading_xy=heading)
    return stack, waypoint, path, length


def test_waypoint_follows_seen_free_space_around_a_wall():
    """The straight line is a wall. The waypoint takes the mapped detour."""
    device = _device()
    robot = torch.tensor([[0.0, 0.0]], device=device)
    goal = torch.tensor([[10.0, 0.0]], device=device)
    _stack, waypoint, path, _length = _plan(
        robot,
        goal,
        [
            ((0.0, 0.0), (0.0, 4.0), 0.8, False),
            ((0.0, 4.0), (10.0, 4.0), 0.8, False),
            ((10.0, 4.0), (10.0, 0.0), 0.8, False),
            ((3.0, -1.5), (3.0, 1.5), 0.35, True),
        ],
    )
    # The carrot stops about 1 m short of the corner, still on the north hall.
    assert float(waypoint[1]) > 2.0
    assert not bool(((path[:, 0] > 2.5) & (path[:, 1] < 2.0)).any())


def test_unknown_space_does_not_freeze_a_walkable_hall():
    """Unscanned cells toward the goal lose to a hall that is already free."""
    device = _device()
    robot = torch.tensor([[0.0, 0.0]], device=device)
    goal = torch.tensor([[0.0, 10.0]], device=device)
    _stack, waypoint, _path, _length = _plan(
        robot,
        goal,
        [
            ((0.0, 0.0), (-4.0, 0.0), 0.8, False),
            ((-4.0, 0.0), (-4.0, 10.0), 0.8, False),
        ],
    )
    # The carrot is the straight start of the hall, not a point up the unscanned line.
    assert float(waypoint[0]) < -0.05
    assert float(waypoint[1]) < 2.0


def test_reverse_pocket_loses_to_the_corridor_ahead():
    """A pocket behind the robot does not beat open space in front."""
    device = _device()
    robot = torch.tensor([[0.0, 0.0]], device=device)
    goal = torch.tensor([[0.0, 12.0]], device=device)
    heading = torch.tensor([1.0, 0.0], device=device)
    _stack, waypoint, _path, _length = _plan(
        robot,
        goal,
        [
            ((0.0, 0.0), (3.0, 0.0), 0.8, False),
            ((3.0, 0.0), (3.0, 12.0), 0.8, False),
            ((0.0, 0.0), (-3.0, 0.0), 0.8, False),
            ((-3.0, 0.0), (-3.0, 12.0), 0.8, False),
        ],
        heading,
    )
    assert float(waypoint[0]) > 1.0


def test_waypoint_follows_the_hall_away_from_the_goal():
    """A wall in front of the goal is not a route. The unseen way around is not commanded."""
    device = _device()
    robot = torch.tensor([[0.0, 0.0]], device=device)
    goal = torch.tensor([[8.0, 0.0]], device=device)
    _stack, waypoint, _path, _length = _plan(
        robot,
        goal,
        [
            ((0.0, 0.0), (-8.0, 0.0), 0.7, False),
            ((0.6, -2.0), (0.6, 2.0), 0.3, True),
        ],
    )
    # The west hall does not reach the goal, and the detour is still unscanned.
    # Either hold, or step west. Do not point through the wall.
    assert float(waypoint[0]) < 0.5


def test_waypoint_ignores_the_maze_lattice():
    """The waypoint is a metric point on free space, not a cell centre."""
    device = _device()
    robot = torch.tensor([[0.2, 0.3]], device=device)
    goal = torch.tensor([[12.0, 0.3]], device=device)
    stack, waypoint, _path, length = _plan(
        robot,
        goal,
        [((0.2, 0.3), (12.0, 0.3), 0.8, False)],
    )
    assert not hasattr(stack, "cell_origin_xy")
    assert 4.0 < float(waypoint[0]) < 6.0
    assert abs(float(waypoint[1]) - 0.3) < 0.4
    assert length <= 5.05


def test_unseen_space_is_not_a_route():
    """An empty map does not step toward the goal through unscanned space."""
    device = _device()
    robot = torch.tensor([[0.0, 0.0]], device=device)
    goal = torch.tensor([[8.0, 0.0]], device=device)
    stack = NavigationStack(1, device)
    stack.reset(torch.tensor([0], device=device), robot, goal)
    waypoint, _path, _length = stack.plan_waypoint(0, robot[0], force=True)
    assert float(torch.linalg.norm(waypoint - robot[0])) < 0.05


def test_short_stub_toward_the_goal_loses_to_the_hall():
    """The gap in front of a wall is not a corridor, even when it faces the goal."""
    device = _device()
    robot = torch.tensor([[0.0, 0.0]], device=device)
    goal = torch.tensor([[10.0, 0.0]], device=device)
    _stack, waypoint, _path, _length = _plan(
        robot,
        goal,
        [
            ((0.0, 0.0), (0.0, 0.0), 1.2, False),
            ((0.0, 0.0), (0.0, 8.0), 0.8, False),
            ((1.4, -2.0), (1.4, 2.0), 0.3, True),
        ],
    )
    assert float(waypoint[1]) > 1.5
    assert float(waypoint[0]) < 1.0


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


def test_distance_field_batch_matches_the_single_grid():
    """The batched field agrees with the single-grid field at the robot."""
    occupied = torch.zeros((7, 7), dtype=torch.bool)
    occupied[3, 1:] = True
    single = distance_field(occupied, (1, 3), 1.0, robot=(5, 3), max_iters=80, settle_iters=8)
    batch = distance_field_batch(
        occupied.unsqueeze(0),
        torch.tensor([[1, 3]]),
        1.0,
        robot=torch.tensor([[5, 3]]),
        max_iters=80,
    )
    assert abs(float(batch[0, 5, 3]) - float(single[5, 3])) < 1e-2
    assert float(batch[0, 5, 3]) < UNREACHABLE * 0.5


def _maze_free(x: float, y: float) -> bool:
    """Dead-end hall toward the goal, with the real door off to the side.

    The north hall is long and points at the goal, and it is capped. The east
    door is shorter than that view, so a longest-ray planner never takes it.
    Corridors are 3 m so one cell of inflation cannot pinch them shut.
    """
    if 1.0 <= x <= 4.0 and 1.0 <= y <= 11.0:
        return True
    if 4.0 <= x <= 8.0 and 1.0 <= y <= 4.0:
        return True
    if 5.0 <= x <= 8.0 and 1.0 <= y <= 15.0:
        return True
    if 1.0 <= x <= 8.0 and 12.0 <= y <= 15.0:
        return True
    return False


def _cast_scan(origin: torch.Tensor, beams: int = 180, max_range: float = 9.5) -> torch.Tensor:
    """2D ray cast against :func:`_maze_free`. Misses stay non-finite."""
    hits = torch.full((1, beams, 3), float("inf"), dtype=torch.float32)
    ox, oy = float(origin[0]), float(origin[1])
    for index, angle in enumerate(torch.linspace(0.0, 2.0 * math.pi, beams + 1)[:-1].tolist()):
        dx, dy = math.cos(angle), math.sin(angle)
        last_free = None
        distance = 0.08
        while distance <= max_range:
            if not _maze_free(ox + dx * distance, oy + dy * distance):
                break
            last_free = distance
            distance += 0.08
        if last_free is None or last_free >= max_range - 0.08:
            continue
        hits[0, index, 0] = ox + dx * last_free
        hits[0, index, 1] = oy + dy * last_free
        hits[0, index, 2] = 0.5
    return hits


def _greedy_opening(stack: NavigationStack, robot: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """The ray fan: among the longest cleared rays, pick the one nearest the goal."""
    grid = stack.grid
    res = float(grid.res[0])
    origin = grid.origin[0]
    half = grid.size // 2
    counts = grid.counts[0]
    free = grid.free_counts[0]
    blocked = (counts >= grid.hit_threshold) | (free <= 0)
    n_rays = 72
    step = 0.25
    reach = 5.0
    angles = torch.arange(n_rays, dtype=torch.float32) * (2.0 * math.pi / n_rays)
    best = None
    best_dist = None
    longest = 0.0
    openings = []
    for angle in angles.tolist():
        direction = (math.cos(angle), math.sin(angle))
        free_dist = 0.0
        distance = step
        while distance <= reach:
            x = float(robot[0]) + direction[0] * distance
            y = float(robot[1]) + direction[1] * distance
            rel_x = (x - float(origin[0])) / res
            rel_y = (y - float(origin[1])) / res
            row = int(math.floor(rel_x)) + half
            col = int(math.floor(rel_y)) + half
            if row < 0 or col < 0 or row >= grid.size or col >= grid.size or bool(blocked[row, col]):
                break
            free_dist = distance
            distance += step
        if free_dist < 1.75:
            continue
        longest = max(longest, free_dist)
        openings.append((direction, free_dist))
    for direction, free_dist in openings:
        if free_dist < longest - 0.75:
            continue
        travel = min(free_dist, reach) if free_dist >= reach - 1e-6 else max(free_dist - 0.75, 0.5)
        end_x = float(robot[0]) + direction[0] * travel
        end_y = float(robot[1]) + direction[1] * travel
        goal_dist = math.hypot(end_x - float(goal[0]), end_y - float(goal[1]))
        if best_dist is None or goal_dist < best_dist:
            best_dist = goal_dist
            best = torch.tensor([end_x, end_y], dtype=torch.float32, device=robot.device)
    if best is None:
        return robot.clone()
    return best


def _roll(robot: torch.Tensor, waypoint: torch.Tensor, speed: float = 0.4) -> torch.Tensor:
    delta = waypoint - robot
    dist = float(torch.linalg.norm(delta))
    if dist < 0.2:
        return robot
    return robot + delta / dist * min(speed, dist)


def test_closed_loop_leaves_a_goal_facing_dead_end():
    """A mapped dead end does not trap the robot. The longest-ray rule does."""
    device = _device()
    start = torch.tensor([[2.5, 2.0]], device=device)
    goal = torch.tensor([[2.5, 13.5]], device=device)
    sensor = torch.tensor([[2.5, 2.0, 0.5]], device=device)

    stack = NavigationStack(1, device)
    stack.reset(torch.tensor([0], device=device), start, goal)
    robot = start[0].clone()
    heading = torch.tensor([0.0, 1.0], device=device)
    reached = False
    for _step in range(180):
        sensor[0, :2] = robot
        stack.update(_cast_scan(robot).to(device), sensor)
        waypoint, _path, _length = stack.plan_waypoint(0, robot, force=True, heading_xy=heading)
        if float(torch.linalg.norm(robot - goal[0])) < 0.8:
            reached = True
            break
        nxt = _roll(robot, waypoint)
        moved = nxt - robot
        if float(torch.linalg.norm(moved)) > 1e-4:
            heading = moved / torch.linalg.norm(moved)
        robot = nxt
    assert reached, f"planner stopped at {robot.tolist()}"

    greedy = NavigationStack(1, device)
    greedy.reset(torch.tensor([0], device=device), start, goal)
    robot = start[0].clone()
    closest = float(torch.linalg.norm(robot - goal[0]))
    for _step in range(180):
        sensor[0, :2] = robot
        greedy.update(_cast_scan(robot).to(device), sensor)
        waypoint = _greedy_opening(greedy, robot, goal[0])
        robot = _roll(robot, waypoint)
        closest = min(closest, float(torch.linalg.norm(robot - goal[0])))
    assert closest > 2.0


def test_commit_prefix_stops_before_a_wall():
    """A blocked vertex is not part of the prefix."""
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    seen = torch.tensor([False, False, False, False])
    blocked = torch.tensor([False, False, True, False])
    prefix, _last_seen = commit_prefix(points, seen, blocked, probe_m=2.5, max_m=15.0)
    assert prefix.shape[0] == 2
    assert float(prefix[-1, 0]) == 1.0


def test_batched_field_trace_and_crops_match_one_env():
    """A joint batch matches one grid, one trace, and one crop at a time.

    Crop sizes and pool factors differ across the batch. Cells outside an
    environment's own crop must not change its costs.
    """
    device = _device()
    torch.manual_seed(4)
    batch, height, width = 5, 21, 17
    occupied = torch.rand((batch, height, width), device=device) < 0.2
    goal = torch.randint(0, height, (batch, 2), device=device)
    goal[:, 1].clamp_(max=width - 1)
    rows = torch.arange(batch, device=device)
    occupied[rows, goal[:, 0], goal[:, 1]] = False
    step = torch.where(torch.rand((batch, height, width), device=device) < 0.3, 3.0, 1.0)
    batched = distance_field_batch(occupied, goal, 0.25, step_cost=step, max_iters=400, check_every=400)
    for index in range(batch):
        single = distance_field(
            occupied[index],
            (int(goal[index, 0]), int(goal[index, 1])),
            0.25,
            step_cost=step[index],
            max_iters=400,
            check_every=400,
        )
        assert torch.allclose(batched[index], single, atol=1e-3, rtol=0.0)
    starts = torch.randint(0, height, (batch, 2), device=device)
    starts[:, 1].clamp_(max=width - 1)
    cells, counts = trace_route_batch(batched, starts, resolution=0.25, max_length_m=6.0, max_steps=40)
    for index in range(batch):
        route = trace_route(
            batched[index],
            (int(starts[index, 0]), int(starts[index, 1])),
            resolution=0.25,
            max_length_m=6.0,
            max_steps=40,
        )
        taken = int(counts[index])
        assert cells[index, :taken].tolist() == [list(cell) for cell in route]

    count = 3
    stack = NavigationStack(count, device)
    robots = torch.tensor([[10.0, 10.0], [12.0, 8.0], [30.0, 40.0]], device=device)
    goals = torch.tensor([[18.0, 14.0], [40.0, 20.0], [70.0, 10.0]], device=device)
    ids = torch.arange(count, device=device)
    stack.reset(ids, robots, goals)
    stack.grid.res[:] = torch.tensor([0.25, 0.25, 0.125], device=device)
    for index in range(count):
        robot = robots[index]
        for step_i in range(12):
            point = (robot + torch.tensor([0.4 * step_i, 0.15 * step_i], device=device)).view(1, 2)
            cell = stack.grid.world_to_cell(point)[index]
            stack.grid.free_counts[index, int(cell[0]), int(cell[1])] = 4
        wall = (robot + torch.tensor([1.2, 2.0], device=device)).view(1, 2)
        cell = stack.grid.world_to_cell(wall)[index]
        stack.grid.counts[index, int(cell[0]), int(cell[1])] = 3
    heading = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]], device=device)
    known_lo, known_hi, _has_free = stack.grid.known_bounds(ids)
    r0, c0, r1, c1, factor, robot_rc, goal_rc = _crop_bounds(
        robots,
        goals,
        stack.grid.origin,
        stack.grid.res,
        stack.grid._half,
        stack.grid.size,
        known_lo,
        known_hi,
    )
    height_c = r1 - r0
    width_c = c1 - c0
    for value in factor.unique().tolist():
        pick = (factor == int(value)).nonzero(as_tuple=False).flatten()
        group = stack._build_group(
            pick,
            r0[pick],
            c0[pick],
            height_c[pick],
            width_c[pick],
            int(value),
            robot_rc[pick],
            goal_rc[pick],
            stack.grid.res[pick],
            stack.grid.origin[pick],
            robots[pick],
            heading[pick],
        )
        for local, index in enumerate(pick.tolist()):
            slot = pick[local : local + 1]
            solo = stack._build_group(
                slot,
                r0[slot],
                c0[slot],
                height_c[slot],
                width_c[slot],
                int(value),
                robot_rc[slot],
                goal_rc[slot],
                stack.grid.res[slot],
                stack.grid.origin[slot],
                robots[slot],
                heading[slot],
            )
            crop_h = int(solo["hc"][0])
            crop_w = int(solo["wc"][0])
            assert torch.equal(group["occupied"][local, :crop_h, :crop_w], solo["occupied"][0])
            assert torch.equal(group["seen"][local, :crop_h, :crop_w], solo["seen"][0])
            assert torch.allclose(group["step"][local, :crop_h, :crop_w], solo["step"][0], atol=1e-5, rtol=0.0)
