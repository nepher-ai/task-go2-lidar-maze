# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-torch checks for the reward scales, the velocity map, and both planners."""

from __future__ import annotations

import ast
import heapq
from pathlib import Path

import torch

from go2_lidar_maze.constants import (
    CELL_PITCH_M,
    GAMMA,
    collision_penalty,
    half_reward_sigma_sq,
    penalty_budget,
    v_safe,
)
from go2_lidar_maze.nav_core.occupancy import OccupancyGrid
from go2_lidar_maze.nav_core.path_utils import (
    cumulative_length,
    fillet_polyline,
    high_water,
    point_at_distance,
    project_monotone,
)
from go2_lidar_maze.nav_core.planner import astar_cells, astar_grid, path_length_m
from go2_lidar_maze.nav_core.stack import NavigationStack, select_waypoint_on_path
from go2_lidar_maze.tasks.hl_maze.maze_layout import MazeLayout, adjacency_from_blocked, index_of
from go2_lidar_maze.tasks.hl_maze.scenario import parse_start_goal
from go2_lidar_maze.velocity_map import map_action_to_velocity, rate_limit

_PACKAGE = Path(__file__).resolve().parents[1] / "source" / "go2_lidar_maze" / "go2_lidar_maze"
_TXT = _PACKAGE / "assets" / "maze" / "enclosed_20x20.txt"


def test_half_reward_sigmas_match_the_derivation():
    # sigma^2 = e_1/2^2 / ln 2. Linear 0.35 m/s -> 0.177. Yaw 0.30 rad/s -> 0.130.
    assert abs(half_reward_sigma_sq(0.35) - 0.177) < 1e-3
    assert abs(half_reward_sigma_sq(0.30) - 0.130) < 1e-3


def test_safe_speed_matches_the_hand_calculation():
    # 1 m of clearance, 0.3 m margin, a_b = 2.5, tau = 0.2 -> 1.44 m/s.
    assert abs(v_safe(1.0) - 1.436) < 0.01
    # 2 m of clearance -> 2.46 m/s.
    assert abs(v_safe(2.0) - 2.458) < 0.01
    assert v_safe(0.2) == 0.0


def test_collision_penalty_dominates_continuing():
    worst = penalty_budget()
    assert abs(worst - 0.9) < 1e-9
    continuing = worst / (1.0 - GAMMA)
    penalty = collision_penalty()
    assert penalty > continuing
    assert abs(penalty - 112.5) < 1e-6


def test_action_mapping_and_rate_limit():
    stopped = map_action_to_velocity(torch.zeros(1, 3))
    assert torch.allclose(stopped, torch.zeros(1, 3))
    forward = map_action_to_velocity(torch.tensor([[1.0, -1.0, 1.0]]))
    assert torch.allclose(forward, torch.tensor([[3.0, -1.0, 2.0]]))
    reverse = map_action_to_velocity(torch.tensor([[-1.0, 0.5, -0.5]]))
    assert torch.allclose(reverse, torch.tensor([[-2.0, 0.5, -1.0]]))
    clipped = map_action_to_velocity(torch.tensor([[2.0, -3.0, 4.0]]))
    assert torch.allclose(clipped, torch.tensor([[3.0, -1.0, 2.0]]))
    limited = rate_limit(
        torch.tensor([3.0, 1.0, 2.0]),
        torch.zeros(3),
        torch.tensor([0.3, 0.3, 0.6]),
    )
    assert torch.allclose(limited, torch.tensor([0.3, 0.3, 0.6]))


def test_projection_and_high_water_progress():
    line = torch.tensor([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]])
    arc, cross, _point = project_monotone(line, torch.tensor([2.0, 0.5]), s_prev=0.0)
    assert abs(float(arc) - 2.0) < 1e-4
    assert abs(float(cross) - 0.5) < 1e-4
    # The window does not let the projection jump backward.
    arc_back, _, _ = project_monotone(line, torch.tensor([0.0, 0.0]), s_prev=3.0, window_m=1.0)
    assert float(arc_back) >= 3.0 - 1e-4
    mark, progress = high_water(torch.tensor([1.0]), torch.tensor([0.4]))
    assert float(progress) == 0.0
    assert float(mark) == 1.0
    mark, progress = high_water(mark, torch.tensor([2.5]))
    assert abs(float(progress) - 1.5) < 1e-5
    smoothed = fillet_polyline(line, radius=0.5)
    assert torch.allclose(smoothed[0], line[0])
    assert torch.allclose(smoothed[-1], line[-1])
    assert float(cumulative_length(smoothed)[-1]) < float(cumulative_length(line)[-1])


def test_cell_astar_matches_bfs():
    # 5x5 open grid with the middle of the first row walled off from its right neighbour.
    neighbours = adjacency_from_blocked([(index_of(0, 2, 5), index_of(0, 3, 5))], n=5)
    start, goal = index_of(0, 0, 5), index_of(0, 4, 5)
    path = astar_cells(neighbours, start, goal, n_cols=5)
    assert path is not None
    assert len(path) - 1 == _bfs_length(neighbours, start, goal)
    # The blocked edge forces a detour: the straight row is 4 steps, the route is longer.
    assert len(path) - 1 > 4
    assert astar_cells(neighbours, start, 99, n_cols=5) is None


def test_grid_astar_matches_dijkstra():
    occupied = torch.zeros(8, 8, dtype=torch.bool)
    occupied[3, 2:6] = True
    start, goal = (0, 0), (7, 7)
    path = astar_grid(occupied, start, goal, resolution=0.25, clearance_weight=0.0, unknown_cost=1.0)
    assert path is not None
    assert path[0] == start and path[-1] == goal
    assert abs(path_length_m(path, 0.25) - _dijkstra_length(occupied, start, goal, 0.25)) < 1e-6
    # A straight diagonal would cross the wall. The path must go around it.
    assert any(cell[0] >= 4 or cell[1] <= 1 or cell[1] >= 6 for cell in path)


def test_privileged_samples_stay_inside_one_to_five_cells():
    neighbours = adjacency_from_blocked([], n=6)
    layout = MazeLayout(neighbours, n=6, n_cols=6)
    for start in range(layout.num_cells):
        for k in range(1, 6):
            if not layout.has_distance(start, k):
                continue
            target, path = layout.sample(start, k, choice=start + k)
            assert len(path) - 1 == k
            assert path[0] == start and path[-1] == target
            assert layout.path_length(start, target) == k


def test_online_selector_respects_the_lookahead():
    short = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    assert torch.allclose(select_waypoint_on_path(short), short[-1])
    long = torch.stack((torch.linspace(0.0, 30.0, 31), torch.zeros(31)), dim=-1)
    waypoint = select_waypoint_on_path(long)
    # The point lies on the polyline at 5 cells (12.5 m).
    along = point_at_distance(long, 5 * CELL_PITCH_M)
    assert torch.allclose(waypoint, along, atol=1e-4)
    assert abs(float(torch.linalg.norm(waypoint - long[0])) - 5 * CELL_PITCH_M) < 1e-3

    stack = NavigationStack(num_envs=1, device="cpu")
    robot = torch.tensor([[0.0, 0.0]])
    near = torch.tensor([[1.0, 0.0]])
    far = torch.tensor([[20.0, 0.0]])
    ids = torch.tensor([0])
    stack.reset(ids, robot, near)
    near_wp, near_path, _ = stack.plan_waypoint(0, robot[0], force=True)
    assert torch.allclose(near_wp, near_path[-1], atol=0.3)
    assert float(cumulative_length(near_path)[-1]) <= CELL_PITCH_M + 0.5
    stack.reset(ids, robot, far)
    far_wp, far_path, _ = stack.plan_waypoint(0, robot[0], force=True)
    far_arc = _arc_to_point(far_path, far_wp)
    assert CELL_PITCH_M - 0.3 <= far_arc <= 5 * CELL_PITCH_M + 0.3


def test_occupancy_records_hits_and_free_space():
    grid = OccupancyGrid(num_envs=1, device="cpu", resolution=0.2, size=40)
    grid.reset(torch.tensor([0]), torch.zeros(1, 2))
    hits = torch.full((1, 2, 3), float("nan"))
    hits[0, 0] = torch.tensor([2.0, 0.0, 0.5])
    sensor = torch.tensor([[0.0, 0.0, 0.2]])
    grid.update(hits, sensor)
    cells = grid.world_to_cell(hits[..., :2])
    row, col = int(cells[0, 0, 0]), int(cells[0, 0, 1])
    assert bool(grid.obstacle_mask()[0, row, col])
    assert bool(grid.seen_free_mask()[0].any())
    assert not bool(grid.seen_free_mask()[0, row, col])


def test_play_parser_ignores_the_path_line():
    text = _TXT.read_text(encoding="utf-8")
    start_x, start_y, end_x, end_y = parse_start_goal(text)
    assert (start_x, start_y, end_x, end_y) == (41.5, 1.5, 49.0, 49.0)
    forged = text.replace("path:", "path: (0,0) -> (19,19)\nignored_path:")
    assert parse_start_goal(forged) == (start_x, start_y, end_x, end_y)


def test_play_sources_do_not_import_the_privileged_layout():
    """The online stack and the command module must not import the maze graph at import time."""
    root = _PACKAGE
    for relative in (
        "nav_core/stack.py",
        "nav_core/planner.py",
        "nav_core/occupancy.py",
        "tasks/hl_maze/scenario.py",
        "tasks/hl_maze/mdp/waypoint_command.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and "maze_layout" in node.module:
                raise AssertionError(f"{relative} imports maze_layout at module level")
    command = (root / "tasks" / "hl_maze" / "mdp" / "waypoint_command.py").read_text(encoding="utf-8")
    assert "MazeLayout" in command
    # The only import sits inside the privileged builder, which Play never calls.
    assert "from go2_lidar_maze.tasks.hl_maze.maze_layout import MazeLayout" in command


def _bfs_length(neighbours, start: int, goal: int) -> int | None:
    if start == goal:
        return 0
    queue = [(start, 0)]
    seen = {start}
    while queue:
        node, dist = queue.pop(0)
        for nxt in neighbours[node]:
            if nxt in seen:
                continue
            if nxt == goal:
                return dist + 1
            seen.add(nxt)
            queue.append((nxt, dist + 1))
    return None


def _dijkstra_length(occupied: torch.Tensor, start: tuple[int, int], goal: tuple[int, int], resolution: float) -> float:
    height, width = occupied.shape
    heap = [(0.0, start)]
    best = {start: 0.0}
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > best.get(node, float("inf")):
            continue
        if node == goal:
            return dist
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            nxt = (node[0] + dr, node[1] + dc)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width) or bool(occupied[nxt]):
                continue
            if dr != 0 and dc != 0 and (bool(occupied[node[0] + dr, node[1]]) or bool(occupied[node[0], node[1] + dc])):
                continue
            step = resolution * (2.0**0.5 if dr != 0 and dc != 0 else 1.0)
            new = dist + step
            if new < best.get(nxt, float("inf")):
                best[nxt] = new
                heapq.heappush(heap, (new, nxt))
    raise AssertionError("dijkstra found no path")


def _arc_to_point(path: torch.Tensor, point: torch.Tensor) -> float:
    best = float("inf")
    cum = cumulative_length(path)
    for index in range(path.shape[0] - 1):
        a, b = path[index], path[index + 1]
        ab = b - a
        denom = torch.dot(ab, ab).clamp(min=1e-8)
        t = torch.clamp(torch.dot(point - a, ab) / denom, 0.0, 1.0)
        closest = a + t * ab
        dist = float(torch.linalg.norm(point - closest))
        if dist < best:
            best = dist
            arc = float(cum[index] + t * (cum[index + 1] - cum[index]))
    return arc
