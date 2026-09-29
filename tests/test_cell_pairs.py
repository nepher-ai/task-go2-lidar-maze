# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Start and goal cells drawn for planner play."""

from __future__ import annotations

import torch
from go2_lidar_maze.constants import CELL_CENTER_ORIGIN_M, CELL_PITCH_M, GRID_SIZE
from go2_lidar_maze.nav_core.planner import astar_cells
from go2_lidar_maze.tasks.hl_maze.cell_pairs import CorridorGraph
from go2_lidar_maze.tasks.hl_maze.maze_layout import adjacency_from_blocked


def _indices(points: torch.Tensor) -> torch.Tensor:
    col = ((points[:, 0] - CELL_CENTER_ORIGIN_M) / CELL_PITCH_M).round().to(torch.long)
    row = ((points[:, 1] - CELL_CENTER_ORIGIN_M) / CELL_PITCH_M).round().to(torch.long)
    return row * GRID_SIZE + col


def test_open_grid_samples_distinct_cell_centres():
    """A fully open grid still returns two different cell centres per row."""
    neighbours = adjacency_from_blocked([])
    graph = CorridorGraph(neighbours)
    start, goal = graph.sample(32, torch.device("cpu"))
    assert start.shape == (32, 2)
    assert not torch.any(torch.all(start == goal, dim=-1))
    start_i = _indices(start)
    goal_i = _indices(goal)
    for s, g in zip(start_i.tolist(), goal_i.tolist()):
        assert astar_cells(neighbours, s, g, n_cols=GRID_SIZE) is not None
    for points in (start, goal):
        col = (points[:, 0] - CELL_CENTER_ORIGIN_M) / CELL_PITCH_M
        row = (points[:, 1] - CELL_CENTER_ORIGIN_M) / CELL_PITCH_M
        assert torch.all(col == col.round())
        assert torch.all(row == row.round())
        assert int(col.min()) >= 0 and int(col.max()) < GRID_SIZE
        assert int(row.min()) >= 0 and int(row.max()) < GRID_SIZE


def test_unreachable_pair_is_rejected():
    """A wall that splits the grid is never crossed."""
    blocked = []
    for row in range(GRID_SIZE):
        left = row * GRID_SIZE + 9
        right = left + 1
        blocked.append((left, right))
    neighbours = adjacency_from_blocked(blocked)
    graph = CorridorGraph(neighbours)
    start, goal = graph.sample(32, torch.device("cpu"))
    start_i = _indices(start)
    goal_i = _indices(goal)
    for s, g in zip(start_i.tolist(), goal_i.tolist()):
        assert astar_cells(neighbours, s, g, n_cols=GRID_SIZE) is not None
        assert (s % GRID_SIZE < 10) == (g % GRID_SIZE < 10)
