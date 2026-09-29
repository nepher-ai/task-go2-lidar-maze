# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Random start and goal cells that A* can join."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from go2_lidar_maze.constants import CELL_CENTER_ORIGIN_M, CELL_PITCH_M, GRID_SIZE
from go2_lidar_maze.nav_core.planner import astar_cells


class CorridorGraph:
    """4-connected maze used to accept a random start and goal."""

    def __init__(self, neighbours: Sequence[Sequence[int]]):
        if not any(neighbours):
            raise ValueError("No corridor cell can reach a different cell.")
        self._neighbours = [list(row) for row in neighbours]

    def sample(self, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return start and goal cell centres ``(count, 2)`` [m].

        Each row draws two different cells uniformly. The pair is kept only
        when A* finds a path. Otherwise both cells are drawn again.
        """
        num_cells = len(self._neighbours)
        start = torch.empty(count, dtype=torch.long)
        goal = torch.empty(count, dtype=torch.long)
        for row in range(count):
            for _ in range(num_cells * num_cells):
                start_cell = int(torch.randint(0, num_cells, ()).item())
                goal_cell = int(torch.randint(0, num_cells - 1, ()).item())
                if goal_cell >= start_cell:
                    goal_cell += 1
                path = astar_cells(self._neighbours, start_cell, goal_cell, n_cols=GRID_SIZE)
                if path is not None:
                    start[row] = start_cell
                    goal[row] = goal_cell
                    break
            else:
                raise RuntimeError("Could not draw a start and goal that A* can join.")
        return _cell_xy(start.to(device)), _cell_xy(goal.to(device))


def _cell_xy(index: torch.Tensor) -> torch.Tensor:
    """World XY [m] of flat cell indices. Column is ``x``, row is ``y``."""
    row = torch.div(index, GRID_SIZE, rounding_mode="floor")
    col = index - row * GRID_SIZE
    return torch.stack(
        (
            CELL_CENTER_ORIGIN_M + CELL_PITCH_M * col.to(torch.float32),
            CELL_CENTER_ORIGIN_M + CELL_PITCH_M * row.to(torch.float32),
        ),
        dim=-1,
    )
