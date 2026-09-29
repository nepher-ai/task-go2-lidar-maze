# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scenario parsing for the go2-maze-v0 benchmark. No simulator."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "go2_lidar_maze"
    / "go2_lidar_maze"
    / "tasks"
    / "hl_maze"
    / "benchmark.py"
)
_SCENARIO_PATH = (
    Path(__file__).resolve().parents[2] / "envhub" / "environments" / "go2-maze-v0" / "maps" / "scenario.json"
)


def _layout_module():
    spec = importlib.util.spec_from_file_location("go2_benchmark_under_test", _BENCHMARK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _tile(origin, maze, pair=0):
    return {"tile": 0, "maze": maze, "pair": pair, "origin": [origin[0], origin[1], 0.0]}


def _scenario():
    """Two tiles: 8×8 at 3.0 m corridors, and 8×8 at 2.0 m on the next tile."""
    return {
        "metadata": {"wall_thickness_m": 0.5, "schema": 4},
        "mazes": [
            {
                "id": 0,
                "n": 8,
                "corridor_width_m": 3.0,
                "path_length_cells": 36,
                "spawn_goal_pairs": [
                    {
                        "spawn": {"pos": [26.5, 12.5, 0.4], "yaw": -1.5707963267948966},
                        "goal": {"pos": [2.0, 26.5, 0.0]},
                    }
                ],
            },
            {
                "id": 1,
                "n": 8,
                "corridor_width_m": 2.0,
                "spawn_goal_pairs": [
                    {
                        "spawn": {"pos": [1.5, 1.5, 0.4], "yaw": 0.0},
                        "goal": {"pos": [4.0, 1.5, 0.0]},
                    }
                ],
            },
        ],
        "tiles": [_tile((0.0, 0.0), 0), _tile((40.0, 0.0), 1)],
    }


def _on_cell(xy, origin, pitch) -> bool:
    col = (float(xy[0]) - float(origin[0])) / pitch
    row = (float(xy[1]) - float(origin[1])) / pitch
    return abs(col - round(col)) < 1e-4 and abs(row - round(row)) < 1e-4


def test_layout_places_spawns_on_the_derived_grid():
    """Hand-checked against enclosed_8x8_w300_00.txt and a 2.0 m corridor."""
    layout = _layout_module().layout_from_scenario(_scenario())
    assert layout.num_tiles == 2

    assert torch.allclose(layout.cell_pitch, torch.tensor([3.5, 2.5]))
    assert torch.equal(layout.grid_n, torch.tensor([8, 8]))
    assert torch.allclose(layout.cell_origin_xy[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(layout.cell_origin_xy[1], torch.tensor([41.5, 1.5]))

    assert torch.allclose(layout.spawn_xy[0], torch.tensor([26.5, 12.5]))
    assert torch.allclose(layout.goal_xy[0], torch.tensor([2.0, 26.5]))
    assert abs(float(layout.spawn_yaw[0]) - (-math.pi / 2)) < 1e-6
    assert _on_cell(layout.spawn_xy[0], layout.cell_origin_xy[0], 3.5)
    assert _on_cell(layout.goal_xy[0], layout.cell_origin_xy[0], 3.5)
    # 36 cells is 35 steps of 3.5 m.
    assert abs(float(layout.optimal_route_m[0]) - 122.5) < 1e-4

    assert torch.allclose(layout.spawn_xy[1], torch.tensor([41.5, 1.5]))
    assert torch.allclose(layout.goal_xy[1], torch.tensor([44.0, 1.5]))
    assert _on_cell(layout.spawn_xy[1], layout.cell_origin_xy[1], 2.5)
    assert _on_cell(layout.goal_xy[1], layout.cell_origin_xy[1], 2.5)
    assert math.isnan(float(layout.optimal_route_m[1]))


def test_ground_plane_covers_every_tile():
    """The visual plane has to reach the far corner, not stop at the origin tile."""
    layout = _layout_module().layout_from_scenario(_scenario())
    center_x, center_y, size_x, size_y = _layout_module().ground_plane_cover(
        layout.tile_origin_xy, layout.grid_n, layout.cell_pitch, margin_m=0.0
    )
    # 8 * 3.5 + 0.5 = 28.5 on the origin tile; the next tile ends at 40 + 8 * 2.5 + 0.5.
    assert abs(size_x - 60.5) < 1e-4
    assert abs(size_y - 28.5) < 1e-4
    assert abs(center_x - 30.25) < 1e-4
    assert abs(center_y - 14.25) < 1e-4
    repeated = _layout_module().layout_from_scenario(_scenario()).for_envs(3)
    again = _layout_module().ground_plane_cover(
        repeated.tile_origin_xy, repeated.grid_n, repeated.cell_pitch, margin_m=0.0
    )
    assert again == (center_x, center_y, size_x, size_y)


def test_for_envs_repeats_tiles():
    layout = _layout_module().layout_from_scenario(_scenario()).for_envs(3)
    assert layout.num_tiles == 3
    assert torch.allclose(layout.spawn_xy[2], layout.spawn_xy[0])
    assert torch.allclose(layout.cell_origin_xy[2], layout.cell_origin_xy[0])


def test_shipped_scenario_matches_enclosed_8x8_w300_00():
    """Tile 14 is enclosed_8x8_w300_00, whose route is 35 steps of 3.5 m."""
    layout = _layout_module().load_benchmark(_SCENARIO_PATH)
    assert layout.num_tiles == 60
    # Tile 14 sits at column 6, row 1 of the 40 m grid.
    assert torch.allclose(layout.tile_origin_xy[14], torch.tensor([240.0, 40.0]))
    assert torch.allclose(layout.spawn_xy[14], torch.tensor([266.5, 52.5]))
    assert abs(float(layout.optimal_route_m[14]) - 122.5) < 1e-3
    assert int(layout.grid_n[14]) == 8
    assert abs(float(layout.cell_pitch[14]) - 3.5) < 1e-4
