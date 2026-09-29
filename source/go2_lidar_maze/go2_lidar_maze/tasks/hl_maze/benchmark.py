# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parse an EnvHub ``go2-maze-v0`` scenario into per-tile tensors.

No simulator imports. Cell geometry follows the enclosed-maze generator:
walls are ``wall_thickness_m`` thick (0.5 m), so the corridor pitch is
``corridor_width_m + wall_thickness_m`` and cell ``(0, 0)`` sits at
``tile_origin + wall_thickness_m + corridor_width_m / 2``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

_DEFAULT_WALL_M = 0.5


@dataclass
class BenchmarkLayout:
    """One row per tile, in scenario order.

    Positions are world-frame metres. ``optimal_route_m`` is NaN when the
    scenario has no ``path_length_cells`` for that maze.
    """

    spawn_xy: torch.Tensor
    spawn_yaw: torch.Tensor
    goal_xy: torch.Tensor
    cell_origin_xy: torch.Tensor
    cell_pitch: torch.Tensor
    grid_n: torch.Tensor
    tile_origin_xy: torch.Tensor
    optimal_route_m: torch.Tensor

    @property
    def num_tiles(self) -> int:
        """Number of tiles in the scenario."""
        return int(self.spawn_xy.shape[0])

    def for_envs(self, num_envs: int) -> BenchmarkLayout:
        """Repeat tiles so environment ``i`` uses tile ``i % num_tiles``."""
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        index = torch.arange(num_envs) % self.num_tiles
        return BenchmarkLayout(
            spawn_xy=self.spawn_xy[index],
            spawn_yaw=self.spawn_yaw[index],
            goal_xy=self.goal_xy[index],
            cell_origin_xy=self.cell_origin_xy[index],
            cell_pitch=self.cell_pitch[index],
            grid_n=self.grid_n[index],
            tile_origin_xy=self.tile_origin_xy[index],
            optimal_route_m=self.optimal_route_m[index],
        )


def ground_plane_cover(
    tile_origin_xy: torch.Tensor,
    grid_n: torch.Tensor,
    cell_pitch: torch.Tensor,
    *,
    wall_m: float = _DEFAULT_WALL_M,
    margin_m: float = 1.0,
) -> tuple[float, float, float, float]:
    """Center and size [m] of a rectangle that covers every tile.

    Returns ``(center_x, center_y, size_x, size_y)``. Repeated tile origins are
    covered once. ``margin_m`` is added on each side. A tile spans
    ``n * cell_pitch + wall_m`` from its origin.
    """
    if tile_origin_xy.shape[0] == 0:
        raise ValueError("No tiles to cover with the ground plane.")
    seen: set[tuple[float, float]] = set()
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for index in range(int(tile_origin_xy.shape[0])):
        origin = (round(float(tile_origin_xy[index, 0]), 3), round(float(tile_origin_xy[index, 1]), 3))
        if origin in seen:
            continue
        seen.add(origin)
        span = float(grid_n[index]) * float(cell_pitch[index]) + wall_m
        min_x = min(min_x, origin[0])
        min_y = min(min_y, origin[1])
        max_x = max(max_x, origin[0] + span)
        max_y = max(max_y, origin[1] + span)
    min_x -= margin_m
    min_y -= margin_m
    max_x += margin_m
    max_y += margin_m
    return ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5, max_x - min_x, max_y - min_y)


def load_benchmark(path: str | Path) -> BenchmarkLayout:
    """Load ``scenario.json`` and return one row per tile."""
    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"scenario.json not found at '{scenario_path}'.")
    with open(scenario_path, encoding="utf-8") as handle:
        data = json.load(handle)
    return layout_from_scenario(data)


def layout_from_scenario(data: dict) -> BenchmarkLayout:
    """Build a :class:`BenchmarkLayout` from a parsed scenario dict."""
    meta = data.get("metadata", {})
    wall_m = float(meta.get("wall_thickness_m", _DEFAULT_WALL_M))
    mazes = {int(maze["id"]): maze for maze in data["mazes"]}
    tiles = data["tiles"]
    if not tiles:
        raise ValueError("scenario.json has no tiles.")

    spawn_xy = []
    spawn_yaw = []
    goal_xy = []
    cell_origin = []
    pitch = []
    grid_n = []
    tile_origin = []
    route_m = []
    for tile in tiles:
        maze = mazes[int(tile["maze"])]
        origin = tile["origin"]
        pair = maze["spawn_goal_pairs"][int(tile.get("pair", 0))]
        width = float(maze["corridor_width_m"])
        cell_pitch = width + wall_m
        ox = float(origin[0]) + wall_m + 0.5 * width
        oy = float(origin[1]) + wall_m + 0.5 * width
        spawn = pair["spawn"]["pos"]
        goal = pair["goal"]["pos"]
        spawn_xy.append((float(origin[0]) + float(spawn[0]), float(origin[1]) + float(spawn[1])))
        spawn_yaw.append(float(pair["spawn"]["yaw"]))
        goal_xy.append((float(origin[0]) + float(goal[0]), float(origin[1]) + float(goal[1])))
        cell_origin.append((ox, oy))
        pitch.append(cell_pitch)
        grid_n.append(int(maze["n"]))
        tile_origin.append((float(origin[0]), float(origin[1])))
        cells = maze.get("path_length_cells")
        if cells is None or int(cells) <= 0:
            route_m.append(float("nan"))
        else:
            route_m.append((int(cells) - 1) * cell_pitch)

    return BenchmarkLayout(
        spawn_xy=torch.tensor(spawn_xy, dtype=torch.float32),
        spawn_yaw=torch.tensor(spawn_yaw, dtype=torch.float32),
        goal_xy=torch.tensor(goal_xy, dtype=torch.float32),
        cell_origin_xy=torch.tensor(cell_origin, dtype=torch.float32),
        cell_pitch=torch.tensor(pitch, dtype=torch.float32),
        grid_n=torch.tensor(grid_n, dtype=torch.long),
        tile_origin_xy=torch.tensor(tile_origin, dtype=torch.float32),
        optimal_route_m=torch.tensor(route_m, dtype=torch.float32),
    )
