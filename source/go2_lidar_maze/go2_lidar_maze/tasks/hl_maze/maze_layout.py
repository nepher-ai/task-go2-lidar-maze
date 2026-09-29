# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged maze graph used only while training the high-level policy.

Adjacency is measured from the wall mesh: a downward ray at each wall-slot
midpoint hits at about 1 m when a wall is present. The resulting 4-connected
graph is searched with A*, and every pair whose path is 1 to 5 cells long is
stored so training can sample waypoints without running a search per reset.

The online planner must not import this module. Reset sampling may use the
adjacency so it can reject a start and goal that A* cannot join.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from go2_lidar_maze.constants import (
    CELL_CENTER_ORIGIN_M,
    CELL_PITCH_M,
    GRID_SIZE,
    MAX_ASTAR_CELLS,
    WALL_HEIGHT_M,
)
from go2_lidar_maze.nav_core.planner import astar_cells, cells_at_distance

_MAX_TARGETS = 32


def cell_center(row: int, col: int) -> tuple[float, float]:
    """World XY [m] of corridor cell ``(row, col)``.

    ``x = 1.5 + 2.5 col`` and ``y = 1.5 + 2.5 row``. Cell ``(0, 16)`` is
    ``(41.5, 1.5)``, matching ``enclosed_20x20.txt``.
    """
    return (
        CELL_CENTER_ORIGIN_M + CELL_PITCH_M * col,
        CELL_CENTER_ORIGIN_M + CELL_PITCH_M * row,
    )


def index_of(row: int, col: int, n: int = GRID_SIZE) -> int:
    """Row-major cell index."""
    return row * n + col


def wall_slot_midpoints(n: int = GRID_SIZE) -> list[tuple[float, float, int, int]]:
    """Midpoints of the slots that may separate two corridor cells.

    Returns:
        Tuples ``(x, y, cell_a, cell_b)`` [m, m, index, index]. A ray that
        hits a wall at one of these points means ``cell_a`` and ``cell_b``
        are not connected.
    """
    slots: list[tuple[float, float, int, int]] = []
    for row in range(n):
        for col in range(n - 1):
            x = CELL_CENTER_ORIGIN_M + CELL_PITCH_M * col + 0.5 * CELL_PITCH_M
            y = CELL_CENTER_ORIGIN_M + CELL_PITCH_M * row
            slots.append((x, y, index_of(row, col, n), index_of(row, col + 1, n)))
    for row in range(n - 1):
        for col in range(n):
            x = CELL_CENTER_ORIGIN_M + CELL_PITCH_M * col
            y = CELL_CENTER_ORIGIN_M + CELL_PITCH_M * row + 0.5 * CELL_PITCH_M
            slots.append((x, y, index_of(row, col, n), index_of(row + 1, col, n)))
    return slots


def adjacency_from_blocked(
    blocked_pairs: Sequence[tuple[int, int]], n: int = GRID_SIZE
) -> list[list[int]]:
    """Build a 4-connected adjacency list, removing blocked wall slots."""
    blocked = {tuple(sorted(pair)) for pair in blocked_pairs}
    neighbours: list[list[int]] = [[] for _ in range(n * n)]
    for row in range(n):
        for col in range(n):
            here = index_of(row, col, n)
            for dr, dc in ((0, 1), (1, 0)):
                nr, nc = row + dr, col + dc
                if nr >= n or nc >= n:
                    continue
                other = index_of(nr, nc, n)
                if tuple(sorted((here, other))) in blocked:
                    continue
                neighbours[here].append(other)
                neighbours[other].append(here)
    return neighbours


def measure_blocked_pairs(usd_path: str | Path, n: int = GRID_SIZE) -> list[tuple[int, int]]:
    """Ray-cast the wall mesh and return cell pairs separated by a wall.

    A downward ray starts 2 m above each wall-slot midpoint. A hit whose
    height is within 0.25 m of the wall top (1 m) counts as a wall.
    """
    points, triangles = _load_wall_triangles(usd_path)
    import warp as wp

    from isaaclab.utils.warp import raycast_mesh

    mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3),
        indices=wp.array(triangles, dtype=wp.int32),
    )
    slots = wall_slot_midpoints(n)
    starts = torch.tensor([[x, y, 2.0] for x, y, _, _ in slots], dtype=torch.float32)
    dirs = torch.zeros_like(starts)
    dirs[:, 2] = -1.0
    hits = raycast_mesh(starts, dirs, mesh, max_dist=3.0)[0]
    blocked: list[tuple[int, int]] = []
    for hit, slot in zip(hits, slots):
        if not torch.isfinite(hit).all():
            continue
        if abs(float(hit[2]) - WALL_HEIGHT_M) < 0.25:
            blocked.append((slot[2], slot[3]))
    return blocked


class MazeLayout:
    """Precomputed A* pairs of length 1 to 5 cells."""

    def __init__(self, neighbours: Sequence[Sequence[int]], n: int = GRID_SIZE, n_cols: int | None = None):
        self.n = n
        self.n_cols = n if n_cols is None else n_cols
        self.neighbours = [list(row) for row in neighbours]
        self.num_cells = n * n
        # targets[start][k - 1] = list of (target, path)
        self.targets: list[list[list[tuple[int, list[int]]]]] = []
        for start in range(self.num_cells):
            per_k: list[list[tuple[int, list[int]]]] = []
            for k in range(1, MAX_ASTAR_CELLS + 1):
                per_k.append(cells_at_distance(self.neighbours, start, k))
            self.targets.append(per_k)

    @classmethod
    def from_usd(cls, usd_path: str | Path, n: int = GRID_SIZE) -> MazeLayout:
        """Build the layout by ray-casting ``usd_path``."""
        blocked = measure_blocked_pairs(usd_path, n)
        return cls(adjacency_from_blocked(blocked, n), n=n, n_cols=n)

    def sample(self, start: int, k: int, choice: int) -> tuple[int, list[int]]:
        """Return ``(target, path)`` for a start cell and an A* length ``k``.

        Args:
            start: Start cell index.
            k: Path length in cells, in ``[1, 5]``.
            choice: Index into the targets at that distance, wrapped modulo
                the number of targets.

        Raises:
            ValueError: If ``start`` has no cell at distance ``k``.
        """
        if not 1 <= k <= MAX_ASTAR_CELLS:
            raise ValueError(f"A* length {k} is outside [1, {MAX_ASTAR_CELLS}].")
        options = self.targets[start][k - 1]
        if not options:
            raise ValueError(f"Cell {start} has no corridor cell at A* distance {k}.")
        return options[choice % len(options)]

    def has_distance(self, start: int, k: int) -> bool:
        """True when ``start`` can reach some cell in exactly ``k`` steps."""
        return bool(self.targets[start][k - 1])

    def path_length(self, start: int, goal: int) -> int | None:
        """A* length in cells, or None when ``goal`` is unreachable."""
        path = astar_cells(self.neighbours, start, goal, n_cols=self.n_cols)
        if path is None:
            return None
        return len(path) - 1


def _load_wall_triangles(usd_path: str | Path) -> tuple[list[tuple[float, float, float]], list[int]]:
    """Read ``/World/Walls`` and triangulate its faces."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    prim = stage.GetPrimAtPath("/World/Walls")
    if not prim.IsValid():
        raise FileNotFoundError(f"USD '{usd_path}' has no /World/Walls mesh.")
    mesh = UsdGeom.Mesh(prim)
    points = [(float(p[0]), float(p[1]), float(p[2])) for p in mesh.GetPointsAttr().Get()]
    counts = list(mesh.GetFaceVertexCountsAttr().Get())
    indices = list(mesh.GetFaceVertexIndicesAttr().Get())
    triangles: list[int] = []
    cursor = 0
    for count in counts:
        face = indices[cursor : cursor + count]
        for k in range(1, count - 1):
            triangles.extend((face[0], face[k], face[k + 1]))
        cursor += count
    return points, triangles
