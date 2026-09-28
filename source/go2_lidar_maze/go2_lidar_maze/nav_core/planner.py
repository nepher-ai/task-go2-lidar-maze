# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A* planners for the maze cell graph and for the LiDAR occupancy grid.

The occupancy-grid search is 8-connected. Its step cost is geometric length
times a clearance factor times 1.15 on unknown cells. Every multiplier is at
least 1, so the octile heuristic (cost 1 on cardinal steps and sqrt(2) on
diagonals) stays admissible.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

import torch

CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
OCTILE = CARDINAL + ((1, 1), (1, -1), (-1, 1), (-1, -1))


def astar_cells(
    neighbours: Sequence[Sequence[int]],
    start: int,
    goal: int,
    n_cols: int = 20,
) -> list[int] | None:
    """Shortest path on an unweighted 4-connected graph.

    Args:
        neighbours: ``neighbours[i]`` lists the cells adjacent to ``i``.
        start: Start cell index.
        goal: Goal cell index.

    Returns:
        Cell indices from ``start`` to ``goal``, or None when unreachable.
        The length in edges is ``len(path) - 1``.
    """
    if start == goal:
        return [start]

    def h(cell: int) -> int:
        gr, gc = divmod(goal, n_cols)
        nr, nc = divmod(cell, n_cols)
        return abs(gr - nr) + abs(gc - nc)

    # (f, tie, g, node). g is stored so a stale heap entry can be skipped.
    heap: list[tuple[int, int, int, int]] = [(h(start), 0, 0, start)]
    came: dict[int, int] = {}
    cost = {start: 0}
    tie = 0
    while heap:
        _, _, g, node = heapq.heappop(heap)
        if g != cost.get(node):
            continue
        if node == goal:
            return _reconstruct(came, node)
        for nxt in neighbours[node]:
            new_cost = cost[node] + 1
            if new_cost < cost.get(nxt, 1_000_000):
                cost[nxt] = new_cost
                came[nxt] = node
                tie += 1
                heapq.heappush(heap, (new_cost + h(nxt), tie, new_cost, nxt))
    return None


def cells_at_distance(neighbours: Sequence[Sequence[int]], start: int, distance: int) -> list[tuple[int, list[int]]]:
    """Every cell whose shortest path from ``start`` has ``distance`` edges.

    Returns:
        Pairs of ``(cell, path)`` with ``path[0] == start`` and
        ``len(path) - 1 == distance``.
    """
    found: list[tuple[int, list[int]]] = []
    queue: list[tuple[int, list[int]]] = [(start, [start])]
    seen = {start}
    while queue:
        node, path = queue.pop(0)
        if len(path) - 1 == distance:
            found.append((node, path))
            continue
        if len(path) - 1 > distance:
            continue
        for nxt in neighbours[node]:
            if nxt in seen:
                continue
            seen.add(nxt)
            queue.append((nxt, path + [nxt]))
    return found


def astar_grid(
    occupied: torch.Tensor,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    unknown: torch.Tensor | None = None,
    resolution: float = 0.25,
    unknown_cost: float = 1.15,
    clearance_weight: float = 4.0,
    clearance_radius: int = 2,
) -> list[tuple[int, int]] | None:
    """A* on one occupancy grid.

    Args:
        occupied: Boolean wall mask ``(H, W)``.
        start: Start cell ``(row, col)``.
        goal: Goal cell ``(row, col)``.
        unknown: Boolean mask of never-seen cells. Unknown is traversable.
        resolution: Cell size [m], used only to scale the geometric step.
        unknown_cost: Multiplier on never-seen cells. Must be >= 1.
        clearance_weight: Extra cost, relative to open space, for hugging a wall.
        clearance_radius: Range of that extra cost [cells].

    Returns:
        Cell path from start to goal, or None.
    """
    height, width = occupied.shape
    if not _inside(start, height, width) or not _inside(goal, height, width):
        return None
    if bool(occupied[goal]):
        goal = _nearest_free(occupied, goal)
        if goal is None:
            return None
    if bool(occupied[start]):
        start = _nearest_free(occupied, start)
        if start is None:
            return None
    penalty = _clearance_penalty(occupied, clearance_radius, clearance_weight)
    unknown_mask = unknown if unknown is not None else torch.zeros_like(occupied)
    res = float(resolution)

    def step_cost(cell: tuple[int, int], diagonal: bool) -> float:
        length = res * (math.sqrt(2.0) if diagonal else 1.0)
        factor = 1.0 + float(penalty[cell])
        if bool(unknown_mask[cell]):
            factor *= unknown_cost
        return length * factor

    def heuristic(cell: tuple[int, int]) -> float:
        dr = abs(cell[0] - goal[0])
        dc = abs(cell[1] - goal[1])
        diagonal = min(dr, dc)
        straight = max(dr, dc) - diagonal
        return res * (math.sqrt(2.0) * diagonal + straight)

    # (f, tie, g, node). g is stored so a stale heap entry can be skipped.
    heap: list[tuple[float, int, float, tuple[int, int]]] = [(heuristic(start), 0, 0.0, start)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}
    tie = 0
    while heap:
        _, _, g, node = heapq.heappop(heap)
        if g > cost.get(node, float("inf")) + 1e-9:
            continue
        if node == goal:
            path = [node]
            while node in came:
                node = came[node]
                path.append(node)
            path.reverse()
            return path
        for dr, dc in OCTILE:
            nxt = (node[0] + dr, node[1] + dc)
            if not _inside(nxt, height, width) or bool(occupied[nxt]):
                continue
            diagonal = dr != 0 and dc != 0
            if diagonal and (bool(occupied[node[0] + dr, node[1]]) or bool(occupied[node[0], node[1] + dc])):
                continue
            new_cost = cost[node] + step_cost(nxt, diagonal)
            if new_cost < cost.get(nxt, float("inf")):
                cost[nxt] = new_cost
                came[nxt] = node
                tie += 1
                heapq.heappush(heap, (new_cost + heuristic(nxt), tie, new_cost, nxt))
    return None


def path_length_m(path: Sequence[tuple[int, int]], resolution: float) -> float:
    """Geometric length of an 8-connected cell path [m]."""
    total = 0.0
    for a, b in zip(path, path[1:]):
        step = math.hypot(b[0] - a[0], b[1] - a[1])
        total += step * resolution
    return total


def _clearance_penalty(occupied: torch.Tensor, radius: int, weight: float) -> torch.Tensor:
    """Extra cost in ``[0, weight]`` that falls off with distance to a wall."""
    if radius <= 0 or weight <= 0.0:
        return torch.zeros_like(occupied, dtype=torch.float32)
    dist = torch.full(occupied.shape, float(radius + 1), dtype=torch.float32)
    dist = torch.where(occupied, torch.zeros_like(dist), dist)
    for _ in range(radius):
        padded = torch.nn.functional.pad(dist, (1, 1, 1, 1), value=float(radius + 1))
        neighbours = torch.stack(
            (
                padded[:-2, 1:-1],
                padded[2:, 1:-1],
                padded[1:-1, :-2],
                padded[1:-1, 2:],
            ),
            dim=0,
        )
        dist = torch.minimum(dist, neighbours.amin(dim=0) + 1.0)
    near = (dist < radius).to(torch.float32)
    return weight * near * (1.0 - dist / float(radius)).clamp(min=0.0)


def _nearest_free(occupied: torch.Tensor, cell: tuple[int, int]) -> tuple[int, int] | None:
    height, width = occupied.shape
    for radius in range(1, max(height, width)):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if max(abs(dr), abs(dc)) != radius:
                    continue
                nxt = (cell[0] + dr, cell[1] + dc)
                if _inside(nxt, height, width) and not bool(occupied[nxt]):
                    return nxt
    return None


def _inside(cell: tuple[int, int], height: int, width: int) -> bool:
    return 0 <= cell[0] < height and 0 <= cell[1] < width


def _reconstruct(came: dict[int, int], node: int) -> list[int]:
    path = [node]
    while node in came:
        node = came[node]
        path.append(node)
    path.reverse()
    return path
