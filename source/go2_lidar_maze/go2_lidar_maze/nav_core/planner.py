# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A* on the maze cell graph, and a distance field on the LiDAR occupancy grid.

The grid search is 8-connected. A cardinal step costs the cell size times a
clearance factor. A diagonal step costs sqrt(2) times that, and is refused
when either adjacent cell is a wall, so the route cannot clip a corner.
Unknown cells stay traversable; the caller folds their extra cost into the
clearance factor. Every multiplier is at least 1.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence

import torch

CARDINAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
OCTILE = CARDINAL + ((1, 1), (1, -1), (-1, 1), (-1, -1))
UNREACHABLE = 1.0e6


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


def distance_field(
    occupied: torch.Tensor,
    goal: tuple[int, int],
    resolution: float,
    step_cost: torch.Tensor | None = None,
    robot: tuple[int, int] | None = None,
    max_iters: int = 2048,
    check_every: int = 32,
    settle_iters: int = 64,
) -> torch.Tensor:
    """Shortest-path distance to ``goal`` on one occupancy grid.

    One relaxation moves the front by a single cell, so ``max_iters`` is the
    longest route the field can represent, in cells. When ``robot`` is given
    the sweep stops after that cell's distance has stopped changing.
    ``settle_iters`` sets how many checks must agree before that stop.

    Args:
        occupied: Boolean wall mask ``(H, W)``.
        goal: Goal cell ``(row, col)``.
        resolution: Cell size [m]. The cardinal step cost is this times
            ``step_cost``.
        step_cost: Per-cell multiplier ``(H, W)``, at least 1. Defaults to 1.
        robot: Optional robot cell. The goal cell and this cell are forced
            traversable so inflation cannot strand the plan.
        max_iters: Relaxation sweeps.

    Returns:
        Distance to the goal [m], shape ``(H, W)``. Cells with no route are
        ``UNREACHABLE``.
    """
    if occupied.ndim != 2:
        raise ValueError(f"distance_field expects one grid, got shape {tuple(occupied.shape)}.")
    height, width = occupied.shape
    device = occupied.device
    goal_r, goal_c = _clamp_cell(goal, height, width)
    free = ~occupied.unsqueeze(0)
    free[0, goal_r, goal_c] = True
    robot_rc: tuple[int, int] | None = None
    if robot is not None:
        robot_rc = _clamp_cell(robot, height, width)
        free[0, robot_rc[0], robot_rc[1]] = True
    wall = ~free
    if step_cost is None:
        mult = torch.ones((1, height, width), dtype=torch.float32, device=device)
    else:
        mult = step_cost.to(dtype=torch.float32, device=device).unsqueeze(0).clamp(min=1.0)
    cardinal = float(resolution) * mult
    diagonal = cardinal * math.sqrt(2.0)
    blocked_cost = torch.full((1, height, width), UNREACHABLE, dtype=torch.float32, device=device)
    dist = blocked_cost.clone()
    dist[0, goal_r, goal_c] = 0.0

    # Non-uniform costs keep improving after the first finite value. Stop only
    # once the robot cell has held still, so a cheap mapped corridor can replace
    # a short hop through unknown space.
    needed_stable = max(2, int(settle_iters) // max(check_every, 1))
    stable = 0
    previous: float | None = None
    for it in range(max_iters):
        updated = dist
        for dr, dc in CARDINAL:
            neigh = _shift(dist, dr, dc, UNREACHABLE)
            updated = torch.minimum(updated, neigh + cardinal)
        for dr, dc in OCTILE[4:]:
            neigh = _shift(dist, dr, dc, UNREACHABLE)
            side_a = _shift(wall, dr, 0, True)
            side_b = _shift(wall, 0, dc, True)
            cand = neigh + diagonal
            cand = torch.where(side_a | side_b, blocked_cost, cand)
            updated = torch.minimum(updated, cand)
        dist = torch.where(free, updated, blocked_cost)
        dist[0, goal_r, goal_c] = 0.0
        if robot_rc is not None and (it + 1) % check_every == 0:
            value = float(dist[0, robot_rc[0], robot_rc[1]])
            if previous is not None and abs(value - previous) < 1.0e-2 and value < UNREACHABLE * 0.5:
                stable += 1
                if stable >= needed_stable:
                    break
            else:
                stable = 0
            previous = value
    return dist.squeeze(0)


def trace_route(
    dist: torch.Tensor,
    start: tuple[int, int],
    *,
    resolution: float,
    max_length_m: float = 16.0,
    max_steps: int = 96,
) -> list[tuple[int, int]]:
    """Walk downhill on ``dist`` and return the cells within ``max_length_m``.

    Args:
        dist: Distance-to-goal ``(H, W)`` from :func:`distance_field`.
        start: Robot cell ``(row, col)``.
        resolution: Cell size [m], used to stop the walk.
        max_length_m: Arc length at which the walk stops [m].
        max_steps: Most cells to visit.

    Returns:
        Cells from ``start`` along the route. The walk ends at the goal, at
        ``max_length_m``, or when every neighbour is uphill.
    """
    height, width = dist.shape
    device = dist.device
    offsets = torch.tensor(OCTILE, dtype=torch.long, device=device)
    diagonal = (offsets[:, 0] != 0) & (offsets[:, 1] != 0)
    side_a_off = torch.stack((offsets[:, 0], torch.zeros_like(offsets[:, 0])), dim=-1)
    side_b_off = torch.stack((torch.zeros_like(offsets[:, 1]), offsets[:, 1]), dim=-1)
    cur = torch.tensor([[int(start[0]), int(start[1])]], dtype=torch.long, device=device)
    walked = torch.empty((max_steps, 2), dtype=torch.long, device=device)
    band = dist.unsqueeze(0)
    rows = torch.zeros(1, dtype=torch.long, device=device)

    def _sample(cells: torch.Tensor) -> torch.Tensor:
        inside = (cells[..., 0] >= 0) & (cells[..., 0] < height) & (cells[..., 1] >= 0) & (cells[..., 1] < width)
        clamped = cells.clamp(min=0)
        clamped[..., 0].clamp_(max=height - 1)
        clamped[..., 1].clamp_(max=width - 1)
        vals = band[rows.view(1, 1), clamped[..., 0], clamped[..., 1]]
        return torch.where(inside, vals, torch.full_like(vals, UNREACHABLE))

    for _step in range(max_steps):
        cand = cur.unsqueeze(1) + offsets.unsqueeze(0)
        vals = _sample(cand)
        # A diagonal step is illegal when either side cell is a wall. Walls are
        # unreachable in the field, and so is any cell the wave never entered.
        corner = diagonal.unsqueeze(0) & (
            (_sample(cur.unsqueeze(1) + side_a_off) >= UNREACHABLE * 0.5)
            | (_sample(cur.unsqueeze(1) + side_b_off) >= UNREACHABLE * 0.5)
        )
        vals = vals.masked_fill(corner, UNREACHABLE)
        best = torch.argmin(vals, dim=1)
        nxt = cand[rows, best]
        # Out-of-grid neighbours are not stored; skip them by holding position.
        nxt_inside = (nxt[:, 0] >= 0) & (nxt[:, 0] < height) & (nxt[:, 1] >= 0) & (nxt[:, 1] < width)
        cur_val = band[rows, cur[:, 0], cur[:, 1]]
        improves = (vals[rows, best] < cur_val - 1.0e-4) & nxt_inside
        cur = torch.where(improves.unsqueeze(1), nxt, cur)
        walked[_step] = cur[0]
    cells = walked.tolist()
    route = [(int(start[0]), int(start[1]))]
    travelled = 0.0
    for raw in cells:
        nxt = (int(raw[0]), int(raw[1]))
        if nxt == route[-1]:
            break
        travelled += math.hypot(nxt[0] - route[-1][0], nxt[1] - route[-1][1]) * float(resolution)
        route.append(nxt)
        if travelled >= max_length_m:
            break
    return route


@torch.no_grad()
def distance_field_batch(
    occupied: torch.Tensor,
    goal: torch.Tensor,
    resolution: float | torch.Tensor,
    step_cost: torch.Tensor | None = None,
    robot: torch.Tensor | None = None,
    max_iters: int = 512,
    check_every: int = 8,
) -> torch.Tensor:
    """Shortest-path distance to the goal on a batch of occupancy grids.

    Same costs as :func:`distance_field`: cardinal steps cost the cell size
    times ``step_cost``, diagonals cost ``sqrt(2)`` times that, and a diagonal
    is refused when either side cell is a wall. When ``robot`` is given, the
    sweep stops once every robot cell has held its distance. If much of the
    crop is unknown, it first waits long enough for a mapped corridor about
    three times longer in cells than the first wave. One expensive cell, such
    as the reverse ring, does not stretch that wait. Without ``robot``, the
    sweep stops once every grid has stopped changing.

    Args:
        occupied: Boolean wall mask ``(B, H, W)``.
        goal: Goal cells ``(B, 2)`` as ``(row, col)``.
        resolution: Cell size [m], a float or ``(B,)``.
        step_cost: Per-cell multiplier ``(B, H, W)``, at least 1. Defaults to 1.
        robot: Optional robot cells ``(B, 2)``. The goal and these cells are
            forced traversable so inflation cannot strand the plan.
        max_iters: Relaxation sweeps. Also the longest route, in cells, that
            the field can represent.
        check_every: Sweeps between convergence checks.

    Returns:
        Distance to the goal [m], shape ``(B, H, W)``. Cells with no route are
        ``UNREACHABLE``.
    """
    if occupied.ndim != 3:
        raise ValueError(f"distance_field_batch expects (B, H, W), got {tuple(occupied.shape)}.")
    batch, height, width = occupied.shape
    device = occupied.device
    goal = goal.to(device=device, dtype=torch.long)
    env = torch.arange(batch, device=device)
    goal_r = goal[:, 0].clamp(0, height - 1)
    goal_c = goal[:, 1].clamp(0, width - 1)
    free = ~occupied
    free[env, goal_r, goal_c] = True
    if robot is not None:
        robot = robot.to(device=device, dtype=torch.long)
        free[env, robot[:, 0].clamp(0, height - 1), robot[:, 1].clamp(0, width - 1)] = True
    wall = ~free
    if step_cost is None:
        mult = torch.ones((batch, height, width), dtype=torch.float32, device=device)
    else:
        mult = step_cost.to(dtype=torch.float32, device=device).clamp(min=1.0)
    if isinstance(resolution, torch.Tensor):
        res = resolution.to(device=device, dtype=torch.float32).reshape(batch, 1, 1)
    else:
        res = torch.full((batch, 1, 1), float(resolution), device=device)
    cardinal = res * mult
    diagonal = cardinal * math.sqrt(2.0)
    dist = torch.full((batch, height, width), UNREACHABLE, dtype=torch.float32, device=device)
    dist[env, goal_r, goal_c] = 0.0
    # One 3x3 step-cost stencil. The centre is 0, so the sweep keeps the
    # current distance. A blocked diagonal costs UNREACHABLE: added to any
    # finite neighbour it stays above every real route, and the centre cell
    # is always a candidate, so the min matches refusing that step.
    wall_pad = torch.nn.functional.pad(wall, (1, 1, 1, 1), value=True)
    north_w = wall_pad[:, :-2, 1:-1]
    south_w = wall_pad[:, 2:, 1:-1]
    west_w = wall_pad[:, 1:-1, :-2]
    east_w = wall_pad[:, 1:-1, 2:]
    cost = diagonal.unsqueeze(-1).expand(batch, height, width, 9).clone().view(batch, height, width, 3, 3)
    cost[..., 1, 1] = 0.0
    cost[..., 0, 1] = cardinal
    cost[..., 2, 1] = cardinal
    cost[..., 1, 0] = cardinal
    cost[..., 1, 2] = cardinal
    cost[..., 0, 0].masked_fill_(north_w | west_w, UNREACHABLE)
    cost[..., 0, 2].masked_fill_(north_w | east_w, UNREACHABLE)
    cost[..., 2, 0].masked_fill_(south_w | west_w, UNREACHABLE)
    cost[..., 2, 2].masked_fill_(south_w | east_w, UNREACHABLE)
    dist_pad = torch.empty((batch, height + 2, width + 2), dtype=torch.float32, device=device)
    dist_pad[:, 0, :] = UNREACHABLE
    dist_pad[:, -1, :] = UNREACHABLE
    dist_pad[:, :, 0] = UNREACHABLE
    dist_pad[:, :, -1] = UNREACHABLE

    # The first finite value is the shortest hop path, often through unknown
    # cells. A cheaper corridor can be longer in hops by about that cost.
    # A one-cell spike, such as the reverse ring, is not a longer route, so
    # the wait follows how much of the crop actually carries the high cost.
    hop_scale = 1
    history = None
    robot_r = robot_c = None
    if robot is not None:
        robot_r = robot[:, 0].clamp(0, height - 1)
        robot_c = robot[:, 1].clamp(0, width - 1)
        share = torch.stack(((mult >= 1.5).float().mean(), (mult >= 2.5).float().mean()))
        mid_share, high_share = share.tolist()
        if high_share > 0.01:
            hop_scale = 3
        elif mid_share > 0.01:
            hop_scale = 2
        history = torch.empty((max_iters, batch), dtype=torch.float32, device=device)
    previous = None if robot is not None else dist.clone()
    quiet = torch.zeros(batch, dtype=torch.long, device=device)
    arrived = torch.full((batch,), -1, dtype=torch.long, device=device)
    for it in range(max_iters):
        dist_pad[:, 1:-1, 1:-1] = dist
        windows = dist_pad.unfold(1, 3, 1).unfold(2, 3, 1)
        updated = (windows + cost).amin(dim=(-2, -1))
        updated.masked_fill_(~free, UNREACHABLE)
        dist = updated
        if history is not None:
            history[it] = dist[env, robot_r, robot_c]
        if (it + 1) % check_every != 0:
            continue
        if history is not None:
            quiet, arrived = _robot_quiet(history[: it + 1], check_every, hop_scale, quiet, arrived)
        else:
            change = (dist - previous).abs().amax(dim=(1, 2))
            quiet = torch.where(change < 1.0e-3, quiet + 1, torch.zeros_like(quiet))
            previous = dist.clone()
        if int(quiet.min()) >= 2:
            break
    return dist


def _robot_quiet(
    history: torch.Tensor,
    check_every: int,
    hop_scale: int,
    quiet: torch.Tensor,
    arrived: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance the robot-cell stop from the newest check in ``history``.

    ``history`` holds the robot distance after every sweep so far, oldest
    first. Only the rows at the check interval are compared, matching a
    running check that samples every ``check_every`` sweeps.
    """
    end = history.shape[0] - 1
    now = history[end]
    if end >= check_every:
        prev = history[end - check_every]
        segment = history[end - check_every + 1 :]
        start = end - check_every + 1
    else:
        prev = torch.full_like(now, UNREACHABLE)
        segment = history
        start = 0
    finite_now = now < UNREACHABLE * 0.5
    changed = (now - prev).abs() >= 1.0e-2
    finite = segment < UNREACHABLE * 0.5
    any_finite = finite.any(dim=0)
    first = finite.to(torch.float32).argmax(dim=0)
    step_no = (start + first + 1).to(dtype=arrived.dtype)
    arrived = torch.where(any_finite & (arrived < 0), step_no, arrived)
    waited = (arrived >= 0) & ((end + 1) >= arrived * hop_scale)
    holding = finite_now & ~changed & waited
    quiet = torch.where(holding, quiet + 1, torch.zeros_like(quiet))
    return quiet, arrived


@torch.no_grad()
def trace_route_batch(
    dist: torch.Tensor,
    start: torch.Tensor,
    *,
    resolution: float | torch.Tensor,
    max_length_m: float = 16.0,
    max_steps: int = 96,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Walk downhill on each distance field.

    Args:
        dist: Distance-to-goal ``(B, H, W)`` [m] from :func:`distance_field_batch`.
        start: Robot cells ``(B, 2)`` as ``(row, col)``.
        resolution: Cell size [m], a float or ``(B,)``.
        max_length_m: Arc length at which a walk stops [m].
        max_steps: Most cells to visit after the start.

    Returns:
        ``(cells, counts)``. ``cells`` is ``(B, max_steps + 1, 2)`` and starts
        at ``start``. ``counts`` is ``(B,)``, the number of cells in each walk
        including the start. Cells past ``counts`` repeat the last step.
    """
    batch, height, width = dist.shape
    device = dist.device
    if isinstance(resolution, torch.Tensor):
        res = resolution.to(device=device, dtype=torch.float32).reshape(batch)
    else:
        res = torch.full((batch,), float(resolution), device=device)
    cur = start.to(device=device, dtype=torch.long).reshape(batch, 2).clone()
    cur[:, 0].clamp_(0, height - 1)
    cur[:, 1].clamp_(0, width - 1)
    cells = cur.new_empty((batch, max_steps + 1, 2))
    cells[:, 0] = cur
    counts = torch.ones(batch, dtype=torch.long, device=device)
    if max_steps <= 0:
        return cells, counts
    # The best downhill step depends only on the field, so follow it instead of
    # re-ranking the eight neighbours at every cell of the walk.
    succ = _best_step(dist)
    env = torch.arange(batch, device=device)
    for _step in range(max_steps):
        cur = succ[env, cur[:, 0], cur[:, 1]]
        cells[:, _step + 1] = cur
    delta = (cells[:, 1:] - cells[:, :-1]).to(torch.float32)
    step_len = torch.linalg.norm(delta, dim=-1) * res.unsqueeze(1)
    moved = step_len > 1.0e-8
    # A step counts only while every earlier step moved and stayed under the cap.
    # The step that crosses ``max_length_m`` is kept; later ones are not.
    prefix = torch.cumprod(moved.to(torch.int32), dim=1).to(dtype=torch.bool)
    travelled = torch.cumsum(step_len, dim=1)
    under = torch.ones_like(prefix)
    under[:, 1:] = travelled[:, :-1] < max_length_m
    keep = prefix & under
    counts = counts + keep.sum(dim=1)
    last = (counts - 1).view(batch, 1, 1).expand(batch, 1, 2)
    held = cells.gather(1, last)
    index = torch.arange(max_steps + 1, device=device).view(1, -1, 1)
    cells = torch.where(index < counts.view(batch, 1, 1), cells, held)
    return cells, counts


def _best_step(dist: torch.Tensor) -> torch.Tensor:
    """Downhill neighbour of every cell, or the cell itself when none improves.

    Neighbours are tried in :data:`OCTILE` order. A diagonal is refused when
    either side cell is unreachable, and an out-of-grid neighbour is unreachable.
    """
    batch, height, width = dist.shape
    device = dist.device
    offsets = torch.tensor(OCTILE, dtype=torch.long, device=device)
    rows = torch.arange(height, device=device)
    cols = torch.arange(width, device=device)
    cand_r = rows.view(1, -1, 1, 1) + offsets[:, 0].view(1, 1, 1, -1)
    cand_c = cols.view(1, 1, -1, 1) + offsets[:, 1].view(1, 1, 1, -1)
    cand_r = cand_r.expand(batch, height, width, 8)
    cand_c = cand_c.expand(batch, height, width, 8)
    inside = (cand_r >= 0) & (cand_r < height) & (cand_c >= 0) & (cand_c < width)
    env = torch.arange(batch, device=device).view(batch, 1, 1, 1).expand(batch, height, width, 8)
    vals = dist[env, cand_r.clamp(0, height - 1), cand_c.clamp(0, width - 1)]
    vals = torch.where(inside, vals, torch.full_like(vals, UNREACHABLE))
    diagonal = (offsets[:, 0] != 0) & (offsets[:, 1] != 0)
    side_r = (rows.view(1, -1, 1, 1) + offsets[:, 0].view(1, 1, 1, -1)).expand(batch, height, width, 8)
    side_c = cols.view(1, 1, -1, 1).expand(batch, height, width, 8)
    side_b_r = rows.view(1, -1, 1, 1).expand(batch, height, width, 8)
    side_b_c = (cols.view(1, 1, -1, 1) + offsets[:, 1].view(1, 1, 1, -1)).expand(batch, height, width, 8)

    def _at(rr: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
        ok = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
        sampled = dist[env, rr.clamp(0, height - 1), cc.clamp(0, width - 1)]
        return torch.where(ok, sampled, torch.full_like(sampled, UNREACHABLE))

    blocked = (_at(side_r, side_c) >= UNREACHABLE * 0.5) | (_at(side_b_r, side_b_c) >= UNREACHABLE * 0.5)
    vals = vals.masked_fill(diagonal.view(1, 1, 1, -1) & blocked, UNREACHABLE)
    improves = vals < dist.unsqueeze(-1) - 1.0e-4
    best = vals.masked_fill(~improves, float("inf")).argmin(dim=-1)
    pick = best.unsqueeze(-1)
    next_r = cand_r.gather(-1, pick).squeeze(-1)
    next_c = cand_c.gather(-1, pick).squeeze(-1)
    any_imp = improves.any(dim=-1)
    self_r = rows.view(1, -1, 1).expand(batch, height, width)
    self_c = cols.view(1, 1, -1).expand(batch, height, width)
    return torch.stack(
        (
            torch.where(any_imp, next_r, self_r),
            torch.where(any_imp, next_c, self_c),
        ),
        dim=-1,
    )


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
    dist = torch.full(occupied.shape, float(radius + 1), dtype=torch.float32, device=occupied.device)
    dist = torch.where(occupied, torch.zeros_like(dist), dist)
    # Ellipsis keeps a leading batch dimension on the last two axes. A single
    # grid has none, so this is the same indexing as slicing the whole tensor.
    for _ in range(radius):
        padded = torch.nn.functional.pad(dist, (1, 1, 1, 1), value=float(radius + 1))
        neighbours = torch.stack(
            (
                padded[..., :-2, 1:-1],
                padded[..., 2:, 1:-1],
                padded[..., 1:-1, :-2],
                padded[..., 1:-1, 2:],
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


def _clamp_cell(cell: tuple[int, int], height: int, width: int) -> tuple[int, int]:
    return (max(0, min(height - 1, int(cell[0]))), max(0, min(width - 1, int(cell[1]))))


def _shift(values: torch.Tensor, dr: int, dc: int, fill: float | bool) -> torch.Tensor:
    """Value of the neighbour ``(r+dr, c+dc)``, with off-grid cells set to ``fill``.

    ``values`` has shape ``(1, H, W)``. ``torch.roll`` wraps, so the wrapped
    edge is overwritten.
    """
    shifted = torch.roll(values, shifts=(-dr, -dc), dims=(-2, -1))
    # A negative roll wraps the high indices; a positive roll wraps the low ones.
    if dr > 0:
        shifted[:, -dr:, :] = fill
    elif dr < 0:
        shifted[:, :-dr, :] = fill
    if dc > 0:
        shifted[:, :, -dc:] = fill
    elif dc < 0:
        shifted[:, :, :-dc] = fill
    return shifted


def _reconstruct(came: dict[int, int], node: int) -> list[int]:
    path = [node]
    while node in came:
        node = came[node]
        path.append(node)
    path.reverse()
    return path
