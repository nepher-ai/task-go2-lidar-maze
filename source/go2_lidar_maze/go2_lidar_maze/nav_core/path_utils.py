# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Polyline smoothing, arc length, and monotone projection.

These helpers are shared by the privileged training path and the online planner
so the high-level policy sees the same waypoint geometry in both modes. They
depend only on torch.
"""

from __future__ import annotations

import math

import torch


def cumulative_length(points: torch.Tensor) -> torch.Tensor:
    """Arc length at each vertex of a polyline.

    Args:
        points: Vertices ``(..., K, 2)`` [m].

    Returns:
        Cumulative length ``(..., K)`` [m], starting at 0.
    """
    seg = torch.linalg.norm(points[..., 1:, :] - points[..., :-1, :], dim=-1)
    zero = torch.zeros(*points.shape[:-2], 1, device=points.device, dtype=points.dtype)
    return torch.cat((zero, torch.cumsum(seg, dim=-1)), dim=-1)


def points_at_arc_length(points: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    """Point at an arc length on each polyline.

    A repeated tail does not add length, so padding the buffer with the last
    vertex does not create a second point.

    Args:
        points: Polylines ``(N, K, 2)`` [m].
        distance: Arc length from the first vertex, ``(N,)`` or ``(N, M)`` [m].
            Clamped to ``[0, length]``.

    Returns:
        Points ``(N, 2)`` when ``distance`` is ``(N,)``, otherwise ``(N, M, 2)`` [m].
    """
    single = distance.ndim == 1
    if single:
        distance = distance.unsqueeze(1)
    cum = cumulative_length(points)
    total = cum[:, -1:]
    target = torch.minimum(distance.clamp(min=0.0), total)
    # Count of vertices strictly before the target: the segment that contains it.
    idx = (cum.unsqueeze(1) < target.unsqueeze(2)).sum(dim=-1)
    idx = idx.clamp(min=1, max=points.shape[1] - 1)
    prev = idx - 1
    cum_hi = cum.gather(1, idx)
    cum_lo = cum.gather(1, prev)
    span = (cum_hi - cum_lo).clamp(min=1e-8)
    frac = ((target - cum_lo) / span).clamp(0.0, 1.0).unsqueeze(-1)
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    gather_prev = prev.unsqueeze(-1).expand(-1, -1, points.shape[-1])
    pt_hi = points.gather(1, gather_idx)
    pt_lo = points.gather(1, gather_prev)
    out = pt_lo + frac * (pt_hi - pt_lo)
    if single:
        return out.squeeze(1)
    return out


def point_at_distance(points: torch.Tensor, distance: float | torch.Tensor) -> torch.Tensor:
    """Point at an arc length along a polyline.

    Args:
        points: Vertices ``(K, 2)`` [m].
        distance: Arc length from the first vertex [m]. Clamped to the polyline.

    Returns:
        Point ``(2,)`` [m].
    """
    cum = cumulative_length(points)
    total = cum[-1]
    if not torch.is_tensor(distance):
        distance_t = torch.tensor(distance, device=points.device, dtype=points.dtype)
    else:
        distance_t = distance.to(device=points.device, dtype=points.dtype)
    target = distance_t.clamp(min=0.0, max=total)
    idx = int(torch.searchsorted(cum, target).item())
    idx = min(max(idx, 1), points.shape[0] - 1)
    span = (cum[idx] - cum[idx - 1]).clamp(min=1e-8)
    frac = ((target - cum[idx - 1]) / span).clamp(0.0, 1.0)
    return points[idx - 1] + frac * (points[idx] - points[idx - 1])


def fillet_polyline(points: torch.Tensor, radius: float, samples: int = 5) -> torch.Tensor:
    """Replace sharp corners with a circular fillet.

    Corners whose adjacent segments are shorter than ``radius`` are left
    unchanged. Colinear corners are also left unchanged.

    Args:
        points: Vertices ``(K, 2)`` [m].
        radius: Fillet radius [m].
        samples: Number of samples placed on each fillet arc.

    Returns:
        Smoothed vertices ``(M, 2)`` [m].
    """
    if points.shape[0] < 3 or radius <= 0.0:
        return points
    out = [points[0]]
    for i in range(1, points.shape[0] - 1):
        prev = points[i - 1]
        corner = points[i]
        nxt = points[i + 1]
        v_in = corner - prev
        v_out = nxt - corner
        len_in = torch.linalg.norm(v_in)
        len_out = torch.linalg.norm(v_out)
        if float(len_in) < 1e-6 or float(len_out) < 1e-6:
            out.append(corner)
            continue
        d_in = v_in / len_in
        d_out = v_out / len_out
        cos_turn = torch.clamp(torch.dot(-d_in, d_out), -1.0, 1.0)
        turn = float(torch.acos(cos_turn))
        if turn < 1e-3 or abs(math.pi - turn) < 1e-3:
            out.append(corner)
            continue
        # Distance from the corner to the fillet tangent points.
        half = turn / 2.0
        tangent = radius / math.tan(half) if half > 1e-3 else float("inf")
        tangent = min(tangent, 0.45 * float(len_in), 0.45 * float(len_out))
        if tangent < 1e-3:
            out.append(corner)
            continue
        start = corner - d_in * tangent
        end = corner + d_out * tangent
        # Arc centre lies along the angle bisector, on the inside of the turn.
        bisector = -d_in + d_out
        bis_norm = torch.linalg.norm(bisector)
        if float(bis_norm) < 1e-6:
            out.append(corner)
            continue
        bisector = bisector / bis_norm
        centre_dist = radius / math.sin(half)
        centre = corner + bisector * centre_dist
        a0 = torch.atan2(start[1] - centre[1], start[0] - centre[0])
        a1 = torch.atan2(end[1] - centre[1], end[0] - centre[0])
        # Shortest signed sweep so the arc stays on the fillet.
        delta = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
        out.append(start)
        for s in range(1, samples):
            ang = a0 + delta * (s / samples)
            out.append(centre + radius * torch.stack((torch.cos(ang), torch.sin(ang))))
        out.append(end)
    out.append(points[-1])
    return torch.stack(out, dim=0)


def project_monotone(
    points: torch.Tensor,
    query: torch.Tensor,
    s_prev: float | torch.Tensor,
    window_m: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a point onto a polyline, searching only forward of ``s_prev``.

    The search window is ``[s_prev, s_prev + window_m]``. Arc length therefore
    cannot jump backward, and it cannot skip more than ``window_m`` ahead.

    Args:
        points: Polyline vertices ``(K, 2)`` [m].
        query: Query point ``(2,)`` [m].
        s_prev: Arc length already achieved [m].
        window_m: Forward search window [m].

    Returns:
        Tuple of arc length [m], absolute cross-track error [m], and the
        closest point ``(2,)`` [m].
    """
    cum = cumulative_length(points)
    if not torch.is_tensor(s_prev):
        s_prev_t = torch.tensor(float(s_prev), device=points.device, dtype=points.dtype)
    else:
        s_prev_t = s_prev.to(device=points.device, dtype=points.dtype)
    best_dist = torch.tensor(float("inf"), device=points.device, dtype=points.dtype)
    best_s = s_prev_t
    best_pt = points[0]
    hi = s_prev_t + window_m
    for i in range(points.shape[0] - 1):
        if float(cum[i + 1]) < float(s_prev_t) - 1e-6:
            continue
        if float(cum[i]) > float(hi) + 1e-6:
            break
        a = points[i]
        b = points[i + 1]
        ab = b - a
        denom = torch.dot(ab, ab).clamp(min=1e-8)
        t = torch.clamp(torch.dot(query - a, ab) / denom, 0.0, 1.0)
        pt = a + t * ab
        s = cum[i] + t * (cum[i + 1] - cum[i])
        if float(s) < float(s_prev_t) - 1e-6 or float(s) > float(hi) + 1e-6:
            continue
        dist = torch.linalg.norm(query - pt)
        if float(dist) < float(best_dist):
            best_dist = dist
            best_s = s
            best_pt = pt
    if not torch.isfinite(best_dist):
        best_s = s_prev_t
        best_pt = point_at_distance(points, s_prev_t)
        best_dist = torch.linalg.norm(query - best_pt)
    return best_s, best_dist, best_pt


def resample_polyline(points: torch.Tensor, spacing: float = 0.5, max_points: int = 64) -> torch.Tensor:
    """Evenly resample a polyline, keeping the endpoints.

    Args:
        points: Vertices ``(K, 2)`` [m].
        spacing: Target spacing [m].
        max_points: Cap on the number of samples.

    Returns:
        Resampled vertices ``(M, 2)`` [m].
    """
    total = float(cumulative_length(points)[-1])
    count = max(2, int(total / max(spacing, 1e-3)) + 1)
    count = min(count, max_points)
    distances = torch.linspace(0.0, total, count, device=points.device, dtype=points.dtype)
    return torch.stack([point_at_distance(points, distance) for distance in distances], dim=0)


def project_batch(
    points: torch.Tensor,
    query: torch.Tensor,
    s_prev: torch.Tensor,
    window_m: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project one query per polyline, searching only forward of ``s_prev``.

    Args:
        points: Padded polylines ``(N, K, 2)`` [m]. Repeated tail vertices are ignored.
        query: Query points ``(N, 2)`` [m].
        s_prev: Arc length already achieved ``(N,)`` [m].
        window_m: Forward window [m].

    Returns:
        Arc length ``(N,)`` [m] and absolute cross-track error ``(N,)`` [m].
    """
    seg_a = points[:, :-1, :]
    seg_b = points[:, 1:, :]
    ab = seg_b - seg_a
    seg_len = torch.linalg.norm(ab, dim=-1)
    denom = (ab * ab).sum(dim=-1).clamp(min=1e-8)
    t = ((query.unsqueeze(1) - seg_a) * ab).sum(dim=-1) / denom
    t = t.clamp(0.0, 1.0)
    closest = seg_a + t.unsqueeze(-1) * ab
    cum = cumulative_length(points)
    arc = cum[:, :-1] + t * seg_len
    dist = torch.linalg.norm(query.unsqueeze(1) - closest, dim=-1)
    lo = s_prev.unsqueeze(1) - 1e-4
    hi = (s_prev + window_m).unsqueeze(1) + 1e-4
    valid = (arc >= lo) & (arc <= hi) & (seg_len > 1e-6)
    dist = dist.masked_fill(~valid, float("inf"))
    best = dist.argmin(dim=-1)
    gathered_s = arc.gather(1, best.unsqueeze(1)).squeeze(1)
    gathered_e = dist.gather(1, best.unsqueeze(1)).squeeze(1)
    missing = ~torch.isfinite(gathered_e)
    gathered_s = torch.where(missing, s_prev, gathered_s)
    gathered_e = torch.where(missing, torch.zeros_like(gathered_e), gathered_e)
    return gathered_s, gathered_e


def high_water(s_bar: torch.Tensor, s_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a high-water mark and return ``(new_mark, progress)``."""
    updated = torch.maximum(s_bar, s_new)
    return updated, updated - s_bar
