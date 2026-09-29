# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR mount frame. ``data.pos_w`` is the parent body, not the ray origin."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from go2_lidar_maze.constants import FOOTPRINT_A_M, FOOTPRINT_B_M


def ray_origin_w(sensor) -> torch.Tensor:
    """World position where the rays start [m].

    The caster keeps ``data.pos_w`` on the parent body and stores ``cfg.offset``
    in the local ray starts. ``ray_alignment`` rotates that offset the same way
    the caster rotates the rays.

    Args:
        sensor: Ray caster. Reading ``data`` refreshes the pose buffers.

    Returns:
        Ray origin for each environment. Shape is (N, 3).
    """
    pos = sensor.data.pos_w.torch
    local = sensor.ray_starts.torch
    if local.ndim == 3:
        local = local[:, 0]
    else:
        local = local[:1].expand(pos.shape[0], -1)
    local = local.to(device=pos.device, dtype=pos.dtype)
    alignment = sensor.cfg.ray_alignment
    if alignment == "world":
        return pos + local
    quat = sensor.data.quat_w.torch
    if alignment == "yaw":
        return pos + math_utils.quat_apply_yaw(quat, local)
    return pos + math_utils.quat_apply(quat, local)


def footprint_exit(sensor_xy: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Distance [m] along each beam from the ray origin to where it leaves the footprint.

    ``sensor_xy`` has shape (N, 2) and ``angles`` has shape (N, B), measured
    from body +X. An origin at (0, 0) returns ``min(a / |cos θ|, b / |sin θ|)``.
    An origin outside the rectangle returns 0 for a beam that misses it. The
    mount at x = 0.36 m is past the front edge, so a forward beam returns 0.

    Args:
        sensor_xy: Ray origin in the yaw frame [m].
        angles: Beam azimuth in that frame [rad].

    Returns:
        Exit distance along each beam. Shape is (N, B).
    """
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)
    sx = sensor_xy[:, 0:1]
    sy = sensor_xy[:, 1:2]
    # A beam parallel to an edge must not divide by zero. The large quotient
    # loses to the edge the beam actually meets.
    cos_s = torch.where(cos_t >= 0, cos_t.clamp(min=1e-4), cos_t.clamp(max=-1e-4))
    sin_s = torch.where(sin_t >= 0, sin_t.clamp(min=1e-4), sin_t.clamp(max=-1e-4))
    tx_far = torch.where(cos_t >= 0, (FOOTPRINT_A_M - sx) / cos_s, (-FOOTPRINT_A_M - sx) / cos_s)
    ty_far = torch.where(sin_t >= 0, (FOOTPRINT_B_M - sy) / sin_s, (-FOOTPRINT_B_M - sy) / sin_s)
    tx_near = torch.where(cos_t >= 0, (-FOOTPRINT_A_M - sx) / cos_s, (FOOTPRINT_A_M - sx) / cos_s)
    ty_near = torch.where(sin_t >= 0, (-FOOTPRINT_B_M - sy) / sin_s, (FOOTPRINT_B_M - sy) / sin_s)
    t_far = torch.minimum(tx_far, ty_far)
    t_near = torch.maximum(tx_near, ty_near)
    crosses = (t_far >= 0) & (t_near <= t_far)
    return torch.where(crosses, t_far, torch.zeros_like(t_far))
