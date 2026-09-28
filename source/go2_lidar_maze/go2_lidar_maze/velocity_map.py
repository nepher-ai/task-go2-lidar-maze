# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Map a high-level action in [-1, 1]^3 onto the low-level velocity envelope.

Zero action is standing still. Positive and negative x use different scales
because the forward limit (3 m/s) is larger than the reverse limit (2 m/s).
"""

from __future__ import annotations

import torch


def map_action_to_velocity(actions: torch.Tensor) -> torch.Tensor:
    """Clip ``actions`` to [-1, 1] and convert them to ``(vx, vy, wz)``.

    Args:
        actions: High-level output ``(..., 3)``.

    Returns:
        Velocity command ``(..., 3)`` with vx in [-2, 3] m/s, vy in [-1, 1] m/s
        and wz in [-2, 2] rad/s.
    """
    action = actions.clamp(-1.0, 1.0)
    vx = torch.where(action[..., 0] >= 0.0, 3.0 * action[..., 0], 2.0 * action[..., 0])
    vy = action[..., 1]
    wz = 2.0 * action[..., 2]
    return torch.stack((vx, vy, wz), dim=-1)


def rate_limit(
    target: torch.Tensor,
    previous: torch.Tensor,
    max_delta: torch.Tensor,
) -> torch.Tensor:
    """Move ``previous`` toward ``target`` by at most ``max_delta`` per axis."""
    delta = (target - previous).clamp(min=-max_delta, max=max_delta)
    return previous + delta
