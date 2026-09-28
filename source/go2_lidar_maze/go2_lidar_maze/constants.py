# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared physical constants for the Go2 LiDAR maze task.

Lengths are metres, angles are radians, and times are seconds unless noted.
"""

from __future__ import annotations

import math

# Maze cell geometry, matching enclosed_20x20.txt.
GRID_SIZE = 20
CELL_PITCH_M = 2.5
CORRIDOR_WIDTH_M = 2.0
WALL_THICKNESS_M = 0.5
WALL_HEIGHT_M = 1.0
CELL_CENTER_ORIGIN_M = 1.5

# A* waypoint contract: path length in maze cells, except the final goal.
MIN_ASTAR_CELLS = 1
MAX_ASTAR_CELLS = 5
REACH_RADIUS_M = 0.5
SPAWN_JITTER_M = 0.4

# Velocity envelope of the low-level policy.
VX_MIN = -2.0
VX_MAX = 3.0
VY_MIN = -1.0
VY_MAX = 1.0
WZ_MIN = -2.0
WZ_MAX = 2.0

# High-level timing.
HL_DT_S = 0.1
GAMMA = 0.99

# Robot footprint used by the clearance reward: half-length and half-width [m].
FOOTPRINT_A_M = 0.35
FOOTPRINT_B_M = 0.20

# Braking model. a_b and tau_ll are replaced by measured values after ll training.
BRAKE_DECEL_MPS2 = 2.5
BRAKE_LATENCY_S = 0.2
BRAKE_MARGIN_M = 0.3

# Reward weights. See README for the derivation of each scale.
PROGRESS_K_PER_M = 1.0 / (VX_MAX * HL_DT_S)
TIME_COST = 0.05
LATERAL_WEIGHT = 0.1
LATERAL_REF_M = 0.6
CLEARANCE_WEIGHT = 0.5
CLEARANCE_SAFE_M = 0.3
BRAKE_WEIGHT = 0.2
ACTION_RATE_WEIGHT = 0.05
WAYPOINT_BONUS = 10.0
PENALTY_MARGIN = 1.25


def penalty_budget() -> float:
    """Worst per-step penalty, before the collision term."""
    return TIME_COST + LATERAL_WEIGHT + CLEARANCE_WEIGHT + BRAKE_WEIGHT + ACTION_RATE_WEIGHT


def collision_penalty(gamma: float = GAMMA) -> float:
    """Return penalty that dominates continuing forever at the worst per-step cost.

    Continuing can cost at most ``penalty_budget() / (1 - gamma)`` in discounted
    return, so the collision penalty is ``margin`` times that value.
    """
    return PENALTY_MARGIN * penalty_budget() / (1.0 - gamma)


def half_reward_sigma_sq(half_error: float) -> float:
    """``sigma^2`` of ``exp(-e^2 / sigma^2)`` so the reward is 1/2 at ``half_error``."""
    return half_error**2 / math.log(2.0)


def v_safe(
    clearance_m: float,
    *,
    a_b: float = BRAKE_DECEL_MPS2,
    tau_s: float = BRAKE_LATENCY_S,
    margin_m: float = BRAKE_MARGIN_M,
) -> float:
    """Largest speed whose stopping distance fits inside ``clearance_m``.

    Stopping distance is ``d(v) = v * tau + v^2 / (2 a_b)``. Solving
    ``d(v) = clearance - margin`` for the positive root gives

        v = a_b * (-tau + sqrt(tau^2 + 2 (clearance - margin) / a_b)).
    """
    available = clearance_m - margin_m
    if available <= 0.0 or a_b <= 0.0:
        return 0.0
    return a_b * (-tau_s + math.sqrt(tau_s**2 + 2.0 * available / a_b))
