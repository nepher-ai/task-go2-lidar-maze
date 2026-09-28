# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Low-level velocity MDP terms that are not already in ``isaaclab.envs.mdp``."""

from .curriculum import widen_velocity_ranges
from .rewards import (
    action_rate_normalized,
    ang_vel_xy_normalized,
    feet_air_time,
    flat_orientation_normalized,
    joint_acc_normalized,
    joint_torque_normalized,
    lin_vel_z_normalized,
    undesired_contacts_capped,
)

__all__ = [
    "action_rate_normalized",
    "feet_air_time",
    "joint_torque_normalized",
    "widen_velocity_ranges",
]
