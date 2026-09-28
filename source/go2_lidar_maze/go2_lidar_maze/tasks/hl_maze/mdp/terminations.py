# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Episode endings for the high-level maze task."""

from __future__ import annotations

import torch

from go2_lidar_maze.constants import REACH_RADIUS_M


def waypoint_timed_out(env, command_name: str) -> torch.Tensor:
    """True when the active waypoint has used its time budget.

    This is registered with ``time_out=True``, so it truncates the episode and
    the critic bootstraps instead of treating it as a failure.
    """
    term = env.command_manager.get_term(command_name)
    return term.time_on_waypoint > term.time_budget


def maze_goal_reached(env, command_name: str) -> torch.Tensor:
    """True when the active waypoint is the maze goal and the robot has reached it."""
    term = env.command_manager.get_term(command_name)
    return term.is_final & (term.distance_to_waypoint() < REACH_RADIUS_M)
