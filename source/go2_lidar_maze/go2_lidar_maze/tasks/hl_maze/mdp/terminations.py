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


def waypoint_reached(env, command_name: str) -> torch.Tensor:
    """True when a privileged episode enters the 0.5 m reach radius.

    Command updates run after terminations, so this latches ``reached_count`` and
    the success metric before the reset. ``bonus_paid`` is left false: the bonus
    reads it on this same step. Online play returns false and still chains waypoints.
    """
    term = env.command_manager.get_term(command_name)
    if term.cfg.source != "privileged":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    arrived = term.distance_to_waypoint() < REACH_RADIUS_M
    term.reached_count[arrived] = 1.0
    issued = term.issued_count.clamp(min=1.0)
    term.metrics["waypoint_success"][:] = term.reached_count / issued
    return arrived


def sustained_illegal_contact(env, threshold: float, sensor_cfg) -> torch.Tensor:
    """True when one body stays above ``threshold`` [N] for every stored contact frame.

    ``illegal_contact`` ends the episode on the strongest frame in the history. A
    one-frame graze near 1 N changes between PhysX builds, so benchmark play uses
    this instead.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_normal_forces_w_history.torch
    norms = torch.linalg.norm(forces[:, :, sensor_cfg.body_ids], dim=-1)
    return torch.any(torch.all(norms > threshold, dim=1), dim=1)
