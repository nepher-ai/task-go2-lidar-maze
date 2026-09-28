# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level rewards. Each value is per control step, before Isaac Lab's ``* dt``.

Isaac Lab multiplies every reward by ``step_dt``. The functions below divide by
``step_dt`` so the value that enters the return is the per-step expression in
the README, not that expression times 0.1 s.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg

from go2_lidar_maze.constants import (
    ACTION_RATE_WEIGHT,
    BRAKE_DECEL_MPS2,
    BRAKE_LATENCY_S,
    BRAKE_MARGIN_M,
    BRAKE_WEIGHT,
    CLEARANCE_SAFE_M,
    CLEARANCE_WEIGHT,
    FOOTPRINT_A_M,
    FOOTPRINT_B_M,
    LATERAL_REF_M,
    LATERAL_WEIGHT,
    PROGRESS_K_PER_M,
    REACH_RADIUS_M,
    TIME_COST,
    VX_MAX,
    WAYPOINT_BONUS,
)
from go2_lidar_maze.nav_core.path_utils import high_water, project_batch


def _scale(env, value: torch.Tensor) -> torch.Tensor:
    return value / env.step_dt


def progress(env, command_name: str) -> torch.Tensor:
    """High-water progress along the active waypoint path, in units of ``k * ds``."""
    term = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]
    query = robot.data.root_pos_w.torch[:, :2]
    arc, cross = project_batch(term.path_xy, query, term.s_bar, window_m=3.0)
    term.cross_track = cross
    term.s_bar, delta = high_water(term.s_bar, arc)
    return _scale(env, PROGRESS_K_PER_M * delta)


def time_cost(env) -> torch.Tensor:
    """Constant per-step cost. Break-even speed is ``c_t / (k dt)``."""
    return _scale(env, torch.full((env.num_envs,), TIME_COST, device=env.device))


def lateral(env, command_name: str) -> torch.Tensor:
    """Squared cross-track error, capped at 1 when the error reaches ``e_ref``."""
    term = env.command_manager.get_term(command_name)
    penalty = (term.cross_track.square() / LATERAL_REF_M**2).clamp(max=1.0)
    return _scale(env, LATERAL_WEIGHT * penalty)


def clearance(env, sensor_cfg: SceneEntityCfg, max_range: float = 10.0) -> torch.Tensor:
    """Squared shortfall of the nearest body-edge clearance below ``c_safe``."""
    c_min = _min_clearance(env, sensor_cfg, max_range)
    shortfall = (1.0 - c_min / CLEARANCE_SAFE_M).clamp(min=0.0).square()
    return _scale(env, CLEARANCE_WEIGHT * shortfall)


def braking(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, max_range: float = 10.0) -> torch.Tensor:
    """Squared excess of planar speed over the speed that can still stop in time."""
    robot = env.scene[asset_cfg.name]
    velocity = robot.data.root_lin_vel_b.torch[:, :2]
    speed = torch.linalg.norm(velocity, dim=-1)
    heading = torch.atan2(velocity[:, 1], velocity[:, 0])
    c_v = _strip_clearance(env, sensor_cfg, heading, max_range)
    safe = _v_safe_tensor(c_v)
    excess = ((speed - safe).clamp(min=0.0) / VX_MAX).square()
    return _scale(env, BRAKE_WEIGHT * excess)


def action_rate(env) -> torch.Tensor:
    """Squared action change divided by 12, its maximum over 3 axes of width 2."""
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    penalty = torch.sum(torch.square(action - prev), dim=1) / 12.0
    return _scale(env, ACTION_RATE_WEIGHT * penalty.clamp(max=1.0))


def waypoint_bonus(env, command_name: str) -> torch.Tensor:
    """``R_g`` once per waypoint, on the step the reach radius is entered.

    The command term marks the bonus paid after the reward, so this reads the
    distance directly and ignores a flag that is still false on the arrival step.
    """
    term = env.command_manager.get_term(command_name)
    arrived = term.distance_to_waypoint() < REACH_RADIUS_M
    fresh = arrived & ~term.bonus_paid
    return _scale(env, WAYPOINT_BONUS * fresh.float())


def failure(env) -> torch.Tensor:
    """One-shot penalty on a real termination (collision or fall), not a timeout."""
    return env.termination_manager.terminated.float() / env.step_dt


def _v_safe_tensor(clearance: torch.Tensor) -> torch.Tensor:
    available = (clearance - BRAKE_MARGIN_M).clamp(min=0.0)
    inside = BRAKE_LATENCY_S**2 + 2.0 * available / BRAKE_DECEL_MPS2
    return BRAKE_DECEL_MPS2 * (-BRAKE_LATENCY_S + torch.sqrt(inside))


def _beam_clearance(ranges: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Clearance beyond the rectangular footprint along each beam [m]."""
    cos_t = torch.cos(angles).abs().clamp(min=1e-4)
    sin_t = torch.sin(angles).abs().clamp(min=1e-4)
    rho = torch.minimum(FOOTPRINT_A_M / cos_t, FOOTPRINT_B_M / sin_t)
    return ranges - rho


def _ranges(env, sensor_cfg: SceneEntityCfg, max_range: float) -> tuple[torch.Tensor, torch.Tensor]:
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch
    origin = sensor.data.pos_w.torch
    # Body-frame ray direction from the yaw-aligned sensor.
    yaw = _yaw(env.scene["robot"].data.root_quat_w.torch)
    delta = hits[..., :2] - origin[:, :2].unsqueeze(1)
    cos_y = torch.cos(yaw).unsqueeze(1)
    sin_y = torch.sin(yaw).unsqueeze(1)
    bx = cos_y * delta[..., 0] + sin_y * delta[..., 1]
    by = -sin_y * delta[..., 0] + cos_y * delta[..., 1]
    ranges = torch.linalg.norm(torch.stack((bx, by), dim=-1), dim=-1)
    ranges = torch.where(torch.isfinite(ranges), ranges, torch.full_like(ranges, max_range))
    ranges = ranges.clamp(max=max_range)
    angles = torch.atan2(by, bx)
    return ranges, angles


def _min_clearance(env, sensor_cfg: SceneEntityCfg, max_range: float) -> torch.Tensor:
    ranges, angles = _ranges(env, sensor_cfg, max_range)
    return _beam_clearance(ranges, angles).amin(dim=-1)


def _strip_clearance(env, sensor_cfg: SceneEntityCfg, heading: torch.Tensor, max_range: float) -> torch.Tensor:
    """Nearest clearance inside a strip of width ``2b`` around the velocity direction."""
    ranges, angles = _ranges(env, sensor_cfg, max_range)
    # ``heading`` is the velocity direction in the body frame.
    delta = torch.atan2(torch.sin(angles - heading.unsqueeze(1)), torch.cos(angles - heading.unsqueeze(1)))
    lateral = ranges * torch.sin(delta).abs()
    in_strip = lateral <= FOOTPRINT_B_M
    ahead = delta.abs() < (math.pi / 2)
    clearance = _beam_clearance(ranges, angles)
    clearance = clearance.masked_fill(~(in_strip & ahead), max_range)
    return clearance.amin(dim=-1)


def _yaw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Yaw [rad] from an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
