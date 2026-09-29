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
from go2_lidar_maze.tasks.hl_maze.lidar_frame import footprint_exit, ray_origin_w


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
    """One-shot penalty on a collision or a fall.

    Reaching the waypoint is also a non-timeout termination, so this does not
    read ``terminated``. A success must not pay ``C``.
    """
    collision = env.termination_manager.get_term("collision")
    fallen = env.termination_manager.get_term("bad_orientation")
    return (collision | fallen).float() / env.step_dt


def _v_safe_tensor(clearance: torch.Tensor) -> torch.Tensor:
    available = (clearance - BRAKE_MARGIN_M).clamp(min=0.0)
    inside = BRAKE_LATENCY_S**2 + 2.0 * available / BRAKE_DECEL_MPS2
    return BRAKE_DECEL_MPS2 * (-BRAKE_LATENCY_S + torch.sqrt(inside))


def _beam_clearance(ranges: torch.Tensor, angles: torch.Tensor, sensor_xy: torch.Tensor) -> torch.Tensor:
    """Clearance [m] beyond the footprint, measured from the ray origin."""
    return ranges - footprint_exit(sensor_xy, angles)


def _ranges(
    env, sensor_cfg: SceneEntityCfg, max_range: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Planar range and azimuth from the ray origin, and that origin in the yaw frame.

    Returns:
        Range [m], azimuth [rad], and the ray origin in the yaw frame [m].
        Shapes are (N, B), (N, B), and (N, 2).
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]
    hits = sensor.data.ray_hits_w.torch
    origin = ray_origin_w(sensor)
    yaw = _yaw(robot.data.root_quat_w.torch)
    root_xy = robot.data.root_pos_w.torch[:, :2]
    rel = origin[:, :2] - root_xy
    cos_y = torch.cos(yaw)
    sin_y = torch.sin(yaw)
    sensor_xy = torch.stack((cos_y * rel[:, 0] + sin_y * rel[:, 1], -sin_y * rel[:, 0] + cos_y * rel[:, 1]), dim=-1)
    delta = hits[..., :2] - origin[:, :2].unsqueeze(1)
    # A miss is NaN. Leaving it in the angle makes the whole clearance term NaN.
    finite = torch.isfinite(delta).all(dim=-1)
    delta = torch.where(finite.unsqueeze(-1), delta, torch.zeros_like(delta))
    cos_b = cos_y.unsqueeze(1)
    sin_b = sin_y.unsqueeze(1)
    bx = cos_b * delta[..., 0] + sin_b * delta[..., 1]
    by = -sin_b * delta[..., 0] + cos_b * delta[..., 1]
    ranges = torch.linalg.norm(torch.stack((bx, by), dim=-1), dim=-1)
    ranges = torch.where(finite, ranges, torch.full_like(ranges, max_range))
    ranges = ranges.clamp(max=max_range)
    angles = torch.atan2(by, bx)
    return ranges, angles, sensor_xy


def _min_clearance(env, sensor_cfg: SceneEntityCfg, max_range: float) -> torch.Tensor:
    ranges, angles, sensor_xy = _ranges(env, sensor_cfg, max_range)
    return _beam_clearance(ranges, angles, sensor_xy).amin(dim=-1)


def _strip_clearance(env, sensor_cfg: SceneEntityCfg, heading: torch.Tensor, max_range: float) -> torch.Tensor:
    """Nearest clearance inside a strip of width ``2b`` around the velocity direction."""
    ranges, angles, sensor_xy = _ranges(env, sensor_cfg, max_range)
    # ``heading`` is the velocity direction in the body frame.
    delta = torch.atan2(torch.sin(angles - heading.unsqueeze(1)), torch.cos(angles - heading.unsqueeze(1)))
    lateral = ranges * torch.sin(delta).abs()
    in_strip = lateral <= FOOTPRINT_B_M
    ahead = delta.abs() < (math.pi / 2)
    clearance = _beam_clearance(ranges, angles, sensor_xy)
    clearance = clearance.masked_fill(~(in_strip & ahead), max_range)
    return clearance.amin(dim=-1)


def _yaw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Yaw [rad] from an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
