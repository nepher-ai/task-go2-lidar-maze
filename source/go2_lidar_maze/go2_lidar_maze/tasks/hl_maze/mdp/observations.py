# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level observations: one waypoint, 360-degree ranges, and proprioception."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from go2_lidar_maze.constants import CELL_PITCH_M, MAX_ASTAR_CELLS


def lidar_ranges(
    env, sensor_cfg: SceneEntityCfg, min_range: float, max_range: float, corrupt: bool = False
) -> torch.Tensor:
    """Normalized LiDAR ranges in ``[0, 1]``. Misses become ``max_range``.

    When ``corrupt`` is true, each beam gets uniform noise of ±0.03 m and a 2%
    chance of being dropped to ``max_range``. Both happen before normalization.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch
    origin = sensor.data.pos_w.torch.unsqueeze(1)
    dist = torch.linalg.norm(hits - origin, dim=-1)
    dist = torch.where(torch.isfinite(dist), dist, torch.full_like(dist, max_range))
    if corrupt:
        dist = dist + (torch.rand_like(dist) - 0.5) * 0.06
        drop = torch.rand_like(dist) < 0.02
        dist = torch.where(drop, torch.full_like(dist, max_range), dist)
    dist = dist.clamp(min_range, max_range)
    return dist / max_range


def waypoint_b(env, command_name: str) -> torch.Tensor:
    """Single body-frame waypoint, scaled by the 5-cell distance and clipped."""
    return env.command_manager.get_term(command_name).waypoint_b()


def remaining_path(env, command_name: str) -> torch.Tensor:
    """Privileged remaining A* length, divided by the 5-cell distance. Critic only."""
    term = env.command_manager.get_term(command_name)
    remaining = (term.path_length - term.s_bar).clamp(min=0.0)
    return (remaining / (MAX_ASTAR_CELLS * CELL_PITCH_M)).unsqueeze(-1)


def base_lin_vel_xy(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Planar body linear velocity [m/s]."""
    return env.scene[asset_cfg.name].data.root_lin_vel_b.torch[:, :2]


def base_yaw_rate(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body yaw rate [rad/s]."""
    return env.scene[asset_cfg.name].data.root_ang_vel_b.torch[:, 2:3]
