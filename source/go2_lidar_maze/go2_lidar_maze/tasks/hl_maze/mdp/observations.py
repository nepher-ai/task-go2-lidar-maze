# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level observations: one waypoint, path look-ahead, ranges, and proprioception."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from go2_lidar_maze.constants import CELL_PITCH_M, MAX_ASTAR_CELLS
from go2_lidar_maze.nav_core.path_utils import points_at_arc_length
from go2_lidar_maze.tasks.hl_maze.lidar_frame import ray_origin_w


def lidar_ranges(
    env, sensor_cfg: SceneEntityCfg, min_range: float, max_range: float, corrupt: bool = False
) -> torch.Tensor:
    """Normalized LiDAR ranges in ``[0, 1]``. Misses become ``max_range``.

    When ``corrupt`` is true, each beam gets uniform noise of ±0.03 m and a 2%
    chance of being dropped to ``max_range``. Both happen before normalization.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w.torch
    origin = ray_origin_w(sensor).unsqueeze(1)
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


def path_lookahead(env, command_name: str, offsets_m: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)) -> torch.Tensor:
    """Body-frame points on the active path, ahead of the high-water mark.

    Each offset is an arc length [m] added to ``s_bar`` and clamped to the end
    of the path. The point is divided by the 5-cell lookahead, 12.5 m, the same
    scale as the waypoint. Four offsets give eight values. The furthest, 4 m,
    covers the 2.4 m stop from 3 m/s.
    """
    term = env.command_manager.get_term(command_name)
    offsets = torch.as_tensor(offsets_m, device=term.s_bar.device, dtype=term.s_bar.dtype)
    distance = term.s_bar.unsqueeze(1) + offsets.unsqueeze(0)
    world = points_at_arc_length(term.path_xy, distance)
    pos = term.robot.data.root_pos_w.torch[:, :2]
    yaw = _yaw(term.robot.data.root_quat_w.torch)
    delta = world - pos.unsqueeze(1)
    cos_y = torch.cos(yaw).unsqueeze(1)
    sin_y = torch.sin(yaw).unsqueeze(1)
    bx = cos_y * delta[..., 0] + sin_y * delta[..., 1]
    by = -sin_y * delta[..., 0] + cos_y * delta[..., 1]
    scale = MAX_ASTAR_CELLS * CELL_PITCH_M
    body = torch.stack((bx, by), dim=-1).clamp(-scale, scale) / scale
    return body.reshape(body.shape[0], -1)


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


def _yaw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Yaw [rad] from an ``(x, y, z, w)`` quaternion."""
    x, y, z, w = quat_xyzw.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
