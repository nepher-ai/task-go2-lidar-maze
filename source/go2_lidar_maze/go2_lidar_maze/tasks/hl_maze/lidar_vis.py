# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw every robot's LiDAR beams and hit points when a window is open."""

from __future__ import annotations

import math
import weakref

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from go2_lidar_maze.tasks.hl_maze.lidar_frame import ray_origin_w

_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/LidarScan",
    markers={
        "hit": sim_utils.SphereCfg(
            radius=0.04,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.0)),
        ),
        "beam": sim_utils.CylinderCfg(
            radius=0.004,
            height=1.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.85, 1.0)),
        ),
    },
)


class LidarScanVis:
    """Beams from the sensor to each return, and a sphere on each hit.

    A ray that misses the maze is stored as infinity. That beam is still drawn,
    out to ``max_distance``, so an open corridor is not a blank wedge. Every
    robot is drawn.
    """

    def __init__(self, env):
        self._env = weakref.ref(env)
        self._markers = VisualizationMarkers(_MARKER_CFG)

    def _debug_vis_callback(self, _event) -> None:
        env = self._env()
        if env is None:
            return
        sensor = env.scene.sensors["lidar"]
        if sensor.data.ray_hits_w is None or sensor.data.pos_w is None:
            return
        show_beams = bool(env.cfg.vis_lidar_beams)
        show_hits = bool(env.cfg.vis_lidar_hits)
        if not show_beams and not show_hits:
            return
        hits = sensor.data.ray_hits_w.torch
        if hits.ndim == 2:
            hits = hits.unsqueeze(0)
        finite = torch.isfinite(hits).all(dim=-1)
        translations = []
        orientations = []
        scales = []
        indices = []
        if show_hits and bool(finite.any()):
            hit_points = hits[finite]
            count = int(hit_points.shape[0])
            identity = torch.zeros(count, 4, device=hits.device, dtype=hits.dtype)
            identity[:, 3] = 1.0
            translations.append(hit_points)
            orientations.append(identity)
            scales.append(torch.ones(count, 3, device=hits.device, dtype=hits.dtype))
            indices.append(torch.zeros(count, dtype=torch.int32, device=hits.device))
        if show_beams:
            origin = ray_origin_w(sensor)
            ends = _beam_ends(sensor, origin, hits, finite)
            drawable = torch.isfinite(ends).all(dim=-1)
            if bool(drawable.any()):
                origin_exp = origin.unsqueeze(1).expand_as(ends)
                ends = ends[drawable]
                delta = ends - origin_exp[drawable]
                length = torch.linalg.norm(delta, dim=-1).clamp(min=1e-4)
                direction = delta / length.unsqueeze(-1)
                mid = origin_exp[drawable] + direction * (length * 0.5).unsqueeze(-1)
                beam_quat = _quat_align_z(direction)
                beam_scale = torch.ones_like(mid)
                beam_scale[:, 2] = length
                translations.append(mid)
                orientations.append(beam_quat)
                scales.append(beam_scale)
                indices.append(torch.ones(mid.shape[0], dtype=torch.int32, device=hits.device))
        if not translations:
            return
        self._markers.visualize(
            torch.cat(translations, dim=0),
            torch.cat(orientations, dim=0),
            torch.cat(scales, dim=0),
            torch.cat(indices, dim=0),
        )


def attach_lidar_vis(env) -> None:
    """Register the scan drawing when the simulation has a visualizer."""
    if not env.sim.visualizers:
        return
    if not env.cfg.vis_lidar_beams and not env.cfg.vis_lidar_hits:
        return
    vis = LidarScanVis(env)
    env.sim.vis_marker_registry.add_debug_vis_callback(vis)
    # The registry keeps only a weak reference.
    env._lidar_vis = vis


def _beam_ends(sensor, origin: torch.Tensor, hits: torch.Tensor, finite: torch.Tensor) -> torch.Tensor:
    """Hit position, or the ray carried out to ``max_distance`` when it misses.

    ``origin`` is (N, 3) and ``hits`` is (N, B, 3). Misses stay at infinity in
    ``ray_hits_w``. Extending them uses the pattern directions. ``yaw`` and
    ``world`` cast those directions unrotated. ``base`` rotates them by the
    sensor quaternion, the same pose the caster applies.
    """
    ends = hits.clone()
    miss = ~finite
    if not bool(miss.any()):
        return ends
    directions = _cast_directions(sensor, hits.shape[0], hits.shape[1], hits.device, hits.dtype)
    ends[miss] = origin.unsqueeze(1).expand_as(directions)[miss] + directions[miss] * float(sensor.cfg.max_distance)
    return ends


def _cast_directions(sensor, num_envs: int, num_rays: int, device, dtype) -> torch.Tensor:
    """Unit directions (N, B, 3), in the frame the ray caster used."""
    local = sensor.ray_directions.torch.to(device=device, dtype=dtype)
    if local.ndim == 2:
        dirs = local.reshape(-1, 3)[:num_rays].unsqueeze(0).expand(num_envs, -1, -1)
    else:
        dirs = local
        if dirs.shape[0] == 1 and num_envs > 1:
            dirs = dirs.expand(num_envs, -1, -1)
        if dirs.shape[1] != num_rays:
            dirs = dirs[:, :num_rays]
    length = torch.linalg.norm(dirs, dim=-1, keepdim=True)
    bad = ~torch.isfinite(dirs).all(dim=-1) | (length.squeeze(-1) < 1e-6)
    if bool(bad.any()):
        # Same angles as a full-circle pattern: linspace(-180, 180, n+1)[:-1].
        angles = torch.arange(num_rays, device=device, dtype=dtype) * (2.0 * math.pi / num_rays) - math.pi
        fallback = torch.stack((torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)), dim=-1)
        fallback = fallback.unsqueeze(0).expand(num_envs, -1, -1)
        dirs = torch.where(bad.unsqueeze(-1), fallback, dirs / length.clamp(min=1e-6))
    else:
        dirs = dirs / length
    if sensor.cfg.ray_alignment == "base":
        quat = sensor.data.quat_w.torch.to(device=device, dtype=dtype)
        quat = quat.unsqueeze(1).expand(-1, num_rays, -1).contiguous()
        dirs = math_utils.quat_apply(quat, dirs)
    return dirs


def _quat_align_z(direction: torch.Tensor) -> torch.Tensor:
    """Quaternion ``(x, y, z, w)`` that rotates +Z onto a unit ``direction``."""
    z_axis = torch.zeros_like(direction)
    z_axis[:, 2] = 1.0
    axis = torch.linalg.cross(z_axis, direction, dim=-1)
    cos = direction[:, 2].clamp(-1.0, 1.0)
    sin = torch.linalg.norm(axis, dim=-1)
    degenerate = sin < 1e-6
    fallback = torch.zeros_like(axis)
    fallback[:, 0] = 1.0
    axis = torch.where(degenerate.unsqueeze(-1), fallback, axis / sin.clamp(min=1e-6).unsqueeze(-1))
    angle = torch.atan2(sin, cos)
    angle = torch.where(degenerate & (cos < 0.0), torch.full_like(angle, math.pi), angle)
    return math_utils.quat_from_angle_axis(angle, axis)
