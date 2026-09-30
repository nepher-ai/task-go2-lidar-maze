# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw maze start, maze goal, and the current target waypoint when a window is open."""

from __future__ import annotations

import weakref

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# Sits on the floor and stays smaller than the 2 m corridor.
_SPHERE_RADIUS_M = 0.1

_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Episode",
    markers={
        "start": sim_utils.SphereCfg(
            radius=_SPHERE_RADIUS_M,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.25)),
        ),
        "goal": sim_utils.SphereCfg(
            radius=_SPHERE_RADIUS_M,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.15, 0.12)),
        ),
        "waypoint": sim_utils.SphereCfg(
            radius=_SPHERE_RADIUS_M,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.45, 1.0)),
        ),
    },
)


class EpisodeMarkerVis:
    """Green sphere at the spawn, red sphere at the maze goal, blue at the target waypoint.

    On the EnvHub benchmark the green and red spheres are the tile's recorded
    start and goal. The blue sphere is only the waypoint the policy is chasing,
    not the samples along the path. Privileged play has no separate maze goal,
    so the red sphere is that episode's waypoint and the blue sphere is omitted.
    Every robot is drawn.
    """

    def __init__(self, env):
        self._env = weakref.ref(env)
        self._markers = VisualizationMarkers(_MARKER_CFG)

    def _debug_vis_callback(self, _event) -> None:
        env = self._env()
        if env is None:
            return
        spawn = getattr(env, "_scenario_spawn_xy", None)
        if spawn is not None:
            self._draw_benchmark(env, spawn)
            return
        term = env.command_manager.get_term("waypoint")
        active = term.issued_count >= 1.0
        if not bool(active.any()):
            return
        start = term.path_xy[active, 0]
        goal = term.waypoint_w[active]
        self._show(start, goal, None)

    def _draw_benchmark(self, env, spawn: torch.Tensor) -> None:
        """Start and goal of each tile, plus the current target waypoint."""
        term = env.command_manager.get_term("waypoint")
        active = term.issued_count >= 1.0
        waypoint = term.waypoint_w[active] if bool(active.any()) else None
        self._show(spawn, env.goal_xy, waypoint)

    def _show(self, start: torch.Tensor, goal: torch.Tensor, waypoint: torch.Tensor | None) -> None:
        """Draw ``start`` as green, ``goal`` as red, and ``waypoint`` as blue."""
        device = start.device
        dtype = start.dtype
        parts = [_lift(start, dtype), _lift(goal, dtype)]
        counts = [start.shape[0], goal.shape[0]]
        kinds = [0, 1]
        if waypoint is not None and waypoint.shape[0] > 0:
            parts.append(_lift(waypoint, dtype))
            counts.append(waypoint.shape[0])
            kinds.append(2)
        translations = torch.cat(parts, dim=0)
        orientations = torch.zeros(translations.shape[0], 4, device=device, dtype=dtype)
        orientations[:, 3] = 1.0
        indices = torch.cat(
            [torch.full((count,), kind, dtype=torch.int32, device=device) for count, kind in zip(counts, kinds)],
            dim=0,
        )
        self._markers.visualize(translations, orientations, marker_indices=indices)


def _lift(xy: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """XY points [m] with the sphere centre one radius above the floor."""
    height = torch.full((xy.shape[0], 1), _SPHERE_RADIUS_M, device=xy.device, dtype=dtype)
    return torch.cat((xy.to(dtype), height), dim=-1)


def attach_episode_vis(env) -> None:
    """Register the start, goal, and waypoint spheres when the simulation has a visualizer."""
    if not env.sim.visualizers or not env.cfg.vis_start_goal:
        return
    vis = EpisodeMarkerVis(env)
    env.sim.vis_marker_registry.add_debug_vis_callback(vis)
    # The registry keeps only a weak reference.
    env._episode_vis = vis
