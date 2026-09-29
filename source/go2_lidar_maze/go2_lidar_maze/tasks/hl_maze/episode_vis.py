# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Draw every robot's episode spawn and waypoint when a window is open."""

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
    },
)


class EpisodeMarkerVis:
    """Green sphere at the spawn, red sphere at the waypoint.

    Privileged play samples both when the episode resets. The path keeps the
    spawn as its first point and the waypoint as its last, so the spheres stay
    put until the next reset. Every robot is drawn.
    """

    def __init__(self, env):
        self._env = weakref.ref(env)
        self._markers = VisualizationMarkers(_MARKER_CFG)

    def _debug_vis_callback(self, _event) -> None:
        env = self._env()
        if env is None:
            return
        term = env.command_manager.get_term("waypoint")
        active = term.issued_count >= 1.0
        if not bool(active.any()):
            return
        start = term.path_xy[active, 0]
        goal = term.waypoint_w[active]
        height = torch.full((start.shape[0], 1), _SPHERE_RADIUS_M, device=start.device, dtype=start.dtype)
        translations = torch.cat((torch.cat((start, height), dim=-1), torch.cat((goal, height), dim=-1)), dim=0)
        orientations = torch.zeros(translations.shape[0], 4, device=start.device, dtype=start.dtype)
        orientations[:, 3] = 1.0
        count = start.shape[0]
        indices = torch.cat(
            (
                torch.zeros(count, dtype=torch.int32, device=start.device),
                torch.ones(count, dtype=torch.int32, device=start.device),
            ),
            dim=0,
        )
        self._markers.visualize(translations, orientations, marker_indices=indices)


def attach_episode_vis(env) -> None:
    """Register the spawn and waypoint spheres when the simulation has a visualizer."""
    if not env.sim.visualizers or not env.cfg.vis_start_goal:
        return
    vis = EpisodeMarkerVis(env)
    env.sim.vis_marker_registry.add_debug_vis_callback(vis)
    # The registry keeps only a weak reference.
    env._episode_vis = vis
