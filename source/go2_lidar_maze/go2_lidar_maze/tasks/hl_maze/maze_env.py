# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym environment class for the Go2 maze task.

Kept separate from the config module so importing the config does not load
USD Python bindings before Kit starts.
"""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv

from go2_lidar_maze.tasks.hl_maze.episode_vis import attach_episode_vis
from go2_lidar_maze.tasks.hl_maze.lidar_vis import attach_lidar_vis
from go2_lidar_maze.tasks.hl_maze.mdp.events import enlarge_physx_aggregate_pairs
from go2_lidar_maze.tasks.hl_maze.maze_env_cfg import Go2MazeEnvCfg


class Go2MazeEnv(ManagerBasedRLEnv):
    """Maze env that publishes the goal tensor before the first reset."""

    def __init__(self, cfg: Go2MazeEnvCfg, render_mode: str | None = None, **kwargs):
        self.goal_xy = torch.zeros((cfg.scene.num_envs, 2), device=cfg.sim.device)
        self.goal_xy[:, 0] = cfg.goal_xy[0]
        self.goal_xy[:, 1] = cfg.goal_xy[1]
        super().__init__(cfg, render_mode, **kwargs)
        attach_lidar_vis(self)
        attach_episode_vis(self)

    def _init_sim(self):
        """Write PhysX GPU capacities as ``UInt`` immediately before physics play."""
        sim_reset = self.sim.reset

        def _reset(soft: bool = False):
            enlarge_physx_aggregate_pairs(self, None)
            sim_reset(soft)

        self.sim.reset = _reset  # type: ignore[method-assign]
        try:
            super()._init_sim()
        finally:
            self.sim.reset = sim_reset  # type: ignore[method-assign]
