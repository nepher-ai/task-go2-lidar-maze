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

from go2_lidar_maze.constants import CELL_CENTER_ORIGIN_M, CELL_PITCH_M, GRID_SIZE
from go2_lidar_maze.tasks.hl_maze.benchmark import load_benchmark
from go2_lidar_maze.tasks.hl_maze.episode_vis import attach_episode_vis
from go2_lidar_maze.tasks.hl_maze.film_recorder import MazeFilm, claim_video
from go2_lidar_maze.tasks.hl_maze.lidar_vis import attach_lidar_vis
from go2_lidar_maze.tasks.hl_maze.mdp.events import enlarge_physx_aggregate_pairs
from go2_lidar_maze.tasks.hl_maze.maze_env_cfg import Go2MazeEnvCfg


class Go2MazeEnv(ManagerBasedRLEnv):
    """Maze env that publishes the goal tensor before the first reset."""

    def __init__(self, cfg: Go2MazeEnvCfg, render_mode: str | None = None, **kwargs):
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        self.goal_xy = torch.zeros((num_envs, 2), device=device)
        self.goal_xy[:, 0] = cfg.goal_xy[0]
        self.goal_xy[:, 1] = cfg.goal_xy[1]
        # Geometry must exist before the command term and the first reset.
        # ``--video`` is claimed first so the stock 10 fps recorder is not built
        # and the viewport renders on the 50 Hz low-level tick.
        film_target = claim_video(cfg)
        self._apply_benchmark(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        attach_lidar_vis(self)
        attach_episode_vis(self)
        self._maze_film = MazeFilm(self, *film_target) if film_target is not None else None

    def close(self):
        film = getattr(self, "_maze_film", None)
        if film is not None:
            film.close()
            self._maze_film = None
        super().close()

    def _apply_benchmark(self, cfg: Go2MazeEnvCfg) -> None:
        """Publish per-env cell geometry. A scenario replaces the single training maze."""
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        scenario_path = getattr(cfg, "scenario_path", "") or ""
        if not scenario_path:
            self.maze_cell_origin_xy = torch.full((num_envs, 2), CELL_CENTER_ORIGIN_M, device=device)
            self.maze_cell_pitch = torch.full((num_envs,), CELL_PITCH_M, device=device)
            self.maze_grid_n = torch.full((num_envs,), GRID_SIZE, dtype=torch.long, device=device)
            self.tile_origin_xy = torch.zeros((num_envs, 2), device=device)
            self.optimal_route_m = torch.full((num_envs,), float("nan"), device=device)
            self._scenario_spawn_xy = None
            self._scenario_spawn_yaw = None
            self._scenario_goal_xy = None
            return

        layout = load_benchmark(scenario_path).for_envs(num_envs)
        self.maze_cell_origin_xy = layout.cell_origin_xy.to(device)
        self.maze_cell_pitch = layout.cell_pitch.to(device)
        self.maze_grid_n = layout.grid_n.to(device)
        self.tile_origin_xy = layout.tile_origin_xy.to(device)
        self.optimal_route_m = layout.optimal_route_m.to(device)
        self._scenario_spawn_xy = layout.spawn_xy.to(device)
        self._scenario_spawn_yaw = layout.spawn_yaw.to(device)
        self._scenario_goal_xy = layout.goal_xy.to(device)
        self.goal_xy[:] = self._scenario_goal_xy

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
