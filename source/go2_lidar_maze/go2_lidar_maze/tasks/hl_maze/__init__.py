# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="Nepher-Go2-LidarMaze-HL-v0",
    entry_point="go2_lidar_maze.tasks.hl_maze.maze_env:Go2MazeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.maze_env_cfg:Go2MazeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2MazePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Nepher-Go2-LidarMaze-HL-Play-v0",
    entry_point="go2_lidar_maze.tasks.hl_maze.maze_env:Go2MazeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.maze_env_cfg:Go2MazeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2MazePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Nepher-Go2-LidarMaze-Planner-Play-v0",
    entry_point="go2_lidar_maze.tasks.hl_maze.maze_env:Go2MazeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.maze_env_cfg:Go2MazeEnvCfg_PLANNER",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2MazePPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
