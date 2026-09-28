# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the Go2 from the maze start to the maze goal.

The online planner supplies one waypoint at a time. The high-level policy
defaults to ``best_policy/best_policy.pt``. The frozen low-level policy is
``best_policy/ll_policy/ll_policy.pt``.

    .\\isaaclab.bat -p source\\task-go2-lidar-maze\\scripts\\navigate_maze.py
    .\\isaaclab.bat -p source\\task-go2-lidar-maze\\scripts\\navigate_maze.py --checkpoint PATH\\model.pt
"""

from __future__ import annotations

import argparse
import importlib

from go2_lidar_maze.paths import hl_policy_path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 LiDAR maze navigation demo.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=str(hl_policy_path()),
    help="High-level RSL-RL checkpoint. Defaults to best_policy/best_policy.pt.",
)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import go2_lidar_maze.tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

TASK = "Nepher-Go2-LidarMaze-HL-Play-v0"


def _load_cfg(task_name: str, entry_point_key: str):
    spec = gym.spec(task_name)
    entry = spec.kwargs[entry_point_key]
    module_name, attr_name = entry.split(":")
    module = importlib.import_module(module_name)
    cfg_cls = getattr(module, attr_name)
    return cfg_cls()


def _tracker(env) -> torch.Tensor:
    """Body-frame P controller. The waypoint is already scaled by 12.5 m."""
    base = env.unwrapped
    term = base.command_manager.get_term("waypoint")
    waypoint = term.waypoint_b()
    action = torch.zeros(base.num_envs, 3, device=waypoint.device)
    action[:, 0] = (1.5 * waypoint[:, 0]).clamp(-1.0, 1.0)
    action[:, 2] = (2.0 * torch.atan2(waypoint[:, 1], waypoint[:, 0].clamp(min=1e-3))).clamp(-1.0, 1.0)
    return action


def main() -> None:
    env_cfg = _load_cfg(TASK, "env_cfg_entry_point")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    policy = None
    if args_cli.checkpoint:
        agent_cfg = _load_cfg(TASK, "rsl_rl_cfg_entry_point")
        agent_cfg.device = args_cli.device
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
    while simulation_app.is_running():
        with torch.inference_mode():
            if policy is None:
                action = _tracker(env)
                env.step(action)
            else:
                action = policy(obs)
                obs, _, dones, _ = env.step(action)
                if hasattr(policy, "reset"):
                    policy.reset(dones)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
