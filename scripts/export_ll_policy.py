# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export a trained low-level checkpoint to TorchScript.

Writes ``best_policy/ll_policy/ll_policy.pt`` under the task project.

Run from f:\\isaacLab, after training Nepher-Go2-LidarMaze-LL-v0:

    .\\isaaclab.bat -p source\\task-go2-lidar-maze\\scripts\\export_ll_policy.py --checkpoint PATH\\model.pt
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export the Go2 velocity policy to TorchScript.")
parser.add_argument("--checkpoint", type=str, required=True, help="RSL-RL model_*.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from packaging import version
from rsl_rl.runners import OnPolicyRunner

import go2_lidar_maze.tasks  # noqa: F401
from go2_lidar_maze.paths import ll_policy_path
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, handle_deprecated_rsl_rl_cfg

TASK = "Nepher-Go2-LidarMaze-LL-Play-v0"
OUTPUT = ll_policy_path()


def _load_cfg(task_name: str, entry_point_key: str):
    spec = gym.spec(task_name)
    entry = spec.kwargs[entry_point_key]
    module_name, attr_name = entry.split(":")
    module = importlib.import_module(module_name)
    cfg_cls = getattr(module, attr_name)
    return cfg_cls()


def main() -> None:
    env_cfg = _load_cfg(TASK, "env_cfg_entry_point")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    agent_cfg = _load_cfg(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    # Drops the legacy ``stochastic`` fields that rsl-rl >= 5 does not accept.
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    installed = version.parse(installed_version)
    if installed >= version.parse("4.0.0"):
        runner.export_policy_to_jit(path=str(OUTPUT.parent), filename=OUTPUT.name)
    else:
        policy = runner.alg.policy
        normalizer = getattr(policy, "actor_obs_normalizer", None)
        export_policy_as_jit(policy, normalizer, path=str(OUTPUT.parent), filename=OUTPUT.name)
    print(f"Wrote {OUTPUT}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
