# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate the high-level policy mean on privileged waypoints.

Reports success, collision and timeout for each A* length. Each length is run
until ``--waypoints`` episodes finish (default 2000). With that count the 95%
half-width of a rate is at most ``1.96 * sqrt(0.25 / 2000)``, about 2.2
percentage points.

The policy is the Gaussian mean. Length ``k`` is forced by setting ``k_max = k``
and ``frontier_fraction = 1``, and the curriculum is turned off.

Run from f:\\isaacLab:

    .\\isaaclab.bat -p source\\task-go2-lidar-maze\\scripts\\eval_hl_policy.py --checkpoint PATH
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate the Go2 maze policy mean, per waypoint length.")
parser.add_argument("--checkpoint", type=str, default="", help="RSL-RL model_*.pt. Defaults to the newest run.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--waypoints", type=int, default=2000, help="Finished episodes required at each length.")
parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.visualizer = ["none"]
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_lidar_maze.tasks  # noqa: F401
import gymnasium as gym
import torch
from go2_lidar_maze.paths import project_root
from go2_lidar_maze.tasks.hl_maze.maze_env import Go2MazeEnv
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

TASK = "Nepher-Go2-LidarMaze-HL-v0"


def _load_cfg(task_name: str, entry_point_key: str):
    spec = gym.spec(task_name)
    entry = spec.kwargs[entry_point_key]
    module_name, attr_name = entry.split(":")
    module = importlib.import_module(module_name)
    cfg_cls = getattr(module, attr_name)
    return cfg_cls()


def _latest_checkpoint() -> str:
    root = project_root() / "logs" / "go2_lidar_maze_hl"
    models = [path for path in root.glob("*/model_*.pt") if path.is_file()]
    if not models:
        raise FileNotFoundError(f"No model_*.pt under {root}. Pass --checkpoint.")
    return str(max(models, key=lambda path: path.stat().st_mtime))


def _empty_counts() -> dict[str, int]:
    return {"n": 0, "success": 0, "collision": 0, "timeout": 0}


def _record(term, done: torch.Tensor, counts: dict[int, dict[str, int]]) -> None:
    ids = done.nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
        return
    lengths = term.last_k[ids]
    reached = term.last_reached[ids]
    collided = term.last_collision[ids]
    timed_out = term.last_timeout[ids]
    for index, length in enumerate(lengths.tolist()):
        slot = counts.get(int(length))
        if slot is None:
            continue
        slot["n"] += 1
        slot["success"] += int(float(reached[index]) > 0.0)
        slot["collision"] += int(bool(collided[index]))
        slot["timeout"] += int(bool(timed_out[index]))


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def main() -> None:
    checkpoint = args_cli.checkpoint or _latest_checkpoint()
    env_cfg = _load_cfg(TASK, "env_cfg_entry_point")
    env_cfg.sim.device = args_cli.device
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.curriculum.waypoint_distance = None
    env_cfg.commands.waypoint.frontier_fraction = 1.0
    env = Go2MazeEnv(cfg=env_cfg)
    agent_cfg = _load_cfg(TASK, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    term = env.unwrapped.command_manager.get_term("waypoint")
    # One episode is at most episode_length_s / step_dt steps. Allow twice that.
    step_cap = int(args_cli.waypoints / args_cli.num_envs * env.unwrapped.max_episode_length * 2) + 1

    print(f"Checkpoint: {checkpoint}")
    print(f"{'k':>3} {'n':>7} {'success':>9} {'collision':>10} {'timeout':>9}")
    for length in args_cli.lengths:
        term.k_max = int(length)
        counts = {int(length): _empty_counts()}
        # The whole episode stays in inference mode. Stepping inside it and
        # resetting outside marks the command buffers as inference tensors.
        with torch.inference_mode():
            obs = env.reset()[0]
            policy.reset(torch.ones(env.num_envs, dtype=torch.long, device=env.unwrapped.device))
            steps = 0
            while counts[int(length)]["n"] < args_cli.waypoints and steps < step_cap:
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                policy.reset(dones)
                _record(term, dones, counts)
                steps += 1
                slot = counts[int(length)]
                if steps % 50 == 0:
                    print(
                        f"{length:3d} {slot['n']:7d} {_rate(slot['success'], slot['n']):9.3f}"
                        f" {_rate(slot['collision'], slot['n']):10.3f} {_rate(slot['timeout'], slot['n']):9.3f}",
                        flush=True,
                    )
        slot = counts[int(length)]
        print(
            f"{length:3d} {slot['n']:7d} {_rate(slot['success'], slot['n']):9.3f}"
            f" {_rate(slot['collision'], slot['n']):10.3f} {_rate(slot['timeout'], slot['n']):9.3f}",
            flush=True,
        )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
