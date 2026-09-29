# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run Nepher-Go2-LidarMaze-Planner-Play-v0 and print the outcome.

The stock play loop never exits. This script uses that gym id, loads the same
latest checkpoint, and scores one episode per environment.

Run from f:\\isaacLab:

    .\\isaaclab.bat -p source\\task-go2-lidar-maze\\scripts\\eval_planner_play.py --viz none --num_envs 64
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play online-planner maze episodes and report the outcome.")
parser.add_argument("--checkpoint", type=str, default="", help="RSL-RL model_*.pt. Defaults to the newest run.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel episodes. Each environment is scored once.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_lidar_maze.tasks  # noqa: F401
import gymnasium as gym
import torch
from go2_lidar_maze.paths import project_root
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner

TASK = "Nepher-Go2-LidarMaze-Planner-Play-v0"


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

    def sort_key(path):
        stem = path.stem
        digits = "".join(ch if ch.isdigit() else " " for ch in stem).split()
        iteration = int(digits[-1]) if digits else -1
        return (path.parent.name, iteration)

    return str(max(models, key=sort_key))


def _termination_names(base) -> list[str]:
    manager = base.termination_manager
    names = []
    for name in manager.active_terms:
        if bool(manager.get_term(name)[0]):
            names.append(name)
    return names


def _run_batch(env, policy, checkpoint: str) -> None:
    """Score one episode on every environment. Success is the maze goal."""
    base = env.unwrapped
    manager = base.termination_manager
    num_envs = base.num_envs
    dt = float(base.step_dt)
    cap = int(base.max_episode_length) + 5
    finished = torch.zeros(num_envs, dtype=torch.bool, device=base.device)
    success = torch.zeros(num_envs, dtype=torch.bool, device=base.device)
    term_counts = {name: 0 for name in manager.active_terms}

    print(f"TASK {TASK}", flush=True)
    print(f"CHECKPOINT {checkpoint}", flush=True)
    print(f"NUM_ENVS {num_envs} EPISODE_S {base.cfg.episode_length_s}", flush=True)

    with torch.inference_mode():
        obs = env.get_observations()
        for step in range(cap):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            newly = dones.to(device=finished.device, dtype=torch.bool) & ~finished
            if bool(newly.any()):
                for name in term_counts:
                    term_counts[name] += int((manager.get_term(name) & newly).sum())
                success[newly] = manager.get_term("maze_goal")[newly]
                finished |= newly
            if step % 100 == 0 or bool(finished.all()):
                done_n = int(finished.sum())
                rate = float(success.sum()) / num_envs
                print(
                    f"t={step * dt:6.1f} finished={done_n}/{num_envs} success_rate={rate:.3f}",
                    flush=True,
                )
            if bool(finished.all()):
                break
        else:
            print(f"NO_DONE finished={int(finished.sum())}/{num_envs}", flush=True)

    rate = float(success.sum()) / num_envs
    print(f"SUCCESS_RATE {rate:.4f} ({int(success.sum())}/{num_envs})", flush=True)
    for name, count in term_counts.items():
        print(f"TERM {name} {count}", flush=True)


def main() -> None:
    checkpoint = args_cli.checkpoint or _latest_checkpoint()
    env_cfg = _load_cfg(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    env = gym.make(TASK, cfg=env_cfg)
    agent_cfg = _load_cfg(TASK, "rsl_rl_cfg_entry_point")
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.num_envs > 1:
        _run_batch(env, policy, checkpoint)
        env.close()
        return

    base = env.unwrapped
    term = base.command_manager.get_term("waypoint")
    robot = base.scene["robot"]
    goal = base.goal_xy[0]
    dt = float(base.step_dt)
    cap = int(base.max_episode_length) + 5

    print(f"TASK {TASK}", flush=True)
    print(f"CHECKPOINT {checkpoint}", flush=True)
    print(f"GOAL {float(goal[0]):.3f} {float(goal[1]):.3f} EPISODE_S {base.cfg.episode_length_s}", flush=True)

    with torch.inference_mode():
        obs = env.get_observations()
        prev_xy = robot.data.root_pos_w.torch[0, :2].clone()
        travel = 0.0
        min_goal = float("inf")
        last_wp = term.waypoint_w[0].clone()
        issues = 0
        for step in range(cap):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
            xy = robot.data.root_pos_w.torch[0, :2]
            travel += float(torch.linalg.norm(xy - prev_xy))
            prev_xy = xy.clone()
            goal_dist = float(torch.linalg.norm(xy - goal))
            min_goal = min(min_goal, goal_dist)
            wp = term.waypoint_w[0]
            if float(torch.linalg.norm(wp - last_wp)) > 0.25:
                issues += 1
                last_wp = wp.clone()
            if step % 20 == 0 or bool(dones[0]):
                wp_dist = float(term.distance_to_waypoint()[0])
                print(
                    f"t={step * dt:6.1f} xy=({float(xy[0]):6.2f},{float(xy[1]):6.2f})"
                    f" goal={goal_dist:6.2f} wp=({float(wp[0]):6.2f},{float(wp[1]):6.2f})"
                    f" wp_d={wp_dist:5.2f} final={int(bool(term.is_final[0]))}"
                    f" issued={int(term.issued_count[0])} reached={int(term.reached_count[0])}"
                    f" travel={travel:6.1f}",
                    flush=True,
                )
            if bool(dones[0]):
                names = _termination_names(base)
                print(
                    f"DONE t={step * dt:.2f} terms={names} min_goal={min_goal:.2f}"
                    f" travel={travel:.1f} waypoint_changes={issues}"
                    f" reached={int(term.reached_count[0])} issued={int(term.issued_count[0])}",
                    flush=True,
                )
                break
        else:
            print(f"NO_DONE steps={cap} min_goal={min_goal:.2f} travel={travel:.1f}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
