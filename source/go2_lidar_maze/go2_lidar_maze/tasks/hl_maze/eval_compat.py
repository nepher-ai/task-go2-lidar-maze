# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation wrapper for the Go2 maze env.

Surfaces the state eval-nav reads:

  ``task_completed``  latched ``maze_goal`` termination
  ``task_failed``     collision or bad orientation
  ``get_raw_state``   batched root pose and velocity, one host sync per step
  ``_log_metadata``   optimal route length and tile origin for the scorer
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EvalCompatEnv:
    """Thin wrapper that surfaces Go2 maze state for eval-nav."""

    def __init__(self, env: ManagerBasedRLEnv):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def unwrapped(self):
        return self._env.unwrapped

    @property
    def task_completed(self) -> torch.Tensor:
        """True where the maze goal was reached before the in-step reset."""
        env = self._env.unwrapped
        try:
            return env.termination_manager.get_term("maze_goal").clone()
        except (AttributeError, KeyError, RuntimeError):
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    @property
    def task_failed(self) -> torch.Tensor:
        """True where the episode ended in a collision or a fall."""
        env = self._env.unwrapped
        failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        try:
            manager = env.termination_manager
            for name in ("collision", "bad_orientation"):
                try:
                    failure = failure | manager.get_term(name)
                except (KeyError, AttributeError):
                    pass
        except AttributeError:
            pass
        return failure

    def get_raw_state(self) -> dict[str, np.ndarray]:
        """Batched root state ``[N, 3]`` for eval-nav's kinematic derivation."""
        env = self._env.unwrapped
        robot = env.scene["robot"]
        return {
            "root_lin_vel_b": robot.data.root_lin_vel_b.torch.detach().cpu().numpy(),
            "root_ang_vel_b": robot.data.root_ang_vel_b.torch.detach().cpu().numpy(),
            "root_pos_w": robot.data.root_pos_w.torch.detach().cpu().numpy(),
        }

    def _log_metadata(self, env_idx: int | None = None) -> dict[str, Any]:
        """Optimal route length [m], control step [s], and tile origin [m]."""
        idx = 0 if env_idx is None else int(env_idx)
        env = self._env.unwrapped
        meta: dict[str, Any] = {"step_dt": float(getattr(env, "step_dt", 0.1))}
        origin = getattr(env, "tile_origin_xy", None)
        if origin is not None:
            xy = origin[idx].detach().cpu()
            meta["env_origin"] = [float(xy[0]), float(xy[1]), 0.0]
        route = getattr(env, "optimal_route_m", None)
        if route is not None:
            length = float(route[idx].detach().cpu())
            if math.isfinite(length):
                meta["path_length"] = length
        return meta

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def step(self, action):
        return self._env.step(action)

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        if hasattr(self._env, "render"):
            return self._env.render(*args, **kwargs)


def wrap_for_eval(env: ManagerBasedRLEnv) -> EvalCompatEnv:
    """Wrap a Go2 maze env so eval-nav can score it."""
    return EvalCompatEnv(env)
