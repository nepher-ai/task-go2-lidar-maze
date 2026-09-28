# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Widen the velocity command ranges once tracking is reliable."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from go2_lidar_maze.constants import VX_MAX, VX_MIN, VY_MAX, VY_MIN, WZ_MAX, WZ_MIN


class widen_velocity_ranges(ManagerTermBase):
    """Move each command bound 10% of the way toward its final limit.

    The step is taken when the mean linear and yaw tracking kernels, averaged
    over the environments that are resetting, are both at least 0.8. At most
    one step is taken per episode, so the next decision uses episodes commanded
    inside the widened range. The kernel is recovered from the episodic reward
    sum, which Isaac Lab stores already multiplied by the term weight and by ``dt``.
    """

    def __init__(self, cfg: CurriculumTermCfg, env):
        super().__init__(cfg, env)
        self._final = {
            "lin_vel_x": (VX_MIN, VX_MAX),
            "lin_vel_y": (VY_MIN, VY_MAX),
            "ang_vel_z": (WZ_MIN, WZ_MAX),
        }
        # common_step_counter at which another widen is allowed.
        self._ready_step = 0

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        command_name: str,
        lin_term: str,
        ang_term: str,
        threshold: float = 0.8,
        step_fraction: float = 0.1,
    ) -> dict[str, float]:
        ranges = env.command_manager.get_term(command_name).cfg.ranges
        state = {
            "lin_vel_x_min": ranges.lin_vel_x[0],
            "lin_vel_x_max": ranges.lin_vel_x[1],
            "lin_vel_y_max": ranges.lin_vel_y[1],
            "ang_vel_z_max": ranges.ang_vel_z[1],
        }
        steps = env.episode_length_buf[env_ids].float()
        if float(steps.mean()) < 2.0:
            return state
        # Episodes finishing before this step were commanded on the previous range.
        if env.common_step_counter < self._ready_step:
            return state
        if not _kernel_ready(env, env_ids, steps, lin_term, threshold):
            return state
        if not _kernel_ready(env, env_ids, steps, ang_term, threshold):
            return state
        ranges.lin_vel_x = _widen(ranges.lin_vel_x, self._final["lin_vel_x"], step_fraction)
        ranges.lin_vel_y = _widen(ranges.lin_vel_y, self._final["lin_vel_y"], step_fraction)
        ranges.ang_vel_z = _widen(ranges.ang_vel_z, self._final["ang_vel_z"], step_fraction)
        self._ready_step = int(env.common_step_counter) + int(env.max_episode_length)
        state["lin_vel_x_min"] = ranges.lin_vel_x[0]
        state["lin_vel_x_max"] = ranges.lin_vel_x[1]
        state["lin_vel_y_max"] = ranges.lin_vel_y[1]
        state["ang_vel_z_max"] = ranges.ang_vel_z[1]
        return state


def _kernel_ready(env, env_ids, steps: torch.Tensor, term_name: str, threshold: float) -> bool:
    """True when the mean exponential tracking kernel is at least ``threshold``."""
    episodic = env.reward_manager._episode_sums[term_name][env_ids]
    weight = env.reward_manager.get_term_cfg(term_name).weight
    # episodic = sum(kernel * weight * dt), and sum(dt) ~= steps * step_dt.
    duration = steps * env.step_dt
    kernel = episodic / (weight * duration.clamp(min=1e-6))
    return float(kernel.mean()) >= threshold


def _widen(current: tuple[float, float], final: tuple[float, float], fraction: float) -> tuple[float, float]:
    lo = current[0] + fraction * (final[0] - current[0])
    hi = current[1] + fraction * (final[1] - current[1])
    return (lo, hi)
