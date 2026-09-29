# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level velocity action executed by the frozen Go2 velocity policy.

The high-level policy outputs ``(ax, ay, az)`` in [-1, 1]. Those values are
mapped onto vx [-2, 3] m/s, vy [-1, 1] m/s and yaw rate [-2, 2] rad/s, then
rate-limited. The resulting command is what the frozen policy observes as
``velocity_commands``. The frozen policy itself runs every 4 physics steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ActionTerm, ObservationManager
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ActionTermCfg, ObservationGroupCfg

from go2_lidar_maze.velocity_map import map_action_to_velocity, rate_limit

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class LlPolicyAction(ActionTerm):
    """Run a TorchScript velocity policy at 50 Hz from a 10 Hz command."""

    cfg: LlPolicyActionCfg

    def __init__(self, cfg: LlPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(
                f"Frozen Go2 velocity policy not found at '{cfg.policy_path}'. "
                "Train Nepher-Go2-LidarMaze-LL-v0 and run scripts/export_ll_policy.py. "
                "That writes best_policy/ll_policy/ll_policy.pt."
            )
        self.policy = torch.jit.load(read_file(cfg.policy_path)).to(env.device).eval()
        for param in self.policy.parameters():
            param.requires_grad_(False)
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)

        def last_action():
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.velocity_commands.params = dict()
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        self._counter = 0
        hl_dt = env.step_dt
        self._max_delta = torch.tensor(
            [cfg.max_lin_accel * hl_dt, cfg.max_lin_accel * hl_dt, cfg.max_ang_accel * hl_dt],
            device=self.device,
        )

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor):
        """Map, clip, and rate-limit the high-level action. Called once per env step."""
        self._prev_cmd[self._env.episode_length_buf == 0] = 0.0
        target = map_action_to_velocity(actions)
        self._raw_actions[:] = rate_limit(target, self._prev_cmd, self._max_delta)
        self._prev_cmd[:] = self._raw_actions

    def apply_actions(self):
        """Step the frozen policy every ``low_level_decimation`` physics steps."""
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            # Eval-nav steps with autograd on. The frozen net must not return a
            # grad tensor; Warp rejects those joint targets. no_grad (not
            # inference_mode) so the result can be copied into the action buffer.
            with torch.no_grad():
                predicted = self.policy(low_level_obs)
            self.low_level_actions[:] = predicted
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1


@configclass
class LlPolicyActionCfg(ActionTermCfg):
    """Configuration for :class:`LlPolicyAction`."""

    class_type: type = LlPolicyAction
    asset_name: str = "robot"
    policy_path: str = ""
    """Path to the frozen TorchScript velocity policy."""
    low_level_decimation: int = 4
    low_level_actions: JointPositionActionCfg | ActionTermCfg = JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )
    low_level_observations: ObservationGroupCfg | None = None
    max_lin_accel: float = 3.0
    """Linear acceleration limit [m/s^2]."""
    max_ang_accel: float = 6.0
    """Yaw acceleration limit [rad/s^2]."""
