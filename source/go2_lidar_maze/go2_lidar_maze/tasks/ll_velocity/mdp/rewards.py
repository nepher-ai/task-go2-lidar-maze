# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Normalized regularizers for the Go2 velocity task.

Each term is divided by its physical maximum so that it lies in ``[0, 1]``.
The environment config then keeps the sum of their weights inside 25% of the
tracking reward evaluated at the half-reward error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import rewards as mdp_rewards
from isaaclab.envs.mdp.rewards import undesired_contacts as _undesired_contacts
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

# Hip and thigh effort limit is 23.7 Nm. The plan's 23.5 Nm is that rating.
# The calf joint limit is higher (45.43 Nm); the result is clamped to 1.
_TAU_MAX_NM = 23.5
_N_JOINTS = 12
# Joint-position actions are in [-1, 1], so a one-step change is at most 2.
_ACTION_DELTA_MAX = 2.0


def joint_torque_normalized(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum of squared joint torques divided by ``12 * tau_max^2``, clamped to 1."""
    asset = env.scene[asset_cfg.name]
    torques = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    raw = torch.sum(torch.square(torques), dim=1) / (_N_JOINTS * _TAU_MAX_NM**2)
    return raw.clamp(max=1.0)


def action_rate_normalized(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared action change divided by its maximum over 12 joints, clamped to 1."""
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    raw = torch.sum(torch.square(action - prev), dim=1) / (_N_JOINTS * _ACTION_DELTA_MAX**2)
    return raw.clamp(max=1.0)


def lin_vel_z_normalized(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Vertical speed squared, clamped at 1 (m/s)^2."""
    return mdp_rewards.lin_vel_z_l2(env, asset_cfg).clamp(max=1.0)


def ang_vel_xy_normalized(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Roll and pitch rate squared, clamped at 1 (rad/s)^2."""
    return mdp_rewards.ang_vel_xy_l2(env, asset_cfg).clamp(max=1.0)


def flat_orientation_normalized(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gravity xy components squared. The sum is at most 1."""
    return mdp_rewards.flat_orientation_l2(env, asset_cfg).clamp(max=1.0)


def joint_acc_normalized(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint acceleration squared, divided by ``12 * (100 rad/s^2)^2`` and clamped to 1."""
    raw = mdp_rewards.joint_acc_l2(env, asset_cfg) / (_N_JOINTS * 100.0**2)
    return raw.clamp(max=1.0)


def undesired_contacts_capped(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """1 when any listed body is in contact, so the weight is the worst-case cost."""
    return _undesired_contacts(env, threshold, sensor_cfg).clamp(max=1.0)


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward feet that stay in the air longer than ``threshold`` [s] while a command is active."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    moving = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward * moving
