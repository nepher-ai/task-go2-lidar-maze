# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from go2_lidar_maze.paths import project_root


@configclass
class Go2VelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO settings for the flat Go2 velocity policy."""

    num_steps_per_env = 24
    # One curriculum widen per episode needs ~35 episodes to reach the final envelope.
    max_iterations = 4000
    save_interval = 50
    # Absolute path. Isaac Lab joins this onto ``logs/rsl_rl``; an absolute
    # component replaces that prefix, so runs land in the task project's logs.
    experiment_name = str(project_root() / "logs" / "go2_lidar_maze_ll")
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 0.01 kept the action std near 1.4, so yaw error never cleared the 0.8 gate.
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
