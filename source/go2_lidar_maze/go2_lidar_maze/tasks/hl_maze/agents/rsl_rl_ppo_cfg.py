# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg

from go2_lidar_maze.paths import hl_policy_path, project_root


@configclass
class Go2MazePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """LSTM actor-critic. One waypoint can sit behind a corner the scan cannot see."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    # Absolute path, so Isaac Lab's ``logs/rsl_rl/<name>`` join keeps this directory.
    experiment_name = str(project_root() / "logs" / "go2_lidar_maze_hl")
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2MazePPORunnerCfg_PLAY(Go2MazePPORunnerCfg):
    """Play loads ``best_policy/best_policy.pt`` unless ``--checkpoint`` overrides it."""

    def __post_init__(self):
        super().__post_init__()
        self.load_checkpoint = str(hl_policy_path())
