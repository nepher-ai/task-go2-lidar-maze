# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg

from go2_lidar_maze.paths import hl_policy_path, project_root


@configclass
class _CappedGaussianCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Log-std Gaussian clamped to ``[0.05, 0.4]``.

    ``rsl_rl`` forwards every field except ``class_name`` to
    ``GaussianDistribution``. ``std_range`` is that constructor's clamp, in the
    same units as ``init_std``.
    """

    std_type: Literal["scalar", "log"] = "log"
    std_range: tuple[float, float] = (0.05, 0.4)


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
        obs_normalization=True,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        distribution_cfg=_CappedGaussianCfg(init_std=0.3),
    )
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=False,
        clip_param=0.2,
        entropy_coef=0.0,
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
class Go2MazePPORunnerCfg_ENVHUB_PLAY(Go2MazePPORunnerCfg):
    """Benchmark play loads the shipped checkpoint, not the newest training log."""

    load_checkpoint = str(hl_policy_path())
