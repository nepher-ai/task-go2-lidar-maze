# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flat-ground Go2 velocity tracking.

The observation is the 48-value proprioceptive layout the high-level action
term later remaps: linear velocity (3), angular velocity (3), projected
gravity (3), velocity command (3), joint position (12), joint velocity (12),
last action (12). There is no height scan.

Command ranges start narrow and widen, by 10% of the remaining gap, toward
vx [-2, 3] m/s, vy [-1, 1] m/s, yaw rate [-2, 2] rad/s.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp

from go2_lidar_maze.constants import half_reward_sigma_sq
from go2_lidar_maze.robots import GO2_CFG
from go2_lidar_maze.tasks.ll_velocity import mdp as ll_mdp

# exp(-e^2 / sigma^2) is 1/2 at these errors.
_LIN_SIGMA_SQ = half_reward_sigma_sq(0.35)
_YAW_SIGMA_SQ = half_reward_sigma_sq(0.30)

# At the half-reward error the tracking return is 1.5*0.5 + 0.75*0.5 = 1.125.
# Penalty weights below sum to 0.28125, which is 25% of that.
_TRACK_LIN_WEIGHT = 1.5
_TRACK_ANG_WEIGHT = 0.75


@configclass
class LlSceneCfg(InteractiveSceneCfg):
    """Flat plane, one Go2, and a contact sensor. No height scanner."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class LlCommandsCfg:
    """Velocity commands. Ranges are the curriculum start, not the final envelope."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class LlActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class LlObservationsCfg:
    """48-value proprioceptive observation. Field names are part of the contract."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class LlEventsCfg:
    """Friction, base mass, and velocity pushes."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (1.0 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class LlRewardsCfg:
    """Tracking kernels plus regularizers whose worst case is 25% of tracking at e_1/2."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=_TRACK_LIN_WEIGHT,
        params={"command_name": "base_velocity", "std": math.sqrt(_LIN_SIGMA_SQ)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=_TRACK_ANG_WEIGHT,
        params={"command_name": "base_velocity", "std": math.sqrt(_YAW_SIGMA_SQ)},
    )
    lin_vel_z = RewTerm(func=ll_mdp.lin_vel_z_normalized, weight=-0.04)
    ang_vel_xy = RewTerm(func=ll_mdp.ang_vel_xy_normalized, weight=-0.03)
    flat_orientation = RewTerm(func=ll_mdp.flat_orientation_normalized, weight=-0.03)
    torque = RewTerm(func=ll_mdp.joint_torque_normalized, weight=-0.08)
    joint_acc = RewTerm(func=ll_mdp.joint_acc_normalized, weight=-0.03)
    action_rate = RewTerm(func=ll_mdp.action_rate_normalized, weight=-0.05)
    feet_air_time = RewTerm(
        func=ll_mdp.feet_air_time,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.4,
        },
    )
    undesired_contacts = RewTerm(
        func=ll_mdp.undesired_contacts_capped,
        weight=-0.02125,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*thigh"), "threshold": 1.0},
    )


@configclass
class LlTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class LlCurriculumCfg:
    velocity_ranges = CurrTerm(
        func=ll_mdp.widen_velocity_ranges,
        params={
            "command_name": "base_velocity",
            "lin_term": "track_lin_vel_xy_exp",
            "ang_term": "track_ang_vel_z_exp",
            "threshold": 0.8,
            "step_fraction": 0.1,
        },
    )


@configclass
class Go2VelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Flat Go2 velocity tracking with the extended command envelope."""

    scene: LlSceneCfg = LlSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: LlObservationsCfg = LlObservationsCfg()
    actions: LlActionsCfg = LlActionsCfg()
    commands: LlCommandsCfg = LlCommandsCfg()
    rewards: LlRewardsCfg = LlRewardsCfg()
    terminations: LlTerminationsCfg = LlTerminationsCfg()
    events: LlEventsCfg = LlEventsCfg()
    curriculum: LlCurriculumCfg = LlCurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Go2VelocityEnvCfg_PLAY(Go2VelocityEnvCfg):
    """Small, noise-free playback of the velocity policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.curriculum.velocity_ranges = None
        # Playback uses the final envelope. Training reaches it through the curriculum.
        self.commands.base_velocity.ranges.lin_vel_x = (-2.0, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
