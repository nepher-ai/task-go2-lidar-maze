# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 maze navigation. One USD maze is shared by every environment."""

from __future__ import annotations

from pathlib import Path

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
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
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import patterns_cfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from go2_lidar_maze.constants import collision_penalty
from go2_lidar_maze.paths import ll_policy_path
from go2_lidar_maze.robots import GO2_CFG
from go2_lidar_maze.tasks.hl_maze import mdp as hl_mdp
from go2_lidar_maze.tasks.hl_maze.mdp.ll_policy_action import LlPolicyActionCfg
from go2_lidar_maze.tasks.hl_maze.mdp.waypoint_command import WaypointCommandCfg
from go2_lidar_maze.tasks.hl_maze.scenario import load_start_goal
from go2_lidar_maze.tasks.ll_velocity.velocity_env_cfg import LlObservationsCfg

_PACKAGE = Path(__file__).resolve().parents[2]
_MAZE_USD = str(_PACKAGE / "assets" / "maze" / "enclosed_20x20.usd")
_MAZE_TXT = _PACKAGE / "assets" / "maze" / "enclosed_20x20.txt"
_POLICY = str(ll_policy_path())


@configclass
class MazeSceneCfg(InteractiveSceneCfg):
    """One maze mesh at the world origin. Robots overlap in space and do not collide."""

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
    maze = AssetBaseCfg(
        prim_path="/World/Maze",
        spawn=sim_utils.UsdFileCfg(usd_path=_MAZE_USD),
    )
    robot = GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    lidar = MultiMeshRayCasterCfg(
        # Base frame, +X forward. The head joint is at x = 0.285 m and the shell
        # tops out at z = 0.089 m. This sits past the trunk, above the head.
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.36, 0.0, 0.23)),
        # Refresh whenever the scan is read. A 0.1 s period equals one control step, but the
        # viewport draws before that period elapses, and float32 time skips the refresh on
        # some steps. The drawn scan then alternates between the current pose and the last one.
        update_period=0.0,
        max_distance=10.0,
        # "base" rotates the ring with the full link pose.
        ray_alignment="base",
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/Maze",
                track_mesh_transforms=False,
                is_shared=True,
            ),
            # Static plane. Without it a downward beam is drawn out to max_distance through the floor.
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground/terrain",
                track_mesh_transforms=False,
                is_shared=True,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Robot/base/visuals",
                track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Robot/[FR][LR]_(hip|thigh|calf|foot)/visuals",
                track_mesh_transforms=True,
            ),
        ],
        pattern_cfg=patterns_cfg.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=2.0,
        ),
        debug_vis=False,
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["/World/Maze/Walls"],
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


def _ll_observations() -> LlObservationsCfg.PolicyCfg:
    """Proprioception for the frozen velocity policy, without observation noise."""
    group = LlObservationsCfg.PolicyCfg()
    group.enable_corruption = False
    return group


@configclass
class HlActionsCfg:
    drive: LlPolicyActionCfg = LlPolicyActionCfg(
        asset_name="robot",
        policy_path=_POLICY,
        low_level_decimation=4,
        low_level_observations=_ll_observations(),
        max_lin_accel=3.0,
        max_ang_accel=6.0,
    )


@configclass
class HlCommandsCfg:
    waypoint: WaypointCommandCfg = WaypointCommandCfg(
        asset_name="robot",
        source="privileged",
        maze_usd=_MAZE_USD,
    )


@configclass
class HlObservationsCfg:
    """199 actor values. The critic adds the remaining path length."""

    @configclass
    class PolicyCfg(ObsGroup):
        lidar = ObsTerm(
            func=hl_mdp.observations.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "min_range": 0.1, "max_range": 10.0, "corrupt": True},
        )
        waypoint = ObsTerm(func=hl_mdp.observations.waypoint_b, params={"command_name": "waypoint"})
        lookahead = ObsTerm(func=hl_mdp.observations.path_lookahead, params={"command_name": "waypoint"})
        lin_vel = ObsTerm(func=hl_mdp.observations.base_lin_vel_xy, params={"asset_cfg": SceneEntityCfg("robot")})
        yaw_rate = ObsTerm(func=hl_mdp.observations.base_yaw_rate, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        lidar = ObsTerm(
            func=hl_mdp.observations.lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "min_range": 0.1, "max_range": 10.0, "corrupt": False},
        )
        waypoint = ObsTerm(func=hl_mdp.observations.waypoint_b, params={"command_name": "waypoint"})
        lookahead = ObsTerm(func=hl_mdp.observations.path_lookahead, params={"command_name": "waypoint"})
        lin_vel = ObsTerm(func=hl_mdp.observations.base_lin_vel_xy, params={"asset_cfg": SceneEntityCfg("robot")})
        yaw_rate = ObsTerm(func=hl_mdp.observations.base_yaw_rate, params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=mdp.last_action)
        remaining = ObsTerm(func=hl_mdp.observations.remaining_path, params={"command_name": "waypoint"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HlEventsCfg:
    # PhysX GPU UInt buffers are written from Go2MazeEnv._init_sim (not a
    # prestartup event): prestartup is blocked when replicate_physics is True.
    reset_robot = EventTerm(func=hl_mdp.events.reset_go2_in_maze, mode="reset")
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )


@configclass
class HlRewardsCfg:
    """Per-step terms. Weights are ±1 because each function already applies its scale.

    ``failure`` is the exception: its weight is ``-C``, with ``C`` derived in
    :func:`go2_lidar_maze.constants.collision_penalty`.
    """

    progress = RewTerm(func=hl_mdp.rewards.progress, weight=1.0, params={"command_name": "waypoint"})
    time_cost = RewTerm(func=hl_mdp.rewards.time_cost, weight=-1.0)
    lateral = RewTerm(func=hl_mdp.rewards.lateral, weight=-1.0, params={"command_name": "waypoint"})
    clearance = RewTerm(
        func=hl_mdp.rewards.clearance,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("lidar")},
    )
    braking = RewTerm(
        func=hl_mdp.rewards.braking,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "asset_cfg": SceneEntityCfg("robot")},
    )
    action_rate = RewTerm(func=hl_mdp.rewards.action_rate, weight=-1.0)
    waypoint_bonus = RewTerm(func=hl_mdp.rewards.waypoint_bonus, weight=1.0, params={"command_name": "waypoint"})
    # Replaced in ``Go2MazeEnvCfg.__post_init__`` by ``-C``.
    failure = RewTerm(func=hl_mdp.rewards.failure, weight=-1.0)


@configclass
class HlTerminationsCfg:
    maze_goal = DoneTerm(
        func=hl_mdp.terminations.maze_goal_reached,
        params={"command_name": "waypoint"},
        time_out=True,
    )
    waypoint_timeout = DoneTerm(
        func=hl_mdp.terminations.waypoint_timed_out,
        params={"command_name": "waypoint"},
        time_out=True,
    )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    waypoint_reached = DoneTerm(
        func=hl_mdp.terminations.waypoint_reached,
        params={"command_name": "waypoint"},
        time_out=False,
    )
    collision = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*hip", ".*thigh", ".*calf"]),
            "threshold": 1.0,
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class HlCurriculumCfg:
    waypoint_distance = CurrTerm(
        func=hl_mdp.curriculum.raise_waypoint_distance,
        params={
            "command_name": "waypoint",
            "threshold": 0.8,
            "min_issued": 512.0,
            "min_episodes": 256.0,
            "max_collision": 0.5,
            "passes_required": 2,
        },
    )


@configclass
class Go2MazeEnvCfg(ManagerBasedRLEnvCfg):
    """Train the high-level policy on privileged waypoints of 1 to 5 cells."""

    scene: MazeSceneCfg = MazeSceneCfg(num_envs=1024, env_spacing=0.0)
    observations: HlObservationsCfg = HlObservationsCfg()
    actions: HlActionsCfg = HlActionsCfg()
    commands: HlCommandsCfg = HlCommandsCfg()
    events: HlEventsCfg = HlEventsCfg()
    rewards: HlRewardsCfg = HlRewardsCfg()
    terminations: HlTerminationsCfg = HlTerminationsCfg()
    curriculum: HlCurriculumCfg = HlCurriculumCfg()

    fixed_spawn: bool = False
    sample_start_goal: bool = False
    scenario_path: str = ""
    """EnvHub ``scenario.json``. Empty keeps the single training maze."""
    """Draw a new start cell and goal cell for each resetting environment."""
    start_xy: tuple[float, float] = (0.0, 0.0)
    goal_xy: tuple[float, float] = (0.0, 0.0)
    # Every robot, and only while a window is open. Each one is independent:
    # ``env.vis_lidar_beams=true env.vis_lidar_hits=true env.vis_start_goal=true``.
    vis_lidar_beams: bool = False
    vis_lidar_hits: bool = False
    vis_start_goal: bool = False

    def __post_init__(self):
        # C = 1.25 * p_max / (1 - gamma). Recomputed here so it tracks the weights.
        self.rewards.failure.weight = -collision_penalty()
        self.decimation = 20
        # The per-waypoint budget is clip(L / 0.5 + 5, 10, 30) s. A 10 s cap
        # would end a 4- or 5-cell waypoint before that budget.
        self.episode_length_s = 30.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics = PhysxAutoCfg(
            isaacsim_physx=PhysxCfg(
                gpu_max_rigid_patch_count=10 * 2**15,
                gpu_collision_stack_size=2**28,
                gpu_found_lost_pairs_capacity=2**25,
                # 1024 robots share one maze. PhysX asked for 36150475 aggregate pairs,
                # which is past 2**25.
                gpu_total_aggregate_pairs_capacity=2**26,
                gpu_found_lost_aggregate_pairs_capacity=2**26,
            )
        )
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class Go2MazeEnvCfg_PLAY(Go2MazeEnvCfg):
    """Training task with one environment. Privileged waypoints in the shared maze."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1


@configclass
class Go2MazeEnvCfg_PLANNER(Go2MazeEnvCfg):
    """Online LiDAR planner. Each reset draws a reachable start and goal per robot."""

    def __post_init__(self):
        super().__post_init__()
        start_x, start_y, end_x, end_y = load_start_goal(_MAZE_TXT)
        self.scene.num_envs = 1
        self.fixed_spawn = False
        self.sample_start_goal = True
        # Placeholder until the first reset draws a pair per environment.
        self.start_xy = (start_x, start_y)
        self.goal_xy = (end_x, end_y)
        self.commands.waypoint.source = "online"
        self.curriculum.waypoint_distance = None
        self.episode_length_s = 180.0


@configclass
class Go2MazeEnvCfg_PLANNER_Envhub_PLAY(Go2MazeEnvCfg_PLANNER):
    """Online planner on the ``go2-maze-v0`` benchmark. One robot per maze tile.

    ``env_id`` and ``scene_id`` are injected by
    ``eval_nav.managers.EnvironmentManager.build_env_cfg``.
    """

    env_id: str = "go2-maze-v0"
    scene_id: str | int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        self._load_envhub()
        self.scene.num_envs = 60
        self.sample_start_goal = False
        self.fixed_spawn = False
        self.episode_length_s = 300.0
        self.events.reset_robot = EventTerm(func=hl_mdp.events.reset_go2_from_scenario, mode="reset")
        # A window draws each tile's start, its maze goal, and the current target waypoint.
        self.vis_start_goal = True
        # Planner play does not read the wall-filtered contact matrix.
        self.scene.contact_forces.filter_prim_paths_expr = []
        # A one-frame graze near 1 N differs across PhysX builds. Require 10 N
        # on every stored contact frame before the benchmark tile fails.
        self.sim.physics.isaacsim_physx.enable_enhanced_determinism = True
        self.terminations.collision.func = hl_mdp.terminations.sustained_illegal_contact
        self.terminations.collision.params["threshold"] = 10.0

    def _load_envhub(self) -> None:
        """Point the shared maze asset at the assembled benchmark world."""
        if not self.env_id:
            return

        from nepher import load_env

        env = load_env(self.env_id, category="navigation")
        bundle_path = env.cache_path

        scene_file = bundle_path / "maze_scene.py"
        if scene_file.is_file():
            import runpy

            namespace = runpy.run_path(str(scene_file))
            namespace["ensure_generated"](bundle_path)

        world_usd = bundle_path / "maps" / "benchmark_world.usd"
        scenario = bundle_path / "maps" / "scenario.json"
        if not world_usd.is_file() or not scenario.is_file():
            raise FileNotFoundError(
                f"go2-maze-v0 bundle at '{bundle_path}' is missing "
                "maps/benchmark_world.usd or maps/scenario.json."
            )
        self.scene.maze.spawn.usd_path = str(world_usd)
        self.scenario_path = str(scenario)
