# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn the Go2 on a maze cell. All robots share the one maze at the origin."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from go2_lidar_maze.constants import CELL_CENTER_ORIGIN_M, CELL_PITCH_M, GRID_SIZE, SPAWN_JITTER_M
from go2_lidar_maze.tasks.hl_maze.cell_pairs import CorridorGraph


def enlarge_physx_aggregate_pairs(env, env_ids=None) -> None:
    """Rewrite GPU pair buffers as ``UInt`` before the first physics play.

    ``PhysxCfg`` writes these through ``safe_set_attribute_on_usd_prim``, which
    creates signed ``Int`` attributes. PhysX only reads the ``UInt`` schema
    names (same path as ``ovphysx_manager``), so the raised capacities never
    took effect and 1024 shared-maze robots overflowed the default 2**25
    aggregate buffer.

    Called from :class:`~go2_lidar_maze.tasks.hl_maze.maze_env.Go2MazeEnv`
    immediately before ``sim.reset()``. Not registered as a ``prestartup``
    event: that mode is rejected when ``replicate_physics`` is True.
    """
    from pxr import Sdf

    del env_ids  # unused; kept so the helper can still be used as an EventTerm
    prim = env.sim.stage.GetPrimAtPath(env.sim.cfg.physics_prim_path)
    # Values match Go2MazeEnvCfg PhysxCfg. Aggregate capacities must clear 3.6e7.
    for name, value in (
        ("physxScene:gpuFoundLostAggregatePairsCapacity", 2**26),
        ("physxScene:gpuTotalAggregatePairsCapacity", 2**26),
        ("physxScene:gpuFoundLostPairsCapacity", 2**25),
        ("physxScene:gpuMaxRigidPatchCount", 10 * 2**15),
        ("physxScene:gpuCollisionStackSize", 2**28),
    ):
        existing = prim.GetAttribute(name)
        if existing.IsValid() and existing.GetTypeName() != Sdf.ValueTypeNames.UInt:
            prim.RemoveProperty(name)
        prim.CreateAttribute(name, Sdf.ValueTypeNames.UInt).Set(value)


def _env_count(env, env_ids: torch.Tensor | slice) -> int:
    """How many environments ``env_ids`` selects.

    A full ``env.reset()`` passes ``slice(None)``. A partial reset passes a tensor.
    """
    if isinstance(env_ids, slice):
        return len(range(env.num_envs)[env_ids])
    return int(env_ids.shape[0])


def reset_go2_in_maze(
    env,
    env_ids: torch.Tensor | slice,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Place robots on corridor centres, with ±0.4 m jitter, or on the fixed start.

    When ``sample_start_goal`` is set, each resetting environment draws a random
    start cell and a different random goal cell, and keeps the pair only when
    A* finds a path. The goal is that cell's centre [m]. The start is the
    centre plus the usual ±0.4 m, and the yaw is uniform in [-π, π).
    """
    robot = env.scene[asset_cfg.name]
    pose = robot.data.default_root_pose.torch[env_ids].clone()
    count = _env_count(env, env_ids)
    if env.cfg.sample_start_goal:
        start_xy, goal_xy = _corridor_graph(env).sample(count, robot.device)
        jitter = (torch.rand(count, 2, device=robot.device) - 0.5) * (2.0 * SPAWN_JITTER_M)
        xy = start_xy + jitter
        yaw = (torch.rand(count, device=robot.device) - 0.5) * (2.0 * torch.pi)
        env.goal_xy[env_ids] = goal_xy
    elif env.cfg.fixed_spawn:
        xy = torch.tensor(env.cfg.start_xy, device=robot.device, dtype=torch.float32).expand(count, 2)
        yaw = torch.full((count,), 1.5708, device=robot.device)
    else:
        index = torch.randint(0, GRID_SIZE * GRID_SIZE, (count,), device=robot.device)
        row = torch.div(index, GRID_SIZE, rounding_mode="floor")
        col = index - row * GRID_SIZE
        jitter = (torch.rand(count, 2, device=robot.device) - 0.5) * (2.0 * SPAWN_JITTER_M)
        xy = torch.stack(
            (
                CELL_CENTER_ORIGIN_M + CELL_PITCH_M * col,
                CELL_CENTER_ORIGIN_M + CELL_PITCH_M * row,
            ),
            dim=-1,
        )
        xy = xy + jitter
        yaw = (torch.rand(count, device=robot.device) - 0.5) * (2.0 * torch.pi)
    pose[:, 0] = xy[:, 0]
    pose[:, 1] = xy[:, 1]
    zeros = torch.zeros(count, device=robot.device)
    pose[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    velocity = torch.zeros(count, 6, device=robot.device)
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)


def reset_go2_from_scenario(
    env,
    env_ids: torch.Tensor | slice,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Place each robot on its scenario spawn. No jitter.

    The goal stays the tile's recorded goal, so the online planner anchors
    on the benchmark pair rather than a random cell.
    """
    robot = env.scene[asset_cfg.name]
    pose = robot.data.default_root_pose.torch[env_ids].clone()
    spawn_xy = env._scenario_spawn_xy[env_ids]
    yaw = env._scenario_spawn_yaw[env_ids]
    pose[:, 0] = spawn_xy[:, 0]
    pose[:, 1] = spawn_xy[:, 1]
    count = _env_count(env, env_ids)
    zeros = torch.zeros(count, device=robot.device)
    pose[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    velocity = torch.zeros(count, 6, device=robot.device)
    robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
    env.goal_xy[env_ids] = env._scenario_goal_xy[env_ids]


def _corridor_graph(env) -> CorridorGraph:
    """Adjacency measured once from the wall mesh and reused on later resets."""
    graph = getattr(env, "_corridor_graph", None)
    if graph is None:
        from go2_lidar_maze.tasks.hl_maze.maze_layout import adjacency_from_blocked, measure_blocked_pairs

        blocked = measure_blocked_pairs(env.cfg.commands.waypoint.maze_usd)
        graph = CorridorGraph(adjacency_from_blocked(blocked))
        env._corridor_graph = graph
    return graph
