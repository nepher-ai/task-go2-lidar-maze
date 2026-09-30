# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym environment class for the Go2 maze task.

Kept separate from the config module so importing the config does not load
USD Python bindings before Kit starts.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch

from isaaclab.envs import ManagerBasedRLEnv

from go2_lidar_maze.constants import CELL_CENTER_ORIGIN_M, CELL_PITCH_M, GRID_SIZE
from go2_lidar_maze.tasks.hl_maze.benchmark import ground_plane_cover, load_benchmark
from go2_lidar_maze.tasks.hl_maze.episode_vis import attach_episode_vis
from go2_lidar_maze.tasks.hl_maze.film_recorder import MazeFilm, claim_video
from go2_lidar_maze.tasks.hl_maze.lidar_vis import attach_lidar_vis
from go2_lidar_maze.tasks.hl_maze.mdp.events import enlarge_physx_aggregate_pairs
from go2_lidar_maze.tasks.hl_maze.maze_env_cfg import Go2MazeEnvCfg

# PhysX logs this channel once per prim while building the stage-wide body view.
_TENSORS_CHANNEL = "omni.physx.tensors.plugin"


class Go2MazeEnv(ManagerBasedRLEnv):
    """Maze env that publishes the goal tensor before the first reset."""

    def __init__(self, cfg: Go2MazeEnvCfg, render_mode: str | None = None, **kwargs):
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        self.goal_xy = torch.zeros((num_envs, 2), device=device)
        self.goal_xy[:, 0] = cfg.goal_xy[0]
        self.goal_xy[:, 1] = cfg.goal_xy[1]
        # Geometry must exist before the command term and the first reset.
        # ``--video`` is claimed first so the stock 10 fps recorder is not built
        # and the viewport renders on the 50 Hz low-level tick.
        film_target = claim_video(cfg)
        self._apply_benchmark(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        if self._scenario_spawn_xy is not None:
            _fit_ground_to_tiles(self)
        attach_lidar_vis(self)
        attach_episode_vis(self)
        self._maze_film = MazeFilm(self, *film_target) if film_target is not None else None

    def close(self):
        film = getattr(self, "_maze_film", None)
        if film is not None:
            film.close()
            self._maze_film = None
        super().close()

    def _apply_benchmark(self, cfg: Go2MazeEnvCfg) -> None:
        """Publish per-env cell geometry. A scenario replaces the single training maze."""
        num_envs = cfg.scene.num_envs
        device = cfg.sim.device
        scenario_path = getattr(cfg, "scenario_path", "") or ""
        if not scenario_path:
            self.maze_cell_origin_xy = torch.full((num_envs, 2), CELL_CENTER_ORIGIN_M, device=device)
            self.maze_cell_pitch = torch.full((num_envs,), CELL_PITCH_M, device=device)
            self.maze_grid_n = torch.full((num_envs,), GRID_SIZE, dtype=torch.long, device=device)
            self.tile_origin_xy = torch.zeros((num_envs, 2), device=device)
            self.optimal_route_m = torch.full((num_envs,), float("nan"), device=device)
            self._scenario_spawn_xy = None
            self._scenario_spawn_yaw = None
            self._scenario_goal_xy = None
            return

        layout = load_benchmark(scenario_path).for_envs(num_envs)
        self.maze_cell_origin_xy = layout.cell_origin_xy.to(device)
        self.maze_cell_pitch = layout.cell_pitch.to(device)
        self.maze_grid_n = layout.grid_n.to(device)
        self.tile_origin_xy = layout.tile_origin_xy.to(device)
        self.optimal_route_m = layout.optimal_route_m.to(device)
        self._scenario_spawn_xy = layout.spawn_xy.to(device)
        self._scenario_spawn_yaw = layout.spawn_yaw.to(device)
        self._scenario_goal_xy = layout.goal_xy.to(device)
        self.goal_xy[:] = self._scenario_goal_xy

    def _init_sim(self):
        """Write PhysX GPU capacities as ``UInt`` immediately before physics play.

        That reset also builds the stage-wide rigid-body view. The scan warns once
        per joint, mesh, and xform, so its channel stays at error until reset returns.
        """
        sim_reset = self.sim.reset

        def _reset(soft: bool = False):
            enlarge_physx_aggregate_pairs(self, None)
            with _quiet_rigid_body_scan():
                sim_reset(soft)

        self.sim.reset = _reset  # type: ignore[method-assign]
        try:
            super()._init_sim()
        finally:
            self.sim.reset = sim_reset  # type: ignore[method-assign]


@contextmanager
def _quiet_rigid_body_scan():
    """Hold the tensors channel at error across the stage-wide rigid-body scan.

    The previous level is restored, including the case where the channel did not
    exist yet. A missing body after startup still warns.
    """
    import omni.log

    log = omni.log.get_log()
    result, level, behavior = log.get_channel_level(_TENSORS_CHANNEL)
    log.set_channel_level(_TENSORS_CHANNEL, omni.log.Level.ERROR, omni.log.SettingBehavior.OVERRIDE)
    try:
        yield
    finally:
        if result == 0:
            log.set_channel_level(_TENSORS_CHANNEL, level, behavior)
        else:
            log.set_channel_level(_TENSORS_CHANNEL, omni.log.Level.WARN, omni.log.SettingBehavior.INHERIT)


def _fit_ground_to_tiles(env: Go2MazeEnv) -> None:
    """Stretch the visual ground plane across every benchmark tile.

    The importer builds a 100 m plane centered on the origin. Tiles continue
    well past that, so the rest of the field is drawn on the sky. The plane
    stays on the origin: a translation here is scaled with the mesh and lands
    beside the mazes, which removes the white ground from the shot.
    """
    center_x, center_y, span_x, span_y = ground_plane_cover(
        env.tile_origin_xy.detach().cpu(),
        env.maze_grid_n.detach().cpu(),
        env.maze_cell_pitch.detach().cpu(),
    )
    size_x = 2.0 * max(abs(center_x - span_x * 0.5), abs(center_x + span_x * 0.5))
    size_y = 2.0 * max(abs(center_y - span_y * 0.5), abs(center_y + span_y * 0.5))
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/ground/terrain/Environment") if stage is not None else None
    if prim is None or not prim.IsValid():
        return
    # The bundled plane asset is 100 m across at unit scale. Same attribute the spawner writes.
    scale = prim.GetAttribute("xformOp:scale")
    if not scale:
        return
    scale.Set((size_x / 100.0, size_y / 100.0, 1.0))
    _rescale_ground_grid(stage, size_x, size_y)
    _sync_ground_scale(prim)


def _rescale_ground_grid(stage, size_x: float, size_y: float) -> None:
    """Keep the 1 m grid metric after the plane mesh is scaled."""
    from pxr import Gf, UsdGeom

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/ground/terrain/Environment/Geometry"))
    if not mesh.GetPrim().IsValid():
        return
    # The plane texture repeats every 5 m.
    half_u = size_x / 10.0
    half_v = size_y / 10.0
    primvar = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    if not primvar:
        return
    primvar.Set(
        [
            Gf.Vec2f(-half_u, -half_v),
            Gf.Vec2f(half_u, -half_v),
            Gf.Vec2f(half_u, half_v),
            Gf.Vec2f(-half_u, half_v),
        ]
    )


def _sync_ground_scale(prim) -> None:
    """Publish the scaled plane into Fabric. The renderer does not reread the USD scale."""
    import omni.usd
    import usdrt.Gf
    import usdrt.Sdf
    import usdrt.Usd
    import usdrt.hierarchy
    from pxr import Usd, UsdGeom

    local = UsdGeom.Xformable(prim).GetLocalTransformation(Usd.TimeCode.Default())
    if isinstance(local, tuple):
        local = local[0]
    rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId()
    )
    hierarchy.set_local_xform(usdrt.Sdf.Path(str(prim.GetPath())), usdrt.Gf.Matrix4d(local))
