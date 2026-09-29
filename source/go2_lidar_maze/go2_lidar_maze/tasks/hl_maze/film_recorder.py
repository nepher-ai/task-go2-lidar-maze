# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write the benchmark film while ``--video`` play is running.

``isaaclab.bat play`` records one frame per environment step. This task's
environment step is 0.1 s, so that file is 10 fps. The film instead grabs the
Kit camera on every low-level tick (``render_interval`` of 4 physics steps)
and encodes those frames at 50 fps.
"""

from __future__ import annotations

import math
from pathlib import Path

from go2_lidar_maze.tasks.hl_maze.film import (
    FPS,
    FRAME_DT_S,
    WALL_THICKNESS_M,
    FilmDirector,
    TileGeom,
    choose_tiles,
    fallback_fov,
)

_CALLBACK = "go2_maze_film"
_TILES_PER_FILM = 4


def claim_video(cfg) -> tuple[str, str] | None:
    """Take over ``--video`` on the benchmark play config.

    Returns ``(output_dir, filename_prefix)`` when this env should film, and
    clears the stock recorders so they do not also write a 10 fps clip.
    Rendering is set to the low-level period. Returns ``None`` when ``--video``
    was not passed or this is not the benchmark scene.
    """
    recorders = list(getattr(cfg, "video_recorders", None) or [])
    if not recorders or not getattr(cfg, "scenario_path", ""):
        return None
    recorder = recorders[0]
    output_dir = recorder.output_dir or "videos/play"
    prefix = getattr(recorder, "output_filename_prefix", None) or "maze_film"
    cfg.video_recorders = []
    cfg.sim.render_interval = max(1, int(round(FRAME_DT_S / cfg.sim.dt)))
    return str(output_dir), str(prefix)


class MazeFilm:
    """Capture the shot list from the Kit viewport and stream it to an mp4."""

    def __init__(self, env, output_dir: str, prefix: str):
        self._env = env
        self._path = Path(output_dir) / f"{prefix}.mp4"
        self._director: FilmDirector | None = None
        self._writer = None
        self._last_counter = -1
        self._capturing = False
        self._closed = False
        self._shot_index = -1
        self._skipped_blank = 0
        self._grabber = _FabricCamera()
        vfov, hfov = _read_fov()
        tiles, self._env_ids = _tiles_from_env(env)
        seed = int(getattr(env.cfg, "seed", None) or 0)
        tile_ids = choose_tiles(tiles, _TILES_PER_FILM, seed)
        self._director = FilmDirector(tiles, tile_ids, vfov, hfov)
        env.sim.add_render_callback(_CALLBACK, self._on_render)
        physics = getattr(env.sim, "physics_manager", None)
        if physics is not None and physics.handles_decimation():
            print(
                "[WARN] Maze film: this physics backend steps a whole environment step at once, "
                "so the clip cannot sample the 50 Hz gait.",
                flush=True,
            )
        seconds = self._director.total_frames / FPS
        print(
            f"[INFO] Maze film: {FPS} fps, {seconds:.0f} s, tiles {tile_ids} -> {self._path}",
            flush=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sim = getattr(self._env, "sim", None)
        if sim is not None and hasattr(sim, "remove_render_callback"):
            sim.remove_render_callback(_CALLBACK)
        self._close_writer()
        self._grabber.close()

    def _on_render(self, _event) -> None:
        if self._closed or self._capturing or self._director is None or self._director.finished:
            return
        interval = int(self._env.cfg.sim.render_interval)
        counter = int(getattr(self._env, "_sim_step_counter", 0))
        if counter < interval or counter % interval != 0 or counter == self._last_counter:
            return
        self._last_counter = counter
        self._capturing = True
        try:
            self._capture_frame()
        finally:
            self._capturing = False

    def _capture_frame(self) -> None:
        director = self._director
        assert director is not None
        viz = _capture_visualizer(self._env)
        if viz is None:
            return
        pose = director.pose(_robot_xy(self._env, self._env_ids))
        if pose is None:
            return
        viz.set_camera_view(pose.eye, pose.lookat)
        # Headless ``--video`` has no viewport. ``set_camera_view`` then only
        # authors a USD xform, and the Kit frame grabber keeps returning the
        # default perspective camera (one frozen view of the maze at the origin).
        frame = self._grabber.grab(pose.eye, pose.lookat, viz)
        if frame is None or getattr(frame, "size", 0) == 0:
            return
        if int(frame.max()) == 0 and self._writer is None and self._skipped_blank < 30:
            self._skipped_blank += 1
            return
        index, shot, _progress = director.shot_at(director.frame)
        if index != self._shot_index:
            self._shot_index = index
            tile = "" if shot.tile is None else f" tile {shot.tile}"
            print(f"[INFO] Maze film: {shot.kind}{tile} ({shot.duration_s:.0f} s)", flush=True)
        self._append(frame)
        director.advance()
        if director.finished:
            self._close_writer()
            self._grabber.close()
            print(f"[INFO] Maze film wrote {self._path}", flush=True)
            _release_play_loop(self._env)

    def _append(self, frame) -> None:
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            import imageio.v2 as imageio

            self._writer = imageio.get_writer(
                self._path,
                fps=FPS,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",
                macro_block_size=1,
            )
        if frame.shape[1] % 2 or frame.shape[0] % 2:
            frame = frame[: frame.shape[0] - frame.shape[0] % 2, : frame.shape[1] - frame.shape[1] % 2]
        self._writer.append_data(frame)

    def _close_writer(self) -> None:
        writer = self._writer
        self._writer = None
        if writer is not None:
            writer.close()


def _release_play_loop(env) -> None:
    """Stop ``play --video`` once the film file is closed.

    That loop exits when a recorder length is reached, then closes the env.
    Claiming ``--video`` removes those recorders, so the process would otherwise
    wait on Ctrl+C after the mp4 exists.
    """
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return
    cfg.video_recorders = [VideoRecorderCfg(video_length=0, step_offset=0)]


def _tiles_from_env(env) -> tuple[list[TileGeom], list[int]]:
    """Unique tiles in environment order, and the environment that stands on each."""
    count = min(int(env.num_envs), int(env.tile_origin_xy.shape[0]))
    origins = env.tile_origin_xy.detach().cpu()
    pitch = env.maze_cell_pitch.detach().cpu()
    grid_n = env.maze_grid_n.detach().cpu()
    tiles: list[TileGeom] = []
    env_ids: list[int] = []
    seen: set[tuple[float, float]] = set()
    for index in range(count):
        origin = (round(float(origins[index, 0]), 3), round(float(origins[index, 1]), 3))
        if origin in seen:
            continue
        seen.add(origin)
        footprint = float(grid_n[index]) * float(pitch[index]) + WALL_THICKNESS_M
        tiles.append(TileGeom(origin[0], origin[1], footprint))
        env_ids.append(index)
    if not tiles:
        raise RuntimeError("Maze film found no tiles to frame.")
    return tiles, env_ids


def _robot_xy(env, env_ids: list[int]) -> dict[int, tuple[float, float]]:
    pos = env.scene["robot"].data.root_pos_w.torch.detach().cpu()
    return {
        tile: (float(pos[env_id, 0]), float(pos[env_id, 1]))
        for tile, env_id in enumerate(env_ids)
        if env_id < pos.shape[0]
    }


class _FabricCamera:
    """RGB camera whose pose is written where the renderer actually reads it.

    A headless Kit visualizer never creates a viewport, so viewport camera
    commands do not move ``/OmniverseKit_Persp``. This camera is a stage prim
    updated through Fabric, then rendered with one Kit frame.
    """

    _PATH = "/World/MazeFilmCamera"

    def __init__(self) -> None:
        self._annotator = None
        self._product = None
        self._translate_op = None
        self._orient_op = None
        self._width = 1280
        self._height = 720

    def grab(self, eye, lookat, viz) -> object | None:
        import numpy as np
        import omni.kit.app

        from isaaclab.app.settings_manager import get_settings_manager

        self._ensure(viz)
        _write_camera_pose(self._PATH, self._translate_op, self._orient_op, eye, lookat)
        import contextlib

        if self._product is not None:
            with contextlib.suppress(Exception):
                self._product.resume()
        settings = get_settings_manager()
        play_flag = settings.get("/app/player/playSimulations")
        settings.set_bool("/app/player/playSimulations", False)
        try:
            omni.kit.app.get_app().update()
        finally:
            if play_flag is not None:
                settings.set_bool("/app/player/playSimulations", bool(play_flag))
        raw = self._annotator.get_data()
        if isinstance(raw, dict):
            raw = raw.get("data")
        if self._product is not None:
            with contextlib.suppress(Exception):
                self._product.pause()
        raw = np.asarray(raw)
        if raw.size == 0:
            return None
        if raw.ndim == 1:
            raw = raw.reshape(self._height, self._width, -1)
        return np.asarray(raw[:, :, :3], dtype=np.uint8)

    def close(self) -> None:
        import contextlib

        if self._annotator is not None:
            with contextlib.suppress(Exception):
                self._annotator.detach()
        if self._product is not None:
            with contextlib.suppress(Exception):
                self._product.destroy()
        self._annotator = None
        self._product = None

    def _ensure(self, viz) -> None:
        if self._annotator is not None:
            return
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom

        cfg = getattr(viz, "cfg", None)
        self._width = int(getattr(cfg, "window_width", 1280) or 1280)
        self._height = int(getattr(cfg, "window_height", 720) or 720)
        stage = omni.usd.get_context().get_stage()
        camera = UsdGeom.Camera.Define(stage, self._PATH)
        xform = UsdGeom.Xformable(camera.GetPrim())
        xform.ClearXformOpOrder()
        # Eyes are world-space. Drop any parent transform so the authored pose is the world pose.
        xform.SetResetXformStack(True)
        prim = camera.GetPrim()
        translate_attr = prim.GetAttribute("xformOp:translate")
        self._translate_op = (
            UsdGeom.XformOp(translate_attr)
            if translate_attr
            else xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
        )
        orient_attr = prim.GetAttribute("xformOp:orient")
        self._orient_op = (
            UsdGeom.XformOp(orient_attr)
            if orient_attr
            else xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        )
        xform.SetXformOpOrder([self._translate_op, self._orient_op], True)
        _copy_lens(stage, camera)
        # Publish the prim to Fabric before the render product binds to it.
        from isaaclab.app.settings_manager import get_settings_manager

        settings = get_settings_manager()
        play_flag = settings.get("/app/player/playSimulations")
        settings.set_bool("/app/player/playSimulations", False)
        try:
            omni.kit.app.get_app().update()
        finally:
            if play_flag is not None:
                settings.set_bool("/app/player/playSimulations", bool(play_flag))
        self._product = rep.create.render_product(self._PATH, (self._width, self._height))
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._annotator.attach([self._product])


def _copy_lens(stage, camera) -> None:
    """Match the perspective camera's field of view, which the shot math uses."""
    from pxr import UsdGeom

    source_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
    if not source_prim or not source_prim.IsValid():
        return
    source = UsdGeom.Camera(source_prim)
    for getter, setter in (
        (source.GetFocalLengthAttr, camera.GetFocalLengthAttr),
        (source.GetHorizontalApertureAttr, camera.GetHorizontalApertureAttr),
        (source.GetVerticalApertureAttr, camera.GetVerticalApertureAttr),
        (source.GetClippingRangeAttr, camera.GetClippingRangeAttr),
    ):
        value = getter().Get()
        if value is not None:
            setter().Set(value)


def _write_camera_pose(prim_path: str, translate_op, orient_op, eye, lookat) -> None:
    """Author a world-space look-at on the film camera.

    Replicator's pose helper crashes here: with Fabric scene delegate off it
    builds a USD transform and then feeds it a Fabric vector. The pose is
    written as USD xform ops, then copied into Fabric so a scene delegate that
    ignores later USD edits still moves.
    """
    import torch
    import usdrt.Gf
    import usdrt.Sdf
    import usdrt.Usd
    import usdrt.hierarchy
    from pxr import Gf, UsdGeom

    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    eye_t = torch.tensor([[float(eye[0]), float(eye[1]), float(eye[2])]], dtype=torch.float32)
    look_t = torch.tensor([[float(lookat[0]), float(lookat[1]), float(lookat[2])]], dtype=torch.float32)
    up_axis = UsdGeom.GetStageUpAxis(stage) if stage is not None else "Z"
    rotation = create_rotation_matrix_from_view(eye_t, look_t, up_axis=up_axis, device="cpu")
    if torch.isnan(rotation).any():
        raise ValueError("Maze film camera eye and look-at are degenerate.")
    quat_xyzw = quat_from_matrix(rotation)[0]
    quat = Gf.Quatd(
        float(quat_xyzw[3]),
        Gf.Vec3d(float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])),
    )
    translation = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
    translate_op.Set(translation)
    orient_op.Set(quat)

    transform = Gf.Transform()
    transform.SetTranslation(translation)
    transform.SetRotation(Gf.Rotation(quat))
    matrix = transform.GetMatrix()
    fabric_matrix = usdrt.Gf.Matrix4d(*[float(matrix[row][col]) for row in range(4) for col in range(4)])
    rt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    hierarchy = usdrt.hierarchy.IFabricHierarchy().get_fabric_hierarchy(
        rt_stage.GetFabricId(), rt_stage.GetStageIdAsStageId()
    )
    hierarchy.set_local_xform(usdrt.Sdf.Path(prim_path), fabric_matrix)


def _capture_visualizer(env):
    visualizers = getattr(env.sim, "visualizers", []) or []
    for viz in visualizers:
        if getattr(getattr(viz, "cfg", None), "visualizer_type", None) == "kit":
            return viz
    for viz in visualizers:
        if hasattr(viz, "render_rgb_array") and hasattr(viz, "set_camera_view"):
            return viz
    return None


def _read_fov() -> tuple[float, float]:
    """Vertical and horizontal field of view [rad] from the Kit perspective camera."""
    try:
        import omni.usd
        from pxr import UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/OmniverseKit_Persp") if stage is not None else None
        if prim is None or not prim.IsValid():
            return fallback_fov()
        camera = UsdGeom.Camera(prim)
        focal = float(camera.GetFocalLengthAttr().Get() or 0.0)
        haperture = float(camera.GetHorizontalApertureAttr().Get() or 0.0)
        if focal <= 0.0 or haperture <= 0.0:
            return fallback_fov()
        vaperture = camera.GetVerticalApertureAttr().Get()
        vaperture = float(vaperture) if vaperture else haperture / (16.0 / 9.0)
        vfov = 2.0 * math.atan(vaperture / (2.0 * focal))
        hfov = 2.0 * math.atan(haperture / (2.0 * focal))
        return vfov, hfov
    except Exception:
        return fallback_fov()
