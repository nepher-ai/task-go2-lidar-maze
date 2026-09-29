# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera path for a 50 fps film of the go2-maze-v0 benchmark.

The high-level policy still steps at 0.1 s. Frames are taken on the low-level
period (0.02 s) and played back at 50 fps, so one second of video is one
second of robot time. The lens looks 10 degrees off vertical.

No simulator imports. The env hooks this clock to the Kit camera while
``--video`` is set.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Corridor walls in go2-maze-v0. Footprint is n * cell_pitch + this thickness.
WALL_THICKNESS_M = 0.5

TILT_RAD = math.radians(10.0)
FPS = 50
FRAME_DT_S = 1.0 / FPS

OVERVIEW_S = 4.0
FILL_S = 4.0
FOLLOW_S = 20.0
# Travel between the overview and a maze, and from one maze to the next.
MOVE_S = 2.0

# Follow sits a little closer than the shot that frames one maze top to bottom.
FOLLOW_HEIGHT_SCALE = 0.8
OVERVIEW_MARGIN = 1.08
FILL_MARGIN = 1.02
# Follow lookat lags the robot so the frame does not chatter.
FOLLOW_LAG_S = 0.30

# Isaac Sim's default 35 mm-equivalent perspective camera, 16:9 gate.
# Used only when the live camera prim cannot be read.
_FALLBACK_FOCAL_MM = 18.147562
_FALLBACK_HAPERTURE_MM = 20.955
_FALLBACK_ASPECT = 16.0 / 9.0


@dataclass(frozen=True)
class TileGeom:
    """One maze tile in the world frame."""

    origin_x: float
    origin_y: float
    footprint: float
    """Outer size of the maze [m]."""

    @property
    def center(self) -> tuple[float, float]:
        half = self.footprint * 0.5
        return (self.origin_x + half, self.origin_y + half)


@dataclass(frozen=True)
class Pose:
    """World-frame camera eye and look-at [m]."""

    eye: tuple[float, float, float]
    lookat: tuple[float, float, float]


@dataclass(frozen=True)
class Shot:
    """One piece of the shot list."""

    kind: str
    duration_s: float
    tile: int | None


def fallback_fov() -> tuple[float, float]:
    """Vertical and horizontal field of view [rad] of the default Kit camera."""
    vaperture = _FALLBACK_HAPERTURE_MM / _FALLBACK_ASPECT
    vfov = 2.0 * math.atan(vaperture / (2.0 * _FALLBACK_FOCAL_MM))
    hfov = 2.0 * math.atan(_FALLBACK_HAPERTURE_MM / (2.0 * _FALLBACK_FOCAL_MM))
    return vfov, hfov


def nadir_pose(look_x: float, look_y: float, height: float, tilt_rad: float = TILT_RAD) -> Pose:
    """Camera above ``(look_x, look_y)``, tilted ``tilt_rad`` off vertical toward +Y.

    The eye sits on the -Y side of the look-at point. Height is the eye's
    world Z [m], not the slant range.
    """
    horiz = height * math.tan(tilt_rad)
    return Pose(eye=(look_x, look_y - horiz, height), lookat=(look_x, look_y, 0.0))


def height_for_vertical_span(span_m: float, vfov_rad: float, tilt_rad: float = TILT_RAD) -> float:
    """Eye height [m] whose vertical field covers ``span_m`` of ground along +Y."""
    half = vfov_rad * 0.5
    near = tilt_rad - half
    far = tilt_rad + half
    if near <= -math.pi / 2 or far >= math.pi / 2:
        raise ValueError("vertical field of view crosses the horizon.")
    denom = math.tan(far) - math.tan(near)
    if denom <= 0.0:
        raise ValueError("vertical field of view does not meet the ground.")
    return span_m / denom


def height_for_horizontal_span(span_m: float, hfov_rad: float, tilt_rad: float = TILT_RAD) -> float:
    """Eye height [m] whose horizontal field covers ``span_m`` of ground along X."""
    return span_m * math.cos(tilt_rad) / (2.0 * math.tan(hfov_rad * 0.5))


def look_y_for_centered_span(center_y: float, height: float, vfov_rad: float, tilt_rad: float = TILT_RAD) -> float:
    """Look-at Y [m] that puts ``center_y`` in the middle of the vertical ground span.

    A tilted perspective camera's ground footprint is not centered on the
    look-at point. This shifts the look-at so the subject is.
    """
    half = vfov_rad * 0.5
    delta = 0.5 * (math.tan(tilt_rad - half) + math.tan(tilt_rad + half)) - math.tan(tilt_rad)
    return center_y - height * delta


def field_bounds(tiles: list[TileGeom]) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of every maze, ``(min_x, min_y, max_x, max_y)`` [m]."""
    min_x = min(tile.origin_x for tile in tiles)
    min_y = min(tile.origin_y for tile in tiles)
    max_x = max(tile.origin_x + tile.footprint for tile in tiles)
    max_y = max(tile.origin_y + tile.footprint for tile in tiles)
    return min_x, min_y, max_x, max_y


def overview_pose(tiles: list[TileGeom], vfov_rad: float, hfov_rad: float) -> Pose:
    """High vantage that fits every maze in frame."""
    min_x, min_y, max_x, max_y = field_bounds(tiles)
    center_x = 0.5 * (min_x + max_x)
    center_y = 0.5 * (min_y + max_y)
    span_x = (max_x - min_x) * OVERVIEW_MARGIN
    span_y = (max_y - min_y) * OVERVIEW_MARGIN
    height = max(
        height_for_vertical_span(span_y, vfov_rad),
        height_for_horizontal_span(span_x, hfov_rad),
    )
    look_y = look_y_for_centered_span(center_y, height, vfov_rad)
    return nadir_pose(center_x, look_y, height)


def fill_pose(tile: TileGeom, vfov_rad: float) -> Pose:
    """Height at which this maze fills the frame from top to bottom."""
    center_x, center_y = tile.center
    height = height_for_vertical_span(tile.footprint * FILL_MARGIN, vfov_rad)
    look_y = look_y_for_centered_span(center_y, height, vfov_rad)
    return nadir_pose(center_x, look_y, height)


def follow_pose(tile: TileGeom, look_xy: tuple[float, float], vfov_rad: float) -> Pose:
    """Slightly lower than :func:`fill_pose`, looking at ``look_xy``."""
    height = height_for_vertical_span(tile.footprint * FILL_MARGIN, vfov_rad) * FOLLOW_HEIGHT_SCALE
    return nadir_pose(look_xy[0], look_xy[1], height)


def choose_tiles(num_tiles: int, count: int, seed: int) -> list[int]:
    """Distinct tile indices. Fewer than ``count`` tiles returns every tile."""
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}.")
    population = list(range(num_tiles))
    take = min(count, num_tiles)
    return random.Random(seed).sample(population, take)


def build_shots(tile_ids: list[int]) -> list[Shot]:
    """Overview, then for each tile a move, a top-down hold, and a follow."""
    shots = [Shot("overview", OVERVIEW_S, None)]
    for tile in tile_ids:
        shots.append(Shot("move", MOVE_S, tile))
        shots.append(Shot("fill", FILL_S, tile))
        shots.append(Shot("follow", FOLLOW_S, tile))
    return shots


def _smoothstep(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3.0 - 2.0 * u)


def _lerp_pose(start: Pose, end: Pose, u: float) -> Pose:
    s = _smoothstep(u)

    def mix(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(a[i] + (b[i] - a[i]) * s for i in range(3))  # type: ignore[return-value]

    return Pose(eye=mix(start.eye, end.eye), lookat=mix(start.lookat, end.lookat))


class FilmDirector:
    """Steps the shot list one 50 fps frame at a time."""

    def __init__(self, tiles: list[TileGeom], tile_ids: list[int], vfov_rad: float, hfov_rad: float):
        if not tiles:
            raise ValueError("tiles must not be empty.")
        self.tiles = tiles
        self.vfov_rad = vfov_rad
        self.hfov_rad = hfov_rad
        self.shots = build_shots(tile_ids)
        self.frame = 0
        self.total_frames = sum(max(1, round(shot.duration_s * FPS)) for shot in self.shots)
        self.overview = overview_pose(tiles, vfov_rad, hfov_rad)
        self.fill_poses = {tile: fill_pose(tiles[tile], vfov_rad) for tile in tile_ids}
        self._move_index = -1
        self._move_start = self.overview
        self._move_end = self.overview
        self._follow_tile = -1
        self._follow_xy: tuple[float, float] | None = None

    @property
    def finished(self) -> bool:
        return self.frame >= self.total_frames

    def shot_at(self, frame: int) -> tuple[int, Shot, float]:
        """Return ``(index, shot, progress in [0, 1))`` for a frame number."""
        cursor = 0
        for index, shot in enumerate(self.shots):
            length = max(1, round(shot.duration_s * FPS))
            if frame < cursor + length:
                return index, shot, (frame - cursor) / length
            cursor += length
        last = len(self.shots) - 1
        return last, self.shots[last], 1.0

    def pose(self, robot_xy: dict[int, tuple[float, float]]) -> Pose | None:
        """Camera pose for the current frame. ``None`` once the shot list is over."""
        if self.finished:
            return None
        index, shot, progress = self.shot_at(self.frame)
        if shot.kind == "overview":
            return self.overview
        if shot.kind == "fill":
            return self.fill_poses[shot.tile]  # type: ignore[index]
        if shot.kind == "follow":
            return self._follow(shot.tile, robot_xy)  # type: ignore[arg-type]
        self._begin_move(index, shot, robot_xy)
        return _lerp_pose(self._move_start, self._move_end, progress)

    def advance(self) -> None:
        self.frame += 1

    def _follow(self, tile: int, robot_xy: dict[int, tuple[float, float]]) -> Pose:
        measured = robot_xy.get(tile, self.tiles[tile].center)
        if self._follow_tile != tile or self._follow_xy is None:
            self._follow_xy = measured
            self._follow_tile = tile
        else:
            alpha = 1.0 - math.exp(-FRAME_DT_S / FOLLOW_LAG_S)
            self._follow_xy = (
                self._follow_xy[0] + alpha * (measured[0] - self._follow_xy[0]),
                self._follow_xy[1] + alpha * (measured[1] - self._follow_xy[1]),
            )
        return follow_pose(self.tiles[tile], self._follow_xy, self.vfov_rad)

    def _begin_move(self, index: int, shot: Shot, robot_xy: dict[int, tuple[float, float]]) -> None:
        if self._move_index == index:
            return
        previous = self.shots[index - 1]
        if previous.kind == "overview":
            start = self.overview
        elif previous.kind == "follow":
            held = self._follow_xy if self._follow_tile == previous.tile and self._follow_xy is not None else None
            look = held or robot_xy.get(previous.tile, self.tiles[previous.tile].center)  # type: ignore[index]
            start = follow_pose(self.tiles[previous.tile], look, self.vfov_rad)  # type: ignore[index]
        else:
            start = self.fill_poses[previous.tile]  # type: ignore[index]
        self._move_start = start
        self._move_end = self.fill_poses[shot.tile]  # type: ignore[index]
        self._move_index = index
