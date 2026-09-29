# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shot list and camera height for the 50 fps benchmark film. No simulator."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_FILM_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "go2_lidar_maze"
    / "go2_lidar_maze"
    / "tasks"
    / "hl_maze"
    / "film.py"
)


def _film():
    spec = importlib.util.spec_from_file_location("go2_film_under_test", _FILM_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _tiles(module):
    # 8 columns, 40 m pitch, two sizes. Matches the benchmark layout at small scale.
    tiles = []
    for index in range(8):
        col, row = index % 4, index // 4
        footprint = 28.5 if index % 2 == 0 else 35.5
        tiles.append(module.TileGeom(col * 40.0, row * 40.0, footprint))
    return tiles


def test_tilt_is_ten_degrees_toward_plus_y():
    film = _film()
    pose = film.nadir_pose(10.0, 20.0, height=50.0)
    assert pose.eye[0] == 10.0
    assert pose.eye[2] == 50.0
    assert pose.lookat == (10.0, 20.0, 0.0)
    assert abs((pose.lookat[1] - pose.eye[1]) - 50.0 * math.tan(film.TILT_RAD)) < 1e-9


def test_vertical_span_round_trip():
    film = _film()
    vfov, _hfov = film.fallback_fov()
    span = 35.5
    height = film.height_for_vertical_span(span, vfov)
    half = vfov * 0.5
    covered = height * (math.tan(film.TILT_RAD + half) - math.tan(film.TILT_RAD - half))
    assert abs(covered - span) < 1e-9


def test_shot_list_times_four_mazes():
    film = _film()
    shots = film.build_shots([1, 4, 0, 7])
    assert [shot.kind for shot in shots] == [
        "overview",
        "move",
        "fill",
        "follow",
        "move",
        "fill",
        "follow",
        "move",
        "fill",
        "follow",
        "move",
        "fill",
        "follow",
    ]
    durations = [shot.duration_s for shot in shots]
    assert durations[0] == 4.0
    assert durations[1:] == [2.0, 4.0, 20.0] * 4
    assert sum(durations) == 108.0


def test_four_distinct_tiles_and_opening_pose():
    film = _film()
    tiles = _tiles(film)
    chosen = film.choose_tiles(len(tiles), 4, seed=7)
    assert len(chosen) == 4
    assert len(set(chosen)) == 4
    assert film.choose_tiles(len(tiles), 4, seed=7) == chosen

    vfov, hfov = film.fallback_fov()
    director = film.FilmDirector(tiles, chosen, vfov, hfov)
    assert director.total_frames == 108 * film.FPS
    opening = director.pose({})
    assert opening == director.overview
    # The whole field fits in both axes.
    min_x, min_y, max_x, max_y = film.field_bounds(tiles)
    assert director.overview.eye[2] + 1e-6 >= film.height_for_vertical_span(
        (max_y - min_y) * film.OVERVIEW_MARGIN, vfov
    )
    assert director.overview.eye[2] + 1e-6 >= film.height_for_horizontal_span(
        (max_x - min_x) * film.OVERVIEW_MARGIN, hfov
    )


def test_follow_is_lower_than_the_fill_and_tracks_the_robot():
    film = _film()
    tiles = _tiles(film)
    vfov, hfov = film.fallback_fov()
    director = film.FilmDirector(tiles, [3], vfov, hfov)
    fill = director.fill_poses[3]
    robot = (tiles[3].center[0] + 1.5, tiles[3].center[1] - 0.5)
    # Jump to the follow shot: overview, move, fill, then follow.
    director.frame = round((4.0 + 2.0 + 4.0) * film.FPS)
    pose = director.pose({3: robot})
    assert pose.eye[2] == fill.eye[2] * film.FOLLOW_HEIGHT_SCALE
    assert abs(pose.lookat[0] - robot[0]) < 1e-9
    assert abs(pose.lookat[1] - robot[1]) < 1e-9
