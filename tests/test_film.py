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
        *(["move", "fill", "approach", "follow"] * 4),
    ]
    durations = [shot.duration_s for shot in shots]
    assert durations[0] == 4.0
    assert durations[1:] == [2.0, 4.0, 2.0, 20.0] * 4
    assert sum(durations) == 116.0


def test_four_distinct_tiles_and_opening_pose():
    film = _film()
    tiles = _tiles(film)
    chosen = film.choose_tiles(tiles, 4, seed=7)
    assert len(chosen) == 4
    assert len(set(chosen)) == 4
    assert film.choose_tiles(tiles, 4, seed=7) == chosen
    for left, right in zip(chosen, chosen[1:]):
        assert right in film.adjacent_tiles(tiles, left)

    vfov, hfov = film.fallback_fov()
    director = film.FilmDirector(tiles, chosen, vfov, hfov)
    assert director.total_frames == 116 * film.FPS
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


def test_later_mazes_are_adjacent():
    film = _film()
    tiles = _tiles(film)
    for seed in range(12):
        chosen = film.choose_tiles(tiles, 4, seed)
        assert len(set(chosen)) == 4
        for left, right in zip(chosen, chosen[1:]):
            assert right in film.adjacent_tiles(tiles, left)


def test_one_maze_stays_inside_the_frame_on_both_axes():
    film = _film()
    tiles = _tiles(film)
    vfov, hfov = film.fallback_fov()
    tile = tiles[3]
    pose = film.fill_pose(tile, vfov, hfov)
    framed = film.height_to_frame_tile(tile, tile.center, vfov, hfov)
    assert abs(pose.eye[2] - framed * film.FILL_HEIGHT_SCALE) < 1e-6
    corner = (tile.origin_x + 0.2, tile.origin_y + 0.2)
    follow = film.follow_pose(tile, corner, vfov, hfov)
    assert abs(follow.eye[2] - framed * film.FOLLOW_HEIGHT_SCALE) < 1e-6


def test_follow_tracks_the_robot_without_cropping_the_maze():
    film = _film()
    tiles = _tiles(film)
    vfov, hfov = film.fallback_fov()
    director = film.FilmDirector(tiles, [3], vfov, hfov)
    fill = director.fill_poses[3]
    robot = (tiles[3].center[0] + 1.5, tiles[3].center[1] - 0.5)
    # Jump to the follow shot: overview, move, fill, approach, then follow.
    director.frame = round((4.0 + 2.0 + 4.0 + 2.0) * film.FPS)
    pose = director.pose({3: robot})
    assert pose == film.follow_pose(tiles[3], robot, vfov, hfov)
    framed = film.height_to_frame_tile(tiles[3], tiles[3].center, vfov, hfov)
    assert abs(fill.eye[2] - framed * film.FILL_HEIGHT_SCALE) < 1e-6
    assert abs(pose.eye[2] - framed * film.FOLLOW_HEIGHT_SCALE) < 1e-6


def test_descent_onto_the_robot_does_not_cut():
    film = _film()
    tiles = _tiles(film)
    vfov, hfov = film.fallback_fov()
    director = film.FilmDirector(tiles, [3], vfov, hfov)
    robot = {3: (tiles[3].center[0] + 4.0, tiles[3].center[1] - 3.0)}
    director.frame = round((4.0 + 2.0 + 4.0) * film.FPS) - 1
    held = director.pose(robot)
    director.advance()
    first = director.pose(robot)
    assert abs(first.eye[2] - held.eye[2]) < 1e-6
    assert abs(first.lookat[0] - held.lookat[0]) < 1e-6
    director.frame = round((4.0 + 2.0 + 4.0 + 2.0) * film.FPS) - 1
    arrived = director.pose(robot)
    expected = film.follow_pose(tiles[3], robot[3], vfov, hfov)
    assert abs(arrived.eye[2] - expected.eye[2]) < 1e-6
    assert abs(arrived.lookat[0] - expected.lookat[0]) < 1e-6
    director.advance()
    following = director.pose(robot)
    assert abs(following.eye[0] - arrived.eye[0]) < 1e-6
    assert abs(following.lookat[0] - arrived.lookat[0]) < 1e-6
