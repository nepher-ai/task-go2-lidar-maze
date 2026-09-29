# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Arc-length queries used by the high-level path observation."""

from __future__ import annotations

import torch
from go2_lidar_maze.nav_core.path_utils import point_at_distance, points_at_arc_length


def test_points_at_arc_length_on_an_l():
    """An L of legs 3 m and 4 m. Distances past the end stay on the last vertex."""
    # Padded with a repeated tail, which must not move the 7 m endpoint.
    poly = torch.tensor(
        [[[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [3.0, 4.0]]],
        dtype=torch.float32,
    )
    distance = torch.tensor([[0.0, 1.5, 3.0, 5.0, 7.0, 9.0]], dtype=torch.float32)
    expected = torch.tensor(
        [[[0.0, 0.0], [1.5, 0.0], [3.0, 0.0], [3.0, 2.0], [3.0, 4.0], [3.0, 4.0]]],
        dtype=torch.float32,
    )
    got = points_at_arc_length(poly, distance)
    assert torch.allclose(got, expected, atol=1e-5)


def test_points_at_arc_length_matches_point_at_distance():
    """The batched query agrees with the single-polyline helper, on two rows."""
    points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], dtype=torch.float32)
    batch = points.unsqueeze(0).expand(2, -1, -1).contiguous()
    distance = torch.tensor([0.5, 2.5, 4.0], dtype=torch.float32)
    got = points_at_arc_length(batch, distance.unsqueeze(0).expand(2, -1).contiguous())
    for index, value in enumerate(distance):
        reference = point_at_distance(points, value)
        assert torch.allclose(got[0, index], reference, atol=1e-5)
        assert torch.allclose(got[1, index], reference, atol=1e-5)
    single = points_at_arc_length(batch, torch.tensor([2.5, 1.0], dtype=torch.float32))
    assert single.shape == (2, 2)
    assert torch.allclose(single[0], point_at_distance(points, 2.5), atol=1e-5)
    assert torch.allclose(single[1], point_at_distance(points, 1.0), atol=1e-5)
