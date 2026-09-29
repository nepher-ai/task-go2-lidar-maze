# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ray origin includes the mount offset that ``data.pos_w`` leaves out."""

from __future__ import annotations

import math

import torch

from go2_lidar_maze.tasks.hl_maze.lidar_frame import footprint_exit, ray_origin_w

_OFFSET = (0.36, 0.0, 0.23)


class _Buf:
    def __init__(self, value: torch.Tensor):
        self.torch = value


class _Data:
    def __init__(self, pos: torch.Tensor, quat: torch.Tensor):
        self.pos_w = _Buf(pos)
        self.quat_w = _Buf(quat)


class _Cfg:
    def __init__(self, alignment: str):
        self.ray_alignment = alignment


class _Sensor:
    def __init__(self, pos: torch.Tensor, quat: torch.Tensor, alignment: str):
        self.data = _Data(pos, quat)
        self.cfg = _Cfg(alignment)
        local = torch.tensor(_OFFSET, dtype=torch.float32).view(1, 1, 3).expand(pos.shape[0], 2, 3).contiguous()
        self.ray_starts = _Buf(local)


def test_ray_origin_adds_forward_mount_at_identity():
    """A +X offset on an unrotated base stays forward of ``pos_w``."""
    pos = torch.tensor([[1.0, 2.0, 0.4]], dtype=torch.float32)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    origin = ray_origin_w(_Sensor(pos, quat, "base"))
    expected = torch.tensor([[1.36, 2.0, 0.63]], dtype=torch.float32)
    assert torch.allclose(origin, expected, atol=1e-5)


def test_ray_origin_yaws_the_mount_with_the_base():
    """90° yaw sends the forward mount onto +Y."""
    half = math.sin(math.pi / 4)
    pos = torch.tensor([[0.0, 0.0, 0.4]], dtype=torch.float32)
    quat = torch.tensor([[0.0, 0.0, half, half]], dtype=torch.float32)
    origin = ray_origin_w(_Sensor(pos, quat, "base"))
    expected = torch.tensor([[0.0, 0.36, 0.63]], dtype=torch.float32)
    assert torch.allclose(origin, expected, atol=1e-5)


def test_world_alignment_does_not_rotate_the_mount():
    """World alignment adds the offset in world axes."""
    half = math.sin(math.pi / 4)
    pos = torch.tensor([[0.0, 0.0, 0.4]], dtype=torch.float32)
    quat = torch.tensor([[0.0, 0.0, half, half]], dtype=torch.float32)
    origin = ray_origin_w(_Sensor(pos, quat, "world"))
    expected = torch.tensor([[0.36, 0.0, 0.63]], dtype=torch.float32)
    assert torch.allclose(origin, expected, atol=1e-5)


def test_footprint_exit_matches_centered_rho_and_a_forward_mount():
    """From the origin, τ is ρ. From x = 0.32 m, a forward beam exits in 0.03 m."""
    angles = torch.tensor([[0.0, math.pi / 2, math.pi / 4, math.pi]], dtype=torch.float32)
    centered = footprint_exit(torch.zeros(1, 2), angles)
    assert torch.allclose(
        centered,
        torch.tensor([[0.35, 0.20, 0.20 / math.sin(math.pi / 4), 0.35]], dtype=torch.float32),
        atol=1e-4,
    )
    mounted = footprint_exit(torch.tensor([[0.32, 0.0]]), angles)
    assert torch.allclose(mounted[0, 0], torch.tensor(0.03), atol=1e-4)
    assert torch.allclose(mounted[0, 3], torch.tensor(0.67), atol=1e-4)


def test_footprint_exit_is_zero_ahead_of_a_mount_past_the_front_edge():
    """The 0.36 m mount is outside the 0.35 m footprint, so a forward beam has already left it."""
    angles = torch.tensor([[0.0, math.pi / 2, math.radians(91.0), math.pi]], dtype=torch.float32)
    mounted = footprint_exit(torch.tensor([[0.36, 0.0]]), angles)
    assert torch.allclose(mounted[0, 0], torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(mounted[0, 1], torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(mounted[0, 2], torch.tensor(0.0), atol=1e-4)
    assert torch.allclose(mounted[0, 3], torch.tensor(0.71), atol=1e-4)
