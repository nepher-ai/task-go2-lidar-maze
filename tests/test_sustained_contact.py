# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark collision requires the force on every stored contact frame."""

from __future__ import annotations

import torch

from go2_lidar_maze.tasks.hl_maze.mdp.terminations import sustained_illegal_contact


class _Buf:
    def __init__(self, value: torch.Tensor):
        self.torch = value


class _Data:
    def __init__(self, forces: torch.Tensor):
        self.net_normal_forces_w_history = _Buf(forces)


class _Sensor:
    def __init__(self, forces: torch.Tensor):
        self.data = _Data(forces)


class _Scene:
    def __init__(self, sensor: _Sensor):
        self.sensors = {"contact_forces": sensor}


class _Env:
    def __init__(self, forces: torch.Tensor):
        self.scene = _Scene(_Sensor(forces))


class _SensorCfg:
    name = "contact_forces"
    body_ids = slice(None)


def _force(history: list[float]) -> torch.Tensor:
    """One env, one body, force along z [N]. Shape (1, T, 1, 3)."""
    values = torch.tensor(history, dtype=torch.float32)
    force = torch.zeros(1, values.shape[0], 1, 3)
    force[0, :, 0, 2] = values
    return force


def test_one_frame_above_threshold_does_not_fail():
    env = _Env(_force([0.0, 50.0, 0.0]))
    assert not bool(sustained_illegal_contact(env, 10.0, _SensorCfg()).item())


def test_every_frame_above_threshold_fails():
    env = _Env(_force([12.0, 11.0, 10.5]))
    assert bool(sustained_illegal_contact(env, 10.0, _SensorCfg()).item())


def test_force_at_threshold_does_not_fail():
    env = _Env(_force([10.0, 10.0, 10.0]))
    assert not bool(sustained_illegal_contact(env, 10.0, _SensorCfg()).item())
