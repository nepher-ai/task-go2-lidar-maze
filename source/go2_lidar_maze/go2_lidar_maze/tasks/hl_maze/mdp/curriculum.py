# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Raise the privileged A* length once waypoints are being reached."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

from go2_lidar_maze.constants import MAX_ASTAR_CELLS


class raise_waypoint_distance(ManagerTermBase):
    """Increase ``k_max`` by one cell when the resetting batch succeeds often enough."""

    def __init__(self, cfg: CurriculumTermCfg, env):
        super().__init__(cfg, env)

    def __call__(self, env, env_ids: Sequence[int], command_name: str, threshold: float = 0.8) -> dict[str, float]:
        term = env.command_manager.get_term(command_name)
        issued = term.issued_count[env_ids]
        total = float(issued.sum())
        if total < 16.0 or term.k_max >= MAX_ASTAR_CELLS:
            return {"k_max": float(term.k_max)}
        success = float(term.reached_count[env_ids].sum() / total)
        if success >= threshold:
            term.k_max += 1
        return {"k_max": float(term.k_max)}
