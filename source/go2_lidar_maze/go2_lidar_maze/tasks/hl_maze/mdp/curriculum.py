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
    """Increase ``k_max`` by one cell when waypoint success holds across two windows.

    Each window pools resetting episodes until it contains at least ``min_episodes``
    episodes and ``min_issued`` issued waypoints. Episodes that were never issued a
    waypoint are ignored. The window passes when the pooled reach rate is at least
    ``threshold`` and the fraction of those episodes with a collision termination is
    strictly below ``max_collision``. ``k_max`` increases only after ``passes_required``
    passing windows in a row. The counters then wait one episode, so the next window
    is commanded at the new length.
    """

    def __init__(self, cfg: CurriculumTermCfg, env):
        super().__init__(cfg, env)
        # common_step_counter at which another increase is allowed.
        self._ready_step = 0
        self._reached = 0.0
        self._issued = 0.0
        self._episodes = 0.0
        self._collisions = 0.0
        self._passes = 0
        # Rates from the last completed window. Partials fall back to these.
        self._success = 0.0
        self._collision = 0.0

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        command_name: str,
        threshold: float = 0.8,
        min_issued: float = 512.0,
        min_episodes: float = 256.0,
        max_collision: float = 0.5,
        passes_required: int = 2,
        collision_term: str = "collision",
    ) -> dict[str, float]:
        term = env.command_manager.get_term(command_name)
        if term.k_max >= MAX_ASTAR_CELLS or env.common_step_counter < self._ready_step:
            return self._reading(term)

        issued = term.issued_count[env_ids]
        reached = term.reached_count[env_ids]
        commanded = issued > 0
        if bool(commanded.any()):
            collided = env.termination_manager.get_term(collision_term)[env_ids]
            self._issued += float(issued[commanded].sum())
            self._reached += float(reached[commanded].sum())
            self._episodes += float(commanded.sum())
            self._collisions += float(collided[commanded].sum())

        if self._issued < min_issued or self._episodes < min_episodes:
            return self._reading(term)

        self._success = self._reached / self._issued
        self._collision = self._collisions / self._episodes
        self._reached = 0.0
        self._issued = 0.0
        self._episodes = 0.0
        self._collisions = 0.0
        if self._success >= threshold and self._collision < max_collision:
            self._passes += 1
        else:
            self._passes = 0
        if self._passes >= passes_required:
            term.k_max += 1
            self._passes = 0
            self._ready_step = int(env.common_step_counter) + int(env.max_episode_length)
        return self._reading(term)

    def _reading(self, term) -> dict[str, float]:
        """``k_max`` plus the pooled rates of the open window, or the last closed one."""
        success = self._reached / self._issued if self._issued > 0.0 else self._success
        collision = self._collisions / self._episodes if self._episodes > 0.0 else self._collision
        return {
            "k_max": float(term.k_max),
            "success": float(success),
            "collision": float(collision),
        }
