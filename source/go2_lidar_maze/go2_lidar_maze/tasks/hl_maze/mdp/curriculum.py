# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Raise the privileged A* length once the hardest waypoints are being reached."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg

from go2_lidar_maze.constants import MAX_ASTAR_CELLS

# Same bodies and force as the collision termination. The sensor filter is the
# maze walls, so the force matrix is wall contact. Net contact also includes
# the ground.
_WALL_BODIES = ["base", ".*hip", ".*thigh", ".*calf"]
_CONTACT_FORCE_N = 1.0


class raise_waypoint_distance(ManagerTermBase):
    """Increase ``k_max`` by one cell when the longest waypoints hold across two windows.

    Privileged episodes issue one waypoint. ``waypoint_reached`` latches
    ``reached_count`` before this term runs, so the pooled rate is the fraction of
    episodes that arrived. Each window waits until it contains at least
    ``min_episodes`` episodes and ``min_issued`` issued waypoints. Episodes that
    were never issued a waypoint are ignored. The window passes when the reach
    rate at length ``k_max`` is at least ``threshold`` and the fraction of those
    episodes with a collision termination is strictly below ``max_collision``.
    Shorter lengths are logged but do not open the gate. ``k_max`` increases
    only after ``passes_required`` passing windows in a row. The counters then wait
    one episode, so the next window is commanded at the new length.
    """

    def __init__(self, cfg: CurriculumTermCfg, env):
        super().__init__(cfg, env)
        # common_step_counter at which another increase is allowed.
        self._ready_step = 0
        self._reached = 0.0
        self._issued = 0.0
        self._episodes = 0.0
        self._collisions = 0.0
        self._wall = 0.0
        self._other = 0.0
        self._passes = 0
        # Rates from the last completed window. Partials fall back to these.
        self._success = 0.0
        self._collision = 0.0
        self._wall_rate = 0.0
        self._other_rate = 0.0
        self._reached_k = [0.0] * MAX_ASTAR_CELLS
        self._issued_k = [0.0] * MAX_ASTAR_CELLS
        self._episodes_k = [0.0] * MAX_ASTAR_CELLS
        self._collisions_k = [0.0] * MAX_ASTAR_CELLS
        self._success_k = [0.0] * MAX_ASTAR_CELLS
        self._collision_k = [0.0] * MAX_ASTAR_CELLS
        self._body_ids: torch.Tensor | slice | None = None

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
        # The post-increase wait keeps the next window on the new length.
        # At ``k_max == 5`` the counters still close, so that length stays visible.
        if env.common_step_counter < self._ready_step:
            return self._reading(term)

        issued = term.issued_count[env_ids]
        reached = term.reached_count[env_ids]
        commanded = issued > 0
        if bool(commanded.any()):
            collided = env.termination_manager.get_term(collision_term)[env_ids]
            wall, other = self._contact_flags(env, env_ids, collided)
            self._issued += float(issued[commanded].sum())
            self._reached += float(reached[commanded].sum())
            self._episodes += float(commanded.sum())
            self._collisions += float(collided[commanded].sum())
            self._wall += float(wall[commanded].sum())
            self._other += float(other[commanded].sum())
            lengths = term.k_issued[env_ids]
            for level in range(MAX_ASTAR_CELLS):
                mask = commanded & (lengths == level + 1)
                if not bool(mask.any()):
                    continue
                self._issued_k[level] += float(issued[mask].sum())
                self._reached_k[level] += float(reached[mask].sum())
                self._episodes_k[level] += float(mask.sum())
                self._collisions_k[level] += float(collided[mask].sum())

        if self._issued < min_issued or self._episodes < min_episodes:
            return self._reading(term)

        self._store_rates()
        level_success, level_collision = self._frontier_rates(term)
        self._clear_counts()
        if term.k_max >= MAX_ASTAR_CELLS:
            return self._reading(term)
        if level_success >= threshold and level_collision < max_collision:
            self._passes += 1
        else:
            self._passes = 0
        if self._passes >= passes_required:
            term.k_max += 1
            self._passes = 0
            self._ready_step = int(env.common_step_counter) + int(env.max_episode_length)
        return self._reading(term)

    def _store_rates(self) -> None:
        """Freeze this window's rates. Lengths with no samples keep their previous rate."""
        self._success = self._reached / self._issued
        self._collision = self._collisions / self._episodes
        self._wall_rate = self._wall / self._episodes
        self._other_rate = self._other / self._episodes
        for level in range(MAX_ASTAR_CELLS):
            if self._issued_k[level] <= 0.0:
                continue
            self._success_k[level] = self._reached_k[level] / self._issued_k[level]
            self._collision_k[level] = self._collisions_k[level] / self._episodes_k[level]

    def _frontier_rates(self, term) -> tuple[float, float]:
        """Reach and collision rates at length ``k_max``. A missing length cannot pass."""
        frontier = int(term.k_max) - 1
        if self._issued_k[frontier] <= 0.0:
            return 0.0, 1.0
        return self._success_k[frontier], self._collision_k[frontier]

    def _clear_counts(self) -> None:
        self._reached = 0.0
        self._issued = 0.0
        self._episodes = 0.0
        self._collisions = 0.0
        self._wall = 0.0
        self._other = 0.0
        self._reached_k = [0.0] * MAX_ASTAR_CELLS
        self._issued_k = [0.0] * MAX_ASTAR_CELLS
        self._episodes_k = [0.0] * MAX_ASTAR_CELLS
        self._collisions_k = [0.0] * MAX_ASTAR_CELLS

    def _contact_flags(self, env, env_ids, collided: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Wall contact, and collision that is not wall contact.

        ``normal_force_matrix_w_history`` is the wall-filtered force. Under
        PhysX, ``force_matrix_w`` is that same normal force and warns on every
        read, and it has no history. The history max matches ``illegal_contact``.
        """
        sensor = env.scene.sensors["contact_forces"]
        history = sensor.data.normal_force_matrix_w_history
        matrix = None if history is None else history.torch
        if matrix is None:
            return torch.zeros_like(collided), collided
        if self._body_ids is None:
            cfg = SceneEntityCfg("contact_forces", body_names=_WALL_BODIES)
            cfg.resolve(env.scene)
            if isinstance(cfg.body_ids, slice):
                self._body_ids = cfg.body_ids
            else:
                self._body_ids = torch.as_tensor(cfg.body_ids, device=matrix.device, dtype=torch.long)
        selected = matrix[env_ids][:, :, self._body_ids]
        peak = torch.linalg.norm(selected, dim=-1).amax(dim=(1, 2, 3))
        wall = peak > _CONTACT_FORCE_N
        return wall, collided & ~wall

    def _reading(self, term) -> dict[str, float]:
        """``k_max``, pooled rates, per-length rates, and the contact split."""
        success = self._reached / self._issued if self._issued > 0.0 else self._success
        collision = self._collisions / self._episodes if self._episodes > 0.0 else self._collision
        wall = self._wall / self._episodes if self._episodes > 0.0 else self._wall_rate
        other = self._other / self._episodes if self._episodes > 0.0 else self._other_rate
        reading = {
            "k_max": float(term.k_max),
            "success": float(success),
            "collision": float(collision),
            "wall_contact": float(wall),
            "other_contact": float(other),
        }
        for level in range(MAX_ASTAR_CELLS):
            length = level + 1
            if self._issued_k[level] > 0.0:
                reading[f"success_k{length}"] = self._reached_k[level] / self._issued_k[level]
                reading[f"collision_k{length}"] = self._collisions_k[level] / self._episodes_k[level]
            else:
                reading[f"success_k{length}"] = self._success_k[level]
                reading[f"collision_k{length}"] = self._collision_k[level]
        return reading
