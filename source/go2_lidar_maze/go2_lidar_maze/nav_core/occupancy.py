# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online occupancy grid built from 360-degree LiDAR hits.

The grid starts empty on every reset. Wall cells are LiDAR endpoints. Free
cells are samples along each beam, stopping short of the hit. Cells with
neither stay unknown. Nothing about the maze is loaded into the grid.
"""

from __future__ import annotations

import torch


class OccupancyGrid:
    """Per-environment wall and free-space evidence.

    Args:
        num_envs: Number of parallel environments.
        device: Torch device.
        resolution: Cell size [m].
        size: Grid edge length in cells.
        hit_threshold: Returns in a cell before it counts as a wall.
        max_valid_range: Returns at or beyond this range are misses [m].
        min_hit_height: Returns below this height are discarded [m].
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        resolution: float = 0.2,
        size: int = 320,
        hit_threshold: int = 1,
        max_valid_range: float = 9.5,
        min_hit_height: float = 0.05,
    ) -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.base_res = float(resolution)
        self.res = torch.full((num_envs,), float(resolution), dtype=torch.float32, device=self.device)
        self.size = int(size)
        self.hit_threshold = int(hit_threshold)
        self.max_valid_range = float(max_valid_range)
        self.min_hit_height = float(min_hit_height)
        self.counts = torch.zeros((num_envs, size, size), dtype=torch.int32, device=self.device)
        self.free_counts = torch.zeros((num_envs, size, size), dtype=torch.int16, device=self.device)
        self.origin = torch.zeros((num_envs, 2), dtype=torch.float32, device=self.device)
        self._half = size // 2

    def reset(
        self,
        env_ids: torch.Tensor,
        origin_xy: torch.Tensor,
        resolution: torch.Tensor | None = None,
    ) -> None:
        """Clear ``env_ids`` and re-anchor the grid on ``origin_xy`` [m]."""
        self.counts[env_ids] = 0
        self.free_counts[env_ids] = 0
        self.origin[env_ids] = origin_xy.to(self.origin.dtype)
        if resolution is None:
            self.res[env_ids] = self.base_res
        else:
            self.res[env_ids] = resolution.to(self.res.dtype)

    def world_to_cell(self, xy: torch.Tensor) -> torch.Tensor:
        """World XY ``(N, ..., 2)`` to integer ``(row, col)``, clamped to the grid."""
        rel = (xy - self._broadcast_origin(xy)) / self._broadcast_res(xy)
        cell = torch.floor(rel).to(torch.long) + self._half
        return cell.clamp_(0, self.size - 1)

    def in_bounds(self, xy: torch.Tensor) -> torch.Tensor:
        """True where ``xy`` lies inside the grid extent."""
        rel = (xy - self._broadcast_origin(xy)) / self._broadcast_res(xy)
        return (rel.abs() < float(self._half)).all(dim=-1)

    def cell_to_world(self, cell: torch.Tensor) -> torch.Tensor:
        """Integer ``(row, col)`` to the world XY of the cell centre [m]."""
        rel = (cell.to(torch.float32) - self._half + 0.5) * self._broadcast_res(cell)
        return rel + self._broadcast_origin(rel)

    def obstacle_mask(self) -> torch.Tensor:
        """Boolean wall mask ``(N, size, size)``."""
        return self.counts >= self.hit_threshold

    def seen_free_mask(self) -> torch.Tensor:
        """Boolean mask of cells a beam has passed through and that are not walls."""
        return (self.free_counts > 0) & ~self.obstacle_mask()

    @torch.no_grad()
    def update(self, ray_hits_w: torch.Tensor, sensor_pos_w: torch.Tensor, free_samples: int = 8) -> None:
        """Fold one scan into the grid.

        Args:
            ray_hits_w: Hit points ``(N, B, 3)`` [m]. Misses are non-finite.
            sensor_pos_w: Sensor origin ``(N, 3)`` [m].
            free_samples: Samples taken along each beam, short of the hit.
        """
        delta = ray_hits_w - sensor_pos_w.unsqueeze(1)
        dist = torch.linalg.norm(delta, dim=-1)
        valid = (
            torch.isfinite(dist)
            & (dist > 0.05)
            & (dist < self.max_valid_range)
            & (ray_hits_w[..., 2] > self.min_hit_height)
            & self.in_bounds(ray_hits_w[..., :2])
        )
        cells = self.world_to_cell(ray_hits_w[..., :2])
        flat = cells[..., 0] * self.size + cells[..., 1]
        sink = self.size * self.size
        flat = torch.where(valid, flat, torch.full_like(flat, sink))
        buf = torch.zeros((self.num_envs, sink + 1), dtype=torch.int32, device=self.device)
        buf.scatter_add_(1, flat, torch.ones_like(flat, dtype=torch.int32))
        self.counts += buf[:, :sink].view(self.num_envs, self.size, self.size)
        self.counts.clamp_(max=1000)

        finite = torch.isfinite(dist) & (dist > 0.05)
        if not bool(finite.any()):
            return
        d_safe = torch.where(finite, dist, torch.zeros_like(dist))
        free_len = (torch.minimum(d_safe, torch.full_like(d_safe, self.max_valid_range)) - 0.3).clamp(min=0.0)
        direction = delta / d_safe.clamp(min=1e-6).unsqueeze(-1)
        direction = torch.where(finite.unsqueeze(-1), direction, torch.zeros_like(direction))
        for sample in range(1, free_samples + 1):
            frac = sample / (free_samples + 1)
            pt = sensor_pos_w.unsqueeze(1) + direction * (free_len * frac).unsqueeze(-1)
            inside = finite & (free_len > 0.05) & self.in_bounds(pt[..., :2])
            free_cells = self.world_to_cell(pt[..., :2])
            free_flat = free_cells[..., 0] * self.size + free_cells[..., 1]
            free_flat = torch.where(inside, free_flat, torch.full_like(free_flat, sink))
            free_buf = torch.zeros((self.num_envs, sink + 1), dtype=torch.int32, device=self.device)
            free_buf.scatter_add_(1, free_flat, torch.ones_like(free_flat, dtype=torch.int32))
            self.free_counts += free_buf[:, :sink].view(self.num_envs, self.size, self.size).to(torch.int16)
        self.free_counts.clamp_(max=1000)
        # A cell that is both hit and traversed counts as a wall, not free.
        self.free_counts.masked_fill_(self.obstacle_mask(), 0)

    def _broadcast_res(self, ref: torch.Tensor) -> torch.Tensor:
        shape = [self.num_envs] + [1] * (ref.dim() - 1)
        return self.res.view(shape)

    def _broadcast_origin(self, ref: torch.Tensor) -> torch.Tensor:
        shape = [self.num_envs] + [1] * (ref.dim() - 2) + [2]
        return self.origin.view(shape)
