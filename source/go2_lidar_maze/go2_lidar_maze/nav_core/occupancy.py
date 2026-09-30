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

    def known_bounds(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inclusive cell bounds of walls and free space for ``env_ids``.

        Returns ``lo`` and ``hi`` shaped ``(K, 2)`` as ``(row, col)``, and
        ``has_free`` shaped ``(K,)``. An environment with no free cells has
        ``has_free`` false. Bounds come from one reduction, not a scan per env.
        """
        counts = self.counts[env_ids]
        free = self.free_counts[env_ids]
        known = (counts >= self.hit_threshold) | (free > 0)
        rows = known.any(dim=2)
        cols = known.any(dim=1)
        index = torch.arange(self.size, device=self.device)
        row_lo = torch.where(rows, index, self.size).amin(dim=1)
        row_hi = torch.where(rows, index, -1).amax(dim=1)
        col_lo = torch.where(cols, index, self.size).amin(dim=1)
        col_hi = torch.where(cols, index, -1).amax(dim=1)
        lo = torch.stack((row_lo, col_lo), dim=-1)
        hi = torch.stack((row_hi, col_hi), dim=-1)
        has_free = (free > 0).flatten(1).any(dim=1)
        return lo, hi, has_free

    def obstacle_mask(self) -> torch.Tensor:
        """Boolean wall mask ``(N, size, size)``."""
        return self.counts >= self.hit_threshold

    def seen_free_mask(self) -> torch.Tensor:
        """Boolean mask of cells a beam has passed through and that are not walls."""
        return (self.free_counts > 0) & ~self.obstacle_mask()

    @torch.no_grad()
    def update(self, ray_hits_w: torch.Tensor, sensor_pos_w: torch.Tensor, free_samples: int | None = None) -> None:
        """Fold one scan into the grid.

        Args:
            ray_hits_w: Hit points ``(N, B, 3)`` [m]. Misses are non-finite.
            sensor_pos_w: Sensor origin ``(N, 3)`` [m].
            free_samples: Samples taken along each beam, short of the hit.
                ``None`` picks a spacing no coarser than the cell size, capped
                at 48, so a beam marks every free cell it crosses.
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
        hit_lin = self._index_add_cells(self.counts, flat, valid, sink)

        finite = torch.isfinite(dist) & (dist > 0.05)
        if free_samples is None:
            # The finest cell the grid can use. Denser than a coarser env's cell is still correct.
            free_samples = int(self.max_valid_range / max(self.base_res, 0.05))
            free_samples = min(max(free_samples, 8), 48)
        d_safe = torch.where(finite, dist, torch.zeros_like(dist))
        free_len = (torch.minimum(d_safe, torch.full_like(d_safe, self.max_valid_range)) - 0.3).clamp(min=0.0)
        direction = delta[..., :2] / d_safe.clamp(min=1e-6).unsqueeze(-1)
        direction = torch.where(finite.unsqueeze(-1), direction, torch.zeros_like(direction))
        fractions = torch.linspace(
            1.0 / (free_samples + 1),
            free_samples / (free_samples + 1),
            free_samples,
            device=self.device,
        )
        travel = (free_len.unsqueeze(-1) * fractions.view(1, 1, -1)).unsqueeze(-1)
        pts = sensor_pos_w[:, :2].unsqueeze(1).unsqueeze(2) + direction.unsqueeze(2) * travel
        flat_pts = pts.reshape(self.num_envs, -1, 2)
        inside = (finite & (free_len > 0.05)).unsqueeze(-1).expand(-1, -1, free_samples)
        inside = inside.reshape(self.num_envs, -1) & self.in_bounds(flat_pts)
        free_cells = self.world_to_cell(flat_pts)
        free_flat = free_cells[..., 0] * self.size + free_cells[..., 1]
        free_lin = self._index_add_cells(self.free_counts, free_flat, inside, sink)
        # A cell that is both hit and traversed counts as a wall, not free.
        # Only the cells this scan touched can have changed, including a new
        # wall that no free sample landed in.
        both = torch.cat((hit_lin, free_lin))
        counts_at = self.counts.view(-1).gather(0, both)
        free_at = self.free_counts.view(-1).gather(0, both)
        free_at = free_at.masked_fill(counts_at >= self.hit_threshold, 0)
        self.free_counts.view(-1).scatter_(0, both, free_at)

    def _index_add_cells(self, grid: torch.Tensor, flat: torch.Tensor, valid: torch.Tensor, sink: int) -> torch.Tensor:
        """Add 1 to each valid flat cell and clamp those cells to 1000.

        Invalid beams contribute a zero weight, so they do not need a compacted
        index list. Duplicate indices add first, then every write of a cell is
        the same clamped value.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)[:, None].expand_as(flat)
        lin = (env_ids * sink + flat.clamp(0, sink - 1)).reshape(-1)
        weight = valid.reshape(-1).to(dtype=grid.dtype)
        flat_grid = grid.view(-1)
        flat_grid.index_add_(0, lin, weight)
        flat_grid.scatter_(0, lin, flat_grid.gather(0, lin).clamp(max=1000))
        return lin

    def _broadcast_res(self, ref: torch.Tensor) -> torch.Tensor:
        shape = [self.num_envs] + [1] * (ref.dim() - 1)
        return self.res.view(shape)

    def _broadcast_origin(self, ref: torch.Tensor) -> torch.Tensor:
        shape = [self.num_envs] + [1] * (ref.dim() - 2) + [2]
        return self.origin.view(shape)
