# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Policy files shipped next to this task project, not inside the Python package."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """``task-go2-lidar-maze`` directory."""
    return Path(__file__).resolve().parents[3]


def ll_policy_path() -> Path:
    """TorchScript velocity policy used by the high-level action term."""
    return project_root() / "best_policy" / "ll_policy" / "ll_policy.pt"


def hl_policy_path() -> Path:
    """Shipped high-level checkpoint for benchmark play."""
    return project_root() / "best_policy" / "best_policy.pt"
