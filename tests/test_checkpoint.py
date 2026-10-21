# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play must resolve the shipped checkpoint even though it imports the resolver first."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "source" / "go2_lidar_maze"


def test_play_import_order_resolves_an_absolute_checkpoint(tmp_path: Path) -> None:
    """A binding taken before task import still returns the checkpoint file."""
    pytest.importorskip("isaaclab_tasks")
    checkpoint = tmp_path / "best_policy.pt"
    checkpoint.write_bytes(b"pt")
    script = tmp_path / "resolve.py"
    script.write_text(
        textwrap.dedent(
            f"""
            from isaaclab_tasks.utils import get_checkpoint_path
            import go2_lidar_maze.tasks
            print("CHECKPOINT=" + get_checkpoint_path("unused-log-dir", ".*", {str(checkpoint)!r}))
            """
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SOURCE) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"CHECKPOINT={checkpoint}" in result.stdout
