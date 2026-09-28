# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read the maze start and goal. The A* path line in the file is never used."""

from __future__ import annotations

from pathlib import Path

_KEYS = ("start_x", "start_y", "end_x", "end_y")


def parse_start_goal(text: str) -> tuple[float, float, float, float]:
    """Return ``(start_x, start_y, end_x, end_y)`` in metres.

    Only those four keys are read. A ``path:`` line, if present, is ignored.
    """
    found: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key in _KEYS:
            found[key] = float(value.strip())
    missing = [key for key in _KEYS if key not in found]
    if missing:
        raise ValueError(f"Maze metadata is missing {missing}.")
    return tuple(found[key] for key in _KEYS)  # type: ignore[return-value]


def load_start_goal(path: str | Path) -> tuple[float, float, float, float]:
    """Parse start and goal coordinates from a maze ``.txt`` file."""
    return parse_start_goal(Path(path).read_text(encoding="utf-8"))
