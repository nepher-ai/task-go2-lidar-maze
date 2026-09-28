# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Let Play treat an existing checkpoint path as a file, not a log-directory regex.

``isaaclab.bat play`` resolves ``agent_cfg.load_checkpoint`` by scanning the
experiment log directory. The maze Play config sets that field to
``best_policy/best_policy.pt``. When the value is already a file, return it.
"""

from __future__ import annotations

import os


def prefer_checkpoint_file() -> None:
    """Install the file-path shortcut on Isaac Lab's checkpoint resolver."""
    import isaaclab_tasks.utils.parse_cfg as parse_cfg

    current = parse_cfg.get_checkpoint_path
    if getattr(current, "_go2_file_shortcut", False):
        return

    def get_checkpoint_path(log_path, run_dir=".*", checkpoint=".*", *args, **kwargs):
        if isinstance(checkpoint, str) and os.path.isfile(checkpoint):
            return os.path.abspath(checkpoint)
        return current(log_path, run_dir, checkpoint, *args, **kwargs)

    get_checkpoint_path._go2_file_shortcut = True  # type: ignore[attr-defined]
    parse_cfg.get_checkpoint_path = get_checkpoint_path
