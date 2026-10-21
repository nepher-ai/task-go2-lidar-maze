# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Let Play treat an existing checkpoint path as a file, not a log-directory regex.

``isaaclab.bat play`` resolves ``agent_cfg.load_checkpoint`` by scanning the
experiment log directory. When that value is already a file, return it.

Play imports that resolver before this package is loaded. Replacing it only on
the defining module leaves Play holding the original function, which then
compiles a Windows path as a regex (``bad escape \\I``) and Kit shuts down.
"""

from __future__ import annotations

import os
import sys


def prefer_checkpoint_file() -> None:
    """Install the file-path shortcut on Isaac Lab's checkpoint resolver.

    Also replaces the function on every module that already imported it, which
    is how ``play`` binds the resolver before task registration runs.
    """
    import isaaclab_tasks.utils.parse_cfg as parse_cfg

    current = parse_cfg.get_checkpoint_path
    if getattr(current, "_go2_file_shortcut", False):
        return

    def get_checkpoint_path(log_path, run_dir=".*", checkpoint=".*", *args, **kwargs):
        if isinstance(checkpoint, str) and os.path.isabs(checkpoint):
            if os.path.isfile(checkpoint):
                return checkpoint
            raise FileNotFoundError(f"Checkpoint file not found: '{checkpoint}'.")
        return current(log_path, run_dir, checkpoint, *args, **kwargs)

    get_checkpoint_path._go2_file_shortcut = True  # type: ignore[attr-defined]
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "get_checkpoint_path", None) is current:
            module.get_checkpoint_path = get_checkpoint_path
