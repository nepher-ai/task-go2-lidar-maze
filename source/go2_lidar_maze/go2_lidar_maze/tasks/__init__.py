# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register the Go2 LiDAR maze tasks. Imported via the isaaclab.tasks entry point."""

from go2_lidar_maze.tasks.checkpoint import prefer_checkpoint_file

from . import hl_maze, ll_velocity  # noqa: F401

prefer_checkpoint_file()
