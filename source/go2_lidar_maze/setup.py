# Copyright (c) 2026, Nepher Robotics
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'go2_lidar_maze' python package."""

import os
import tomllib

from setuptools import find_packages, setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(EXTENSION_PATH, "config", "extension.toml"), "rb") as _ext:
    EXTENSION_TOML_DATA = tomllib.load(_ext)

setup(
    name="go2_lidar_maze",
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    license="BSD-3-Clause",
    include_package_data=True,
    package_data={"go2_lidar_maze": ["assets/maze/*", "assets/policies/*"]},
    python_requires=">=3.11",
    entry_points={"isaaclab.tasks": ["go2_lidar_maze = go2_lidar_maze.tasks"]},
    zip_safe=False,
)
