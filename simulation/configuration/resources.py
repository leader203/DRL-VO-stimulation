# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve simulation resources without loading the SDK or contacting servers."""

from pathlib import Path
from urllib.parse import unquote, urlsplit

DEFAULT_MOTION_LIBRARY = "Isaac/People/MotionLibrary/HumanMotionLibrary.usd"


def _local_asset_root(value: str | None) -> Path | None:
    if value and "://" in value:
        uri = urlsplit(value)
        value = (
            unquote(uri.path)
            if uri.scheme == "file" and uri.netloc in ("", "localhost") and uri.path.startswith("/")
            else None
        )
    return Path(value).expanduser() if value else None


def resolve_ros2_motion_library(value: object, asset_root: str | None) -> Path:
    """Resolve the ROS 2 controller's required local motion library.

    Args:
        value: Compiled local path or the default SDK motion-library token.
        asset_root: Local path or file URI used for the default token.

    Returns:
        Absolute path to the existing local library file.

    Raises:
        ValueError: If the selected library cannot be loaded from a local file.
    """
    if not isinstance(value, str) or not value or "://" in value:
        raise ValueError("ROS character animation requires a local motion_library path.")
    motion = Path(value).expanduser()
    if value == DEFAULT_MOTION_LIBRARY:
        root = _local_asset_root(asset_root)
        if root is None:
            raise ValueError("ROS character animation requires a local asset_root or an explicit local motion_library.")
        motion = root / motion
    if not motion.is_file():
        raise ValueError(f"Motion library not found: {motion}")
    return motion.resolve()
