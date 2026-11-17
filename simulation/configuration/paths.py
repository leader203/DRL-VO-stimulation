# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Packaged resources and path resolution for simulation configurations."""

from pathlib import Path

SIMULATION_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = SIMULATION_ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_ROOT = SIMULATION_ROOT / "configs"
ASSETS_ROOT = PACKAGE_ROOT / "assets"
ROS2_NAVIGATION_SETTINGS_PATH = CONFIG_ROOT / "runtime/ros2_navigation.yaml"
ROS2_KIT_EXPERIENCE_PATH = CONFIG_ROOT / "runtime/ros2_app.kit"


def resolve_config_path(path: Path) -> Path:
    """Resolve an explicit local input without guessing a replacement configuration.

    Args:
        path: Local input path, relative to the current working directory.

    Returns:
        Absolute path. The caller validates that the required input exists.

    Raises:
        ValueError: If home expansion or symlink resolution fails.
    """
    try:
        return path.expanduser().resolve()
    except RuntimeError as error:
        raise ValueError(f"Cannot resolve configuration path {path}: {error}") from error


def resolve_simulation_profile(path: Path) -> Path:
    """Resolve an explicit profile or select its unambiguous project-directory entry.

    Args:
        path: Local configuration file or project directory.

    Returns:
        Absolute profile path. Explicit files retain ordinary path resolution.

    Raises:
        ValueError: If a directory contains both standard profile names, neither
            name, or a selected entry that is not a regular file.
    """
    path = resolve_config_path(path)
    if not path.is_dir():
        return path
    names = ("simulation.yaml", "simulation.toml")
    candidates = [path / name for name in names if (path / name).exists() or (path / name).is_symlink()]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous simulation project {path}: multiple profiles found; select a file explicitly.")
    if not candidates:
        raise ValueError(f"No simulation profile in {path}; expected simulation.yaml or simulation.toml.")
    selected = candidates[0]
    if not selected.is_file():
        raise ValueError(f"Simulation profile must be a regular file: {selected}")
    return resolve_config_path(selected)


def resolve_output_path(path: Path) -> Path:
    """Resolve an output's parent while preserving its final filesystem entry.

    Args:
        path: Local output path, relative to the current working directory.

    Returns:
        Absolute destination whose final name can still be checked for existing
        files or dangling symlinks before writing.

    Raises:
        ValueError: If home expansion or parent symlink resolution fails.
    """
    try:
        expanded = path.expanduser()
        return expanded.parent.resolve() / expanded.name
    except RuntimeError as error:
        raise ValueError(f"Cannot resolve output path {path}: {error}") from error


def resolve_config_reference(value: object, *, source: Path, field: str, output: bool = False) -> Path:
    """Resolve a local reference against the configuration file that declares it.

    Args:
        value: Nonempty filesystem-path string, with optional user-home expansion.
        source: Configuration file containing the reference.
        field: Field name included in validation diagnostics.
        output: Preserve the final filesystem entry for output collision checks.

    Returns:
        Absolute path, independent of the process working directory.

    Raises:
        ValueError: If the value is not a nonempty local path or cannot be resolved.
    """
    source = resolve_config_path(source)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} must be a nonempty local filesystem path.")
    if "://" in value:
        raise ValueError(f"{source}: {field} requires a local resource; remote URLs are not supported.")
    try:
        reference = Path(value).expanduser()
        target = reference if reference.is_absolute() else source.parent / reference
        return resolve_output_path(target) if output else resolve_config_path(target)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"{source}: {field} cannot resolve filesystem path: {error}") from error
