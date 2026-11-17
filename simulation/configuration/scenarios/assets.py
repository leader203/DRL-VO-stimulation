# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local character discovery and filtering by explicit asset-name labels, without SDK imports."""

import re
from pathlib import Path

from ..resources import _local_asset_root
from .models import PedestrianGender, _gender

_USD_SUFFIXES = (".usd", ".usda")


def character_catalog_files(source: Path, asset_root: str | None) -> tuple[Path, ...]:
    """Return a catalog's ordered local USD files, or an empty tuple for other sources."""
    if source.suffix.lower() not in (".yaml", ".yml") or not source.is_file():
        return ()
    from ..loading import load_yaml_mapping

    root = _local_asset_root(asset_root)
    if root is None:
        raise ValueError(f"{source}: character catalogs require a local navigation.asset_root.")
    data = load_yaml_mapping(source)
    entries = data.get("assets")
    if data.keys() != {"assets"} or not isinstance(entries, list) or not entries:
        raise ValueError(f"{source}: expected one nonempty assets list.")
    files: list[Path] = []
    seen: set[Path] = set()
    for entry in entries:
        if not isinstance(entry, str) or not entry or "://" in entry:
            raise ValueError(f"{source}: each asset must be a relative USD/USDA path.")
        path = Path(entry)
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in _USD_SUFFIXES:
            raise ValueError(f"{source}: expected a relative USD/USDA path under asset_root: {entry}")
        asset = (root / path).absolute()
        if not asset.is_file():
            raise ValueError(f"{source}: character asset not found: {asset}")
        resolved = asset.resolve()
        if resolved in seen:
            raise ValueError(f"{source}: duplicate character asset: {entry}")
        seen.add(resolved)
        files.append(asset)
    return tuple(files)


def find_character_assets(
    source: Path, gender: PedestrianGender = "any", *, asset_root: str | None = None
) -> tuple[Path, ...]:
    """Find character entry files matching an optional asset gender label.

    Args:
        source: Local USD file, directory, or YAML asset catalog. Direct USD files take precedence;
            otherwise one entry file per immediate non-hidden child folder is used.
        gender: ``male`` or ``female`` selects labelled assets; ``any`` includes all.
        asset_root: Local directory or file URI for catalog-relative USD paths.

    Returns:
        Deterministically ordered absolute USD/USDA paths. No USD data is loaded.

    Raises:
        ValueError: If no matching entry exists or the requested label is invalid.

    Labels come only from filename/direct-parent tokens ``male``/``female`` or
    ``M_``/``F_`` prefixes. Unlabelled or conflicting entries cannot satisfy a
    specific gender. This does not infer identity or alter an asset's appearance.
    """
    _gender(gender, "gender")
    source = source.expanduser().resolve()

    def entries(directory: Path) -> list[Path]:
        return sorted(
            path
            for path in directory.iterdir()
            if not path.name.startswith(".") and path.is_file() and path.suffix.lower() in _USD_SUFFIXES
        )

    candidates = []
    if source.is_file():
        candidates = (
            (source,) if source.suffix.lower() in _USD_SUFFIXES else character_catalog_files(source, asset_root)
        )
    elif source.is_dir():
        candidates = entries(source)
        if not candidates:
            for folder in sorted(source.iterdir()):
                if folder.is_dir() and not folder.name.startswith("."):
                    candidates.extend(entries(folder)[:1])
    matches = tuple(path for path in candidates if gender == "any" or _asset_gender(path) == gender)
    if not matches:
        raise ValueError(
            f"No character assets match gender={gender!r} in {source}. "
            "Use male_/female_ or M_/F_ file/folder labels, or gender: any for unlabelled assets."
        )
    return matches


def _asset_gender(path: Path) -> str | None:
    labels = set()
    for name in (path.stem, path.parent.name):
        tokens = re.split(r"[^a-z0-9]+", name.casefold())
        labels.update(token for token in tokens if token in ("male", "female"))
        if tokens[0] in ("m", "f"):
            labels.add("male" if tokens[0] == "m" else "female")
    return next(iter(labels)) if len(labels) == 1 else None
