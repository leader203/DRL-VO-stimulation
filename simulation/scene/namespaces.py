# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select writable runtime namespaces without assuming an environment root."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Sdf, Usd


def select_runtime_prim_path(stage: Usd.Stage, name: str, *, unique: bool = False) -> Sdf.Path:
    """Choose a temporary prim destination beneath an editable scene root.

    Args:
        stage: Stage receiving session-layer runtime objects.
        name: Single USD prim identifier for the runtime object.
        unique: Append a numeric suffix until the destination is unused.
            False preserves the destination so callers can reject collisions.

    Returns:
        Absolute path below an active, defined, loaded non-instance default
        prim, an existing compatible World root, or the pseudo-root fallback.
        This function does not author any USD data.
    """
    from pxr import Sdf

    if not isinstance(name, str) or not Sdf.Path.IsValidIdentifier(name):
        raise ValueError("Runtime prim name must be a single valid USD identifier.")
    parent = Sdf.Path.absoluteRootPath
    for root in (stage.GetDefaultPrim(), stage.GetPrimAtPath("/World")):
        if _editable_parent(root):
            parent = root.GetPath()
            break
    path = parent.AppendChild(name)
    if unique:
        index = 1
        while stage.GetPrimAtPath(path):
            path = parent.AppendChild(f"{name}_{index}")
            index += 1
    return path


def _editable_parent(prim: Usd.Prim) -> bool:
    return bool(
        prim
        and prim.IsActive()
        and prim.IsDefined()
        and prim.IsLoaded()
        and not prim.IsAbstract()
        and not prim.IsInstance()
        and not prim.IsInstanceProxy()
        and not prim.IsPrototype()
        and not prim.IsInPrototype()
    )
