# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bake the training room into a USD that Isaac Lab can clone per environment.

``simulation/assets/THIRD_EXP_1.usd`` is authored as a *stage*: ``/World`` holds the
room **and** a ``PhysicsScene``. Isaac Lab's scene owns the physics scene, and with
``replicate_physics=True`` PhysX refuses to clone one ("Replication of this type is
not supported ... /Room/PhysicsScene"), so the room is rewritten without it::

    OMNI_KIT_ACCEPT_EULA=YES python simulation/tools/bake_room_usd.py

Everything else under ``/World`` is copied verbatim, including the instanced
geometry, whose ``Flattened_Prototype_*`` prims are copied as well.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SIMULATION = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = SIMULATION / "assets" / "THIRD_EXP_1.usd"
DEFAULT_OUTPUT = SIMULATION / "rl" / "assets" / "third_room.usd"
WORLD = "/World"
# The clone must not carry a physics scene: Isaac Lab creates the single one.
DROP_CHILDREN = ("PhysicsScene",)


def bake(source: Path, output: Path) -> Path:
    """Write a cloneable copy of the room.

    Args:
        source: Room asset that may contain a physics scene.
        output: Layer to write.

    Returns:
        The written path.
    """
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Cannot open {source}")
    if not stage.GetPrimAtPath(WORLD):
        raise RuntimeError(f"{source} has no {WORLD} prim")

    flattened = stage.Flatten()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    target = Usd.Stage.CreateNew(str(output))
    root = target.GetRootLayer()

    copied = []
    for spec in flattened.pseudoRoot.nameChildren:
        path = str(spec.path)
        Sdf.CopySpec(flattened, path, root, path)
        copied.append(path)

    # ``Sdf.Layer`` has no prim removal API, and a deactivated prim is neither
    # traversed nor replicated, which is exactly what is needed here.
    for name in DROP_CHILDREN:
        child = target.GetPrimAtPath(f"{WORLD}/{name}")
        if child and child.IsValid():
            child.SetActive(False)

    target.SetDefaultPrim(target.GetPrimAtPath(WORLD))
    for key in ("metersPerUnit", "upAxis"):
        value = stage.GetMetadata(key)
        if value is not None:
            target.SetMetadata(key, value)
    root.Save()
    target = None

    verify(output)
    print(
        f"[bake] {output} ({output.stat().st_size / 1e6:.1f} MB) floor_children={len(copied)}",
        flush=True,
    )
    return output


def verify(path: Path) -> None:
    """Reopen the baked room and prove it still has colliders and no physics scene.

    Args:
        path: Layer written by :func:`bake`.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Cannot reopen {path}")
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if scenes:
        raise RuntimeError(f"{path} still carries {len(scenes)} physics scenes: {[str(p.GetPath()) for p in scenes]}")
    if len(colliders) < 100 or len(meshes) < 100:
        raise RuntimeError(f"{path} looks incomplete: {len(colliders)} colliders, {len(meshes)} meshes")
    print(f"[bake] verified: {len(colliders)} colliders, {len(meshes)} meshes, no physics scene", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    bake(args.source, args.output)


if __name__ == "__main__":
    main()
