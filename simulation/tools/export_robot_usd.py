# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bake the EVT robot into a standalone USD that Isaac Lab can clone per environment.

``simulation/EVT-ROS.usd`` is a whole *stage*: it carries the robot under
``/EVT_3`` together with a ``/PhysicsScene``, a ``/Render`` scope and an
``/EVT_3/GroundPlane`` collider. Spawning it once per environment would clone the
physics scene and the 50 m ground plane with it, so this tool flattens the stage
and copies only the robot side of the layer stack:

* ``/EVT_3`` — the articulation, its visuals, materials and joint drives;
* every ``/Flattened_Prototype_*`` prim — ``Flatten()`` keeps instanced geometry
  as internal references to those prototypes, so copying ``/EVT_3`` alone leaves
  the wheels and gripper pointing at nothing;
* ``/Render`` and ``/PhysicsScene`` are dropped, and ``/EVT_3/GroundPlane`` and
  ``/EVT_3/ROS_cmd`` are deactivated: none of them may be duplicated per
  environment.

Run once, and again whenever the source asset changes::

    OMNI_KIT_ACCEPT_EULA=YES python simulation/tools/export_robot_usd.py

The result is written under ``simulation/rl/assets/`` and is what
:mod:`.vec_task` spawns for every environment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

SIMULATION = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = SIMULATION / "EVT-ROS.usd"
DEFAULT_OUTPUT = SIMULATION / "rl" / "assets" / "evt_robot.usd"
ROBOT_ROOT = "/EVT_3"
# Stage-level prims that must not follow the robot into a cloned environment.
SKIP_ROOTS = ("/Render", "/PhysicsScene")
# Robot-level prims that are only valid once: a 50x50 m collision plane and a ROS
# 2 command graph per robot.
DEACTIVATE = ("GroundPlane", "ROS_cmd")


def export(source: Path, output: Path) -> Path:
    """Copy the robot subtree of ``source`` into a fresh, self-contained layer.

    Args:
        source: Stage that references the robot, normally ``simulation/EVT-ROS.usd``.
        output: Layer to write.

    Returns:
        The written path.
    """
    from pxr import Sdf, Usd

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise RuntimeError(f"Cannot open {source}")
    robot = stage.GetPrimAtPath(ROBOT_ROOT)
    if not robot or not robot.IsValid():
        raise RuntimeError(f"{source} has no {ROBOT_ROOT} prim")

    flattened = stage.Flatten()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    target = Usd.Stage.CreateNew(str(output))
    root = target.GetRootLayer()

    copied = []
    # ``Flatten()`` returns a ``Sdf.Layer``: its root specs are the layer's
    # ``pseudoRoot`` prim specs, which is where the prototype siblings live.
    for spec in flattened.pseudoRoot.nameChildren:
        path = str(spec.path)
        if path in SKIP_ROOTS:
            continue
        Sdf.CopySpec(flattened, path, root, path)
        copied.append(path)

    target.SetDefaultPrim(target.GetPrimAtPath(ROBOT_ROOT))
    for key in ("metersPerUnit", "upAxis"):
        value = stage.GetMetadata(key)
        if value is not None:
            target.SetMetadata(key, value)
    for name in DEACTIVATE:
        prim = target.GetPrimAtPath(f"{ROBOT_ROOT}/{name}")
        if prim and prim.IsValid():
            prim.SetActive(False)
    root.Save()
    target = None

    verify(output)
    print(f"[export] {output} ({output.stat().st_size / 1e6:.1f} MB) copied={copied}", flush=True)
    return output


def verify(path: Path) -> None:
    """Reopen the written layer and prove the internal references still resolve.

    A flattened layer keeps instanced geometry as references to
    ``/Flattened_Prototype_*`` prims in the same file; losing one leaves an empty
    instance that only fails much later, inside PhysX. The check re-opens the file
    and requires every link the source has.

    Args:
        path: USD file written by :func:`export`.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Cannot reopen {path}")
    roots = [str(prim.GetPath()) for prim in stage.GetPseudoRoot().GetChildren()]
    if ROBOT_ROOT not in roots:
        raise RuntimeError(f"{path} lost {ROBOT_ROOT}: {roots}")
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    articulations = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    children = [child.GetName() for child in stage.GetPrimAtPath(f"{ROBOT_ROOT}/base_link").GetChildren()]
    if len(bodies) < 40 or len(joints) < 30 or not articulations:
        raise RuntimeError(
            f"{path} is incomplete: {len(bodies)} rigid bodies, {len(joints)} joints, "
            f"{len(articulations)} articulation roots"
        )
    if not children:
        raise RuntimeError(f"{path}: {ROBOT_ROOT}/base_link has no children, the instance did not resolve")
    print(
        f"[export] verified: roots={roots} bodies={len(bodies)} joints={len(joints)} "
        f"articulations={[str(p.GetPath()) for p in articulations]}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.source, args.output)


if __name__ == "__main__":
    main()
