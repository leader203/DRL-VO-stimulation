"""Work around the ``joint_acc`` defect in ``isaaclab_physx`` 3.0.0.

``ArticulationData.update`` touches ``self.joint_acc`` on every physics step only
to advance a finite-difference buffer at a higher rate. That accessor launches the
``get_joint_acc_from_joint_vel`` Warp kernel, which faults with CUDA error 700
(illegal memory access) under Isaac Sim 6.2, so no environment can be instantiated.

The DRL-VO task never reads joint acceleration — rewards, observations and the
differential-wheel action all work from root pose/velocity and joint positions —
so this script drops the eager refresh. The property still computes on demand for
anyone who genuinely wants it.

Run it with the interpreter of the affected environment::

    python simulation/tools/patch_isaaclab_physx.py [--revert]

Idempotent; a ``.orig`` backup is written once.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

BROKEN = """        # update the simulation timestamp
        self._sim_timestamp += dt
        # Trigger an update of the joint acceleration buffer at a higher frequency
        # since we do finite differencing.
        self.joint_acc
"""

PATCHED = """        # update the simulation timestamp
        self._sim_timestamp += dt
        # Upstream touches ``self.joint_acc`` here to advance the finite-difference
        # buffer at a higher rate. That accessor launches the
        # ``get_joint_acc_from_joint_vel`` Warp kernel, which faults with CUDA error
        # 700 under Isaac Sim 6.2, so the refresh is left to the first real read.
"""


def locate() -> Path:
    """Return the ``articulation_data.py`` of the installed ``isaaclab_physx``."""
    spec = importlib.util.find_spec("isaaclab_physx")
    if spec is None or not spec.origin:
        raise SystemExit("[patch] isaaclab_physx is not importable from this interpreter")
    root = Path(spec.origin).parent
    target = root / "assets" / "articulation" / "articulation_data.py"
    if not target.is_file():
        raise SystemExit(f"[patch] missing {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revert", action="store_true", help="Restore the backup instead of patching.")
    args = parser.parse_args()

    target = locate()
    backup = target.with_suffix(".py.orig")

    if args.revert:
        if not backup.exists():
            raise SystemExit(f"[patch] no backup at {backup}")
        shutil.copy2(backup, target)
        print(f"[patch] restored {target}")
        return 0

    source = target.read_text(encoding="utf-8")
    if PATCHED in source:
        print(f"[patch] already patched: {target}")
        return 0
    if BROKEN not in source:
        raise SystemExit(f"[patch] anchor not found in {target}; isaaclab_physx changed upstream")

    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"[patch] backup written to {backup}")
    target.write_text(source.replace(BROKEN, PATCHED, 1), encoding="utf-8")
    compile(target.read_text(encoding="utf-8"), str(target), "exec")
    print(f"[patch] patched {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
