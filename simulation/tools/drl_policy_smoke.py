#!/usr/bin/env python3
"""Load the frozen DRL-VO checkpoint and run a deterministic forward pass."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from pedstrain_simulation.simulation.ros2_navigation.drl.package import DrlPackage
from pedstrain_simulation.simulation.ros2_navigation.drl.policy import FrozenDrlVoPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Default: <package-root>/frozen_policy.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    package = DrlPackage(args.package_root, args.checkpoint or args.package_root / "frozen_policy.pt")
    policy = FrozenDrlVoPolicy(package, device=args.device, expected_sha256=args.expected_sha256)
    print(f"checkpoint={package.checkpoint}")
    print(f"sha256={policy.checkpoint_sha256}")
    print(f"parameters={policy.parameter_count}")
    observation = np.zeros(19202, dtype=np.float32)
    for _ in range(3):
        start = time.perf_counter()
        action, metadata = policy.infer(observation)
        elapsed = (time.perf_counter() - start) * 1000.0
        print(f"action={action.tolist()} value={metadata['value']} inference_ms={elapsed:.2f}")
    policy.close()


if __name__ == "__main__":
    main()
