# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the native simulation module with ``python -m pedstrain_simulation.simulation``."""

from .cli.launcher import main

if __name__ == "__main__":
    main()
