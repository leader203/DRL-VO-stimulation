# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Readable YAML scenarios, ROS 2 resources and immutable actor plans."""

from importlib import import_module

from .models import PedestrianPlan as PedestrianPlan
from .models import Waypoint as Waypoint

__all__ = ["PedestrianPlan", "Waypoint", "compile_scenario_config", "load_scenario_config"]


def __getattr__(name: str):
    if name not in ("compile_scenario_config", "load_scenario_config"):
        raise AttributeError(name)
    module = ".compilation" if name == "compile_scenario_config" else ".loading"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
