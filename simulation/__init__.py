# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent simulation APIs, safe to import before Isaac Sim starts."""

from importlib import import_module
from typing import TYPE_CHECKING

from .compatibility import install_module_aliases as _install_module_aliases

if TYPE_CHECKING:
    from .configuration import (
        PedestrianConfig,
        PedestrianPlan,
        RecordingSettings,
        RobotConfig,
        SceneConfig,
        SimulationConfig,
        SimulationSettings,
        Waypoint,
        load_scenario_config,
        load_simulation_config,
    )
    from .projects import (
        create_simulation_project,
        inspect_config_dependencies,
        inspect_simulation_config,
        run_simulation_workflow,
    )
    from .runtime import run_simulation

_EXPORTS = {
    "NavigationConfig": "configuration.navigation",
    "Ros2NavigationSettings": "configuration.navigation",
    "load_ros2_navigation_settings": "configuration.navigation",
    "Ros2SimulationSession": "ros2_navigation.runtime",
    "PedestrianConfig": "configuration",
    "PedestrianPlan": "configuration",
    "Waypoint": "configuration",
    "RecordingSettings": "configuration",
    "RobotConfig": "configuration",
    "SceneConfig": "configuration",
    "SimulationConfig": "configuration",
    "SimulationSettings": "configuration",
    "load_simulation_config": "configuration",
    "load_scenario_config": "configuration",
    "run_simulation": "runtime.runner",
    "create_simulation_project": "projects",
    "inspect_config_dependencies": "projects",
    "inspect_simulation_config": "projects",
    "run_simulation_workflow": "projects",
}
__all__ = list(_EXPORTS)

globals().update(_install_module_aliases(__name__))


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
