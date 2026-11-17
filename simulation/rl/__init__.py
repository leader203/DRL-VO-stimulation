# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional single-world ROS 2 integration for Isaac Lab training."""

from importlib import import_module

_EXPORTS = {
    "Ros2TrainingScene": "scene",
    "Ros2PedestrianBridge": "bridge",
    "PedestrianState": "bridge",
    "Ros2PedestrianActionCfg": "isaaclab",
    "pedestrian_observations": "observations",
    "pedestrian_clearance": "mdp",
    "pedestrian_collision": "mdp",
    "pedestrian_proximity_cost": "mdp",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
