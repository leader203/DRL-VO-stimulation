# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keep historical module imports working after functional package organization."""

import sys
from importlib import import_module
from types import ModuleType

_MODULE_ALIASES = {
    "config": "configuration",
    "launcher": "cli.launcher",
    "paths": "configuration.paths",
    "runner": "runtime.runner",
    "configuration.inspection": "projects.inspection",
    "configuration.projects": "projects.creation",
    "configuration.yaml": "configuration.loading",
    "runtime.recording": "recording",
}


def install_module_aliases(package_name: str) -> dict[str, ModuleType]:
    """Register SDK-free legacy modules as the same objects as their replacements.

    Returning the modules also lets the package preserve attribute-style imports
    and monkeypatching without duplicating modules or installing import hooks.
    """
    aliases = {}
    for previous, current in _MODULE_ALIASES.items():
        module = import_module(f"{package_name}.{current}")
        sys.modules[f"{package_name}.{previous}"] = module
        parent, separator, attribute = previous.rpartition(".")
        if separator:
            setattr(import_module(f"{package_name}.{parent}"), attribute, module)
        else:
            aliases[previous] = module
    return aliases
