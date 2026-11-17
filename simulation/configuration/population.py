# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Select the effective actor population and its independent settings source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .models import SimulationConfig


@dataclass(frozen=True)
class PopulationSelection:
    """Effective actor configuration and the reason for that selection."""

    config: SimulationConfig | None
    source: Literal["disabled", "population_override", "scenario"]


def resolve_population(config: SimulationConfig) -> PopulationSelection:
    """Select a single population override without validating unused assets.

    Args:
        config: Final session configuration after profile and command overrides.

    Returns:
        Selected population and source. A replacement's own population
        override is not followed.

    Raises:
        ValueError: If the enabled population override is missing or malformed.
    """
    if not config.pedestrians.enabled:
        return PopulationSelection(None, "disabled")
    if config.pedestrians.config_path is not None:
        from .loading import _load_simulation_config

        source = config.pedestrians.config_path
        if not source.is_file():
            raise ValueError(f"Input file not found for pedestrians.config_path: {source}")
        return PopulationSelection(_load_simulation_config(source), "population_override")
    return PopulationSelection(config, "scenario")
