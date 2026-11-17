# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compose robot assets before native IRA loads actors and bakes the NavMesh."""

import math
from pathlib import Path

from ..configuration.models import SimulationConfig


def create_runtime_layer(config: SimulationConfig, base_stage: str, directory: Path) -> Path | None:
    """Author a session-owned USD layer with robot references and initial poses.

    Args:
        config: Scene layers and robots to compose.
        base_stage: Resolved environment USD used for length/mass units and up-axis.
        directory: Temporary directory retained until the application closes.

    Returns:
        Runtime layer path, or None when no additional composition is requested.

    Raises:
        ValueError: If referenced robot metrics or authored overlay metrics differ
            from the environment. Convert an asset copy before using it here.
    """
    if not config.robots and not config.scene.layers:
        return None
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    base = Usd.Stage.Open(base_stage, load=Usd.Stage.LoadNone)
    if not base:
        raise RuntimeError(f"Cannot open environment: {base_stage}")
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(base)
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("Scene composition requires finite positive stage metersPerUnit.")
    kilograms_per_unit = UsdPhysics.GetStageKilogramsPerUnit(base)
    if not math.isfinite(kilograms_per_unit) or kilograms_per_unit <= 0:
        raise ValueError("Scene composition requires finite positive stage kilogramsPerUnit.")
    up_axis = UsdGeom.GetStageUpAxis(base)
    if up_axis not in ("Y", "Z"):
        raise ValueError("Scene composition requires a supported Y or Z upAxis.")
    sources = [(path, False) for path in config.scene.layers]
    sources.extend(
        (robot.asset_path if robot.asset_path is not None else robot.layer_path, robot.asset_path is not None)
        for robot in config.robots
    )
    for path, asset in dict.fromkeys(sources):
        _validate_source_metrics(path, meters_per_unit, up_axis, kilograms_per_unit, asset=asset)
    layer = Sdf.Layer.CreateNew(str(directory / "simulation.usda"))
    layer.subLayerPaths = list(dict.fromkeys(str(path) for path in config.scene.layers))
    for robot in config.robots:
        if robot.layer_path is not None and str(robot.layer_path) not in layer.subLayerPaths:
            layer.subLayerPaths.append(str(robot.layer_path))
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdPhysics.SetStageKilogramsPerUnit(stage, kilograms_per_unit)
    UsdGeom.SetStageUpAxis(stage, up_axis)
    stage.SetTimeCodesPerSecond(base.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(base.GetFramesPerSecond())
    for robot in config.robots:
        if robot.asset_path is not None:
            if base.GetPrimAtPath(robot.prim_path) or stage.GetPrimAtPath(robot.prim_path):
                raise ValueError(f"Robot destination already exists: {robot.prim_path}")
            prim = stage.DefinePrim(robot.prim_path, "Xform")
            if not prim.GetReferences().AddReference(str(robot.asset_path), robot.source_prim_path or ""):
                raise ValueError(f"Cannot reference robot: {robot.asset_path}")
        else:
            prim = stage.GetPrimAtPath(robot.prim_path)
            if not prim or not prim.IsActive() or not prim.IsDefined():
                raise ValueError(f"Robot layer has no active robot at {robot.prim_path}")
        if robot.position is not None or robot.yaw_degrees is not None:
            xform = UsdGeom.Xformable(prim)
            if not xform:
                raise ValueError(f"Robot prim is not transformable: {robot.prim_path}")
            transform = Gf.Transform(xform.GetLocalTransformation())
            if robot.position is not None:
                position = tuple(value / meters_per_unit for value in robot.position)
                if not all(math.isfinite(value) for value in position):
                    raise ValueError("Robot position must remain finite after conversion to stage units.")
                transform.SetTranslation(Gf.Vec3d(*position))
            if robot.yaw_degrees is not None:
                axis = Gf.Vec3d(0, 0, 1) if up_axis == "Z" else Gf.Vec3d(0, 1, 0)
                transform.SetRotation(Gf.Rotation(axis, robot.yaw_degrees))
            xform.MakeMatrixXform().Set(transform.GetMatrix())
    layer.Save()
    return Path(layer.realPath)


def _validate_source_metrics(path: Path, units: float, up_axis: str, mass_units: float, *, asset: bool) -> None:
    """Reject incompatible source metrics before composing unconverted physics.

    Complete assets use USD's effective metadata, including fallback values.
    An overlay without authored metrics is expressed in its destination frame.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    source = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    if not source:
        raise ValueError(f"Cannot inspect composition units: {path}")
    source_units = UsdGeom.GetStageMetersPerUnit(source)
    if (asset or source.HasAuthoredMetadata("metersPerUnit")) and (
        not math.isfinite(source_units)
        or source_units <= 0
        or not math.isclose(source_units, units, rel_tol=1e-9, abs_tol=0.0)
    ):
        raise ValueError(
            f"Composition metersPerUnit mismatch at {path}: source={source_units}, destination={units}. "
            "Convert a separate asset/layer to the destination units before referencing it."
        )
    source_axis = UsdGeom.GetStageUpAxis(source)
    if (asset or source.HasAuthoredMetadata("upAxis")) and source_axis != up_axis:
        raise ValueError(
            f"Composition upAxis mismatch at {path}: source={source_axis}, destination={up_axis}. "
            "Convert a separate asset/layer to the destination axis before referencing it."
        )
    source_mass_units = UsdPhysics.GetStageKilogramsPerUnit(source)
    if (asset or source.HasAuthoredMetadata("kilogramsPerUnit")) and (
        not math.isfinite(source_mass_units)
        or source_mass_units <= 0
        or not math.isclose(source_mass_units, mass_units, rel_tol=1e-9, abs_tol=0.0)
    ):
        raise ValueError(
            f"Composition kilogramsPerUnit mismatch at {path}: source={source_mass_units}, destination={mass_units}. "
            "Convert a separate asset/layer to the destination mass units before referencing it."
        )
