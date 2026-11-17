# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native physics overrides and post-load robot validation."""

from ..configuration.models import SimulationConfig, SimulationSettings
from .namespaces import select_runtime_prim_path


def configure_physics(settings: SimulationSettings) -> None:
    """Apply requested physics overrides while preserving unspecified scene settings."""
    import omni.usd
    from pxr import PhysxSchema, Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if not stage:
        raise RuntimeError("IRA did not open the environment stage.")
    scenes = [prim for prim in stage.Traverse(Usd.TraverseInstanceProxies()) if prim.IsA(UsdPhysics.Scene)]
    if len(scenes) > 1:
        raise ValueError("Multiple physics scenes found; reference the robot model root, not its preview scene.")
    if settings.physics_hz is None and settings.solver is None:
        return
    if scenes and scenes[0].IsInstanceProxy():
        raise ValueError("Cannot override a physics scene inside an instance; provide an editable scene layer.")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        scene = (
            scenes[0]
            if scenes
            else UsdPhysics.Scene.Define(stage, select_runtime_prim_path(stage, "PhysicsScene", unique=True)).GetPrim()
        )
        api = PhysxSchema.PhysxSceneAPI.Apply(scene)
        if settings.physics_hz is not None:
            api.CreateTimeStepsPerSecondAttr(settings.physics_hz)
        if settings.solver is not None:
            api.CreateSolverTypeAttr(settings.solver)


def validate_robots(config: SimulationConfig) -> None:
    """Require enabled rigid bodies, including instance proxies, at robot destinations."""
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    if not stage:
        raise RuntimeError("IRA did not open the environment stage.")
    for robot in config.robots:
        prim = stage.GetPrimAtPath(robot.prim_path)
        if (
            not prim
            or not prim.IsActive()
            or not prim.IsDefined()
            or not any(
                child.HasAPI(UsdPhysics.RigidBodyAPI)
                and UsdPhysics.RigidBodyAPI(child).GetRigidBodyEnabledAttr().Get() is not False
                for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
            )
        ):
            raise ValueError(f"Robot has no active rigid bodies: {robot.prim_path}")
