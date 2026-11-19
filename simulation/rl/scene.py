# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Prepare a ROS crowd on a stage that an Isaac Lab task will step."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from ..configuration.navigation import Ros2NavigationSettings, load_ros2_navigation_settings
from ..configuration.population import resolve_population
from ..ros2_navigation.actors import Ros2Actors
from ..ros2_navigation.doors import Ros2Doors
from ..ros2_navigation.frames import StageFrame
from ..ros2_navigation.mapping import DynamicObstacleObserver, build_occupancy_grid, save_occupancy_grid
from ..ros2_navigation.planner import OccupancyGrid
from ..ros2_navigation.runtime import (
    _assets,
    _bootstrap_plans,
    _canonical_plans,
    _compose_world,
    _validate_fixed_spawn,
    _validate_walk_routes,
)
from ..ros2_navigation.solver import build_solver
from ..ros2_navigation.transport import Ros2Transport
from ..scene.namespaces import select_runtime_prim_path

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    from pxr import Usd

    from ..configuration.models import SimulationConfig
    from .bridge import Ros2PedestrianBridge

_BOUND_SCENE: ContextVar[Ros2TrainingScene | None] = ContextVar("ros2_training_scene", default=None)


def _navigation_settings(path: Path | None, *, required: bool) -> Ros2NavigationSettings:
    """Load the configured ROS settings, tolerating a missing file without a crowd.

    A pedestrian-free smoke stage still ray-casts its rewards against the map the
    deployed controller would build, so the configured resolution and clearances
    are kept; only a missing settings file falls back to the dataclass defaults.
    """
    try:
        return load_ros2_navigation_settings(path)
    except (OSError, ValueError):
        if required:
            raise
        return Ros2NavigationSettings()


def _bound_training_scene(stage: Usd.Stage) -> Ros2TrainingScene:
    prepared = _BOUND_SCENE.get()
    if prepared is None or prepared.stage != stage:
        raise RuntimeError("Construct the Isaac Lab environment inside prepared.bind() on its prepared ROS stage.")
    return prepared


class Ros2TrainingScene:
    """Own a composed USD and ROS actors while the caller owns Kit and physics.

    Args:
        config: ROS 2 scenario with explicit, fixed pedestrian spawns.
        max_pedestrians: Observation capacity, at least the selected population.
        output: Optional fresh directory for map, DDS and bootstrap diagnostics.
    """

    def __init__(
        self, config: SimulationConfig, *, max_pedestrians: int | None = None, output: Path | None = None
    ) -> None:
        config.validate()
        if config.recording.enabled:
            raise ValueError("The Isaac Lab task must own recording when using a ROS training scene.")
        self.config = config
        population = resolve_population(config).config
        self._input_plans = population.pedestrian_plans if population is not None else ()
        if any(plan.position is None for plan in self._input_plans):
            raise ValueError("ROS training requires an explicit spawn position for every pedestrian.")
        minimum = max(1, len(self._input_plans))
        capacity = minimum if max_pedestrians is None else max_pedestrians
        if type(capacity) is not int or capacity < minimum:
            raise ValueError("max_pedestrians must cover the selected population and be at least one.")
        self.max_pedestrians = capacity
        self.navigation = _navigation_settings(config.navigation.settings_path, required=population is not None)
        self._output = output.resolve() if output is not None else None
        self._temporary: Path | None = None
        self._world_path: Path | None = None
        self._stage: Usd.Stage | None = None
        self._occupancy: OccupancyGrid | None = None
        self._actors: Ros2Actors | None = None
        self._bridge: Ros2PedestrianBridge | None = None
        self._bridge_attempted = False

    def __enter__(self) -> Ros2TrainingScene:
        if self._temporary is not None or self._world_path is not None:
            raise RuntimeError("Enter a fresh ROS training scene once.")
        self._temporary = Path(tempfile.mkdtemp(prefix="ros2_pedestrian_training_"))
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note(f"ROS training scene cleanup failed: {cleanup!r}")

    @property
    def has_crowd(self) -> bool:
        """Whether this scene authored ROS pedestrian bodies on the stage."""
        return self._actors is not None

    @property
    def stage(self) -> Usd.Stage:
        """Prepared USD stage to pass to Isaac Lab's ``use_stage`` context."""
        if self._stage is None:
            raise RuntimeError("Complete ROS training scene setup before requesting its stage.")
        return self._stage

    @property
    def frame(self) -> StageFrame:
        """Canonical world to ROS map conversion of the prepared stage."""
        self.stage
        return self._frame

    def occupancy(self) -> OccupancyGrid:
        """Return the raster the bridge planned on, building it when no crowd is configured.

        The training reward ray-casts against the same static map the deployed
        controller uses, so a pedestrian-free smoke run still sees the geometry.
        """
        self.stage
        if self._occupancy is None:
            settings = self.navigation.to_dict()
            settings.update(seed=self.config.settings.seed, plans=[plan.to_dict() for plan in self._plans])
            roots = (self._actors.root_path,) if self._actors is not None else ()
            grid_data = build_occupancy_grid(self.stage, settings, self._frame, excluded_roots=roots)
            self._occupancy = OccupancyGrid(
                grid_data["occupancy"],
                grid_data["resolution_m"],
                grid_data["origin_xy_m"],
                goal_projection_tolerance_m=self.navigation.goal_projection_tolerance_m,
            )
        return self._occupancy

    async def setup(self) -> None:
        """Open the scene and author fixed pedestrian bodies before physics starts."""
        import omni.kit.app
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics

        from isaaclab.sim import SimulationContext

        if self._temporary is None or self._world_path is not None:
            raise RuntimeError("Enter a fresh training scene and call setup once.")
        if SimulationContext.instance() is not None:
            raise RuntimeError("Prepare the ROS stage before constructing Isaac Lab's SimulationContext.")
        self._world_path, _ = _compose_world(self.config, self._temporary)
        context = omni.usd.get_context()
        # Open synchronously so the stage is live when this method returns; the
        # awaited variant needs a coroutine runner that standalone Kit lacks.
        opened = context.open_stage(str(self._world_path))
        error = None
        stage = context.get_stage()
        if not opened or stage is None or stage.GetRootLayer().realPath != str(self._world_path):
            raise RuntimeError(f"Cannot open ROS training scene: {error or self._world_path}")
        if UsdGeom.GetStageUpAxis(stage) != "Z" or not math.isclose(
            UsdGeom.GetStageMetersPerUnit(stage), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("ROS pedestrian training requires a Z-up stage in meters.")
        scenes = [prim for prim in stage.Traverse(Usd.TraverseInstanceProxies()) if prim.IsA(UsdPhysics.Scene)]
        if len(scenes) > 1 or any(prim.IsInstanceProxy() for prim in scenes):
            raise ValueError("ROS pedestrian training requires one editable PhysX scene or a scene without physics.")
        self.physics_prim_path = str(
            scenes[0].GetPath() if scenes else select_runtime_prim_path(stage, "PhysicsScene", unique=True)
        )
        self._frame = StageFrame(1.0, "Z", self.navigation.origin_m, self.navigation.yaw_rad)
        self._plans = _canonical_plans(self._input_plans, self._frame, self.config.navigation.waypoint_frame)
        if self._plans:
            manager = omni.kit.app.get_app().get_extension_manager()
            if any(
                manager.is_extension_enabled(name)
                for name in ("isaacsim.replicator.agent.core", "omni.anim.navigation.core", "omni.anim.behavior.core")
            ):
                raise RuntimeError("Start Kit without IRA or native navigation extensions for ROS training.")
            manager.set_extension_enabled_immediate("omni.anim.retarget.core", True)
            if not manager.is_extension_enabled("omni.anim.retarget.core"):
                raise RuntimeError("Cannot enable the ROS pedestrian animation retarget extension.")
            sources, motion = _assets(self.config, self.navigation, self._plans)
            self._actors = Ros2Actors(
                stage,
                self._plans,
                sources,
                motion,
                str(select_runtime_prim_path(stage, "Ros2Pedestrians", unique=True)),
                self._frame,
                self.config.settings.seed or 0,
                self.navigation.body_radius_m,
                self.navigation.body_height_m,
            )
        self._stage = stage

    @contextmanager
    def bind(self) -> Iterator[None]:
        """Expose this prepared stage to the ROS action term during environment construction."""
        self.stage
        if _BOUND_SCENE.get() is not None:
            raise RuntimeError("A ROS training scene is already bound in this construction scope.")
        token = _BOUND_SCENE.set(self)
        try:
            yield
        finally:
            _BOUND_SCENE.reset(token)

    def create_bridge(self, *, physics_dt_s: float, device: str) -> Ros2PedestrianBridge:
        """Create the single-world bridge after Isaac Lab has attached PhysX.

        Args:
            physics_dt_s: Duration of each task-owned physics substep [s].
            device: Device for the fixed-size observation tensors.
        """
        from .bridge import Ros2PedestrianBridge, _physics_ratio, _TrainingPhysicsBodies

        self.stage
        if self._bridge_attempted:
            raise RuntimeError("Only one ROS training bridge may be created per prepared scene.")
        _physics_ratio(physics_dt_s, self.navigation.control_hz)
        self._bridge_attempted = True
        obstacles = doors = transport = kinematics = None
        try:
            if self._actors is not None:
                output = self._output or self._temporary / "runtime"
                output.mkdir(parents=True, exist_ok=False)
                roots = (self._actors.root_path,)
                settings = self.navigation.to_dict()
                settings.update(seed=self.config.settings.seed, plans=[plan.to_dict() for plan in self._plans])
                grid_data = build_occupancy_grid(self.stage, settings, self._frame, excluded_roots=roots)
                grid = OccupancyGrid(
                    grid_data["occupancy"],
                    grid_data["resolution_m"],
                    grid_data["origin_xy_m"],
                    goal_projection_tolerance_m=self.navigation.goal_projection_tolerance_m,
                )
                self._occupancy = grid
                kinematics = _TrainingPhysicsBodies(self.stage, self._actors)
                obstacles = DynamicObstacleObserver(
                    self.stage,
                    self._frame,
                    roots,
                    dynamic_roots=self.navigation.dynamic_roots,
                    rigid_pose_reader=partial(kinematics.read_poses, group="obstacles"),
                )
                solid = obstacles.sample()
                from ..ros2_navigation.planner import convex_footprint

                floor = self.navigation.floor_height_m
                solid = [
                    {
                        **item,
                        "footprint_xy_m": (
                            convex_footprint(item["footprint_xy_m"]) if item.get("footprint_xy_m") is not None else None
                        ),
                    }
                    for item in solid
                    if item.get("z_max_m", math.inf) >= floor
                    and item.get("z_min_m", -math.inf) <= floor + self.navigation.body_height_m
                ]
                assigned = []
                for plan in self._plans:
                    point = self._frame.world_to_ros(plan.position)
                    if abs(point[2] - floor) > plan.arrival_tolerance:
                        raise ValueError(f"{plan.name}: ROS training requires pedestrians on the mapped floor.")
                    _validate_fixed_spawn(plan, point, grid, self.navigation, solid, assigned)
                    assigned.append((plan, point))
                _validate_walk_routes(self._plans, self._frame, grid, self.navigation)
                paths = save_occupancy_grid(grid_data, output / "map")
                doors = Ros2Doors(
                    self.stage,
                    self._frame,
                    navigation_interlock=True,
                    rigid_pose_reader=partial(kinematics.read_poses, group="doors"),
                )
                settings["door_specs"] = doors.descriptors()
                bootstrap = output / "bootstrap.json"
                bootstrap.write_text(
                    json.dumps(
                        {
                            "plans": _bootstrap_plans(self._plans, self._input_plans, self._frame, {}),
                            "input_plans": [plan.to_dict() for plan in self._input_plans],
                            "world_plans": [plan.to_dict() for plan in self._plans],
                            "settings": settings,
                            "agents": self._actors.states(),
                            "map_npz": str(paths["npz"]),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                transport = Ros2Transport(
                    bootstrap, build_solver(output / "build"), settings, output, publish_clock=False
                )
            self._bridge = Ros2PedestrianBridge(
                frame=self._frame,
                navigation=self.navigation,
                actors=self._actors,
                obstacles=obstacles,
                doors=doors,
                transport=transport,
                kinematics=kinematics,
                capacity=self.max_pedestrians,
                device=device,
                physics_dt_s=physics_dt_s,
            )
        except BaseException as error:
            for resource in (transport, kinematics, obstacles, doors):
                if resource is not None:
                    try:
                        resource.close()
                    except BaseException as cleanup:
                        error.add_note(f"ROS training initialization cleanup failed: {cleanup!r}")
            raise
        return self._bridge

    def close(self) -> None:
        """Release the crowd and temporary files after the caller detaches the stage."""
        if self._temporary is None:
            return
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        if self._world_path is not None:
            import omni.usd

            current = omni.usd.get_context().get_stage()
            if current is not None and current.GetRootLayer().realPath == str(self._world_path):
                raise RuntimeError(f"Detach the ROS training stage before cleanup: {self._world_path}")
        if self._actors is not None:
            self._actors.close()
            self._actors = None
        shutil.rmtree(self._temporary)
        self._temporary = None
        self._stage = None
        self._frame = None
        self._occupancy = None
