# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Independent Isaac/ROS 2 execution; ROS owns all pedestrian navigation decisions."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import select
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..configuration.models import SimulationConfig
from ..configuration.navigation import Ros2NavigationSettings, load_ros2_navigation_settings
from ..configuration.paths import ROS2_KIT_EXPERIENCE_PATH
from ..configuration.population import resolve_population
from ..configuration.resources import resolve_ros2_motion_library
from ..configuration.scenarios.assets import find_character_assets
from ..recording.output import _validate_report_path
from ..scene.composition import create_runtime_layer
from ..scene.namespaces import select_runtime_prim_path
from ..scene.physics import configure_physics, validate_robots
from .actors import Ros2Actors
from .frames import StageFrame
from .mapping import DynamicObstacleObserver, build_occupancy_grid, save_occupancy_grid
from .planner import OccupancyGrid, convex_footprint, segment_polygon_clear
from .solver import build_solver
from .transport import Ros2Transport


def _spawn_geometry(stage, frame, navigation, plans):
    """Read current solid poses [m] for placement, without predicting motion."""
    if not plans:
        return ()
    observer = DynamicObstacleObserver(stage, frame, (), dynamic_roots=navigation.dynamic_roots)
    try:
        return tuple(
            {
                **obstacle,
                "footprint_xy_m": (
                    None if obstacle.get("footprint_xy_m") is None else convex_footprint(obstacle["footprint_xy_m"])
                ),
            }
            for obstacle in observer.sample()
            if obstacle.get("z_max_m", math.inf) >= navigation.floor_height_m
            and obstacle.get("z_min_m", -math.inf) <= navigation.floor_height_m + navigation.body_height_m
        )
    finally:
        observer.close()


def _spawn_clear(point, clearance, obstacles):
    """Check the proposed body's current footprint clearance [m]."""
    point = np.asarray(point[:2])
    for obstacle in obstacles:
        if obstacle["footprint_xy_m"] is not None:
            if not segment_polygon_clear(point, point, obstacle["footprint_xy_m"], clearance):
                return False
        elif np.linalg.norm(point - obstacle["position"][:2]) <= clearance + float(obstacle["radius_m"]):
            return False
    return True


def _certifiable_obstacles(obstacles):
    """Drop zero-area dynamic footprints that cannot block a planar body [m]."""
    kept = []
    for obstacle in obstacles:
        try:
            convex_footprint(obstacle["footprint_xy_m"])
        except ValueError:
            continue
        kept.append(obstacle)
    return kept


def _validate_fixed_spawn(plan, point, grid, navigation, obstacles, assigned):
    """Reject an invalid fixed disk [m] without moving its requested position."""
    clearance = navigation.body_radius_m + navigation.map_clearance_m
    if not grid.segment_clear(point, point, clearance):
        raise ValueError(f"{plan.name}: fixed spawn is outside disk-clear occupancy space.")
    if not _spawn_clear(point, clearance, obstacles):
        raise ValueError(f"{plan.name}: fixed spawn intersects current physical geometry.")
    for other, other_point in assigned:
        if math.dist(point[:2], other_point[:2]) < 2 * navigation.body_radius_m - 1e-9:
            raise ValueError(f"{plan.name}: fixed spawn body overlaps {other.name}.")


def _resolve_fixed_spawn(plan, point, grid, navigation, obstacles, assigned, future, frame):
    """Keep a safe requested pose or choose a nearby certified raster center [m]."""
    try:
        _validate_fixed_spawn(plan, point, grid, navigation, obstacles, assigned)
        return point
    except ValueError as error:
        tolerance = navigation.spawn_tolerance_m
        if tolerance == 0:
            raise
        mask = grid.free_mask(navigation.body_radius_m + navigation.map_clearance_m)
        cells = grid._near_cells(point[:2], tolerance, mask)
        centers = sorted((grid.cell_center(cell) for cell in cells), key=lambda xy: math.dist(point[:2], xy))
        nonterminal = plan.waypoints if len(plan.waypoints) == 1 else plan.waypoints[:-1]
        reserved_waypoints = {waypoint.position for waypoint in nonterminal}
        for center in centers:
            candidate = (*center, point[2])
            if math.dist(candidate, point) > tolerance or frame.ros_to_world(candidate) in reserved_waypoints:
                continue
            try:
                _validate_fixed_spawn(plan, candidate, grid, navigation, obstacles, (*assigned, *future))
            except ValueError:
                continue
            return candidate
        raise ValueError(f"{error} No safe fixed spawn within spawn_tolerance_m={tolerance:g} m.") from error


def _create_app(config: SimulationConfig, navigation, extra_args):
    needs_ros = config.pedestrians.enabled or (config.drl is not None and config.drl.enabled)
    if needs_ros and importlib.util.find_spec("rclpy") is None:
        raise RuntimeError(
            f"Source {navigation.ros_setup_path} before starting ROS 2 simulation; "
            "no dependencies are installed automatically."
        )
    os.environ.setdefault("OMNI_CRASHREPORTER_ENABLED", "0")
    os.environ.setdefault("OMNI_CRASHREPORTER_SKIPOLDDUMPUPLOAD", "1")
    os.environ.setdefault("ROS_LOG_DIR", str(Path(tempfile.gettempdir()) / "pedestrian_ros_logs"))
    import isaacsim
    from isaacsim import SimulationApp

    return SimulationApp(
        {
            "headless": config.settings.headless,
            "width": config.settings.width,
            "height": config.settings.height,
            "multi_gpu": False,
            "limit_cpu_threads": 8,
            "extra_args": [
                "--/crashreporter/enabled=false",
                "--/crashreporter/skipOldDumpUpload=true",
                "--/app/uploadDumpsOnStartup=false",
                *([f"--/persistent/isaac/asset_root/default={navigation.asset_root}"] if navigation.asset_root else []),
                *[
                    arg
                    for name in ("exts", "extscache", "extsDeprecated")
                    for arg in ("--ext-folder", str(Path(isaacsim.__file__).parent / name))
                ],
                *(extra_args or []),
            ],
        },
        experience=str(ROS2_KIT_EXPERIENCE_PATH),
    )


def _compose_world(config, directory: Path) -> tuple[Path, list[str]]:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    base = config.scene.asset_path
    if base is None:
        raise ValueError("ROS 2 simulation requires an explicit scene asset.")
    source = Usd.Stage.Open(str(base))
    overlay = create_runtime_layer(config, str(base), directory)
    layer = Sdf.Layer.CreateNew(str(directory / "ros2_world.usda"))
    layer.subLayerPaths = [str(p) for p in (overlay, base) if p is not None]
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    UsdPhysics.SetStageKilogramsPerUnit(stage, UsdPhysics.GetStageKilogramsPerUnit(source))
    stage.SetTimeCodesPerSecond(source.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(source.GetFramesPerSecond())
    suppressed = []
    door_controllers = (
        tuple(prim.GetParent().GetPath() for prim in stage.Traverse() if prim.GetRelationship("automaticDoor:joint"))
        if config.pedestrians.enabled
        else ()
    )
    # A composed asset must not start a second controller behind the ROS owner.
    for prim in stage.Traverse():
        if prim.GetTypeName() in ("NavMesh", "NavMeshVolume"):
            physical = [
                str(child.GetPath())
                for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                if child.HasAPI(UsdPhysics.CollisionAPI)
                and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get() is not False
            ]
            if physical:
                raise ValueError(
                    f"Navigation container {prim.GetPath()} owns enabled physical collision geometry: "
                    f"{physical}. Provide a neutral physics overlay or independent map configuration; "
                    "its physical world cannot be deactivated with navigation."
                )
            suppressed.append(str(prim.GetPath()))
            prim.SetActive(False)
            continue
        for attr in prim.GetAttributes():
            name = str(attr.GetName())
            if name == "inputs:script":
                script = attr.Get() or ""
                owns_door_drive = (
                    any(prim.GetPath().HasPrefix(path) for path in door_controllers)
                    and "automaticDoor:" in script
                    and "DriveAPI" in script
                )
                if owns_door_drive or any(
                    token in script
                    for token in (
                        "omni.anim.behavior",
                        "omni.anim.navigation",
                        "isaacsim.replicator.agent",
                        "simulation.behaviors.character",
                        "simulation.behaviors.interaction",
                        "simulation.behaviors.robot_yield",
                    )
                ):
                    attr.Set("def compute(db):\n    return True\n")
                    suppressed.append(str(attr.GetPath()))
            elif name == "omni:scripting:scripts":
                scripts = attr.Get()
                if scripts:
                    attr.Set([])
                    suppressed.append(str(attr.GetPath()))
    layer.Save()
    return Path(layer.realPath), suppressed


def _canonical_plans(input_plans, frame, input_frame):
    """Normalize authored waypoints to canonical world positions [m]."""
    if input_frame == "stage_world":
        plans = tuple(
            replace(
                plan,
                position=frame.stage_to_world(plan.position) if plan.position is not None else None,
                waypoints=tuple(
                    replace(waypoint, position=frame.stage_to_world(waypoint.position)) for waypoint in plan.waypoints
                ),
            )
            for plan in input_plans
        )
    elif input_frame == "world_m":
        plans = input_plans
    else:
        raise ValueError("navigation.waypoint_frame must be world_m or stage_world.")
    return plans


def _validate_walk_routes(plans, frame, grid, navigation):
    """Check original route legs on a measured occupancy grid [m]."""
    for plan in plans:
        if plan.behavior != "walk":
            continue
        start = frame.world_to_ros(plan.position)
        for index, waypoint in enumerate(plan.waypoints):
            goal = frame.world_to_ros(waypoint.position)
            if math.dist(start, goal) <= plan.arrival_tolerance:
                continue
            dz = abs(start[2] - goal[2])
            try:
                if dz >= plan.arrival_tolerance:
                    raise ValueError("Goal height exceeds the planar map's arrival tolerance.")
                grid.plan(
                    start,
                    goal,
                    navigation.body_radius_m,
                    math.sqrt(plan.arrival_tolerance**2 - dz**2),
                    navigation.map_clearance_m,
                )
            except ValueError as error:
                raise ValueError(f"{plan.name} waypoint {index + 1}: static map validation failed: {error}") from error


def _bootstrap_plans(plans, input_plans, frame, spawn_adjustments):
    """Serialize fixed ROS route data [m, deg] without altering authored plans."""
    bootstrap_plans = []
    for plan, authored in zip(plans, input_plans, strict=True):
        data = plan.to_dict()
        # A safe adjusted spawn cannot change the authored route topology.
        if plan.name in spawn_adjustments and plan.behavior == "walk" and plan.repeat:
            data["authored_closed_route"] = authored.waypoints[-1].position == authored.position
        data["position"] = list(frame.world_to_ros(plan.position))
        for point in data["waypoints"]:
            point["position"] = list(frame.world_to_ros(point["position"]))
            if point["yaw"] is not None:
                point["yaw"] -= math.degrees(frame.yaw_rad)
        bootstrap_plans.append(data)
    return bootstrap_plans


def _assets(config: SimulationConfig, navigation, plans):
    population = resolve_population(config).config or config
    sources = dict(population.pedestrian_assets)
    motion = resolve_ros2_motion_library(population.motion_library, navigation.asset_root)
    generator = np.random.default_rng(config.settings.seed)
    selected = {}
    pools = {}
    for plan in plans:
        source = sources[plan.name]
        key = (source, plan.gender)
        if key not in pools:
            pools[key] = find_character_assets(source, plan.gender, asset_root=navigation.asset_root)
        candidates = pools[key]
        selected[plan.name] = candidates[int(generator.integers(len(candidates)))]
    return selected, motion


class _ControlTimebase:
    """Own a seconds-based main-loop step without changing USD animation metrics."""

    _fixed_setting = "/app/player/useFixedTimeStepping"

    def __init__(self, control_hz: int) -> None:
        import carb.settings
        from omni.kit.loop import _loop

        self._module = _loop
        self._settings = carb.settings.get_settings()
        self._runner = _loop.acquire_loop_interface()
        self._previous_mode = self._runner.get_manual_mode("main")
        self._previous_step = self._runner.get_manual_step_size("main")
        self._previous_fixed = self._settings.get(self._fixed_setting)
        try:
            self._runner.set_manual_step_size(1.0 / control_hz, "main")
            self._runner.set_manual_mode(True, "main")
            # Kit's fixed player mode advances one animation time code, rather
            # than consuming the deterministic seconds supplied by this loop.
            self._settings.set_bool(self._fixed_setting, False)
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup:
                error.add_note(f"Control timebase restoration failed: {cleanup}")
            raise

    def close(self) -> None:
        if self._runner is None:
            return
        errors = []
        for restore in (
            lambda: self._runner.set_manual_step_size(self._previous_step, "main"),
            lambda: self._runner.set_manual_mode(self._previous_mode, "main"),
            lambda: (
                self._settings.destroy_item(self._fixed_setting)
                if self._previous_fixed is None
                else self._settings.set_bool(self._fixed_setting, self._previous_fixed)
            ),
        ):
            try:
                restore()
            except BaseException as error:
                errors.append(error)
        if errors:
            raise errors[0]
        self._module.release_loop_interface(self._runner)
        self._runner = None


class _DrlCmdVelTransport:
    """Minimal ROS 2 publisher for a robot-only DRL session."""

    def __init__(self, navigation) -> None:
        import rclpy
        from geometry_msgs.msg import Twist
        from rclpy.context import Context
        from rclpy.node import Node

        self._rclpy = rclpy
        self._context = Context()
        rclpy.init(context=self._context, domain_id=int(navigation.ros_domain_id))
        self.namespace = str(navigation.ros_namespace).rstrip("/") + "/drl_only"
        self.monitor_manifest = None
        self._node = Node("drl_cmd_vel", namespace=self.namespace, context=self._context)
        self._publisher = self._node.create_publisher(Twist, "/cmd_vel", 10)

    def create_publisher(self, message_type, topic: str, qos: int = 10):
        if topic == "/cmd_vel":
            return self._publisher
        return self._node.create_publisher(message_type, topic, qos)

    def close(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._context is not None:
            if self._rclpy.ok(context=self._context):
                self._rclpy.shutdown(context=self._context)
            self._context = None


class Ros2SimulationSession:
    """Own one scene and ROS transport, with no IRA or navigation-mesh runtime."""

    def __init__(self, config: SimulationConfig, *, extra_args=None, output: Path | None = None):
        config.validate()
        self.config = config
        needs_navigation = config.pedestrians.enabled or (
            config.drl is not None and config.drl.enabled
        )
        self.navigation = (
            load_ros2_navigation_settings(config.navigation.settings_path)
            if needs_navigation
            else Ros2NavigationSettings()
        )
        self.output = output or (
            config.settings.report_path.parent if config.settings.report_path else Path.cwd() / "output"
        ) / ("ros2_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6])
        self.output = self.output.resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        self._extra_args = extra_args
        self._temporary = tempfile.TemporaryDirectory(prefix="ros2_pedestrian_")
        self._app = self.timeline = self.actors = self.transport = None
        self.stage = None
        self._world_path: Path | None = None
        self._quit = False
        self._last_time = 0.0
        self._native_origin = 0.0
        self._samples = 0
        self._last_command = {"agents": []}
        self._suppressed = []
        self._map = None
        self._grid = None
        self._drl = None
        self._trace = None
        self._dynamic_previous = {}
        self.doors = None
        self._obstacles = None
        self._timebase = None
        self.input_plans = ()
        self.coordinate_contract = {}

    @property
    def app(self):
        """Owned native Kit application, available after setup starts."""
        return self._app

    def setup(self) -> None:
        """Load physics, map it, spawn visuals, then connect an independent ROS node."""
        self._app = _create_app(self.config, self.navigation, self._extra_args)
        import omni.kit.app
        import omni.timeline
        import omni.usd
        from pxr import UsdGeom

        manager = omni.kit.app.get_app().get_extension_manager()
        for extension in self.config.settings.extensions:
            if extension.startswith(("isaacsim.replicator.agent", "omni.anim.navigation", "omni.anim.behavior")):
                raise ValueError(f"ROS 2 mode rejects competing motion extension {extension}.")
            manager.set_extension_enabled_immediate(extension, True)
            if not manager.is_extension_enabled(extension):
                raise RuntimeError(f"Cannot enable requested extension {extension}.")
        if self.config.pedestrians.enabled:
            manager.set_extension_enabled_immediate("omni.anim.retarget.core", True)
            if not manager.is_extension_enabled("omni.anim.retarget.core"):
                raise RuntimeError("Independent animation retarget extension failed to start.")
        world, self._suppressed = _compose_world(self.config, Path(self._temporary.name))
        self._world_path = world
        context = omni.usd.get_context()
        opened, error = self.app.run_coroutine(context.open_stage_async(str(world)))
        self.stage = self._owned_stage(context) if opened else None
        if self.stage is None:
            reason = error or "composed USD did not attach"
            raise RuntimeError(f"Cannot open the ROS simulation scene {world}: {reason}")
        self._enable_loaded_graphs()
        configure_physics(self.config.settings)
        validate_robots(self.config)
        self.frame = StageFrame(
            UsdGeom.GetStageMetersPerUnit(self.stage),
            str(UsdGeom.GetStageUpAxis(self.stage)),
            self.navigation.origin_m,
            self.navigation.yaw_rad,
        )
        input_frame = getattr(self.config.navigation, "waypoint_frame", "world_m")
        self.coordinate_contract = {
            "input_waypoint_frame": input_frame,
            "effective_plans_frame": "world_m",
            "bootstrap_plans_frame": "ros_map",
            "stage_meters_per_unit": self.frame.meters_per_unit,
            "stage_up_axis": self.frame.up_axis,
            "map_origin_world_m": list(self.frame.origin_m),
            "map_yaw_rad": self.frame.yaw_rad,
            "speed_unit": "m/s",
            "arrival_tolerance_unit": "m",
            "heading_unit": "deg",
        }
        self.timeline = omni.timeline.get_timeline_interface()
        self._timebase = _ControlTimebase(self.navigation.control_hz)
        self.timeline.set_end_time(max(self.timeline.get_end_time(), (self.config.settings.duration or 86400.0) + 1.0))
        self.timeline.set_target_framerate(float(self.navigation.control_hz))
        self.timeline.play()
        self.timeline.commit()
        for _ in range(10):
            self.app.update()
        self.timeline.pause()
        self.timeline.commit()
        self.app.update()
        if not self.config.pedestrians.enabled:
            self.plans = ()
            self.input_plans = ()
            if self.config.drl is not None and self.config.drl.enabled:
                settings = self.navigation.to_dict()
                settings["seed"] = self.config.settings.seed
                settings["plans"] = []
                self._map = build_occupancy_grid(self.stage, settings, self.frame)
                paths = save_occupancy_grid(self._map, self.output / "map")
                grid = OccupancyGrid(
                    self._map["occupancy"],
                    self._map["resolution_m"],
                    self._map["origin_xy_m"],
                    goal_projection_tolerance_m=self.navigation.goal_projection_tolerance_m,
                )
                self._grid = grid
                self.transport = _DrlCmdVelTransport(self.navigation)
                from .drl.controller import DrlVoController

                self._drl = DrlVoController(
                    self.config.drl,
                    grid=grid,
                    frame=self.frame,
                    transport=self.transport,
                    output=self.output,
                    robots=self.config.robots,
                    stage=self.stage,
                    navigation=self.navigation,
                )
                self._drl.start()
                self._trace = (self.output / "trace.jsonl").open("w")
            self.timeline.stop()
            self.timeline.commit()
            self.app.update()
            self._native_origin = self.timeline.get_current_time()
            self._assert_isolated()
            self._configure_viewport()
            return
        self.input_plans = (resolve_population(self.config).config or self.config).pedestrian_plans
        # Normalize scene-authored positions once. Every downstream consumer,
        # including animation and recording, receives canonical metric plans.
        plans = _canonical_plans(self.input_plans, self.frame, input_frame)
        settings = self.navigation.to_dict()
        settings["seed"] = self.config.settings.seed
        settings["plans"] = [p.to_dict() for p in plans]
        self._map = build_occupancy_grid(self.stage, settings, self.frame)
        paths = save_occupancy_grid(self._map, self.output / "map")
        grid = OccupancyGrid(
            self._map["occupancy"],
            self._map["resolution_m"],
            self._map["origin_xy_m"],
            goal_projection_tolerance_m=self.navigation.goal_projection_tolerance_m,
        )
        self._grid = grid
        spawn_obstacles = _spawn_geometry(self.stage, self.frame, self.navigation, plans)
        rng = np.random.default_rng(self.config.settings.seed)
        free = np.argwhere(grid.free_mask(self.navigation.body_radius_m + self.navigation.map_clearance_m))
        # Reserve later fixed spawns before any random actor is sampled.
        reserved = {
            plan.name: (plan, self.frame.world_to_ros(plan.position)) for plan in plans if plan.position is not None
        }
        assigned = []
        effective = []
        spawn_adjustments = {}
        for plan in plans:
            _, requested = reserved.pop(plan.name, (None, None))
            point = requested
            if point is None:
                unavailable = (*assigned, *(reserved_point for _, reserved_point in reserved.values()))
                for index in rng.permutation(len(free)):
                    row, column = free[index]
                    candidate = (
                        self._map["origin_xy_m"][0] + (column + 0.5) * self.navigation.resolution_m,
                        self._map["origin_xy_m"][1] + (row + 0.5) * self.navigation.resolution_m,
                        self.navigation.floor_height_m,
                    )
                    if _spawn_clear(
                        candidate,
                        self.navigation.body_radius_m + self.navigation.map_clearance_m,
                        spawn_obstacles,
                    ) and all(
                        math.dist(candidate, other)
                        > 2 * self.navigation.body_radius_m + self.navigation.pair_clearance_m
                        for other in unavailable
                    ):
                        point = candidate
                        break
                if point is None:
                    raise RuntimeError(f"No separated random spawn for {plan.name}.")
            if abs(point[2] - self.navigation.floor_height_m) > plan.arrival_tolerance:
                raise ValueError("A 2D ROS map cannot represent a different waypoint/spawn floor.")
            if requested is not None:
                point = _resolve_fixed_spawn(
                    plan,
                    point,
                    grid,
                    self.navigation,
                    spawn_obstacles,
                    tuple(zip(effective, assigned)),
                    tuple(reserved.values()),
                    self.frame,
                )
            if point != requested:
                position = self.frame.ros_to_world(point)
                if requested is not None:
                    spawn_adjustments[plan.name] = [
                        actual - original for actual, original in zip(position, plan.position)
                    ]
                plan = replace(plan, position=position)
            assigned.append(point)
            effective.append(plan)
        self.plans = tuple(effective)
        _validate_walk_routes(self.plans, self.frame, grid, self.navigation)
        self.timeline.stop()
        self.timeline.commit()
        self.app.update()
        self._native_origin = self.timeline.get_current_time()
        sources, motion = _assets(self.config, self.navigation, self.plans)
        self.asset_sources, self.motion_library = sources, motion
        self.actors = Ros2Actors(
            self.stage,
            self.plans,
            sources,
            motion,
            str(select_runtime_prim_path(self.stage, "Ros2Pedestrians", unique=True)),
            self.frame,
            self.config.settings.seed or 0,
            self.navigation.body_radius_m,
            self.navigation.body_height_m,
        )
        from .doors import Ros2Doors

        self.doors = Ros2Doors(self.stage, self.frame, navigation_interlock=True)
        settings["door_specs"] = self.doors.descriptors()
        self._obstacles = DynamicObstacleObserver(
            self.stage, self.frame, (self.actors.root_path,), dynamic_roots=self.navigation.dynamic_roots
        )
        bootstrap_plans = _bootstrap_plans(self.plans, self.input_plans, self.frame, spawn_adjustments)
        bootstrap = self.output / "bootstrap.json"
        bootstrap.write_text(
            json.dumps(
                {
                    "plans": bootstrap_plans,
                    "input_plans": [plan.to_dict() for plan in self.input_plans],
                    "world_plans": [plan.to_dict() for plan in self.plans],
                    "spawn_adjustments_world_m": spawn_adjustments,
                    "coordinate_contract": self.coordinate_contract,
                    "requested_duration_s": self.config.settings.duration,
                    "settings": settings,
                    "agents": self.actors.states(),
                    "map_npz": str(paths["npz"]),
                },
                indent=2,
            )
            + "\n"
        )
        library = build_solver(self.output / "build")
        self.transport = Ros2Transport(bootstrap, library, settings, self.output)
        if self.config.drl is not None:
            from .drl.controller import DrlVoController

            self._drl = DrlVoController(
                self.config.drl,
                grid=grid,
                frame=self.frame,
                transport=self.transport,
                output=self.output,
                robots=self.config.robots,
                stage=self.stage,
                navigation=self.navigation,
            )
            self._drl.start()
        self._trace = (self.output / "trace.jsonl").open("w")
        self._assert_isolated()
        self._configure_viewport()

    def _enable_loaded_graphs(self) -> None:
        """Enable OmniGraphs loaded disabled from referenced assets before playback starts."""
        try:
            import omni.graph.core as og
        except ImportError:
            return
        for prim in self.stage.Traverse():
            if prim.GetTypeName() != "OmniGraph":
                continue
            graph = og.get_graph_by_path(str(prim.GetPath()))
            if graph is not None and graph.is_disabled():
                graph.set_disabled(False)

    def _configure_viewport(self) -> None:
        if not self.config.settings.render_viewport:
            from omni.kit.viewport.utility import get_active_viewport

            viewport = get_active_viewport()
            if viewport is not None:
                viewport.updates_enabled = False

    def _assert_isolated(self) -> None:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        forbidden = ["isaacsim.replicator.agent.core", "omni.anim.navigation.core", "omni.anim.behavior.core"]
        active = [name for name in forbidden if manager.is_extension_enabled(name)]
        if active:
            raise RuntimeError(f"ROS navigation cannot share motion ownership with: {active}")

    def play(self) -> None:
        """Start or resume the native timeline without changing the ROS plan."""
        self.timeline.play()
        self.timeline.commit()
        if self._drl is not None:
            self._drl.resume()

    def pause(self) -> None:
        """Pause physics and ROS control at the current simulation time [s]."""
        self.timeline.pause()
        self.timeline.commit()
        if self._drl is not None:
            self._drl.pause()

    def request_exit(self) -> None:
        """Request a clean owner-loop exit."""
        self._quit = True

    def is_running(self) -> bool:
        """Return whether this application still owns the active run."""
        return not self._quit and self.app.is_running()

    def update(self) -> None:
        """Advance one Kit frame, then execute one matching ROS command batch."""
        self._assert_isolated()
        self.app.update()
        self._assert_isolated()
        if not self.timeline.is_playing():
            return
        now = self.timeline.get_current_time() - self._native_origin
        dt = now - self._last_time
        if dt < -1e-8:
            raise RuntimeError("Timeline rewound during a ROS control sequence; restart the session to reset its plan.")
        if dt <= 0:
            return
        if abs(dt - 1 / self.navigation.control_hz) > 1e-5:
            raise RuntimeError(f"Native timeline step {dt} does not match configured ROS control frequency.")
        if not self.config.pedestrians.enabled:
            if self._drl is not None:
                self._drl.step(now, dt, [], [])
            self._last_time = now
            self._samples += 1
            return
        states = self.actors.states()
        door_states = self.doors.observations(now) if self.doors else []
        dynamic = self._obstacles.sample()
        dynamic = [
            obstacle
            for obstacle in dynamic
            if obstacle.get("z_max_m", math.inf) >= self.navigation.floor_height_m
            and obstacle.get("z_min_m", -math.inf) <= self.navigation.floor_height_m + self.navigation.body_height_m
        ]
        observed = {}
        for obstacle in dynamic:
            key = obstacle["prim_path"]
            previous = self._dynamic_previous.get(key)
            point = tuple(obstacle["position"])
            observed[key] = now, point
            if previous is not None and math.isclose(now - previous[0], dt, rel_tol=1e-6, abs_tol=1e-8):
                obstacle["velocity"] = [(point[i] - previous[1][i]) / (now - previous[0]) for i in range(3)]
                obstacle["velocity_valid"] = True
        self._dynamic_previous = observed
        dynamic = _certifiable_obstacles(dynamic)
        if self._drl is not None:
            self._drl.step(now, dt, states, dynamic)
        if any(not obstacle["velocity_valid"] for obstacle in dynamic):
            if self._samples:
                raise RuntimeError("A new dynamic obstacle needs a measured velocity before ROS motion can continue.")
            if self.doors:
                self.doors.update(states, now, dt)
            self._last_time = now
            return
        command = self.transport.step(now, dt, states, obstacles=dynamic, door_states=door_states)
        self._last_command = command
        failures = [
            f"{agent.get('name')}: {agent.get('error') or 'failed navigation'}"
            for agent in command["agents"]
            if agent.get("error") or agent.get("phase") == "failed"
        ]
        if failures:
            self._trace.write(
                json.dumps(
                    {
                        "time_s": now,
                        "observed": states,
                        "obstacles": dynamic,
                        "doors": door_states,
                        "command": command,
                        "drl": self._drl.trace_state() if self._drl else None,
                        "failed": True,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            self._trace.flush()
            raise RuntimeError("ROS 2 pedestrian navigation failed: " + "; ".join(failures))
        self.actors.apply(command["agents"], dt)
        if self.doors:
            self.doors.set_navigation_intents(command.get("door_commands", []))
            self.doors.update(states, now, dt)
        self._last_time = now
        self._last_command = command
        self._samples += 1
        self._trace.write(
            json.dumps(
                {
                    "time_s": now,
                    "observed": states,
                    "states": self.actors.states(),
                    "obstacles": dynamic,
                    "doors": door_states,
                    "command": command,
                    "drl": self._drl.trace_state() if self._drl else None,
                },
                allow_nan=False,
            )
            + "\n"
        )
        self._assert_isolated()

    def status(self) -> dict[str, object]:
        """Read ROS navigation, native pose and map diagnostics without advancing tasks."""
        return {
            "controller": (
                "ros2_sfm_orca"
                if self.config.pedestrians.enabled
                else ("drl_only" if self._drl is not None else "disabled")
            ),
            "ira_enabled": False,
            "navmesh_enabled": False,
            "simulation_time_s": self._last_time,
            "control_frames": self._samples,
            "coordinate_contract": self.coordinate_contract,
            "ros_namespace": self.transport.namespace if self.transport else None,
            "monitor_manifest": (
                str(self.transport.monitor_manifest)
                if self.transport and getattr(self.transport, "monitor_manifest", None)
                else None
            ),
            "pedestrian_schedules": self._last_command["agents"],
            "passing": self._last_command.get("passing", {}),
            "goal_occupancy": self._last_command.get("goal_occupancy", {}),
            "door_closure": self._last_command.get("door_closure", {}),
            "actors": self.actors.status() if self.actors else {},
            "map": self._map["diagnostics"] if self._map else {},
            "suppressed_conflicting_scripts": self._suppressed,
            "doors": self.doors.status() if self.doors else {},
            "drl_vo": self._drl.status() if self._drl else None,
            "output_directory": str(self.output),
        }

    def _owned_stage(self, context):
        # The private root layer also identifies a partially attached stage
        # when the SDK reports an open failure.
        stage = context.get_stage()
        if stage is not None and stage.GetRootLayer().realPath == str(self._world_path):
            return stage
        return None

    def close_scene(self) -> dict[str, bool]:
        """Stop ROS, detach native state and release temporary USD only after stage close."""
        errors = []
        if self._drl is not None:
            try:
                self._drl.close()
            except BaseException as error:
                errors.append(error)
            self._drl = None
        for resource in ("transport", "_trace"):
            value = getattr(self, resource, None)
            if value is not None:
                try:
                    value.close()
                    setattr(self, resource, None)
                except BaseException as error:
                    errors.append(error)
        if self.timeline is not None:
            try:
                self.timeline.stop()
                self.timeline.commit()
                if self.app is not None and self.app.is_running():
                    self.app.update()
            except BaseException as error:
                errors.append(error)
            finally:
                self.timeline = None
        for resource in ("_obstacles", "actors", "doors"):
            value = getattr(self, resource, None)
            if value is not None:
                try:
                    value.close()
                except BaseException as error:
                    errors.append(error)
        closed = True
        if self._world_path is not None:
            closed = False
            try:
                import omni.usd

                context = omni.usd.get_context()
                if self._owned_stage(context) is not None:
                    if self.app.is_running():
                        self.app.run_coroutine(context.close_stage_async())
                    else:
                        context.close_stage()
                closed = self._owned_stage(context) is None
            except BaseException as error:
                errors.append(error)
        if not closed:
            self._temporary._finalizer.detach()
            errors.append(RuntimeError(f"ROS simulation stage is still open; retained {self._temporary.name}."))
        else:
            self.stage = None
            try:
                self._temporary.cleanup()
            except BaseException as error:
                errors.append(error)
            else:
                self._world_path = None
        if getattr(self, "_timebase", None) is not None:
            try:
                self._timebase.close()
                self._timebase = None
            except BaseException as error:
                errors.append(error)
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"Additional ROS cleanup failure: {error}")
            raise errors[0]
        return {"stage_closed": True, "temporary_removed": True, "ros_process_stopped": True}


def _finalize_ros2_run(
    session: Ros2SimulationSession,
    report: dict[str, object],
    report_path: Path | None,
    error: BaseException | None,
) -> BaseException | None:
    """Attempt every owned finalizer and preserve the earliest failure."""
    if error is not None:
        report["passed"] = False

    def remember(phase: str, failure: BaseException) -> None:
        nonlocal error
        report["passed"] = False
        report.setdefault("finalization_errors", []).append({"phase": phase, "error": repr(failure)})
        if error is None:
            error = failure
            report.setdefault("error", f"{type(failure).__name__}: {failure}")
        elif failure is not error:
            error.add_note(f"ROS {phase} failed: {type(failure).__name__}: {failure}")

    try:
        report["cleanup"] = session.close_scene()
    except BaseException as failure:
        report["cleanup_error"] = repr(failure)
        remember("scene cleanup", failure)

    targets = [("runtime report", session.output / "result.json")]
    if report_path is not None:
        targets.append(("configured report", report_path))

    def write_reports(paths):
        written, failed = [], False
        for phase, path in paths:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
                written.append((phase, path))
            except BaseException as failure:
                remember(phase, failure)
                failed = True
        return written, failed

    def print_result() -> None:
        print(
            "ROS2_SIMULATION_RESULT "
            + json.dumps({"passed": report["passed"], "output": str(session.output), "error": report.get("error")}),
            flush=True,
        )

    # Kit can terminate the process inside close(), so persist every currently
    # known result before asking it to exit.
    written, failed = write_reports(targets)
    if failed:
        write_reports(written)
    try:
        print_result()
    except BaseException as failure:
        remember("result notification", failure)
        write_reports(written)
    if session.app is not None:
        try:
            session.app.close(exit_code=0 if report["passed"] else 1)
        except BaseException as failure:
            report["app_close_error"] = repr(failure)
            remember("application close", failure)
            written, _ = write_reports(targets)
            try:
                print_result()
            except BaseException as notification:
                remember("result notification", notification)
                write_reports(written)
    return error


def run_ros2_simulation(config: SimulationConfig, *, extra_args=None, console=None, report_path=None) -> None:
    """Run independent ROS navigation to the configured simulation duration [s]."""
    report_path = config.settings.report_path if report_path is None else report_path
    console = config.settings.console if console is None else console
    _validate_report_path(config, report_path)
    if config.recording.enabled:
        from ..recording.output import validate_recording_runtime

        validate_recording_runtime(config)
        if console:
            raise ValueError("ROS recording requires automatic playback without console controls.")
    session = Ros2SimulationSession(config, extra_args=extra_args)
    started = time.monotonic()
    playback = None
    report = {"passed": False, "requested_duration_s": config.settings.duration}
    error = None
    try:
        session.setup()
        playback = time.monotonic()
        if config.recording.enabled:
            from ..recording.ros2_recorder import record_ros2_simulation

            report["recording"] = record_ros2_simulation(session)
            complete = report["recording"]["recorded_elapsed_s"] >= config.settings.duration - 1e-8
            report.update(session.status(), passed=complete, completed_duration=complete)
            return
        if not config.settings.start_paused:
            session.play()
        print("ROS2_SIMULATION_READY " + json.dumps(session.status()), flush=True)
        progress = time.monotonic()
        while session.is_running():
            if console and select.select([sys.stdin], [], [], 0)[0]:
                command = sys.stdin.readline().strip()
                if command in ("play", "pause"):
                    getattr(session, command)()
                elif command == "quit":
                    session.request_exit()
                elif command == "status":
                    print(json.dumps(session.status()), flush=True)
            session.update()
            if (
                not config.settings.start_paused
                and session.timeline.is_stopped()
                and config.settings.duration > session._last_time
            ):
                raise RuntimeError("The native timeline stopped before the configured simulation duration.")
            if time.monotonic() - progress >= 5:
                print("ROS2_SIMULATION_PROGRESS " + json.dumps(session.status()), flush=True)
                progress = time.monotonic()
            if config.settings.duration and session._last_time >= config.settings.duration - 1e-8:
                break
        complete = not config.settings.duration or session._last_time >= config.settings.duration - 1e-8
        report.update(session.status(), passed=complete, completed_duration=complete)
    except BaseException as failure:
        error = failure
        report["error"] = traceback.format_exc()
        try:
            report.update(session.status())
        except Exception as status_error:
            report["status_error"] = repr(status_error)
        raise
    finally:
        original_error = error
        report["wall_time_s"] = time.monotonic() - started
        report["playback_wall_time_s"] = None if playback is None else time.monotonic() - playback
        error = _finalize_ros2_run(session, report, report_path, error)
        if original_error is None and error is not None:
            raise error
