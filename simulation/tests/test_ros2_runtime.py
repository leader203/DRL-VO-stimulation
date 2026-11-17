# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS execution boundary regressions without Kit, DDS or GPU startup."""

import asyncio
import io
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from pedstrain_simulation.simulation.cli.arguments import create_parser
from pedstrain_simulation.simulation.cli.overrides import _simulation_from_args
from pedstrain_simulation.simulation.configuration.models import RobotConfig, SimulationSettings
from pedstrain_simulation.simulation.configuration.navigation import NavigationConfig, Ros2NavigationSettings
from pedstrain_simulation.simulation.configuration.scenarios.models import PedestrianPlan, Waypoint
from pedstrain_simulation.simulation.projects.inspection import _inspect_simulation_config
from pedstrain_simulation.simulation.ros2_navigation import doors, node, runtime, transport


def _modules(monkeypatch, names):
    result = {name: ModuleType(name) for name in names}
    for name, module in result.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(result[parent], child, module)
    return result


class _Timeline:
    def __init__(self, origin=0.0):
        self.origin = self.now = origin
        self.playing = True
        self.end = 100.0

    def get_current_time(self):
        return self.now

    def is_playing(self):
        return self.playing

    def is_stopped(self):
        return not self.playing

    def play(self):
        self.playing = True

    def pause(self):
        self.playing = False

    def stop(self):
        self.playing = False
        self.now = self.origin

    def commit(self):
        pass

    def get_end_time(self):
        return self.end

    def set_end_time(self, value):
        self.end = value

    def set_target_framerate(self, value):
        self.rate = value


class _App:
    def __init__(self, timeline):
        self.timeline = timeline
        self.running = True
        self.updates = 0
        self.advance = False

    def update(self):
        if not self.running:
            raise RuntimeError("app already quit")
        self.updates += 1
        if self.advance and self.timeline.playing:
            self.timeline.now += 1 / 60

    def is_running(self):
        return self.running

    def close(self, exit_code=0):
        self.running = False
        self.exit_code = exit_code

    def run_coroutine(self, coroutine):
        return asyncio.run(coroutine)


class _Resource:
    def __init__(self, error=None):
        self.closed = False
        self.error = error

    def close(self):
        self.closed = True
        if self.error:
            raise RuntimeError(self.error)

    def update(self, *args):
        pass

    def sample(self):
        return []


class _Doors(_Resource):
    def descriptors(self):
        return [{"name": "door_metadata"}]

    def observations(self, time_s):
        return [{"name": "door_metadata", "fully_open": False, "time_s": time_s}]

    def set_navigation_intents(self, intents):
        self.intents = intents


class _Actors(_Resource):
    root_path = "/FakeActors"

    def __init__(self):
        super().__init__()
        self.applied = []

    def states(self):
        return [
            {
                "name": "walker",
                "position": [0.0, 0.0, 0.0],
                "yaw": 0.0,
                "velocity": [0.0, 0.0, 0.0],
                "radius_m": 0.25,
                "height_m": 1.8,
            }
        ]

    def apply(self, commands, dt):
        self.applied.append((commands, dt))


class _Transport(_Resource):
    namespace = "/fake_session"

    def __init__(self):
        super().__init__()
        self.requests = []
        self.failure = {}

    def step(self, time_s, dt, states, obstacles=(), door_states=()):
        self.requests.append(
            dict(time_s=time_s, dt=dt, states=states, obstacles=list(obstacles), doors=list(door_states))
        )
        return {"time_s": time_s, "agents": [{**states[0], "phase": "walking", "error": None, **self.failure}]}


class _Temporary:
    name = "/tmp/owned_ros_test_composition"

    def __init__(self):
        self.cleaned = False
        self._finalizer = SimpleNamespace(detach=lambda: None)

    def cleanup(self):
        self.cleaned = True


def _session(monkeypatch, *, origin=0.0):
    session = runtime.Ros2SimulationSession.__new__(runtime.Ros2SimulationSession)
    session.timeline = _Timeline(origin)
    session._app = _App(session.timeline)
    session.navigation = Ros2NavigationSettings()
    session.actors, session.transport, session.doors = _Actors(), _Transport(), None
    session.stage = session.frame = None
    session._native_origin, session._last_time, session._samples = origin, 0.0, 0
    session._dynamic_previous, session._last_command = {}, {"agents": []}
    session._trace, session._temporary = io.StringIO(), _Temporary()
    session._assert_isolated = lambda: None
    session._obstacles = _Resource()
    session.config = SimpleNamespace(pedestrians=SimpleNamespace(enabled=True))
    return session


def test_nonzero_native_clock_sends_a_single_relative_control_step(monkeypatch):
    session = _session(monkeypatch, origin=5.0)
    session.timeline.now = 5.0 + 1 / 60
    session.update()
    assert session.transport.requests[0]["time_s"] == pytest.approx(1 / 60)
    assert session.transport.requests[0]["dt"] == pytest.approx(1 / 60)
    assert len(session.actors.applied) == 1


def test_door_observation_and_command_share_the_navigation_tick(monkeypatch):
    session = _session(monkeypatch)
    events = []
    observation = [{"name": "door", "fully_open": False}]
    intent = [{"name": "door", "hold": True, "request_open": False}]

    class Doors(_Doors):
        def observations(self, time_s):
            events.append("observe")
            return observation

        def set_navigation_intents(self, intents):
            events.append("intent")
            assert intents == intent

        def update(self, states, time_s, dt):
            events.append("drive")
            assert len(session.actors.applied) == 1

    def command(time_s, dt, states, obstacles=(), door_states=()):
        events.append("navigate")
        assert door_states == observation
        return {"agents": states, "door_commands": intent}

    session.doors = Doors()
    session.transport.step = command
    session.timeline.now = 1 / 60
    session.update()
    assert events == ["observe", "navigate", "intent", "drive"]
    assert json.loads(session._trace.getvalue())["doors"] == observation


def test_rewound_clock_is_rejected_before_any_navigation_or_pose_write(monkeypatch):
    session = _session(monkeypatch)
    session._last_time, session.timeline.now = 10.0, 9.0
    with pytest.raises(RuntimeError, match="rewound"):
        session.update()
    assert session.transport.requests == [] and session.actors.applied == []


def _obstacle(x=0.0):
    return {
        "prim_path": "/door",
        "position": [x, 0.0, 0.0],
        "radius_m": 0.3,
        "velocity": [0.0, 0.0, 0.0],
        "velocity_valid": False,
    }


def test_initial_dynamic_obstacle_waits_for_a_contiguous_velocity_measurement(monkeypatch):
    session = _session(monkeypatch)
    observations = iter([[_obstacle()], [_obstacle(0.01)]])
    session._obstacles.sample = lambda: next(observations)
    session.timeline.now = 1 / 60
    session.update()
    assert session.transport.requests == [] and session.actors.applied == []
    session.timeline.now = 2 / 60
    session.update()
    observed = session.transport.requests[0]["obstacles"][0]
    assert observed["velocity_valid"] is True
    assert observed["velocity"] == pytest.approx([0.6, 0.0, 0.0])


def test_reappearing_dynamic_path_cannot_reuse_history_across_a_missing_frame(monkeypatch):
    session = _session(monkeypatch)
    observations = iter([[_obstacle()], [], [_obstacle(0.1)]])
    session._obstacles.sample = lambda: next(observations)
    for tick in (1, 2):
        session.timeline.now = tick / 60
        session.update()
    session.timeline.now = 3 / 60
    with pytest.raises(RuntimeError, match="measured velocity"):
        session.update()
    assert len(session.transport.requests) == len(session.actors.applied) == 1


@pytest.mark.parametrize("failure", [{"error": "unreachable"}, {"phase": "failed"}])
def test_failed_agent_command_rejects_the_whole_batch_before_pose_application(monkeypatch, failure):
    session = _session(monkeypatch)
    session.transport.failure = failure
    session.timeline.now = 1 / 60
    with pytest.raises(RuntimeError, match="walker.*(?:unreachable|failed navigation)"):
        session.update()
    assert session.actors.applied == [] and session._samples == 0


def _stage_context(monkeypatch, session, *, remain_open=False):
    modules = _modules(monkeypatch, ("omni", "omni.usd"))
    session._world_path = Path("/fake/ros2_world.usda")

    class Context:
        stage = SimpleNamespace(GetRootLayer=lambda: SimpleNamespace(realPath=str(session._world_path)))

        def get_stage(self):
            return self.stage

        def close_stage(self):
            if not remain_open:
                self.stage = None

        async def close_stage_async(self):
            self.close_stage()

    context = Context()
    modules["omni.usd"].get_context = lambda: context
    return context


def test_quit_application_skips_native_update_and_still_releases_all_owned_resources(monkeypatch):
    session = _session(monkeypatch)
    context = _stage_context(monkeypatch, session)
    session._app.running = False
    session.doors = _Resource()
    assert session.close_scene()["temporary_removed"]
    assert session._app.updates == 0
    assert session.actors.closed and session.doors.closed and session._temporary.cleaned
    assert context.get_stage() is None


def test_cleanup_failure_does_not_skip_other_resources_or_closed_stage_temporary_cleanup(monkeypatch):
    session = _session(monkeypatch)
    context = _stage_context(monkeypatch, session)
    session.transport.error = "transport cleanup failed"
    session.actors.error = "actor cleanup failed"
    session.doors = _Resource()
    with pytest.raises(RuntimeError, match="transport cleanup failed") as caught:
        session.close_scene()
    assert session.actors.closed and session.doors.closed and session._temporary.cleaned
    assert context.get_stage() is None
    assert any("actor cleanup failed" in note for note in caught.value.__notes__)


def test_live_stage_keeps_the_owned_temporary_composition_after_a_failed_close(monkeypatch):
    session = _session(monkeypatch)
    context = _stage_context(monkeypatch, session, remain_open=True)
    with pytest.raises(RuntimeError, match="stage is still open"):
        session.close_scene()
    assert context.get_stage() is not None and not session._temporary.cleaned


def _setup_environment(monkeypatch, tmp_path):
    modules = _modules(
        monkeypatch,
        ("omni", "omni.kit", "omni.kit.app", "omni.kit.loop", "omni.timeline", "omni.usd", "carb", "carb.settings"),
    )
    loop = SimpleNamespace(manual=False, dt=0.025, released=0)
    loop.get_manual_mode = lambda name: loop.manual
    loop.get_manual_step_size = lambda name: loop.dt
    loop.set_manual_mode = lambda value, name: setattr(loop, "manual", value)
    loop.set_manual_step_size = lambda value, name: setattr(loop, "dt", value)
    modules["omni.kit.loop"]._loop = SimpleNamespace(
        acquire_loop_interface=lambda: loop,
        release_loop_interface=lambda value: setattr(loop, "released", loop.released + 1),
    )
    modules["omni.kit.loop"].runner = loop
    settings = SimpleNamespace(values={"/app/player/useFixedTimeStepping": True})
    settings.get = lambda key: settings.values.get(key)
    settings.set_bool = lambda key, value: settings.values.__setitem__(key, value)
    settings.destroy_item = lambda key: settings.values.pop(key, None)
    modules["carb.settings"].get_settings = lambda: settings
    world = tmp_path / "world.usda"
    stage = Usd.Stage.CreateNew(str(world))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    timeline = _Timeline(5.0)
    app = _App(timeline)
    app.advance = True
    enabled = set()
    manager = SimpleNamespace(
        set_extension_enabled_immediate=lambda name, value: enabled.add(name),
        is_extension_enabled=lambda name: name in enabled,
    )
    modules["omni.kit.app"].get_app = lambda: SimpleNamespace(get_extension_manager=lambda: manager)
    modules["omni.timeline"].get_timeline_interface = lambda: timeline

    class Context:
        current = stage

        async def open_stage_async(self, path):
            self.current = stage
            return True, ""

        def get_stage(self):
            return self.current

        async def close_stage_async(self):
            self.current = None

    context = Context()
    modules["omni.usd"].get_context = lambda: context
    navigation = Ros2NavigationSettings(origin_m=(1.3, -2.7, 0.0), yaw_rad=0.37)
    plan = PedestrianPlan(
        "walker",
        position=(3.14, 4.71, 0.0),
        behavior="walk",
        speed=0.6,
        repeat=True,
        waypoints=(Waypoint((4.14, 4.71, 0.0), wait=3.0, yaw=90.0), Waypoint((3.14, 4.71, 0.0), wait=4.0, yaw=30.0)),
    )
    config = SimpleNamespace(
        validate=lambda: None,
        navigation=SimpleNamespace(settings_path=None),
        pedestrians=SimpleNamespace(enabled=True),
        robots=(),
        settings=SimulationSettings(seed=7),
        pedestrian_plans=(plan,),
    )
    monkeypatch.setattr(runtime, "load_ros2_navigation_settings", lambda path: navigation)
    monkeypatch.setattr(runtime, "resolve_population", lambda config: SimpleNamespace(config=config))
    monkeypatch.setattr(runtime, "_create_app", lambda *args: app)
    monkeypatch.setattr(runtime, "_compose_world", lambda *args: (world, []))
    monkeypatch.setattr(runtime, "configure_physics", lambda *args: None)
    monkeypatch.setattr(runtime, "validate_robots", lambda *args: None)
    monkeypatch.setattr(
        runtime,
        "build_occupancy_grid",
        lambda *args, **kwargs: dict(
            occupancy=np.zeros((200, 200), dtype=np.int8), resolution_m=0.1, origin_xy_m=(-10.0, -10.0), diagnostics={}
        ),
    )
    monkeypatch.setattr(runtime, "save_occupancy_grid", lambda *args: {"npz": tmp_path / "map.npz"})
    monkeypatch.setattr(runtime, "_assets", lambda *args: ({"walker": Path("asset.usd")}, Path("motion.usd")))
    monkeypatch.setattr(runtime, "Ros2Actors", lambda *args: _Actors())
    monkeypatch.setattr(doors, "Ros2Doors", lambda *args, **kwargs: _Doors())
    monkeypatch.setattr(runtime, "build_solver", lambda *args: tmp_path / "solver.so")
    monkeypatch.setattr(runtime, "Ros2Transport", lambda *args: _Transport())
    monkeypatch.setattr(runtime, "select_runtime_prim_path", lambda *args, **kwargs: "/ChosenActors")
    monkeypatch.setattr(runtime, "DynamicObstacleObserver", lambda *args, **kwargs: _Resource())
    return config, plan, app, timeline, enabled


@pytest.mark.parametrize("partial_attachment", [False, True])
def test_rejected_stage_open_stops_setup_and_only_closes_its_own_stage(monkeypatch, tmp_path, partial_attachment):
    config, _, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    context = sys.modules["omni.usd"].get_context()
    composed = context.current
    previous = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(previous, 1.0)
    UsdGeom.SetStageUpAxis(previous, "Z")
    context.current = previous
    physics_calls = []

    async def reject_open(path):
        if partial_attachment:
            context.current = composed
        return False, "SDK rejected the composed USD"

    monkeypatch.setattr(context, "open_stage_async", reject_open)
    monkeypatch.setattr(runtime, "configure_physics", physics_calls.append)
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "run")
    temporary = Path(session._temporary.name)
    try:
        with pytest.raises(RuntimeError, match="SDK rejected the composed USD"):
            session.setup()
        assert physics_calls == [] and session.actors is None and session.transport is None
    finally:
        assert session.close_scene()["stage_closed"]
    assert context.get_stage() == (None if partial_attachment else previous)
    assert not temporary.exists()


def test_setup_captures_native_clock_origin_and_forwards_seed_and_world_plan(monkeypatch, tmp_path):
    config, plan, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    original = plan.to_dict()
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "run")
    try:
        session.setup()
        bootstrap = json.loads((session.output / "bootstrap.json").read_text())
        session.play()
        session.update()
        assert session.transport.requests[0]["time_s"] == pytest.approx(1 / 60)
        assert bootstrap["settings"]["seed"] == 7
        assert bootstrap["settings"]["door_specs"] == [{"name": "door_metadata"}]
        assert session.transport.requests[0]["doors"][0]["name"] == "door_metadata"
        assert session.plans[0].to_dict() == original
        assert bootstrap["plans"][0]["position"] == bootstrap["plans"][0]["waypoints"][-1]["position"]
        assert bootstrap["plans"][0]["waypoints"][0]["yaw"] == pytest.approx(90.0 - math.degrees(0.37))
        assert bootstrap["plans"][0]["waypoints"][0]["wait"] == 3.0
        assert plan.to_dict() == original
    finally:
        session.close_scene()


def test_ros_map_keeps_robot_colliders_available_for_static_or_live_classification(monkeypatch, tmp_path):
    from pedstrain_simulation.simulation.scene.physics import validate_robots

    config, _, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    asset = tmp_path / "robot.usda"
    robot_stage = Usd.Stage.CreateNew(str(asset))
    UsdGeom.SetStageMetersPerUnit(robot_stage, 1.0)
    UsdGeom.SetStageUpAxis(robot_stage, "Z")
    root = UsdGeom.Xform.Define(robot_stage, "/Robot").GetPrim()
    robot_stage.SetDefaultPrim(root)
    body = UsdGeom.Cube.Define(robot_stage, "/Robot/Base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.CollisionAPI.Apply(body)
    joint = UsdPhysics.FixedJoint.Define(robot_stage, "/Robot/WorldAnchor")
    joint.CreateBody1Rel().SetTargets([body.GetPath()])
    robot_stage.GetRootLayer().Save()
    robot = RobotConfig(asset_path=asset, prim_path="/World/Robot")
    config.robots = (robot,)
    stage = sys.modules["omni.usd"].get_context().get_stage()
    UsdGeom.Xform.Define(stage, robot.prim_path).GetPrim().GetReferences().AddReference(str(asset))
    monkeypatch.setattr(runtime, "validate_robots", validate_robots)
    build_map = runtime.build_occupancy_grid
    excluded = []

    def observe_map(*args, **kwargs):
        excluded.extend(kwargs.get("excluded_roots", ()))
        return build_map(*args, **kwargs)

    monkeypatch.setattr(runtime, "build_occupancy_grid", observe_map)
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "run")
    try:
        session.setup()
        assert robot.prim_path not in excluded
    finally:
        session.close_scene()


@pytest.mark.parametrize("tcps,fps", [(24.0, 24.0), (48.0, 25.0)])
def test_control_clock_is_independent_of_usd_timecodes_and_restores_owned_loop(monkeypatch, tmp_path, tcps, fps):
    config, _, app, timeline, _ = _setup_environment(monkeypatch, tmp_path)
    stage = sys.modules["omni.usd"].get_context().get_stage()
    stage.SetTimeCodesPerSecond(tcps)
    stage.SetFramesPerSecond(fps)
    animation = UsdGeom.Xform.Define(stage, "/Animated").AddTranslateOp()
    animation.Set(Gf.Vec3d(0), 0.0)
    animation.Set(Gf.Vec3d(1, 0, 0), tcps)
    original = stage.GetRootLayer().ExportToString()
    loop = sys.modules["omni.kit.loop"].runner
    settings = sys.modules["carb.settings"].get_settings()

    def update():
        app.updates += 1
        if timeline.playing:
            # Kit's fixed player mode advances an animation frame, whereas the
            # Isaac manual runloop supplies a controlled seconds-based delta.
            dt = 1 / stage.GetTimeCodesPerSecond()
            if not settings.get("/app/player/useFixedTimeStepping") and loop.manual:
                dt = loop.dt
            timeline.now += dt

    app.update = update
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "run")
    try:
        session.setup()
        session.play()
        for _ in range(6):
            session.update()
        assert len(session.transport.requests) == 6
        assert [r["dt"] for r in session.transport.requests] == pytest.approx([1 / 60] * 6)
        assert session._last_time == pytest.approx(0.1)
        assert tuple(animation.Get(session._last_time * tcps)) == pytest.approx((0.1, 0.0, 0.0))
        session.pause()
        paused = timeline.get_current_time()
        app.update()
        session.update()
        assert timeline.get_current_time() == paused and len(session.transport.requests) == 6
        assert stage.GetTimeCodesPerSecond() == tcps and stage.GetFramesPerSecond() == fps
        assert stage.GetRootLayer().ExportToString() == original
    finally:
        session.close_scene()
    session.close_scene()
    assert not loop.manual and loop.dt == 0.025 and loop.released == 1
    assert settings.get("/app/player/useFixedTimeStepping") is True


@pytest.mark.parametrize("axis", ["Y", "Z"])
@pytest.mark.parametrize("waypoint_frame", ["stage_world", "world_m"])
def test_setup_converts_explicit_stage_waypoints_once_and_retains_raw_inputs(
    monkeypatch, tmp_path, axis, waypoint_frame
):
    config, metric_plan, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    stage = sys.modules["omni.usd"].get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdGeom.SetStageUpAxis(stage, axis)

    def stage_point(point):
        x, y, z = point
        return tuple(value / 0.01 for value in ((x, z, -y) if axis == "Y" else point))

    raw_plan = (
        replace(
            metric_plan,
            position=stage_point(metric_plan.position),
            waypoints=tuple(replace(w, position=stage_point(w.position)) for w in metric_plan.waypoints),
        )
        if waypoint_frame == "stage_world"
        else metric_plan
    )
    config.pedestrian_plans = (raw_plan,)
    config.navigation = SimpleNamespace(settings_path=None, waypoint_frame=waypoint_frame)
    original = raw_plan.to_dict()
    observed = {}
    build_map = runtime.build_occupancy_grid

    def capture_map(stage, settings, frame, **kwargs):
        observed["map_plans"] = settings["plans"]
        return build_map(stage, settings, frame, **kwargs)

    def create_actors(stage, plans, *args):
        observed["actor_plans"] = plans
        return _Actors()

    monkeypatch.setattr(runtime, "build_occupancy_grid", capture_map)
    monkeypatch.setattr(runtime, "Ros2Actors", create_actors)
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "run")
    try:
        session.setup()
        effective = session.plans[0]
        assert effective.position == pytest.approx(metric_plan.position)
        for actual, expected in zip(effective.waypoints, metric_plan.waypoints):
            assert actual.position == pytest.approx(expected.position)
            assert (actual.wait, actual.yaw) == (expected.wait, expected.yaw)
        assert effective.speed == metric_plan.speed
        assert effective.arrival_tolerance == metric_plan.arrival_tolerance
        assert observed["actor_plans"] == session.plans
        assert observed["map_plans"][0]["position"] == pytest.approx(metric_plan.position)
        bootstrap = json.loads((session.output / "bootstrap.json").read_text())
        assert bootstrap["plans"][0]["position"] == pytest.approx(session.frame.world_to_ros(metric_plan.position))
        assert bootstrap["plans"][0]["waypoints"][0]["position"] == pytest.approx(
            session.frame.world_to_ros(metric_plan.waypoints[0].position)
        )
        assert bootstrap["plans"][0]["waypoints"][0]["yaw"] == pytest.approx(90 - math.degrees(0.37))
        assert bootstrap["input_plans"] == [original]
        assert bootstrap["world_plans"] == [effective.to_dict()]
        assert bootstrap["coordinate_contract"] == session.coordinate_contract
        assert session.coordinate_contract["input_waypoint_frame"] == waypoint_frame
        assert session.coordinate_contract["effective_plans_frame"] == "world_m"
        assert session.input_plans == (raw_plan,)
        assert raw_plan.to_dict() == original
    finally:
        session.close_scene()


def test_ros_waypoint_frame_defaults_to_metric_world_and_accepts_stage_world(tmp_path):
    from pedstrain_simulation.simulation.configuration.loading import load_simulation_config

    assert NavigationConfig().waypoint_frame == "world_m"
    _, profile, _ = _profile_config(tmp_path, enabled=True)
    data = yaml.safe_load(profile.read_text())
    data["navigation"]["waypoint_frame"] = "stage_world"
    profile.write_text(yaml.safe_dump(data))
    assert load_simulation_config(profile).navigation.waypoint_frame == "stage_world"
    for invalid in ("local", None):
        data["navigation"]["waypoint_frame"] = invalid
        profile.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match="navigation.waypoint_frame"):
            load_simulation_config(profile)
    assert NavigationConfig(waypoint_frame="stage_world").backend == "ros2"


def test_all_static_route_goals_are_checked_before_creating_actors_or_ros(monkeypatch, tmp_path):
    config, plan, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    config.pedestrian_plans = (replace(plan, waypoints=(Waypoint((100.0, 100.0, 0.0)),)),)

    def forbidden(*args, **kwargs):
        raise AssertionError("Actors and ROS cannot start before static route validation.")

    monkeypatch.setattr(runtime, "Ros2Actors", forbidden)
    monkeypatch.setattr(runtime, "Ros2Transport", forbidden)
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "unreachable")
    try:
        with pytest.raises(ValueError, match="walker waypoint 1: static map validation failed"):
            session.setup()
        assert session.actors is None and session.transport is None
    finally:
        session.close_scene()


@pytest.mark.parametrize("case", ["outside_stand", "outside_walk", "occupied_stand"])
def test_fixed_spawns_are_static_disk_clear_before_actors_or_ros_start(monkeypatch, tmp_path, case):
    config, plan, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    navigation = replace(runtime.load_ros2_navigation_settings(None), origin_m=(0.0, 0.0, 0.0), yaw_rad=0.0)
    monkeypatch.setattr(runtime, "load_ros2_navigation_settings", lambda path: navigation)
    point = (0.0, 0.0, 0.0) if case == "occupied_stand" else (100.0, 100.0, 0.0)
    config.pedestrian_plans = (
        replace(
            plan,
            position=point,
            behavior="walk" if case == "outside_walk" else "stand",
            repeat=False,
            waypoints=(Waypoint((point[0] + 0.1, point[1], point[2])),) if case == "outside_walk" else (),
        ),
    )
    if case == "occupied_stand":
        cells = np.zeros((200, 200), dtype=np.int8)
        cells[95:105, 95:105] = 100
        monkeypatch.setattr(
            runtime,
            "build_occupancy_grid",
            lambda *args, **kwargs: dict(occupancy=cells, resolution_m=0.1, origin_xy_m=(-10.0, -10.0), diagnostics={}),
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("Actors and ROS cannot start before fixed spawn validation.")

    monkeypatch.setattr(runtime, "Ros2Actors", forbidden)
    monkeypatch.setattr(runtime, "Ros2Transport", forbidden)
    original = config.pedestrian_plans
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "invalid_fixed")
    try:
        with pytest.raises(ValueError, match="fixed spawn.*disk-clear"):
            session.setup()
        assert config.pedestrian_plans == original
        assert session.actors is None and session.transport is None
    finally:
        session.close_scene()


@pytest.mark.parametrize("distance", [0.4, 0.5, 0.6])
def test_fixed_spawn_physical_overlap_is_rejected_but_contact_and_comfort_deficits_are_preserved(
    monkeypatch, tmp_path, distance
):
    config, plan, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    navigation = replace(runtime.load_ros2_navigation_settings(None), origin_m=(0.0, 0.0, 0.0), yaw_rad=0.0)
    monkeypatch.setattr(runtime, "load_ros2_navigation_settings", lambda path: navigation)
    config.pedestrian_plans = (
        replace(plan, position=(0.0, 0.0, 0.0), behavior="stand", repeat=False, waypoints=()),
        replace(plan, name="other", position=(distance, 0.0, 0.0), behavior="stand", repeat=False, waypoints=()),
    )
    original = config.pedestrian_plans
    if distance < 2 * navigation.body_radius_m:

        def forbidden(*args, **kwargs):
            raise AssertionError("Actors cannot start with overlapping fixed bodies.")

        monkeypatch.setattr(runtime, "Ros2Actors", forbidden)
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "fixed_pair")
    try:
        if distance < 2 * navigation.body_radius_m:
            with pytest.raises(ValueError, match="fixed spawn.*overlap"):
                session.setup()
            assert session.actors is None and session.transport is None
        else:
            session.setup()
            assert tuple(p.position for p in session.plans) == tuple(p.position for p in original)
        assert config.pedestrian_plans == original
    finally:
        session.close_scene()


@pytest.mark.parametrize("tolerance", [0.10, 0.25])
@pytest.mark.parametrize("closed", [False, True])
def test_opt_in_fixed_spawn_correction_uses_only_safe_centers_within_its_radius(
    monkeypatch, tmp_path, tolerance, closed
):
    config, plan, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    navigation = replace(
        runtime.load_ros2_navigation_settings(None), origin_m=(0, 0, 0), yaw_rad=0.0, spawn_tolerance_m=tolerance
    )
    monkeypatch.setattr(runtime, "load_ros2_navigation_settings", lambda path: navigation)
    requested = (0.15, 0.05, 0.0)
    waypoints = (Waypoint((1.0, 0.05, 0.0)), Waypoint(requested if closed else (0.35, 0.05, 0.0)))
    config.pedestrian_plans = (replace(plan, position=requested, arrival_tolerance=0.25, waypoints=waypoints),)
    cells = np.zeros((200, 200), dtype=np.int8)
    cells[:, 95:100] = 100  # Wall ends at x=0, so disk centers need the configured clearance in x.
    monkeypatch.setattr(
        runtime,
        "build_occupancy_grid",
        lambda *args, **kwargs: dict(occupancy=cells, resolution_m=0.1, origin_xy_m=(-10, -10), diagnostics={}),
    )
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "projected_spawn")
    try:
        if tolerance < 0.13:
            with pytest.raises(ValueError, match="No safe fixed spawn within spawn_tolerance_m"):
                session.setup()
            assert session.actors is None
        else:
            session.setup()
            actual = session.plans[0].position
            assert actual == pytest.approx((0.35, 0.05, 0.0))
            assert math.dist(actual, requested) <= tolerance
            bootstrap = json.loads((session.output / "bootstrap.json").read_text())
            assert bootstrap["spawn_adjustments_world_m"][plan.name] == pytest.approx((0.2, 0, 0))
            assert bootstrap["plans"][0]["authored_closed_route"] is closed
            assert session.plans[0].waypoints == waypoints
        assert config.pedestrian_plans[0].position == requested
    finally:
        session.close_scene()


@pytest.mark.parametrize("overhead", [False, True])
def test_fixed_spawns_respect_current_physical_footprints_without_predicting_velocity(monkeypatch, tmp_path, overhead):
    config, plan, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    navigation = replace(runtime.load_ros2_navigation_settings(None), origin_m=(0.0, 0.0, 0.0), yaw_rad=0.0)
    monkeypatch.setattr(runtime, "load_ros2_navigation_settings", lambda path: navigation)
    config.pedestrian_plans = (replace(plan, position=(0.0, 0.0, 0.0), behavior="stand", repeat=False, waypoints=()),)

    class Observations(_Resource):
        def sample(self):
            return [
                {
                    "prim_path": "/current_physical_leaf",
                    "position": [0.0, 0.0, 0.0],
                    "radius_m": 0.2,
                    "footprint_xy_m": [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
                    "z_min_m": 3.0 if overhead else 0.0,
                    "z_max_m": 4.0 if overhead else 1.8,
                    "velocity": [0.0, 0.0, 0.0],
                    "velocity_valid": False,
                }
            ]

    monkeypatch.setattr(runtime, "DynamicObstacleObserver", lambda *args, **kwargs: Observations())
    if not overhead:

        def forbidden(*args, **kwargs):
            raise AssertionError("Actors cannot start inside an observed physical footprint.")

        monkeypatch.setattr(runtime, "Ros2Actors", forbidden)
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "fixed_physical")
    try:
        if overhead:
            session.setup()
            assert session.plans[0].position == (0.0, 0.0, 0.0)
        else:
            with pytest.raises(ValueError, match="fixed spawn.*physical"):
                session.setup()
            assert session.actors is None and session.transport is None
    finally:
        session.close_scene()


def test_random_spawn_reserves_a_later_explicit_spawn_without_moving_its_point(monkeypatch, tmp_path):
    config, _, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    navigation = replace(
        runtime.load_ros2_navigation_settings(None), resolution_m=0.1, origin_m=(0.0, 0.0, 0.0), yaw_rad=0.0
    )
    monkeypatch.setattr(runtime, "load_ros2_navigation_settings", lambda path: navigation)
    # Independently verified first seed-7 free cell in this fixture's 0.1m grid.
    fixed = PedestrianPlan("fixed", position=(1.45, 6.15, 0.0), behavior="stand")
    config.pedestrian_plans = (PedestrianPlan("random", behavior="wander"), fixed)
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "mixed_spawns")
    try:
        session.setup()
        random_plan, fixed_plan = session.plans
        assert fixed_plan is fixed and fixed_plan.position == (1.45, 6.15, 0.0)
        minimum = 2 * navigation.body_radius_m + navigation.pair_clearance_m
        assert math.dist(random_plan.position, fixed_plan.position) > minimum
    finally:
        session.close_scene()


@pytest.mark.parametrize("overhead", [False, True])
def test_random_spawns_respect_current_physical_shapes_in_their_height_band(monkeypatch, tmp_path, overhead):
    config, _, _, _, _ = _setup_environment(monkeypatch, tmp_path)
    navigation = replace(
        runtime.load_ros2_navigation_settings(None), resolution_m=0.1, origin_m=(0.0, 0.0, 0.0), yaw_rad=0.0
    )
    monkeypatch.setattr(runtime, "load_ros2_navigation_settings", lambda path: navigation)
    config.pedestrian_plans = (PedestrianPlan("random", behavior="wander"),)
    # This box covers the independently verified first seed-7 candidate.
    lower, upper = (1.0, 5.7), (1.9, 6.6)

    class Observations(_Resource):
        def sample(self):
            return [
                {
                    "prim_path": "/movable_shape",
                    "position": [1.45, 6.15, 0.0],
                    "radius_m": 0.7,
                    "footprint_xy_m": [
                        [lower[0], lower[1]],
                        [upper[0], lower[1]],
                        [upper[0], upper[1]],
                        [lower[0], upper[1]],
                    ],
                    "z_min_m": 3.0 if overhead else 0.0,
                    "z_max_m": 4.0 if overhead else 1.8,
                    "velocity": [0.0, 0.0, 0.0],
                    "velocity_valid": False,
                }
            ]

    monkeypatch.setattr(runtime, "DynamicObstacleObserver", lambda *args, **kwargs: Observations())
    session = runtime.Ros2SimulationSession(config, output=tmp_path / "physical_spawns")
    try:
        session.setup()
        point = session.plans[0].position
        if overhead:
            assert point == pytest.approx((1.45, 6.15, 0.0))
        else:
            distance = math.hypot(
                *(max(low - value, 0.0, value - high) for value, low, high in zip(point, lower, upper))
            )
            assert distance > navigation.body_radius_m + navigation.map_clearance_m
    finally:
        session.close_scene()


def _ros_transport(monkeypatch, tmp_path, *, send_future=False, first_reply=None, publish_clock=True):
    names = (
        "rclpy",
        "rclpy.context",
        "rclpy.executors",
        "rclpy.node",
        "rclpy.qos",
        "nav_msgs",
        "nav_msgs.msg",
        "rosgraph_msgs",
        "rosgraph_msgs.msg",
        "std_msgs",
        "std_msgs.msg",
    )
    modules = _modules(monkeypatch, names)
    record = SimpleNamespace(
        contexts=[], nodes=[], executors=[], messages=[], pending=[], process=None, first_reply_sent=False
    )

    class Context:
        active = False

        def __init__(self):
            record.contexts.append(self)

        def ok(self):
            return self.active

        def shutdown(self):
            self.active = False

    def init(*, context, domain_id):
        context.active = True
        context.domain = domain_id

    modules["rclpy"].init = init
    modules["rclpy"].spin_once = lambda *args, **kwargs: pytest.fail("Private context used the global ROS executor")
    modules["rclpy.context"].Context = Context

    class String:
        def __init__(self, data=""):
            self.data = data

    class Node:
        def __init__(self, name, *, namespace, context):
            self.context, self.subscriptions, self.destroyed = context, {}, False
            record.nodes.append(self)

        def create_publisher(self, message_type, topic, qos):
            def publish(message):
                record.messages.append((topic, message))
                if topic.endswith("/state"):
                    record.pending.append(json.loads(message.data))

            return SimpleNamespace(publish=publish)

        def create_subscription(self, message_type, topic, callback, qos):
            self.subscriptions[topic] = callback

        def destroy_node(self):
            self.destroyed = True

    class Executor:
        def __init__(self, *, context):
            self.context = context
            self.closed = False
            record.executors.append(self)

        def add_node(self, node):
            assert node.context is self.context
            self.node = node

        def remove_node(self, node):
            assert node is self.node

        def shutdown(self, timeout_sec):
            self.closed = True

        def spin_once(self, timeout_sec):
            for topic, callback in self.node.subscriptions.items():
                if topic.endswith("/map"):
                    callback(SimpleNamespace(info=SimpleNamespace(width=1, height=1), data=[0]))
            if record.pending:
                request = record.pending.pop(0)
                callback = next(value for key, value in self.node.subscriptions.items() if key.endswith("/command"))
                if send_future and request["sequence"] == 0:
                    callback(String(json.dumps({"sequence": 1, "time_s": 999.0, "agents": []})))
                if first_reply is not None and not record.first_reply_sent:
                    record.first_reply_sent = True
                    record.pending.insert(0, request)
                    callback(String(json.dumps(first_reply(request))))
                    return
                callback(
                    String(
                        json.dumps(
                            dict(sequence=request["sequence"], time_s=request["time_s"], agents=request["agents"])
                        )
                    )
                )

    modules["rclpy.node"].Node = Node
    modules["rclpy.executors"].SingleThreadedExecutor = Executor
    modules["rclpy.qos"].DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL="transient")
    modules["rclpy.qos"].QoSProfile = lambda **kwargs: SimpleNamespace(**kwargs)
    modules["nav_msgs.msg"].OccupancyGrid = type("OccupancyGrid", (), {})
    modules["rosgraph_msgs.msg"].Clock = lambda: SimpleNamespace(clock=SimpleNamespace(sec=0, nanosec=0))
    modules["std_msgs.msg"].String = String

    class Process:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

        def kill(self):
            self.returncode = -9

    def start(*args, **kwargs):
        record.process = Process()
        return record.process

    monkeypatch.setattr(transport.subprocess, "Popen", start)
    settings = {"ros_setup_path": "/unused", "ros_domain_id": 59, "ros_namespace": "/test", "request_timeout_s": 0.5}
    np.savez(tmp_path / "map.npz", occupancy=np.zeros((1, 1), dtype=np.int8), resolution_m=0.1, origin_xy_m=[0, 0])
    (tmp_path / "bootstrap.json").write_text(json.dumps({"plans": [], "map_npz": "map.npz"}))
    client = transport.Ros2Transport(
        tmp_path / "bootstrap.json", tmp_path / "solver.so", settings, tmp_path, publish_clock=publish_clock
    )
    return client, record


def test_transport_uses_one_owned_context_executor_and_cleans_it_after_a_clocked_exchange(monkeypatch, tmp_path):
    client, record = _ros_transport(monkeypatch, tmp_path)
    try:
        response = client.step(1.9999999999, 1 / 60, [])
        assert response["sequence"] == 0 and response["time_s"] == 1.9999999999
        clock = next(message for topic, message in record.messages if topic == "/clock")
        assert (clock.clock.sec, clock.clock.nanosec) == (2, 0)
        assert record.nodes[0].context is record.executors[0].context is record.contexts[0]
        assert record.contexts[0].domain == 59
        monitor = json.loads(client.monitor_manifest.read_text())
        assert monitor["namespace"] == client.namespace and monitor["ros_domain_id"] == 59
        assert monitor["topics"]["map"] == client.namespace + "/map"
    finally:
        client.close()
    assert record.process.returncode == 0 and record.nodes[0].destroyed
    assert record.executors[0].closed and not record.contexts[0].ok()


def test_unsolicited_future_reply_cannot_satisfy_a_later_state_sequence(monkeypatch, tmp_path):
    client, record = _ros_transport(monkeypatch, tmp_path, send_future=True)
    try:
        assert client.step(1 / 60, 1 / 60, [])["time_s"] == 1 / 60
        assert client.step(2 / 60, 1 / 60, [])["time_s"] == 2 / 60
        assert record.pending == []
    finally:
        client.close()


def test_training_transport_uses_request_time_without_publishing_global_clock(monkeypatch, tmp_path):
    client, record = _ros_transport(monkeypatch, tmp_path, publish_clock=False)
    try:
        assert client.step(1 / 60, 1 / 60, [])["time_s"] == pytest.approx(1 / 60)
        assert [topic for topic, _ in record.messages] == [client.namespace + "/state"]
    finally:
        client.close()


@pytest.mark.parametrize("reply_time", [999.0, True, "0.1", None, math.nan])
def test_transport_waits_for_the_matching_time_as_well_as_sequence(monkeypatch, tmp_path, reply_time):
    client, record = _ros_transport(
        monkeypatch,
        tmp_path,
        first_reply=lambda request: dict(sequence=request["sequence"], time_s=reply_time, agents=[]),
    )
    try:
        assert client.step(0.1, 0.1, [])["time_s"] == 0.1
        assert not record.pending
    finally:
        client.close()


@pytest.mark.parametrize("include_time", [False, True])
def test_transport_keeps_geometry_error_replies_including_legacy_ones_without_time(monkeypatch, tmp_path, include_time):
    def reply(request):
        return {
            "sequence": request["sequence"],
            "error": "No reachable geometry",
            "agents": [],
            **({"time_s": request["time_s"]} if include_time else {}),
        }

    client, _ = _ros_transport(monkeypatch, tmp_path, first_reply=reply)
    try:
        with pytest.raises(RuntimeError, match="No reachable geometry"):
            client.step(0.1, 0.1, [])
    finally:
        client.close()


def _ros_navigation_node(monkeypatch, tmp_path, requests, *, failed_observer=None):
    modules = _modules(
        monkeypatch,
        (
            "rclpy",
            "rclpy.node",
            "rclpy.executors",
            "rclpy.parameter",
            "rclpy.qos",
            "rclpy.time",
            "nav_msgs",
            "nav_msgs.msg",
            "geometry_msgs",
            "geometry_msgs.msg",
            "std_msgs",
            "std_msgs.msg",
        ),
    )
    messages, warnings = [], []
    observer_failed = False
    agent = {"name": "visitor", "position": [0.0, 0.0, 0.0], "yaw": 0.0}

    class Engine:
        def __init__(self, *args):
            self.names = ["visitor"]
            self.routes = {"visitor": SimpleNamespace(path=[(0.0, 0.0), (1.0, 0.0)])}
            self.bodies = {"visitor": agent}
            self.distance = 0.0

        def step(self, time_s, dt, agents, *args):
            if agents[0].get("geometry_error"):
                raise ValueError("No reachable geometry")
            self.distance += dt
            return {"time_s": time_s, "agents": [{**agent, "position": [self.distance, 0.0, 0.0]}]}

        def close(self):
            pass

    class Node:
        def __init__(self, *args, **kwargs):
            self.context = SimpleNamespace(ok=lambda: True)

        def create_publisher(self, message_type, topic, qos):
            def publish(message):
                nonlocal observer_failed
                if failed_observer == "path" and topic.endswith("/path") and not observer_failed:
                    observer_failed = True
                    raise RuntimeError("Path observer failed")
                messages.append((topic, message))

            return SimpleNamespace(publish=publish)

        def create_subscription(self, *args):
            return object()

        def get_logger(self):
            return SimpleNamespace(
                info=lambda *args: None,
                error=lambda *args: None,
                warning=lambda message, **kwargs: warnings.append(message),
            )

        def destroy_node(self):
            pass

    def observe(*args):
        nonlocal observer_failed
        if failed_observer == "marker" and not observer_failed:
            observer_failed = True
            raise RuntimeError("Marker observer failed")

    modules["rclpy"].init = lambda **kwargs: None
    modules["rclpy"].try_shutdown = lambda: None
    modules["rclpy"].spin = lambda instance: [
        instance.on_state(SimpleNamespace(data=json.dumps(request))) for request in requests
    ]
    modules["rclpy.node"].Node = Node
    modules["rclpy.executors"].ExternalShutdownException = type("ExternalShutdownException", (Exception,), {})
    modules["rclpy.parameter"].Parameter = lambda *args, **kwargs: None
    modules["rclpy.qos"].QoSProfile = SimpleNamespace
    modules["rclpy.qos"].DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL="transient")
    modules["rclpy.qos"].ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable")
    modules["rclpy.time"].Time = lambda **kwargs: SimpleNamespace(to_msg=lambda: SimpleNamespace())
    modules["std_msgs.msg"].String = lambda **kwargs: SimpleNamespace(**kwargs)
    modules["nav_msgs.msg"].Path = lambda: SimpleNamespace(header=SimpleNamespace(), poses=[])
    modules["nav_msgs.msg"].OccupancyGrid = lambda: SimpleNamespace(
        header=SimpleNamespace(),
        info=SimpleNamespace(origin=SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace())),
    )
    modules["geometry_msgs.msg"].PoseStamped = lambda: SimpleNamespace(
        header=SimpleNamespace(), pose=SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace())
    )
    monkeypatch.setattr(node, "NavigationEngine", Engine)
    monkeypatch.setattr(node, "NavigationMonitor", lambda *args: SimpleNamespace(publish=observe))
    np.savez(tmp_path / "map.npz", occupancy=np.zeros((30, 30), dtype=np.int8), resolution_m=0.1, origin_xy_m=[0, 0])
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text(json.dumps({"plans": [], "agents": [agent], "settings": {}, "map_npz": "map.npz"}))
    node.main(["--bootstrap", str(bootstrap), "--library", str(tmp_path / "unused.so")])
    return (
        [json.loads(message.data) for topic, message in messages if topic == "command"],
        [message for topic, message in messages if topic.endswith("/path")],
        warnings,
    )


def _navigation_request(sequence=0, time_s=0.1, dt=0.1, **state):
    return {
        "sequence": sequence,
        "time_s": time_s,
        "dt": dt,
        "agents": [{"name": "visitor", "position": [0.0, 0.0, 0.0], "yaw": 0.0, **state}],
    }


@pytest.mark.parametrize("observer", ["path", "marker"])
def test_observation_failure_cannot_publish_a_second_navigation_answer_or_lose_a_path(monkeypatch, tmp_path, observer):
    request = _navigation_request()
    replies, paths, warnings = _ros_navigation_node(
        monkeypatch, tmp_path, [request, request, _navigation_request(1, 0.2)], failed_observer=observer
    )
    assert len(replies) == 3 and all(not reply.get("error") for reply in replies)
    assert replies[0] == replies[1] and replies[2]["agents"][0]["position"][0] == pytest.approx(0.2)
    assert len(paths) == 1 and len(paths[0].poses) == 2 and warnings


@pytest.mark.parametrize("time_s,dt", [(True, 0.1), ("0.1", 0.1), (-0.1, 0.1), (0.1, True), (0.1, "0.1"), (0.1, 0.0)])
def test_node_rejects_invalid_clock_values_before_moving_a_body(monkeypatch, tmp_path, time_s, dt):
    replies, _, _ = _ros_navigation_node(
        monkeypatch, tmp_path, [_navigation_request(0, time_s, dt), _navigation_request(1, 0.2)]
    )
    assert len(replies) == 2 and replies[0].get("error")
    assert replies[1]["agents"][0]["position"][0] == pytest.approx(0.1)


@pytest.mark.parametrize("time_s,dt", [(0.2, 0.1), (0.1, 0.1), (0.4, 0.1), (0.3, 0.2)])
def test_node_rejects_repeated_rewound_or_incoherent_time_without_advancing_the_route(
    monkeypatch, tmp_path, time_s, dt
):
    replies, _, _ = _ros_navigation_node(
        monkeypatch,
        tmp_path,
        [_navigation_request(0, 0.2), _navigation_request(1, time_s, dt), _navigation_request(2, 0.3)],
    )
    assert len(replies) == 3 and replies[1].get("error")
    assert replies[2]["agents"][0]["position"][0] == pytest.approx(0.2)


def test_node_accepts_nonzero_initial_clock_and_duplicate_retry_only_with_the_same_snapshot(monkeypatch, tmp_path):
    request = _navigation_request(0, 0.37, 0.01)
    changed = {**request, "obstacles": [{"name": "changed snapshot"}]}
    replies, _, _ = _ros_navigation_node(
        monkeypatch, tmp_path, [request, request, changed, _navigation_request(1, 0.39, 0.02)]
    )
    assert replies[0] == replies[1] and replies[2].get("error")
    assert replies[3]["agents"][0]["position"][0] == pytest.approx(0.03)


def test_node_geometry_error_echoes_the_valid_request_time(monkeypatch, tmp_path):
    replies, _, _ = _ros_navigation_node(monkeypatch, tmp_path, [_navigation_request(geometry_error=True)])
    assert len(replies) == 1 and replies[0]["error"] == "No reachable geometry" and replies[0]["time_s"] == 0.1


def test_transport_timeout_retains_the_exact_request_and_the_original_failure(monkeypatch, tmp_path):
    client, record = _ros_transport(monkeypatch, tmp_path)

    def timeout(*args, **kwargs):
        raise TimeoutError("deliberate RPC timeout")

    monkeypatch.setattr(client, "_wait", timeout)
    try:
        with pytest.raises(TimeoutError, match="deliberate RPC timeout"):
            client.step(1.0, 1 / 60, [], door_states=[{"name": "door", "fully_open": False}])
        request = json.loads((tmp_path / "last_failed_request.json").read_text())
        sent = next(message for topic, message in record.messages if topic.endswith("/state"))
        assert request == json.loads(sent.data)
        assert request["sequence"] == 0 and request["doors"][0]["name"] == "door"
    finally:
        client.close()


def _profile_config(tmp_path, *, enabled=False, bad_navigation=None):
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n")
    characters = tmp_path / "characters"
    characters.mkdir(exist_ok=True)
    (characters / "M_actor.usda").write_text("#usda 1.0\n")
    motion = tmp_path / "motion.usda"
    motion.write_text("#usda 1.0\n")
    navigation = tmp_path / "navigation.yaml"
    if bad_navigation != "missing":
        navigation.write_text("invalid: [" if bad_navigation == "invalid" else "{}\n")
    data = {
        "schema_version": 1,
        "scene": {"asset_path": str(scene)},
        "simulation": {"duration": 0.05, "seed": 7, "report_path": "status.json"},
        "navigation": {"settings_path": str(navigation)},
        "pedestrians": {
            "count": 1,
            "assets": str(characters) if enabled else "missing_characters",
            "motion_library": str(motion) if enabled else "missing_motion.usda",
        },
    }
    profile = tmp_path / "simulation.yaml"
    profile.write_text(yaml.safe_dump(data))
    argv = ["--simulation-config", str(profile)]
    if not enabled:
        argv.append("--no-pedestrians")
    return _simulation_from_args(create_parser().parse_args(argv)), profile, navigation


@pytest.mark.parametrize("bad_navigation", ["missing", "invalid"])
def test_no_pedestrians_cli_does_not_read_unused_navigation_or_motion_inputs(tmp_path, bad_navigation):
    config, _, _ = _profile_config(tmp_path, bad_navigation=bad_navigation)
    assert config.navigation.backend == "ros2" and not config.pedestrians.enabled
    assert config.settings.duration == 0.05


def test_disabled_inspection_omits_unread_ros_settings_and_population_dependencies(tmp_path):
    config, profile, navigation = _profile_config(tmp_path, bad_navigation="invalid")
    summary = _inspect_simulation_config(config, profile)
    assert summary["pedestrian_controller"] == "disabled" and summary["pedestrian_parameters"] is None
    assert summary["pedestrian_effective_avoidance"] is None and summary["pedestrians"] == 0
    assert summary["ros2_clearance"] is None
    assert str(navigation) not in summary["dependencies"]["configuration_files"]


@pytest.mark.parametrize("enabled", [False, True])
def test_app_checks_for_rclpy_only_when_the_population_needs_ros(monkeypatch, tmp_path, enabled):
    modules = _modules(monkeypatch, ("isaacsim",))
    modules["isaacsim"].__file__ = str(tmp_path / "isaacsim/__init__.py")
    calls = []
    modules["isaacsim"].SimulationApp = lambda *args, **kwargs: calls.append((args, kwargs)) or object()
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda name: None)
    config = SimpleNamespace(pedestrians=SimpleNamespace(enabled=enabled), settings=SimulationSettings())
    if enabled:
        with pytest.raises(RuntimeError, match="Source"):
            runtime._create_app(config, Ros2NavigationSettings(), None)
        assert calls == []
    else:
        runtime._create_app(config, Ros2NavigationSettings(), None)
        assert len(calls) == 1


def test_disabled_runtime_runs_duration_without_mapping_actors_ros_or_unused_settings(monkeypatch, tmp_path):
    _, _, app, _, enabled = _setup_environment(monkeypatch, tmp_path)
    config, _, _ = _profile_config(tmp_path, bad_navigation="invalid")

    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled population reached a navigation or character dependency")

    for name in (
        "load_ros2_navigation_settings",
        "build_occupancy_grid",
        "save_occupancy_grid",
        "Ros2Actors",
        "Ros2Transport",
        "DynamicObstacleObserver",
        "build_solver",
        "_assets",
    ):
        monkeypatch.setattr(runtime, name, forbidden)
    monkeypatch.setattr(doors, "Ros2Doors", forbidden)
    runtime.run_ros2_simulation(config)
    report = json.loads(config.settings.report_path.read_text())
    assert report["passed"] and report["simulation_time_s"] == pytest.approx(0.05)
    assert report["controller"] == "disabled" and report["pedestrian_schedules"] == []
    assert report["map"] == {} and report["ros_namespace"] is None and report["actors"] == {}
    assert report["cleanup"]["stage_closed"] and report["cleanup"]["temporary_removed"]
    assert not app.is_running() and "omni.anim.retarget.core" not in enabled


@pytest.mark.parametrize("destination", ["result", "summary", "console"])
def test_run_output_failure_still_attempts_both_reports_and_closes_the_app(monkeypatch, tmp_path, destination):
    _, _, app, _, _ = _setup_environment(monkeypatch, tmp_path)
    config, _, _ = _profile_config(tmp_path, bad_navigation="invalid")
    writes = []
    original_write = Path.write_text

    def write(path, *args, **kwargs):
        writes.append(path)
        if (destination == "result" and path.name == "result.json") or (
            destination == "summary" and path == config.settings.report_path
        ):
            raise OSError(f"{destination} output failure")
        return original_write(path, *args, **kwargs)

    if destination == "console":
        import builtins

        def output(*args, **kwargs):
            if args and str(args[0]).startswith("ROS2_SIMULATION_RESULT"):
                raise OSError("console output failure")
            return builtins.print(*args, **kwargs)

        monkeypatch.setattr(runtime, "print", output, raising=False)
    monkeypatch.setattr(Path, "write_text", write)
    with pytest.raises(OSError, match=f"{destination} output failure"):
        runtime.run_ros2_simulation(config)
    assert not app.is_running() and app.exit_code == 1
    assert any(path.name == "result.json" for path in writes) and config.settings.report_path in writes
    readable = config.settings.report_path if destination == "result" else next(tmp_path.rglob("result.json"))
    assert json.loads(readable.read_text())["passed"] is False


def test_run_preserves_the_original_failure_when_every_finalizer_fails(monkeypatch, tmp_path):
    _, _, app, _, _ = _setup_environment(monkeypatch, tmp_path)
    config, _, _ = _profile_config(tmp_path, bad_navigation="invalid")
    primary = RuntimeError("original simulation failure")
    setup, cleanup = runtime.Ros2SimulationSession.setup, runtime.Ros2SimulationSession.close_scene

    def failed_setup(session):
        setup(session)
        raise primary

    def failed_cleanup(session):
        cleanup(session)
        raise RuntimeError("scene cleanup failure")

    closed = []

    def failed_close(*, exit_code):
        closed.append(exit_code)
        raise RuntimeError("application close failure")

    def failed_write(path, *args, **kwargs):
        raise OSError(f"report write failure: {path.name}")

    monkeypatch.setattr(runtime.Ros2SimulationSession, "setup", failed_setup)
    monkeypatch.setattr(runtime.Ros2SimulationSession, "close_scene", failed_cleanup)
    monkeypatch.setattr(app, "close", failed_close)
    monkeypatch.setattr(Path, "write_text", failed_write)
    with pytest.raises(RuntimeError, match="original simulation failure") as caught:
        runtime.run_ros2_simulation(config)
    assert caught.value is primary and closed == [1]
    notes = "\n".join(getattr(primary, "__notes__", []))
    for failure in (
        "scene cleanup failure",
        "report write failure: result.json",
        "report write failure: status.json",
        "application close failure",
    ):
        assert failure in notes


def test_application_close_failure_marks_both_completed_reports_failed(monkeypatch, tmp_path):
    _, _, app, _, _ = _setup_environment(monkeypatch, tmp_path)
    config, _, _ = _profile_config(tmp_path, bad_navigation="invalid")
    failure = RuntimeError("application close failure")

    def close(*, exit_code):
        assert exit_code == 0
        raise failure

    monkeypatch.setattr(app, "close", close)
    with pytest.raises(RuntimeError, match="application close failure") as caught:
        runtime.run_ros2_simulation(config)
    assert caught.value is failure
    for path in (config.settings.report_path, *tmp_path.rglob("result.json")):
        report = json.loads(path.read_text())
        assert report["passed"] is False and "application close failure" in report["error"]


def test_mixed_nav_container_with_enabled_collision_is_not_deactivated(tmp_path):
    source = tmp_path / "mixed.usda"
    stage = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    stage.DefinePrim("/World/Navigation", "NavMeshVolume")
    collider = UsdGeom.Cube.Define(stage, "/World/Navigation/PhysicalWall")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    stage.GetRootLayer().Save()
    original = source.read_bytes()
    config = SimpleNamespace(
        scene=SimpleNamespace(asset_path=source, layers=()), robots=(), pedestrians=SimpleNamespace(enabled=True)
    )
    directory = tmp_path / "composed"
    directory.mkdir()
    with pytest.raises(ValueError, match="(?:[Cc]ollision|[Pp]hysic)"):
        runtime._compose_world(config, directory)
    assert source.read_bytes() == original


@pytest.mark.parametrize("pedestrians_enabled", [False, True])
def test_composition_suppresses_metadata_door_scripts_only_when_ros_owns_doors(tmp_path, pedestrians_enabled):
    from pxr import Sdf

    source = tmp_path / "door_scene.usda"
    stage = Usd.Stage.CreateNew(str(source))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    stage.DefinePrim("/World/Doors")
    configuration = stage.DefinePrim("/World/Doors/Configuration")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/LeafJoint")
    configuration.CreateRelationship("automaticDoor:joint").SetTargets([joint.GetPath()])
    controller = stage.DefinePrim("/World/Doors/Graph/Script")
    script = "from pxr import UsdPhysics\n# automaticDoor:joint\nUsdPhysics.DriveAPI(joint, 'angular')\n"
    controller.CreateAttribute("inputs:script", Sdf.ValueTypeNames.String).Set(script)
    observer = stage.DefinePrim("/World/Observer")
    observer.CreateAttribute("inputs:script", Sdf.ValueTypeNames.String).Set(script)
    legacy = stage.DefinePrim("/World/LegacyPedestrianScript")
    old_script = "import pedstrain_simulation.simulation.behaviors.character_body_colliders"
    legacy.CreateAttribute("inputs:script", Sdf.ValueTypeNames.String).Set(old_script)
    stage.GetRootLayer().Save()
    original = source.read_bytes()
    config = SimpleNamespace(
        scene=SimpleNamespace(asset_path=source, layers=()),
        robots=(),
        pedestrians=SimpleNamespace(enabled=pedestrians_enabled),
    )
    directory = tmp_path / "composed"
    directory.mkdir()
    world, suppressed = runtime._compose_world(config, directory)
    composed = Usd.Stage.Open(str(world))
    path = str(controller.GetPath()) + ".inputs:script"
    assert (path in suppressed) is pedestrians_enabled
    assert (composed.GetAttributeAtPath(path).Get() != script) is pedestrians_enabled
    assert composed.GetPrimAtPath(observer.GetPath()).GetAttribute("inputs:script").Get() == script
    legacy_path = legacy.GetPath().AppendProperty("inputs:script")
    assert str(legacy_path) in suppressed
    assert composed.GetAttributeAtPath(legacy_path).Get() != old_script
    assert source.read_bytes() == original
