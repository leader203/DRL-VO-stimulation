# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression checks for final configuration selection and offline previews."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from pedstrain_simulation.simulation.cli.arguments import create_parser
from pedstrain_simulation.simulation.cli.overrides import _simulation_from_args
from pedstrain_simulation.simulation.configuration import PedestrianPlan, SceneConfig, load_simulation_config
from pedstrain_simulation.simulation.configuration.scenarios import compile_scenario_config
from pedstrain_simulation.simulation.configuration.scenarios.models import Waypoint
from pedstrain_simulation.simulation.projects import inspect_simulation_config


@pytest.mark.parametrize("points", [[(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]])
def test_initial_position_is_not_an_interior_or_only_walking_target(points):
    with pytest.raises(ValueError, match="initial position"):
        PedestrianPlan(
            "invalid",
            position=(0.0, 0.0, 0.0),
            behavior="walk",
            repeat=True,
            waypoints=tuple(Waypoint(point) for point in points),
        )


@pytest.fixture
def scenario(tmp_path: Path) -> tuple[Path, dict]:
    """Provide local placeholder inputs for SDK-free configuration validation."""
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n")
    motion_library = tmp_path / "motion.usda"
    motion_library.write_text("#usda 1.0\n")
    characters = tmp_path / "characters"
    characters.mkdir()
    (characters / "M_actor.usda").write_text("#usda 1.0\n")
    return tmp_path / "simulation.yaml", {
        "schema_version": 1,
        "scene": {"asset_path": str(scene)},
        "pedestrians": {"count": 1, "assets": str(characters), "motion_library": motion_library.name},
    }


@pytest.mark.parametrize("population_source", ["pool", "actor", "motion_library", "gender_pool"])
def test_disabled_population_does_not_validate_unused_resources(scenario, population_source):
    path, data = scenario
    if population_source == "pool":
        data["pedestrians"]["assets"] = "missing"
    elif population_source == "actor":
        data["pedestrians"]["actors"] = {"walker_01": {"asset_path": "missing.usda"}}
    elif population_source == "motion_library":
        data["pedestrians"]["motion_library"] = "missing.usda"
    else:
        data["pedestrians"]["defaults"] = {"gender": "female"}
    path.write_text(yaml.safe_dump(data))
    args = create_parser().parse_args(["--simulation-config", str(path), "--no-pedestrians"])

    assert not _simulation_from_args(args).pedestrians.enabled
    with pytest.raises(ValueError):
        load_simulation_config(path)


@pytest.mark.parametrize("unused_source", ["base_pool", "replacement_scene"])
def test_cli_population_override_validates_only_selected_population(scenario, unused_source):
    path, data = scenario
    replacement = path.with_name("population.yaml")
    scene = data["scene"]["asset_path"]
    if unused_source == "replacement_scene":
        data["scene"]["asset_path"] = "unused_missing_scene.usda"
    replacement.write_text(yaml.safe_dump(data))
    data["scene"]["asset_path"] = scene
    if unused_source == "base_pool":
        data["pedestrians"]["assets"] = "missing"
    path.write_text(yaml.safe_dump(data))
    args = create_parser().parse_args(["--simulation-config", str(path), "--pedestrians-config", str(replacement)])

    config = _simulation_from_args(args)

    assert config.pedestrians.config_path == replacement
    assert path in config._source_paths


def test_standalone_cli_population_config_retains_profile_population(scenario):
    from pedstrain_simulation.simulation.cli.launcher import main
    from pedstrain_simulation.simulation.configuration.population import resolve_population

    path, data = scenario
    replacement = path.with_name("population.yaml")
    data["pedestrians"]["actors"] = {"visitor": {}}
    replacement.write_text(yaml.safe_dump(data))
    data["pedestrians"].update(assets="unused_missing_pool", actors={"unused": {}})
    path.write_text(yaml.safe_dump(data))
    profile = path.with_suffix(".toml")
    profile.write_text(f'scenario = "{path.name}"\n[pedestrians]\nconfig_path = "{replacement.name}"\n')
    launched = []

    main(["--pedestrians-config", str(profile)], runner=lambda config, **_: launched.append(config))

    assert launched == [load_simulation_config(profile)]
    assert tuple(plan.name for plan in resolve_population(launched[0]).config.pedestrian_plans) == ("visitor",)


def test_wrapper_validates_only_replaced_scene_and_robot_resources(scenario):
    path, data = scenario
    scene = data["scene"]["asset_path"]
    data["scene"]["asset_path"] = "missing_scene.usda"
    data["robots"] = [{"asset_path": "missing_robot.usda"}]
    path.write_text(yaml.safe_dump(data))
    wrapper = path.with_suffix(".toml")
    wrapper.write_text(f'scenario = "{path.name}"\nrobots = []\n[scene]\nasset_path = "{scene}"\n')

    config = load_simulation_config(wrapper)

    assert config.scene.asset_path == Path(scene)
    assert config.robots == ()
    assert config._source_paths == (path, wrapper)
    with pytest.raises(ValueError):
        load_simulation_config(path)


@pytest.mark.parametrize("entry", ["profile", "report"])
def test_configuration_check_reports_symlink_loop_without_traceback(scenario, capsys, entry):
    from pedstrain_simulation.simulation.cli.projects import main

    path, data = scenario
    link = path if entry == "profile" else path.with_name("status.json")
    if entry == "report":
        data["simulation"] = {"report_path": link.name}
        path.write_text(yaml.safe_dump(data))
    link.symlink_to(link.name)

    with pytest.raises(SystemExit) as error:
        main(["check", str(path)])

    assert error.value.code == 2
    assert str(link) in capsys.readouterr().err


def test_cli_scene_override_validates_final_scene(scenario):
    path, data = scenario
    scene = data["scene"]["asset_path"]
    data["scene"]["asset_path"] = "missing_scene.usda"
    path.write_text(yaml.safe_dump(data))
    args = create_parser().parse_args(["--simulation-config", str(path), "--scene", scene])

    assert _simulation_from_args(args).scene.asset_path == Path(scene)


def test_cli_scene_and_population_override_retains_outer_runtime(scenario):
    path, data = scenario
    path.write_text(yaml.safe_dump(data))
    profile = path.with_suffix(".toml")
    profile.write_text(f'scenario = "{path.name}"\n[simulation]\nduration = 45.0\n')
    replacement = path.with_name("replacement.yaml")
    scene, motion = (path.with_name(name) for name in ("different_scene.usda", "different_motion.usda"))
    for asset in (scene, motion):
        asset.write_text("#usda 1.0\n")
    data["scene"]["asset_path"] = scene.name
    data["pedestrians"].update(actors={"visitor": {}}, motion_library=motion.name)
    replacement.write_text(yaml.safe_dump(data))
    args = create_parser().parse_args(["--simulation-config", str(profile), "--config", str(replacement)])

    config = _simulation_from_args(args)

    assert config.settings.duration == 45.0
    assert config.scene.asset_path == scene
    assert [plan.name for plan in config.pedestrian_plans] == ["visitor"]
    assert config.motion_library == str(motion)
    assert {profile, path, replacement} == set(config._source_paths)


@pytest.mark.parametrize("source", ["pool", "actor", "motion_library"])
def test_public_compile_rejects_missing_used_resources(scenario, source):
    path, data = scenario
    plan = PedestrianPlan("walker_01", asset_path=path.parent / "missing.usda" if source == "actor" else None)
    kwargs = {"motion_library": path.parent / "missing.usda"} if source == "motion_library" else {}
    assets = path.parent / "missing" if source == "pool" else Path(data["pedestrians"]["assets"])

    with pytest.raises(ValueError, match="not found"):
        compile_scenario_config(
            scene=SceneConfig(asset_path=Path(data["scene"]["asset_path"])), plans=(plan,), assets=assets, **kwargs
        )


def test_public_compile_allows_unused_missing_shared_pool(scenario):
    path, data = scenario
    asset = Path(data["pedestrians"]["assets"]) / "M_actor.usda"

    config = compile_scenario_config(
        scene=SceneConfig(asset_path=Path(data["scene"]["asset_path"])),
        plans=(PedestrianPlan("walker_01", asset_path=asset),),
        assets=path.parent / "unused_missing_pool",
    )
    assert config.pedestrian_plans[0].asset_path == asset


@pytest.mark.parametrize("path_style", ["relative", "home"])
def test_public_compile_freezes_local_inputs_before_working_directory_changes(tmp_path, monkeypatch, path_style):
    import os

    from pedstrain_simulation.simulation import RobotConfig

    base = tmp_path / "inputs with spaces"
    (base / "characters" / "model").mkdir(parents=True)
    for name in ("scene.usda", "layer.usda", "motion.usda", "robot.usda", "characters/model/Actor.usda"):
        (base / name).write_text("#usda 1.0\n")
    other = tmp_path / "another directory"
    other.mkdir()
    monkeypatch.chdir(tmp_path)
    prefix = Path(base.name) if path_style == "relative" else Path("~") / os.path.relpath(base, Path.home())
    scene = SceneConfig(asset_path=prefix / "scene.usda", layers=(prefix / "layer.usda",))
    robot = RobotConfig(asset_path=prefix / "robot.usda")

    config = compile_scenario_config(
        scene=scene,
        plans=(PedestrianPlan("visitor", behavior="stand"),),
        assets=prefix / "characters",
        motion_library=prefix / "motion.usda",
        robots=(robot,),
    )
    monkeypatch.chdir(other)
    config.validate()

    assert config.scene.asset_path == base / "scene.usda"
    assert config.scene.layers == (base / "layer.usda",)
    assert config.robots[0].asset_path == base / "robot.usda"
    assert dict(config.pedestrian_assets) == {"visitor": base / "characters"}
    assert config.motion_library == str(base / "motion.usda")
    assert scene.asset_path == prefix / "scene.usda"
    assert scene.layers == (prefix / "layer.usda",)
    assert robot.asset_path == prefix / "robot.usda"


@pytest.mark.parametrize("selection", ["sdk_token", "explicit_path", "relative_string"])
def test_public_compile_distinguishes_motion_library_files_from_the_sdk_token(scenario, monkeypatch, selection):
    from pedstrain_simulation.simulation.configuration.scenarios.compilation import DEFAULT_MOTION_LIBRARY

    path, data = scenario
    monkeypatch.chdir(path.parent)
    library = Path(DEFAULT_MOTION_LIBRARY) if selection != "relative_string" else Path("Motion Library.usda")
    library.parent.mkdir(parents=True, exist_ok=True)
    library.write_text("#usda 1.0\n")
    selected = (
        DEFAULT_MOTION_LIBRARY
        if selection == "sdk_token"
        else library
        if selection == "explicit_path"
        else str(library)
    )
    config = compile_scenario_config(
        scene=SceneConfig(asset_path=Path(data["scene"]["asset_path"])),
        plans=(PedestrianPlan("visitor", behavior="stand"),),
        assets=Path(data["pedestrians"]["assets"]),
        motion_library=selected,
    )
    assert config.motion_library == (DEFAULT_MOTION_LIBRARY if selection == "sdk_token" else str(library.resolve()))


def test_ros2_settings_keep_authored_plans(scenario):
    path, data = scenario
    data["pedestrians"]["actors"] = {
        "visitor": {"position": [0, 0, 0], "waypoints": [{"position": [1, 0, 0], "wait": 2, "yaw": 90}]}
    }
    path.write_text(yaml.safe_dump(data))
    original = load_simulation_config(path)
    settings = path.with_name("ros2.yaml")
    settings.write_text(
        "time_horizon_s: 3.0\nobstacle_time_horizon_s: 0.5\n"
        "body_radius_m: 0.24\nmap_clearance_m: 0.04\npreferred_path_clearance_m: 0.06\npair_clearance_m: 0.16\n"
    )
    data["navigation"] = {"backend": "ros2", "settings_path": settings.name}
    path.write_text(yaml.safe_dump(data))

    actual = load_simulation_config(path)
    summary = inspect_simulation_config(path)
    assert actual.pedestrian_plans == original.pedestrian_plans
    assert summary["pedestrian_controller"] == "ros2_sfm_orca"
    assert summary["pedestrian_parameters"]["time_horizon_s"] == 3.0
    assert summary["pedestrian_parameters"]["obstacle_time_horizon_s"] == 0.5
    assert summary["ros2_clearance"] == pytest.approx(
        {
            "planning_radius_m": 0.28,
            "preferred_path_radius_m": 0.34,
            "fixed_spawn_body_center_distance_m": 0.48,
            "pedestrian_target_center_distance_m": 0.64,
        }
    )
    assert summary["random_spawn_minimum_center_distance_m"] == pytest.approx(0.64)
    assert str(settings) in summary["dependencies"]["configuration_files"]


def test_run_ros2_navigation_override_applies_to_preview_and_runner(scenario, monkeypatch, capsys):
    from pedstrain_simulation.simulation.cli.launcher import main

    path, data = scenario
    inherited = path.with_name("missing_navigation.yaml")
    data["navigation"] = {"backend": "ros2", "settings_path": inherited.name}
    path.write_text(yaml.safe_dump(data))
    original = path.read_bytes()
    tuning = path.parent / "tuning"
    tuning.mkdir()
    selected = tuning / "navigation.yaml"
    selected.write_text("map_clearance_m: 0.08\n")
    monkeypatch.chdir(tuning)
    args = ["run", str(path), "--navigation-settings", selected.name]

    def unexpected(*args, **kwargs):
        pytest.fail("Dry run must not launch a simulator")

    main([*args, "--dry-run"], runner=unexpected)
    summary = json.loads(capsys.readouterr().out)
    assert summary["navigation"]["settings_path"] == str(selected)
    assert summary["pedestrian_parameters"]["map_clearance_m"] == 0.08
    assert summary["dependencies"]["pedestrian_settings"] == {"path": str(selected), "selection": "explicit"}
    assert str(inherited) not in summary["dependencies"]["configuration_files"]

    def verify(config, *, extra_args, **kwargs):
        assert extra_args == []
        assert config.navigation.settings_path == selected
        assert [plan.to_dict() for plan in config.pedestrian_plans] == summary["pedestrian_plans"]

    main(args, runner=verify)
    assert path.read_bytes() == original


def test_run_navigation_override_requires_enabled_pedestrians(scenario, capsys):
    from pedstrain_simulation.simulation.cli.launcher import main

    path, data = scenario
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit) as error:
        main(["run", str(path), "--dry-run", "--navigation-settings", "unused.yaml", "--no-pedestrians"])
    assert error.value.code == 2
    assert "--navigation-settings requires enabled pedestrians" in capsys.readouterr().err


@pytest.mark.parametrize("option", ["--legacy-example", "--native-ira-settings=old.toml", "--agent-config=old.toml"])
def test_retired_cli_options_fail_before_launch(option, capsys):
    from pedstrain_simulation.simulation.cli.launcher import main

    with pytest.raises(SystemExit) as error:
        main([option], runner=lambda *_a, **_k: pytest.fail("retired option launched Kit"))
    assert error.value.code == 2
    assert "was removed" in capsys.readouterr().err


def test_ira_backend_requires_migration(scenario, capsys):
    from pedstrain_simulation.simulation.cli.launcher import main

    path, data = scenario
    data["navigation"] = {"backend": "ira"}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit) as error:
        main(["run", str(path), "--dry-run"], runner=lambda *_a, **_k: pytest.fail("IRA profile launched Kit"))
    assert error.value.code == 2
    assert "ros2" in capsys.readouterr().err.lower()


@pytest.mark.parametrize(
    ("name", "content", "diagnostic"),
    [
        ("actor_sdg.yaml", "isaacsim.replicator.agent: {}\n", "project YAML"),
        ("old_launcher.toml", 'ira_config = "old.yaml"\n', "scenario ="),
        ("old_training.toml", 'simulation_config = "old.toml"\n', "training profile was retired"),
    ],
)
def test_legacy_input_fails_with_migration_guidance(tmp_path, name, content, diagnostic):
    path = tmp_path / name
    path.write_text(content)
    with pytest.raises(ValueError, match=diagnostic):
        load_simulation_config(path)


@pytest.mark.parametrize("frame", ["world_m", "stage_world", "stage_world_without_metadata"])
def test_ros2_check_reports_fixed_spawn_overlap_in_meters(scenario, frame):
    path, data = scenario
    if frame == "stage_world":
        pytest.importorskip("pxr.Sdf")
        Path(data["scene"]["asset_path"]).write_text('#usda 1.0\n(\n    metersPerUnit = 0.01\n    upAxis = "Y"\n)\n')
    second = [0.0, 0.3, 0.0] if frame == "world_m" else [0.0, 80.0, -30.0]
    navigation = path.with_name("navigation.yaml")
    navigation.write_text("body_radius_m: 0.2\npair_clearance_m: 0.3\norigin_m: [1, 2, 3]\nyaw_rad: 1.2\n")
    data["navigation"] = {
        "backend": "ros2",
        "settings_path": navigation.name,
        "waypoint_frame": "world_m" if frame == "world_m" else "stage_world",
    }
    data["pedestrians"].update(
        count=3,
        actors={"first": {"position": [0, 0, 0]}, "second": {"position": second, "start_time": 10}, "random": {}},
    )
    path.write_text(yaml.safe_dump(data))

    summary = inspect_simulation_config(path)
    assert summary["random_spawn_minimum_center_distance_m"] == pytest.approx(0.7)
    placement = summary["fixed_spawn_check"]
    assert placement["minimum_center_distance_m"] == pytest.approx(0.4)
    assert placement["fixed_count"] == 2
    if frame == "stage_world_without_metadata":
        assert placement["status"] == "not_checked"
        assert placement["overlaps"] == []
    else:
        assert placement["status"] == "overlap"
        assert placement["overlaps"] == [{"actors": ["first", "second"], "center_distance_m": pytest.approx(0.3)}]
    assert next(plan for plan in summary["pedestrian_plans"] if plan["name"] == "second")["position"] == second


def test_ros2_project_creation_only_writes_active_navigation_files(scenario):
    from pedstrain_simulation.simulation.configuration.navigation import load_ros2_navigation_settings
    from pedstrain_simulation.simulation.projects import create_simulation_project

    path, data = scenario
    directory = path.parent / "ros2_project"
    profile = create_simulation_project(
        directory,
        scene_asset=Path(data["scene"]["asset_path"]),
        character_assets=Path(data["pedestrians"]["assets"]),
        navigation_backend="ros2",
    )
    assert {p.name for p in directory.iterdir()} == {"simulation.yaml", "navigation_settings.yaml"}
    actual = load_simulation_config(profile)
    assert actual.navigation.backend == "ros2"
    assert actual.navigation.settings_path == directory / "navigation_settings.yaml"
    assert actual.settings.report_path == directory / "output/status.json"
    copied = yaml.safe_load(actual.navigation.settings_path.read_text())
    assert copied["auto_return_heading"] == "follow_route"
    assert load_ros2_navigation_settings(actual.navigation.settings_path).auto_return_heading == "follow_route"


def test_ros2_auto_return_heading_preserves_omitted_external_settings_and_shared_project_policy(tmp_path):
    from pedstrain_simulation.simulation.configuration.navigation import (
        Ros2NavigationSettings,
        load_ros2_navigation_settings,
    )

    assert Ros2NavigationSettings().auto_return_heading == "preserve"
    settings = tmp_path / "external_navigation.yaml"
    settings.write_text("{}\n")
    assert load_ros2_navigation_settings(settings).auto_return_heading == "preserve"
    assert load_ros2_navigation_settings().auto_return_heading == "follow_route"
    for policy in ("preserve", "follow_route"):
        settings.write_text(yaml.safe_dump({"auto_return_heading": policy}))
        actual = load_ros2_navigation_settings(settings)
        assert actual.auto_return_heading == policy
        assert actual.to_dict()["auto_return_heading"] == policy


@pytest.mark.parametrize("policy", [None, True, 1, "route", "FOLLOW_ROUTE", ["follow_route"], {}])
def test_ros2_auto_return_heading_rejects_invalid_direct_and_yaml_values(tmp_path, policy):
    from pedstrain_simulation.simulation.configuration.navigation import (
        Ros2NavigationSettings,
        load_ros2_navigation_settings,
    )

    with pytest.raises(ValueError, match="auto_return_heading"):
        Ros2NavigationSettings(auto_return_heading=policy)
    settings = tmp_path / "external_navigation.yaml"
    settings.write_text(yaml.safe_dump({"auto_return_heading": policy}))
    with pytest.raises(ValueError, match="auto_return_heading"):
        load_ros2_navigation_settings(settings)


@pytest.mark.parametrize("fps", [24, 60])
def test_ros2_recording_preflight_rejects_incompatible_capture_cadence(scenario, monkeypatch, fps):
    from pedstrain_simulation.simulation.recording import output

    path, data = scenario
    navigation = path.with_name("navigation.yaml")
    navigation.write_text("control_hz: 30\n")
    data["navigation"] = {"backend": "ros2", "settings_path": navigation.name}
    data["simulation"] = {"duration": 2.0}
    data["recording"] = {"enabled": True, "output_path": "output/capture.mp4", "fps": fps}
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(output.shutil, "which", lambda name: "/existing/ffmpeg")
    with pytest.raises(ValueError, match="recording.fps.*control_hz"):
        inspect_simulation_config(path)
    assert not (path.parent / "output").exists()


def test_ros2_recording_preview_exposes_the_effective_capture_settings(scenario, monkeypatch):
    from pedstrain_simulation.simulation.recording import output

    path, data = scenario
    navigation = path.with_name("navigation.yaml")
    navigation.write_text("control_hz: 60\n")
    data["navigation"] = {"backend": "ros2", "settings_path": navigation.name}
    data["simulation"] = {"duration": 2.0}
    data["recording"] = {
        "enabled": True,
        "output_path": "output/capture.mp4",
        "fps": 12,
        "warmup_s": 0.5,
        "camera_eye_m": [1.0, 2.0, 8.0],
        "camera_target_m": [1.0, 2.0, 0.0],
        "max_stationary_s": 3.0,
    }
    path.write_text(yaml.safe_dump(data))
    original = path.read_bytes()
    monkeypatch.setattr(output.shutil, "which", lambda name: "/existing/ffmpeg")
    summary = inspect_simulation_config(path)
    assert summary["recording"] == {
        **data["recording"],
        "output_path": str(path.parent / "output/capture.mp4"),
        "duration_s": 2.0,
        "camera_view": "oblique",
    }
    assert path.read_bytes() == original and not (path.parent / "output").exists()


@pytest.mark.parametrize(
    ("camera", "diagnostic"),
    [
        ({"camera_view": "tilted"}, "camera_view must be oblique or overhead"),
        (
            {"camera_view": "overhead", "camera_eye_m": [1, 2, 3], "camera_target_m": [1, 2, 0]},
            "overhead uses automatic framing",
        ),
    ],
)
def test_recording_camera_view_rejects_conflicting_configuration(scenario, camera, diagnostic):
    path, data = scenario
    data["simulation"] = {"duration": 2.0}
    data["recording"] = {"enabled": True, "output_path": "output/capture.mp4", **camera}
    path.write_text(yaml.safe_dump(data))

    with pytest.raises(ValueError, match=diagnostic):
        load_simulation_config(path)


def test_recording_requires_ffprobe_during_preflight(scenario, monkeypatch):
    from pedstrain_simulation.simulation.recording import output

    path, data = scenario
    data["simulation"] = {"duration": 2.0}
    data["recording"] = {"enabled": True, "output_path": "output/capture.mp4"}
    path.write_text(yaml.safe_dump(data))
    monkeypatch.setattr(output.shutil, "which", lambda name: "/existing/ffmpeg" if name == "ffmpeg" else None)
    with pytest.raises(ValueError, match="ffprobe"):
        inspect_simulation_config(path)


def test_default_dispatch_starts_ros2(scenario, monkeypatch):
    from pedstrain_simulation.simulation.ros2_navigation import runtime as ros2
    from pedstrain_simulation.simulation.runtime import runner

    path, data = scenario
    path.write_text(yaml.safe_dump(data))
    config = load_simulation_config(path)
    assert config.navigation.backend == "ros2"
    calls = []
    monkeypatch.setattr(ros2, "run_ros2_simulation", lambda actual, **kwargs: calls.append(actual))
    runner.run_simulation(config)
    assert calls == [config]


@pytest.mark.parametrize("source", ["local", "remote", "omitted"])
def test_preview_reports_declared_asset_root_without_sdk_resolution(scenario, source):
    path, data = scenario
    asset_root = {
        "local": str(path.parent / "local SDK assets"),
        "remote": "https://assets.example.test/Isaac/6.1",
        "omitted": None,
    }[source]
    navigation = path.with_name("navigation.yaml")
    navigation.write_text(yaml.safe_dump({"asset_root": asset_root}))
    data["navigation"] = {"settings_path": navigation.name}
    path.write_text(yaml.safe_dump(data))

    summary = inspect_simulation_config(path)
    assert summary["dependencies"]["runtime_settings"]["asset_root"] == asset_root


@pytest.mark.parametrize(
    ("source", "diagnostic"),
    [
        ("unset", "requires a local asset_root"),
        ("remote", "requires a local asset_root"),
        ("missing", "Motion library not found"),
        ("directory", "Motion library not found"),
        ("local", None),
        ("file_uri", None),
        ("explicit", None),
    ],
)
def test_ros2_motion_library_preflight_agrees_with_runtime_selection(scenario, source, diagnostic):
    from pedstrain_simulation.simulation.configuration.navigation import load_ros2_navigation_settings
    from pedstrain_simulation.simulation.configuration.resources import DEFAULT_MOTION_LIBRARY
    from pedstrain_simulation.simulation.ros2_navigation.runtime import _assets

    path, data = scenario
    navigation = path.with_name("navigation.yaml")
    asset_root = path.parent / "local SDK assets"
    motion = asset_root / DEFAULT_MOTION_LIBRARY
    if source == "explicit":
        motion = path.parent / data["pedestrians"]["motion_library"]
    else:
        data["pedestrians"].pop("motion_library")
    if source in ("local", "file_uri"):
        motion.parent.mkdir(parents=True)
        motion.write_text("#usda 1.0\n")
    elif source == "directory":
        motion.mkdir(parents=True)
    if source == "remote":
        root_setting = "https://assets.example.test/Isaac/6.1"
    elif source == "file_uri":
        root_setting = asset_root.as_uri()
    elif source in ("unset", "explicit"):
        root_setting = None
    else:
        root_setting = str(asset_root)
    navigation.write_text(yaml.safe_dump({"asset_root": root_setting}))
    data["navigation"] = {"backend": "ros2", "settings_path": navigation.name}
    path.write_text(yaml.safe_dump(data))
    config = load_simulation_config(path)
    settings = load_ros2_navigation_settings(navigation)

    if diagnostic is not None:
        with pytest.raises(ValueError, match=diagnostic):
            inspect_simulation_config(path)
        with pytest.raises(ValueError, match=diagnostic):
            _assets(config, settings, config.pedestrian_plans)
    else:
        summary = inspect_simulation_config(path)
        _, selected = _assets(config, settings, config.pedestrian_plans)
        assert selected == motion
        assert str(selected) in summary["dependencies"]["assets"]


def test_ros2_report_protects_navigation_motion_library_over_file_relative_namesake(scenario):
    from pedstrain_simulation.simulation.configuration.resources import DEFAULT_MOTION_LIBRARY
    from pedstrain_simulation.simulation.recording.output import _validate_report_path

    path, data = scenario
    asset_root = path.parent / "sdk_assets"
    motion = asset_root / DEFAULT_MOTION_LIBRARY
    unused_motion = path.parent / DEFAULT_MOTION_LIBRARY
    for file in (motion, unused_motion):
        file.parent.mkdir(parents=True)
        file.write_text("#usda 1.0\n")
    navigation = path.with_name("navigation.yaml")
    navigation.write_text(yaml.safe_dump({"asset_root": str(asset_root)}))
    data["navigation"] = {"backend": "ros2", "settings_path": navigation.name}
    data["pedestrians"].pop("motion_library")
    path.write_text(yaml.safe_dump(data))
    config = load_simulation_config(path)

    with pytest.raises(ValueError, match="must not overwrite an input"):
        _validate_report_path(config, motion)
    _validate_report_path(config, unused_motion)


def test_preview_omits_disabled_population_controls(scenario):
    from pedstrain_simulation.simulation.projects.inspection import _inspect_simulation_config

    path, data = scenario
    path.write_text(yaml.safe_dump(data))
    config = load_simulation_config(path)
    config = replace(config, pedestrians=replace(config.pedestrians, enabled=False))
    summary = _inspect_simulation_config(config, path)
    assert summary["pedestrian_effective_avoidance"] is None
    assert summary["pedestrian_parameters"] is None
    assert summary["pedestrian_controller"] == "disabled"


def test_compilation_discovers_a_shared_gender_pool_once(scenario, monkeypatch):
    from pedstrain_simulation.simulation.configuration.scenarios import compilation

    _, data = scenario
    assets = Path(data["pedestrians"]["assets"])
    plans = tuple(PedestrianPlan(f"walker_{index}", gender="male") for index in range(4))
    calls = []
    original = compilation.find_character_assets

    def discover(source, gender, *, asset_root):
        calls.append((source, gender))
        return original(source, gender, asset_root=asset_root)

    monkeypatch.setattr(compilation, "find_character_assets", discover)
    compile_scenario_config(scene=SceneConfig(asset_path=Path(data["scene"]["asset_path"])), plans=plans, assets=assets)
    assert calls == [(assets.resolve(), "male")]


def test_session_model_does_not_select_a_historical_scene():
    from pedstrain_simulation.simulation import SimulationConfig

    config = SimulationConfig()
    assert config.scene.asset_path is None and not config.pedestrian_plans
    with pytest.raises(ValueError, match="scene.asset_path"):
        config.validate()


def test_launcher_requires_a_scene_selection_instead_of_loading_an_example():
    args = create_parser().parse_args([])
    with pytest.raises(ValueError, match="explicit|project|PROJECT"):
        _simulation_from_args(args)


def test_scenario_does_not_infer_a_repository_character_pool(scenario):
    path, data = scenario
    del data["pedestrians"]["assets"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="pedestrians.assets"):
        load_simulation_config(path)


def test_actor_specific_assets_need_no_shared_pool(scenario):
    path, data = scenario
    asset = Path(data["pedestrians"].pop("assets")) / "M_actor.usda"
    data["pedestrians"]["actors"] = {"visitor_alpha": {"asset_path": str(asset), "behavior": "stand"}}
    path.write_text(yaml.safe_dump(data))
    config = load_simulation_config(path)
    assert config.pedestrian_plans[0].name == "visitor_alpha"
    assert config.pedestrian_plans[0].asset_path == asset


def test_init_requires_explicit_assets_before_creating_a_directory(scenario):
    from pedstrain_simulation.simulation import create_simulation_project

    path, data = scenario
    directory = path.parent / "new_project"
    with pytest.raises(ValueError, match="character_assets"):
        create_simulation_project(directory, scene_asset=Path(data["scene"]["asset_path"]))
    assert not directory.exists()


@pytest.fixture
def report_inputs(scenario):
    path, data = scenario
    resources = {"scene": Path(data["scene"]["asset_path"])}
    resources["actor"] = Path(data["pedestrians"]["assets"]) / "M_actor.usda"
    for name in ("motion_library", "robot"):
        resources[name] = path.with_name(f"{name}.usda")
        resources[name].write_text("#usda 1.0\n")
    resources["navigation"] = path.with_name("navigation.yaml")
    resources["navigation"].write_text("{}\n")
    data["pedestrians"]["motion_library"] = str(resources["motion_library"])
    data["navigation"] = {"settings_path": str(resources["navigation"])}
    data["robots"] = [{"asset_path": str(resources["robot"])}]
    path.write_text(yaml.safe_dump(data))
    return load_simulation_config(path), path, data, resources


@pytest.mark.parametrize("resource", ["scene", "actor", "motion_library", "robot", "navigation"])
def test_report_cannot_overwrite_effective_assets(report_inputs, resource):
    from pedstrain_simulation.simulation.recording.output import _validate_report_path

    config, _, _, resources = report_inputs
    before = resources[resource].read_bytes()
    with pytest.raises(ValueError, match="must not overwrite an input"):
        _validate_report_path(config, resources[resource])
    assert resources[resource].read_bytes() == before


def test_wrapper_report_protects_the_consumed_project_yaml(report_inputs):
    from pedstrain_simulation.simulation.recording.output import _validate_report_path

    _, path, _, _ = report_inputs
    wrapper = path.with_suffix(".toml")
    wrapper.write_text('scenario = "simulation.yaml"\n')
    with pytest.raises(ValueError, match="must not overwrite an input"):
        _validate_report_path(load_simulation_config(wrapper), path)


def test_report_allows_unrelated_outputs_and_replaced_assets(report_inputs):
    from pedstrain_simulation.simulation.configuration.models import PedestrianConfig
    from pedstrain_simulation.simulation.recording.output import _validate_report_path

    config, path, data, resources = report_inputs
    replacement = path.with_name("replacement.yaml")
    other_scene = path.with_name("other_scene.usda")
    other_scene.write_text("#usda 1.0\n")
    other_actor = path.with_name("other_actor.usda")
    other_actor.write_text("#usda 1.0\n")
    other_motion = path.with_name("other_motion.usda")
    other_motion.write_text("#usda 1.0\n")
    data["pedestrians"].pop("assets")
    data["pedestrians"].update(actors={"visitor": {"asset_path": str(other_actor)}}, motion_library=str(other_motion))
    replacement.write_text(yaml.safe_dump(data))
    selected = replace(
        config,
        scene=SceneConfig(asset_path=other_scene),
        pedestrians=PedestrianConfig(config_path=replacement),
    )
    selected.validate()
    for output in (
        path.with_name("status.json"),
        resources["scene"],
        resources["actor"],
        resources["motion_library"],
    ):
        _validate_report_path(selected, output)
    for active in (other_scene, other_actor, other_motion):
        with pytest.raises(ValueError, match="must not overwrite an input"):
            _validate_report_path(selected, active)


@pytest.mark.parametrize("source_records", [False, True])
def test_headless_profile_reuses_the_original_scenario_without_changing_motion(scenario, source_records):
    path, data = scenario
    data["simulation"] = {"duration": 37.0, "seed": 17, "physics_hz": 120, "report_path": "output/original.json"}
    if source_records:
        data["recording"] = {"enabled": True, "output_path": "output/capture.mp4"}
    data["pedestrians"]["actors"] = {
        "visitor": {"position": [0.0, 0.0, 0.0], "waypoints": [{"position": [1.0, 0.0, 0.0], "wait": 2.0}]}
    }
    path.write_text(yaml.safe_dump(data))
    original_bytes = path.read_bytes()
    wrapper = path.with_name("headless.toml")
    wrapper.write_text((Path(__file__).parents[1] / "configs/third_3/headless.toml").read_text())

    source = load_simulation_config(path)
    actual = load_simulation_config(wrapper)
    assert actual.settings == replace(
        source.settings,
        headless=True,
        render_viewport=False,
        report_path=path.parent / "output/headless_status.json",
    )
    assert actual.scene == source.scene and actual.pedestrians == source.pedestrians
    assert actual.pedestrian_plans == source.pedestrian_plans
    assert actual.pedestrian_assets == source.pedestrian_assets
    assert actual.motion_library == source.motion_library
    assert actual.robots == source.robots
    assert actual.recording == replace(source.recording, enabled=False)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("unsupported_file", [None, "character.usdc"])
def test_generation_rejects_asset_pools_without_supported_entries(scenario, unsupported_file):
    from pedstrain_simulation.simulation import create_simulation_project

    path, data = scenario
    assets = Path(data["pedestrians"]["assets"])
    (assets / "M_actor.usda").unlink()
    if unsupported_file is not None:
        (assets / unsupported_file).write_bytes(b"unsupported candidate")
    directory = path.parent / "uncreated_parent" / "project"
    with pytest.raises(ValueError, match="character_assets"):
        create_simulation_project(directory, scene_asset=Path(data["scene"]["asset_path"]), character_assets=assets)
    assert not directory.parent.exists()


def test_generation_standing_preserves_explicit_navigation_settings(scenario):
    from pedstrain_simulation.simulation import create_simulation_project

    path, data = scenario
    settings = path.with_name("navigation.yaml")
    settings.write_text("map_clearance_m: 0.08\n")
    original = settings.read_bytes()
    directory = path.parent / "standing_project"
    profile = create_simulation_project(
        directory,
        scene_asset=Path(data["scene"]["asset_path"]),
        character_assets=Path(data["pedestrians"]["assets"]),
        navigation_settings=settings,
        behavior="standing",
        pedestrian_count=2,
    )
    config = load_simulation_config(profile)
    assert [plan.behavior for plan in config.pedestrian_plans] == ["stand", "stand"]
    assert config.navigation.settings_path == settings
    assert settings.read_bytes() == original
    assert {file.name for file in directory.iterdir()} == {"simulation.yaml"}


def test_generation_validates_navigation_before_creating_directories(scenario):
    from pedstrain_simulation.simulation import create_simulation_project

    path, data = scenario
    navigation = path.with_name("broken_navigation.yaml")
    navigation.write_text("[broken")
    directory = path.parent / "uncreated_parent" / "project"
    with pytest.raises(ValueError, match="broken_navigation.yaml"):
        create_simulation_project(
            directory,
            scene_asset=Path(data["scene"]["asset_path"]),
            character_assets=Path(data["pedestrians"]["assets"]),
            navigation_settings=navigation,
        )
    assert not directory.parent.exists()


@pytest.mark.parametrize("unsupported", [None, "not_a_model.txt", "character.usdc"])
def test_selected_unrestricted_population_requires_discoverable_assets(scenario, unsupported):
    path, data = scenario
    assets = Path(data["pedestrians"]["assets"])
    (assets / "M_actor.usda").unlink()
    if unsupported is not None:
        (assets / unsupported).write_text("not a supported character entry")
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="pedestrians.assets.*USD|No character assets"):
        load_simulation_config(path)


@pytest.mark.parametrize("layout", ["flat", "flat_and_nested", "nested"])
def test_asset_pool_prefers_direct_entries_or_uses_nested_entries(tmp_path, layout):
    from pedstrain_simulation.simulation.configuration.scenarios.assets import find_character_assets

    source = tmp_path / "characters"
    direct = source / "Visitor.usda"
    nested = source / "model" / "Nested.usda"
    source.mkdir()
    if layout != "nested":
        direct.touch()
    if layout != "flat":
        nested.parent.mkdir()
        nested.touch()
    assert find_character_assets(source) == ((nested,) if layout == "nested" else (direct,))


def test_asset_catalog_preserves_order_and_protects_resolved_inputs(scenario):
    from pedstrain_simulation.simulation.configuration.scenarios.assets import find_character_assets
    from pedstrain_simulation.simulation.recording.output import _validate_report_path

    path, data = scenario
    root = path.parent / "SDK assets"
    root.mkdir()
    entries = ["M_second.usda", "F_visitor.usda", "M_first.usda"]
    for entry in entries:
        (root / entry).write_text("#usda 1.0\n")
    catalog = path.with_name("catalog.yaml")
    catalog.write_text(yaml.safe_dump({"assets": entries}))
    navigation = path.with_name("navigation.yaml")
    navigation.write_text(yaml.safe_dump({"asset_root": root.as_uri()}))
    data["navigation"] = {"settings_path": navigation.name}
    data["pedestrians"].update(assets=catalog.name, defaults={"gender": "male"})
    path.write_text(yaml.safe_dump(data))

    assert find_character_assets(catalog, "male", asset_root=root.as_uri()) == (
        root / entries[0],
        root / entries[2],
    )
    config = load_simulation_config(path)
    dependencies = inspect_simulation_config(path)["dependencies"]
    assert set(map(str, (catalog, *(root / entry for entry in entries)))) <= set(dependencies["assets"])
    with pytest.raises(ValueError, match="must not overwrite"):
        _validate_report_path(config, root / entries[2])


@pytest.mark.parametrize(
    ("entries", "reason"),
    [
        (["../outside.usda"], "relative"),
        (["M_actor.usda", "M_actor.usda"], "duplicate"),
        (["missing.usda"], "not found"),
    ],
)
def test_asset_catalog_rejects_invalid_entries(tmp_path, entries, reason):
    from pedstrain_simulation.simulation.configuration.scenarios.assets import find_character_assets

    (tmp_path / "M_actor.usda").touch()
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(yaml.safe_dump({"assets": entries}))
    with pytest.raises(ValueError, match=reason):
        find_character_assets(catalog, asset_root=str(tmp_path))


def test_inspection_retains_the_consumed_wrapper_yaml_sources(scenario):
    path, data = scenario
    path.write_text(yaml.safe_dump(data))
    wrapper = path.with_suffix(".toml")
    wrapper.write_text('scenario = "simulation.yaml"\n')
    summary = inspect_simulation_config(wrapper)
    assert {str(path), str(wrapper)} <= set(summary["dependencies"]["configuration_files"])


def test_config_layout_has_only_maintained_inputs():
    root = Path(__file__).resolve().parents[1]
    assert {path.name for path in root.glob("*.py")} == {"__init__.py", "__main__.py", "run.py"}


def test_public_configuration_and_execution_imports_need_no_native_sdk(tmp_path):
    import os
    import subprocess
    import sys

    script = tmp_path / "cold_import.py"
    script.write_text(
        "import importlib, importlib.abc, sys\n"
        "class RejectNative(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in {'omni', 'isaacsim', 'pxr'}:\n"
        "            raise AssertionError('Native SDK imported before startup: '+fullname)\n"
        "sys.meta_path.insert(0, RejectNative())\n"
        "import pedstrain_simulation.simulation as simulation\n"
        "for name in simulation.__all__: getattr(simulation, name)\n"
        "for suffix in ('configuration.resources', 'ros2_navigation.runtime', 'cli.launcher'):\n"
        "    importlib.import_module('pedstrain_simulation.simulation.'+suffix)\n"
    )
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[3]))
    completed = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=environment, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ros_namespace", "//pedestrians"),
        ("dynamic_roots", ["/World/Door.angle"]),
        ("dynamic_roots", ["/World/Door/"]),
        ("request_timeout_s", 10**400),
        ("origin_m", [0, 10**400, 0]),
    ],
    ids=["namespace", "property-path", "trailing-prim-slash", "scalar-overflow", "coordinate-overflow"],
)
def test_ros2_configuration_reports_invalid_runtime_values_with_source(tmp_path, field, value):
    from pedstrain_simulation.simulation.configuration.navigation import load_ros2_navigation_settings

    path = tmp_path / "navigation.yaml"
    path.write_text(yaml.safe_dump({field: value}))
    with pytest.raises(ValueError, match=rf"navigation\.yaml:.*{field}"):
        load_ros2_navigation_settings(path)


def test_ros2_configuration_accepts_root_prim_and_namespace_with_trailing_separator():
    from pedstrain_simulation.simulation.configuration.navigation import Ros2NavigationSettings

    config = Ros2NavigationSettings(ros_namespace="/pedestrians/", dynamic_roots=("/", "/World/Door_1"))
    assert config.ros_namespace == "/pedestrians/"
    assert config.dynamic_roots == ("/", "/World/Door_1")


@pytest.mark.parametrize(
    ("recording", "diagnostic"),
    [
        ({"warmup_s": 10**400}, "warmup_s"),
        ({"max_stationary_s": 10**400}, "max_stationary_s"),
        ({"camera_eye_m": [10**400, 0, 1], "camera_target_m": [0, 0, 0]}, "camera_eye_m"),
        ({"enabled": True, "output_path": "video.mp4", "fps": 60}, "frame count"),
    ],
    ids=["warmup", "stationary-limit", "camera-coordinate", "frame-count"],
)
def test_recording_configuration_rejects_overflow_with_source(scenario, recording, diagnostic):
    path, data = scenario
    data["simulation"] = {"duration": 1.0e308}
    data["recording"] = recording
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=rf"simulation\.yaml:.*{diagnostic}"):
        load_simulation_config(path)


@pytest.mark.parametrize(
    ("settings", "diagnostic"),
    [
        ({"body_radius_m": 5e307, "body_height_m": 1e308, "pair_clearance_m": 1e308}, "pair_clearance_m"),
        ({"map_clearance_m": 1e308, "preferred_path_clearance_m": 1e308}, "preferred_path_clearance_m"),
    ],
)
def test_ros2_configuration_rejects_overflow_in_combined_clearance(tmp_path, settings, diagnostic):
    from pedstrain_simulation.simulation.configuration.navigation import load_ros2_navigation_settings

    path = tmp_path / "navigation.yaml"
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match=rf"navigation\.yaml:.*{diagnostic}"):
        load_ros2_navigation_settings(path)


def test_robot_path_can_use_an_authored_character_namespace(scenario):
    path, data = scenario
    prim_path = "/World/Characters/Robot"
    data["robots"] = [{"asset_path": data["scene"]["asset_path"], "prim_path": prim_path}]
    path.write_text(yaml.safe_dump(data))
    assert load_simulation_config(path).robots[0].prim_path == prim_path
