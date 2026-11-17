# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only map/marker monitoring, session binding and no-SDK launcher contracts."""

import copy
import json
import math
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from pedstrain_simulation.simulation.cli import launcher
from pedstrain_simulation.simulation.projects import workflow
from pedstrain_simulation.simulation.ros2_navigation import monitor, visualization


@pytest.fixture
def messages(monkeypatch):
    modules = {
        name: ModuleType(name) for name in ("geometry_msgs.msg", "rclpy.qos", "visualization_msgs.msg", "tf2_msgs.msg")
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    class Marker:
        ARROW, CUBE, SPHERE, CYLINDER, LINE_STRIP, TEXT_VIEW_FACING = 0, 1, 2, 3, 4, 9
        ADD, DELETEALL = 0, 3

        def __init__(self):
            self.header = SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0))
            self.pose = SimpleNamespace(position=SimpleNamespace(), orientation=SimpleNamespace())
            self.color, self.scale = SimpleNamespace(), SimpleNamespace()

    modules["visualization_msgs.msg"].Marker = Marker
    modules["visualization_msgs.msg"].MarkerArray = SimpleNamespace
    modules["geometry_msgs.msg"].Point = SimpleNamespace
    modules["geometry_msgs.msg"].TransformStamped = lambda: SimpleNamespace(
        header=SimpleNamespace(), transform=SimpleNamespace(translation=SimpleNamespace(), rotation=SimpleNamespace())
    )
    modules["tf2_msgs.msg"].TFMessage = SimpleNamespace
    modules["rclpy.qos"].DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL="transient")
    modules["rclpy.qos"].ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable")
    modules["rclpy.qos"].QoSProfile = SimpleNamespace
    published, references = [], []

    def publisher(message_type, topic, qos):
        assert topic in ("markers", "/tf_static")
        assert qos.depth == 1 and qos.durability == "transient" and qos.reliability == "reliable"
        return SimpleNamespace(publish=(published if topic == "markers" else references).append)

    return SimpleNamespace(create_publisher=publisher, reference_messages=references), published, Marker


def _bootstrap(tmp_path):
    grid = np.zeros((10, 20), dtype=np.int8)
    np.savez(tmp_path / "map.npz", occupancy=grid, resolution_m=0.5, origin_xy_m=[-10.0, 3.0])
    bootstrap = {
        "map_npz": "map.npz",
        "plans": [{"name": "visitor", "arrival_tolerance": 0.17, "waypoints": [{"position": [4.0, 5.0, 0.0]}]}],
        "agents": [{"name": "visitor", "position": [1.0, 2.0, 0.25], "yaw": math.pi / 2, "radius_m": 0.3}],
        "settings": {
            "floor_height_m": 0.0,
            "door_specs": [{"name": "/Room/Portal", "hinge_position_m": [2.0, 3.0, 1.0]}],
        },
    }
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(bootstrap))
    return bootstrap, path


def test_markers_use_actual_map_pose_and_radius_while_commands_and_goals_remain_labelled(messages, tmp_path):
    node, published, marker_type = messages
    bootstrap, _ = _bootstrap(tmp_path)
    bootstrap["settings"]["door_specs"].append({"name": "/Room/Disabled", "enabled": False})
    bootstrap["settings"]["door_specs"][0].update(
        doorway_segment_xy_m=[[2.0, 3.0], [3.0, 3.0]],
        swept_footprint_xy_m=[[2.0, 3.0], [3.0, 3.0], [3.0, 4.0], [2.0, 4.0]],
    )
    observer = visualization.NavigationMonitor(node, bootstrap)
    request = {
        "time_s": 0.2,
        "agents": bootstrap["agents"],
        "obstacles": [{"footprint_xy_m": [[0, 0], [1, 0], [1, 1]], "z_min_m": 0.2}],
        "doors": [
            {
                "name": "/Room/Portal",
                "valid": True,
                "actual_angle_deg": 32.0,
                "fully_closed": False,
                "fully_open": False,
            }
        ],
    }
    command = {
        "agents": [{"name": "visitor", "position": [9.0, 9.0, 0.25], "velocity": [0.9, 0, 0], "phase": "walking"}],
        "door_commands": [{"name": "/Room/Portal", "request_open": False}],
        "guidance": {"visitor": {"target": [8.0, 6.0, 0.25], "phase": "walking"}},
    }
    original = copy.deepcopy((bootstrap, request, command))
    assert observer.publish(request, command)
    markers = published[-1].markers
    assert markers[0].action == marker_type.DELETEALL
    assert all(item.header.frame_id == "map" and item.header.stamp.nanosec == 200000000 for item in markers)
    body = next(item for item in markers if item.ns == "observed/body")
    assert (body.pose.position.x, body.pose.position.y, body.pose.position.z) == (1.0, 2.0, 0.25)
    assert (body.pose.orientation.z, body.pose.orientation.w) == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))
    assert (body.scale.x, body.scale.y) == (0.6, 0.6)
    heading = next(item for item in markers if item.ns == "observed/heading")
    assert (heading.points[-1].x, heading.points[-1].y) == pytest.approx((1.0, 2.6))
    goal = next(item for item in markers if item.ns == "requested/yaml_goal_tolerance")
    assert (goal.pose.position.x, goal.pose.position.y, goal.scale.x) == (4.0, 5.0, 0.34)
    velocity = next(item for item in markers if item.ns == "issued/velocity_1s")
    assert (velocity.points[-1].x, velocity.points[-1].y) == pytest.approx((1.9, 2.0))
    assert "control phase: walking" in next(item.text for item in markers if item.ns == "observed/labels")
    footprint = next(item for item in markers if item.ns == "observed/dynamic_footprints")
    assert (footprint.points[0].x, footprint.points[0].y) == (footprint.points[-1].x, footprint.points[-1].y)
    assert "32.0deg" in next(item.text for item in markers if item.ns == "observed/door_state")
    guidance = next(item for item in markers if item.ns == "requested/current_guidance")
    assert (guidance.points[-1].x, guidance.points[-1].y, guidance.points[-1].z) == (8.0, 6.0, 0.25)
    guide_point = next(item for item in markers if item.ns == "requested/guide_point")
    assert (guide_point.pose.position.x, guide_point.pose.position.y, guide_point.pose.position.z) == (8.0, 6.0, 0.25)
    portal = next(item for item in markers if item.ns == "reference/door_portal")
    assert [(p.x, p.y) for p in portal.points] == [(2.0, 3.0), (3.0, 3.0)]
    sweep = next(item for item in markers if item.ns == "requested/door_sweep")
    assert (sweep.points[0].x, sweep.points[0].y) == (sweep.points[-1].x, sweep.points[-1].y)
    assert 0 < sweep.color.a < footprint.color.a
    assert (bootstrap, request, command) == original


def test_marker_rate_uses_simulation_time_and_reset_replaces_the_complete_snapshot(messages, tmp_path):
    node, published, marker_type = messages
    bootstrap, _ = _bootstrap(tmp_path)
    observer = visualization.NavigationMonitor(node, bootstrap)
    for tick in range(1, 61):
        observer.publish({"time_s": tick / 60, "agents": bootstrap["agents"]}, {"agents": []})
    assert len(published) == 11
    assert not observer.publish({"time_s": 1.0, "agents": []}, {"agents": []})
    assert observer.publish({"time_s": 0.0, "agents": []}, {"agents": []})
    assert published[-1].markers[0].action == marker_type.DELETEALL
    assert all(item.ns != "observed/body" for item in published[-1].markers)


@pytest.mark.parametrize("actual_namespace", ["/people/session_unique", None])
def test_reference_tf_uses_canonical_world_inverse_and_the_actual_unique_session(messages, tmp_path, actual_namespace):
    node, _, _ = messages
    bootstrap, _ = _bootstrap(tmp_path)
    bootstrap["settings"].update(
        origin_m=[1.0, 2.0, 3.0], yaw_rad=math.pi / 2, ros_namespace="/fallback/session_snapshot"
    )
    if actual_namespace is not None:
        node.get_namespace = lambda: actual_namespace
    observer = visualization.NavigationMonitor(node, bootstrap)
    reference = node.reference_messages[0].transforms[0]
    expected_child = (actual_namespace or bootstrap["settings"]["ros_namespace"]).strip("/") + "/world_m"
    assert reference.header.frame_id == "map" and reference.child_frame_id == expected_child
    assert observer.world_frame == reference.child_frame_id
    t, q = reference.transform.translation, reference.transform.rotation
    assert (t.x, t.y, t.z) == pytest.approx((-2.0, 1.0, -3.0))
    assert (q.z, q.w) == pytest.approx((-math.sqrt(0.5), math.sqrt(0.5)))
    # A +90 degree map has world +X along map -Y; its declared world origin maps to zero.
    rotation = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert rotation @ np.array(bootstrap["settings"]["origin_m"]) + [t.x, t.y, t.z] == pytest.approx([0.0, 0.0, 0.0])
    assert rotation @ np.array([1.0, 0.0, 0.0]) == pytest.approx([0.0, -1.0, 0.0])


def test_generated_rviz_binds_the_real_session_and_only_read_only_tools(tmp_path):
    _, bootstrap = _bootstrap(tmp_path)
    original = bootstrap.read_bytes()
    path = monitor.write_monitor_files(
        bootstrap,
        tmp_path,
        "/people/session_123abc",
        {"ros_domain_id": 59, "ros_setup_path": "/opt/ros/jazzy/setup.bash"},
    )
    manifest = json.loads(path.read_text())
    full_config = yaml.safe_load((tmp_path / manifest["rviz_config"]).read_text())
    config = full_config["Visualization Manager"]
    assert manifest["namespace"] == "/people/session_123abc" and manifest["read_only"]
    assert config["Displays"][0]["Name"] == "Static map"
    assert any(display["Name"] == "Route: visitor" for display in config["Displays"])
    assert manifest["canonical_world_frame"] == "people/session_123abc/world_m"
    assert manifest["topics"]["tf_static"] == "/tf_static"
    assert [panel["Class"] for panel in full_config["Panels"]] == ["rviz_common/Displays"]
    assert full_config["Window Geometry"] == {"Width": 1600, "Height": 1000}
    assert config["Views"]["Current"]["Scale"] == pytest.approx(500 / (20 * 0.5))
    assert config["Global Options"]["Fixed Frame"] == "map"
    assert config["Views"]["Current"]["Class"] == "rviz_default_plugins/TopDownOrtho"
    assert (config["Views"]["Current"]["X"], config["Views"]["Current"]["Y"]) == (-5.0, 5.5)
    assert all("SetGoal" not in item["Class"] and "SetInitialPose" not in item["Class"] for item in config["Tools"])
    assert all(item["Topic"]["Value"].startswith(manifest["namespace"] + "/") for item in config["Displays"])
    assert all(item["Topic"]["Durability Policy"] == "Transient Local" for item in config["Displays"])
    assert bootstrap.read_bytes() == original


def test_monitor_cli_resolves_status_and_dry_run_never_starts_a_simulation(monkeypatch, tmp_path, capsys):
    _, bootstrap = _bootstrap(tmp_path)
    monitor.write_monitor_files(
        bootstrap,
        tmp_path,
        "/people/session_private",
        {"ros_domain_id": 59, "ros_setup_path": "/opt/ros/jazzy/setup.bash"},
    )
    status = tmp_path / "status.json"
    status.write_text(json.dumps({"output_directory": str(tmp_path)}))
    monkeypatch.setattr(workflow, "_ros_environment", lambda path: {"PATH": "/fake/bin"})
    monkeypatch.setattr(monitor.shutil, "which", lambda *args, **kwargs: "/fake/bin/rviz2")
    monkeypatch.setattr(launcher, "run_simulation", lambda *args, **kwargs: pytest.fail("Monitor started Kit"))
    monkeypatch.setattr(monitor.subprocess, "run", lambda *args, **kwargs: pytest.fail("Dry run started RViz"))
    launcher.main(["monitor", str(status), "--dry-run"])
    result = json.loads(capsys.readouterr().out)
    assert result["read_only"] and result["namespace"] == "/people/session_private" and result["ros_domain_id"] == 59
    assert result["command"] == [
        "/fake/bin/rviz2",
        "-d",
        str(tmp_path / "monitor.rviz"),
        "--ros-args",
        "-p",
        "use_sim_time:=true",
    ]


def test_monitor_rejects_cyclic_status_references_and_config_escape(tmp_path):
    path = tmp_path / "status.json"
    path.write_text(json.dumps({"output_directory": str(path)}))
    with pytest.raises(ValueError, match="cycle"):
        monitor.launch_ros2_monitor(path, dry_run=True)
    path.write_text(json.dumps({"schema_version": 1, "rviz_config": "../foreign.rviz"}))
    with pytest.raises(ValueError, match="inside"):
        monitor.launch_ros2_monitor(path, dry_run=True)
