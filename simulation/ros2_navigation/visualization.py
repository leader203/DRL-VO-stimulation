# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only, simulation-time RViz snapshots in the existing ROS map frame."""

from __future__ import annotations

import colorsys
import hashlib
import math
from typing import TYPE_CHECKING

from .frames import StageFrame

if TYPE_CHECKING:
    from rclpy.node import Node


def monitor_color(name: str) -> tuple[float, float, float]:
    """Return a stable display color independently of actor order or scene names."""
    hue = int.from_bytes(hashlib.sha256(name.encode()).digest()[:2], "big") / 65536
    return colorsys.hsv_to_rgb(hue, 0.65, 0.95)


def monitor_world_frame(namespace: str) -> str:
    """Return the session-qualified canonical-world TF child without a leading slash."""
    return f"{namespace.strip('/') or 'pedestrian_sim'}/world_m"


class NavigationMonitor:
    """Publish complete observed MarkerArray snapshots at most every 0.1 [s].

    Positions and footprints use the ROS ``map`` frame [m], actor heading [rad],
    and timestamps use simulation time [s]. Plans retain their original yaw [deg].
    Issued velocity arrows show a one-second command projection, never measured
    motion. This publisher has no navigation subscriptions or command publisher.
    """

    def __init__(self, node: Node, bootstrap: dict) -> None:
        from geometry_msgs.msg import Point, TransformStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from tf2_msgs.msg import TFMessage
        from visualization_msgs.msg import Marker, MarkerArray

        self._marker, self._array, self._point = Marker, MarkerArray, Point
        self._plans = bootstrap["plans"]
        self._settings = bootstrap["settings"]
        self._last_time: float | None = None
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher = node.create_publisher(MarkerArray, "markers", qos)
        self.reference_publisher = node.create_publisher(TFMessage, "/tf_static", qos)
        get_namespace = getattr(node, "get_namespace", None)
        namespace = (
            get_namespace() if callable(get_namespace) else self._settings.get("ros_namespace", "/pedestrian_sim")
        )
        self.world_frame = monitor_world_frame(namespace)
        # This reference describes canonical meters/Z-up, independently of raw USD units or axes.
        frame = StageFrame(
            1.0, "Z", self._settings.get("origin_m", (0.0, 0.0, 0.0)), self._settings.get("yaw_rad", 0.0)
        )
        reference = TransformStamped()
        reference.header.frame_id, reference.child_frame_id = "map", self.world_frame
        translation = reference.transform.translation
        translation.x, translation.y, translation.z = frame.world_to_ros((0.0, 0.0, 0.0))
        reference.transform.rotation.z = math.sin(-frame.yaw_rad / 2)
        reference.transform.rotation.w = math.cos(-frame.yaw_rad / 2)
        self.reference_publisher.publish(TFMessage(transforms=[reference]))
        self.publish({"time_s": 0.0, "agents": bootstrap["agents"]}, {"agents": []})

    def publish(self, request: dict, command: dict) -> bool:
        """Publish measured state and separately labelled intentions; never mutate a batch.

        Args:
            request: Original observed state, obstacles and door readings [m, rad, s].
            command: Issued next-step poses/velocities [m, rad, m/s] and diagnostics.

        Returns:
            Whether a complete snapshot was published on this simulation tick.
        """
        now = float(request["time_s"])
        if self._last_time is not None and 0 <= now - self._last_time < 0.1 - 1e-9:
            return False
        marker_type = self._marker
        markers = []
        seconds, nanoseconds = divmod(round(now * 1e9), 1_000_000_000)

        def marker(namespace, kind, color):
            item = marker_type()
            item.header.frame_id = "map"
            item.header.stamp.sec, item.header.stamp.nanosec = seconds, nanoseconds
            item.ns, item.id, item.type = namespace, len(markers), kind
            item.action = marker_type.ADD
            item.pose.orientation.w = 1.0
            item.scale.x = item.scale.y = item.scale.z = 1.0
            item.color.r, item.color.g, item.color.b, item.color.a = map(float, color)
            markers.append(item)
            return item

        def point(position, offset=0.0):
            return self._point(x=float(position[0]), y=float(position[1]), z=float(position[2]) + offset)

        def line(namespace, positions, color, *, width=0.025):
            item = marker(namespace, marker_type.LINE_STRIP, color)
            item.scale.x = width
            item.points = [point(p) for p in positions]
            return item

        def label(namespace, position, text, color):
            item = marker(namespace, marker_type.TEXT_VIEW_FACING, color)
            item.pose.position = point(position, 0.18)
            item.scale.z, item.text = 0.14, text

        def arrow(namespace, start, end, color):
            item = marker(namespace, marker_type.ARROW, color)
            item.points = [point(start, 0.08), point(end, 0.08)]
            item.scale.x, item.scale.y, item.scale.z = 0.035, 0.075, 0.10

        reset = marker("", marker_type.CUBE, (0, 0, 0, 0))
        reset.action = marker_type.DELETEALL
        observed = {state["name"]: state for state in request["agents"]}
        issued = {state["name"]: state for state in command["agents"]}
        waits = {wait["name"]: wait for wait in command.get("goal_occupancy", {}).get("queue", ())}
        for name, state in observed.items():
            color = monitor_color(name)
            position = state["position"]
            radius = float(state.get("radius_m", self._settings.get("body_radius_m", 0.25)))
            body = marker("observed/body", marker_type.CYLINDER, (*color, 0.75))
            body.pose.position = point(position)
            body.pose.orientation.z = math.sin(state["yaw"] / 2)
            body.pose.orientation.w = math.cos(state["yaw"] / 2)
            body.scale.x = body.scale.y = 2 * radius
            body.scale.z = 0.05
            heading = [
                position[0] + 2 * radius * math.cos(state["yaw"]),
                position[1] + 2 * radius * math.sin(state["yaw"]),
                position[2],
            ]
            arrow("observed/heading", position, heading, (*color, 1.0))
            phase = issued.get(name, {}).get("phase", "initial")
            text = f"{name} OBSERVED\ncontrol phase: {phase}"
            if name in waits:
                text += f"\nqueue: {waits[name]['holder']}"
            label("observed/labels", position, text, (*color, 1.0))
            if name in issued:
                velocity = issued[name]["velocity"]
                if math.hypot(*velocity[:2]) > 1e-6:
                    projected = [position[i] + velocity[i] for i in range(3)]
                    arrow("issued/velocity_1s", position, projected, (1.0, 0.55, 0.0, 0.85))

        for plan in self._plans:
            color = monitor_color(plan["name"])
            tolerance = float(plan.get("arrival_tolerance", 0.15))
            for index, goal in enumerate(plan.get("waypoints", [])):
                item = marker("requested/yaml_goal_tolerance", marker_type.SPHERE, (*color, 0.18))
                item.pose.position = point(goal["position"])
                item.scale.x = item.scale.y = item.scale.z = 2 * tolerance
                label("requested/yaml_goal_labels", goal["position"], f"{plan['name']} YAML {index + 1}", (*color, 0.9))

        floor = float(self._settings.get("floor_height_m", 0.0))
        for obstacle in request.get("obstacles", ()):
            polygon = obstacle.get("footprint_xy_m")
            if polygon:
                height = float(obstacle.get("z_min_m", floor)) + 0.03
                vertices = [[*p, height] for p in polygon]
                line("observed/dynamic_footprints", [*vertices, vertices[0]], (1.0, 0.35, 0.15, 1.0))
            elif obstacle.get("radius_m") is not None:
                item = marker("observed/dynamic_discs", marker_type.CYLINDER, (1.0, 0.35, 0.15, 0.5))
                item.pose.position = point(obstacle["position"])
                item.scale.x = item.scale.y = 2 * float(obstacle["radius_m"])
                item.scale.z = 0.05

        doors = {door["name"]: door for door in request.get("doors", ())}
        for spec in self._settings.get("door_specs", ()):
            if not spec.get("enabled", True):
                continue
            portal = spec.get("doorway_segment_xy_m")
            if portal and len(portal) >= 2:
                line(
                    "reference/door_portal",
                    [[p[0], p[1], floor + 0.06] for p in portal],
                    (0.2, 0.9, 1.0, 0.8),
                    width=0.035,
                )
            else:
                portal = None
            sweep = spec.get("swept_footprint_xy_m")
            if sweep and len(sweep) >= 3:
                vertices = [[p[0], p[1], floor + 0.015] for p in sweep]
                line("requested/door_sweep", [*vertices, vertices[0]], (0.2, 0.9, 1.0, 0.20), width=0.015)
            state = doors.get(spec["name"], {})
            location = spec.get("hinge_position_m")
            if location is None:
                if portal is None:
                    continue
                location = [(portal[0][i] + portal[1][i]) / 2 for i in (0, 1)] + [floor]
            angle = state.get("actual_angle_deg")
            angle_text = "unobserved" if not state.get("valid") or angle is None else f"{angle:.1f}deg"
            condition = "closed" if state.get("fully_closed") else "open" if state.get("fully_open") else "moving/held"
            if not state.get("valid"):
                condition = "unobserved"
            color = (0.25, 0.9, 0.4, 1.0) if state.get("fully_closed") else (1.0, 0.7, 0.15, 1.0)
            label(
                "observed/door_state", location, f"{spec['name'].rsplit('/', 1)[-1]} {angle_text}\n{condition}", color
            )

        for wait in command.get("goal_occupancy", {}).get("queue", ()):
            name = wait["name"]
            if name not in observed:
                continue
            body = observed[name]
            refuge = wait.get("refuge_xy_m")
            if refuge is not None:
                line("requested/queue_refuge", [body["position"], [*refuge, body["position"][2]]], (1, 1, 0.3, 0.7))
        for name, transaction in command.get("door_closure", {}).get("transactions", {}).items():
            if name in observed and transaction.get("target") is not None:
                position = observed[name]["position"]
                line("requested/door_refuge", [position, [*transaction["target"], position[2]]], (1, 0.3, 1, 0.8))
        for name, directive in command.get("guidance", {}).items():
            target = directive.get("target")
            if name in observed and target is not None:
                position = observed[name]["position"]
                destination = target if len(target) == 3 else [*target, position[2]]
                line("requested/current_guidance", [position, destination], (1.0, 0.85, 0.1, 0.9))
                item = marker("requested/guide_point", marker_type.SPHERE, (1.0, 0.85, 0.1, 1.0))
                item.pose.position = point(destination)
                item.scale.x = item.scale.y = item.scale.z = 0.09

        message = self._array(markers=markers)
        self.publisher.publish(message)
        self._last_time = now
        return True
