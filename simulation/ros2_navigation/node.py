# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run occupancy planning and crowd control as an independent ROS 2 process."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from .planner import OccupancyGrid
from .solver import NavigationEngine
from .visualization import NavigationMonitor


def main(argv: list[str] | None = None) -> None:
    """Start a real ROS node using the explicitly selected bootstrap and library."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--namespace", default="/pedestrian_sim")
    args, ros_args = parser.parse_known_args(argv)
    bootstrap_path = args.bootstrap.expanduser().resolve()
    os.environ.setdefault("ROS_LOG_DIR", str(bootstrap_path.parent / "ros_logs"))
    bootstrap = json.loads(bootstrap_path.read_text())
    map_path = Path(bootstrap["map_npz"])
    if not map_path.is_absolute():
        map_path = bootstrap_path.parent / map_path
    with np.load(map_path, allow_pickle=False) as data:
        grid = OccupancyGrid(
            data["occupancy"],
            float(data["resolution_m"]),
            tuple(data["origin_xy_m"]),
            goal_projection_tolerance_m=bootstrap["settings"].get("goal_projection_tolerance_m"),
        )
    engine = NavigationEngine(bootstrap["plans"], grid, bootstrap["settings"], bootstrap["agents"], args.library)

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import OccupancyGrid as MapMessage
    from nav_msgs.msg import Path as PathMessage
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from std_msgs.msg import String

    class NavigationNode(Node):
        def __init__(self):
            super().__init__(
                "pedestrian_navigation",
                namespace=args.namespace,
                parameter_overrides=[Parameter("use_sim_time", value=True)],
            )
            latched = QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
            )
            self.map_publisher = self.create_publisher(MapMessage, "map", latched)
            self.command_publisher = self.create_publisher(String, "command", 1)
            self.subscription = self.create_subscription(String, "state", self.on_state, 1)
            self.paths = {name: self.create_publisher(PathMessage, f"{name}/path", latched) for name in engine.names}
            self.path_signatures = {}
            self.last_sequence = -1
            self.last_time = None
            self.cached = None
            self.last_request = None
            message = MapMessage()
            message.header.frame_id = "map"
            message.info.resolution = grid.resolution_m
            message.info.width, message.info.height = grid.width, grid.height
            message.info.origin.position.x, message.info.origin.position.y = grid.origin_xy_m
            message.info.origin.position.z = float(bootstrap["settings"].get("floor_height_m", 0.0))
            message.info.origin.orientation.w = 1.0
            message.data = grid.occupancy.ravel().tolist()
            self.map_publisher.publish(message)
            self.monitor = NavigationMonitor(self, bootstrap)
            self.get_logger().info("ROS occupancy planner and RVO2 crowd controller ready")

        def on_state(self, message):
            sequence = None
            time_s = None
            try:
                request = json.loads(message.data)
                sequence = request["sequence"]
                fingerprint = json.dumps(request, sort_keys=True, allow_nan=False)
                if type(sequence) is not int or sequence < 0:
                    raise ValueError("sequence must be a nonnegative integer.")
                request_time, dt = request["time_s"], request["dt"]
                if type(request_time) not in (int, float) or not math.isfinite(request_time) or request_time < 0:
                    raise ValueError("time_s must be a finite nonnegative number [s].")
                time_s = float(request_time)
                if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
                    raise ValueError("dt must be a finite positive number [s].")
                if sequence == self.last_sequence and time_s == self.last_time:
                    if fingerprint != self.last_request:
                        raise ValueError("A duplicate sequence cannot change its body or obstacle snapshot.")
                    self.command_publisher.publish(String(data=self.cached))
                    return
                if sequence <= self.last_sequence:
                    raise ValueError("State sequences must strictly increase; a duplicate tick is never reintegrated.")
                if self.last_time is not None:
                    if time_s <= self.last_time:
                        raise ValueError("New navigation requests must strictly advance simulation time [s].")
                    if not math.isclose(time_s - self.last_time, dt, rel_tol=1e-6, abs_tol=1e-9):
                        raise ValueError("Navigation dt must match the simulation-time increment [s].")
                output = engine.step(
                    time_s,
                    float(request["dt"]),
                    request["agents"],
                    request.get("obstacles", []),
                    request.get("doors", []),
                )
                output["sequence"] = sequence
                encoded = json.dumps(output, allow_nan=False)
                self.last_sequence, self.last_time, self.cached = sequence, time_s, encoded
                self.last_request = fingerprint
                self.command_publisher.publish(String(data=encoded))
            except Exception as error:
                if not self.context.ok():
                    return
                self.get_logger().error(f"Navigation request rejected: {type(error).__name__}: {error}")
                response = {"sequence": sequence, "error": str(error), "agents": []}
                if time_s is not None:
                    response["time_s"] = time_s
                self.command_publisher.publish(String(data=json.dumps(response, allow_nan=False)))
                return
            # Observation failures must never replace an already accepted control answer.
            self._publish_paths(time_s)
            try:
                self.monitor.publish(request, output)
            except Exception as error:
                self.get_logger().warning(f"RViz observation snapshot failed: {error}", throttle_duration_sec=5.0)

        def _publish_paths(self, time_s: float) -> None:
            for name, route in engine.routes.items():
                try:
                    signature = tuple(tuple(point) for point in route.path)
                    if self.path_signatures.get(name) == signature:
                        continue
                    path = PathMessage()
                    path.header.frame_id = "map"
                    path.header.stamp = Time(seconds=time_s).to_msg()
                    for point in route.path:
                        pose = PoseStamped()
                        pose.header = path.header
                        pose.pose.position.x, pose.pose.position.y = point
                        pose.pose.position.z = engine.bodies[name]["position"][2]
                        pose.pose.orientation.w = 1.0
                        path.poses.append(pose)
                    self.paths[name].publish(path)
                    self.path_signatures[name] = signature
                except Exception as error:
                    self.get_logger().warning(f"RViz path for {name} failed: {error}", throttle_duration_sec=5.0)

    rclpy.init(args=ros_args)
    node = None
    try:
        node = NavigationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        engine.close()
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
