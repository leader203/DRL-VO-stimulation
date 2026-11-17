# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sequenced ROS 2 DDS transport between the simulator and navigation process."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


class Ros2Transport:
    """Exchange complete, timestamped batches over ROS 2; never plan or integrate motion."""

    def __init__(
        self, bootstrap: Path, library: Path, settings: dict[str, object], output: Path, *, publish_clock: bool = True
    ) -> None:
        """Start an owned node and bind simulation-time ROS messages [s]."""
        self._context = self._node = self._process = self._log = self._executor = None
        self._output = Path(output)
        try:
            self._initialize(bootstrap, library, settings, output, publish_clock)
        except BaseException:
            self.close()
            raise

    def _initialize(
        self, bootstrap: Path, library: Path, settings: dict[str, object], output: Path, publish_clock: bool
    ) -> None:
        try:
            import rclpy
            from nav_msgs.msg import OccupancyGrid
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import DurabilityPolicy, QoSProfile
            from rosgraph_msgs.msg import Clock
            from std_msgs.msg import String
        except ImportError as error:
            raise RuntimeError(
                f"ROS 2 Python is unavailable. Source {settings['ros_setup_path']} before starting the simulator."
            ) from error
        self._rclpy, self._String, self._Clock = rclpy, String, Clock
        self._context = Context()
        rclpy.init(context=self._context, domain_id=int(settings["ros_domain_id"]))
        self.namespace = str(settings["ros_namespace"]).rstrip("/") + "/session_" + uuid.uuid4().hex[:8]
        self._node = Node("isaac_pedestrian_adapter", namespace=self.namespace, context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._replies: dict[int, dict[str, object]] = {}
        self._map_received = False
        self._sequence = 0
        self._inflight = None
        self._inflight_time = None
        self._timeout = float(settings["request_timeout_s"])
        self._publisher = self._node.create_publisher(String, self.namespace + "/state", 10)
        self._clock = self._node.create_publisher(Clock, "/clock", 10) if publish_clock else None
        self._node.create_subscription(String, self.namespace + "/command", self._on_command, 10)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._node.create_subscription(OccupancyGrid, self.namespace + "/map", self._on_map, qos)
        self._log = (output / "ros2_node.log").open("w")
        environment = dict(os.environ)
        package_root = str(Path(__file__).resolve().parents[3])
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        environment["ROS_DOMAIN_ID"] = str(settings["ros_domain_id"])
        environment["ROS_LOG_DIR"] = str(output / "ros_logs")
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pedstrain_simulation.simulation.ros2_navigation.node",
                "--bootstrap",
                str(bootstrap),
                "--library",
                str(library),
                "--namespace",
                self.namespace,
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        self._wait(lambda: self._map_received, "ROS 2 navigation map discovery")
        from .monitor import write_monitor_files

        self.monitor_manifest = write_monitor_files(bootstrap, output, self.namespace, settings)

    def create_publisher(self, message_type, topic: str, qos: int = 10):
        """Create a publisher on the owned ROS 2 node without exposing the node."""
        if self._node is None:
            raise RuntimeError("ROS 2 transport is closed.")
        return self._node.create_publisher(message_type, topic, qos)

    def _on_map(self, message) -> None:
        if (
            message.info.width <= 0
            or message.info.height <= 0
            or len(message.data) != message.info.width * message.info.height
        ):
            raise RuntimeError("ROS 2 navigation published an invalid OccupancyGrid.")
        self._map_received = True

    def _on_command(self, message) -> None:
        data = json.loads(message.data)
        sequence = data.get("sequence")
        if type(sequence) is int and sequence == getattr(self, "_inflight", self._sequence):
            # Older nodes omitted the timestamp only on explicit failure replies.
            if not data.get("error") or "time_s" in data:
                reply_time = data.get("time_s")
                expected_time = self._inflight_time
                if (
                    type(reply_time) not in (int, float)
                    or not math.isfinite(reply_time)
                    or expected_time is None
                    or not math.isclose(reply_time, expected_time, rel_tol=0.0, abs_tol=1e-9)
                ):
                    return
            self._replies[sequence] = data

    def _wait(self, predicate, operation: str, retry=None) -> None:
        deadline = time.monotonic() + self._timeout
        next_retry = time.monotonic() + 0.2
        while not predicate():
            if self._process.poll() is not None:
                self._log.flush()
                raise RuntimeError(f"{operation} failed: navigation process exited; inspect {self._log.name}.")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{operation} timed out after {self._timeout:g} wall seconds.")
            if retry is not None and time.monotonic() >= next_retry:
                retry()
                next_retry = time.monotonic() + 0.2
            self._executor.spin_once(timeout_sec=min(0.02, max(0.0, deadline - time.monotonic())))

    def step(
        self, time_s: float, dt: float, states: list[dict[str, object]], obstacles=(), door_states=()
    ) -> dict[str, object]:
        """Request exactly one navigation step at simulation time [s], interval [s]."""
        if self._clock is not None:
            clock = self._Clock()
            clock.clock.sec = int(time_s)
            clock.clock.nanosec = int(round((time_s - int(time_s)) * 1e9))
            if clock.clock.nanosec >= 1000000000:
                clock.clock.sec += 1
                clock.clock.nanosec = 0
            self._clock.publish(clock)
        message = self._String()
        message.data = json.dumps(
            {
                "sequence": self._sequence,
                "time_s": time_s,
                "dt": dt,
                "agents": states,
                "obstacles": list(obstacles),
                "doors": list(door_states),
            },
            allow_nan=False,
        )
        self._publisher.publish(message)
        self._inflight = self._sequence
        self._inflight_time = float(time_s)
        try:
            self._wait(
                lambda: self._sequence in self._replies,
                f"ROS 2 navigation step {self._sequence}",
                retry=lambda: self._publisher.publish(message),
            )
        except Exception as error:
            self._retain_failed_request(message.data, error)
            raise
        finally:
            self._inflight = None
            self._inflight_time = None
        result = self._replies.pop(self._sequence)
        self._sequence += 1
        if result.get("error"):
            failure = RuntimeError(f"ROS 2 navigation failed: {result['error']}")
            self._retain_failed_request(message.data, failure)
            raise failure
        return result

    def _retain_failed_request(self, request: str, failure: Exception) -> None:
        try:
            (self._output / "last_failed_request.json").write_text(request, encoding="utf-8")
        except Exception as diagnostic_error:
            failure.add_note(f"Could not retain failed ROS request: {diagnostic_error}")

    def close(self) -> None:
        """Stop only this owned process and release its ROS context."""
        process = getattr(self, "_process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if getattr(self, "_node", None) is not None:
            if self._executor is not None:
                self._executor.remove_node(self._node)
                self._executor.shutdown(timeout_sec=2.0)
                self._executor = None
            self._node.destroy_node()
            self._node = None
        if getattr(self, "_context", None) is not None and self._context.ok():
            self._context.shutdown()
        if getattr(self, "_log", None) is not None:
            self._log.close()
