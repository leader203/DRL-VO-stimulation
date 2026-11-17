# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build the frozen DRL-VO 19,202-element observation from ROS map state."""

from __future__ import annotations

from types import ModuleType

import numpy as np

OBSERVATION_SIZE = 19202
SCAN_HISTORY = 10
FRONT_BEAM_COUNT = 720


class DrlObservationAdapter:
    """Own the scan history and assemble the external package's observation."""

    def __init__(self, api: ModuleType) -> None:
        self.api = api
        self.history = api.ScanHistory()

    def reset(self) -> None:
        self.history.reset()

    def append_scan(self, scan: np.ndarray, timestamp_ns: int) -> None:
        self.history.append(scan, timestamp_ns)

    def sample_history(self, timestamp_ns: int) -> bool:
        return bool(self.history.sample(timestamp_ns))

    @property
    def history_ready(self) -> bool:
        return bool(self.history.ready)

    def history_array(self) -> np.ndarray:
        return np.asarray(self.history.array(), dtype=np.float32)

    def build_observation(
        self,
        states: list[dict[str, object]],
        robot_pose_xyyaw: tuple[float, float, float],
        local_goal_base_xy: np.ndarray | tuple[float, float],
    ) -> tuple[np.ndarray, dict[str, object]]:
        pose = np.asarray(robot_pose_xyyaw, dtype=np.float32)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("robot_pose_xyyaw must contain three finite values.")
        goal = np.asarray(local_goal_base_xy, dtype=np.float32)
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("local_goal_base_xy must contain two finite values [m].")
        if states:
            positions = np.asarray([state["position"][:2] for state in states], dtype=np.float32)
            velocities = np.asarray([state["velocity"][:2] for state in states], dtype=np.float32)
        else:
            positions = np.zeros((0, 2), dtype=np.float32)
            velocities = np.zeros((0, 2), dtype=np.float32)
        if positions.shape != (len(states), 2) or velocities.shape != (len(states), 2):
            raise ValueError("Pedestrian states must expose planar position and velocity.")
        pedestrian_map = self.api.oracle_pedestrian_velocity_map(positions, velocities, pose)
        observation = self.api.build_observation(
            pedestrian_map, self.history_array(), goal, clip_pedestrian=False
        )
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
            raise RuntimeError(f"DRL observation is invalid: shape={observation.shape}")
        metadata = {
            "pedestrian_count": len(states),
            "history_frames": int(len(self.history.frames)),
            "local_goal_base_xy_m": [float(goal[0]), float(goal[1])],
            "pedestrian_map_min_mps": float(pedestrian_map.min()) if pedestrian_map.size else 0.0,
            "pedestrian_map_max_mps": float(pedestrian_map.max()) if pedestrian_map.size else 0.0,
        }
        return observation, metadata
