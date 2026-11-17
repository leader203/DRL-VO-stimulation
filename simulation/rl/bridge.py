# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clock ROS pedestrians from an external PhysX loop and expose measured state."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class PedestrianState:
    """One world's stable, name-sorted slots, padded to a fixed capacity.

    Positions [m] and velocities [m/s] are canonical Z-up world vectors with
    shape [1, N, 3]. Radii [m] and validity masks have shape [1, N].
    """

    positions_w: torch.Tensor
    velocities_w: torch.Tensor
    radii: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class _ObservationConfig:
    max_pedestrians: int


def _physics_ratio(physics_dt_s: float, control_hz: int) -> int:
    if (
        isinstance(physics_dt_s, bool)
        or not isinstance(physics_dt_s, Real)
        or not math.isfinite(physics_dt_s)
        or not 0 < physics_dt_s <= 1.0 / control_hz + 1e-12
    ):
        raise ValueError("Physics dt must be finite, positive and no longer than the ROS control interval [s].")
    ratio = (1.0 / control_hz) / physics_dt_s
    if not math.isfinite(ratio) or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("Each ROS control interval must contain an integer number of physics substeps.")
    return round(ratio)


class _TrainingPhysicsBodies:
    def __init__(self, stage, actors) -> None:
        import torch

        import omni.physics.tensors as tensors
        from pxr import UsdUtils

        paths = {f"{actors.root_path}/{body['name']}": body["name"] for body in actors.states()}
        stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
        self._simulation = tensors.create_simulation_view("torch", stage_id=stage_id, backend="physx")
        self._simulation.set_subspace_roots("/")
        self._view = self._simulation.create_rigid_body_view(list(paths))
        actual = tuple(self._view.prim_paths)
        if len(actual) != len(paths) or set(actual) != set(paths):
            raise RuntimeError("PhysX did not bind exactly the prepared pedestrian kinematic roots.")
        self._names = tuple(paths[path] for path in actual)
        self._targets = self._view.get_transforms()
        self._indices = torch.arange(len(actual), dtype=torch.int32, device=self._targets.device)
        self._read_views = {}

    def read_poses(self, paths: tuple[str, ...], *, group: str) -> dict[str, dict]:
        if not paths:
            self._read_views.pop(group, None)
            return {}
        requested, view = self._read_views.get(group, ((), None))
        if requested != paths:
            view = self._simulation.create_rigid_body_view(list(paths))
            actual = tuple(view.prim_paths)
            if len(actual) != len(paths) or set(actual) != set(paths):
                raise RuntimeError(f"PhysX could not bind the live ROS training {group}: {paths}")
            self._read_views[group] = paths, view
        return {
            path: {"ret_val": True, "position": pose[:3], "rotation": pose[3:]}
            for path, pose in zip(view.prim_paths, view.get_transforms().tolist(), strict=True)
        }

    def apply(self, commands, frame) -> None:
        by_name = {body["name"]: body for body in commands}
        transforms = []
        for name in self._names:
            body = by_name[name]
            transforms.append((*frame.ros_to_stage(body["position"]), *frame.ros_yaw_to_stage_quaternion(body["yaw"])))
        self._targets.copy_(self._targets.new_tensor(transforms))
        if self._targets.is_cuda:
            # The GPU rigid-body view does not implement kinematic targets.
            # Apply the certified pose before stepping the task's physics.
            self._view.set_transforms(self._targets, self._indices)
        else:
            self._view.set_kinematic_targets(self._targets, self._indices)

    def close(self) -> None:
        self._read_views.clear()
        self._view = self._simulation = None


class Ros2PedestrianBridge:
    """Drive a prepared single-world crowd before the task's own physics step."""

    def __init__(
        self, *, frame, navigation, actors, obstacles, doors, transport, kinematics, capacity, device, physics_dt_s
    ) -> None:
        self._ratio = _physics_ratio(physics_dt_s, navigation.control_hz)
        self._thread = threading.get_ident()
        self._physics_dt_s = physics_dt_s
        self._frame = frame
        self._navigation = navigation
        self._actors, self._obstacles, self._doors, self._transport = actors, obstacles, doors, transport
        self._kinematics = kinematics
        self.config = _ObservationConfig(capacity)
        self._device = device
        self._substeps = self._control_steps = 0
        self._previous: dict[str, tuple[float, ...]] = {}
        self._closed = self._failed = False
        bodies = self._read_bodies()
        self._names = tuple(body["name"] for body in bodies)
        self._state = self._capture_state(bodies)

    @property
    def control_steps(self) -> int:
        """Number of applied ROS 2 command batches."""
        return self._control_steps

    @property
    def state(self) -> PedestrianState:
        """Return the latest measured observation snapshot, treated as read-only."""
        self._require_ready()
        return self._state

    def _read_bodies(self):
        return sorted(self._actors.states(), key=lambda body: body["name"]) if self._actors is not None else []

    def _capture_state(self, bodies) -> PedestrianState:
        import torch

        count, capacity = len(bodies), self.config.max_pedestrians
        if count > capacity or tuple(body["name"] for body in bodies) != self._names:
            raise RuntimeError("Prepared ROS pedestrian slots changed during training.")
        positions = torch.zeros((1, capacity, 3), dtype=torch.float32, device=self._device)
        velocities = torch.zeros_like(positions)
        radii = positions.new_zeros((1, capacity))
        valid = torch.zeros((1, capacity), dtype=torch.bool, device=self._device)
        if count:
            positions[0, :count] = positions.new_tensor([self._frame.ros_to_world(b["position"]) for b in bodies])
            velocities[0, :count] = positions.new_tensor(
                [self._frame.ros_vector_to_world(b["velocity"]) for b in bodies]
            )
            radii[0, :count] = positions.new_tensor([b["radius_m"] for b in bodies])
            valid[0, :count] = True
        return PedestrianState(positions, velocities, radii, valid)

    def advance(self, physics_dt_s: float) -> None:
        """Run a scheduled ROS update before one task-owned physics substep [s]."""
        self._require_ready()
        if (
            isinstance(physics_dt_s, bool)
            or not isinstance(physics_dt_s, Real)
            or not math.isclose(physics_dt_s, self._physics_dt_s, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError("Pass the configured physics dt once per physics substep [s].")
        self._substeps += 1
        if self._transport is None or self._substeps % self._ratio:
            return
        try:
            self._advance_crowd()
        except BaseException:
            self._failed = True
            self._state = None
            raise

    def _advance_crowd(self) -> None:
        dt = 1.0 / self._navigation.control_hz
        now = (self._control_steps + 1) / self._navigation.control_hz
        bodies = self._read_bodies()
        doors = self._doors.observations(now)
        obstacles = []
        observed = {}
        floor = self._navigation.floor_height_m
        for obstacle in self._obstacles.sample():
            if obstacle.get("z_max_m", math.inf) < floor or obstacle.get("z_min_m", -math.inf) > (
                floor + self._navigation.body_height_m
            ):
                continue
            name = obstacle["prim_path"]
            position = tuple(obstacle["position"])
            previous = self._previous.get(name)
            velocity = (
                [0.0, 0.0, 0.0] if previous is None else [(p - q) / dt for p, q in zip(position, previous, strict=True)]
            )
            obstacles.append({**obstacle, "velocity": velocity, "velocity_valid": True})
            observed[name] = position
        command = self._transport.step(now, dt, bodies, obstacles=obstacles, door_states=doors)
        for actor in command["agents"]:
            if actor.get("error") or actor.get("phase") == "failed":
                raise RuntimeError(f"ROS pedestrian {actor.get('name')} failed: {actor.get('error') or 'navigation'}")
        self._actors.apply(command["agents"], dt)
        # Isaac Lab steps PhysX directly; the USD visual pose alone is not a
        # kinematic target for the collision body on this execution path.
        self._kinematics.apply(command["agents"], self._frame)
        self._doors.set_navigation_intents(command.get("door_commands", []))
        self._doors.update(bodies, now, dt)
        self._state = self._capture_state(self._read_bodies())
        self._previous = observed
        self._control_steps += 1

    def reset(self, env_ids=None) -> None:
        """Clear obstacle velocity history at a robot reset, preserving the crowd."""
        self._require_ready()
        if env_ids is not None:
            if isinstance(env_ids, slice):
                ids = range(1)[env_ids]
            else:
                ids = tuple(index.item() if hasattr(index, "item") else index for index in env_ids)
            if not ids:
                return
            if len(ids) != 1 or isinstance(ids[0], bool) or not isinstance(ids[0], Integral) or ids[0] != 0:
                raise ValueError("ROS pedestrian training supports only environment ID 0.")
        self._previous.clear()

    def _check_thread(self) -> None:
        if threading.get_ident() != self._thread:
            raise RuntimeError("ROS training bridge access must stay on its owning Kit thread.")

    def _require_ready(self) -> None:
        self._check_thread()
        if self._closed:
            raise RuntimeError("The ROS training bridge is closed.")
        if self._failed:
            raise RuntimeError("A ROS training tick failed; close the bridge and rebuild the training scene.")

    def close(self) -> None:
        """Close DDS and its observers without advancing or stopping the task's physics."""
        self._check_thread()
        self._closed = True
        self._state = None
        errors = []
        for name in ("_transport", "_kinematics", "_obstacles", "_doors"):
            resource = getattr(self, name)
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                errors.append(error)
            else:
                setattr(self, name, None)
        if errors:
            for error in errors[1:]:
                errors[0].add_note(f"Additional ROS training cleanup failure: {error!r}")
            raise errors[0]
