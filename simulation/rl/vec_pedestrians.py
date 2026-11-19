# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The workshop's scripted pedestrians, replayed inside every cloned room.

``configs/third_3/simulation.yaml`` choreographs four waypoint walkers: a spawn
pose and yaw, a start delay, then a ring of waypoints with a pause and a facing
angle each. The deployed runtime plays that script through a ROS 2 pedestrian
node, and the bodies the robot can actually hit are invisible kinematic capsules
of radius 0.25 m and height 1.75 m — the Isaac People skeletons on top are
visual only.

This module keeps exactly the parts that change the robot's outcome:

* the same four routes, speeds, pauses and start delays, in the room's own
  coordinates (the robot's spawn pose in the same file matches the robot asset,
  so both live in one frame);
* the same capsule footprint, used as a planar keep-out disk so that driving into
  a pedestrian fails the episode the way a wall does;
* per-step positions and velocities so the frozen policy's 12,800-element
  pedestrian velocity map carries real people instead of zeros;
* the Isaac People character USDs from ``Character-assets/``, one per walker in
  every cloned room, moved with the route replay so the rooms are not empty.

The ROS 2 transport and the retargeted skeleton animation stay out: the routes are
replayed here in NumPy, and the characters keep their bind pose while their root
transform follows the walker. Nothing in the success protocol depends on either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .bridge import PedestrianState

PEDESTRIAN_RADIUS_M = 0.25
PEDESTRIAN_HEIGHT_M = 1.75
TURN_RATE_RAD_S = math.radians(90.0)
SCENARIO_RELATIVE = "configs/third_3/simulation.yaml"


@dataclass(frozen=True)
class Waypoint:
    """One stop of a route, in room coordinates.

    Args:
        x: World x [m].
        y: World y [m].
        yaw_rad: Facing the walker holds while it pauses [rad].
        wait_s: Pause after arriving and turning [s].
    """

    x: float
    y: float
    yaw_rad: float
    wait_s: float


@dataclass(frozen=True)
class Route:
    """The scripted walk of one pedestrian.

    Args:
        name: Actor key from the scenario file.
        start_xy: Spawn position [m].
        yaw_rad: Spawn facing [rad].
        start_time_s: Delay before the first leg [s].
        speed_m_s: Walking speed [m/s].
        arrival_tolerance_m: Distance that counts as reaching a waypoint [m].
        repeat: Whether the walker loops back to the first waypoint.
        waypoints: The ordered stops.
    """

    name: str
    start_xy: tuple[float, float]
    yaw_rad: float
    start_time_s: float
    speed_m_s: float
    arrival_tolerance_m: float
    repeat: bool
    waypoints: tuple[Waypoint, ...]


@dataclass(frozen=True)
class _Capacity:
    """Mirror of the observation bridge's shape contract."""

    max_pedestrians: int


def load_routes(path) -> tuple[Route, ...]:
    """Read the walkers out of the scenario YAML.

    The project's own loader validates that every character USD exists, which it
    cannot on these machines, so the pedestrian block is read directly.

    Args:
        path: Scenario YAML with a ``pedestrians`` block.

    Returns:
        One :class:`Route` per actor, in file order.

    Raises:
        ValueError: If the block is missing or a route is malformed.
    """
    import yaml

    with open(path, encoding="utf-8") as handle:
        scenario = yaml.safe_load(handle) or {}
    block = scenario.get("pedestrians")
    if not isinstance(block, dict) or not block.get("actors"):
        raise ValueError(f"{path} has no pedestrians.actors block")
    defaults = block.get("defaults") or {}
    speed = float(defaults.get("speed", 0.9))
    tolerance = float(defaults.get("arrival_tolerance", 0.2))
    default_wait = float(defaults.get("wait", 0.0))
    repeat = bool(defaults.get("repeat", True))

    routes: list[Route] = []
    for name, actor in block["actors"].items():
        position = actor.get("position")
        stops = actor.get("waypoints") or ()
        if position is None or not stops:
            raise ValueError(f"{name}: a pedestrian needs a position and at least one waypoint")
        waypoints = tuple(
            Waypoint(
                x=float(stop["position"][0]),
                y=float(stop["position"][1]),
                yaw_rad=math.radians(float(stop.get("yaw", 0.0))),
                wait_s=float(stop.get("wait", default_wait)),
            )
            for stop in stops
        )
        routes.append(
            Route(
                name=str(name),
                start_xy=(float(position[0]), float(position[1])),
                yaw_rad=math.radians(float(actor.get("yaw", 0.0))),
                start_time_s=float(actor.get("start_time", 0.0)),
                speed_m_s=float(actor.get("speed", speed)),
                arrival_tolerance_m=float(actor.get("arrival_tolerance", tolerance)),
                repeat=bool(actor.get("repeat", repeat)),
                waypoints=waypoints,
            )
        )
    return tuple(routes)


def _turn_toward(current: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
    """Rotate ``current`` toward ``target`` at :data:`TURN_RATE_RAD_S` [rad]."""
    difference = (np.asarray(target) - np.asarray(current) + math.pi) % (2.0 * math.pi) - math.pi
    limit = TURN_RATE_RAD_S * dt
    return current + np.clip(difference, -limit, limit)


class VecPedestrians:
    """Replay the routes for ``num_envs`` rooms and publish their planar state.

    The class mimics the runtime bridge closely enough that the reward and the
    frozen observation can read it: ``config.max_pedestrians`` plus a ``state``
    with ``positions_w``, ``velocities_w``, ``radii`` and ``valid``, one row per
    environment.

    Args:
        routes: The scripted walks, in the room's coordinates.
        num_envs: How many rooms share the choreography.
        device: Device of the published tensors.
        dt: Seconds between two calls to :meth:`advance`.
        radius_m: Capsule radius used as the keep-out disk [m].
    """

    def __init__(
        self,
        routes: tuple[Route, ...],
        *,
        num_envs: int,
        device: torch.device | str = "cpu",
        dt: float = 0.05,
        radius_m: float = PEDESTRIAN_RADIUS_M,
    ) -> None:
        if not routes:
            raise ValueError("At least one pedestrian route is required")
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError("dt must be finite and positive")
        self.routes = tuple(routes)
        self.num_envs = int(num_envs)
        self.count = len(self.routes)
        self.dt = float(dt)
        self.radius_m = float(radius_m)
        self.device = torch.device(device)
        self.config = _Capacity(self.count)

        self._pos = np.zeros((self.num_envs, self.count, 2), dtype=np.float64)
        self._yaw = np.zeros((self.num_envs, self.count), dtype=np.float64)
        self._index = np.zeros((self.num_envs, self.count), dtype=np.int64)
        self._clock = np.zeros((self.num_envs, self.count), dtype=np.float64)
        self._wait = np.zeros((self.num_envs, self.count), dtype=np.float64)
        self._velocity = np.zeros((self.num_envs, self.count, 2), dtype=np.float64)
        self._stamp: tuple[int, int] | None = None

        self.state = PedestrianState(
            positions_w=torch.zeros((self.num_envs, self.count, 3), dtype=torch.float32, device=self.device),
            velocities_w=torch.zeros((self.num_envs, self.count, 3), dtype=torch.float32, device=self.device),
            radii=torch.full((self.num_envs, self.count), self.radius_m, dtype=torch.float32, device=self.device),
            valid=torch.ones((self.num_envs, self.count), dtype=torch.bool, device=self.device),
        )
        self.reset(None)
        self._publish(np.zeros((self.num_envs, 2), dtype=np.float64))

    # -- lifecycle -------------------------------------------------------

    def reset(self, env: object = None, env_ids: object = None) -> None:
        """Put the selected rooms back at the start of the choreography."""
        if env_ids is None:
            ids = np.arange(self.num_envs, dtype=np.int64)
        else:
            # The manager hands over a tensor on the environment device, but the
            # route replay works in NumPy, so normalise once here.
            ids = torch.as_tensor(env_ids).detach().cpu().numpy().astype(np.int64).reshape(-1)
        if ids.size == 0:
            return
        for slot, route in enumerate(self.routes):
            self._pos[ids, slot, 0] = route.start_xy[0]
            self._pos[ids, slot, 1] = route.start_xy[1]
            self._yaw[ids, slot] = route.yaw_rad
            self._index[ids, slot] = 0
            self._clock[ids, slot] = 0.0
            self._wait[ids, slot] = 0.0
            self._velocity[ids, slot] = 0.0
        self._stamp = None
        if env is not None:
            self._publish(self._origins(env))

    def advance(self, env: object) -> None:
        """Move the walkers one ``dt`` forward, once per policy step.

        Args:
            env: Environment owning ``common_step_counter`` and ``episode_length_buf``.
        """
        stamp = (
            int(getattr(env, "common_step_counter", 0)),
            int(torch.as_tensor(getattr(env, "episode_length_buf", [0])).reshape(-1)[0].item()),
        )
        if self._stamp == stamp:
            return
        self._stamp = stamp
        self._advance_arrays()
        self._publish(self._origins(env))

    # -- internals -------------------------------------------------------

    def _advance_arrays(self) -> None:
        previous = self._pos.copy()
        for slot, route in enumerate(self.routes):
            stops = np.array([[stop.x, stop.y] for stop in route.waypoints], dtype=np.float64)
            facings = np.array([stop.yaw_rad for stop in route.waypoints], dtype=np.float64)
            pauses = np.array([stop.wait_s for stop in route.waypoints], dtype=np.float64)
            position = self._pos[:, slot]
            yaw = self._yaw[:, slot]
            index = self._index[:, slot]
            clock = self._clock[:, slot] + self.dt
            wait = self._wait[:, slot]
            self._clock[:, slot] = clock

            active = clock >= route.start_time_s
            if not bool(active.any()):
                continue
            target = stops[index]
            delta = target - position
            distance = np.linalg.norm(delta, axis=1)
            idle = wait > 0.0

            if bool(idle.any()):
                self._wait[idle, slot] = np.maximum(wait[idle] - self.dt, 0.0)
                yaw = np.where(idle, _turn_toward(yaw, facings[index], self.dt), yaw)

            travel = active & ~idle
            moving = travel & (distance > 1e-6)
            if bool(moving.any()):
                stride = np.minimum(route.speed_m_s * self.dt, distance)
                direction = delta / np.maximum(distance, 1e-9)[:, None]
                position = np.where(moving[:, None], position + direction * stride[:, None], position)
                heading = np.arctan2(delta[:, 1], delta[:, 0])
                yaw = np.where(moving, _turn_toward(yaw, heading, self.dt), yaw)
                reached = travel & (distance <= max(route.arrival_tolerance_m, route.speed_m_s * self.dt))
                if bool(reached.any()):
                    # Snap onto the stop so repeated episodes do not drift.
                    position = np.where(reached[:, None], target, position)
                    self._wait[reached, slot] = pauses[index[reached]]
                    yaw = np.where(reached, facings[index], yaw)
                    following = index[reached] + 1
                    if route.repeat:
                        following = following % len(route.waypoints)
                    else:
                        following = np.minimum(following, len(route.waypoints) - 1)
                    self._index[reached, slot] = following

            self._pos[:, slot] = position
            self._yaw[:, slot] = yaw
        self._velocity = (self._pos - previous) / self.dt

    def _origins(self, env: object) -> np.ndarray:
        origins = getattr(getattr(env, "scene", None), "env_origins", None)
        if not isinstance(origins, torch.Tensor) or origins.numel() < 2:
            return np.zeros((self.num_envs, 2), dtype=np.float64)
        values = origins if origins.dtype == torch.float32 else origins.float()
        matrix = values.detach().cpu().numpy().astype(np.float64)
        if matrix.shape[0] != self.num_envs:
            return np.zeros((self.num_envs, 2), dtype=np.float64)
        return matrix[:, :2]

    def _publish(self, origins: np.ndarray) -> None:
        world = self._pos + origins[:, None, :]
        positions = np.zeros((self.num_envs, self.count, 3), dtype=np.float32)
        positions[:, :, 0] = world[:, :, 0]
        positions[:, :, 1] = world[:, :, 1]
        positions[:, :, 2] = PEDESTRIAN_HEIGHT_M / 2.0
        velocities = np.zeros((self.num_envs, self.count, 3), dtype=np.float32)
        velocities[:, :, 0] = self._velocity[:, :, 0]
        velocities[:, :, 1] = self._velocity[:, :, 1]
        self.state = PedestrianState(
            positions_w=torch.as_tensor(positions, device=self.device),
            velocities_w=torch.as_tensor(velocities, device=self.device),
            radii=torch.full((self.num_envs, self.count), self.radius_m, dtype=torch.float32, device=self.device),
            valid=torch.ones((self.num_envs, self.count), dtype=torch.bool, device=self.device),
        )

    # -- diagnostics -----------------------------------------------------

    def room_xy(self) -> np.ndarray:
        """Walkers in room coordinates, shape ``[num_envs, count, 2]`` [m]."""
        return self._pos.copy()

    def room_yaw(self) -> np.ndarray:
        """Walker facings in room coordinates, shape ``[num_envs, count]`` [rad]."""
        return self._yaw.copy()



# ---------------------------------------------------------------------------
# Character assets
# ---------------------------------------------------------------------------

CHARACTER_CATALOG_RELATIVE = "configs/runtime/characters.yaml"
CHARACTER_ASSET_ROOT_RELATIVE = "Character-assets"


def default_asset_root() -> Path:
    """Return the repository's ``Character-assets`` directory."""
    return Path(__file__).resolve().parents[2] / CHARACTER_ASSET_ROOT_RELATIVE


def load_character_pool(
    asset_root: Path | str | None = None,
    catalog: Path | str | None = None,
    gender: str = "any",
) -> tuple[Path, ...]:
    """List the character USDs the catalog declares, in its own order.

    Args:
        asset_root: Directory that contains ``Isaac/People/...``; defaults to this
            repository's ``Character-assets``.
        catalog: ``characters.yaml``; defaults to the scenario configuration.
        gender: ``male``, ``female`` or ``any``, as the project loader defines it.

    Returns:
        Absolute paths to existing character USDs.

    Raises:
        ValueError: If the catalog cannot be resolved against ``asset_root``.
    """
    from ..configuration.scenarios.assets import find_character_assets

    root = Path(asset_root) if asset_root is not None else default_asset_root()
    source = (
        Path(catalog)
        if catalog is not None
        else Path(__file__).resolve().parents[1] / CHARACTER_CATALOG_RELATIVE
    )
    return find_character_assets(source, gender, asset_root=str(root))


def pick_characters(pool: tuple[Path, ...], count: int, *, offset: int = 0) -> tuple[Path, ...]:
    """Take ``count`` characters from ``pool``, cycling and keeping the order stable.

    Args:
        pool: Candidate USDs.
        count: Walkers per room.
        offset: Rotation applied to the pool, so different machines can show
            different people without changing the physics.
    """
    if not pool:
        raise ValueError("At least one character asset is required")
    if int(count) < 1:
        raise ValueError("count must be positive")
    return tuple(pool[(int(offset) + index) % len(pool)] for index in range(int(count)))


class VecCharacterStage:
    """One Isaac People character per walker and room, driven by the route replay.

    The characters are visual: the collision the protocol scores is the planar
    capsule of :class:`VecPedestrians`. Their root transform is written straight
    into USD, once per call to :meth:`sync`, which is why the caller can throttle
    it without touching the physics.

    Args:
        characters: One USD per walker; the list repeats across rooms.
        num_envs: Number of cloned rooms.
        visual_stride: Policy steps between two transform writes; 1 writes every
            step, 0 disables the writes entirely.
    """

    def __init__(self, characters: tuple[Path, ...], *, num_envs: int, visual_stride: int = 2) -> None:
        if not characters:
            raise ValueError("At least one character asset is required")
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        self.characters = tuple(Path(path) for path in characters)
        self.num_envs = int(num_envs)
        self.count = len(self.characters)
        self.visual_stride = max(int(visual_stride), 0)
        self.prim_paths: list[list[str]] = []
        self._ops: list[list[tuple[object, object]]] = []
        self._foot_z: list[float] = []
        self.spawned = False
        self.writes = 0

    # -- construction ----------------------------------------------------

    def spawn(self, stage: object, routes: tuple[Route, ...], *, root: str = "/World/envs") -> None:
        """Reference every character USD into every room at its route start pose.

        Args:
            stage: Open USD stage that already holds the cloned rooms.
            routes: Scripted walks; one character per route.
            root: Parent of the cloned rooms.
        """
        from pxr import Gf, Usd, UsdGeom

        from isaaclab import sim as sim_utils

        if len(routes) != self.count:
            raise ValueError(f"Got {len(routes)} routes for {self.count} characters")

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for index, route in enumerate(routes):
            cfg = sim_utils.UsdFileCfg(usd_path=str(self.characters[index]))
            paths: list[str] = []
            ops: list[tuple[object, object]] = []
            for env_index in range(self.num_envs):
                prim_path = f"{root}/env_{env_index}/Pedestrian_{index}"
                # No translation here: ``sync`` owns the root pose, so the base it
                # reads back stays the asset's own offset instead of the spawn one.
                cfg.func(prim_path, cfg)
                prim = stage.GetPrimAtPath(prim_path)
                if not prim or not prim.IsValid():
                    raise RuntimeError(f"Character prim was not created: {prim_path}")
                xformable = UsdGeom.Xformable(prim)
                translate, orient, base_xyz, base_quat = self._transform_ops(xformable, UsdGeom)
                paths.append(prim_path)
                ops.append((translate, orient, base_xyz, base_quat))
            # The character origin sits slightly above its soles, so lift the root
            # by the local bound instead of guessing a constant per asset.
            local_min = cache.ComputeLocalBound(stage.GetPrimAtPath(paths[0])).ComputeAlignedRange().GetMin()
            foot = float(-local_min[2]) if local_min[2] < 0.0 else 0.0
            self._foot_z.append(foot)
            self.prim_paths.append(paths)
            self._ops.append(ops)
        self.spawned = True

    @staticmethod
    def _transform_ops(xformable: object, UsdGeom: object) -> tuple[object, object, tuple, tuple]:
        """Return the pose ops plus the transform the asset already carries.

        Two details force the bookkeeping:

        * the character assets ship double-precision transform ops, and
          ``XformOp.Set`` rejects the other precision, so the op keeps its type;
        * the referenced asset can already rotate its root, so the walker yaw is
          composed on top of that orientation instead of replacing it.
        """
        translate = None
        orient = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate is None:
                translate = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient is None:
                orient = op

        if translate is None:
            translate = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            base_xyz = (0.0, 0.0, 0.0)
        else:
            value = translate.Get()
            base_xyz = tuple(float(component) for component in value) if value is not None else (0.0, 0.0, 0.0)

        if orient is None:
            orient = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            base_quat = (1.0, 0.0, 0.0, 0.0)
        else:
            value = orient.Get()
            if value is None:
                base_quat = (1.0, 0.0, 0.0, 0.0)
            else:
                imaginary = value.GetImaginary()
                base_quat = (
                    float(value.GetReal()),
                    float(imaginary[0]),
                    float(imaginary[1]),
                    float(imaginary[2]),
                )
        return translate, orient, base_xyz, base_quat

    # -- per-step --------------------------------------------------------

    def sync(self, crowd: VecPedestrians, *, step: int = 0, force: bool = False) -> bool:
        """Write the walker poses into the stage.

        Args:
            crowd: Route replay that owns the positions and facings.
            step: Policy step counter, used for the stride.
            force: Ignore the stride, e.g. right after a reset.

        Returns:
            Whether the stage was written.
        """
        if not self.spawned or self.visual_stride == 0:
            return False
        if not force and int(step) % self.visual_stride != 0:
            return False
        from pxr import Gf, UsdGeom

        positions = crowd.room_xy()
        facings = crowd.room_yaw()
        double = UsdGeom.XformOp.PrecisionDouble
        for index in range(self.count):
            z = self._foot_z[index]
            for env_index in range(self.num_envs):
                translate, orient, base_xyz, base_quat = self._ops[index][env_index]
                x, y = positions[env_index, index]
                yaw = float(facings[env_index, index])
                shift = (base_xyz[0] + float(x), base_xyz[1] + float(y), base_xyz[2] + z)
                if translate.GetPrecision() == double:
                    translate.Set(Gf.Vec3d(*shift))
                else:
                    translate.Set(Gf.Vec3f(*shift))
                cosine, sine = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
                quaternion = _quat_mul((cosine, 0.0, 0.0, sine), base_quat)
                if orient.GetPrecision() == double:
                    orient.Set(Gf.Quatd(*quaternion))
                else:
                    orient.Set(Gf.Quatf(*quaternion))
        self.writes += 1
        return True

    def attach(self, env: object, crowd: VecPedestrians, *, root: str = "/World/envs") -> "VecCharacterStage":
        """Spawn the characters for ``env`` and return this stage wrapper."""
        self.spawn(env.sim.stage, crowd.routes, root=root)
        self.sync(crowd, force=True)
        return self


def _quat_mul(left: tuple, right: tuple) -> tuple:
    """Hamilton product of two ``(w, x, y, z)`` quaternions."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )
