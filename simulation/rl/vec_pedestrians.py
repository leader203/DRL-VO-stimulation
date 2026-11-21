# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The deployed crowd, replayed inside every cloned room.

The real pedestrian stack has two layers, and this module mirrors both:

* **Motion.** A ROS 2 process owns each root's planar pose. Every control tick it
  plans an A* path on the 2 cm occupancy map, blends social forces, resolves the
  crowd with RVO2 and schedules waypoints, turns and dwells. Isaac never plans;
  it consumes ``position``, ``yaw`` and ``velocity``.
* **Contact and appearance.** The body the robot can hit is an invisible
  kinematic capsule of radius 0.25 m and height 1.8 m; the Isaac People skeleton
  on top is visual only, skinned by a retargeted Walk/Idle cycle.

None of the four walkers observes the robot, so their trajectories are a
deterministic function of simulation time and are identical in every clone of
the room. :mod:`gen_ped_trace` therefore runs the solver once, offline, and this
module replays that table by index lookup: 512 rooms cost one gather per step
instead of 2048 planner calls.

The replay keeps the invariants the robot actually feels:

* no walker ever leaves the free space of the occupancy map — the table is baked
  from A* paths that respect the 0.25 m body disk, and :meth:`VecPedestrians.audit`
  re-checks that claim on every frame;
* walkers keep RVO2's pair clearance, so they never fuse into one body;
* ``positions_w`` and ``velocities_w`` carry the true crowd into the reward and
  into the policy's pedestrian velocity map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .bridge import PedestrianState

PEDESTRIAN_RADIUS_M = 0.25
PEDESTRIAN_HEIGHT_M = 1.8
SCENARIO_RELATIVE = "configs/third_3/simulation.yaml"


@dataclass(frozen=True)
class Waypoint:
    """One stop of a route, in room coordinates.

    Args:
        x: Room x [m].
        y: Room y [m].
        yaw_rad: Facing the walker holds while it pauses [rad].
        wait_s: Pause after arriving and turning [s].
    """

    x: float
    y: float
    yaw_rad: float
    wait_s: float


@dataclass(frozen=True)
class Route:
    """The scripted walk of one pedestrian, as the scenario declares it.

    Only the spawn pose survives into the baked table; the rest documents where
    the walker came from and lets the start sampler keep robots off its disk.

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
    waypoints: tuple[Waypoint, ...] = ()


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


class VecPedestrians:
    """Replay the baked ROS 2 crowd for ``num_envs`` rooms.

    The class mimics the runtime bridge closely enough that the reward and the
    frozen observation can read it: ``config.max_pedestrians`` plus a ``state``
    with ``positions_w``, ``velocities_w``, ``radii`` and ``valid``, one row per
    environment.

    Args:
        trace: ``npz`` written by :mod:`gen_ped_trace` — ``positions`` and
            ``velocities`` of shape ``[frames, walkers, 3]`` in room
            coordinates, ``yaws`` of shape ``[frames, walkers]``, ``names`` and
            ``sample_hz``.
        num_envs: How many rooms replay the same table.
        device: Device of the published tensors.
        dt: Policy step [s]; converts an episode length into a table index.
        radius_m: Capsule radius used as the keep-out disk [m].
        phase: ``episode`` restarts the crowd with every episode; ``global``
            keeps one clock for the whole run.

    Raises:
        ValueError: If the table is empty, malformed or shorter than one step.
    """

    def __init__(
        self,
        trace,
        *,
        num_envs: int,
        device: torch.device | str = "cpu",
        dt: float = 0.05,
        radius_m: float = PEDESTRIAN_RADIUS_M,
        height_m: float = PEDESTRIAN_HEIGHT_M,
        phase: str = "episode",
        speed_scale: float = 1.0,
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError("dt must be finite and positive")
        if phase not in ("episode", "global"):
            raise ValueError("phase must be 'episode' or 'global'")
        if not (math.isfinite(speed_scale) and speed_scale > 0.0):
            raise ValueError("speed_scale must be finite and positive")

        with np.load(trace, allow_pickle=False) as data:
            positions = np.array(data["positions"], dtype=np.float32)
            velocities = np.array(data["velocities"], dtype=np.float32)
            yaws = np.array(data["yaws"], dtype=np.float32)
            names = [str(name) for name in data["names"]]
            sample_hz = float(data["sample_hz"])
        if positions.ndim != 3 or positions.shape[2] != 3:
            raise ValueError(f"{trace}: positions must be [frames, walkers, 3], got {positions.shape}")
        if positions.shape[0] < 2:
            raise ValueError(f"{trace}: need at least two frames, got {positions.shape[0]}")
        if velocities.shape != positions.shape or yaws.shape != positions.shape[:2]:
            raise ValueError(f"{trace}: positions, velocities and yaws disagree in shape")
        if not (math.isfinite(sample_hz) and sample_hz > 0):
            raise ValueError(f"{trace}: sample_hz must be finite and positive")

        self.trace_path = Path(trace)
        self.num_envs = int(num_envs)
        self.count = int(positions.shape[1])
        self.frames = int(positions.shape[0])
        self.sample_hz = sample_hz
        self.duration_s = self.frames / sample_hz
        self.dt = float(dt)
        self.radius_m = float(radius_m)
        self.height_m = float(height_m)
        self.phase = phase
        # Table index per policy step. Below 1 the crowd walks its baked path
        # slower without ever leaving it, so the zero-penetration guarantee of
        # the bake survives: only the rate along the path changes.
        self.speed_scale = float(speed_scale)
        self.device = torch.device(device)
        self.names = names
        self.config = _Capacity(self.count)

        self.table_positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        self.table_velocities = torch.as_tensor(velocities, dtype=torch.float32, device=self.device)
        self.table_yaws = torch.as_tensor(yaws, dtype=torch.float32, device=self.device)
        # The planner reports z on the floor; the bridge and the reward expect the
        # capsule centre, and the scan padding assumes the same body height.
        self.table_positions[:, :, 2] = float(height_m) * 0.5

        spawn = positions[0]
        self.routes = tuple(
            Route(
                name=name,
                start_xy=(float(spawn[index, 0]), float(spawn[index, 1])),
                yaw_rad=float(yaws[0, index]),
                start_time_s=0.0,
                speed_m_s=float(np.linalg.norm(velocities[:, index, :2], axis=1).max()),
                arrival_tolerance_m=0.2,
                repeat=True,
            )
            for index, name in enumerate(names)
        )

        self.state = PedestrianState(
            positions_w=torch.zeros((self.num_envs, self.count, 3), dtype=torch.float32, device=self.device),
            velocities_w=torch.zeros((self.num_envs, self.count, 3), dtype=torch.float32, device=self.device),
            radii=torch.full((self.num_envs, self.count), self.radius_m, dtype=torch.float32, device=self.device),
            valid=torch.ones((self.num_envs, self.count), dtype=torch.bool, device=self.device),
        )
        self._index = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._room = torch.zeros((self.num_envs, self.count, 2), dtype=torch.float32, device=self.device)
        self._stamp: tuple[int, int] | None = None
        self.advance(None, force=True)

    # -- lifecycle -------------------------------------------------------

    def reset(self, env: object = None, env_ids: object = None) -> None:
        """Put the selected rooms back at the first frame of the table.

        The table index follows the episode clock, so the next :meth:`advance`
        lands on frame zero on its own; forcing one here keeps the published
        state consistent for whichever reward runs before that.
        """
        if env_ids is None:
            self._index.fill_(-1)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
            if ids.numel() == 0:
                return
            self._index[ids] = -1
        self._publish_frame(torch.zeros_like(self._index), env)

    def advance(self, env: object, *, force: bool = False) -> None:
        """Publish the crowd for the current policy step.

        Args:
            env: Environment owning ``episode_length_buf``; ``None`` publishes the
                first frame on the room origin, which is what construction needs.
            force: Recompute even when the index did not move.
        """
        index = self._frame_index(env)
        if not force and torch.equal(index, self._index):
            return
        self._index = index
        self._publish_frame(index, env)

    # -- internals -------------------------------------------------------

    def _frame_index(self, env: object) -> torch.Tensor:
        """Map the environment clock onto a table row."""
        if self.phase == "global":
            step = int(getattr(env, "common_step_counter", 0) or 0)
            frame = int(round(step * self.dt * self.sample_hz * self.speed_scale))
            return torch.full((self.num_envs,), frame % self.frames, dtype=torch.long, device=self.device)
        lengths = getattr(env, "episode_length_buf", None) if env is not None else None
        if lengths is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        lengths = torch.as_tensor(lengths).to(device=self.device, dtype=torch.float32).reshape(-1)
        if lengths.numel() != self.num_envs:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        frame = torch.round(lengths * self.dt * self.sample_hz * self.speed_scale).to(torch.long)
        return frame.clamp_(0, self.frames - 1)

    def _publish_frame(self, index: torch.Tensor, env: object) -> None:
        positions = self.table_positions[index]
        self._room.copy_(positions[:, :, :2])
        origins = self._origins(env)
        world = positions.clone()
        world[:, :, 0] += origins[:, None, 0]
        world[:, :, 1] += origins[:, None, 1]
        self.state.positions_w.copy_(world)
        self.state.velocities_w.copy_(self.table_velocities[index])
        self._yaw = self.table_yaws[index]

    def _origins(self, env: object) -> torch.Tensor:
        """Room origins on the environment device, shape ``[num_envs, 2]``."""
        scene = getattr(env, "scene", None) if env is not None else None
        origins = getattr(scene, "env_origins", None)
        empty = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        if not isinstance(origins, torch.Tensor) or origins.numel() < 2:
            return empty
        values = origins.to(device=self.device, dtype=torch.float32)
        if values.shape[0] != self.num_envs:
            return empty
        return values[:, :2]

    # -- diagnostics -----------------------------------------------------

    def room_xy(self) -> np.ndarray:
        """Walkers in room coordinates, shape ``[num_envs, count, 2]`` [m]."""
        return self._room.detach().cpu().numpy()

    def room_yaw(self) -> np.ndarray:
        """Walker facings in room coordinates, shape ``[num_envs, count]`` [rad]."""
        return self._yaw.detach().cpu().numpy()

    def summary(self) -> dict:
        """Describe the baked table without touching the map.

        The wall audit lives in :mod:`gen_ped_trace`, which runs it against the
        same ``OccupancyGrid`` the planner used; this keeps one authority for the
        claim that no walker ever leaves free space.
        """
        positions = self.table_positions.detach().cpu().numpy()
        steps = np.linalg.norm(np.diff(positions[:, :, :2], axis=0), axis=2)
        speeds = np.linalg.norm(self.table_velocities.detach().cpu().numpy()[:, :, :2], axis=2)
        return {
            "frames": self.frames,
            "duration_s": round(self.duration_s, 2),
            "sample_hz": self.sample_hz,
            "walkers": self.names,
            "travel_m": [round(float(value), 2) for value in steps.sum(axis=0)],
            "max_speed_mps": round(float(speeds.max()), 3),
            "max_step_m": round(float(steps.max()), 4),
        }


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
    """One Isaac People character per walker and room, driven by the baked replay.

    The characters are visual. The body the robot can hit is the planar capsule of
    :class:`VecPedestrians`; this class only skins a mesh onto the same root pose.

    Two layers live here, mirroring the deployed stack:

    * the **root** comes straight from the replay table, written into the prim's
      translate and orient ops once per :meth:`sync`;
    * the **skeleton** is a retargeted Walk/Idle cycle, sampled by travelled
      distance so a slow walker takes short steps instead of sliding, and written
      into a ``Pose`` ``SkelAnimation`` bound to the asset's own skeleton.

    The retargeting reuses :class:`~...ros2_navigation.actors._Rig`, which aims each
    target bone at the direction the matching clip bone points. That is the only
    mapping that survives the two rigs having different up axes, different local
    bone rolls and different leg lengths; a straight copy of the source rotations
    leaves the character T-posed or with a leg above its head.

    Args:
        characters: One USD per walker; the list repeats across rooms.
        num_envs: Number of cloned rooms.
        visual_stride: Policy steps between two transform writes; 1 writes every
            step, 0 disables the writes entirely.
        visual_envs: Rooms that carry the skinned USDs. Only the room the
            evaluation camera looks into needs them, so a 512-room rollout pays
            for four characters instead of two thousand and forty-eight.
        motion_library: ``HumanMotionLibrary.usd``; defaults to the one under the
            asset root.
        animate: Skinning on/off. A still bind pose is enough when only the
            capsule matters, and skipping the skeleton write saves the retarget.
        dt: Policy step length [s], used to turn travel into speed for the stride.
    """

    def __init__(
        self,
        characters: tuple[Path, ...],
        *,
        num_envs: int,
        visual_stride: int = 2,
        visual_envs: object = None,
        motion_library: Path | str | None = None,
        animate: bool = True,
        dt: float = 0.05,
    ) -> None:
        if not characters:
            raise ValueError("At least one character asset is required")
        if int(num_envs) < 1:
            raise ValueError("num_envs must be positive")
        if not (math.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be finite and positive [s]")
        self.characters = tuple(Path(path) for path in characters)
        self.num_envs = int(num_envs)
        self.count = len(self.characters)
        self.visual_stride = max(int(visual_stride), 0)
        if visual_envs is None:
            self.visual_envs = tuple(range(self.num_envs))
        else:
            self.visual_envs = tuple(sorted({int(index) for index in visual_envs}))
            if not self.visual_envs or self.visual_envs[0] < 0 or self.visual_envs[-1] >= self.num_envs:
                raise ValueError(f"visual_envs must index the {self.num_envs} rooms")
        self.motion_library = None if motion_library is None else Path(motion_library)
        self._motion_cache = None
        self.animate = bool(animate)
        self.dt = float(dt)
        self.rigs: list[object] = []
        self.prim_paths: list[list[str]] = []
        self._ops: list[list[tuple[object, object, tuple, tuple]]] = []
        self._anims: list[list[object]] = []
        self._gait: list[list[_Gait]] = []
        self._foot_z: list[float] = []
        self._last_step: int | None = None
        self.spawned = False
        self.writes = 0
        self.pose_writes = 0

    # -- construction ----------------------------------------------------

    def _motion(self):
        """Load the shared motion library, once for every character."""
        from ..ros2_navigation.actors import _Motion

        if getattr(self, "_motion_cache", None) is None:
            path = self.motion_library
            if path is None:
                from ..configuration.resources import DEFAULT_MOTION_LIBRARY

                path = Path(default_asset_root()) / DEFAULT_MOTION_LIBRARY
            self._motion_cache = _Motion(Path(path))
        return self._motion_cache

    def spawn(self, stage: object, routes: tuple[Route, ...], *, root: str = "/World/envs") -> None:
        """Reference every character USD into the visual rooms at its start pose.

        Args:
            stage: Open USD stage that already holds the cloned rooms.
            routes: Scripted walks; one character per route.
            root: Parent of the cloned rooms.
        """
        from pxr import Gf, Usd, UsdGeom, UsdSkel, Vt

        from isaaclab import sim as sim_utils

        if len(routes) != self.count:
            raise ValueError(f"Got {len(routes)} routes for {self.count} characters")

        motion = self._motion() if self.animate else None
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for index, route in enumerate(routes):
            rig = None
            if self.animate:
                from ..ros2_navigation.actors import _Rig

                rig = _Rig(self.characters[index], motion)
            self.rigs.append(rig)
            cfg = sim_utils.UsdFileCfg(usd_path=str(self.characters[index]))
            paths: list[str] = []
            ops: list[tuple[object, object, tuple, tuple]] = []
            anims: list[object] = []
            gaits: list[_Gait] = []
            for env_index in self.visual_envs:
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
                if rig is not None:
                    skeleton = next(child for child in Usd.PrimRange(prim) if child.IsA(UsdSkel.Skeleton))
                    animation = UsdSkel.Animation.Define(stage, f"{prim_path}/Pose")
                    animation.CreateJointsAttr(rig.joints)
                    animation.CreateTranslationsAttr(Vt.Vec3fArray([Gf.Vec3f()] * len(rig.joints)))
                    animation.CreateRotationsAttr(Vt.QuatfArray([Gf.Quatf(1.0)] * len(rig.joints)))
                    animation.CreateScalesAttr(Vt.Vec3hArray([Gf.Vec3h(1.0)] * len(rig.joints)))
                    UsdSkel.BindingAPI(skeleton).CreateAnimationSourceRel().SetTargets([animation.GetPath()])
                    anims.append(animation)
                gaits.append(_Gait(xy=None, idle_time=0.37 * index, walk_position=0.31 * index, weight=0.0))
            # The character origin sits slightly above its soles, so lift the root
            # by the local bound instead of guessing a constant per asset.
            local_min = cache.ComputeLocalBound(stage.GetPrimAtPath(paths[0])).ComputeAlignedRange().GetMin()
            foot = float(-local_min[2]) if local_min[2] < 0.0 else 0.0
            self._foot_z.append(foot)
            self.prim_paths.append(paths)
            self._ops.append(ops)
            self._anims.append(anims)
            self._gait.append(gaits)
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
        """Write the walker poses and their skinned skeletons into the stage.

        Args:
            crowd: Replay table that owns the positions and facings.
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

        # Time since the previous write, so a throttled crowd still reports the
        # true root speed and the stride follows it.
        if force or self._last_step is None:
            elapsed = self.dt
        else:
            elapsed = self.dt * max(1, int(step) - self._last_step)
        self._last_step = int(step)

        positions = crowd.room_xy()
        facings = crowd.room_yaw()
        double = UsdGeom.XformOp.PrecisionDouble
        for index in range(self.count):
            z = self._foot_z[index]
            rig = self.rigs[index] if self.animate else None
            authored = float(rig.authored_yaw) if rig is not None else 0.0
            for slot, env_index in enumerate(self.visual_envs):
                translate, orient, base_xyz, base_quat = self._ops[index][slot]
                x, y = positions[env_index, index]
                yaw = float(facings[env_index, index])
                shift = (base_xyz[0] + float(x), base_xyz[1] + float(y), base_xyz[2] + z)
                if translate.GetPrecision() == double:
                    translate.Set(Gf.Vec3d(*shift))
                else:
                    translate.Set(Gf.Vec3f(*shift))
                # The asset faces its own authored direction where ROS heading zero
                # faces +X, so the prim carries the heading minus that offset.
                half = (yaw - authored) * 0.5
                quaternion = _quat_mul((math.cos(half), 0.0, 0.0, math.sin(half)), base_quat)
                if orient.GetPrecision() == double:
                    orient.Set(Gf.Quatd(*quaternion))
                else:
                    orient.Set(Gf.Quatf(*quaternion))
                if rig is not None:
                    self._write_pose(index, slot, rig, (float(x), float(y)), elapsed)
        self.writes += 1
        return True

    def _write_pose(self, index: int, slot: int, rig: object, xy: tuple[float, float], elapsed: float) -> None:
        """Advance one walker's gait by the travelled distance and push the pose."""
        from pxr import Gf, Vt

        from ..ros2_navigation.actors import _mix, _sample

        gait = self._gait[index][slot]
        travel = 0.0 if gait.xy is None else math.dist(xy, gait.xy)
        speed = travel / elapsed if elapsed > 0.0 else 0.0
        gait.xy = xy
        gait.idle_time += elapsed
        # Same fade as the deployed actor: 0.12 s exponential smoothing, steps shrink
        # below 0.5 m/s, and the walk clip fades out under 0.03 m/s so a standing
        # walker plays Idle instead of marching in place.
        alpha = 1.0 - math.exp(-elapsed / 0.12)
        stride = min(1.0, max(0.25, speed / 0.5))
        target = stride * min(1.0, speed / 0.03)
        gait.weight = min(1.0, max(0.0, gait.weight + alpha * (target - gait.weight)))
        gait.walk_position = (gait.walk_position + travel / stride) % rig.arc[-1]

        walk = _sample(rig.walk, rig.arc, gait.walk_position)
        idle = _sample(rig.idle, rig.idle_times, gait.idle_time % rig.idle_times[-1])
        pose = _mix(idle, walk, gait.weight)
        animation = self._anims[index][slot]
        animation.GetTranslationsAttr().Set(Vt.Vec3fArray(pose[0]))
        animation.GetRotationsAttr().Set(Vt.QuatfArray(pose[1]))
        animation.GetScalesAttr().Set(Vt.Vec3hArray([Gf.Vec3h(1.0)] * len(rig.joints)))
        self.pose_writes += 1

    def attach(self, env: object, crowd: VecPedestrians, *, root: str = "/World/envs") -> "VecCharacterStage":
        """Spawn the characters for ``env`` and return this stage wrapper."""
        self.dt = float(getattr(env, "step_dt", self.dt))
        self.spawn(env.sim.stage, crowd.routes, root=root)
        self.sync(crowd, force=True)
        return self

    def close(self) -> None:
        """Release the in-memory retarget stages held by every rig."""
        for rig in self.rigs:
            if rig is not None:
                try:
                    rig.close()
                except Exception:
                    pass
        self.rigs = []


@dataclass
class _Gait:
    """Per-walker animation state, advanced only by travelled distance.

    Attributes:
        xy: Last root position in room coordinates; ``None`` on the first write, so
            the spawn jump is not counted as travel.
        idle_time: Seconds of wall clock, driving the Idle clip.
        walk_position: Arc length [m] into the walk cycle.
        weight: Blend weight between Idle and Walk, exponentially smoothed.
    """

    xy: tuple[float, float] | None
    idle_time: float
    walk_position: float
    weight: float


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
