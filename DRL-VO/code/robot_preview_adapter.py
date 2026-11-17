"""EVT physical wheel-drive adapter for the existing NewMetro SimulationContext.

No SimulationApp or PhysicsScene is created here. The owning preview creates
the articulation through its InteractiveScene, steps PhysX, then calls update.
Root pose/velocity writes occur only during an explicit spawn/reset.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

ROOT = Path("/home/user/workspace/social_navigation_runtime_20261010")
ROBOT_USD = ROOT / "project/resources/releases/pedestrians_20261010/robot/EVT-EXP.usd"
EXPECTED_SHA256 = "d5e82de58510753ddaf6160a185d7c5c64333f0089eb40b3c98e6f9bd7837ab6"
WHEEL_NAMES = ("wheel_fl_joint", "wheel_fr_joint", "wheel_rl_joint", "wheel_rr_joint")
WHEEL_RADIUS_M = 0.07500000298023224
HALF_LENGTH_PLUS_HALF_WIDTH_M = 0.1575 + 0.1725


def _source_initial_joint_positions():
    from pxr import Usd, UsdPhysics

    if hashlib.sha256(ROBOT_USD.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("EVT robot source hash changed; stop and review asset provenance")
    stage = Usd.Stage.Open(str(ROBOT_USD))
    positions = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        angle = prim.GetAttribute("state:angular:physics:position").Get() or 0.0
        lower = prim.GetAttribute("physics:lowerLimit").Get()
        upper = prim.GetAttribute("physics:upperLimit").Get()
        if lower is not None:
            angle = max(float(lower), float(angle))
        if upper is not None:
            angle = min(float(upper), float(angle))
        positions[prim.GetName()] = math.radians(float(angle))
    return positions


def make_robot_cfg(spawn_xy, spawn_yaw=0.0, spawn_z=0.02):
    """Return the existing EVT USD config, with original geometry/masses/gains.

    UsdFileCfg applies explicit root translation and overrides the source's
    saved -1.242 m X translation. Quaternion order in this pinned SDK is XYZW.
    Contact sensors are enabled only on this robot in the preview overlay.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    return ArticulationCfg(
        prim_path="/World/NavigationRobot",
        articulation_root_prim_path="/base_link",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD), activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(float(spawn_xy[0]), float(spawn_xy[1]), float(spawn_z)),
            rot=(0.0, 0.0, math.sin(0.5 * spawn_yaw), math.cos(0.5 * spawn_yaw)),
            joint_pos=_source_initial_joint_positions(), joint_vel={".*": 0.0},
        ),
        actuators={
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=list(WHEEL_NAMES), stiffness=0.0, damping=None,
            ),
            "rollers": ImplicitActuatorCfg(
                joint_names_expr=["wheel_.*_roller_.*_joint"],
                stiffness=0.0, damping=0.0, effort_limit_sim=0.0,
            ),
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"], stiffness=None, damping=None,
            ),
        },
    )


def wheel_velocities(v, w, lateral=0.0):
    """Mecanum inverse kinematics in FL/FR/RL/RR order, rad/s.

    Geometry and radius come from the USD, rather than Nova Carter defaults.
    Expected signs still require post-physics forward/reverse/yaw validation.
    """
    k = HALF_LENGTH_PLUS_HALF_WIDTH_M
    return [(v - lateral - k * w) / WHEEL_RADIUS_M,
            (v + lateral + k * w) / WHEEL_RADIUS_M,
            (v + lateral - k * w) / WHEEL_RADIUS_M,
            (v - lateral + k * w) / WHEEL_RADIUS_M]


def _torch(value):
    import torch
    import warp
    if isinstance(value, torch.Tensor):
        return value
    view = getattr(value, "torch", None)
    if view is not None:
        return view
    return warp.to_torch(value)


class RobotPreviewAdapter:
    """Apply physical wheel targets and observe the same PhysX articulation."""

    def __init__(self, robot, sim, max_linear_m_s=None, max_angular_rad_s=None):
        import torch
        self.robot, self.sim = robot, sim
        self.max_linear_m_s = None if max_linear_m_s is None else float(max_linear_m_s)
        self.max_angular_rad_s = None if max_angular_rad_s is None else float(max_angular_rad_s)
        self.wheel_joint_ids = [list(robot.joint_names).index(n) for n in WHEEL_NAMES]
        self._targets = torch.zeros((1, 4), dtype=torch.float32, device=robot.device)
        self._command = [0.0, 0.0]
        self._initial_observation = None
        self.command_count = 0
        self.reset_count = 0

    def initialize(self, spawn_pose=None):
        """Explicit accepted spawn after Kit's first physics step; no stepping."""
        import torch
        if self.robot.num_instances != 1:
            raise RuntimeError("Preview expects exactly one navigation robot")
        cfg = self.robot.cfg.init_state
        if spawn_pose is None:
            position, rotation = cfg.pos, cfg.rot
        else:
            if len(spawn_pose) != 3 or not all(math.isfinite(float(v)) for v in spawn_pose):
                raise ValueError('Spawn override must be finite world X, Y, yaw')
            x, y, yaw = map(float, spawn_pose)
            position = (x, y, float(cfg.pos[2]))
            rotation = (0., 0., math.sin(yaw/2), math.cos(yaw/2))
        pose = torch.tensor([(*position, *rotation)], dtype=torch.float32, device=self.robot.device)
        zero_velocity = torch.zeros((1, 6), dtype=torch.float32, device=self.robot.device)
        joint_pos = _torch(self.robot.data.default_joint_pos).clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_root_pose_to_sim_index(root_pose=pose)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=zero_velocity)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.set_joint_position_target(joint_pos)
        self.stop()
        self.robot.reset()
        self.robot.update(0.0)
        self.reset_count += 1
        self._initial_observation = self.observe()
        return self._initial_observation

    def command(self, v, w):
        """Set SI body command; optional caps, wheel targets alone drive PhysX."""
        if not all(math.isfinite(float(x)) for x in (v, w)):
            raise ValueError("Non-finite robot command")
        v, w = float(v), float(w)
        if self.max_linear_m_s is not None:
            v = max(-self.max_linear_m_s, min(self.max_linear_m_s, v))
        if self.max_angular_rad_s is not None:
            w = max(-self.max_angular_rad_s, min(self.max_angular_rad_s, w))
        self._command[:] = [v, w]
        self._targets[0] = self._targets.new_tensor(wheel_velocities(v, w))
        self.robot.set_joint_velocity_target(self._targets, joint_ids=self.wheel_joint_ids)
        self.command_count += 1
        return list(self._command)

    def pre_physics_step(self):
        """Owner calls immediately before each existing 120 Hz sim step."""
        self.robot.set_joint_velocity_target(self._targets, joint_ids=self.wheel_joint_ids)
        self.robot.write_data_to_sim()

    def apply_current(self):
        """Alias for the owning preview's before-apply substep hook."""
        self.pre_physics_step()

    def update(self, dt):
        """Owner calls after PhysX advances; no additional physics tick."""
        self.robot.update(float(dt))

    def observe(self):
        pose = _torch(self.robot.data.root_link_pose_w)[0].detach().cpu().tolist()
        velocity = _torch(self.robot.data.root_link_vel_w)[0].detach().cpu().tolist()
        joint_velocity = _torch(self.robot.data.joint_vel)[0, self.wheel_joint_ids].detach().cpu().tolist()
        x, y, z, qx, qy, qz, qw = pose
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (qx * qx + qy * qy))))
        return {
            "xyz": [x, y, z], "quaternion_xyzw": [qx, qy, qz, qw],
            "linear_velocity_world": velocity[:3], "angular_velocity_world": velocity[3:],
            "wheel_actual_velocities_rad_s": joint_velocity,
            "root_position_m": [x, y, z], "root_quaternion_xyzw": [qx, qy, qz, qw],
            "yaw_rad": yaw, "tilt_rad": tilt,
            "physical_velocity_world": velocity,
            "physical_forward_velocity_m_s": velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw),
            "physical_yaw_velocity_rad_s": velocity[5],
            "requested_body_command_m_s_rad_s": list(self._command),
            "wheel_names": list(WHEEL_NAMES),
            "wheel_target_velocity_rad_s": self._targets[0].detach().cpu().tolist(),
            "wheel_actual_velocity_rad_s": joint_velocity,
            "root_pose_integrated_from_command": False,
            "root_velocity_set_from_command": False,
            "finite": all(math.isfinite(a) for a in [*pose, *velocity, *joint_velocity]),
            "initial_spawn_reset_count": self.reset_count,
        }

    def stop(self):
        self.command(0.0, 0.0)
        self.pre_physics_step()

    def provenance(self):
        return {
            "robot_usd": str(ROBOT_USD), "sha256": EXPECTED_SHA256,
            "drive_type": "mecanum_real_roller_joints_and_PhysX_contacts",
            "wheel_radius_m": WHEEL_RADIUS_M,
            "half_length_plus_half_width_m": HALF_LENGTH_PLUS_HALF_WIDTH_M,
            "source_physics_geometry_masses_preserved": True,
            "wheel_gains": "original USD damping/maxForce; zero stiffness velocity drive",
            "passive_roller_actuators": "zero stiffness, damping and drive effort",
            "command_limits": [self.max_linear_m_s, self.max_angular_rad_s],
            "uses_owner_simulation_context": True, "training": False,
            "source_control_reference": "a_pipeline robot_controller.py formula only; root control discarded",
            "validation_boundary": "physical movement/contact/stop pending runtime measurements",
        }
