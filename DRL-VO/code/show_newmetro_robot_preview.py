"""NewMetro physical preview plus isolated SFM/AMP DRLVO training mode."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
p.add_argument("--drlvo-smoke", action="store_true")
p.add_argument("--optimized-preview", action="store_true",
               help="manual preview with admitted CPU route candidates, TRACE fallback and CUDA AMP")
p.add_argument("--sfm-preview", action="store_true",
               help="Gazebo-style social-force trajectory targets with physical AMP execution")
p.add_argument("--isolated-demo",action="store_true")
p.add_argument("--demo-headless",action="store_true")
p.add_argument("--isolated-training", action="store_true",
               help="isolated four-person SFM/AMP and physical DRLVO policy owner")
p.add_argument("--continuous-trace", action="store_true",
               help="one asynchronous real TRACE worker; current-state re-admission and admitted bridge")
p.add_argument("--manual-robot", action="store_true", help="start robot stopped; skip automatic motion check")
p.add_argument("--calibration-fixture", action="store_true",
               help="bounded stationary PhysX dual-LiDAR calibration; no training")
p.add_argument("--open-doors", action="store_true", help="fixed-open session poses and current PhysX navigation grid")
p.add_argument("--drlvo-long-steps", type=int, default=0,
               help="after the in-process smoke gate, run this many DRL-VO PPO transitions")
args = p.parse_args()
owned_policy = args.isolated_training or args.isolated_demo
if args.isolated_training and args.isolated_demo:p.error("Choose demo or training")
if args.demo_headless and not args.isolated_demo:p.error("demo-headless requires demo")
if args.calibration_fixture and (not args.manual_robot or args.drlvo_smoke or args.drlvo_long_steps or owned_policy):
    p.error('--calibration-fixture requires a manual robot and excludes training')
if args.continuous_trace and not args.optimized_preview:
    p.error('--continuous-trace requires --optimized-preview')
if args.sfm_preview and (args.optimized_preview or args.continuous_trace or args.drlvo_smoke or args.drlvo_long_steps):
    p.error('--sfm-preview is an independent manual preview mode')
if owned_policy and not (args.sfm_preview and args.manual_robot and args.open_doors):
    p.error("--isolated-training requires SFM, manual-robot and fixed-open doors")
if args.drlvo_long_steps < 0:
    p.error("--drlvo-long-steps must be non-negative")
if args.optimized_preview and (args.drlvo_smoke or args.drlvo_long_steps):
    p.error("optimized preview is separate from DRL-VO training")
if args.open_doors and (args.drlvo_smoke or args.drlvo_long_steps):
    p.error('DRL-VO repair is restricted to the authored closed scene')
OUT = args.output.resolve()
source = json.loads((OUT / "SOURCE.json").read_text())
closed_task = None
if args.drlvo_smoke or args.drlvo_long_steps:
    # Admission before Kit creation; no unreviewed endpoint fallback.
    from drlvo_closed_map import load_task
    if 'drlvo_task' not in source:
        p.error('Training requires an admitted closed DRLVO_TASK.json')
    closed_task = load_task(source['drlvo_task']['path'], expected_sha256=source['drlvo_task']['sha256'])
    if source['robot_spawn_xy'] != closed_task['start_xy'] or source['robot_goal_xy'] != closed_task['goal_xy']:
        p.error('Robot spawn/goal differs from admitted closed task')
sys.argv = [sys.argv[0]]
# IsaacLab's pinned ProxyArray API requires the installed Warp 1.15 build;
# retain it before Kit's older omni.warp.core Python package is placed first.
import warp
from isaacsim import SimulationApp
app = SimulationApp({"headless": bool(args.isolated_training or args.demo_headless or args.drlvo_smoke or args.drlvo_long_steps), "renderer": "RaytracedLighting", "multi_gpu": False,
                     "width": 1440, "height": 900, "window_width": 1700, "window_height": 1050})
import torch
import inspect
import omni.ui as ui
import omni.timeline
import omni.usd
import carb.settings
from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.amp import PacerAMPController
from CrowdSim.humanoid.amp_world import AMPWorld
from CrowdSim.tools.smpl_mesh_visualizer import ProtoMotionsHumanMeshAdapter
from isaaclab.utils.warp.proxy_array import ProxyArray

# Process-local interop only: pass the same underlying arrays to Warp.
# This SDK snapshot passes ProxyArray directly at several kernel boundaries;
# the installed Warp rejects its interface for transform-typed arrays.
original_launch = warp.launch
launch_signature = inspect.signature(original_launch)
def proxy_launch(*args, **kwargs):
    bound = launch_signature.bind_partial(*args, **kwargs)
    for name in ("inputs", "outputs", "adj_inputs", "adj_outputs"):
        values = bound.arguments.get(name)
        if values is not None:
            bound.arguments[name] = [v.warp if isinstance(v, ProxyArray) else v for v in values]
    return original_launch(*bound.args, **bound.kwargs)
warp.launch = proxy_launch
original_to_torch = warp.to_torch
def proxy_to_torch(value, *args, **kwargs):
    return original_to_torch(value.warp if isinstance(value, ProxyArray) else value, *args, **kwargs)
warp.to_torch = proxy_to_torch
import omni.physics.tensors.api as tensor_api
def cpu_index_setter(method):
    signature = inspect.signature(method)
    def adapted(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        # This Sim 6 tensor backend accepts host writeback buffers, while
        # the newer Lab snapshot supplies device buffers. Values are unchanged.
        for name, value in list(bound.arguments.items()):
            if isinstance(value, ProxyArray):
                value = value.warp
            if isinstance(value, warp.array):
                bound.arguments[name] = value if value.device.is_cpu else warp.clone(value, device="cpu")
            elif isinstance(value, torch.Tensor):
                bound.arguments[name] = warp.from_torch(value.detach().cpu().contiguous())
        return method(*bound.args, **bound.kwargs)
    return adapted
cpu_index_methods = []
for name, method in inspect.getmembers(tensor_api.ArticulationView, inspect.isfunction):
    if name.startswith("set_") and "indices" in inspect.signature(method).parameters:
        setattr(tensor_api.ArticulationView, name, cpu_index_setter(method))
        cpu_index_methods.append(name)

report = dict(source, status="STARTING", pid=os.getpid(), amp=True, actual_controls=0, falls=0,
              training=False, camera_control="user", window_open=True,
              warp_version=warp.__version__, process_local_proxy_array_interop=True, SDK_source_modified=False)
report["legacy_physx_cpu_index_adapter"] = cpu_index_methods
report["legacy_physx_host_writeback_buffers"] = True
report["inference_device"] = "cpu"
report["physics_device"] = "cpu"
report["planner_device"] = "cuda:0"
report["pedestrian_planner_mode"] = "SFM to AMP" if args.sfm_preview else "TRACE to AMP"
report["terminal_teleop_enabled"] = not owned_policy
report["terminal_teleop_style"] = "teleop_twist_keyboard_continuous_velocity"
report["fall_events"] = []
report["route_vertices_reached_per_actor"] = [0] * source["actor_count"]
def save():
    tmp = OUT / "REPORT.tmp"
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(OUT / "REPORT.json")
def pump_gui():
    # SimulationCfg has no KitVisualizer by default in this SDK snapshot.
    # Follow KitVisualizer.step(): refresh Kit without an extra physics tick.
    settings = carb.settings.get_settings()
    play_flag = settings.get("/app/player/playSimulations")
    settings.set_bool("/app/player/playSimulations", False)
    try:
        app.update()
    finally:
        settings.set_bool("/app/player/playSimulations", True if play_flag is None else bool(play_flag))
    report["gui_update_count"] = report.get("gui_update_count", 0) + 1
    stage = omni.usd.get_context().get_stage()
    report["gui_actor_count"] = sum(bool(stage and stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")) for i in range(source["actor_count"]))
    report["gui_stage_has_robot"] = report["gui_actor_count"] == source["actor_count"]
    report["gui_stage_has_scene"] = bool(stage and stage.GetPrimAtPath("/NewMetro"))
    report["gui_stage_has_navigation_robot"] = bool(stage and stage.GetPrimAtPath("/World/NavigationRobot/base_link"))
world = None
robot = None
robot_file = None
file = None
error_code = 0
save()
try:
    import numpy as np
    from dataclasses import fields, replace
    from robot_preview_adapter import make_robot_cfg, RobotPreviewAdapter
    from lidar_preview_adapter import LidarPreviewAdapter
    from protomotions.simulator.isaaclab.utils import usd_only_scene as source_usd_scene
    if source['actor_count'] == 2:
        from usd_only_scene_two_preview import USDOnlyInteractiveScene
        source_usd_scene.USDOnlyInteractiveScene = USDOnlyInteractiveScene
        report['two_pedestrian_USD_adapter'] = dict(num_envs_required=2,
            SDK_source_modified=False, official_USD_copy_retained=True)
    else:
        USDOnlyInteractiveScene = source_usd_scene.USDOnlyInteractiveScene
    def add_two_owned_articulations(self):
        # The project class admits one vectorized humanoid asset. This preview
        # adds exactly one global robot, and retains its USD/direct-PhysX path.
        from isaaclab.scene import InteractiveScene
        from isaaclab.cloner.replicate_session import REPLICATION_QUEUE
        from isaaclab_physx.cloner.replicate import PhysxReplicateContext
        expected = {id(self.cfg.robot), id(self.cfg.navigation_robot)}
        owned = {id(cfg) for cfg in self._collect_asset_cfgs()}
        InteractiveScene._add_entities_from_cfg(self)
        removed = [(cfg, backend) for cfg, backend in REPLICATION_QUEUE
            if id(cfg) in owned and backend is PhysxReplicateContext]
        if len(removed) != 2 or {id(cfg) for cfg, _ in removed} != expected:
            raise RuntimeError("Require the exact humanoid and navigation robot queue owners")
        kept = [entry for entry in REPLICATION_QUEUE
            if not (id(entry[0]) in owned and entry[1] is PhysxReplicateContext)]
        if not kept:
            raise RuntimeError("Official USD replication must remain queued")
        REPLICATION_QUEUE[:] = kept
        self.physics_replication_report.update(owned_Physx_queue_entries_removed=2,
            preserved_backend_entries=len(kept), global_navigation_robot="/World/NavigationRobot",
            preview_process_local_two_asset_adaptation=True)
    USDOnlyInteractiveScene._add_entities_from_cfg = add_two_owned_articulations
    report["two_articulation_asset_queue_process_adapter"] = True
    from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
    from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
    from CrowdSim.humanoid.diffusion.planner import DiffusionPlanner
    from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidanceConfig, TrajectoryAdmissionError
    from CrowdSim.humanoid.runtime.trace_pace import TracePaceClosedLoop
    sys.path.insert(0, "/home/user/workspace/restroom_isaac_preview_20261010_031340/tools")
    from ceiling_visibility import CeilingVisibility
    torch.manual_seed(7)
    torch.set_num_threads(1 if args.continuous_trace else int(os.environ.get("NEWMETRO_TORCH_THREADS", "4")))
    scene = json.loads((OUT / "scene_config.json").read_text())
    profile = json.loads((OUT / "profile.json").read_text())
    count = source["actor_count"]
    print("NEWMETRO_ADMITTING_EXISTING_MAP", flush=True)
    snap = scene["probe_script_snapshot"]
    provider = DynBenchSceneProvider.from_preflight(scene["preflight"]["path"],
        expected_sha256=scene["preflight"]["sha256"], probe_script_snapshot=snap["path"])
    provider.admit_native_map(scene["native_map"]["path"], expected_sha256=scene["native_map"]["sha256"])
    provider.configure_policy_height_feature(scene["amp_policy_height_mode"])
    native_options = {k: scene[k] for k in ["clearance_m", "arrival_radius_m", "goal_lookahead_m",
        "cyclic_route_lookahead", "start_recovery_radius_m", "start_recovery_clearance_m",
        "route_clearance_weight", "route_comfort_clearance_m"] if k in scene}
    if owned_policy:
        # SFM consumes the measured signed grid as obstacle force only.
        # The robot gets a separate, full-footprint planner below.
        native_options.update(clearance_m=0., start_recovery_radius_m=0.,
                              start_recovery_clearance_m=0., cyclic_route_lookahead=False)
        report["pedestrian_astar_calls"] = 0
        report["pedestrian_route_gate"] = "disabled_for_sfm_force_only"
    native = NativeGridScene.from_provider(provider,
        origins_xy=scene["origins_xy"], routes_xy=scene["routes_xy"], device="cpu",
        **native_options)
    if closed_task:
        load_task(source['drlvo_task']['path'], expected_sha256=source['drlvo_task']['sha256'], native=native)
        report['closed_task_static_admission'] = closed_task['static_certificate']
        report['doors'] = dict(mode='authored_closed', source_poses_preserved=True, opened_by_this_run=False)
    original_scene_cfg = provider.prepare_scene_cfg
    if args.open_doors:
        original_spawn = provider._spawn
        def spawn_with_open_doors(*spawn_args, **spawn_kwargs):
            prim = original_spawn(*spawn_args, **spawn_kwargs)
            from CrowdSim.humanoid.runtime.dynbench_scene import _active_spawn_stage
            from door_pose_preview import open_doors
            open_doors(_active_spawn_stage(), report)
            return prim
        provider._spawn = spawn_with_open_doors
    def scene_with_robot(cfg):
        result = original_scene_cfg(cfg)
        result.navigation_robot = make_robot_cfg(source["robot_spawn_xy"],
            spawn_yaw=source.get("robot_spawn_yaw_rad", 0.), spawn_z=.02)
        original_spawn = result.navigation_robot.spawn.func
        def spawn_without_gamepad(*args, **kwargs):
            prim = original_spawn(*args, **kwargs)
            import isaaclab.sim as sim_utils
            from pxr import Usd
            spawn_stage = sim_utils.get_current_stage()
            graph = spawn_stage.GetPrimAtPath("/World/NavigationRobot/Gamepad")
            if not graph:
                raise RuntimeError("Expected source robot Gamepad graph missing")
            with Usd.EditContext(spawn_stage, spawn_stage.GetSessionLayer()):
                graph.SetActive(False)
            report["robot_Gamepad_disabled_only_in_session"] = True
            return prim
        result.navigation_robot.spawn.func = spawn_without_gamepad
        return result
    provider.prepare_scene_cfg = scene_with_robot
    # Add only this robot's colliders to the provider's explicit dynamic set.
    # Room static paths and all humanoid identities keep their original contracts.
    original_bind = provider.bind_native_query
    def bind_with_robot(query, *, physics_step_count, dynamic_robot_roots=()):
        return original_bind(query, physics_step_count=physics_step_count,
            dynamic_robot_roots=tuple(dynamic_robot_roots) + ("/World/NavigationRobot",))
    provider.bind_native_query = bind_with_robot
    corpus = AMASSCorpus(OUT / "preview_manifest.json", split="validation", device="cpu", seed=7,
                         dof_velocity_frame="child_joint")
    initialization_rng = corpus.generator.get_state()
    print("NEWMETRO_CREATING_SHARED_PHYSICAL_WORLD", flush=True)
    world = AMPWorld(app, corpus, num_envs=count, device="cpu", shared_world=True,
        native_navigation_scene=native, episode_steps=1000000000, contact_reporting=False,
        **{k: scene[k] for k in ["align_spawn_heading_to_route", "initial_reference_max_speed_m_s",
            "initialize_at_rest", "initial_reference_arm_cone_deg", "scene_physics_replication"] if k in scene})
    robot = RobotPreviewAdapter(world.simulator._scene["navigation_robot"], world.simulator._sim,
                                max_linear_m_s=2.0 if args.optimized_preview else None)
    original_apply = world.simulator._apply_control
    def apply_all_controls():
        original_apply()
        robot.pre_physics_step()
    world.simulator._apply_control = apply_all_controls
    world.simulator._sim.step(render=False)
    world.simulator._scene.update(dt=world.dt / 4)
    corpus.generator.set_state(initialization_rng)
    robot.initialize()
    world.reset(torch.arange(count, device=world.device))
    world.simulator._scene.write_data_to_sim()
    world.simulator._sim.forward()
    world.simulator._scene.update(dt=world.dt / 4)
    report["initialization_pre_spawn_physics_steps"] = 1
    report["initial_backend_root_pose"] = warp.to_torch(world.simulator._robot.root_view.get_root_transforms()).tolist()
    report["initial_observed_root_pose"] = world.state().root_pos.tolist()
    report["physics"] = world.physics_manifest()
    report["navigation_robot"] = robot.provenance()
    import omni.physx
    if closed_task:
        from drlvo_runtime_geometry import check_route_geometry
        report['drlvo_runtime_geometry'] = check_route_geometry(native, closed_task,
            omni.physx.get_physx_scene_query_interface(), OUT)
        if closed_task.get('schema') == 'drlvo.closed_observed_heading_task.v2':
            from drlvo_observed_task import navigation_for_task
            robot_navigation = navigation_for_task(closed_task, native)
    if args.open_doors:
        from door_pose_preview import refresh_grid
        refresh_grid(native, omni.physx.get_physx_scene_query_interface(), OUT, report, pump_gui,
                     sfm_force_only=owned_policy)
    training_mode = bool(owned_policy or args.drlvo_smoke or args.drlvo_long_steps)
    if args.calibration_fixture:
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        kept = set(provider.floor_paths)
        disabled = []
        for path in provider.static_paths:
            if path in kept:
                continue
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                api = UsdPhysics.CollisionAPI(prim)
                attr = api.GetCollisionEnabledAttr()
                if attr and attr.IsValid():
                    attr.Set(False)
                    disabled.append(str(path))
        robot_pose = robot.observe()
        from pxr import Gf
        import isaaclab.sim as sim_utils
        target_angle = math.pi / 2
        target_distance = 2.5
        rx, ry, rz = robot_pose['root_position_m']
        yaw = float(robot_pose['yaw_rad'])
        tx = rx + target_distance * math.cos(yaw + target_angle)
        ty = ry + target_distance * math.sin(yaw + target_angle)
        obstacle = sim_utils.CuboidCfg(size=(.1,.1,.25),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.))
        sim_utils.spawn_cuboid('/World/CalibrationObstacle', obstacle,
            translation=(tx,ty,rz+.208))
        calibration_floor = sim_utils.CuboidCfg(size=(6.,6.,.2),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.))
        sim_utils.spawn_cuboid('/World/CalibrationFloor', calibration_floor,
            translation=(rx,ry,rz-.12))
        report['lidar_calibration_fixture'] = dict(
            status='FIXTURE_READY', training=False, robot_motion='manual_zero',
            static_collision_paths_disabled=disabled, floor_paths_retained=sorted(kept),
            external_obstacle_path='/World/CalibrationObstacle',
            calibration_floor_path='/World/CalibrationFloor',
            external_obstacle_beam_scan_01=1500,
            external_obstacle_target_distance_m=target_distance)
    lidar = LidarPreviewAdapter(omni.physx.get_physx_scene_query_interface(),
        pedestrian_roots=tuple(f"/World/envs/env_{i}/Robot" for i in range(count)),
        capture_self_echoes=bool(args.manual_robot),
        **({'range_min_m': .1, 'range_max_m': 8.}
           if training_mode or args.calibration_fixture else {}))
    report["robot_initial_observation"] = robot.observe()
    report["robot_effective_wheel_damping"] = warp.to_torch(robot.robot.root_view.get_dof_dampings())[:, robot.wheel_joint_ids].tolist()
    report["robot_effective_wheel_max_force"] = warp.to_torch(robot.robot.root_view.get_dof_max_forces())[:, robot.wheel_joint_ids].tolist()
    report["robot_movement_status"] = "SHORT_SMOKE_PENDING"
    report["lidar_scan_count"] = 0
    report["lidar_max_pedestrian_hit_count"] = 0
    report["lidar_backend"] = "actual_current_PhysX_scene_query"
    report["lidar_installed_relative_to"] = "/World/NavigationRobot/base_link"
    report["lidar_self_occlusion_mode"] = "carrying_robot_excluded; AMP colliders retained"
    report["robot_oracle_pedestrian_state_for_training"] = "PREPARATION_ONLY_NOT_NETWORK_PERCEPTION"
    report["robot_contact_safety"] = "NOT_VALIDATED"
    robot_file = (OUT / "robot_motion.csv").open("w", newline="")
    robot_writer = csv.writer(robot_file)
    robot_writer.writerow(["control", "sim_time", "segment", "cmd_v", "cmd_w", "root_x", "root_y", "root_z", "yaw", "tilt", "actual_forward_v", "actual_yaw_w", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"])
    robot_rows = []
    smoke_segments = [("settle",30,0.,0.), ("forward",60,.18,0.), ("stop_after_forward",30,0.,0.),
        ("backward",60,-.18,0.), ("left",60,0.,.4), ("stop_after_left",30,0.,0.),
        ("right",60,0.,-.4), ("final_stop",45,0.,0.)]
    robot_auto = not args.manual_robot
    robot_manual_until = 0
    lidar_rows = []
    lidar_last = None
    report["native_scene_live_observation"] = world.native_scene_live_observation
    report["controller_provenance_verified"] = False
    controller = PacerAMPController.from_checkpoint(source["checkpoint"], world.lower, world.upper,
        body_names=world.mapping.body_names, dof_names=world.mapping.dof_names,
        device="cpu", expected_sha256=source["checkpoint_sha256"])
    original_manifest = json.loads(Path(source["original_manifest"]).read_text())
    assert controller.local_finetune["manifest_sha256"] == source["original_manifest_sha256"]
    assert set(controller.local_finetune["expert_unique_motion_ids"]) == {r["motion_id"] for r in original_manifest["splits"]["train"]["clips"]}
    report["controller_provenance_verified"] = True
    os.environ["CROWDSIM_SMPL_MODEL_DIR"] = "/home/user/workspace/social_navigation_runtime_20261010/project/data/smpl"
    os.environ["CROWDSIM_SMPL_SHAPE_STD"] = "0"
    os.environ["CROWDSIM_HIDE_HUMANOID"] = "1"
    mesh = ProtoMotionsHumanMeshAdapter.from_simulator(world.simulator)
    mesh.create()
    mesh.update()
    report["physical_skin_overlay"] = True
    stage = omni.usd.get_context().get_stage()
    ceiling = CeilingVisibility(stage)
    hidden = ui.SimpleBoolModel(True)
    def apply_ceiling(model):
        report["ceiling_hidden"] = model.as_bool
        report["ceiling_geometry_count"] = len(ceiling.set_hidden(model.as_bool))
    hidden.add_value_changed_fn(apply_ceiling)
    apply_ceiling(hidden)
    # The analytical reference floor supplies collisions; authored tiles supply visuals.
    from pxr import Usd, UsdGeom
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Imageable(stage.GetPrimAtPath("/__PaperMapProbeReferenceGround")).MakeInvisible()
    report["reference_floor_render_hidden"] = True
    lidar_points = UsdGeom.Points.Define(stage, "/World/LiveLidarHits")
    lidar_points.CreateDisplayColorAttr([(0.1, 1., .25)])
    lidar_points.CreateWidthsAttr([])
    show_lidar = ui.SimpleBoolModel(True)
    def lidar_visibility(model):
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            lidar_points.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited if model.as_bool else UsdGeom.Tokens.invisible)
    show_lidar.add_value_changed_fn(lidar_visibility)
    pump_gui()
    print("NEWMETRO_LOADING_EXISTING_TRACE" if not args.sfm_preview else "NEWMETRO_LOADING_SFM_AMP", flush=True)
    if args.sfm_preview:
        from sfm_amp_preview import SfmAmpPreview
        report.update(planner_device="cpu", actual_diffusion_steps=0,
                      sfm_source="a_pipeline Gazebo desired/social/robot/obstacle force defaults",
                      sfm_route_source="original cyclic NewMetro route vertices; no A-star path query",
                      sfm_contact_safety="NOT_VALIDATED")
        def new_runtime():
            return SfmAmpPreview(controller, scene["routes_xy"],
                scene["arrival_radius_m"], native, robot, report)
    else:
        resources = source["resources"]
        planner = DiffusionPlanner(resources["diffusion_checkpoint"]["path"],
            resources["diffusion_config"]["path"], device="cuda:0")
        original_context = planner.prepare_context
        def aligned_context(states, *args, **kwargs):
            previous_device = planner.device
            planner.device = states.device
            try:
                return original_context(states, *args, **kwargs)
            finally:
                planner.device = previous_device
        planner.prepare_context = aligned_context
        original_plan = planner.plan
        def host_plan(**kwargs):
            transferred = {k: v.to(planner.device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            plan = original_plan(**transferred)
            return replace(plan, **{f.name: getattr(plan, f.name).cpu() for f in fields(plan)
                if isinstance(getattr(plan, f.name), torch.Tensor)})
        planner.plan = host_plan
        if args.drlvo_smoke or args.drlvo_long_steps:
            from drlvo_trace_retry import install_admission_retry
            install_admission_retry(planner, OUT, report)
        report["planner_host_transfer_only"] = True
        report["runtime_guard_context_device"] = "cpu"
        report["actual_diffusion_steps"] = planner.model.n_timesteps
        runtime_keys = ["replan_controls", "num_samples", "allow_safety_braking", "stationary_route_heading",
            "braking_agent_buffer_m", "mixed_safety_brakes", "guard_horizon_s", "guard_tracking_margin_m",
            "map_guard_clearance_m", "pedestrian_extent_m", "actual_body_map_guard", "body_guard_horizon_s",
            "body_guard_gap_m", "persistent_braking_target", "settle_braking_target", "amp_sample_start_s"]
        attach_runtime = lambda runtime: runtime
        if args.drlvo_smoke or args.drlvo_long_steps:
            from drlvo_robot_temporal import install_strict_robot_admission
            attach_runtime = install_strict_robot_admission(planner, robot, report)
        def new_runtime():
            return attach_runtime(TracePaceClosedLoop(planner, controller, PedestrianGuidanceConfig(**profile["guidance"]),
                seed=7, **{k: profile[k] for k in runtime_keys if k in profile}))
    if not args.sfm_preview:
        from robot_pedestrian_neighbor import install_robot_neighbor
        install_robot_neighbor(robot, report, radius=.6,
            temporal_admission=bool(args.optimized_preview or args.drlvo_smoke or args.drlvo_long_steps))
    if not args.sfm_preview:
        from pedestrian_robot_stop_guard import install_stop_guard
        install_stop_guard(controller, robot, report,
            strict_planned_escape=bool(args.drlvo_smoke or args.drlvo_long_steps))
    runtime = new_runtime()
    if owned_policy:
        drlvo_root = "/home/user/navigation_project/a_pipeline/methods/experiments/drl_vo_ros2_offline"
        sys.path.insert(0, str(Path(drlvo_root).parents[2]))
        sys.path.insert(0, drlvo_root)
    if args.drlvo_smoke or args.drlvo_long_steps:
        long_steps = int(args.drlvo_long_steps)
        report.update(status="DRLVO_ISAAC_SMOKE_STARTING", training=False,
            long_training=False, drlvo_smoke=True,
            source_scene="NewMetro_static_v2_restroom_source",
            scene_source_path=source.get("scene_source", {}).get("path"),
            pedestrian_source="oracle",
            training_scope="NewMetro restroom source + four AMP people + physical robot + dual LiDAR; DRL-VO smoke/long mode")
        m = sys.modules[__name__]
        m.runtime = runtime
        # Goal comes only from the admitted task; no endpoint fallback in training.
        m.drlvo_goal = np.asarray(source.get("robot_goal_xy", [5.45, -1.85]), dtype=np.float32)
        m.drlvo_route_indices = torch.zeros(count, dtype=torch.long, device=world.device)
        # Reuse the protected historical DRL-VO implementation verbatim.
        drlvo_root = "/home/user/navigation_project/a_pipeline/methods/experiments/drl_vo_ros2_offline"
        sys.path.insert(0, str(Path(drlvo_root).parents[2]))
        sys.path.insert(0, drlvo_root)
        from drlvo_isaac_env import IsaacDrlVoEnv
        from online_ppo import RolloutBuffer, load_online_policy
        from drlvo_gazebo_contract import CheckedPPO, verified_model
        policy_path = verified_model()
        policy, policy_items = load_online_policy(policy_path)
        smoke_ppo = CheckedPPO(policy, torch.device('cpu'))
        smoke_env = IsaacDrlVoEnv(m, max_episode_steps=512)
        observation, reset_info = smoke_env.reset()
        buffer = RolloutBuffer()
        transitions = []
        for _ in range(256):
            raw, action, logp, value = smoke_ppo.select_action(observation)
            next_observation, reward, terminated, truncated, info = smoke_env.step(action)
            buffer.add(observation, raw, reward, terminated or truncated, logp, value)
            transitions.append(dict(reward=reward, terminated=terminated, truncated=truncated,
                done_reason=info.get("done_reason"), actual_forward_m_s=info["actual_forward_m_s"],
                wheel_actual_velocity_rad_s=info['wheel_actual_velocity_rad_s'],
                local_goal_base_xy=info['local_goal_base_xy'], local_goal_world_xy=info['local_goal_world_xy'],
                robot_xy=info['robot_xy'], global_path_length_m=info['global_path_length_m'],
                minimum_scan_m=info["minimum_scan_m"]))
            observation = next_observation
            if terminated or truncated:
                observation, _ = smoke_env.reset()
        ppo_metrics = smoke_ppo.update(buffer, bootstrap_value=0. if terminated or truncated else smoke_ppo.value(observation))
        # Cover an explicit reset even if no natural episode ended in 256 steps.
        smoke_env.reset()
        wheel_motion = max(abs(value) for row in transitions for value in row['wheel_actual_velocity_rad_s'])
        if smoke_env.max_move < .02 or wheel_motion < .1:
            raise RuntimeError('Real wheel-drive/motion smoke gate failed')
        smoke_checkpoint = OUT / "drlvo_policy_smoke.pt"
        torch.save(smoke_ppo.policy.state_dict(), smoke_checkpoint)
        report.update(status="DRLVO_ISAAC_SMOKE_PASS", isaac_runtime=True,
            policy_checkpoint=str(policy_path), policy_items=policy_items,
            observation_size=int(observation.shape[0]), reset_info=reset_info,
            real_scan_history_frames=len(smoke_env.history.frames), transitions=transitions,
            ppo_update=ppo_metrics, smoke_checkpoint=str(smoke_checkpoint),
            optimizer_state_restored=False,
            collision_metric="Gazebo reward contract; minimum valid dual scan; not PhysX contact",
            action_period_s=.1, lidar_range_m=[.1,8.], local_goal_source='admitted A* path',
            rollout_action='raw_action; environment executes clipped_action',
            timeout_bootstrap='zero on timeout, preserving successful Gazebo reward/done learning semantics',
            long_training=False, training=False, smoke_update=True)
        report['acceptance_checks'] = dict(real_dual_lidar='PASS_NATIVE_QUERIES',
            action_period='PASS_3_CONTROL_TICKS', actor_update='PASS_CHECKED_PPO',
            KL_and_clip_fraction='PASS_CHECKED_PPO', reset='PASS_EXPLICIT_POST_ROLLOUT_RESET',
            measured_wheel_motion='PASS_BOUNDED', robot_displacement_m=smoke_env.max_move,
            max_actual_wheel_velocity_rad_s=wheel_motion, local_goal='PASS_RUNTIME_ASTAR_OBSERVATIONS',
            environment_fault_handling='NOT_FAULT_INJECTED_IN_ISAAC', PhysX_contact='NOT_VALIDATED',
            navigation_goal_reach='NOT_REQUIRED_FOR_256_TRANSITION_SMOKE')
        report['scan_evidence'] = smoke_env.scan_evidence
        report['reset_evidence'] = smoke_env.reset_evidence
        (OUT / "DRLVO_ISAAC_SMOKE.json").write_text(json.dumps(report, indent=2) + "\n")
        if long_steps <= 0:
            smoke_env.log.close()
            smoke_env.tracking_log.close()
            raise SystemExit(0)

        # Match Gazebo online PPO for the actual long phase.
        smoke_env.limit = 512
        report.update(status="DRLVO_LONG_TRAINING_RUNNING", training=True,
            long_training=True, long_training_authorized=True,
            long_target_steps=long_steps, smoke_max_episode_steps=512,
            long_max_episode_steps=512, long_rollout_steps=256,
            long_reward_contract="gazebo_compute_training_reward",
            long_source_checkpoint=str(smoke_checkpoint), long_optimizer_state_restored=False)
        ppo = CheckedPPO(policy, torch.device('cpu'))
        long_dir = OUT / "long_checkpoints"
        long_dir.mkdir(parents=True, exist_ok=True)
        long_buffer = RolloutBuffer()
        long_observation, _ = smoke_env.reset()
        long_transitions = 0
        long_updates = 0
        long_returns = []
        last_episode_return = 0.0
        for _ in range(long_steps):
            raw, action, logp, value = ppo.select_action(long_observation)
            next_observation, reward, terminated, truncated, info = smoke_env.step(action)
            done = bool(terminated or truncated)
            long_buffer.add(long_observation, raw, reward, done, logp, value)
            last_episode_return += float(reward)
            long_transitions += 1
            long_observation = next_observation
            if done:
                long_returns.append(last_episode_return)
                last_episode_return = 0.0
                long_observation, _ = smoke_env.reset()
            if len(long_buffer) >= 256:
                update_metrics = ppo.update(long_buffer,
                    bootstrap_value=0. if done else ppo.value(long_observation))
                long_updates += 1
                torch.save({"policy_state_dict": ppo.policy.state_dict(),
                            "optimizer_state_dict": ppo.optimizer.state_dict(),
                            "global_step": long_transitions,
                            "ppo_update": update_metrics,
                            "source_checkpoint": str(policy_path)},
                           long_dir / f"checkpoint_{long_transitions}.pt")
                report.update(long_transitions=long_transitions,
                    long_updates=long_updates, last_ppo_update=update_metrics,
                    episode_returns=long_returns[-100:])
                save()
        if len(long_buffer) >= 2:
            update_metrics = ppo.update(long_buffer, bootstrap_value=ppo.value(long_observation))
            long_updates += 1
        final_checkpoint = long_dir / "latest.pt"
        torch.save({"policy_state_dict": ppo.policy.state_dict(),
                    "optimizer_state_dict": ppo.optimizer.state_dict(),
                    "global_step": long_transitions, "ppo_updates": long_updates,
                    "last_ppo_update": update_metrics if long_updates else None,
                    "source_checkpoint": str(policy_path)}, final_checkpoint)
        smoke_env.log.close()
        report.update(status="DRLVO_LONG_TRAINING_COMPLETE", long_transitions=long_transitions,
            long_updates=long_updates, final_checkpoint=str(final_checkpoint),
            episode_returns=long_returns[-100:], training=True, long_training=True)
        (OUT / "DRLVO_LONG_TRAINING.json").write_text(json.dumps(report, indent=2) + "\n")
        save()
        raise SystemExit(0)
    file = (OUT / "motion.csv").open("w", newline="")
    writer = csv.writer(file)
    writer.writerow(["replay", "control", "sim_time", "actor", "root_x", "root_y", "root_z", "speed_xy", "joint_delta", "fallen", "route_index"])
    state = world.state()
    start = state.root_pos.clone()
    first_joints = state.dof_pos.clone()
    max_displacements = torch.zeros(count)
    route_indices = torch.zeros(count, device=world.device, dtype=torch.long)
    control, active, replay_requested = 0, True, False
    calibration_mask = {name: set() for name in ("scan_01", "scan_02")}
    calibration_self_paths = {name: {} for name in calibration_mask}
    calibration_min_external = math.inf
    calibration_target_rows = []
    calibration_start_pose = None
    planner_blocked = False
    manual_planner_wait = None
    if args.optimized_preview and args.manual_robot:
        from manual_preview_planner_wait import ManualPreviewPlannerWait
        def record_planner_wait(event):
            with (OUT / 'manual_planner_wait.jsonl').open('a') as stream:
                stream.write(json.dumps(event) + '\n')
        manual_planner_wait = ManualPreviewPlannerWait(
            TrajectoryAdmissionError, report, record_planner_wait)
        report['manual_planner_failure_policy'] = 'AMP waiting and periodic replan; robot teleop remains independent'
    terminal_sequence = None
    terminal_continuous = False
    terminal_deadline = 0.
    last_save = wall_step = time.monotonic()
    def toggle():
        global active
        if planner_blocked:
            label.text = "Planning blocked: click Replay to reset manually."
            return
        active = not active
    def replay():
        global replay_requested
        replay_requested = True
    def robot_manual(v, w):
        global robot_auto, robot_manual_until, active
        if owned_policy:
            raise RuntimeError("Training policy owns wheel targets")
        if planner_blocked:
            robot.stop()
            robot_label.text = "Planning blocked. Click Replay for a manual reset."
            return
        if control >= 1800 and not (args.optimized_preview or args.sfm_preview):
            robot.stop()
            robot_label.text = "Round finished. Click Replay before moving the robot."
            return
        robot_auto = False
        report["robot_movement_status"] = report.get("robot_movement_status", "MANUAL") if report.get("robot_smoke") else "AUTO_CHECK_INTERRUPTED_BY_USER"
        robot.command(v, w)
        robot_manual_until = control + 60
        active = True
    def robot_stop():
        global robot_auto, robot_manual_until
        robot_auto = False
        robot_manual_until = control
        robot.stop()
    window = ui.Window(f"NewMetro - robot / LiDAR / {count} pedestrians", width=450, height=315)
    with window.frame:
        with ui.VStack(spacing=8):
            label = ui.Label("SFM + AMP: continuous patrol" if args.sfm_preview else
                             "Route + TRACE fallback + AMP: continuous patrol" if args.optimized_preview
                             else f"TRACE + AMP: {count} pedestrians / 60s")
            ui.Button("Pause / Resume", clicked_fn=toggle)
            ui.Button("Replay (manual reset)", clicked_fn=replay,enabled=not owned_policy)
            with ui.HStack():
                ui.Label("Hide ceiling")
                ui.CheckBox(model=hidden)
            ui.Label("Robot: manual wheel-drive" if args.manual_robot else "Robot: physical wheel-drive check, then manual controls")
            with ui.HStack():
                ui.Button("Forward (2s)", clicked_fn=lambda: robot_manual(.18,0.))
                ui.Button("Backward (2s)", clicked_fn=lambda: robot_manual(-.18,0.))
            with ui.HStack():
                ui.Button("Left (2s)", clicked_fn=lambda: robot_manual(0.,.4))
                ui.Button("Right (2s)", clicked_fn=lambda: robot_manual(0.,-.4))
                ui.Button("Stop", clicked_fn=robot_stop)
            with ui.HStack():
                ui.Label("Show LiDAR hits")
                ui.CheckBox(model=show_lidar)
            robot_label = ui.Label("Robot/LiDAR check starting")
            ui.Label("Adjust view freely. No RL training.")
    report["status"] = "TRAINING_STARTING" if owned_policy else "RUNNING"
    if args.optimized_preview:
        import preview_runtime_optimization
        preview_runtime_optimization.install(sys.modules[__name__])
    if not owned_policy:
        from apply_preview_performance_patch import apply_to_preview
        apply_to_preview(sys.modules[__name__])
    training_bridge = None
    if args.isolated_training:
        from sfm_training_bridge import SfmTrainingBridge
        training_bridge = SfmTrainingBridge(sys.modules[__name__])
        from performance_probe import install
        install(sys.modules[__name__])
    elif args.isolated_demo:
        from sfm_demo_bridge import SfmDemoBridge
        training_bridge = SfmDemoBridge(sys.modules[__name__])
        if os.environ.get('NEWMETRO_GPU_EXECUTION')=='1':
            from gpu_execution import install as gpu_install
            gpu_install(sys.modules[__name__])
    if args.continuous_trace:
        import continuous_trace_preview
        continuous_trace_preview.install(sys.modules[__name__])
    save()
    print("NEWMETRO_GUI_RUNNING=" + str(OUT), flush=True)
    while app.is_running():
        if training_bridge is not None and training_bridge.should_stop(control):
            break
        if replay_requested:
            if args.continuous_trace:
                invalidate_continuous_trace()
            previous_smoke = OUT / "ROBOT_SMOKE.json"
            if previous_smoke.exists():
                previous_smoke.replace(OUT / f"ROBOT_SMOKE_replay_{report.get('manual_replays', 0)}.json")
            world.reset(torch.arange(count, device=world.device))
            robot.initialize()
            lidar.reset_clock()
            robot_rows = []
            robot_auto = not args.manual_robot
            robot_manual_until = 0
            report["robot_smoke"] = None
            report["robot_movement_status"] = "SHORT_SMOKE_PENDING"
            runtime = new_runtime()
            if manual_planner_wait is not None:
                manual_planner_wait = ManualPreviewPlannerWait(
                    TrajectoryAdmissionError, report, record_planner_wait)
                report['pedestrian_planning_wait'] = False
            state = world.state()
            start = state.root_pos.clone()
            first_joints = state.dof_pos.clone()
            max_displacements.zero_()
            route_indices.zero_()
            if args.optimized_preview:
                route_target_cache.clear()
            control, active, replay_requested = 0, True, False
            planner_blocked = False
            report["planner_blocked"] = False
            report["status"] = "RUNNING"
            report["manual_replays"] = report.get("manual_replays", 0) + 1
            report["route_vertices_reached_per_actor"] = [0] * count
            label.text = ("SFM + AMP: continuous patrol" if args.sfm_preview
                          else f"TRACE + AMP: {count} pedestrians / 60s")
        if terminal_continuous and time.time() > terminal_deadline:
            robot.stop()
            robot_label.text = "Terminal disconnected: robot stopped"
        if active and (control < 1800 or terminal_continuous or args.optimized_preview or args.sfm_preview):
            try:
                before = world.planar()
                policy_command = None
                if training_bridge is not None:
                    policy_command = training_bridge.command(control, control*world.dt,
                                                             before.detach().cpu().numpy())
                    if policy_command.get("training_done"):
                        if training_bridge.handle_terminal(control, control*world.dt):
                            break
                        continue
                if args.sfm_preview:
                    route_indices, reached = runtime.advance_route(before[:, :2], route_indices)
                else:
                    route_indices, reached = native.advance_route(before[:, :2], route_indices)
                report["route_vertices_reached_per_actor"] = [a + int(b) for a, b in zip(report["route_vertices_reached_per_actor"], reached.tolist())]
                if args.sfm_preview:
                    started = time.monotonic()
                    action, diag = runtime.action(world, control, route_indices)
                    if not owned_policy: pump_gui()
                else:
                    def fresh_targets():
                        def produce():
                            old_failures = native.global_path_failures
                            targets, _, _ = native.route_waypoints(before[:, :2], route_indices, advance=False)
                            if native.global_path_failures != old_failures:
                                raise RuntimeError("Existing global route query failed; preserved without changing route")
                            report['route_targets_without_robot'] = targets.tolist()
                            from robot_pedestrian_neighbor import detour_targets
                            return detour_targets(before, targets, native, robot.observe(), report=report)
                        if args.optimized_preview:
                            with preview_timings.measure('route_query_cpu'):
                                return route_target_cache.get(control, route_indices, produce, force=True)
                        return produce()
                    if args.optimized_preview:
                        targets = route_target_cache.get(control, route_indices, fresh_targets)
                    else:
                        targets = fresh_targets()
                        pump_gui()
                    started = time.monotonic()
                    if manual_planner_wait is not None:
                        action, diag = manual_planner_wait.action(
                            runtime, world, control, targets, route_target_provider=fresh_targets)
                    else:
                        action, diag = runtime.action(world, control, targets, route_target_provider=fresh_targets)
                if diag["replanned"]:
                    report["replan_count"] = report.get("replan_count", 0) + 1
                    report["last_replan_wall_seconds"] = time.monotonic() - started
                segment = "manual"
                if training_bridge is not None:
                    robot.command(policy_command["desired_v_m_s"], policy_command["desired_w_rad_s"])
                    segment = policy_command.get('owner', 'FROZEN_DRLVO_DEMO') if args.isolated_demo else ('DRLVO_PPO' if policy_command.get('owner') == 'DRLVO_NATIVE_SB3_PPO' else 'SENSOR_WARMUP')
                elif robot_auto:
                    boundary = 0
                    for name, length, v, w in smoke_segments:
                        boundary += length
                        if control < boundary:
                            segment = name
                            robot.command(v,w)
                            break
                    else:
                        robot.stop()
                        robot_auto = False
                elif not terminal_continuous and control >= robot_manual_until:
                    robot.stop()
                    segment = "idle"
                with torch.no_grad():
                    state, fallen, truncated = world.step(action)
                if training_bridge is not None:
                    from fall_recovery import check_after_physics
                    check_after_physics(world,state,action,fallen,control+1,runtime,OUT,
                        training_bridge.driver.policy,report)
                mesh.update()
                control += 1
                observed = robot.observe()
                if training_bridge is not None:
                    training_bridge.post_physics(control, control*world.dt, observed)
                report["robot_observation"] = observed
                row = dict(control=control,sim_time=control*world.dt,segment=segment,**observed)
                robot_rows.append(row)
                robot_writer.writerow([control,control*world.dt,segment,*observed["requested_body_command_m_s_rad_s"],
                    *observed["root_position_m"],observed["yaw_rad"],observed["tilt_rad"],
                    observed["physical_forward_velocity_m_s"], observed["physical_yaw_velocity_rad_s"],
                    *observed["wheel_actual_velocity_rad_s"]])
                if not observed["finite"] or observed["tilt_rad"] > .45 or not -.08 < observed["root_position_m"][2] < .5:
                    robot.stop()
                    report["robot_movement_status"] = "PHYSICAL_STATE_GATE_FAILED"
                    if training_bridge is not None:
                        from episode_semantics import EnvironmentEpisodeFault
                        raise EnvironmentEpisodeFault("robot_physical_state_invalid", observed)
                    raise RuntimeError("Navigation robot state gate failed; original state retained")
                scan = (None if training_bridge is not None or
                        (args.calibration_fixture and control < 30) else
                        lidar.scan(observed["root_position_m"], observed["root_quaternion_xyzw"], control*world.dt))
                if scan is not None:
                    lidar_last = scan
                    report["lidar_scan_count"] += 1
                    report["lidar_max_pedestrian_hit_count"] = max(report["lidar_max_pedestrian_hit_count"],scan["stats"]["pedestrian_hit_count"])
                    stats = dict(scan["stats"],sim_time=control*world.dt,
                        beam_count_per_sensor=[len(scan[n]["ranges"]) for n in ("scan_01","scan_02")])
                    report["lidar_latest_stats"] = stats
                    lidar_rows.append(stats)
                    point_rows = []
                    for name in ("scan_01", "scan_02"):
                        packet = scan[name]
                        mask = np.isfinite(packet["ranges"])
                        point_rows.extend(packet["points_world_m"][mask][::8].tolist())
                    lidar_points.GetPointsAttr().Set(point_rows)
                    lidar_points.GetWidthsAttr().Set([.015] * len(point_rows))
                    report["lidar_visual_hit_point_count"] = len(point_rows)
                    if args.calibration_fixture:
                        pose_now = np.asarray([*observed["root_position_m"],
                                               *observed["root_quaternion_xyzw"]], dtype=float)
                        if calibration_start_pose is None:
                            calibration_start_pose = pose_now.copy()
                        if float(np.max(np.abs(pose_now-calibration_start_pose))) > .01:
                            raise RuntimeError('Robot moved during fixed-mask calibration')
                        for name in calibration_mask:
                            packet = scan[name]
                            ranges = np.asarray(packet["ranges"], dtype=float)
                            valid = np.isfinite(ranges)
                            if valid.any():
                                calibration_min_external = min(calibration_min_external,
                                                               float(np.min(ranges[valid])))
                            for beam, hit_paths in enumerate(packet.get("self_hit_paths", [])):
                                for hit_path in hit_paths:
                                    if not hit_path.startswith('/World/NavigationRobot/'):
                                        raise RuntimeError('Self-hit path is outside NavigationRobot root')
                                    calibration_mask[name].add(beam)
                                    counts = calibration_self_paths[name]
                                    counts[hit_path] = counts.get(hit_path, 0) + 1
                        target_rows_by_sensor = {}
                        for sensor_name in calibration_mask:
                            sensor_packet = scan[sensor_name]
                            self_by_beam = sensor_packet.get('self_hit_paths', [[] for _ in range(2000)])
                            candidates = []
                            for beam, hit_path in enumerate(sensor_packet['hit_paths']):
                                if (str(hit_path).startswith('/World/CalibrationObstacle') and
                                        not self_by_beam[beam]):
                                    candidates.append(dict(beam_index=beam,
                                        range_m=float(sensor_packet['ranges'][beam]),
                                        hit_path=str(hit_path)))
                            target_rows_by_sensor[sensor_name] = candidates
                        target_row = dict(scan=report['lidar_scan_count'],
                            unmasked_external_hits=target_rows_by_sensor,
                            scan_01_self_occluded_target_beams=[i for i,p in enumerate(scan['scan_01']['self_hit_paths'])
                                if p and str(scan['scan_01']['hit_paths'][i]).startswith('/World/CalibrationObstacle')],
                            scan_02_self_occluded_target_beams=[i for i,p in enumerate(scan['scan_02']['self_hit_paths'])
                                if p and str(scan['scan_02']['hit_paths'][i]).startswith('/World/CalibrationObstacle')])
                        calibration_target_rows.append(target_row)
                        if report['lidar_scan_count'] >= 10:
                            if calibration_min_external < 2.0:
                                raise RuntimeError('Calibration fixture has an external return under 2 m')
                            if not all(any(2.3 <= hit['range_m'] <= 2.7 and
                                               hit['hit_path'].startswith('/World/CalibrationObstacle') and
                                               hit['beam_index'] not in calibration_mask[sensor_name]
                                           for hit in row['unmasked_external_hits']['scan_01'])
                                       for row in calibration_target_rows):
                                raise RuntimeError('External obstacle was not preserved on the unmasked beam')
                            if not all(calibration_mask[n] for n in calibration_mask):
                                raise RuntimeError('Self-hit paths were not identified for both LiDARs')
                            sensor_config = dict(beam_count=lidar.beam_count,
                                angle_min_rad=-math.pi,
                                angle_increment_rad=2*math.pi/lidar.beam_count,
                                range_min_m=lidar.range_min, range_max_m=lidar.range_max,
                                rate_hz=lidar.rate_hz,
                                mounts=[dict(name=n, xyz_m=mount.tolist(), yaw_rad=yaw)
                                        for n,mount,yaw,_ in lidar.mounts])
                            mask_record = dict(schema='newmetro.dual_lidar_fixed_self_mask.v1',
                                status='CALIBRATION_PASS',
                                robot_usd=source['robot_usd'],
                                robot_usd_sha256=source['robot_sha256'],
                                robot_root_path='/World/NavigationRobot',
                                sensor=sensor_config,
                                fixed_mask_beam_indices={n: sorted(map(int,calibration_mask[n]))
                                                         for n in calibration_mask},
                                self_hit_paths=calibration_self_paths,
                                calibration=dict(post_physics_frames=report['lidar_scan_count'],
                                    robot_stationary_max_delta_m=0.01,
                                    empty_area_min_external_range_m=calibration_min_external,
                                    external_obstacle_path='/World/CalibrationObstacle',
                                    external_test_unmasked_hits=target_rows_by_sensor,
                                    scan_02_target_occlusion='robot_self_paths_recorded_separately',
                                    external_test_rows=calibration_target_rows,
                                    external_test_passed_unmasked=True,
                                    disabled_static_collision_paths=disabled),
                                training=False, ppo_updates=0, long_training_started=False)
                            (OUT/'LIDAR_SELF_MASK.json').write_text(json.dumps(mask_record,indent=2)+'\n')
                            report['lidar_calibration_fixture'].update(status='CALIBRATION_PASS',
                                mask_path=str(OUT/'LIDAR_SELF_MASK.json'),
                                mask_beam_counts={n:len(calibration_mask[n]) for n in calibration_mask},
                                frames=report['lidar_scan_count'],
                                minimum_external_range_m=calibration_min_external,
                                external_obstacle_test='PASS_UNMASKED')
                            report['status']='LIDAR_SELF_MASK_CALIBRATION_PASS'
                            robot.stop(); active=False
                            save(); file.flush(); robot_file.flush()
                            break
                smoke_length = sum(x[1] for x in smoke_segments)
                if control == smoke_length and robot_auto:
                    from robot_smoke_metrics import evaluate_robot_smoke
                    report["robot_smoke"] = evaluate_robot_smoke(robot_rows)
                    report["robot_smoke"]["replay_id"] = report.get("manual_replays", 0)
                    report["robot_movement_status"] = report["robot_smoke"]["status"]
                    (OUT / "ROBOT_SMOKE.json").write_text(json.dumps(report["robot_smoke"],indent=2)+"\n")
                    robot.stop()
                    robot_auto = False
                    robot_label.text = report["robot_movement_status"] + "; manual controls available"
                    label.text = f"{count} pedestrians running; robot check complete"
                elif control % 30 == 0:
                    robot_label.text = f"Robot {segment}; scans {report['lidar_scan_count']}; people hits {report['lidar_max_pedestrian_hit_count']}"
                displacements = (state.root_pos[:, :2] - start[:, :2]).norm(dim=-1)
                max_displacements = torch.maximum(max_displacements, displacements)
                delta = (state.dof_pos - first_joints).abs().amax(dim=-1)
                positions = state.root_pos.tolist()
                pair_centers = [math.dist(positions[i][:2], positions[j][:2])
                    for i in range(count) for j in range(i+1, count)]
                measured_pair_center = min(pair_centers, default=math.inf)
                report.update(actual_controls=control, sim_time=control * world.dt,
                    max_root_displacement_per_actor_m=max_displacements.tolist(),
                    max_joint_delta_per_actor_rad=delta.tolist(), positions=positions,
                    motion_observed_per_actor=((max_displacements > .3) & (delta > .1)).tolist(),
                    speed_per_actor_m_s=state.root_vel[:, :2].norm(dim=-1).tolist(),
                    safety_braking_plans=runtime.safety_braking_plans,
                    route_indices=route_indices.tolist(), truncated=truncated.tolist(),
                    minimum_measured_pedestrian_pair_center_m=min(
                        report.get('minimum_measured_pedestrian_pair_center_m', math.inf), measured_pair_center))
                for i in range(count):
                    writer.writerow([report.get("manual_replays", 0), control, control * world.dt, i,
                        *positions[i], report["speed_per_actor_m_s"][i], float(delta[i]), bool(fallen[i]), int(route_indices[i])])
                if training_bridge is not None and measured_pair_center < .8:
                    report["pedestrian_pair_close_events"] = report.get("pedestrian_pair_close_events", 0) + 1
                    if report["pedestrian_pair_close_events"] == 1 or control % 30 == 0:
                        with (OUT / "PEDESTRIAN_NEAR_EVENTS.jsonl").open("a") as events:
                            events.write(json.dumps(dict(control=control, sim_time=control*world.dt,
                                minimum_pair_center_m=measured_pair_center,
                                batch_paused=False, counted_as_robot_collision=False)) + "\n")
                if args.sfm_preview and measured_pair_center < .8 and training_bridge is None:
                    report['status'] = 'PEDESTRIAN_PAIR_CLOSE_PAUSED'
                    report['pair_close_control'] = control
                    report['pair_close_center_m'] = measured_pair_center
                    active = False
                    robot.stop()
                    label.text = 'Pedestrian pair too close: paused for inspection.'
                elif bool(fallen.any()):
                    if training_bridge is not None:
                        from episode_semantics import EnvironmentEpisodeFault
                        raise EnvironmentEpisodeFault("pedestrian_fall", {"control":control,"actors":torch.where(fallen)[0].tolist()})
                    report["falls"] += int(fallen.sum())
                    report["fall_events"].append({"control":control, "actors":torch.where(fallen)[0].tolist()})
                    report["status"] = "FALL_PAUSED"
                    active = False
                    robot.stop()
                    label.text = "Physical fall: paused. Replay is manual."
                elif bool(truncated.any()):
                    raise RuntimeError("Unexpected reset boundary")
                elif report.get('pedestrian_planning_wait'):
                    report['status'] = 'PEDESTRIANS_WAITING_ROBOT_TELEOP_ACTIVE'
                    label.text = 'Pedestrians waiting; replanning. Robot teleop available.'
                elif any(report.get('pedestrian_execution_stalled', [])):
                    report['status'] = 'PEDESTRIAN_EXECUTION_STALLED_REPLANNING'
                    label.text = 'Measured pedestrian stall; TRACE replanning. Robot teleop available.'
                elif args.optimized_preview and args.manual_robot and report.get('pedestrian_recent_motion_ready'):
                    report['status'] = ('PEDESTRIANS_MOVING_GUI_UPDATING'
                        if all(value >= .05 for value in report['pedestrian_recent_motion_m'])
                        else 'PEDESTRIANS_PARTLY_STATIONARY_GUI_UPDATING')
                elif training_bridge is not None:
                    report['status'] = training_bridge.status if args.isolated_demo else 'TRAINING_RUNNING'
                elif all(report["motion_observed_per_actor"]):
                    report["status"] = "FOUR_PERSON_PHYSICAL_MOTION_OBSERVED_GUI_UPDATING"
                if control == 1800 and not terminal_continuous and not (args.optimized_preview or args.sfm_preview):
                    robot.stop()
                    active = False
                    label.text = "60s preview finished. Replay is manual."
                    robot_label.text = "Robot stopped. Click Replay for a new round."
                if control % 30 == 0:
                    print(json.dumps({k: report.get(k) for k in ["status", "actual_controls", "falls", "positions", "route_vertices_reached_per_actor", "last_replan_wall_seconds"]}), flush=True)
                wall_step += world.dt
                delay = wall_step - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, world.dt))
                else:
                    wall_step = time.monotonic()
            except TrajectoryAdmissionError as error:
                robot.stop()
                robot_auto = terminal_continuous = active = False
                planner_blocked = True
                report.update(status="PLANNING_BLOCKED_GUI_OPEN", planner_blocked=True,
                    last_admission_error=str(error), last_admission_control=control,
                    last_admission_traceback=traceback.format_exc(),
                    planner_blocked_count=report.get("planner_blocked_count", 0)+1)
                label.text = "No admitted trajectory: simulation paused. Replay resets manually."
                robot_label.text = "Robot stopped; scene and window retained."
                save()
                file.flush()
                robot_file.flush()
                print("PLANNING_BLOCKED_GUI_RETAINED", control, str(error), flush=True)
        else:
            time.sleep(.01)
            wall_step = time.monotonic()
        world.simulator._sim.render()
        pump_gui()
        if time.monotonic() - last_save > 1:
            save()
            file.flush()
            robot_file.flush()
            if lidar_last is not None:
                np.savez_compressed(OUT / "latest_lidar.npz",
                    scan_01=lidar_last["scan_01"]["ranges"],scan_02=lidar_last["scan_02"]["ranges"],
                    scan_01_points_world=lidar_last["scan_01"]["points_world_m"],
                    scan_02_points_world=lidar_last["scan_02"]["points_world_m"],
                    scan_01_hit_paths=np.asarray(lidar_last["scan_01"]["hit_paths"]),
                    scan_02_hit_paths=np.asarray(lidar_last["scan_02"]["hit_paths"]),
                    scan_01_self_hit_paths=np.asarray([json.dumps(v) for v in lidar_last["scan_01"].get("self_hit_paths", [])]),
                    scan_02_self_hit_paths=np.asarray([json.dumps(v) for v in lidar_last["scan_02"].get("self_hit_paths", [])]),
                    sim_time=lidar_rows[-1]["sim_time"])
                (OUT / "lidar_stats.json").write_text(json.dumps(lidar_rows,indent=2)+"\n")
            last_save = time.monotonic()
        terminal_request = OUT / "ROBOT_REQUEST.json"
        if terminal_request.exists() and not owned_policy:
            try:
                message = json.loads(terminal_request.read_text())
                if message.get("sequence") != terminal_sequence:
                    terminal_sequence = message.get("sequence")
                    operation = message.get("key")
                    commands = {"w":(.18,0.), "s":(-.18,0.), "a":(0.,.4), "d":(0.,-.4)}
                    if time.time() - float(message.get("sent_at", 0)) > (3. if operation == "velocity" else 10.):
                        result = "EXPIRED"
                    elif operation == "velocity":
                        v, w = float(message["v"]), float(message["w"])
                        if planner_blocked:
                            robot.stop()
                            result = "PLANNING_BLOCKED_REPLAY_REQUIRED"
                        else:
                            robot.command(v, w)
                            robot_auto = False
                            terminal_continuous = True
                            terminal_deadline = float(message["sent_at"]) + 3.
                            active = True
                            result = "VELOCITY_APPLIED"
                            report["robot_movement_status"] = "TERMINAL_VELOCITY_CONTROL"
                            robot_label.text = f"Terminal: v={robot._command[0]:.2f} m/s w={robot._command[1]:.2f} rad/s"
                    elif operation in commands:
                        robot_manual(*commands[operation])
                        result = "APPLIED" if control < 1800 else "ROUND_FINISHED_USE_R"
                    elif operation == "x":
                        robot_stop()
                        result = "STOPPED"
                    elif operation == "r":
                        replay()
                        result = "REPLAY_REQUESTED"
                    elif operation == "p":
                        toggle()
                        result = "PAUSE_TOGGLED"
                    else:
                        result = "UNKNOWN_KEY"
                    acknowledgement = dict(sequence=terminal_sequence, key=operation,
                        result=result, control=control, received_at=time.time())
                    (OUT / "ROBOT_ACK.json").write_text(json.dumps(acknowledgement)+"\n")
                    report["terminal_command"] = acknowledgement
            except (ValueError, TypeError, KeyError) as error:
                report["terminal_command_error"] = str(error)
        request = OUT / "VIEW_REQUEST.json"
        if request.exists() and json.loads(request.read_text()).get("close"):
            if args.isolated_demo:
                report["status"] = "DEMO_USER_CLOSED"
                report["demo_close_source"] = "owned_view_request"
            break
except SystemExit:
    raise
except BaseException as error:
    error_code = 1
    from episode_semantics import EnvironmentEpisodeFault
    if isinstance(error, EnvironmentEpisodeFault):
        report.update(status="ENVIRONMENT_FAULT_STOPPED", error=str(error),
                      environment_fault=error.reason, environment_diagnostics=error.diagnostics,
                      not_robot_collision_reward=True, traceback=traceback.format_exc())
        (OUT / "ENVIRONMENT_FAULT.json").write_text(json.dumps(dict(
            reason=error.reason, diagnostics=error.diagnostics,
            not_robot_collision_reward=True), indent=2) + "\n")
    else:
        report.update(status="FAILED", error=str(error), traceback=traceback.format_exc())
    traceback.print_exc()
finally:
    if 'training_bridge' in globals() and training_bridge is not None:
        training_bridge.close(failed=bool(error_code))
    if args.continuous_trace and 'close_continuous_trace' in globals():
        close_continuous_trace()
    if robot is not None:
        robot.stop()
    if robot_file is not None:
        robot_file.close()
    if file is not None:
        file.close()
    report["window_open"] = False
    save()
    if world is not None:
        world.close()
    app.close(wait_for_replicator=False, exit_code=error_code)
