"""Bounded isolated PhysX calibration of the fixed EVT dual-LiDAR self mask.

This creates a clean floor, the exact versioned EVT robot USD, and one 2.5 m
external test target. It performs no policy inference or training.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from robot_preview_adapter import EXPECTED_SHA256, ROBOT_USD, make_robot_cfg, RobotPreviewAdapter
from lidar_preview_adapter import LidarPreviewAdapter

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
out = a.output.resolve()
out.mkdir(parents=True, exist_ok=False)
sys.argv = [sys.argv[0]]
import warp
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'renderer': 'RaytracedLighting', 'multi_gpu': False})
sim = None
try:
    print('CAL_STAGE=app_ready', flush=True)
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom
    import omni.physx

    if hashlib.sha256(ROBOT_USD.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError('EVT robot USD hash changed')
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device='cpu'))
    print('CAL_STAGE=simulation_context', flush=True)
    ground = sim_utils.GroundPlaneCfg()
    ground.func('/World/CalibrationGround', ground)

    stage = sim_utils.get_current_stage()
    print('CAL_STAGE=stage_ready', flush=True)
    robot_root = UsdGeom.Xform.Define(stage, '/World/NavigationRobot')
    robot_root.GetPrim().GetReferences().AddReference(str(ROBOT_USD))
    # EVT-EXP.usd stores its base_link at X=-1.2420253 m. Match the existing
    # UsdFileCfg spawn, which overrides that authored source offset.
    root_ops = robot_root.GetOrderedXformOps()
    if not root_ops or root_ops[0].GetOpType() != UsdGeom.XformOp.TypeTranslate:
        raise RuntimeError('Expected the robot source root translation op')
    root_ops[0].Set(Gf.Vec3d(1.2420253144257412, 0., .02))

    # A single external obstacle, 2.5 m from the scan_01 mount on a beam
    # outside the robot silhouette. This gives an explicit retained-hit check.
    beam_count = 2000
    target_beam = 1500  # local angle +pi/2 for scan_01
    target_angle = -math.pi + target_beam * (2 * math.pi / beam_count)
    target_distance_m = 2.5
    target_origin = np.array([.2, .13, .208], dtype=np.float64)
    target_center = target_origin + target_distance_m * np.array(
        [math.cos(target_angle), math.sin(target_angle), 0.])
    obstacle = sim_utils.CuboidCfg(
        size=(.1, .1, .25),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.))
    sim_utils.spawn_cuboid('/World/CalibrationObstacle', obstacle,
                           translation=tuple(target_center))
    sim.reset()
    print('CAL_STAGE=physics_reset', flush=True)
    sim.play()
    sim.step(render=False)
    app.update()
    pose = np.array([0., 0., .02, 0., 0., 0., 1.], dtype=np.float64)
    query = omni.physx.get_physx_scene_query_interface()
    print('CAL_STAGE=scene_query_ready', flush=True)
    lidar = LidarPreviewAdapter(query, range_min_m=.1, range_max_m=8.,
                                capture_self_echoes=True)

    masks = {n: set() for n in ('scan_01', 'scan_02')}
    path_counts = {n: {} for n in masks}
    frame_data = []
    target_rows = []
    started = time.monotonic()
    # 5 independent post-physics sweeps over 0.267 simulated seconds.
    for frame in range(5):
        app.update()
        pose_now = pose.copy()
        packet = lidar.scan(pose_now[:3], pose_now[3:], (frame + 1) / 15.)
        if packet is None:
            raise RuntimeError('Expected scheduled calibration scan missing')
        row = {'sim_time': packet['stats']['stamp_sim_time'],
               'scan_01_min_m': None, 'scan_02_min_m': None,
               'external_hits': {}}
        for name in masks:
            scan = packet[name]
            ranges = np.asarray(scan['ranges'], dtype=np.float64)
            valid = np.isfinite(ranges)
            row[name + '_min_m'] = float(ranges[valid].min()) if valid.any() else None
            for i, hit_paths in enumerate(scan['self_hit_paths']):
                for path in hit_paths:
                    if not (path == '/World/NavigationRobot' or
                            path.startswith('/World/NavigationRobot/')):
                        raise RuntimeError(f'Non-robot path classified as self: {path}')
                    masks[name].add(i)
                    path_counts[name][path] = path_counts[name].get(path, 0) + 1
            ext = np.flatnonzero(valid)
            if len(ext):
                row['external_hits'][name] = {
                    'beam_count': int(len(ext)),
                    'nearest_range_m': float(ranges[ext].min()),
                    'nearest_hit_path': str(scan['hit_paths'][int(ext[np.argmin(ranges[ext])])]),
                }
        ti = target_beam
        target_rows.append({
            'frame': frame,
            'beam_index': ti,
            'range_m': float(packet['scan_01']['ranges'][ti]),
            'hit_path': str(packet['scan_01']['hit_paths'][ti]),
            'self_hit_paths': packet['scan_01']['self_hit_paths'][ti],
        })
        frame_data.append(row)

    # Empty-world gate: all external returns in the initial calibration zone
    # must be at least 2 m away. The 2.5 m target is the only authored object.
    min_external = min((r[n + '_min_m'] for r in frame_data
                        for n in masks if r[n + '_min_m'] is not None), default=math.inf)
    if min_external < 2.0:
        raise RuntimeError(f'Calibration area too close to an external hit: {min_external:.3f} m')
    if target_beam in masks['scan_01']:
        raise RuntimeError('External validation beam overlaps calibrated self mask')
    if not all(2.3 <= r['range_m'] <= 2.7 and
               r['hit_path'].startswith('/World/CalibrationObstacle') and
               not r['self_hit_paths'] for r in target_rows):
        raise RuntimeError('External obstacle was not preserved on its unmasked beam')
    if not masks['scan_01'] or not masks['scan_02']:
        raise RuntimeError('No self echoes identified on one or both sensors')

    source = {
        'schema': 'newmetro.dual_lidar_fixed_self_mask.v1',
        'status': 'CALIBRATION_PASS',
        'robot_usd': str(ROBOT_USD),
        'robot_usd_sha256': EXPECTED_SHA256,
        'robot_root_path': '/World/NavigationRobot',
        'robot_mount_reference': '/World/NavigationRobot/EVT_3/base_link',
        'sensor': {'beam_count': beam_count, 'angle_min_rad': -math.pi,
                   'angle_increment_rad': 2*math.pi/beam_count,
                   'range_min_m': .1, 'range_max_m': 8., 'rate_hz': 15.,
                   'mounts': [{'name': n, 'xyz_m': xyz, 'yaw_rad': yaw}
                              for n, xyz, yaw in LidarPreviewAdapter.__init__.__globals__['DEFAULT_MOUNTS']]},
        'fixed_mask_beam_indices': {n: sorted(int(i) for i in masks[n]) for n in masks},
        'self_hit_paths': {n: dict(sorted(path_counts[n].items())) for n in masks},
        'calibration': {'post_physics_frames': len(frame_data),
                        'stationary_pose_xyz_xyzw': pose.tolist(),
                        'stationary_pose_max_delta_abs': 0.0,
                        'minimum_external_return_m': min_external,
                        'empty_area_gate_m': 2.0,
                        'external_test_obstacle_path': '/World/CalibrationObstacle',
                        'external_test_obstacle_beam_scan_01': target_beam,
                        'external_test_obstacle_target_distance_m': target_distance_m,
                        'external_test_rows': target_rows,
                        'external_test_passed_unmasked': True},
        'evidence': {'frames': frame_data, 'elapsed_wall_s': time.monotonic()-started,
                     'physics_backend': 'omni.physx.scene_query', 'training': False,
                     'ppo_updates': 0, 'robot_motion': 'stationary'}
    }
    (out/'LIDAR_SELF_MASK.json').write_text(json.dumps(source, indent=2)+'\n')
    (out/'CALIBRATION_PASS.json').write_text(json.dumps({
        'status': 'CALIBRATION_PASS', 'mask_path': str(out/'LIDAR_SELF_MASK.json'),
        'scan_01_mask_beams': len(masks['scan_01']),
        'scan_02_mask_beams': len(masks['scan_02']),
        'minimum_external_return_m': min_external,
        'external_obstacle_test': 'PASS_UNMASKED', 'training_started': False,
    }, indent=2)+'\n')
    print('LIDAR_SELF_MASK_CALIBRATION_PASS', flush=True)
except BaseException as e:
    (out/'CALIBRATION_FAULT.json').write_text(json.dumps({
        'error': str(e), 'type': type(e).__name__, 'traceback': __import__('traceback').format_exc(),
        'training_started': False,
        'diagnostics': {k: v for k, v in {
            'frame_data': locals().get('frame_data'),
            'target_rows': locals().get('target_rows'),
            'mask_beams': {n: sorted(masks[n]) for n in masks} if 'masks' in locals() else None,
            'pose': (locals().get('pose').tolist()
                     if isinstance(locals().get('pose'), np.ndarray) else locals().get('pose')),
        }.items()},
    }, indent=2)+'\n')
    raise
finally:
    if sim is not None:
        sim.stop()
    app.close(wait_for_replicator=False)
