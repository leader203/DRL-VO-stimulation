"""Compare prepared data contract against executable AST from protected sources.

Does not import ROS, Gym, Isaac or PPO. Synthetic inputs exercise numerical
equivalence, not sensor accuracy, physical movement or RL readiness.
"""
from __future__ import annotations
import ast
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import numpy.matlib
import drlvo_observation_adapter as new

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/home/user/navigation_project/a_pipeline')
ENV_SOURCE = OLD/'github_src/drl_vo_nav-drl_vo/drl_vo/src/turtlebot_gym/turtlebot_gym/envs/drl_nav_env.py'
ADAPTER_SOURCE = OLD/'methods/experiments/drl_vo_ros2_offline/observation_adapter.py'
OUT = ROOT/'audit'


def functions_from_source(path, names, *, methods=False):
    tree = ast.parse(path.read_text())
    nodes = list(ast.walk(tree)) if methods else tree.body
    selected = [node for node in nodes if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != len(names):
        raise RuntimeError(f'Could not identify all reference functions in {path}')
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = dict(np=np, math=math, SCAN_HISTORY=10, FULL_SCAN_BEAMS=1080,
                     FRONT_SCAN=slice(180,900), LEGACY_ANGLE_MIN=-3*math.pi/4,
                     LEGACY_ANGLE_INCREMENT=math.pi/720, PED_MAP_SHAPE=(2,80,80))
    namespace['rospy'] = SimpleNamespace(logdebug=lambda *a:None, logwarn=lambda *a:None,
        Rate=lambda rate:SimpleNamespace(sleep=lambda:None))
    namespace['Twist'] = lambda: SimpleNamespace(linear=SimpleNamespace(x=0), angular=SimpleNamespace(z=0))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = functions_from_source(ENV_SOURCE, {'_get_observation', '_take_action'}, methods=True)
    ros2 = functions_from_source(ADAPTER_SOURCE, {'compress_scan_history','dual_lidar_to_legacy_scan',
        '_dual_lidar_to_legacy_front','pedestrian_velocity_map','rotate_map_to_base'})
    rng = np.random.default_rng(20261010)
    obs_errors, lidar_matches, pedestrian_matches = [], [], []
    for case in range(24):
        ped = rng.uniform(-3,3,(2,80,80)).astype(np.float32)
        hist = rng.uniform(0,30,(10,720)).astype(np.float32)
        hist[:,::37] = 0
        goal = rng.uniform(-8,8,2).astype(np.float32)
        state = SimpleNamespace(cnn_data=SimpleNamespace(ped_pos_map=ped.reshape(-1),scan=hist.reshape(-1),goal_cart=goal,vel=[0,0]))
        expected = original['_get_observation'](state)
        original_observation_dtype = str(expected.dtype)
        actual = new.build_observation(ped,hist,goal)
        error = float(np.max(np.abs(actual-expected)))
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=1e-6)
        obs_errors.append(error)
        np.testing.assert_array_equal(new.compress_scan_history(hist),ros2['compress_scan_history'](hist))
        ranges = rng.uniform(.02,30,4000).astype(np.float32)
        angles = rng.uniform(-math.pi,math.pi,4000).astype(np.float32)
        ranges[::31] = np.inf; ranges[::57] = np.nan; ranges[::97] = 0
        actual_lidar = new.dual_lidar_to_legacy_scan(ranges,angles)
        expected_lidar = ros2['dual_lidar_to_legacy_scan'](ranges,angles)
        np.testing.assert_array_equal(actual_lidar[0],expected_lidar[0])
        assert actual_lidar[1:] == expected_lidar[1:]
        lidar_matches.append(True)
        pose = np.asarray([1,2,1.2],dtype=np.float32)
        pos = rng.uniform(-10,20,(12,2)).astype(np.float32)
        vel = rng.uniform(-2,2,(12,2)).astype(np.float32)
        expected_map = ros2['pedestrian_velocity_map'](pos,vel,pose)[0]
        actual_map = new.oracle_pedestrian_velocity_map(pos,vel,pose)
        np.testing.assert_array_equal(actual_map,expected_map)
        pedestrian_matches.append(True)
    action_pairs = []
    for action in ([ -1,-1 ],[0,0],[1,1],[.4,-.8]):
        captured=[]
        original['_take_action'](SimpleNamespace(_cmd_vel_pub=SimpleNamespace(publish=captured.append)),action)
        expected = np.array([captured[0].linear.x,captured[0].angular.z])
        np.testing.assert_allclose(new.normalized_to_physical(action),expected,atol=1e-7)
        action_pairs.append(dict(action=action,physical=expected.tolist()))
    # Analytic bin sentinels distinguish scan/ped order, temporal min/mean rows,
    # four repetitions of the 1600 block, and absolute-vs-relative velocity.
    hist=np.zeros((10,720),dtype=np.float32); hist[7,27:36]=np.arange(9)+1
    compressed=new.compress_scan_history(hist)
    assert compressed[14*80+3] == 1 and compressed[15*80+3] == 5
    for block in range(4):
        assert compressed[block*1600+14*80+3] == 1
    ped=new.oracle_pedestrian_velocity_map([[1,0]],[[.8,.2]],[0,0,0])
    assert np.allclose(ped[:,4,40],[.8,.2])
    # Invalid timestamp / incomplete history must not silently become an input.
    history=new.ScanHistory()
    try: history.array()
    except RuntimeError: pass
    else: raise AssertionError('warmup accepted')
    samples = []
    for tick in range(1801):
        stamp = round(tick*1e9/30)
        if tick % 2 == 0:
            history.append(np.full(720,tick+1),stamp)
        if history.sample(stamp):
            samples.append(stamp)
            assert history.last_timestamp_ns <= stamp
            np.testing.assert_array_equal(history.frames[-1],history.latest_scan)
    assert len(samples) == 1201
    assert set(np.round(np.diff(samples)/1e9,6)) == {.033333,.066667}
    assert len(set(history.source_timestamps_ns)) < 10
    assert all(a <= b for a,b in zip(history.source_timestamps_ns,history.sample_timestamps_ns))
    assert history.ready
    for bad_stamp in (history.latest_scan_timestamp_ns,history.latest_scan_timestamp_ns+300000000):
        try: history.append(np.ones(720),bad_stamp)
        except ValueError: pass
        else: raise AssertionError('invalid timestamp accepted')
    history.reset(); assert not history.ready and history.latest_scan is None
    assert history.last_timestamp_ns is None and not history.source_timestamps_ns
    assert not history.sample(0)
    history.append(np.ones(720),100)
    try: history.sample(99)
    except ValueError: pass
    else: raise AssertionError('future scan accepted')
    metadata=dict(angle_min=-math.pi,angle_increment=2*math.pi/2000,range_min=.02,
        range_max=30,translation_xy=(0,0),yaw=0)
    # Original ROS1 no-return semantics: no fabricated maximum-range endpoints,
    # including with translated sensors. A real hit at range_max remains valid.
    metadata['translation_xy'] = (.2, .13)
    for missing in (np.inf, -np.inf, np.nan, 0.):
        ranges,angles=new.sensor_ranges_to_base(np.full(2000,missing),**metadata)
        assert np.isnan(ranges).all() and np.isnan(angles).all()
        front,coverage,minimum=new.dual_lidar_to_legacy_scan(
            np.tile(ranges,2),np.tile(angles,2))
        assert not front.any() and coverage == 0 and math.isnan(minimum)
        obs=new.build_observation(np.zeros((2,80,80)),np.tile(front,(10,1)),[0,0])
        np.testing.assert_array_equal(obs[12800:19200],-np.ones(6400))
    raw=np.full(2000,np.inf); raw[1000]=metadata['range_max']
    ranges,angles=new.sensor_ranges_to_base(raw,**metadata)
    assert np.isfinite(ranges).sum() == 1
    np.testing.assert_allclose(ranges[1000],np.hypot(30.2,.13),atol=3e-6)
    front,coverage,minimum=new.dual_lidar_to_legacy_scan(
        np.r_[np.full(2000,np.nan),ranges],np.r_[np.full(2000,np.nan),angles])
    assert np.count_nonzero(front) == 1 and np.isclose(minimum,ranges[1000])
    try: new.sensor_ranges_to_base(np.ones(720),**metadata)
    except ValueError: pass
    else: raise AssertionError('wrong raw beam count accepted')
    report=dict(status='CPU_CONTRACT_PASS',not_claimed=['IsaacLab environment runtime','physical sensor accuracy',
        'robot dynamics','navigation success','PPO or RL training'],seed=20261010,synthetic_cases=24,
        original_observation_max_abs_error=max(obs_errors),original_output_dtype=original_observation_dtype,
        prepared_output_dtype='float32',ros2_front_scan_exact_cases=sum(lidar_matches),
        ros2_pedestrian_map_exact_cases=sum(pedestrian_matches),action_reference_pairs=action_pairs,
        history_reset_and_clock_gates='PASS',history_sampling_hz=20,physical_lidar_hz=15,
        history_samples_in_60_seconds_inclusive=len(samples),analytic_order_sentinels='PASS',
        versions={name:importlib.metadata.version(name) for name in ('numpy','torch','gymnasium')},
        sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (ENV_SOURCE,ADAPTER_SOURCE,Path(new.__file__),Path(__file__))})
    (OUT/'CPU_CONTRACT_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
