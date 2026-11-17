"""Real-map task checks and actual reset-method tests with mocked physics, no Isaac."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import numpy as np

from drlvo_observation_adapter import ScanHistory, normalized_to_physical
from local_goal import ActionClock
from reward_logic import OriginalReward
from episode_semantics import EnvironmentEpisodeFault
from task_sampling import TaskSampler, TaskSamplingError, TaskAdmissionError, make_navigation, terminal_requires_teleport

BATCH = Path(__file__).resolve().parents[1]


def source_class(filename, name):
    path = BATCH/'code'/filename
    tree = ast.parse(path.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name)
    namespace = dict(np=np, math=math, json=json, Path=Path, time=time, ScanHistory=ScanHistory,
                     normalized_to_physical=normalized_to_physical, ActionClock=ActionClock,
                     EnvironmentEpisodeFault=EnvironmentEpisodeFault,
                     TaskAdmissionError=TaskAdmissionError,
                     terminal_requires_teleport=terminal_requires_teleport)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[name]


def check_boundaries(navigation, admission_failures=0):
    Bridge = source_class('sfm_training_bridge.py', 'SfmTrainingBridge')
    Driver = source_class('training_driver.py', 'TrainingPolicyDriver')
    Policy = source_class('training_policy.py', 'TrainingPolicy')
    class Tensor:
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.arange(16, dtype=float).reshape(4, 4).copy()
    class Robot:
        def __init__(self):
            self.pose = np.array([-6.85, -1.65, math.pi/2])
            self.reset_count = 0
        def stop(self): pass
        def observe(self):
            return dict(root_position_m=[*self.pose[:2], .02], yaw_rad=float(self.pose[2]))
        def initialize(self, spawn_pose=None):
            assert spawn_pose is not None
            self.pose = np.asarray(spawn_pose).astype(np.float32).astype(float)
            self.reset_count += 1
    results = []
    for reason in ('goal', 'collision', 'timeout'):
        with tempfile.TemporaryDirectory(prefix='newmetro_task_boundary_') as temp:
            driver = Driver.__new__(Driver)
            driver.native = navigation
            driver.goal = np.array([5.45, -1.85])
            driver.history = ScanHistory()
            for i in range(20):
                stamp = round(i*1e9/30)
                if i % 2 == 0: driver.history.append(np.ones(720), stamp)
                driver.history.sample(stamp)
            driver.fixed_self_masks = {name: np.arange(2000) % 3 == i for i, name in enumerate(('scan_01', 'scan_02'))}
            fixed_masks_before = {name: mask.copy() for name, mask in driver.fixed_self_masks.items()}
            driver.last_action_control = 100
            driver.last_action_scan_ns = 1000000000
            driver.cached_local_goal = np.array([9., 9.])
            driver.last_goal_time = 999.
            driver.latest_pedestrian_planar = np.ones((4, 4))
            driver.minimum_full_scan = .2
            policy = Policy.__new__(Policy)
            driver.policy = policy
            policy.pending = None
            policy.done, policy.reason = True, reason
            policy.steps, policy.updates, policy.episodes = 100, 1, 1
            policy.episode_steps, policy.episode_return = 90, -99.
            policy.smoke_pass = False
            policy.reward = OriginalReward(42)
            policy.reward.bump_num = 0 if reason=="collision" else 1
            policy.cfg = dict(total_timesteps=100000)
            policy.buffer = SimpleNamespace(dones=[1.], marker='completed_previous_episode')
            policy.checkpoint = lambda: None
            flushes = []
            robot = Robot()
            initial_pose = robot.pose.copy()
            world = SimpleNamespace(planar=lambda: Tensor(), simulator=SimpleNamespace(
                _scene=SimpleNamespace(write_data_to_sim=lambda: flushes.append('write'),
                                       update=lambda dt: flushes.append('update')),
                _sim=SimpleNamespace(forward=lambda: flushes.append('forward'))))
            bridge = Bridge.__new__(Bridge)
            bridge.driver, bridge.out = driver, Path(temp)
            bridge.m = SimpleNamespace(robot=robot, world=world, report={}, save=lambda: None)
            bridge.sampler = TaskSampler(BATCH, navigation, 700+len(results))
            if admission_failures:
                original_sample = bridge.sampler.sample
                remaining_failures = [admission_failures]
                def sample_with_failures(*args, **kwargs):
                    if remaining_failures[0]:
                        remaining_failures[0] -= 1
                        raise TaskAdmissionError('injected bounded admission failure', diagnostics={'injected': True})
                    return original_sample(*args, **kwargs)
                bridge.sampler.sample = sample_with_failures
            bridge.navigation, bridge.phase = navigation, 'long'
            bridge.start_wall = time.monotonic()
            bridge.config = dict(wall_limit_s=300.)
            bridge.task_admission_recoveries = 0
            bridge.reset_count, bridge.task_switch_count = 0, 0
            queried = []
            bridge.query_task_geometry = lambda task, out: queried.append(task['task_id'])
            assert bridge.handle_terminal(100, 3.3) is False
            expected_teleport = reason != 'goal' or admission_failures > 0
            assert robot.reset_count == int(expected_teleport)
            assert len(flushes) == (3 if expected_teleport else 0)
            if not expected_teleport: np.testing.assert_array_equal(robot.pose, initial_pose)
            else:
                selected = next(s for s in bridge.sampler.starts if s['id'] == bridge.active_task['spawn_candidate_id'])
                np.testing.assert_allclose(robot.pose, [*selected['xy_m'], selected['yaw_rad']], atol=1e-6)
            assert not driver.history.ready and len(driver.history.frames) == 0
            assert set(driver.fixed_self_masks) == set(fixed_masks_before)
            for name, mask in fixed_masks_before.items():
                np.testing.assert_array_equal(driver.fixed_self_masks[name], mask)
            assert driver.cached_local_goal is None and driver.last_goal_time is None
            assert driver.last_action_control is driver.last_action_scan_ns is None
            assert driver.latest_pedestrian_planar is None and math.isinf(driver.minimum_full_scan)
            assert driver.current['desired_v_m_s'] == driver.current['desired_w_rad_s'] == 0
            assert policy.pending is None and not policy.done and policy.reason is None
            assert policy.episode_steps == policy.episode_return == 0
            assert policy.buffer.dones == [1.]  # Preserve completed PPO transitions.
            assert policy.task_id == bridge.active_task['task_id']
            np.testing.assert_array_equal(driver.certified_goal, driver.goal)
            np.testing.assert_allclose(navigation.reference_poses[-1, :2], driver.goal)
            distance = float(np.linalg.norm(robot.pose[:2]-driver.goal))
            assert 4.2-1e-6 <= distance < 7.
            # First new action establishes a new distance baseline; it must not
            # close a transition from the old goal, history, or teleported pose.
            policy.agent = SimpleNamespace(select_action=lambda obs: (np.zeros(2), np.zeros(2), 0., 0.))
            action = policy.command(np.zeros(19202, dtype=np.float32), distance=distance,
                                    minimum=1., control=120, sim_time=4., reward_context={})
            assert not action['training_done'] and policy.pending['distance'] == distance
            assert policy.episode_steps == 0 and policy.buffer.dones == [1.]
            event = json.loads((Path(temp)/'RESET_EVENTS.jsonl').read_text())
            assert event['teleported'] == expected_teleport and event['pedestrians_unchanged']
            assert event['task_admission_recovery'] == (admission_failures > 0)
            assert bridge.task_admission_recoveries == int(admission_failures > 0)
            assert policy.steps == 100 and policy.updates == policy.episodes == 1
            admission_events=[json.loads(line) for line in (Path(temp)/'TASK_ADMISSION_EVENTS.jsonl').read_text().splitlines()]
            assert sum(e['status']=='REJECTED' for e in admission_events)==admission_failures
            results.append(dict(reason=reason, admission_failures=admission_failures, status='PASS_MOCKED_PHYSICS_BOUNDARY',
                                teleported=event['teleported'], distance_m=distance,
                                source_methods='actual copied bridge, driver, policy methods via AST'))
    return results


def main():
    begun = time.monotonic()
    protected_checkpoint = BATCH/'training/session_state.pt'
    checkpoint_before = hashlib.sha256(protected_checkpoint.read_bytes()).hexdigest() if protected_checkpoint.exists() else None
    for path in (BATCH/'code').glob('*.py'): compile(path.read_text(), str(path), 'exec')
    navigation = make_navigation(BATCH)
    manifest = json.loads((BATCH/'BASELINE_MANIFEST.json').read_text())
    original = Path(manifest['source_batch'])
    spec = importlib.util.spec_from_file_location('original_heading_reference', original/'code/drlvo_heading_planner.py')
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    reference = reference_module.HeadingPlanner(navigation.native, navigation.lower, navigation.upper, planning_tracking_margin=0.)
    sampler = TaskSampler(BATCH, navigation, 20261016)
    map_counts, routes = [], []
    # Compare the vectorized map-valid mask to literal original square loops.
    labels = navigation._labels
    rng = np.random.default_rng(42)
    for r, c in rng.integers([0, 0], labels.shape, size=(1000, 2)):
        expected = r >= 8 and c >= 8 and r+8 <= labels.shape[0] and c+8 <= labels.shape[1]
        if expected: expected = bool(np.all(labels[r-8:r+8, c-8:c+8] == 254))
        assert sampler.goal_valid[r, c] == expected, (r, c)
    for start in sampler.starts:
        pose = [*start['xy_m'], start['yaw_rad']]
        eligible = len(sampler.eligible_goals(pose[:2]))
        assert eligible > 0
        map_counts.append(dict(start_id=start['id'], eligible_goal_cells=eligible))
        task = sampler.sample(pose)
        poses = np.asarray(task['poses_xy_yaw'])
        assert 4.2 <= task['straight_distance_m'] < 7.
        assert all(navigation.sweep_clear(a, b) for a, b in zip(poses[:-1], poses[1:]))
        assert all(reference.sweep_clear(a, b) for a, b in zip(poses[:-1], poses[1:]))
        routes.append(dict(start_id=start['id'], goal_xy=task['goal_xy'],
                           distance_m=task['straight_distance_m'], poses=len(poses),
                           rejected=len(task['rejected_candidates']), status='PASS_STATIC_ORIENTED_ROUTE'))
        print(json.dumps(routes[-1]), flush=True)
    # Exact interval boundaries and impossible-goal behavior.
    probe = TaskSampler.__new__(TaskSampler)
    probe.goal_xy = np.array([[4.2, 0.], [7., 0.], [4.19999, 0.], [6.99999, 0.]])
    assert probe.eligible_goals([0., 0.]).tolist() == [0, 3]
    try: sampler.sample([1000., 1000., 0.])
    except TaskSamplingError: pass
    else: raise AssertionError('Impossible task accepted')
    state = sampler.state()
    a = sampler.sample([*sampler.starts[0]['xy_m'], sampler.starts[0]['yaw_rad']], require_route=False)
    sampler.restore(state)
    b = sampler.sample([*sampler.starts[0]['xy_m'], sampler.starts[0]['yaw_rad']], require_route=False)
    assert a == b
    # No straight-line fallback after planner rejection; old reference restored.
    state = sampler.state()
    original_astar = navigation.astar_path
    reference = navigation.reference_poses
    navigation.astar_path = lambda *a, **kw: None
    try:
        sampler.sample([*sampler.starts[0]['xy_m'], sampler.starts[0]['yaw_rad']])
    except TaskSamplingError: pass
    else: raise AssertionError('Planner rejection accepted')
    finally:
        navigation.astar_path = original_astar
        sampler.restore(state)
    assert navigation.reference_poses is reference
    boundaries = check_boundaries(navigation)
    for name, digest in manifest['source_hashes'].items():
        assert hashlib.sha256((original/name).read_bytes()).hexdigest() == digest, name
    checkpoint_after = hashlib.sha256(protected_checkpoint.read_bytes()).hexdigest() if protected_checkpoint.exists() else None
    assert checkpoint_before == checkpoint_after, 'CPU task check modified protected checkpoint'
    report = dict(status='OFFLINE_TASK_AND_MOCK_RESET_PASS', source_start_count=len(sampler.starts),
                  map_valid_goal_cells=len(sampler.goal_xy), goal_counts=map_counts, routes=routes,
                  boundary_checks=boundaries, original_neighborhood_comparison_cases=1000,
                  interval_boundary_check='PASS', bounded_no_goal_rejection='PASS',
                  original_unoptimized_swept_geometry_recheck='PASS_ALL_11_ROUTES',
                  rng_state_replay='PASS', bounded_no_route_rejection='PASS',
                  original_batch_unchanged=True, failed_checkpoint_loaded=False,
                  real_simulator_reset='NOT_EXECUTED', runtime_PhysX_geometry='NOT_EXECUTED',
                  physical_contact_safety='UNVERIFIED', training_started=False,
                  elapsed_s=time.monotonic()-begun,
                  code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BATCH/'code').glob('*.py')})
    (BATCH/'OFFLINE_TASK_CHECK.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('code_hashes', 'routes', 'goal_counts')}, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
