"""CPU-only feedback/control tests. No Isaac, checkpoint, or GPU is loaded."""
import io
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from first_subgoal_alignment import FirstSubgoalAlignment

# Exercise the real demo command path with a counting policy, without torch.
driver_spec = importlib.util.spec_from_file_location(
    '_alignment_test_demo_driver', Path(__file__).with_name('demo_driver.py'))
driver_module = importlib.util.module_from_spec(driver_spec)
with patch.dict(sys.modules, {'demo_policy': SimpleNamespace(DemoPolicy=None)}):
    driver_spec.loader.exec_module(driver_module)
DemoDriver = driver_module.DemoDriver
from drlvo_observation_adapter import ScanHistory
from local_goal import ActionClock


class CountingPolicy:
    def __init__(self):
        self.cfg = {'local_goal_lookahead_m': .5}
        self.steps = 0
        self.inputs = []
        self.task_id = 0

    def command(self, observation, **kwargs):
        self.steps += 1
        self.inputs.append(observation.copy())
        return dict(desired_v_m_s=.2, desired_w_rad_s=0.,
                    owner='FROZEN_DRLVO_DEMO', training_done=False)

    def reset_episode(self):
        pass


def warm_history(driver):
    for n in range(10):
        ns = round(n*1e9/15)
        driver.history.append(np.full(720, 3., dtype=np.float32), ns)
        driver.history.sample(ns)
    assert driver.history.ready


def make_driver(out):
    driver = DemoDriver.__new__(DemoDriver)
    driver.out = Path(out)
    driver.robot_state = dict(root_position_m=[0., 0., 0.], yaw_rad=0.,
                              physical_forward_velocity_m_s=0., physical_yaw_velocity_rad_s=0.)
    driver.robot = SimpleNamespace(observe=lambda: driver.robot_state.copy())
    driver.plan_calls = []

    def plan(start, goal):
        driver.plan_calls.append((start.copy(), goal.copy()))
        return np.array([start, goal])

    driver.native = SimpleNamespace(astar_path=plan, current_yaw=0., _labels=np.zeros((2, 2)))
    driver.goal = np.array([0., 4.])
    driver.certified_reference = None
    driver.certified_goal = driver.goal.copy()
    driver.certified_map = driver.native._labels.copy()
    driver.policy = CountingPolicy()
    driver.initial_alignment = FirstSubgoalAlignment()
    driver.action_clock = ActionClock(.05)
    driver.history = ScanHistory()
    driver.last_action_control = driver.last_action_scan_ns = None
    driver.cached_local_goal = driver.last_goal_time = None
    driver.minimum_full_scan = driver.minimum_reward_scan = 3.
    driver.policy_log = io.StringIO()
    driver.current = dict(desired_v_m_s=0., desired_w_rad_s=0., owner='SENSOR_WARMUP')
    (driver.out/'policy_inputs').mkdir()
    return driver


class AlignmentTests(unittest.TestCase):
    def test_shortest_direction_and_wraparound(self):
        for yaw_deg, goal_deg, expected_sign in [(0, 90, 1), (0, -90, -1),
                                                (179, -170, 1), (-179, 170, -1),
                                                (0, 180, 1), (0, -180, -1)]:
            with self.subTest(yaw=yaw_deg, goal=goal_deg):
                yaw, bearing = math.radians(yaw_deg), math.radians(goal_deg)
                local_angle = bearing-yaw
                align = FirstSubgoalAlignment()
                align.start([math.cos(local_angle), math.sin(local_angle)], [0, 0, yaw])
                command = align.command([0, 0, yaw], 0, 1.)
                self.assertEqual(command['desired_v_m_s'], 0.)
                self.assertGreater(expected_sign*command['desired_w_rad_s'], 0.)
                self.assertLessEqual(abs(command['desired_w_rad_s']), .8)
                self.assertLessEqual(abs(command['initial_alignment_error_rad']), math.pi)

    def test_dwell_requires_heading_and_actual_angular_stop(self):
        align = FirstSubgoalAlignment()
        align.start([1, 0], [0, 0, 0])
        for t in (0., .2, .4):
            self.assertFalse(align.command([0, 0, 0], .2, t)['inference_started'])
            self.assertFalse(align.done)
        align.command([0, 0, 0], 0., .5)
        align.command([0, 0, .2], 0., .6)  # leaving tolerance resets dwell
        align.command([0, 0, 0], 0., .7)
        align.command([0, 0, 0], 0., .8)
        self.assertFalse(align.done)
        result = align.command([0, 0, 0], 0., .9)
        self.assertTrue(align.done)
        self.assertEqual(result['desired_w_rad_s'], 0.)
        self.assertFalse(result['inference_started'])

    def test_target_stays_fixed_in_world(self):
        align = FirstSubgoalAlignment()
        align.start([0, .5], [2., 3., 0.])
        result = align.command([2., 3., math.pi/2], 0., 1.)
        np.testing.assert_allclose(result['local_goal_base_xy'], [.5, 0], atol=1e-10)
        self.assertEqual(result['initial_alignment_target_world_xy_m'], [2., 3.5])

    def test_degenerate_and_invalid_inputs(self):
        align = FirstSubgoalAlignment()
        align.start([0, 0], [0, 0, 1.])
        self.assertEqual(align.command([0, 0, 1.], 0, 0)['desired_w_rad_s'], 0.)
        for cfg in ({'max_w_rad_s': float('nan')}, {'kp': 0}, {'heading_tolerance_rad': 4}):
            with self.assertRaises(ValueError):
                FirstSubgoalAlignment(cfg)
        with self.assertRaises(ValueError):
            align.command([0, 0, float('nan')], 0, 0)

    def test_driver_no_inference_until_alignment_then_normal_cadence(self):
        with tempfile.TemporaryDirectory() as out:
            driver = make_driver(out)
            pedestrians = np.zeros((0, 4))
            self.assertEqual(driver.command(0, 0., pedestrians)['owner'], 'SENSOR_WARMUP')
            self.assertEqual(driver.policy.steps, 0)
            warm_history(driver)
            release_control = None
            for control in range(30, 450):
                result = driver.command(control, control/30., pedestrians)
                if result['owner'] == 'FROZEN_DRLVO_DEMO':
                    release_control = control
                    break
                self.assertEqual(result['desired_v_m_s'], 0.)
                self.assertIsNone(result['raw_action'])
                self.assertEqual(driver.policy.steps, 0)
                self.assertIsNone(driver.action_clock.origin)
                self.assertIsNone(driver.last_action_control)
                self.assertEqual(len(driver.plan_calls), 1)
                self.assertFalse(list((driver.out/'policy_inputs').iterdir()))
                driver.robot_state['yaw_rad'] += result['desired_w_rad_s']/30.
                driver.robot_state['physical_yaw_velocity_rad_s'] = result['desired_w_rad_s']
            self.assertIsNotNone(release_control, 'controller failed to release inference')
            self.assertLessEqual(abs(math.pi/2-driver.robot_state['yaw_rad']), math.radians(5))
            self.assertEqual(driver.policy.steps, 1)
            # Real observation builder sees a goal in the CURRENT robot frame.
            np.testing.assert_allclose(driver.policy.inputs[0][-2:]*2,
                                       driver.cached_local_goal, atol=1e-6)
            self.assertLess(abs(driver.policy.inputs[0][-1]*2), .05)
            for control in range(release_control+1, release_control+12):
                result = driver.command(control, control/30., pedestrians)
                self.assertEqual(result['owner'], 'FROZEN_DRLVO_DEMO')
            self.assertGreater(driver.policy.steps, 3)
            self.assertTrue(driver.initial_alignment.done)

            steps = driver.policy.steps
            driver.set_goal([4., 0.], [[0., 0., driver.robot_state['yaw_rad']], [4., 0., 0.]])
            self.assertFalse(driver.initial_alignment.done)
            self.assertIsNone(driver.initial_alignment.target_world)
            self.assertIsNone(driver.last_action_control)
            self.assertEqual(driver.command(500, 500/30., pedestrians)['owner'], 'SENSOR_WARMUP')
            warm_history(driver)
            result = driver.command(501, 501/30., pedestrians)
            self.assertEqual(result['owner'], 'INITIAL_SUBGOAL_ALIGNMENT')
            self.assertLess(result['desired_w_rad_s'], 0.)
            self.assertEqual(driver.policy.steps, steps)

    def test_already_aligned_still_skips_inference_during_dwell(self):
        with tempfile.TemporaryDirectory() as out:
            driver = make_driver(out)
            driver.goal = np.array([4., 0.])
            warm_history(driver)
            for control in range(30, 36):
                result = driver.command(control, control/30., np.zeros((0, 4)))
                self.assertEqual(result['desired_v_m_s'], 0.)
                self.assertEqual(result['desired_w_rad_s'], 0.)
                self.assertEqual(driver.policy.steps, 0)
            self.assertTrue(driver.initial_alignment.done)
            driver.command(36, 36/30., np.zeros((0, 4)))
            self.assertEqual(driver.policy.steps, 1)


if __name__ == '__main__':
    unittest.main()
