"""One persistent Isaac owner for successive frozen-policy map goals."""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from demo_driver import DemoDriver
from drlvo_heading_planner import HeadingPlanner
from episode_semantics import EnvironmentEpisodeFault
from newmetro_geometry import verify_low_height_route


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


class SfmDemoBridge:
    def __init__(self, m):
        self.m = m
        self.out = m.OUT
        self.batch = self.out.parents[1]
        self.config = json.loads((self.out / 'ROBOT_CONTROL.json').read_text())
        self.start_wall = time.monotonic()
        self.closed = False
        self.state = 'waiting'
        self.status = 'DEMO_WAITING_FOR_GOAL'
        self.sequence = 0
        self.goal = None
        self.settle_since = None
        self.completed_goals = 0
        self.last_subgoal_time = None
        self.subgoal_log = (self.out / 'SUBGOAL_EVENTS.jsonl').open('w', buffering=1)
        assert m.args.isolated_demo and m.args.sfm_preview and m.args.open_doors
        assert m.count == m.source['actor_count'] == 4
        assert m.report['pedestrian_planner_mode'] == 'SFM to AMP'
        with np.load(self.batch / 'scene/open_doors_native_grid.npz', allow_pickle=False) as data:
            accepted = data['labels'].copy()
        with np.load(self.out / 'open_doors_native_grid.npz', allow_pickle=False) as data:
            observed = data['labels'].copy()
        overrides=json.loads((self.batch/'MAP_OVERRIDE.json').read_text())
        allowed=np.zeros_like(accepted,dtype=bool);allowed[42:54,129]=True;allowed[41,206:213]=True
        for r,c in overrides['lower_block']['row_col_cells']:allowed[r,c]=True
        if np.any((accepted!=observed)&~allowed) or not np.array_equal(observed,m.native._labels):
            raise EnvironmentEpisodeFault('demo_grid_changed')
        from task_sampling import make_navigation
        self.navigation=make_navigation(self.batch)
        self.navigation.current_yaw=float(m.robot.observe()['yaw_rad'])
        self.map_hash=hashlib.sha256(accepted.tobytes()).hexdigest()
        np.savez_compressed(self.out/'robot_navigation_grid.npz',labels=accepted)
        atomic_json(self.out/'ROBOT_MAP_BINDING.json',dict(source=str(self.batch/'scene/open_doors_native_grid.npz'),sha256=digest(self.batch/'scene/open_doors_native_grid.npz'),allowed_native_differences=int(np.count_nonzero(accepted!=observed)),planning_tracking_margin_m=0.,sampled_margin_m=.03))
        import omni.physx
        self.query = omni.physx.get_physx_scene_query_interface()
        # Create the LiDAR and load frozen weights once; no goal is sent yet.
        self.driver = DemoDriver(self.out, m.robot, self.navigation, m.count,
                                 m.source['robot_goal_xy'], self.query)
        policy = self.driver.policy
        m.report.update(training=False, RL_training=False, demo=True, demo_policy='frozen_deterministic',
            demo_goal_xy=None, demo_checkpoint=policy.checkpoint,
            demo_checkpoint_sha256=policy.checksum, demo_actions=0,
            demo_completed_goals=0,
            demo_local_goal_lookahead_m=policy.cfg['local_goal_lookahead_m'],
            source_trained_lookahead_m=policy.trained_lookahead,
            sensor_training='two actual PhysX LiDARs, 2000 beams each, 15 Hz, 0.1..8 m',
            pedestrian_observation='oracle physical pedestrian velocity',
            manual_teleop_disabled=True, contact_safety='UNVERIFIED', status=self.status)
        loaded = {'entry': inspect.getfile(m), 'bridge': __file__,
                  'driver': inspect.getfile(sys.modules['demo_driver']),
                  'policy': inspect.getfile(sys.modules['demo_policy']),
                  'initial_alignment': inspect.getfile(sys.modules['first_subgoal_alignment']),
                  'python': sys.executable, 'pid': os.getpid()}
        loaded['sha256'] = {key: digest(value) for key, value in loaded.items()
                            if key in ('entry', 'bridge', 'driver', 'policy', 'initial_alignment')}
        atomic_json(self.out / 'LOADED_PATHS.json', loaded)
        m.save()

    def event(self, name, **extra):
        payload = {'schema': 'newmetro_drl_demo_goal_event/v1', 'event': name,
                   'sequence': self.sequence, 'control': self.m.report.get('actual_controls', 0),
                   'sim_time': self.m.report.get('sim_time', 0), 'goal_xy_m':
                   None if self.goal is None else self.goal.tolist(), **extra}
        atomic_json(self.out / 'GOAL_EVENT.json', payload)
        with (self.out / 'GOAL_EVENTS.jsonl').open('a') as stream:
            stream.write(json.dumps(payload) + '\n')

    def accept_request(self):
        path = self.out / 'GOAL_REQUEST.json'
        if not path.exists():
            return
        try:
            request = json.loads(path.read_text())
            sequence = int(request['sequence'])
            if sequence <= self.sequence:
                return
            self.sequence = sequence
            if request.get('schema') != 'newmetro_drl_demo_goal_request/v1':
                raise ValueError('Goal request schema differs')
            if request.get('map_labels_sha256') != self.map_hash:
                raise ValueError('Selected map differs from Isaac map')
            goal = np.asarray(request['goal_xy_m'], dtype=float)
            if goal.shape != (2,) or not np.isfinite(goal).all():
                raise ValueError('Goal must contain two finite coordinates')
            if self.state != 'waiting':
                raise ValueError('Robot is still following the previous goal')
            initial = self.m.robot.observe()
            start = np.array([*initial['root_position_m'][:2], initial['yaw_rad']])
            self.navigation.current_yaw = float(start[2])
            self.navigation.reference_poses=None
            self.navigation.reference_only=False
            route=self.navigation.astar_path(start[:2],goal,time_budget_s=6.)
            poses=self.navigation.last_poses
            if route is None or poses is None or np.linalg.norm(poses[-1,:2]-goal)>1e-6:
                raise ValueError('Selected goal has no admitted full-footprint route within 6 seconds')
            route_source='current_training_3x3_attachment_oriented_astar'
            verify_low_height_route(self.m.native, self.navigation, poses, self.query, self.out)
            self.goal = goal
            self.navigation.reference_poses = poses.copy()
            self.navigation.reference_only = True
            self.navigation.last_poses = poses.copy()
            self.driver.set_goal(goal, poses)
            self.last_subgoal_time = None
            self.state = 'active'
            self.status = 'DEMO_RUNNING'
            self.m.report.update(demo_goal_xy=goal.tolist(), demo_route_source=route_source,
                                 demo_last_goal_distance_m=float(np.linalg.norm(start[:2]-goal)),
                                 demo_goal_sequence=sequence, status=self.status)
            atomic_json(self.out / 'ROBOT_GLOBAL_ROUTE.json', {
                'sequence': sequence, 'start_pose': start.tolist(), 'goal_xy': goal.tolist(),
                'poses': poses.tolist(), 'route_source': route_source,
                'static_footprint_margin_m': .03, 'source_grid_sha256':
                digest(self.batch / 'scene/open_doors_native_grid.npz'),
                'physical_contact_safety': 'UNVERIFIED'})
            self.event('accepted', route_source=route_source)
            self.m.save()
        except (KeyError, TypeError, ValueError, RuntimeError, EnvironmentEpisodeFault) as exc:
            self.status = 'DEMO_WAITING_FOR_GOAL' if self.state == 'waiting' else self.status
            self.event('rejected', reason=str(exc))

    def should_stop(self, control):
        request = self.out / 'VIEW_REQUEST.json'
        if request.exists() and json.loads(request.read_text()).get('close'):
            self.m.report['status'] = 'DEMO_USER_CLOSED'
            return True
        if control >= int(self.config['max_controls']):
            self.m.report['status'] = 'DEMO_CONTROL_LIMIT'
            return True
        if time.monotonic() - self.start_wall > float(self.config['max_wall_seconds']):
            self.m.report['status'] = 'DEMO_WALL_LIMIT'
            return True
        self.accept_request()
        return False

    def command(self, control, sim_time, pedestrian_planar):
        if self.state != 'active':
            return dict(desired_v_m_s=0., desired_w_rad_s=0., training_done=False,
                        owner='DEMO_WAITING_FOR_GOAL' if self.state == 'waiting' else 'DEMO_SETTLING')
        result = self.driver.command(control, sim_time, pedestrian_planar)
        self.status = ('DEMO_ALIGNING_FIRST_SUBGOAL' if not self.driver.initial_alignment.done
                       else 'DEMO_RUNNING')
        self.m.report.update(status=self.status,
            demo_initial_alignment_state=result.get('initial_alignment_state',
                'aligned' if self.driver.initial_alignment.done else 'waiting_for_first_subgoal'),
            demo_initial_alignment_error_rad=result.get('initial_alignment_error_rad'),
            demo_initial_alignment_target_world_xy_m=(None if self.driver.initial_alignment.target_world is None
                else list(self.driver.initial_alignment.target_world)),
            demo_control_owner=result.get('owner'))
        publication = result.get('local_goal_publication_sim_time')
        if publication is not None and publication != self.last_subgoal_time:
            local = np.asarray(result['local_goal_base_xy'], dtype=float)
            obs = self.m.robot.observe()
            yaw = float(obs['yaw_rad'])
            rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            world = np.asarray(obs['root_position_m'][:2]) + rotation @ local
            lookahead = float(self.driver.policy.cfg['local_goal_lookahead_m'])
            self.m.report.update(demo_subgoal_base_xy_m=local.tolist(), demo_subgoal_world_xy_m=world.tolist(),
                demo_subgoal_publication_sim_time=publication)
            self.subgoal_log.write(json.dumps(dict(control=control, sim_time=sim_time,
                publication_sim_time=publication, base_xy_m=local.tolist(), world_xy_m=world.tolist(),
                robot_pose_xy_yaw=[*obs['root_position_m'][:2], yaw], lookahead_m=lookahead)) + '\n')
            self.last_subgoal_time = publication
        self.m.report.update(demo_actions=self.driver.policy.steps,
                             demo_last_goal_distance_m=result.get('goal_distance_m'),
                             demo_minimum_full_scan_m=result.get('minimum_full_scan_m'),
                             physical_control=control)
        return result

    def post_physics(self, control, sim_time, observed):
        self.driver.post_physics(control, sim_time)
        self.m.report.update(real_scan_history_frames=len(self.driver.history.frames),
            lidar_scan_count=int(self.driver.lidar.sequence),
            physical_control=control, physical_sim_time=sim_time,
            robot_actual_forward_m_s=observed['physical_forward_velocity_m_s'],
            robot_actual_wheels_rad_s=observed['wheel_actual_velocity_rad_s'])
        if self.state == 'settling':
            distance = float(np.linalg.norm(np.asarray(observed['root_position_m'][:2])-self.goal))
            stopped = abs(observed['physical_forward_velocity_m_s']) < .05 and abs(observed['physical_yaw_velocity_rad_s']) < .10
            if stopped and distance <= float(self.config['goal_tolerance_m']):
                if self.settle_since is None:
                    self.settle_since = sim_time
                if sim_time - self.settle_since >= .5:
                    self.state = 'waiting'
                    self.status = 'DEMO_WAITING_FOR_NEXT_GOAL'
                    self.completed_goals += 1
                    self.m.report.update(status=self.status, demo_completed_goals=self.completed_goals,
                                         demo_last_goal_distance_m=distance)
                    self.event('ready', final_goal_distance_m=distance,
                               robot_stopped=True, arrival_dwell_s=sim_time-self.settle_since)
                    self.m.save()
            else:
                self.settle_since = None

    def handle_terminal(self, control, sim_time):
        self.m.robot.stop()
        reason = self.driver.policy.reason
        self.m.report.update(demo_done_reason=reason, demo_actions=self.driver.policy.steps,
                             physical_control=control, physical_sim_time=sim_time)
        if reason == 'goal':
            self.state = 'settling'
            self.status = 'DEMO_GOAL_SETTLING'
            self.settle_since = None
            self.m.save()
            return False
        self.status = 'DEMO_' + str(reason).upper()
        self.m.report['status'] = self.status
        self.event('failed', reason=reason)
        self.m.save()
        return True

    def close(self, failed=False):
        if self.closed:
            return
        self.closed = True
        frozen=self.driver.policy.verify_frozen()
        atomic_json(self.out/'FROZEN_POLICY_CHECK.json',dict(unchanged=True,model_fingerprint=frozen,optimizer_updates=0,training=False))
        self.driver.policy_log.close()
        self.driver.scan_log.close()
        self.subgoal_log.close()
        self.m.report['demo_actions'] = self.driver.policy.steps
