"""Four-person SFM/AMP physics owner with a robot-only DRLVO PPO bridge."""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

from drlvo_heading_planner import HeadingPlanner
from episode_semantics import EnvironmentEpisodeFault
from newmetro_geometry import verify_low_height_route
from training_driver import TrainingPolicyDriver
from task_sampling import TaskSampler, TaskAdmissionError, terminal_requires_teleport


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SfmTrainingBridge:
    # Includes the first request, followed by at most three safe-spawn retries.
    TASK_ADMISSION_MAX_RECOVERY_SPAWNS = 3
    TASK_ADMISSION_TOTAL_BUDGET_S = 60.

    def __init__(self, m):
        self.m = m
        self.out = m.OUT
        self.batch = self.out.parents[1]
        self.config = json.loads((self.out / "ROBOT_CONTROL.json").read_text())
        self.phase = self.config["training_phase"]
        self.start_wall = time.monotonic()
        self.reset_count = 0
        self.task_switch_count = 0
        self.task_admission_recoveries = 0
        self.closed = False
        assert m.count == m.source["actor_count"] == 4
        assert m.args.sfm_preview and m.args.open_doors and m.args.isolated_training
        assert m.report["pedestrian_planner_mode"] == "SFM to AMP"
        assert m.report["doors"]["count"] == 15
        assert m.report["open_doors_map"]["all_patrol_segments_reachable"] == "NOT_EXECUTED_SFM_FORCE_ONLY"
        accepted = np.load(self.batch / "scene/open_doors_native_grid.npz")["labels"]
        observed = np.load(self.out / "open_doors_native_grid.npz")["labels"]
        overrides = json.loads((self.batch / "MAP_OVERRIDE.json").read_text())
        allowed = np.zeros_like(accepted, dtype=bool)
        allowed[42:54,129] = True
        allowed[41,206:213] = True
        for r,c in overrides["lower_block"]["row_col_cells"]: allowed[r,c] = True
        if np.any((accepted != observed) & ~allowed):
            raise EnvironmentEpisodeFault("unapproved_current_open_door_grid_changed")
        class RobotMap:
            resolution = m.native.resolution
            _bounds = m.native._bounds
            _labels = accepted.copy()
            pixel_to_world = staticmethod(m.native.pixel_to_world)
            world_to_pixel = staticmethod(m.native.world_to_pixel)
        self.goal = np.asarray(self.config["robot_goal"], dtype=float)
        bounds = json.loads((self.batch / "SIZE_REFERENCE.json").read_text())["robot_static_asset_bounds_relative_base_link"]
        self.navigation = HeadingPlanner(RobotMap(), bounds["min_xyz"], bounds["max_xyz"], planning_tracking_margin=0.)
        (self.out / "ROBOT_MAP_BINDING.json").write_text(json.dumps(dict(
            source=str(self.batch / "scene/open_doors_native_grid.npz"),
            sha256=digest(self.batch / "scene/open_doors_native_grid.npz"),
            user_override_cells=int(allowed.sum()), differences_from_native=int(np.count_nonzero(accepted != observed)),
            pedestrian_map_unchanged=True, heading_count=8, sampled_margin_m=.03, additional_tracking_margin_m=0.),indent=2))
        initial = m.robot.observe()
        start = np.array([*initial["root_position_m"][:2], initial["yaw_rad"]])
        self.navigation.current_yaw = float(start[2])
        if self.spawn_is_occupied(start):
            raise EnvironmentEpisodeFault("initial_user_spawn_overlaps_pedestrian_proxy")
        self.sampler = TaskSampler(self.batch, self.navigation, self.config['task_sampling_seed'])
        self.active_task = json.loads((self.out/'INITIAL_TASK.json').read_text())
        self.sampler.restore(self.active_task['sampler_state'])
        wanted = np.asarray(self.active_task['start_pose'])
        yaw_error = abs((start[2]-wanted[2]+math.pi)%(2*math.pi)-math.pi)
        if np.linalg.norm(start[:2]-wanted[:2]) > 1e-4 or yaw_error > 1e-4:
            raise EnvironmentEpisodeFault('random_task_initial_pose_mismatch')
        poses = np.asarray(self.active_task['poses_xy_yaw'], dtype=float)
        connector = self.navigation.connector(start, poses[0], extra_margin=0.)
        if connector is None or not all(self.navigation.sweep_clear(a, b, extra_margin=0.)
                                        for a, b in zip(poses[:-1], poses[1:])):
            raise EnvironmentEpisodeFault("current_robot_reference_footprint_rejected")
        poses = np.concatenate((connector, poses[1:]))
        if np.linalg.norm(poses[-1, :2] - self.goal) > 1e-6:
            raise EnvironmentEpisodeFault("robot_reference_goal_mismatch")
        self.navigation.reference_poses = poses.copy()
        self.navigation.reference_only = True
        self.navigation.last_poses = poses.copy()
        import omni.physx
        query = omni.physx.get_physx_scene_query_interface()
        verify_low_height_route(m.native, self.navigation, poses, query, self.out)
        (self.out / "ROBOT_GLOBAL_ROUTE.json").write_text(json.dumps({
            "start_pose": start.tolist(), "goal_xy": self.goal.tolist(),
            "poses": poses.tolist(), "static_footprint_margin_m": .03,
            "tracking_margin_m": 0., "source_grid_sha256":
                digest(self.batch / "scene/open_doors_native_grid.npz"),
            "physical_contact_safety": "UNVERIFIED"}, indent=2) + "\n")
        self.driver = TrainingPolicyDriver(self.out, m.robot, self.navigation,
                                           m.count, self.goal, query)
        self.active_task['poses_xy_yaw'] = poses.tolist()
        self.driver.install_task(self.active_task)
        self.driver.policy.task_state_provider = lambda: dict(sampler=self.sampler.state(), active_task=self.active_task)
        self.start_steps = self.driver.policy.steps
        if self.phase == "smoke":
            assert self.start_steps == self.driver.policy.updates == len(self.driver.policy.buffer) == 0
        m.report.update(training=True, training_phase=self.phase,
            training_goal_xy=self.goal.tolist(), training_source="four original SFM routes; frozen AMP; real EVT wheels",
            sensor_training="two actual PhysX LiDARs, 2000 beams each, 15 Hz, 0.1..8 m",
            pedestrian_observation="oracle current physical velocity; predicted Gazebo perception NOT_REPRODUCED",
            manual_teleop_disabled=True, initial_steps=self.start_steps,
            initial_updates=self.driver.policy.updates, contact_safety="UNVERIFIED",
            status="TRAINING_PHYSICAL_PREFLIGHT_PASSED")
        loaded = {"entry": inspect.getfile(m), "bridge": __file__,
                  "sfm": inspect.getfile(sys.modules["sfm_amp_preview"]),
                  "driver": inspect.getfile(sys.modules["training_driver"]),
                  "policy": inspect.getfile(sys.modules["training_policy"]),
                  "wheel_adapter": inspect.getfile(m.RobotPreviewAdapter),
                  "native_ppo": inspect.getfile(sys.modules["native_ppo"]),
                  "reward_logic": inspect.getfile(sys.modules["reward_logic"]),
                  "python": sys.executable, "pid": os.getpid(),
                  "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]}
        loaded["sha256"] = {key: digest(value) for key, value in loaded.items()
                            if key in ("entry", "bridge", "sfm", "driver", "policy", "wheel_adapter", "native_ppo", "reward_logic")}
        (self.out / "LOADED_PATHS.json").write_text(json.dumps(loaded, indent=2) + "\n")
        m.save()

    def should_stop(self, control):
        request = self.out / "VIEW_REQUEST.json"
        if request.exists() and json.loads(request.read_text()).get("close"):
            self.driver.policy.interruption_reason = "external_close_request"
            self.m.report["status"] = "EXTERNALLY_INTERRUPTED"
            return True
        if time.monotonic() - self.start_wall > self.config["wall_limit_s"]:
            self.driver.policy.interruption_reason = "smoke_wall_limit"
            self.m.report["status"] = "SMOKE_WALL_LIMIT"
            return True
        return False

    def command(self, control, sim_time, pedestrian_planar):
        result = self.driver.command(control, sim_time, pedestrian_planar)
        p = self.driver.policy
        self.m.report.update(training_total_steps=p.steps, training_updates=p.updates,
            training_episodes=p.episodes, training_last_done_reason=p.reason,
            physical_control=control, contact_safety="UNVERIFIED")
        return result

    def post_physics(self, control, sim_time, observed):
        self.driver.post_physics(control, sim_time)
        p = self.driver.policy
        self.m.report.update(training_total_steps=p.steps, training_updates=p.updates,
            real_scan_history_frames=len(set(self.driver.history.source_timestamps_ns)),
            history_samples=len(self.driver.history.frames), history_sampling_hz=20,
            physical_control=control, physical_sim_time=sim_time,
            robot_actual_forward_m_s=observed["physical_forward_velocity_m_s"],
            robot_actual_wheels_rad_s=observed["wheel_actual_velocity_rad_s"])

    def handle_terminal(self, control, sim_time):
        m, p = self.m, self.driver.policy
        p.checkpoint()
        m.robot.stop()
        m.report.update(training_total_steps=p.steps, training_updates=p.updates,
                        training_episodes=p.episodes, training_last_done_reason=p.reason)
        if self.phase == "smoke" and p.smoke_pass and p.steps >= 512 and p.updates >= 1:
            before = m.world.planar().detach().cpu().numpy().copy()
            selected = self.choose_clear_spawn()
            self._verify_spawn([*selected['xy_m'], selected['yaw_rad']], 'smoke_final_reset')
            m.robot.initialize(spawn_pose=[*selected['xy_m'], selected['yaw_rad']])
            m.world.simulator._scene.write_data_to_sim()
            m.world.simulator._sim.forward()
            m.world.simulator._scene.update(dt=0.)
            after = m.world.planar().detach().cpu().numpy().copy()
            if not np.array_equal(before, after):
                raise EnvironmentEpisodeFault("smoke_robot_reset_modified_human_state")
            self.reset_count += 1
            m.report["in_process_robot_resets"] = self.reset_count
            m.report["smoke_post_update_robot_reset"] = "PASS"
            m.report["status"] = "DRLVO_REAL512_NATIVE_PPO_SMOKE_PASS"
            m.save()
            return True
        if p.steps >= p.cfg["total_timesteps"]:
            m.report["status"] = "DRLVO_CONFIGURED_BUDGET_REACHED"
            m.save()
            return True
        if p.reason not in ("collision", "goal", "timeout"):
            m.report["status"] = "TRAINING_TERMINAL_DIAGNOSTIC_STOP"
            m.save()
            return True
        reason = p.reason
        before = m.world.planar().detach().cpu().numpy().copy()
        teleport = terminal_requires_teleport(reason)
        current = m.robot.observe()
        requested_pose = np.array([*current['root_position_m'][:2], current['yaw_rad']])
        task, selected, requested_pose, recovered = self.sample_boundary_task(
            requested_pose, reason=reason, control=control, sim_time=sim_time)
        teleport = teleport or recovered
        task['spawn_candidate_id'] = None if selected is None else selected['id']
        task_dir = self.out/'tasks'/f"task_{task['task_id']:06d}"
        task_dir.mkdir(parents=True, exist_ok=False)
        self.query_task_geometry(task, task_dir)
        if teleport:
            m.robot.initialize(spawn_pose=requested_pose)
            m.world.simulator._scene.write_data_to_sim()
            m.world.simulator._sim.forward()
            m.world.simulator._scene.update(dt=0.)
            self.reset_count += 1
        actual = m.robot.observe()
        actual_pose = np.array([*actual['root_position_m'][:2], actual['yaw_rad']])
        if np.linalg.norm(actual_pose[:2]-requested_pose[:2]) > 1e-4:
            raise EnvironmentEpisodeFault('task_boundary_robot_position_mismatch')
        yaw_error = abs((actual_pose[2]-requested_pose[2]+math.pi)%(2*math.pi)-math.pi)
        if yaw_error > 1e-4:
            raise EnvironmentEpisodeFault('task_boundary_robot_yaw_mismatch')
        if not teleport and not np.array_equal(actual_pose, requested_pose):
            raise EnvironmentEpisodeFault('successful_task_modified_robot_pose')
        after = m.world.planar().detach().cpu().numpy().copy()
        if not np.array_equal(before, after):
            raise EnvironmentEpisodeFault('task_boundary_modified_human_state')
        self.goal = np.asarray(task['goal_xy'], dtype=float)
        self.active_task = task
        self.driver.reset_episode(task=task)
        if recovered:
            self.task_admission_recoveries += 1
        self.navigation.current_yaw = float(actual_pose[2])
        self.task_switch_count += 1
        task['actual_boundary_pose'] = actual_pose.tolist()
        (task_dir/'TASK.json').write_text(json.dumps(task, indent=2)+'\n')
        (self.out/'TASK_STATE.json').write_text(json.dumps(dict(sampler=self.sampler.state(), active_task=task), indent=2)+'\n')
        with (self.out/'RESET_EVENTS.jsonl').open('a') as f:
            f.write(json.dumps(dict(control=control, sim_time=sim_time, reason=reason,
                task_id=task['task_id'], spawn_candidate_id=task['spawn_candidate_id'],
                teleported=teleport, actual_pose=actual_pose.tolist(), goal_xy=task['goal_xy'],
                task_admission_recovery=recovered,
                reset_cause='task_admission_recovery' if recovered else reason,
                robot_reset_count=m.robot.reset_count, pedestrians_unchanged=True,
                history_cleared=True, old_pending_action=False))+'\n')
        p.checkpoint()
        m.report.update(status='TRAINING_RUNNING_AFTER_RANDOM_TASK_SWITCH',
                        in_process_robot_resets=self.reset_count, task_switches=self.task_switch_count,
                        training_goal_xy=self.goal.tolist(), current_task_id=task['task_id'],
                        task_admission_recoveries=self.task_admission_recoveries,
                        last_boundary_teleported=teleport)
        m.save()
        return False

    def sample_boundary_task(self, current_pose, *, reason, control, sim_time):
        """Admit before moving. Only bounded sampling failures permit recovery."""
        deadline = min(time.monotonic()+self.TASK_ADMISSION_TOTAL_BUDGET_S,
                       self.start_wall+self.config['wall_limit_s'])
        last_error = None
        for attempt in range(self.TASK_ADMISSION_MAX_RECOVERY_SPAWNS+1):
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                break
            rng_before = self.sampler.state()
            selected = self.choose_clear_spawn() if attempt or terminal_requires_teleport(reason) else None
            pose = np.asarray(current_pose if selected is None else [*selected['xy_m'], selected['yaw_rad']], dtype=float)
            event = dict(control=control, sim_time=sim_time, terminal_reason=reason,
                         attempt=attempt+1, recovery_attempt=attempt>0,
                         requested_pose=pose.tolist(),
                         spawn_candidate_id=None if selected is None else selected['id'],
                         sampler_before=rng_before)
            try:
                task = self.sampler.sample(pose, time_budget_s=max(0.,deadline-time.monotonic()))
            except TaskAdmissionError as error:
                last_error = error
                event.update(status='REJECTED', error=str(error), diagnostics=error.diagnostics,
                             sampler_after=self.sampler.state())
                with (self.out/'TASK_ADMISSION_EVENTS.jsonl').open('a') as file:
                    file.write(json.dumps(event)+'\n')
                continue
            event.update(status='ADMITTED_PENDING_GEOMETRY', task_id=task['task_id'],
                         sampler_after=self.sampler.state())
            with (self.out/'TASK_ADMISSION_EVENTS.jsonl').open('a') as file:
                file.write(json.dumps(event)+'\n')
            task['boundary_admission'] = dict(attempt=attempt+1, recovery=attempt>0,
                original_terminal_reason=reason, budget_s=self.TASK_ADMISSION_TOTAL_BUDGET_S)
            return task, selected, pose, attempt>0
        # No robot movement, reset, episode or transition is manufactured here.
        self.driver.policy.interruption_reason = 'task_admission_recovery_exhausted'
        self.m.report.update(status='TASK_ADMISSION_RECOVERY_EXHAUSTED',
            task_admission_last_error=None if last_error is None else str(last_error))
        with (self.out/'TASK_ADMISSION_EVENTS.jsonl').open('a') as file:
            file.write(json.dumps(dict(control=control,sim_time=sim_time,status='EXHAUSTED',
                sampler_after=self.sampler.state(), terminal_reason=reason))+'\n')
        self.driver.policy.checkpoint()
        self.m.save()
        raise TaskAdmissionError('Task admission recovery exhausted; stopped without reset',
            diagnostics=dict(last_error=None if last_error is None else last_error.diagnostics,
                             sampler_state=self.sampler.state())) from last_error

    def spawn_is_occupied(self, pose):
        measured = self.m.world.planar().detach().cpu().numpy()
        yaw = float(pose[2])
        rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        local = (measured[:,:2]-np.asarray(pose[:2]))@rotation
        outside = np.maximum(np.maximum(self.navigation.lower[:2]-local, local-self.navigation.upper[:2]), 0.)
        # Existing SIZE_REFERENCE proxy only; full humanoid contact safety is not claimed.
        return bool(np.any(np.linalg.norm(outside,axis=1) <= .4510317735765819/2+.03))

    def choose_clear_spawn(self):
        for attempt in range(32):
            selected = self.sampler.random_start()
            pose = [*selected['xy_m'], selected['yaw_rad']]
            if not self.spawn_is_occupied(pose):return selected
            with (self.out/'SPAWN_REJECTIONS.jsonl').open('a') as file:
                file.write(json.dumps(dict(candidate_id=selected['id'],reason='current_humanoid_circle_proxy_overlap',attempt=attempt+1))+'\n')
        raise EnvironmentEpisodeFault('no_unoccupied_user_spawn_within_32_attempts')

    def query_task_geometry(self, task, output):
        import omni.physx
        verify_low_height_route(self.m.native, self.navigation,
            np.asarray(task['poses_xy_yaw'], dtype=float),
            omni.physx.get_physx_scene_query_interface(), output)

    def _verify_spawn(self, pose, name):
        output = self.out/'tasks'/name
        output.mkdir(parents=True, exist_ok=False)
        self.query_task_geometry(dict(poses_xy_yaw=[list(pose), list(pose)]), output)

    def close(self, failed=False):
        if self.closed:
            return
        self.closed = True
        p = self.driver.policy
        p.close(failed=failed)
        self.driver.policy_log.close()
        self.driver.scan_log.close()
        checkpoint = self.batch / "training/session_state.pt"
        self.m.report.update(training_total_steps=p.steps, training_updates=p.updates,
            training_episodes=p.episodes, in_process_robot_resets=self.reset_count,
            accepted_checkpoint=str(checkpoint) if checkpoint.exists() else "NOT_OBSERVED",
            accepted_checkpoint_sha256=digest(checkpoint) if checkpoint.exists() else "NOT_OBSERVED")
