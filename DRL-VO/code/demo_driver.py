"""Causal post-physics sensors and frozen inference. Never calls world.step."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from demo_policy import DemoPolicy
from drlvo_observation_adapter import ScanHistory,build_observation,oracle_pedestrian_velocity_map,rotate_world_to_base,sensor_ranges_to_base,dual_lidar_to_legacy_scan
from lidar_preview_adapter import LidarPreviewAdapter
from episode_semantics import EnvironmentEpisodeFault
from local_goal import ActionClock,local_goal_base
from first_subgoal_alignment import FirstSubgoalAlignment
class DemoDriver:
 def __init__(self,out,robot,native,count,goal,query):
  self.out=Path(out);self.robot=robot;self.native=native;self.goal=np.asarray(goal)
  self.certified_reference=np.asarray(native.reference_poses[:,:2]).copy() if getattr(native,'reference_poses',None) is not None else None;self.certified_goal=self.goal.copy();self.certified_map=native._labels.copy()
  self.policy=DemoPolicy(self.out);self.action_clock=ActionClock(.05);self.history=ScanHistory()
  run_config=json.loads((self.out/'ROBOT_CONTROL.json').read_text())
  self.initial_alignment=FirstSubgoalAlignment(run_config.get('first_subgoal_alignment'))
  self.lidar=LidarPreviewAdapter(query,pedestrian_roots=[f'/World/envs/env_{i}/Robot' for i in range(count)],range_min_m=.1,range_max_m=8.)
  self_mask_path=Path(__file__).resolve().parents[1]/'audit'/'LIDAR_SELF_MASK.json'
  if not self_mask_path.is_file():raise RuntimeError(f'Calibrated dual-LiDAR self mask missing: {self_mask_path}')
  self_mask=json.loads(self_mask_path.read_text())
  if self_mask.get('schema')!='newmetro.dual_lidar_fixed_self_mask.v1' or self_mask.get('status')!='CALIBRATION_PASS':
   raise RuntimeError('Dual-LiDAR self mask is not a completed calibration')
  run_source_path=self.out/'SOURCE.json'
  if not run_source_path.is_file():raise RuntimeError('Run SOURCE.json is required to bind the calibrated robot identity')
  run_source=json.loads(run_source_path.read_text())
  if (self_mask.get('robot_usd_sha256')!=run_source.get('robot_sha256') or
      self_mask.get('robot_root_path')!='/World/NavigationRobot'):
   raise RuntimeError('Calibrated self mask robot identity differs from this experiment')
  sensor=self_mask.get('sensor',{})
  if (sensor.get('beam_count')!=self.lidar.beam_count or
      not math.isclose(float(sensor.get('angle_min_rad',math.nan)),-math.pi,abs_tol=1e-12) or
      not math.isclose(float(sensor.get('angle_increment_rad',math.nan)),2*math.pi/self.lidar.beam_count,abs_tol=1e-12) or
      not math.isclose(float(sensor.get('range_min_m',math.nan)),self.lidar.range_min,abs_tol=1e-12) or
      not math.isclose(float(sensor.get('range_max_m',math.nan)),self.lidar.range_max,abs_tol=1e-12) or
      not math.isclose(float(sensor.get('rate_hz',math.nan)),self.lidar.rate_hz,abs_tol=1e-12)):
   raise RuntimeError('Calibrated self mask scan parameters differ from this experiment')
  calibrated_mounts=sensor.get('mounts',[])
  if len(calibrated_mounts)!=len(self.lidar.mounts):raise RuntimeError('Calibrated sensor count differs')
  for (name,xyz,yaw,_directions),entry in zip(self.lidar.mounts,calibrated_mounts,strict=True):
   if (entry.get('name')!=name or not np.allclose(xyz,np.asarray(entry.get('xyz_m'),dtype=float),atol=1e-9,rtol=0) or
       not math.isclose(float(entry.get('yaw_rad',math.nan)),yaw,abs_tol=1e-9)):
    raise RuntimeError(f'Calibrated mount pose differs for {name}')
  self.fixed_self_masks={}
  for name in ('scan_01','scan_02'):
   indices=self_mask.get('fixed_mask_beam_indices',{}).get(name)
   if not isinstance(indices,list) or not indices or any(type(i) is not int or i<0 or i>=self.lidar.beam_count for i in indices):
    raise RuntimeError(f'Invalid fixed self-mask beam list for {name}')
   fixed=np.zeros(self.lidar.beam_count,dtype=bool);fixed[indices]=True;self.fixed_self_masks[name]=fixed
  self.self_mask_sha256=hashlib.sha256(self_mask_path.read_bytes()).hexdigest()
  (self.out/'LIDAR_SELF_MASK_BINDING.json').write_text(json.dumps(dict(
   status='LOADED_AND_BOUND',path=str(self_mask_path),sha256=self.self_mask_sha256,
   robot_usd_sha256=run_source['robot_sha256'],
   fixed_mask_beam_counts={name:int(mask.sum()) for name,mask in self.fixed_self_masks.items()},
   calibration_status=self_mask['status']),indent=2)+'\n')
  self.last_action_control=None;self.last_action_scan_ns=None;self.cached_local_goal=None;self.last_goal_time=None;self.latest_pedestrian_planar=None;self.current=dict(raw_action=None,clipped_action=None,desired_v_m_s=0.,desired_w_rad_s=0.,owner='SENSOR_WARMUP',external_safety_override=False)
  self.policy_log=(self.out/'policy.jsonl').open('w',buffering=1);self.scan_log=(self.out/'lidar.jsonl').open('w',buffering=1);(self.out/'scans').mkdir();(self.out/'policy_inputs').mkdir()
 def post_physics(self,control,time):
  robot=self.robot.observe();packet=self.lidar.scan(robot['root_position_m'],robot['root_quaternion_xyzw'],time)
  if packet is None:
   self.history.sample(round(time*1e9))
   return
  ranges=[];angles=[]
  for name in ('scan_01','scan_02'):
   scan=packet[name]
   if np.asarray(scan['ranges']).shape != (2000,) or scan['range_min'] != .1 or scan['range_max'] != 8.:
    raise EnvironmentEpisodeFault('lidar_packet_contract_damaged',dict(sensor=name,control=control,shape=np.asarray(scan['ranges']).shape))
   r,a=sensor_ranges_to_base(scan['ranges'],angle_min=scan['angle_min'],angle_increment=scan['angle_increment'],range_min=scan['range_min'],range_max=scan['range_max'],translation_xy=scan['mount_position_robot_m'][:2],yaw=scan['mount_yaw_robot_rad'])
   mask=self.fixed_self_masks[name]
   if mask.shape!=r.shape:raise EnvironmentEpisodeFault('fixed_self_mask_shape_mismatch',dict(sensor=name,mask_shape=mask.shape,range_shape=r.shape))
   r[mask]=np.nan;a[mask]=np.nan;ranges.extend(r);angles.extend(a)
  valid=np.asarray(ranges);valid=valid[np.isfinite(valid)&(valid>0)];self.minimum_full_scan=float(valid.min()) if len(valid) else math.inf
  front,coverage,minimum=dual_lidar_to_legacy_scan(ranges,angles)
  # The legacy Gazebo adapter keeps empty projection bins as zero. Near a wall,
  # the offset sensor origins can create more than 5% empty bins without a
  # failed scan. Retain that observation contract and diagnose complete loss.
  if not np.isfinite(front).all() or not np.any(front>0):
   raise EnvironmentEpisodeFault('lidar_observation_unusable',dict(control=control,sim_time=time,coverage=coverage,valid_projected_beams=int(np.count_nonzero(front)),valid_full_beams=int(len(valid))))
  self.minimum_reward_scan=float(np.min(front[front!=0]))
  self.history.append(front,round(time*1e9));self.history.sample(round(time*1e9));seq=packet['stats']['sequence']
  if seq<=12 or seq%256==0:
   np.savez_compressed(self.out/'scans'/f'{seq:06d}.npz',scan_01=packet['scan_01']['ranges'],scan_02=packet['scan_02']['ranges'],front=front,sim_time=time,control=control,robot_pose=[*robot['root_position_m'][:2],robot['yaw_rad']])
  self.scan_log.write(json.dumps(dict(control=control,sim_time=time,stats=packet['stats'],coverage=coverage,minimum=None if not math.isfinite(minimum) else minimum,history_samples=len(self.history.frames),history_sampling_hz=20,history_unique_scans=len(set(self.history.source_timestamps_ns)),fixed_self_mask_sha256=self.self_mask_sha256,fixed_self_mask_beams={n:int(m.sum()) for n,m in self.fixed_self_masks.items()}))+'\n')
  if seq%5==0:
   target=self.out/'latest_lidar.npz';temp=self.out/'latest_lidar.tmp'
   with temp.open('wb') as stream:
    np.savez_compressed(stream,scan_01=packet['scan_01']['ranges'],scan_02=packet['scan_02']['ranges'],scan_01_points_world=packet['scan_01']['points_world_m'],scan_02_points_world=packet['scan_02']['points_world_m'],sim_time=time)
   temp.replace(target)

 def command(self,control,time,pedestrian_planar):
  self.latest_pedestrian_planar=np.asarray(pedestrian_planar).copy()
  if not self.history.ready:return self.current.copy()
  if not self.initial_alignment.done and self.initial_alignment.target_world is not None:
   return self.alignment_command(control,time)
  scan_ns=self.history.last_timestamp_ns
  action_due=self.action_clock.due(time)
  goal_due=action_due or self.cached_local_goal is None
  if not action_due and not goal_due:return self.current.copy()
  robot=self.robot.observe();pose=np.array([*robot['root_position_m'][:2],robot['yaw_rad']])
  # Reward/done belongs to TrainingPolicy.command after a real action interval,
  # matching the successful Gazebo trainer. No geometric proximity latch here.
  if hasattr(self.native,'current_yaw'):self.native.current_yaw=float(pose[2])
  if hasattr(self.native,'astar_path'):
   path=self.native.astar_path(pose[:2],self.goal)
  elif hasattr(self.native,'_planning_paths'):
   if not bool(self.native.in_domain(self.native.tensor(np.array([pose[:2],self.goal]))).all()):raise RuntimeError('DRLVO start/goal outside native planning domain')
   # ReconstructionScene's actual source API; retain its A* and rejection.
   path=self.native._planning_paths(self.native.tensor(pose[None,:2]),self.native.tensor(self.goal[None]),timed=False)[0]
  else:raise RuntimeError('Unsupported native global planning contract')
  connector_admitted=path is not None
  reference_source='fresh_oriented_connector'
  if path is None:
   # A global reference guides learning; it is not a per-action safety controller.
   # Reuse only the launch-certified route on byte-identical static map/goal.
   # Robot footprint margin is diagnostic only; actions and done use Gazebo semantics.
   if self.certified_reference is not None and np.array_equal(self.certified_map,self.native._labels) and np.array_equal(self.certified_goal,self.goal):
    path=self.certified_reference;reference_source='launch_certified_reference_current_connector_unavailable'
    with (self.out/'REFERENCE_CONNECTOR_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(control=control,sim_time=time,robot_pose=pose.tolist(),source=reference_source,current_connector_admitted=False,not_current_motion_safety_certificate=True))+'\n')
   else:
    from episode_semantics import RobotEpisodeTerminal
    raise RobotEpisodeTerminal('robot_reference_path_unavailable')
  # Open-ground .55m boundary clearance contains static base_link radius .509m.
  path=np.asarray(path)
  if goal_due:
   self.cached_local_goal=local_goal_base(path,pose,lookahead=self.policy.cfg['local_goal_lookahead_m']);self.last_goal_time=time
   with (self.out/'LOCAL_GOAL_EVENTS.jsonl').open('a') as f:f.write(json.dumps(dict(control=control,sim_time=time,base_xy=self.cached_local_goal.tolist(),role='initial_heading_alignment_target' if not self.initial_alignment.done else 'global_reference_only_no_velocity_control'))+'\n')
  if not self.initial_alignment.done:
   self.initial_alignment.start(self.cached_local_goal,pose)
   return self.alignment_command(control,time)
  if not action_due:return self.current.copy()
  local=self.cached_local_goal.copy();ped=np.asarray(pedestrian_planar)
  pmap=oracle_pedestrian_velocity_map(ped[:,:2],ped[:,2:4],pose);observation=build_observation(pmap,self.history.array(),local,clip_pedestrian=False)
  if self.policy.steps<256 or self.policy.steps%256==0:
   np.save(self.out/'policy_inputs'/f'{control:06d}.npy',observation)
  relative_positions=rotate_world_to_base(ped[:,:2]-pose[:2],pose[2]);relative_velocities=rotate_world_to_base(ped[:,2:4],pose[2])
  reward_context=dict(actual_v=robot['physical_forward_velocity_m_s'],actual_w=robot['physical_yaw_velocity_rad_s'],goal_base_xy=local.tolist(),pedestrians=np.c_[relative_positions,relative_velocities],full_scan_minimum=self.minimum_full_scan,scan_timestamp_ns=scan_ns)
  result=self.policy.command(observation,distance=float(np.linalg.norm(pose[:2]-self.goal)),minimum=self.minimum_reward_scan,control=control,sim_time=time,reward_context=reward_context);result.update(control=control,sim_time=time,scan_time_ns=self.history.last_timestamp_ns,history_frames=len(self.history.frames),local_goal_base_xy=local.tolist(),global_goal=self.goal.tolist(),local_goal_publication_sim_time=self.last_goal_time,global_reference_source=reference_source,current_pose_connector_admitted=connector_admitted,pedestrian_input='oracle',external_safety_override=False)
  result.update(history_sample_times_ns=list(self.history.sample_timestamps_ns),history_source_scan_times_ns=list(self.history.source_timestamps_ns),history_sampling_hz=20)
  if self.last_action_control is not None:
   assert control-self.last_action_control in (1,2), '20Hz deadline must quantize to 1 or 2 of the unchanged 30Hz controls'
   assert scan_ns>=self.last_action_scan_ns
  assert self.history.last_timestamp_ns<=round(time*1e9)
  self.action_clock.commit(time);self.last_action_scan_ns=scan_ns;self.last_action_control=control;self.current=result;self.policy_log.write(json.dumps(result)+'\n');return result.copy()

 def alignment_command(self,control,time):
  robot=self.robot.observe();pose=[*robot['root_position_m'][:2],robot['yaw_rad']]
  result=self.initial_alignment.command(pose,robot['physical_yaw_velocity_rad_s'],time)
  result.update(control=control,sim_time=time,local_goal_publication_sim_time=self.last_goal_time,
   global_goal=self.goal.tolist(),goal_distance_m=float(np.linalg.norm(np.asarray(pose[:2])-self.goal)),
   minimum_full_scan_m=self.minimum_full_scan,policy_steps=self.policy.steps)
  self.current=result
  with (self.out/'INITIAL_ALIGNMENT_EVENTS.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
  # Do not advance the policy clock, build a policy input, or call the model.
  return result.copy()

 def install_task(self,task):
  self.goal=np.asarray(task['goal_xy'],dtype=float)
  poses=np.asarray(task['poses_xy_yaw'],dtype=float)
  self.native.reference_poses=poses.copy();self.native.reference_only=True;self.native.last_poses=poses.copy()
  self.certified_reference=poses[:,:2].copy();self.certified_goal=self.goal.copy();self.certified_map=self.native._labels.copy()
  self.policy.task_id=int(task['task_id'])

 def reset_episode(self,task=None):
  if task is not None:self.install_task(task)
  self.minimum_full_scan=math.inf;self.minimum_reward_scan=math.inf;self.latest_pedestrian_planar=None;self.action_clock=ActionClock(.05)
  self.history=ScanHistory();self.last_action_control=None;self.last_action_scan_ns=None;self.cached_local_goal=None;self.last_goal_time=None
  self.initial_alignment.reset()
  self.current=dict(raw_action=None,clipped_action=None,desired_v_m_s=0.,desired_w_rad_s=0.,owner='SENSOR_WARMUP',external_safety_override=False)
  self.policy.reset_episode()

 def set_goal(self,goal,reference):
  task=dict(task_id=getattr(self.policy,'task_id',0)+1,goal_xy=np.asarray(goal).tolist(),poses_xy_yaw=np.asarray(reference).tolist())
  self.reset_episode(task=task)
