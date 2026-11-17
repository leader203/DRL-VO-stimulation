"""Pure execution-device optimization. Frozen weights; same CPU PhysX state."""
from dataclasses import replace
from pathlib import Path
import json,time,copy,hashlib
import torch

def fingerprint(module):
 h=hashlib.sha256()
 for name,value in module.state_dict().items():
  h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

@torch.no_grad()
def install(m):
 visual=m.mesh.visualizer
 amp_before=fingerprint(m.controller.network);smpl_before=fingerprint(visual.smpl_model)
 from CrowdSim.control.drive import convert_quaternion_order
 robot=m.mesh.simulator._robot
 def view(v):return v if isinstance(v,torch.Tensor) else getattr(v,'torch',v)
 joint_names=getattr(robot,'joint_names',None)
 if joint_names is None:joint_names=robot.data.joint_names
 actual_pose=m.mesh.pose_mapper(view(robot.data.joint_pos),list(joint_names))
 actual_root=view(robot.data.root_pos_w)
 actual_quat=convert_quaternion_order(view(robot.data.root_quat_w),getattr(m.mesh.simulator,'native_quaternion_order','xyzw'),'wxyz')
 actual_cpu_vertices=visual.compute_world_vertices(actual_pose,actual_root,actual_quat).cpu()
 # Check both neutral and the actual current physical pose using the existing API.
 neutral_pose=torch.zeros(m.count,69);neutral_root=torch.zeros(m.count,3);neutral_quat=torch.zeros(m.count,4);neutral_quat[:,0]=1.
 cpu_vertices=visual.compute_world_vertices(neutral_pose,neutral_root,neutral_quat).cpu()
 visual.device=torch.device('cuda:0');visual.cfg=replace(visual.cfg,device='cuda:0');visual.smpl_model.to(visual.device);visual.betas=visual.betas.to(visual.device);visual.shape_floor_offsets=visual.shape_floor_offsets.to(visual.device)
 gpu_vertices=visual.compute_world_vertices(neutral_pose,neutral_root,neutral_quat).cpu()
 skin_error=float((cpu_vertices-gpu_vertices).abs().max());assert skin_error<=1e-4,skin_error
 actual_gpu_vertices=visual.compute_world_vertices(actual_pose,actual_root,actual_quat).cpu()
 actual_skin_error=float((actual_cpu_vertices-actual_gpu_vertices).abs().max());assert actual_skin_error<=1e-4,actual_skin_error
 controller=m.controller
 # CPU normalization stays bit-identical; only the frozen MLP moves to CUDA.
 rng=torch.Generator(device='cpu').manual_seed(77109)
 samples=[torch.randn(m.count,1401,generator=rng).clamp(-5.,5.) for _ in range(16)]
 cpu=[controller.network.eval_actor(x)[0].clamp(-1.,1.).cpu() for x in samples]
 controller.network.to('cuda:0').eval()
 gpu=[controller.network.eval_actor(x.to('cuda:0'))[0].clamp(-1.,1.).cpu() for x in samples]
 action_error=max(float((a-b).abs().max()) for a,b in zip(cpu,gpu));assert action_error<=1e-4,action_error
 @torch.no_grad()
 def gpu_act(state,trajectory,terrain,*,trajectory_frame='world',return_raw=False,deterministic=True,generator=None):
  controller._require_checkpoint()
  if not deterministic:raise RuntimeError('This experiment requires unchanged deterministic AMP')
  observation=controller.raw_observation(state,trajectory,terrain,trajectory_frame=trajectory_frame)
  normalized=controller.observation_normalizer(observation.float())
  mean,_=controller.network.eval_actor(normalized.to('cuda:0'))
  action=mean.clamp(-1.,1.).to(state.root_pos.device)
  return action if return_raw else controller.decode_action(action)
 controller.act=gpu_act
 amp_after=fingerprint(controller.network);smpl_after=fingerprint(visual.smpl_model)
 assert amp_before==amp_after and smpl_before==smpl_after,'Frozen weights changed during execution-device move' 
 record=dict(AMP_actor_device=str(next(controller.network.parameters()).device),skinning_device=str(visual.device),AMP_max_abs_error=action_error,AMP_comparison_observations=64,AMP_atol=1e-4,skinning_neutral_max_abs_error_m=skin_error,skinning_actual_pose_max_abs_error_m=actual_skin_error,AMP_weights_sha256_before=amp_before,AMP_weights_sha256_after=amp_after,SMPL_weights_sha256_before=smpl_before,SMPL_weights_sha256_after=smpl_after,skinning_atol_m=1e-4,physics_device=str(m.world.device),weights_unchanged=True,normalization_cpu_unchanged=True,deterministic_AMP_unchanged=True,render_calls_unchanged=True,outer_Kit_pump_retained=True,training_ready=False,physical_contact_safety='UNVERIFIED')
 (m.OUT/'GPU_EXECUTION_CHECK.json').write_text(json.dumps(record,indent=2));m.report.update(gpu_execution=record,inference_device='cuda:0',skinning_device='cuda:0');m.save()
