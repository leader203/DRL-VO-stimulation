"""Process-local timing only; no sensor, policy or physics semantics changes."""
import os,time,json,functools,collections
from contextlib import contextmanager
from pathlib import Path
import torch

class Probe:
 def __init__(self,m):
  self.m=m;self.out=m.OUT;self.p=m.training_bridge.driver.policy
  self.profile=os.environ.get('NEWMETRO_PROFILE','0')=='1'
  self.target=int(os.environ.get('NEWMETRO_TARGET_UPDATES','0'))
  self.totals=collections.defaultdict(float);self.counts=collections.Counter();self.stack=[];self.beg=time.perf_counter();self.prev_update=self.p.updates
  self.events=(self.out/'PERF_EVENTS.jsonl').open('w',buffering=1)
  self.phasefile=self.out/'PERF_PHASE.json';self.phase('rollout')
  self.event('restore',device=str(self.p.agent.device),policy_device=str(next(self.p.model.parameters()).device),optimizer_devices=sorted(set(str(v.device) for s in self.p.agent.optimizer.state.values() for v in s.values() if isinstance(v,torch.Tensor))),torch_threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),world_device=str(m.world.device),profile=self.profile,target_updates=self.target,model_fingerprint=self.p.loaded_state_sha256)
 def event(self,kind,**kw):self.events.write(json.dumps(dict(kind=kind,epoch=time.time(),perf=time.perf_counter(),steps=self.p.steps,updates=self.p.updates,control=self.m.control,sim_time=self.m.control*self.m.world.dt,**kw))+'\n')
 def phase(self,s):
  tmp=self.phasefile.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=s,epoch=time.time(),steps=self.p.steps)));tmp.replace(self.phasefile)
 @contextmanager
 def measure(self,name):
  if not self.profile:yield;return
  t=time.perf_counter();entry=[name,t,0.];self.stack.append(entry)
  try:yield
  finally:
   elapsed=time.perf_counter()-t;self.stack.pop();self.totals[name]+=elapsed-entry[2];self.counts[name]+=1
   if self.stack:self.stack[-1][2]+=elapsed
 def wrap(self,obj,name,label):
  original=getattr(obj,name)
  @functools.wraps(original)
  def call(*a,**kw):
   with self.measure(label):return original(*a,**kw)
  setattr(obj,name,call)
 def dump(self):
  v=dict(profile=self.profile,epoch=time.time(),wall_s=time.perf_counter()-self.beg,steps=self.p.steps,updates=self.p.updates,control=self.m.control,sim_time=self.m.control*self.m.world.dt,exclusive_seconds=dict(self.totals),calls=dict(self.counts))
  t=self.out/'PERF_TIMINGS.tmp';t.write_text(json.dumps(v,indent=2));t.replace(self.out/'PERF_TIMINGS.json')

def install(m):
 if os.environ.get('NEWMETRO_GPU_EXECUTION','0')=='1':
  from gpu_execution import install as gpu_install
  gpu_install(m)
 q=Probe(m);m.performance_probe=q;p=q.p;b=m.training_bridge;d=b.driver
 # Profiler wraps only when requested; final throughput runs use update/window markers.
 if q.profile:
  for obj,name,label in [(m.world,'step','simulation_physics'),(m.world.simulator._sim,'render','render'),(m,'pump_gui','kit_gui_pump'),(m.mesh,'update','mesh_visualization'),(m.runtime,'action','pedestrian_sfm_amp'),(m.runtime.controller,'act','pedestrian_amp_inference'),(d,'command','observation_route_reward_rollout'),(d,'post_physics','lidar_projection_history'),(d.lidar,'scan','physx_lidar_raycast'),(p.agent,'select_action','policy_inference_transfer_sync'),(p.agent,'value','policy_bootstrap'),(p,'checkpoint','checkpoint_log_save'),(m,'save','report_log_save'),(b,'handle_terminal','task_reset_planning')]:q.wrap(obj,name,label)
  # Fine-grained policy H2D/forward/D2H timings only in diagnostic runs.
  def select(observation):
   import numpy as np
   p.agent.policy.train(True)
   with torch.no_grad():
    with q.measure('policy_h2d'):
     obs=torch.as_tensor(np.asarray(observation,np.float32)[None],device=p.agent.device)
    with q.measure('policy_cuda_forward_enqueue'):
     action,value,logp=p.agent.policy.forward(obs)
    with q.measure('policy_d2h_and_sync'):
     raw=action[0].cpu().numpy();lp=float(logp.item());v=float(value.item())
   return raw,np.clip(raw,-1.,1.),lp,v
  p.agent.select_action=select
 original_update=p.agent.update
 def update(*a,**kw):
  q.phase('ppo_update');q.event('ppo_begin',buffer_length=len(a[0]));t=time.perf_counter()
  try:
   with q.measure('ppo_update'):result=original_update(*a,**kw)
   q.event('ppo_end',wall_s=time.perf_counter()-t,metrics=result)
   return result
  finally:q.phase('rollout')
 p.agent.update=update
 original_checkpoint=p.checkpoint
 def checkpoint(*a,**kw):
  r=original_checkpoint(*a,**kw)
  if p.updates!=q.prev_update:
   q.prev_update=p.updates;q.event('update_checkpoint_end');q.dump()
  return r
 p.checkpoint=checkpoint
 orig_stop=b.should_stop
 def stop(control):
  if q.target and p.updates>=q.target:
   m.report['status']='PERFORMANCE_BOUND_COMPLETED';m.report['performance_bound_updates']=q.target;q.event('bounded_complete');q.dump();return True
  if q.profile and control%128==0:q.dump()
  return orig_stop(control)
 b.should_stop=stop
 orig_close=b.close
 def close(*a,**kw):
  try:return orig_close(*a,**kw)
  finally:q.event('closed');q.dump();q.phase('exited')
 b.close=close
 m.performance_probe=q
