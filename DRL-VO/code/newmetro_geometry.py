"""Current oriented full-envelope static/floor route queries; no physical step."""
import json,math
from pathlib import Path
import numpy as np
def verify_low_height_route(native,planner,poses,query,out):
 floors=set(native.provider.floor_paths);statics=set(native.provider.static_paths);z=native.provider.ground_height_m
 lower,upper=planner.lower[:2],planner.upper[:2];offset=(upper+lower)/2;half=(upper-lower)/2+.03
 rows=[];failures=[];samples=[]
 for a,b in zip(poses[:-1],poses[1:]):
  delta=(b[2]-a[2]+math.pi)%(2*math.pi)-math.pi
  n=max(1,math.ceil((np.linalg.norm(b[:2]-a[:2])+planner.radius*abs(delta))/.01))
  samples.extend(np.c_[np.linspace(a[:2],b[:2],n+1),a[2]+np.linspace(0,1,n+1)*delta])
 for i,pose in enumerate(samples):
  yaw=float(pose[2]);rot=np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
  center=pose[:2]+rot@offset;hits=[]
  def overlap(hit):
   path=str(hit.collision)
   if path.startswith('/World/NavigationRobot/') or path.startswith('/World/envs/'):return True
   if path not in floors and path not in statics:raise RuntimeError('Unexpected static route collider: '+path)
   if path in statics:hits.append(path)
   return True
  query.overlap_box((float(half[0]),float(half[1]),.8975),(float(center[0]),float(center[1]),z+.9025),(0.,0.,math.sin(yaw/2),math.cos(yaw/2)),overlap,False)
  unsupported=[]
  for ox,oy in [(0,0),(-1,-1),(-1,1),(1,-1),(1,1),(0,-1),(0,1),(-1,0),(1,0)]:
   xy=center+rot@(half*np.array([ox,oy]));supported=[False]
   def ray(hit):
    if str(hit.collision) in floors and hit.normal[2]>=.9 and abs(hit.position[2]-z)<.08:supported[0]=True
    return True
   query.raycast_all((float(xy[0]),float(xy[1]),z+.1),(0.,0.,-1.),.2,ray,False)
   if not supported[0]:unsupported.append(xy.tolist())
  row=dict(pose_xy_yaw=pose.tolist(),static_hits=hits,unsupported_floor_samples=unsupported);rows.append(row)
  if hits or unsupported:failures.append(row)
 result=dict(status='CURRENT_ORIENTED_FULL_HEIGHT_ROUTE_PASS' if not failures else 'CURRENT_ORIENTED_FULL_HEIGHT_ROUTE_FAILED',height_band_m=[.005,1.8],footprint='source oriented rectangle with base_link offset',total_sample_margin_m=.03,doors='current fixed-open session',source_walls_preserved=True,contact_safety='UNVERIFIED',rows=rows,failures=failures)
 (Path(out)/'NEWMETRO_RUNTIME_GEOMETRY.json').write_text(json.dumps(result,indent=2))
 if failures:raise RuntimeError('Current NewMetro oriented full-height route gate failed')
