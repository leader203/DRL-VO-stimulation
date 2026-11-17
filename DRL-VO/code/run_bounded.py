"""Own one training launch, bound stalled startup/progress, retain failure evidence."""
from pathlib import Path
import json,os,sys,subprocess,time,signal
B=Path(__file__).resolve().parents[1];out=Path(sys.argv[1]).resolve()
assert out.parent==B/'runs'
cfg=json.loads((out/'ROBOT_CONTROL.json').read_text());limit=cfg['wall_limit_s']
p=subprocess.Popen([sys.executable,str(B/'code/launch_sfm_training.py'),str(out),*sys.argv[2:]])
start=last=time.monotonic();previous=None;reason=None
while p.poll() is None:
 time.sleep(10);now=time.monotonic()
 try:
  report=json.loads((out/'REPORT.json').read_text());marker=(report.get('training_total_steps'),report.get('training_updates'))
  if marker!=(None,None) and marker!=previous:previous=marker;last=now
 except (OSError,ValueError):pass
 if now-start>limit:reason='wall_limit'
 elif now-last>600:reason='no_real_step_progress_600s'
 if reason:
  (out/'WATCHDOG.json').write_text(json.dumps(dict(reason=reason,wall_seconds=now-start,last_progress=previous)))
  owner=json.loads((out/'OWNER.json').read_text()) if (out/'OWNER.json').exists() else None
  if owner:
   proc=Path('/proc')/str(owner['pid'])
   def matches():
    try:return proc.stat().st_uid==os.getuid() and proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==owner['start_ticks'] and str(out).encode() in proc.joinpath('cmdline').read_bytes().split(b'\0')
    except OSError:return False
   if matches():(out/'VIEW_REQUEST.json').write_text(json.dumps(dict(close=True,reason=reason)))
   for _ in range(30):
    if p.poll() is not None:break
    time.sleep(1)
   if matches():os.kill(owner['pid'],signal.SIGTERM)
   for _ in range(15):
    if p.poll() is not None:break
    time.sleep(1)
   if matches():os.kill(owner['pid'],signal.SIGKILL)
  if p.poll() is None:p.terminate()
  break
try:code=p.wait(timeout=15)
except subprocess.TimeoutExpired:p.kill();code=p.wait()
(out/'BOUNDED_EXIT.json').write_text(json.dumps(dict(exit_code=code,watchdog_reason=reason,elapsed_seconds=time.monotonic()-start)))
raise SystemExit(1 if reason else code)
