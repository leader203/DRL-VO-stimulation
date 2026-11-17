"""Launch exactly one owned Isaac process for one prepared phase."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

B = Path(__file__).resolve().parents[1]
ROOT = Path("/home/user/workspace/social_navigation_runtime_20261010")
OUT = Path(sys.argv[1]).resolve()
if OUT.parent != B / "runs" or not (OUT / "ROBOT_CONTROL.json").exists():
    raise SystemExit("Expected a prepared run inside this batch")
if (OUT / "launcher.log").exists():
    raise SystemExit("Run directory already launched")
preflight = json.loads((B / 'OFFLINE_PREFLIGHT.json').read_text())
if preflight['status'] != 'OFFLINE_TASK_PREFLIGHT_PASS':
    raise SystemExit('Offline task admission missing')
for name, expected in preflight['code_hashes'].items():
    if hashlib.sha256((B/'code'/name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Code changed after offline admission: '+name)
for name, expected in preflight['config_hashes'].items():
    if hashlib.sha256((B/name).read_bytes()).hexdigest() != expected:
        raise SystemExit('Task/config changed after offline admission: '+name)
for name, expected in preflight['native_reference_hashes'].items():
    if hashlib.sha256((B/name).read_bytes()).hexdigest()!=expected:
        raise SystemExit('Native reference changed: '+name)
phase = json.loads((OUT / "ROBOT_CONTROL.json").read_text())["training_phase"]
if phase == "long":
    smoke = json.loads((B / "training/SMOKE_PASS.json").read_text())
    if not smoke['passed'] or not smoke.get('checks') or not all(smoke['checks'].values()):
        raise SystemExit("Real smoke checks not passed")
    if not smoke.get('training_ready', False):
        if (sys.argv[2:] != ['--accept-known-linear-saturation'] or
                smoke.get('linear_action_saturation_fraction', 0) < .99):
            raise SystemExit("Long training readiness missing; known saturation needs explicit acceptance")
        (OUT / 'KNOWN_LIMITATION_ACCEPTED.json').write_text(json.dumps(dict(
            limitation='linear_action_saturation',
            saturation_fraction=smoke['linear_action_saturation_fraction'],
            training_ready_stays_false=True, original_actions_and_rewards_unchanged=True,
            acceptance='explicit launch flag; only allowed after user approval',
            smoke_sha256=hashlib.sha256((B / 'training/SMOKE_PASS.json').read_bytes()).hexdigest()
        ), indent=2) + '\n')
lock = (B / "ISAAC_OWNER.lock").open("a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
subprocess.run(["python3", str(ROOT / "tools/check_single_isaac.py")], check=True)
code = B / "code"
shutil.copytree(code, OUT / "code_at_launch", ignore=shutil.ignore_patterns("__pycache__"))
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code.glob("*.py")}
(OUT / "CODE_VERSION.json").write_text(json.dumps(hashes, indent=2) + "\n")
cmd = [str(ROOT / "python.sh"), str(code / "show_newmetro_robot_preview.py"),
       "--output", str(OUT), "--sfm-preview", "--manual-robot", "--open-doors",
       "--isolated-training"]
env = os.environ.copy()
env.update(OMNI_KIT_ACCEPT_EULA="YES", OMNI_KIT_DISABLE_TELEMETRY="1",
           PYTHONDONTWRITEBYTECODE="1")
with (OUT / "launcher.log").open("w") as log:
    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    owner = {"pid": child.pid, "argv": cmd,
             "start_ticks": Path(f"/proc/{child.pid}/stat").read_text().split()[21],
             "launcher_pid": os.getpid(), "phase": phase, "uid": os.getuid()}
    (OUT / "OWNER.json").write_text(json.dumps(owner, indent=2) + "\n")
    exit_code = child.wait()
report = json.loads((OUT / "REPORT.json").read_text()) if (OUT / "REPORT.json").exists() else {}
result = {"exit_code": exit_code, "status": report.get("status"),
          "steps": report.get("training_total_steps", "NOT_OBSERVED"),
          "updates": report.get("training_updates", "NOT_OBSERVED"),
          "finished_at": time.time()}
(OUT / "EXIT.json").write_text(json.dumps(result, indent=2) + "\n")
if exit_code or report.get("error"):
    raise SystemExit(1)
