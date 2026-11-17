"""Read-only current PID, transition, update, episode and checkpoint evidence."""
import collections
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path

B = Path(__file__).resolve().parents[1]
# Resumed segments retain absolute counters. Discover the unique current owner
# instead of reporting the sealed first segment as current progress.
candidates = []
for owner_path in (B / "runs").glob("*/OWNER.json"):
    try:
        candidate = json.loads(owner_path.read_text())
        proc = Path("/proc") / str(candidate["pid"])
        argv = [x.decode(errors="replace") for x in proc.joinpath("cmdline").read_bytes().split(b"\0") if x]
        identity = (candidate.get("uid") == os.getuid() == proc.stat().st_uid
                    and proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19] == candidate["start_ticks"]
                    and argv[1:] == candidate["argv"][1:])
        if identity:
            candidates.append((owner_path.parent, candidate, argv))
    except (OSError, KeyError, json.JSONDecodeError):
        continue
if len(candidates) != 1:
    raise SystemExit("NOT_OBSERVED: expected exactly one live owner in this training batch; no snapshot overwritten")
run, owner, argv = candidates[0]
identity = True


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


transitions = rows(run / "transitions.jsonl")
updates = rows(run / "updates.jsonl")
episodes = rows(B / "training/episodes.jsonl")
resets = rows(run / "RESET_EVENTS.jsonl")
report = json.loads((run / "REPORT.json").read_text()) if (run / "REPORT.json").exists() else {}
progress = json.loads((B / "training/PROGRESS.json").read_text()) if (B / "training/PROGRESS.json").exists() else {}
checkpoint = B / "training/session_state.pt"
previous_path = B / "STATUS_SNAPSHOT.json"
previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
observed_steps = transitions[-1]["total_step"] if transitions else "NOT_OBSERVED"
restored = json.loads((run / "CHECKPOINT_RESTORE.json").read_text()) if (run / "CHECKPOINT_RESTORE.json").exists() else {}
update_counters = [x["update"] for x in updates if isinstance(x.get("update"), int)]
if isinstance(restored.get("updates"), int):
    update_counters.append(restored["updates"])
last_steps = previous.get("observed_accepted_steps")
summary = {
    "checked_at_myt": datetime.now(timezone(timedelta(hours=8))).isoformat(),
    "target_accepted_steps": 100000,
    "active_run": str(run),
    "owner_pid": owner["pid"],
    "owner_start_ticks": owner["start_ticks"],
    "owner_identity_matches_proc": identity,
    "owner_argv": argv if identity else "NOT_OBSERVED",
    "runtime_status": report.get("status", "NOT_OBSERVED"),
    "observed_accepted_steps": observed_steps,
    "new_long_run_steps": len(transitions),
    "accepted_steps_delta_since_previous_snapshot":
        observed_steps - last_steps if isinstance(observed_steps, int) and isinstance(last_steps, int) else "NOT_OBSERVED",
    "effective_updates": max(update_counters) if update_counters else "NOT_OBSERVED",
    "latest_update": updates[-1] if updates else "NOT_OBSERVED",
    "episode_terminations": dict(collections.Counter(x["reason"] for x in episodes)),
    "long_episode_resets": len(resets),
    "latest_30_episode_goal_rate": (
        sum(x["reason"] == "goal" for x in episodes[-30:]) / len(episodes[-30:])
        if episodes else "NOT_OBSERVED"),
    "latest_30_episode_mean_return": (
        sum(x["return_sum"] for x in episodes[-30:]) / len(episodes[-30:])
        if episodes else "NOT_OBSERVED"),
    "checkpoint_path": str(checkpoint) if checkpoint.exists() else "NOT_OBSERVED",
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint.exists() else "NOT_OBSERVED",
    "checkpoint_accepted_steps": progress.get("total_steps", "NOT_OBSERVED"),
    "checkpoint_effective_updates": progress.get("total_updates", "NOT_OBSERVED"),
    "loaded_paths": json.loads((run / "LOADED_PATHS.json").read_text())
        if (run / "LOADED_PATHS.json").exists() else "NOT_OBSERVED",
    "current_physical_control": report.get("physical_control", "NOT_OBSERVED"),
    "real_scan_history_frames": report.get("real_scan_history_frames", "NOT_OBSERVED"),
    "pedestrian_near_events": report.get("pedestrian_pair_close_events", "NOT_OBSERVED"),
    "contact_safety": "UNVERIFIED",
    "independent_evaluation": "NOT_EXECUTED",
    "target_reached": observed_steps >= 100000 if isinstance(observed_steps, int) else "NOT_OBSERVED",
}
tmp = B / "STATUS_SNAPSHOT.tmp"
tmp.write_text(json.dumps(summary, indent=2) + "\n")
tmp.replace(previous_path)
print(json.dumps({key: summary[key] for key in (
    "owner_identity_matches_proc", "observed_accepted_steps",
    "accepted_steps_delta_since_previous_snapshot", "effective_updates",
    "episode_terminations", "checkpoint_accepted_steps", "runtime_status")}))
