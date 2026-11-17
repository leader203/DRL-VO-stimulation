"""Apply from Isaac's Script Editor to the running preview without resetting it.

No SimulationApp is constructed. No USD, camera, physics rate, AMP weights, or
body state is changed. The live preview owns all physics and UI updates.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

class StreamingScanHistory(list):
    def __init__(self, rows, path, max_rows=32):
        super().__init__()
        self.stream = Path(path).open('a', buffering=1)
        self.max_rows = max_rows
        for row in rows:
            self.append(row)

    def append(self, row):
        self.stream.write(json.dumps(row, separators=(',', ':'))+'\n')
        super().append(row)
        if len(self) > self.max_rows:
            del self[:-self.max_rows]


def apply_to_preview(m, num_samples=8):
    if m.report.get('live_performance_patch_applied'):
        print('Performance patch already active')
        return
    if num_samples not in (8, 16):
        raise ValueError('Use 8 preview candidates or the original 16')
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='preview_trace')
    original_plan = m.planner.plan
    original_save = m.save
    original_new_runtime = m.new_runtime
    original_physics_step = m.world.simulator._physics_step
    original_render = m.world.simulator._sim.render
    state = {'inside_physics': False}

    def responsive_plan(**kwargs):
        started = time.monotonic()
        future = executor.submit(original_plan, **kwargs)
        updates = 0
        while not future.done():
            # Physical state stays fixed until the same full plan is available.
            # pump_gui explicitly disables Kit's automatic physical stepping.
            m.pump_gui()
            updates += 1
            time.sleep(.025)
        result = future.result()
        m.report['planner_wait_gui_updates'] = m.report.get('planner_wait_gui_updates', 0) + updates
        m.report['planner_worker_wall_seconds'] = time.monotonic() - started
        return result

    def physics_step():
        started = time.monotonic()
        state['inside_physics'] = True
        try:
            return original_physics_step()
        finally:
            state['inside_physics'] = False
            m.report['last_physics_control_wall_seconds'] = time.monotonic()-started

    def render(*args, **kwargs):
        # The existing outer loop renders AFTER SMPL and LiDAR points update.
        # The inner physics render occurs before those updates and is redundant.
        if state['inside_physics']:
            m.report['redundant_physics_renders_skipped'] = m.report.get('redundant_physics_renders_skipped', 0)+1
            return
        return original_render(*args, **kwargs)

    def new_runtime():
        result = original_new_runtime()
        result.num_samples = num_samples
        return result

    def save():
        if len(m.robot_rows) > 600 and not m.robot_auto:
            del m.robot_rows[:-600]
        m.report['lidar_history_format'] = 'full append-only lidar_stats.jsonl; latest 32 in lidar_stats.json'
        m.report['robot_history_in_memory_rows'] = len(m.robot_rows)
        original_save()

    previous_history = m.OUT / 'lidar_stats.json'
    if previous_history.exists():
        previous_history.rename(m.OUT / 'lidar_stats_before_performance_patch.json')
    history = StreamingScanHistory(list(m.lidar_rows), m.OUT / 'lidar_stats.jsonl')
    m.lidar_rows = history
    m.planner.plan = responsive_plan
    m.runtime.num_samples = num_samples
    m.new_runtime = new_runtime
    m.world.simulator._physics_step = physics_step
    m.world.simulator._sim.render = render
    m.save = save
    m._preview_performance_executor = executor
    m.report.update(live_performance_patch_applied=True,
        preview_candidates=num_samples, original_profile_candidates=m.profile['num_samples'],
        preview_sampling_changed=num_samples != m.profile['num_samples'],
        diffusion_steps_unchanged=m.planner.model.n_timesteps,
        physics_held_during_planning=True, physics_advanced_from_gui=False,
        camera_changed_by_patch=False, body_pose_reset_by_patch=False,
        patch_scope='Responsive GUI during planning; bounded memory and append-only scan evidence; one render per control; CPU physics remains',
        patch_wall_time=time.time())
    save()
    print('Live preview performance patch applied; no reset or camera change')


def apply_current_preview():
    m = sys.modules['__main__']
    if Path(getattr(m, '__file__', '')).name != 'show_newmetro_robot_preview.py':
        raise RuntimeError('This command must run inside the existing NewMetro robot preview, from Script Editor')
    apply_to_preview(m)

if __name__ in ('__main__', '<run_path>'):
    apply_current_preview()
