"""Turn toward one fixed world subgoal before frozen-policy inference starts."""
import math


class FirstSubgoalAlignment:
    def __init__(self, config=None):
        config = config or {}
        self.enabled = bool(config.get('enabled', True))
        self.tolerance = float(config.get('heading_tolerance_rad', math.radians(5)))
        self.max_w = float(config.get('max_w_rad_s', .8))
        self.kp = float(config.get('kp', 2.))
        self.settle_seconds = float(config.get('settle_seconds', .15))
        self.stopped_w = float(config.get('stopped_w_rad_s', .1))
        values = (self.tolerance, self.max_w, self.kp, self.settle_seconds, self.stopped_w)
        if not all(math.isfinite(v) and v > 0 for v in values) or self.tolerance >= math.pi:
            raise ValueError('Invalid first-subgoal alignment configuration')
        self.reset()

    def reset(self):
        self.target_world = None
        self.settle_since = None
        self.done = not self.enabled

    def start(self, local_xy, pose):
        """Latch the FIRST subgoal in world coordinates; never follow replans here."""
        x, y, yaw = map(float, pose)
        lx, ly = map(float, local_xy)
        if not all(map(math.isfinite, (x, y, yaw, lx, ly))):
            raise ValueError('Non-finite first subgoal or robot pose')
        self.target_world = (x + math.cos(yaw)*lx - math.sin(yaw)*ly,
                             y + math.sin(yaw)*lx + math.cos(yaw)*ly)

    def command(self, pose, actual_w, sim_time):
        if self.target_world is None:
            raise RuntimeError('First subgoal has not been published')
        x, y, yaw = map(float, pose)
        actual_w, sim_time = float(actual_w), float(sim_time)
        if not all(map(math.isfinite, (x, y, yaw, actual_w, sim_time))):
            raise ValueError('Non-finite alignment feedback')
        dx, dy = self.target_world[0]-x, self.target_world[1]-y
        c, s = math.cos(yaw), math.sin(yaw)
        local = [c*dx+s*dy, -s*dx+c*dy]
        # atan2(sin, cos) selects the shortest turn across the +/-pi seam.
        bearing = math.atan2(dy, dx)-yaw
        error = math.atan2(math.sin(bearing), math.cos(bearing)) if math.hypot(dx, dy) > 1e-6 else 0.
        aligned = abs(error) <= self.tolerance
        w = 0. if aligned else max(-self.max_w, min(self.max_w, self.kp*error))
        if aligned and abs(actual_w) <= self.stopped_w:
            if self.settle_since is None:
                self.settle_since = sim_time
            self.done = sim_time-self.settle_since >= self.settle_seconds-1e-9
        else:
            self.settle_since = None
        return dict(desired_v_m_s=0., desired_w_rad_s=w,
                    raw_action=None, clipped_action=None, training_done=False,
                    owner='INITIAL_SUBGOAL_ALIGNMENT', external_safety_override=False,
                    initial_alignment_state='aligned' if self.done else 'aligning',
                    initial_alignment_error_rad=error,
                    initial_alignment_target_world_xy_m=list(self.target_world),
                    local_goal_base_xy=local, inference_started=False)
