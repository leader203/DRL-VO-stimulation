"""Manual preview only: AMP waiting while planning retries; no robot ownership."""
import torch


class ManualPreviewPlannerWait:
    def __init__(self, admission_error, report, on_failure, retry_controls=30):
        self.admission_error = admission_error
        self.report = report
        self.on_failure = on_failure
        self.retry_controls = retry_controls
        self.retry_at = 0
        self.anchor = None

    def action(self, runtime, world, control, targets, **kwargs):
        if self.anchor is None or control >= self.retry_at:
            try:
                result = runtime.action(world, control, targets, **kwargs)
            except self.admission_error as error:
                runtime.invalidate_after_reset()
                if self.anchor is None:
                    self.anchor = world.state().root_pos[:, :2].detach().clone()
                self.retry_at = control + self.retry_controls
                self.report.update(
                    last_admission_error=str(error), last_admission_control=control,
                    planner_wait_failures=self.report.get('planner_wait_failures', 0) + 1)
                self.on_failure(dict(control=control, error=str(error),
                    retry_at_control=self.retry_at, robot_command_overridden=False,
                    waiting_target_admitted=False, contact_safety='UNVERIFIED'))
            else:
                self.anchor = None
                self.report['pedestrian_planning_wait'] = False
                return result
        # A desired holding point, not a physical freeze or an admitted route.
        # AMP/PD and world.step still execute for every real control interval.
        state = world.state()
        hold = self.anchor[:, None, :].expand(-1, 10, -1).clone()
        action = runtime.controller.act(state, hold, world, return_raw=True)
        if not torch.isfinite(action).all():
            raise FloatingPointError('Nonfinite AMP waiting action')
        runtime.last_control = control
        self.report.update(pedestrian_planning_wait=True,
            planner_wait_controls=self.report.get('planner_wait_controls', 0) + 1,
            planner_wait_target_admitted=False, planner_blocked=False)
        return action, dict(replanned=False, safety_braking=False,
            pedestrian_planning_wait=True)
