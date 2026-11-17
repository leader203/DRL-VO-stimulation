"""Measured execution feedback for the manual preview, never a stop command."""
from collections import deque
import numpy as np


class PreviewProgress:
    def __init__(self, window_controls=90, minimum_motion_m=.05):
        self.window = window_controls
        self.minimum_motion = minimum_motion_m
        self.samples = deque()
        self.stalled = None

    def update(self, control, xy, intended, waiting):
        xy = np.asarray(xy, dtype=float)
        if not self.samples or control <= self.samples[-1][0]:
            self.samples.clear()
            self.stalled = np.zeros(len(xy), dtype=bool)
        self.samples.append((control, xy.copy()))
        while len(self.samples) > 1 and self.samples[1][0] <= control-self.window:
            self.samples.popleft()
        positions = np.stack([value for _, value in self.samples])
        motion = np.linalg.norm(positions.max(0)-positions.min(0), axis=-1)
        ready = control-self.samples[0][0] >= self.window
        # Do not retry the same ineffective candidate family until measured
        # translation resumes. Braking plans must not erase a known stall.
        self.stalled[motion >= self.minimum_motion] = False
        if ready:
            self.stalled |= ((motion < self.minimum_motion)
                             & np.asarray(intended) & ~np.asarray(waiting))
        return motion, self.stalled.copy(), ready
