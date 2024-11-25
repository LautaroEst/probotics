
from .base import BaseTask
import time

class RunInCircles(BaseTask):

    def __init__(self):
        self.start_w = 0
        self.v = 0.1

    def run_cycle(self, global_state):
        if time.time() - global_state["start_time"] < 5:
            w = -0.1
        elif time.time() - global_state["start_time"] >= 7.5:
            w = 0.1
        return {
            "linear_velocity": self.v,
            "angular_velocity": w
        }
        