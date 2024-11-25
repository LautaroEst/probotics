
import time

from .base import BaseTask

class GoToPoint(BaseTask):

    def __init__(self, point):
        self.point = point

    def run_cycle(self, state):

        if self.has_arrived(state['robot'].current_pose):
            state['task_status'] = "finished"
            v, w = 0., 0.
        else:
            # Planificar trayectoria
            v, w = self.plan_going_to(state['robot'].current_pose, state['robot'].world)
        return {
            "linear_velocity": v,
            "angular_velocity": w,
        }

    def has_arrived(self, pose):
        return True
    
    def plan_going_to(self, pose, world):
        pass