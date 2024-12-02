
import numpy as np
from .base import BaseTask
import time

class Explore(BaseTask):

    def __init__(self, safe_distance=0.5, max_linear_velocity=0.5, max_angular_velocity=0.5):
        self.safe_distance = safe_distance
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.min_linear_velocity = 0.1

    def run_cycle(self, global_state):

        robot_pose = global_state["robot"].current_pose
        ranges = global_state["sensor"].ranges
        angles = global_state["sensor"].scan_angles

        # Identify dangerous regions
        close_points = (ranges < self.safe_distance) & (~np.isnan(ranges))
        if np.any(close_points):
            # Obstacle detected, find direction of the most open space
            free_angles = angles[~close_points]
            if len(free_angles) > 0:
                # Rotate towards the center of the open area
                # argmax_range = np.argmax(ranges[~close_points])
                # target_angle = free_angles[argmax_range]
                target_angle = np.median(free_angles)
                # TODO: Probar mirando los ranges en lugar de los ángulos
                angular_velocity = np.clip(target_angle, -self.max_angular_velocity, self.max_angular_velocity)
                linear_velocity = self.min_linear_velocity
            else:
                # No open space; rotate in place
                angular_velocity = self.max_angular_velocity
                linear_velocity = self.min_linear_velocity
        else:
            # No obstacles, move forward
            angular_velocity = 0.0
            linear_velocity = self.max_linear_velocity
        
        return {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }
    
    
        