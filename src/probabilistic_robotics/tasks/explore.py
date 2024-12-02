
import numpy as np
from .base import BaseTask
import time

def find_quantile(support, values, q):
    """Finds the q-th quantile of a distribution with support and values"""
    if len(support) == 0:
        return None
    if len(support) == 1:
        return values[0]
    if q == 0:
        return np.min(values)
    if q == 1:
        return np.max(values)
    if q < 0 or q > 1:
        raise ValueError("Quantile must be between 0 and 1")
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_support = support[sorted_indices]
    total_mass = np.sum(sorted_values)
    target_mass = q * total_mass
    cumsum = np.cumsum(sorted_values)
    index = np.searchsorted(cumsum, target_mass)
    return sorted_support[index]


class Explore(BaseTask):

    def __init__(self, safe_distance=0.5, min_linear_velocity=0.0, max_linear_velocity=0.5, max_angular_velocity=0.5):
        self.safe_distance = safe_distance
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.min_linear_velocity = min_linear_velocity

    def run_cycle(self, global_state):

        robot_pose = global_state["robot"].current_pose
        ranges = global_state["sensor"].ranges
        angles = global_state["sensor"].scan_angles

        # Identify dangerous regions
        close_points = (ranges < self.safe_distance) & (~np.isnan(ranges))
        if sum(close_points) > len(ranges) * 3 / 4:
            # If the robot is surrounded by obstacles, stop
            return {
                "linear_velocity": self.min_linear_velocity,
                "angular_velocity": self.max_angular_velocity,
            }
            
        if np.any(close_points):
            # Obstacle detected, find direction of the most open space
            free_angles = angles[~close_points]
            free_ranges = ranges[~close_points]
            if len(free_angles) > 0:
                # Rotate towards the center of the open area
                # argmax_range = np.argmax(ranges[~close_points])
                # target_angle = free_angles[argmax_range]
                target_angle = find_quantile(free_angles, free_ranges, 0.5)
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
    
    
        