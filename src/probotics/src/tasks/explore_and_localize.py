
import numpy as np
from .base import BaseTask
import time
from ..mapping import Map2D
from ..robots import NoisyDiffDriveRobot
from ..sensors import Lidar

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


class ParticleFilterLocalization:

    def __init__(
        self, 
        N_particles, 
        map2d, 
        radius, 
        wheels_radius, 
        wheels_distance, 
        alpha,
        sensor_offset, 
        num_scans, 
        start_angle, 
        end_angle, 
        min_range, 
        max_range, 
        occupation_threshold,
        seed
    ):
        self.N_particles = N_particles
        self.map2d = map2d

        self.seed = seed
        rs = np.random.RandomState(seed)

        possible_initial_poses = np.array(np.where(map2d.map_array > occupation_threshold)).T * map2d.map_resolution
        initial_poses = possible_initial_poses[rs.permutation(len(possible_initial_poses))[:N_particles]]
        
        particles = []
        for i in range(N_particles):
            initial_pose = np.array([initial_poses[i,1], initial_poses[i,0], rs.rand() * 2 * np.pi - np.pi])
            p = NoisyDiffDriveRobot(initial_pose, radius, wheels_radius, wheels_distance, alpha, seed)
            particles.append(p)

        self.particles = particles
        self.weights = np.ones(N_particles) / N_particles
        self.sensor = Lidar(sensor_offset, num_scans, start_angle, end_angle, min_range, max_range, occupation_threshold, seed)
        self.rs = rs

        self.radius = radius
        self.wheels_radius = wheels_radius
        self.wheels_distance = wheels_distance
        self.alpha = alpha
        self.sensor_offset = sensor_offset
        self.num_scans = num_scans
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.min_range = min_range
        self.max_range = max_range
        self.occupation_threshold = occupation_threshold

    def prediction_step(self, odometry):
        # Implementación del paso de predicción
        for particle in self.particles:
            particle.apply_movement(odometry["linear_velocity"], odometry["angular_velocity"], odometry["dt"])

    def correction_step(self, sensor):
        # Implementación del paso de corrección
        new_weights = [
            w * self.sensor.compute_prob_of_measure(p.current_pose, sensor['ranges'], sensor['scan_angles'])
            for w, p in zip(self.weights, self.particles)
        ]
        self.weights = np.array(new_weights) / sum(new_weights)


    def systematic_resample(self):
        # Implementación del algoritmo de Muestreo Estocástico Universal
        new_particles = []

        N = len(self.weights)
        positions = (np.arange(N) + self.rs.rand()) / N
        cumulative_sum = np.cumsum(self.weights)

        seeds = self.rs.permutation(N)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                new_particles.append(
                    NoisyDiffDriveRobot(self.particles[j].current_pose, self.radius, self.wheels_radius, self.wheels_distance, self.alpha, seeds[i])
                )
                i += 1
            else:
                j += 1

        self.particles = new_particles
        self.weights = np.ones(N) / N
        return self


    def update(self, odometry, sensor):
        self.prediction_step(odometry) # predicción
        self.correction_step(sensor) # corrección
        self.systematic_resample() # remuestreo

    def get_particles_poses(self):
        return np.vstack([p.current_pose for p in self.particles])

    def get_mean_robot(self):
        # Función que obtiene la pose promedio de las partículas
        poses = self.get_particles_poses()
        mean_pose = np.average(poses, axis=0, weights=self.weights)
        mean_robot = NoisyDiffDriveRobot(mean_pose, self.radius, self.wheels_radius, self.wheels_distance, self.alpha, self.seed)
        return mean_robot


class ExploreAndLocalize(BaseTask):

    def __init__(self, 
        safe_distance=0.5, min_linear_velocity=0.0, max_linear_velocity=0.5, max_angular_velocity=0.5,
        N_particles=100, alpha=0.1, occupation_threshold=0.5, seed=None
    ):
        self.safe_distance = safe_distance
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.min_linear_velocity = min_linear_velocity
        self.N_particles = N_particles
        self.alpha = alpha
        self.occupation_threshold = occupation_threshold
        self.seed = seed
        self.is_first_cycle = True

    def first_cycle(self, global_state):
        self.pf = ParticleFilterLocalization(
            self.N_particles,
            global_state["map"],
            global_state["robot"].radius,
            global_state["robot"].wheels_radius,
            global_state["robot"].wheels_distance,
            self.alpha,
            global_state["sensor"].sensor_offset,
            global_state["sensor"].num_scans,
            global_state["sensor"].start_angle,
            global_state["sensor"].end_angle,
            global_state["sensor"].min_range,
            global_state["sensor"].max_range,
            self.occupation_threshold,
            seed=self.seed
        )

    def run_cycle(self, global_state):
        if self.is_first_cycle:
            self.first_cycle(global_state)
            self.is_first_cycle = False

        ranges = global_state["sensor"].ranges
        angles = global_state["sensor"].scan_angles

        # Identify dangerous regions
        close_points = (ranges < self.safe_distance) & (~np.isnan(ranges))
        if sum(close_points) > len(ranges) * 3 / 4:
            # If the robot is surrounded by obstacles, stop
            output = {
                "linear_velocity": self.min_linear_velocity,
                "angular_velocity": self.max_angular_velocity,
            }
            
        elif np.any(close_points):
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
        
        output = {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
        }

        # Update the particle filter
        odometry = {
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "dt": global_state["sample_time"],
        }
        sensor = {
            "ranges": ranges,
            "scan_angles": angles,
        }
        self.pf.update(odometry, sensor)

        return output
    
    
        