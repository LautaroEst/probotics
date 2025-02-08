
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from ..robots.noisyodom import NoisyOdometryRobot
from ..sensors.landmarks import LandmarkIdentificator

class ParticleFilter:

    def __init__(self, N, world_data_path, odometry_noise_params, measurement_noise, radius=0.5, seed=None):
        rs = np.random.RandomState(seed)
        particles = []
        for i in range(N):
            initial_pose = np.array([rs.randn(), rs.randn(), rs.rand() * 2 * np.pi - np.pi])
            p = NoisyOdometryRobot(initial_pose, odometry_noise_params, radius, seed+i)
            particles.append(p)

        self.radius = radius
        self.N_init = N
        self.particles = particles
        self.weights = np.ones(N) / N
        self.sensor = LandmarkIdentificator.from_file(world_data_path, measurement_noise)
        self.world_data_path = world_data_path
        self.odometry_noise_params = odometry_noise_params
        self.measurement_noise = measurement_noise
        self.rs = rs


    def prediction_step(self, odometry):
        # Implementación del paso de predicción
        for particle in self.particles:
            particle.apply_movement(odometry['r1'], odometry['t'], odometry['r2'])


    def correction_step(self, sensor):
        # Implementación del paso de corrección
        new_weights = [
            w * self.sensor.measurement_prob_range(p.current_pose, sensor['id'], sensor['range'])
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
                    NoisyOdometryRobot(self.particles[j].current_pose, self.odometry_noise_params, self.radius, seeds[i])
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
        mean_robot = NoisyOdometryRobot(mean_pose, self.odometry_noise_params, self.radius)
        return mean_robot


    
    

    