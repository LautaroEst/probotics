
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
            initial_pose = np.array([rs.rand() * 15, rs.rand() * 15, rs.rand() * 2 * np.pi - np.pi])
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
        for particle in self.particles:
            particle.apply_movement(odometry['r1'], odometry['t'], odometry['r2'])

    def correction_step(self, sensor):
        new_weights = []
        for particle in self.particles:
            logprob = self.sensor.measurement_prob_range(particle.current_pose, sensor['id'], sensor['range'])
            new_weights.append(logprob)
        self.weights = softmax(new_weights)

    def update(self, odometry, sensor):

        # Paso de predicción
        self.prediction_step(odometry)

        # Paso de corrección
        self.correction_step(sensor)

        # Remuestreo usando Muestreo Estocástico Universal
        self.systematic_resampling()

    def get_mean_robot(self):
        x = 0.0
        y = 0.0
        theta = 0.0
        
        for p in self.particles:
            px, py, ptheta = p.current_pose
            x += px
            y += py
            theta += ptheta
            
        x /= len(self.particles)
        y /= len(self.particles)
        theta /= len(self.particles)
        
        mean_pos = np.array([x, y, theta])
        mean_robot = NoisyOdometryRobot(mean_pos, self.odometry_noise_params, self.radius)
        return mean_robot
    
    def get_best_particle(self):
        best_particle = self.particles[np.argmax(self.weights)]
        return best_particle

    def get_particles_poses(self):
        return np.vstack([p.current_pose for p in self.particles])
    
    def systematic_resampling(self):
        
        N = len(self.particles)
        cum_w = np.cumsum(self.weights)

        seed = self.rs.rand() / N
        pointers = np.arange(N) / N + seed

        new_particles = []
        new_weights = []
        for point in pointers:
            i = 0
            while point > cum_w[i]:
                i += 1
            new_particles.append(self.particles[i])
            new_weights.append(1/N)

        # Normalize weights    
        new_weights = np.array(new_weights) / np.sum(new_weights)
        self.particles = new_particles