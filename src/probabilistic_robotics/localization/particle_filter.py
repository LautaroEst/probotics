
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from ..robots.noisyodom import NoisyOdometryRobot
from ..sensors.landmarks import LandmarkIdentificator

class ParticleFilter:

    def __init__(self, N, world_data_path, odometry_noise_params, measurement_noise, seed=None):
        rs = np.random.RandomState(seed)
        particles = []
        for i in range(N):
            initial_pose = np.array([rs.rand() * 15, rs.rand() * 15, rs.rand() * 2 * np.pi - np.pi])
            p = NoisyOdometryRobot(initial_pose, odometry_noise_params, seed+i)
            particles.append(p)

        self.N_init = N
        self.particles = particles
        self.weights = np.ones(N) / N
        self.sensor = LandmarkIdentificator.from_file(world_data_path, measurement_noise)
        self.world_data_path = world_data_path
        self.odometry_noise_params = odometry_noise_params
        self.measurement_noise = measurement_noise
        self.rs = rs

    def predict(self, sensor_data):

        history = [(0, [(*p.current_pose, weight) for p, weight in zip(self.particles, self.weights)])]
        for t in tqdm(range(len(sensor_data)),total=len(sensor_data)):
            
            odometry = sensor_data[t]['odom']
            sensor = sensor_data[t]['sensor']

            new_weights = []
            for particle in self.particles:
                
                # motion update (prediction)
                particle.apply_movement(odometry['r1'], odometry['t'], odometry['r2'])

                # measurement update
                logprob = self.sensor.measurement_prob_range(particle.current_pose, sensor['id'], sensor['range'])
                new_weights.append(logprob)
            
            self.weights = softmax(new_weights)
                
            # Remuestreo usando Muestreo Estocástico Universal
            self.systematic_resampling()

            # Guardamos el estado de las partículas
            history = [(t+1, [(*p.current_pose, weight) for p, weight in zip(self.particles, self.weights)])]

        return history
    
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
            initial_pose = self.particles[i].current_pose
            particle = NoisyOdometryRobot(initial_pose, self.odometry_noise_params, point)
            new_particles.append(particle)
            new_weights.append(1/N)

        # Normalize weights    
        new_weights = np.array(new_weights) / np.sum(new_weights)
        self.particles = new_particles