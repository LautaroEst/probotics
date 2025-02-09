import numpy as np
import pandas as pd
from copy import deepcopy

from ..robots import GaussianNoisyOdometryRobot
from ..sensors.landmarks import GaussianLandmarkIdentificator



class Particle:
    def __init__(self, robot, sensor, weight):
        self.robot = robot
        self.sensor = sensor
        self.weight = weight
    
    def copy(self):
        new_robot = GaussianNoisyOdometryRobot(None, self.robot.noise_params, radius=0.5, seed=self.robot.seed)
        new_robot._current_pose = deepcopy(self.robot._current_pose)
        new_robot._pose_history = deepcopy(self.robot._pose_history)
        new_sensor = GaussianLandmarkIdentificator(self.sensor.N_landmarks, self.sensor.sensor_noise)
        new_sensor.landmarks = self.sensor.landmarks.copy()
        return Particle(new_robot, new_sensor, self.weight)



class FastSLAM:
    
    def __init__(self, N_particles, N_landmarks, odometry_noise_params, measurement_noise, use_neff=False, seed=None):
        rs = np.random.RandomState(seed)
        particles = []
        for i in range(N_particles):
            initial_pose = np.array([0, 0, 0])
            robot = GaussianNoisyOdometryRobot(initial_pose, odometry_noise_params, radius=0.5, seed=seed+i)
            sensor = GaussianLandmarkIdentificator(N_landmarks, measurement_noise)
            particles.append(Particle(robot, sensor, 1/N_particles))

        self.N_initial_particles = N_particles
        self.N_landmarks = N_landmarks
        self.particles = particles
        self.odometry_noise_params = odometry_noise_params
        self.measurement_noise = measurement_noise
        self.rs = rs
        self.use_neff = use_neff

    def update(self, odometry, sensor):

        # Prediction step
        self.prediction_step(odometry)

        # Correction step
        self.correction_step(sensor)

        # Resampling step
        self.resampling_step()


    def prediction_step(self, odometry):
        for particle in self.particles:
            particle.robot.apply_movement(odometry['r1'], odometry['t'], odometry['r2'])

    def correction_step(self, sensor):
        sorted_ids = np.argsort(sensor["id"])
        ids = np.array(sensor["id"])[sorted_ids]
        ranges = np.array(sensor["range"])[sorted_ids]
        bearings = np.array(sensor["bearing"])[sorted_ids]

        Qt = np.eye(2) * self.measurement_noise

        for i, particle in enumerate(self.particles):

            for id_, r, b in zip(ids, ranges, bearings):
                
                if not particle.sensor.landmarks.loc[id_,"observed"]:
                    x, y, theta = particle.robot.current_pose
                    
                    # TODO: Initialize its position based on the measurement and the current robot pose:
                    particle.sensor.landmarks.at[id_,"mu"] = np.array([x + r * np.cos(theta + b), y + r * np.sin(theta + b)])
                    
                    # Get the Jacobian with respect to the landmark position
                    h, H = particle.sensor.measurement_model(particle.robot.current_pose, id_)

                    # TODO: Initialize the covariance for this landmark
                    particle.sensor.landmarks.at[id_,"sigma"] = np.linalg.inv(H) @ Qt @ np.linalg.inv(H).T

                    # Indicate that this landmark has been observed
                    particle.sensor.landmarks.loc[id_,"observed"] = True
                
                else:
                    # Get the Jacobian with respect to the landmark position
                    expected_Z, H = particle.sensor.measurement_model(particle.robot.current_pose, id_)
                    expected_Z = expected_Z.reshape(2,1)
                    
                    # TODO: Compute the measurement covariance
                    sigma_hat = particle.sensor.landmarks.loc[id_,'sigma']
                    Q = H @ sigma_hat @ H.T + Qt
                    
                    # TODO: Calculate the Kalman gain
                    K = sigma_hat @ H.T @ np.linalg.inv(Q)
                    
                    # TODO: Update the mean and covariance of the EKF for this landmark
                    mu_hat = particle.sensor.landmarks.loc[id_,"mu"].reshape(2,1)
                    measured_Z = np.array([r, b]).reshape(2,1)
                    mu = mu_hat + K @ (measured_Z - expected_Z)
                    particle.sensor.landmarks.at[id_,"mu"] = mu.flatten()
                    particle.sensor.landmarks.at[id_,"sigma"] = (np.eye(2) - K @ H) @ sigma_hat

                    exp = np.exp( -(measured_Z - expected_Z).T @ np.linalg.inv(Q) @ (measured_Z - expected_Z) / 2)
                    exp = float(exp)
                    particle.weight = particle.weight * np.sqrt( 1 / (2 * np.pi * np.abs(np.linalg.det(Q)))) * exp
        

    def resampling_step(self):

        N_particles = len(self.particles)
        weights = np.array([particle.weight for particle in self.particles])
        weights = weights / np.sum(weights)

        if self.use_neff:
            neff = 1 / np.sum(weights ** 2)
            if neff > N_particles / 2:
                for w, p in zip(weights, self.particles):
                    p.weight = w
                return
            
        cum_w = np.cumsum(weights)
        weight_sum = cum_w[-1]
        step = weight_sum / N_particles
        position = self.rs.rand() * weight_sum
        idx = 0

        new_particles = []
        for i in range(N_particles):
            position += step
            if position > weight_sum:
                position -= weight_sum
                idx = 0
            while position > cum_w[idx]:
                idx += 1
            new_particle = self.particles[idx].copy()
            new_particle.weight = 1 / N_particles
            new_particles.append(new_particle)
        self.particles = new_particles

    def get_particles_poses(self):
        return np.vstack([p.robot.current_pose for p in self.particles])
    
    def get_weights(self):
        return np.array([p.weight for p in self.particles])

    def get_best_particle(self):
        return max(self.particles, key=lambda p: p.weight)


        



                    

        
        
