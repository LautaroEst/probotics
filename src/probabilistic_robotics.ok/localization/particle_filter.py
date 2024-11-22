
import numpy as np
from scipy.special import softmax
from tqdm import tqdm

class Robot:

    """ creates robot and initializes location/orientation 
    """

    sigma_r = .2

    def __init__(self, weight, random_state):
        rs = np.random.RandomState(random_state)
        self.x = rs.rand() * 15  # initial x position
        self.y = rs.rand() * 15 # initial y position
        self.orientation = rs.rand() * 2 * np.pi - np.pi # initial orientation
        self.weight = weight
        self.random_state = random_state
        
    def mov_odom(self, r1, t, r2, noise_params):
        
        """ Takes in Odometry data
            
        odom: diccionario que contiene r1, r2, t (rotación 1, rotación 2, traslación)
        noise: pesos del ruido    
        
        """
        rs = np.random.RandomState(self.random_state)
    
        # Calculate the distance and Guassian noise      
        dist  = t
        
        # calculate delta rotation 1 and delta rotation 1
        delta_rot1  = r1
        delta_rot2 = r2
        
        # noise sigma for delta_rot1 
        sigma_delta_rot1 = noise_params[0] * abs(delta_rot1)  + noise_params[1] * abs(dist)
        delta_rot1_noisy = delta_rot1 + rs.randn() * sigma_delta_rot1

        # noise sigma for translation
        sigma_translation = noise_params[2] * abs(dist)  + noise_params[3] * abs(delta_rot1+delta_rot2)
        translation_noisy = dist + rs.randn() * sigma_translation

        # noise sigma for delta_rot2
        sigma_delta_rot2 = noise_params[0] * abs(delta_rot2)  + noise_params[1] * abs(dist)
        delta_rot2_noisy = delta_rot2 + rs.randn() * sigma_delta_rot2

        # Estimate of the new position of the robot
        self.x = self.x  + translation_noisy * np.cos(self.orientation + delta_rot1_noisy)
        self.y = self.y  + translation_noisy * np.sin(self.orientation + delta_rot1_noisy)
        self.orientation = self.orientation + delta_rot1_noisy + delta_rot2_noisy
        
        #if new_orientation < -math.pi or new_orientation >= math.pi:
        #    raise ValueError, 'Orientation must be in [-pi..pi]'
        if self.orientation > np.pi:
            self.orientation = self.orientation -2 * np.pi
        elif self.orientation < -np.pi:
            self.orientation = self.orientation + 2 * np.pi

    @staticmethod
    def _evaluate_lognormal(x, mu, sigma):
        return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x - mu) / sigma) ** 2

    def measurement_prob_range(self, indices, ranges, world_data):
        """ computes the probability of a measurement 
        """
        # p = lambda x, mu: 1 / (np.sqrt(2*np.pi) * sigma_r) * np.exp(-(x - mu)**2 / 2 / sigma_r**2)
        
        N = len(indices)
        logprob = 0
        for i in range(N):
            x_landmark = world_data.loc[indices[i],'x']
            y_landmark = world_data.loc[indices[i],'y']
            mu = np.sqrt((self.x - x_landmark)**2 + (self.y - y_landmark)**2)
            logprob += self._evaluate_lognormal(ranges[i], mu, self.sigma_r)
        return logprob
    

class ParticleFilter:

    def __init__(self, N, random_state=None):
        self.N_init = N
        self.particles = [Robot(1/N, random_state+i) for i in range(N)]
        self.rs = np.random.RandomState(random_state)

    def fit(self, sensor_data, world_data, noise_params):

        history = [(0, [(p.x, p.y, p.orientation, p.weight) for p in self.particles])]
        for t in tqdm(range(len(sensor_data)),total=len(sensor_data)):
            
            odometry = sensor_data[t]['odom']
            sensor = sensor_data[t]['sensor']

            new_weights = []
            for particle in self.particles:
                
                # motion update (prediction)
                particle.mov_odom(odometry['r1'], odometry['t'], odometry['r2'], noise_params)

                # measurement update
                logprob = particle.measurement_prob_range(sensor['id'], sensor['range'], world_data)
                new_weights.append(logprob)
            
            new_weights = softmax(new_weights)
            for i in range(len(self.particles)):
                self.particles[i].weight = new_weights[i]
                
            # Remuestreo usando Muestreo Estocástico Universal
            self.systematic_resampling()

            # Guardamos el estado de las partículas
            history.append((t+1, [(p.x, p.y, p.orientation, p.weight) for p in self.particles]))

        return history
    
    def systematic_resampling(self):
        
        N = len(self.particles)
        w = np.array([particle.weight for particle in self.particles])
        cum_w = np.cumsum(w)

        seed = self.rs.rand() / N
        pointers = np.arange(N) / N + seed

        new_particles = []
        new_weights = []
        for point in pointers:
            i = 0
            while point > cum_w[i]:
                i += 1
            particle = Robot(weight=1/N, random_state=self.rs.randint(0, 1000))
            particle.x = self.particles[i].x
            particle.y = self.particles[i].y    
            particle.orientation = self.particles[i].orientation
            new_particles.append(particle)
            new_weights.append(1/N)

        # Normalize weights    
        new_weights = np.array(new_weights) / np.sum(new_weights)
        for i, particle in enumerate(new_particles):
            particle.weight = new_weights[i]
        
        self.particles = new_particles