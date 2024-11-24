import numpy as np

class Robot:

    def __init__(self, wheels_radius, wheels_distance, alpha, seed=None):
        self.wheels_radius = wheels_radius
        self.wheel_distance = wheels_distance
        self.alpha = alpha
        self._rs = np.random.RandomState(seed)

    @property
    def current_pose(self):
        return self._current_pose
    
    @current_pose.setter
    def current_pose(self, pose):
        self._current_pose = pose
    
    def inverse_kinematics(self, v, w):
        # Velocidades de las ruedas
        wL = (v - w * self.wheel_distance / 2) / self.wheel_radius
        wR = (v + w * self.wheel_distance / 2) / self.wheel_radius

        # Velocidades con ruido
        v = self.wheels_radius * (wR + wL) / 2 + self._rs.randn() * self.alpha * (wL**2 + wR**2)
        w = self.wheels_radius * (wR - wL) / self.wheel_distance + self._rs.randn() * self.alpha * (wL**2 + wR**2)
        vL = v - w * self.wheel_distance / 2
        vR = v + w * self.wheel_distance / 2
        
        return vL, vR
    
    def diffdrive(self, x, y, theta, vL, vR, dt):
        w = (vR - vL) / self.wheel_distance # Velocidad angular 
        if w == 0:
            v = vR
            x_n = x + v * np.cos(theta) * dt
            y_n = y + v * np.sin(theta) * dt
            theta_n = theta
        else:
            R = self.wheel_distance / 2 * (vL + vR) / (vR - vL) # distancia del ICC al robot
            icc_x = x - R * np.sin(theta) # ICC_x
            icc_y = y + R * np.cos(theta) # ICC_y
            x_n = np.cos(w * dt) * (x - icc_x) - np.sin(w * dt) * (y - icc_y) + icc_x
            y_n = np.sin(w * dt) * (x - icc_x) + np.cos(w * dt) * (y - icc_y) + icc_y
            theta_n = theta + w * dt
        return x_n, y_n, theta_n


    def apply_diffdrive(self, v, w, dt):

        # Obtenemos la pose actual
        x, y, theta = self._current_pose

        # Calculamos las velocidades lineales de las ruedas
        vL, vR = self.inverse_kinematics(v, w)
        
        # Calculamos la pose después de aplicar una velocidad v y w durante un intervalo de tiempo dt
        x_n, y_n, theta_n = self.diffdrive(x, y, theta, vL, vR, dt)
        
        # Actualizamos la pose actual
        self.current_pose = np.array([x_n, y_n, theta_n])


    def body2world(self, measurements):
        x, y, theta = self.current_pose

        # Matriz de cambio de base de la terna global a la terna local.
        T = np.array([
            [np.cos(theta), -np.sin(theta), x], 
            [np.sin(theta), np.cos(theta), y], 
            [0, 0, 1]
        ])
        
        measurements_1 = np.hstack((measurements[:,:2], np.ones((measurements.shape[0],1)))) # Concateno columna de 1's
        measurements_global = measurements_1 @ T.T
        measurements_global[:,2] = measurements[:,2] + theta
        return measurements_global



class Lidar:

    def __init__(self, sensor_offset, num_scans, start_angle, end_angle, max_range, seed=None):
        self.sensor_offset = sensor_offset
        self.num_scans = num_scans
        self.start_angle = start_angle
        self.end_angle = end_angle
        self.scan_angles = np.linspace(start_angle, end_angle, num_scans)
        self.max_range = max_range
        self._rs = np.random.RandomState(seed)

    def measure(self, pose):
        pass
    