
import numpy as np

def diffdrive(pose, action, l):
    """
    Argumentos
    ----------
        x, y, theta: pose del robot
        v_l, v_r: velocidades de la rueda izquierda y derecha
        dt: intervalo de tiempo en movimiento
        l: distancia entre las ruedas del robot
    
    Devuelve
    --------
        x_n, y_n, theta_n: la nueva pose del robot
    """
    
    x, y, theta = pose
    v_l, v_r, dt = action
    
    w = (v_r - v_l) / l # Velocidad angular 
    if w == 0:
        v = v_r
        x_n = x + v * np.cos(theta) * dt
        y_n = y + v * np.sin(theta) * dt
        theta_n = theta
    else:
        R = l / 2 * (v_l + v_r) / (v_r - v_l) # distancia del ICC al robot
        icc_x = x - R * np.sin(theta) # ICC_x
        icc_y = y + R * np.cos(theta) # ICC_y
        x_n = np.cos(w * dt) * (x - icc_x) - np.sin(w * dt) * (y - icc_y) + icc_x
        y_n = np.sin(w * dt) * (x - icc_x) + np.cos(w * dt) * (y - icc_y) + icc_y
        theta_n = theta + w * dt
    
    return np.array([x_n, y_n, theta_n])