import importlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
import yaml

def read_sensor_data(filename):
    t = 0
    data = {}
    is_first = True
    with open(filename, "r") as f:
        for line in f:
            line_s = line.split('\n') # remove the new line character
            line_spl = line_s[0].split(' ') # split the line
            if line_spl[0] == 'ODOMETRY':
                if not is_first:
                    data[t] = {'odom': odom, 'sensor': sensor}
                    t += 1
                else:
                    is_first = False
                odom = {'r1':float(line_spl[1]),'t':float(line_spl[2]),'r2':float(line_spl[3])}
                sensor = {"id": [], "range": [], "bearing": []}
            elif line_spl[0] == 'SENSOR':
                sensor["id"].append(int(line_spl[1]))
                sensor["range"].append(float(line_spl[2]))
                sensor["bearing"].append(float(line_spl[3]))
            else:
                raise ValueError("Invalid data type")
        data[t] = {'odom': odom, 'sensor': sensor}
    return data


def load_yaml(path):
    with open(path, "r") as f:
        file = yaml.safe_load(f)
    if file is None:
        return {}
    return file

def read_tasks(tasks_cfgs):
    tasks = []
    module = importlib.import_module('.tasks', package=__package__)
    for cfg in tasks_cfgs:
        cfg = load_yaml(f"../configs/{cfg}.yaml")
        task_name = cfg.pop('task')
        task_cls = getattr(module, task_name)
        task = task_cls(**cfg)
        tasks.append(task)
    return tasks

def evaluate_lognormal(x, mu, sigma):
    return -np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((x - mu) / sigma) ** 2

def plot_ellipse(mu, sigma, ax, color="k"):
    """
    Draws ellipse from xy of covariance matrix
    """

    # Compute eigenvalues and associated eigenvectors
    vals, vecs = np.linalg.eigh(sigma)
    # Compute "tilt" of ellipse using first eigenvector
    x, y = vecs[:, 0]
    theta = np.degrees(np.arctan2(y, x))

    # Eigenvalues give length of ellipse along each eigenvector
    w, h = 2 * np.sqrt(vals)

    ellipse = Ellipse(mu, w, h, angle=theta, color=color)
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)


def plot_landmarks_with_ellipsoids(ax, robot_pose, landmarks):
    for i, landmark in landmarks[landmarks["observed"]].iterrows():
        plot_ellipse(landmark['mu'], landmark['sigma'], ax, color='b')
        x = np.asarray([robot_pose[0], landmark['mu'][0]])
        y = np.asarray([robot_pose[1], landmark['mu'][1]])
        ax.plot(x, y, 'b--')