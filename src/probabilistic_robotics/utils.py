import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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


def read_world_data(filename):
    return pd.read_csv(filename, sep=' ', header=None, names=["id", "x", "y"]).set_index("id")


def plot_robot(xt, ax, radius=1, **kwargs):
    x, y, theta = xt
    circle = plt.Circle((x,y), radius, fill=False, color=kwargs.get('color', None))
    ax.add_artist(circle)
    ax.plot([x, x + radius * np.cos(theta)], [y, y + radius * np.sin(theta)], linewidth=2, **kwargs)
