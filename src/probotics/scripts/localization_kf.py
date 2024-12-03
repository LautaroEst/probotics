
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from ..localization import KalmanFilter
from ..utils import read_sensor_data, plot_ellipse


Qt = np.array([[0.2, 0.0, 0.0],[0.0, 0.2, 0.0],[0.0, 0.0, 0.02]])
sensor_noise = 0.5


def  main(sensor_data, world_data, plots_dir):

    # Initialize Kalman Filter
    kf = KalmanFilter(np.zeros(3), np.eye(3), world_data, Qt, sensor_noise)

    # Read data
    sensor_data = read_sensor_data(sensor_data)

    # Plot
    for t in tqdm(range(len(sensor_data)),total=len(sensor_data)):

        # Update Kalman Filter
        kf.update(sensor_data[t]['odom'], sensor_data[t]['sensor'])
        mu, sigma = kf.mu, kf.sigma

        fig, ax = plt.subplots()
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Localización con filtro Kalman y landmarks. Timestamp: " + str(t))
        ax.plot(kf.sensor.landmarks['x'], kf.sensor.landmarks['y'], 'ko', markersize=5)
        plot_ellipse(mu[:2], sigma[:2, :2], ax, color='r')
        kf.robot.plot(ax, radius=0.5, color='k')
        plt.savefig(f"{plots_dir}/plot_{t:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sensor_data", help="File containing sensor data")
    parser.add_argument("--world_data", help="File containing world data")
    parser.add_argument("--plots_dir", help="Directory to save plots", default="plots")

    args = parser.parse_args()
    main(args.sensor_data, args.world_data, args.plots_dir)