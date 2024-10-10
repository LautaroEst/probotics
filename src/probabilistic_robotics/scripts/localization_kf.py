
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from tqdm import tqdm

from ..localization import KalmanFilter
from ..utils import read_sensor_data, read_world_data, plot_robot


Qt = np.array([[0.2, 0.0, 0.0],[0.0, 0.2, 0.0],[0.0, 0.0, 0.02]])
sensor_noise = 0.5


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

    ellipse = Ellipse(mu, w, h, theta, color=color)  # color="k")
    ellipse.set_clip_box(ax.bbox)
    ellipse.set_alpha(0.2)
    ax.add_artist(ellipse)



def  main(sensor_data, world_data, plots_dir):

    sensor_data = read_sensor_data(sensor_data)
    world_data = read_world_data(world_data)

    pf = KalmanFilter(mu0=np.zeros(3), sigma0=np.eye(3))
    print("Fitting Kalman filter...")
    history = pf.fit(sensor_data, world_data, odometry_noise=Qt, sensor_noise=sensor_noise)

    # Plot
    print("Creating plots...")
    for t, mu, sigma in tqdm(history):

        fig, ax = plt.subplots()
        ax.set_xlim(-2, 12)
        ax.set_ylim(-2, 12)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title("Filtro Kalman para ubicar al robot. Timestamp: " + str(t))
        ax.plot(world_data['x'], world_data['y'], 'ko', markersize=5)
        plot_ellipse(mu[:2], sigma[:2, :2], ax, color='r')
        plot_robot(mu, ax, radius=0.5, color='k')
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