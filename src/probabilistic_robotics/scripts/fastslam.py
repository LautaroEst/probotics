
from ..robots.noisyodom import NoisyOdometryRobot


def main(sensor_data, world_data, plots_dir):
    pass


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--sensor_data", help="File containing sensor data")
    parser.add_argument("--world_data", help="File containing world data")
    parser.add_argument("--plots_dir", help="Directory to save plots", default="plots")

    args = parser.parse_args()
    main(args.sensor_data, args.world_data, args.plots_dir)