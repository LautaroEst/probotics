#!/bin/bash -ex

OUTPUTS_DIR=outputs/fastslam
PLOTS_DIR=$OUTPUTS_DIR/plots
rm -rf $OUTPUTS_DIR
mkdir -p $PLOTS_DIR
python -m probabilistic_robotics.scripts.fastslam \
    --N 200 \
    --odometry_noise_params "0.005,0.01,0.005" \
    --sensor_noise 0.1 \
    --sensor_data data/slam/sensor_data.dat\
    --world_data data/slam/world.dat \
    --plots_dir $PLOTS_DIR
ffmpeg -r 10 -i $PLOTS_DIR/plot_%03d.png $OUTPUTS_DIR/fastslam.mp4