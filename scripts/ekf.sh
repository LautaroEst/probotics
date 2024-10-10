#!/bin/bash -ex

OUTPUTS_DIR=outputs/kalman_filter
PLOTS_DIR=$OUTPUTS_DIR/plots
rm -rf $OUTPUTS_DIR
mkdir -p $PLOTS_DIR
python -m probabilistic_robotics.scripts.localization_kf \
    --sensor_data data/sensor_data.dat\
    --world_data data/world.dat \
    --plots_dir $PLOTS_DIR
ffmpeg -r 10 -i $PLOTS_DIR/plot_%03d.png $OUTPUTS_DIR/pf.mp4