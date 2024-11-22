#!/bin/bash -ex

OUTPUTS_DIR=outputs/fastslam
PLOTS_DIR=$OUTPUTS_DIR/plots
rm -rf $OUTPUTS_DIR
mkdir -p $PLOTS_DIR
python -m probabilistic_robotics.scripts.fastslam \
    --sensor_data data/sensor_data.dat\
    --world_data data/world.dat \
    --plots_dir $PLOTS_DIR
# ffmpeg -r 10 -i $PLOTS_DIR/plot_%03d.png $OUTPUTS_DIR/fastslam.mp4