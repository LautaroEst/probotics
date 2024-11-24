#!/bin/bash -ex

OUTPUTS_DIR=outputs/planning
PLOTS_DIR=$OUTPUTS_DIR/plots
rm -rf $OUTPUTS_DIR
mkdir -p $PLOTS_DIR
python -m probabilistic_robotics.scripts.planning \
    --map_path data/planning/map.mat \
    --start 34,23 \
    --goal 16,41 \
    --plots_dir $PLOTS_DIR \
