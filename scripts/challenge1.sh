#!/bin/bash -ex

# Params
initial_pose="10,7,0"
# map=1er_piso_ala_sur_alig.tiff
map=mapa_fiuba_1p.tiff
map_resolution=0.04 # meters per pixel
scans_scale_factor=10

# Directories
OUTPUTS_DIR=outputs/simulation
PLOTS_DIR=$OUTPUTS_DIR/plots
CONFIGS_DIR=configs

# Tasks
TASKS="run_in_circles"
TASKS=$(echo $TASKS | sed "s/\([^,]\+\)/$CONFIGS_DIR\/\1/g")

# Run simulation
rm -rf $OUTPUTS_DIR
mkdir -p $PLOTS_DIR
python -m probabilistic_robotics.scripts.simulator \
    --initial_pose $initial_pose \
    --map data/simulation/$map \
    --map_resolution $map_resolution \
    --scans_scale_factor $scans_scale_factor \
    --tasks $TASKS \
    --plots_dir $PLOTS_DIR \
    --seed 1234
