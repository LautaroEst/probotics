#!/bin/bash -ex

# Params
initial_pose="0,0,0"
map=mapa_fiuba_1p.tiff
scans_scale_factor=10

# Directories
OUTPUTS_DIR=outputs/simulation
PLOTS_DIR=$OUTPUTS_DIR/plots
CONFIGS_DIR=configs

# Tasks
declare -a TASKS=(run_in_circles)
__TASKS="${TASKS[@]/#/$CONFIGS_DIR/}"
_TASKS=$(echo "${__TASKS[@]}" | sed 's/ /,/g')

# Run simulation
rm -rf $OUTPUTS_DIR
mkdir -p $PLOTS_DIR
python -m probabilistic_robotics.scripts.simulator \
    --initial_pose $initial_pose \
    --map data/simulation/$map \
    --scans_scale_factor $scans_scale_factor \
    --tasks $_TASKS \
    --plots_dir $PLOTS_DIR \
    --seed 1234
