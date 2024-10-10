#!/bin/bash -ex

number_of_particles=200 # Number of particles
noise_params="0.1,0.1,0.05,0.05" # Noise parameters for the motion model

OUTPUTS_DIR=outputs/particle_filter
PLOTS_DIR=$OUTPUTS_DIR/plots
rm -rf $OUTPUTS_DIR
mkdir -p $PLOTS_DIR
python -m probabilistic_robotics.scripts.localization_particle_filter \
    --sensor_data data/sensor_data.dat\
    --world_data data/world.dat \
    --N $number_of_particles \
    --noise_params $noise_params \
    --seed 1234 \
    --plots_dir $PLOTS_DIR
ffmpeg -r 10 -i $PLOTS_DIR/plot_%03d.png $OUTPUTS_DIR/pf.mp4