#!/bin/bash -ex

OUTPUTS_DIR=outputs/planning
PLOTS_DIR=$OUTPUTS_DIR/plots
rm -rf $OUTPUTS_DIR

start=34,23
goal=16,41
threshold=0.5

mkdir -p $PLOTS_DIR/dijkstra
python -m probabilistic_robotics.scripts.planning \
    --map_path data/planning/map.mat \
    --start $start \
    --goal $goal \
    --method dijkstra \
    --threshold $threshold \
    --plots_dir $PLOTS_DIR/dijkstra
ffmpeg -r 10 -i $PLOTS_DIR/dijkstra/plot_%04d.png $OUTPUTS_DIR/dijkstra.mp4

for factor in 1 2 5 10; do
    mkdir -p $PLOTS_DIR/a_star/factor_$factor
    python -m probabilistic_robotics.scripts.planning \
        --map_path data/planning/map.mat \
        --start $start \
        --goal $goal \
        --method a_star \
        --factor $factor \
        --threshold $threshold \
        --plots_dir $PLOTS_DIR/a_star/factor_$factor
    ffmpeg -r 10 -i $PLOTS_DIR/a_star/factor_$factor/plot_%04d.png $OUTPUTS_DIR/a_star_h2=$factor.mp4
done