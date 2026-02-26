#!/bin/bash
# conda activate gsworld
# Define dataset path
DATA_DIR=/home/guangqi/wanglab/GSWorld/data/xhand0107
# LABEL=xhand

# Go to scripts
cd /home/guangqi/wanglab/GSWorld/gsworld/real2sim/scripts

# Run colmap and rescale
python sfm.py --source_path "$DATA_DIR" -v
python aruco_rescale.py --source_path "$DATA_DIR"

# Go to gaussian splatting submodule
cd /home/guangqi/wanglab/GSWorld/submodules/gaussian-splatting

# Training
CUDA_VISIBLE_DEVICES=0 python train.py -s "$DATA_DIR" --disable_viewer # --label "$LABEL"