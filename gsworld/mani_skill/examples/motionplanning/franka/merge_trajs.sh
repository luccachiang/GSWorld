#!/bin/bash

# Set output file path
# OUTPUT_PATH="/data/guangqi/maniskill/demos/PnpSingleYCBBox-v1-merged/motionplanning/PnpSingleYCBBox_500.h5"
# OUTPUT_PATH="/data/guangqi/maniskill/demos/PourMustard-v1-merged/motionplanning/PourMustard_500.h5"
OUTPUT_PATH="/data/guangqi/maniskill/demos/PnpAlignYCB-v1-merged/motionplanning/PnpAlignYCB_500.h5"
# OUTPUT_PATH="/data/guangqi/maniskill/demos/PnpStackYCB-v1-merged/motionplanning/PnpStackYCB_500.h5"

# Set input directories here
# INPUT_DIRS=(
#     "/data/guangqi/maniskill/demos/PnpSingleYCBBox-v1/motionplanning/20250413_183631.h5"
#     "/data/guangqi/maniskill/demos/PnpSingleYCBBox-v1/motionplanning/20250414_144800.h5"
#     "/data/guangqi/maniskill/demos/PnpSingleYCBBox-v1/motionplanning/20250413_205032.h5"
#     "/data/guangqi/maniskill/demos/PnpSingleYCBBox-v1/motionplanning/20250413_213531.h5"
#     "/data/guangqi/maniskill/demos/PnpSingleYCBBox-v1/motionplanning/20250413_221832.h5"
# )
INPUT_DIRS=(
    # "/data/guangqi/maniskill/demos/PnpSingleYCBBox-v1/motionplanning/"
    # "/data/guangqi/maniskill/demos/PourMustard-v1/motionplanning/"
    "/data/guangqi/maniskill/demos/PnpAlignYCB-v1/motionplanning/"
    # "/data/guangqi/maniskill/demos/PnpStackYCB-v1/motionplanning/"
)

# Optional: pattern of h5 files to look for
PATTERN="*.h5"

# Join input directories into a single string
INPUT_DIRS_STR=""
for dir in "${INPUT_DIRS[@]}"; do
    INPUT_DIRS_STR+="$dir "
done

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Run the Python merging script
cd /home/guangqi/wanglab/projects/GSWorld_private/third_party/ManiSkill/mani_skill/trajectory
python merge_trajectory.py \
    -i $INPUT_DIRS_STR \
    -o "$OUTPUT_PATH" \
    -p "$PATTERN"
