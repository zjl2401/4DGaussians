#!/bin/bash
VIDEO_PATH=""
RUN_NAME=""
ITERATIONS=30000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video) VIDEO_PATH="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    *) shift ;;
  esac
done

if [ -z "$RUN_NAME" ]; then RUN_NAME=$(basename "$VIDEO_PATH" | cut -f 1 -d "."); fi

DATASET_DIR="data/monocular_custom/$RUN_NAME"
OUTPUT_DIR="output/monocular_custom/$RUN_NAME"

echo "[INFO] Processing: $VIDEO_PATH"

python scripts/prepare_monocular_spin_demo.py --video "$VIDEO_PATH" --out "$DATASET_DIR"

python train.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" --configs arguments/dnerf/monocular_spin_demo.py --iterations "$ITERATIONS" --save_iterations "$ITERATIONS" --white_background

python render.py -s "$DATASET_DIR" -m "$OUTPUT_DIR" --iteration "$ITERATIONS" --skip_train --configs arguments/dnerf/monocular_spin_demo.py