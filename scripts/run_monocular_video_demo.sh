#!/usr/bin/env bash
set -euo pipefail

# Unified monocular video demo runner for Linux/WSL.
# Modes:
#   fixed_camera : static camera + rotating object
#   moving_camera: moving camera (with optional dynamic object) + COLMAP

usage() {
  cat <<'USAGE'
Usage:
  bash scripts/run_monocular_video_demo.sh --video <video.mp4> [options]

Required:
  --video PATH                  Input video path.

Options:
  --mode MODE                   fixed_camera | moving_camera (default: moving_camera)
  --run-name NAME               Experiment name (default: derived from video file name)
  --dataset-root PATH           Dataset root directory (default: data/monocular_custom)
  --exp-root PATH               Output experiment root (default: monocular_custom)

  # fixed_camera options
  --max-frames N                Max extracted frames (default: 120)
  --stride N                    Use every N-th frame (default: 2)
  --size N                      Resize short side to N (default: 800)

  # moving_camera options
  --fps F                       Frame extraction FPS for COLMAP path (default: 3)
  --colmap-video-interp N       Smooth camera interpolation for rendering (default: 160)

  --train-port PORT             Train viewer port (default: 6017)
  --skip-render                 Train only
  --help                        Show this message

Examples:
  # fixed camera + moving object (single video)
  bash scripts/run_monocular_video_demo.sh \
    --mode fixed_camera \
    --video /mnt/c/Users/me/Desktop/spin.mp4 \
    --run-name spin_case

  # moving camera (and possibly moving object)
  bash scripts/run_monocular_video_demo.sh \
    --mode moving_camera \
    --video /mnt/c/Users/me/Desktop/walkaround.mp4 \
    --run-name walk_case
USAGE
}

MODE="moving_camera"
VIDEO=""
RUN_NAME=""
DATASET_ROOT="data/monocular_custom"
EXP_ROOT="monocular_custom"
TRAIN_PORT=6017

MAX_FRAMES=120
STRIDE=2
SIZE=800

FPS=3
COLMAP_VIDEO_INTERP=160
SKIP_RENDER=0

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: $1" >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --video) VIDEO="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --exp-root) EXP_ROOT="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --stride) STRIDE="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --colmap-video-interp) COLMAP_VIDEO_INTERP="$2"; shift 2 ;;
    --train-port) TRAIN_PORT="$2"; shift 2 ;;
    --skip-render) SKIP_RENDER=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[ERROR] Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$VIDEO" ]]; then
  echo "[ERROR] --video is required." >&2
  usage
  exit 1
fi

if [[ ! -f "$VIDEO" ]]; then
  echo "[ERROR] Video not found: $VIDEO" >&2
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  base="$(basename "$VIDEO")"
  RUN_NAME="${base%.*}"
fi

DATASET_DIR="$DATASET_ROOT/$RUN_NAME"
EXP_NAME="$EXP_ROOT/$RUN_NAME"

cd "$(dirname "$0")/.."

echo "[INFO] Workspace: $(pwd)"
echo "[INFO] Mode: $MODE"
echo "[INFO] Video: $VIDEO"
echo "[INFO] Run name: $RUN_NAME"
echo "[INFO] Dataset: $DATASET_DIR"
echo "[INFO] Experiment: $EXP_NAME"

require_cmd python

if [[ "$MODE" == "fixed_camera" ]]; then
  echo "[STEP 1/3] Prepare fixed-camera monocular dataset..."
  python scripts/prepare_monocular_spin_demo.py \
    --video "$VIDEO" \
    --out "$DATASET_DIR" \
    --max_frames "$MAX_FRAMES" \
    --stride "$STRIDE" \
    --size "$SIZE"

  echo "[STEP 2/3] Train 4DGaussians (D-NeRF style)..."
    -s "$DATASET_DIR" \
    --expname "$EXP_NAME" \
    --configs arguments/dnerf/monocular_spin_demo.py \
    --port "$TRAIN_PORT"

  if [[ "$SKIP_RENDER" -eq 0 ]]; then
    echo "[STEP 3/3] Render fixed-camera test sequence..."
    python render.py \
      --model_path "output/$EXP_NAME" \
      --skip_train \
      --skip_video \
      --configs arguments/dnerf/monocular_spin_demo.py
  fi

elif [[ "$MODE" == "moving_camera" ]]; then
  require_cmd ffmpeg
  require_cmd colmap

  echo "[STEP 1/4] Extract frames for COLMAP..."
  python scripts/prepare_orbit_video.py \
    --video "$VIDEO" \
    --out "$DATASET_DIR" \
    --fps "$FPS" \
    --overwrite

  echo "[STEP 2/4] Run COLMAP conversion pipeline..."
  python convert.py -s "$DATASET_DIR"

  echo "[STEP 3/4] Train 4DGaussians (COLMAP trajectory + time)..."
    -s "$DATASET_DIR" \
    -m "output/$EXP_NAME" \
    --no_eval \
    --colmap_video_interp "$COLMAP_VIDEO_INTERP" \
    --port "$TRAIN_PORT"

  if [[ "$SKIP_RENDER" -eq 0 ]]; then
    echo "[STEP 4/4] Render result video..."
    python render.py \
      -s "$DATASET_DIR" \
      -m "output/$EXP_NAME" \
      --skip_train \
      --skip_test
  fi

else
  echo "[ERROR] Invalid --mode: $MODE (expected fixed_camera or moving_camera)" >&2
  exit 1
fi

echo
echo "[DONE] Finished."
if [[ "$MODE" == "fixed_camera" ]]; then
  echo "[RESULT] output/$EXP_NAME/test/ours_*/video_rgb.mp4"
else
  echo "[RESULT] output/$EXP_NAME/video/ours_*/video_rgb.mp4"
fi
