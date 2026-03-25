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
  --zoom-scale F                Post-render center zoom factor, >1 enables zoom (default: 1.0)
  --highlight-foreground        Generate motion-highlight video to separate object/background
  --desktop-dir PATH            Export final videos to PATH (default: ~/Desktop)
  --no-export-desktop           Do not copy post-render videos to desktop
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
ZOOM_SCALE=1.0
HIGHLIGHT_FOREGROUND=0
DESKTOP_DIR="${HOME}/Desktop"
EXPORT_DESKTOP=1

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
    --zoom-scale) ZOOM_SCALE="$2"; shift 2 ;;
    --highlight-foreground) HIGHLIGHT_FOREGROUND=1; shift ;;
    --desktop-dir) DESKTOP_DIR="$2"; shift 2 ;;
    --no-export-desktop) EXPORT_DESKTOP=0; shift ;;
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

post_process_video() {
  local src_video="$1"
  local out_prefix="$2"
  local mode="$3"
  local current="$src_video"

  if [[ "$ZOOM_SCALE" != "1.0" ]]; then
    require_cmd ffmpeg
    local zoom_video="${out_prefix}_zoom.mp4"
    ffmpeg -y -i "$current" \
      -vf "crop=iw/${ZOOM_SCALE}:ih/${ZOOM_SCALE}:(iw-iw/${ZOOM_SCALE})/2:(ih-ih/${ZOOM_SCALE})/2,scale=iw:ih" \
      -an "$zoom_video" >/dev/null 2>&1
    echo "[POST] Zoom video: $zoom_video"
    current="$zoom_video"
  fi

  if [[ "$HIGHLIGHT_FOREGROUND" -eq 1 ]]; then
    require_cmd ffmpeg
    local motion_video="${out_prefix}_motion.mp4"
    ffmpeg -y -i "$src_video" \
      -vf "tblend=all_mode=difference,eq=contrast=2.0:brightness=0.03:saturation=1.8" \
      -an "$motion_video" >/dev/null 2>&1
    echo "[POST] Motion-highlight video: $motion_video"

    local compare_video="${out_prefix}_compare.mp4"
    ffmpeg -y -i "$current" -i "$motion_video" \
      -filter_complex "[0:v][1:v]hstack=inputs=2[v]" -map "[v]" -an "$compare_video" >/dev/null 2>&1
    echo "[POST] Side-by-side video: $compare_video"
    current="$compare_video"
    if [[ "$mode" != "fixed_camera" ]]; then
      echo "[WARN] Foreground highlight is motion-based; best quality is in fixed_camera scenes."
    fi
  fi

  if [[ "$EXPORT_DESKTOP" -eq 1 ]]; then
    mkdir -p "$DESKTOP_DIR"
    cp -f "$current" "$DESKTOP_DIR/"
    echo "[POST] Exported to desktop: $DESKTOP_DIR/$(basename "$current")"
  fi
}

if [[ "$MODE" == "fixed_camera" ]]; then
  echo "[STEP 1/3] Prepare fixed-camera monocular dataset..."
  python scripts/prepare_monocular_spin_demo.py \
    --video "$VIDEO" \
    --out "$DATASET_DIR" \
    --max_frames "$MAX_FRAMES" \
    --stride "$STRIDE" \
    --size "$SIZE"

  echo "[STEP 2/3] Train 4DGaussians (D-NeRF style)..."
  python train.py \
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
  python train.py \
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
  RESULT_VIDEO="$(ls -1t output/$EXP_NAME/test/ours_*/video_rgb.mp4 2>/dev/null | head -n 1 || true)"
  echo "[RESULT] output/$EXP_NAME/test/ours_*/video_rgb.mp4"
else
  RESULT_VIDEO="$(ls -1t output/$EXP_NAME/video/ours_*/video_rgb.mp4 2>/dev/null | head -n 1 || true)"
  echo "[RESULT] output/$EXP_NAME/video/ours_*/video_rgb.mp4"
fi

if [[ "$SKIP_RENDER" -eq 0 ]] && [[ -n "${RESULT_VIDEO:-}" ]] && [[ -f "$RESULT_VIDEO" ]]; then
  post_process_video "$RESULT_VIDEO" "${RESULT_VIDEO%.mp4}" "$MODE"
fi
