import argparse
import json
import os
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare fixed-camera monocular video dataset for 4DGaussians."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input monocular video file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output dataset directory (will contain images/transforms_*.json).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=120,
        help="Maximum number of frames to keep.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every N-th frame from the video.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio in (0, 1).",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=800,
        help="Resize the shorter side to this value while preserving aspect ratio.",
    )
    parser.add_argument(
        "--camera_angle_x",
        type=float,
        default=0.8575560450553894,
        help="Horizontal field-of-view in radians for Blender-format transforms.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resize_with_short_side(frame, short_side):
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("Invalid frame size.")
    if min(h, w) == short_side:
        return frame
    if h < w:
        new_h = short_side
        new_w = int(round(w * (short_side / h)))
    else:
        new_w = short_side
        new_h = int(round(h * (short_side / w)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def extract_frames(video_path, image_dir, stride, max_frames, short_side):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved = 0
    idx = 0
    frame_paths = []
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame = resize_with_short_side(frame, short_side)
            out_name = f"frame_{saved:05d}.png"
            out_path = os.path.join(image_dir, out_name)
            if not cv2.imwrite(out_path, frame):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            frame_paths.append(f"images/{out_name}")
            saved += 1
        idx += 1
    cap.release()
    return frame_paths


def build_frames(file_paths, start_idx, end_idx):
    count = end_idx - start_idx
    if count <= 0:
        return []

    transform_identity = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]

    out = []
    for i, fp in enumerate(file_paths[start_idx:end_idx]):
        if count == 1:
            t = 0.0
        else:
            t = float(i / (count - 1))
        out.append(
            {
                "file_path": fp.replace("\\", "/").rsplit(".", 1)[0],
                "time": t,
                "transform_matrix": transform_identity,
            }
        )
    return out


def write_transforms(path, camera_angle_x, frames):
    payload = {"camera_angle_x": camera_angle_x, "frames": frames}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    video_path = os.path.abspath(args.video)
    out_dir = os.path.abspath(args.out)
    image_dir = os.path.join(out_dir, "images")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if args.max_frames < 2:
        raise ValueError("--max_frames must be >= 2")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not (0.1 <= args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in [0.1, 1.0)")

    ensure_dir(out_dir)
    ensure_dir(image_dir)

    frame_paths = extract_frames(
        video_path=video_path,
        image_dir=image_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        short_side=args.size,
    )
    if len(frame_paths) < 2:
        raise RuntimeError("Not enough frames extracted. Try lower --stride or higher --max_frames.")

    split_idx = max(1, min(len(frame_paths) - 1, int(len(frame_paths) * args.train_ratio)))
    train_frames = build_frames(frame_paths, 0, split_idx)
    test_frames = build_frames(frame_paths, split_idx, len(frame_paths))

    write_transforms(
        os.path.join(out_dir, "transforms_train.json"),
        args.camera_angle_x,
        train_frames,
    )
    write_transforms(
        os.path.join(out_dir, "transforms_test.json"),
        args.camera_angle_x,
        test_frames,
    )

    print("Prepared dataset:")
    print(f"  video: {video_path}")
    print(f"  out:   {out_dir}")
    print(f"  total frames: {len(frame_paths)}")
    print(f"  train/test:   {len(train_frames)}/{len(test_frames)}")
    print("Now you can run train.py with this output folder as -s.")


if __name__ == "__main__":
    main()
