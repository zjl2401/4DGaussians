import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


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
    parser.add_argument(
        "--video_camera_phi",
        type=float,
        default=-30.0,
        help="导出 video_cameras 的俯仰角(度)。值越大/越小会改变俯视角；负数表示向下。默认 -30.0。",
    )
    parser.add_argument(
        "--video_camera_radius",
        type=float,
        default=4.0,
        help="导出 video_cameras 的相机轨道半径。半径越小，相机越靠近模型(可能更容易裁剪)。默认 4.0。",
    )
    parser.add_argument(
        "--auto-crop",
        action="store_true",
        help="自动前景裁剪（基于相邻抽帧差分的运动区域），减少背景杂点对训练/渲染的干扰。",
    )
    parser.add_argument(
        "--static-bg-clean",
        action="store_true",
        help="静态背景清理：假设相机与背景不动，用所有抽帧的下采样中位数估计背景，然后把背景像素替换为白色（减少背景杂点）。",
    )
    parser.add_argument(
        "--static-bg-threshold",
        type=int,
        default=25,
        help="静态背景清理阈值：像素差分灰度大于该值认为属于前景。",
    )
    parser.add_argument(
        "--static-bg-downscale-short-side",
        type=int,
        default=256,
        help="静态背景估计/分割使用的下采样短边大小（越小越快，但分割可能不够准）。",
    )
    parser.add_argument(
        "--write-fg-masks",
        action="store_true",
        help="输出前景 mask 到 masks/（用于训练时 mask-aware loss，强烈推荐转台/静态背景场景）。",
    )
    parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.15,
        help="自动裁剪的边缘留白比例（0~1）。",
    )
    parser.add_argument(
        "--crop-threshold",
        type=int,
        default=25,
        help="自动裁剪的差分阈值（灰度差，越大越保守）。",
    )
    parser.add_argument(
        "--crop-downscale-short-side",
        type=int,
        default=256,
        help="用于估计裁剪框的下采样短边大小（越小越快，但可能不够准）。",
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


def _compute_motion_bbox(video_path, stride, max_frames, short_side, downscale_short_side, threshold):
    """
    在下采样帧上，基于相邻抽帧的差分做 motion mask，得到 bbox。
    bbox 会映射回短边 = short_side 的坐标系。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    prev_gray = None
    bbox_left, bbox_top, bbox_right, bbox_bottom = None, None, None, None
    saved = 0
    idx = 0
    motion_pixels = 0

    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % stride == 0:
            frame_down = resize_with_short_side(frame, downscale_short_side)
            gray = cv2.cvtColor(frame_down, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diff = cv2.GaussianBlur(diff, (5, 5), 0)
                _, th = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
                ys, xs = np.where(th > 0)
                if xs.size > 0:
                    motion_pixels += xs.size
                    l, r = int(xs.min()), int(xs.max())
                    t, b = int(ys.min()), int(ys.max())
                    bbox_left = l if bbox_left is None else min(bbox_left, l)
                    bbox_top = t if bbox_top is None else min(bbox_top, t)
                    bbox_right = r if bbox_right is None else max(bbox_right, r)
                    bbox_bottom = b if bbox_bottom is None else max(bbox_bottom, b)

            prev_gray = gray
            saved += 1

        idx += 1

    cap.release()

    # motion 太少则无法稳定估计 bbox
    if motion_pixels < 200 or bbox_left is None:
        return None

    scale = float(short_side) / float(downscale_short_side)
    l_s = int(bbox_left * scale)
    r_s = int(bbox_right * scale)
    t_s = int(bbox_top * scale)
    b_s = int(bbox_bottom * scale)
    return l_s, t_s, r_s, b_s


def extract_frames_with_auto_crop(
    video_path, image_dir, stride, max_frames, short_side, margin, threshold, downscale_short_side
):
    """
    两次扫描视频：
    - 第一次估计 motion bbox
    - 第二次真正抽帧并对每帧做裁剪
    """
    bbox = _compute_motion_bbox(
        video_path=video_path,
        stride=stride,
        max_frames=max_frames,
        short_side=short_side,
        downscale_short_side=downscale_short_side,
        threshold=threshold,
    )

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
            frame_resized = resize_with_short_side(frame, short_side)
            if bbox is not None:
                h, w = frame_resized.shape[:2]
                l, t, r, b = bbox

                bw = max(r - l, 1)
                bh = max(b - t, 1)
                m_x = int(bw * margin)
                m_y = int(bh * margin)

                l2 = max(0, l - m_x)
                t2 = max(0, t - m_y)
                r2 = min(w - 1, r + m_x)
                b2 = min(h - 1, b + m_y)

                if r2 - l2 > 10 and b2 - t2 > 10:
                    frame_resized = frame_resized[t2:b2, l2:r2]

            out_name = f"frame_{saved:05d}.png"
            out_path = os.path.join(image_dir, out_name)
            if not cv2.imwrite(out_path, frame_resized):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            frame_paths.append(f"images/{out_name}")
            saved += 1

        idx += 1

    cap.release()
    return frame_paths


def extract_frames_with_static_bg_clean(
    video_path,
    image_dir,
    stride,
    max_frames,
    short_side,
    bg_threshold,
    bg_downscale_short_side,
):
    """
    假设相机与背景静止：用所有抽帧的下采样中位数估计背景。
    将每帧中与背景差异过大的区域视为前景，并把背景像素替换成白色。
    """
    # Pass 1: estimate background median on downscaled frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames_ds = []
    saved = 0
    idx = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame_down = resize_with_short_side(frame, bg_downscale_short_side)
            frames_ds.append(frame_down)
            saved += 1
        idx += 1
    cap.release()

    if len(frames_ds) < 5:
        print("[WARN] static-bg-clean: 提取帧太少，跳过背景清理。")
        # fallback: use plain extraction
        return extract_frames(video_path, image_dir, stride, max_frames, short_side)

    bg = np.median(np.stack(frames_ds, axis=0).astype(np.float32), axis=0).astype(np.uint8)

    # Pass 2: whiten background for each selected frame
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
            frame_resized = resize_with_short_side(frame, short_side)

            frame_down = resize_with_short_side(frame, bg_downscale_short_side)
            diff = cv2.absdiff(frame_down, bg)
            # Use mean over channels as simple "difference intensity"
            diff_gray = diff.mean(axis=2)
            mask_foreground = diff_gray > float(bg_threshold)

            h_res, w_res = frame_resized.shape[:2]
            mask_up = cv2.resize(
                mask_foreground.astype(np.uint8),
                (w_res, h_res),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            # Replace background with white (matches default renderer white background)
            frame_resized[~mask_up] = 255

            out_name = f"frame_{saved:05d}.png"
            out_path = os.path.join(image_dir, out_name)
            if not cv2.imwrite(out_path, frame_resized):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            frame_paths.append(f"images/{out_name}")
            saved += 1
        idx += 1

    cap.release()
    return frame_paths


def extract_frames_with_static_bg_masks(
    video_path,
    image_dir,
    mask_dir,
    stride,
    max_frames,
    short_side,
    bg_threshold,
    bg_downscale_short_side,
):
    """
    生成前景 mask（背景静止假设），并保存到 mask_dir（0=背景,255=前景）。
    不改动原始图像内容（避免把主体洗掉）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames_ds = []
    saved = 0
    idx = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame_down = resize_with_short_side(frame, bg_downscale_short_side)
            frames_ds.append(frame_down)
            saved += 1
        idx += 1
    cap.release()

    if len(frames_ds) < 5:
        print("[WARN] write-fg-masks: 提取帧太少，跳过 mask 输出。")
        return extract_frames(video_path, image_dir, stride, max_frames, short_side)

    bg = np.median(np.stack(frames_ds, axis=0).astype(np.float32), axis=0).astype(np.uint8)

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
            frame_resized = resize_with_short_side(frame, short_side)
            frame_down = resize_with_short_side(frame, bg_downscale_short_side)
            diff = cv2.absdiff(frame_down, bg)
            diff_gray = diff.mean(axis=2)
            mask_foreground = diff_gray > float(bg_threshold)

            h_res, w_res = frame_resized.shape[:2]
            mask_up = cv2.resize(
                mask_foreground.astype(np.uint8) * 255,
                (w_res, h_res),
                interpolation=cv2.INTER_NEAREST,
            )

            out_name = f"frame_{saved:05d}.png"
            out_path = os.path.join(image_dir, out_name)
            if not cv2.imwrite(out_path, frame_resized):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            mask_path = os.path.join(mask_dir, out_name)
            if not cv2.imwrite(mask_path, mask_up):
                raise RuntimeError(f"Failed to write mask: {mask_path}")

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


def write_transforms(path, camera_angle_x, frames, video_camera_phi: float, video_camera_radius: float):
    payload = {
        "camera_angle_x": camera_angle_x,
        "video_camera_phi": video_camera_phi,
        "video_camera_radius": video_camera_radius,
        "frames": frames,
    }
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
    mask_dir = os.path.join(out_dir, "masks")
    if args.write_fg_masks:
        ensure_dir(mask_dir)

    if args.write_fg_masks:
        print(
            f"[INFO] Foreground masks enabled (static-bg): threshold={args.static_bg_threshold}, downscale_short_side={args.static_bg_downscale_short_side}"
        )
        frame_paths = extract_frames_with_static_bg_masks(
            video_path=video_path,
            image_dir=image_dir,
            mask_dir=mask_dir,
            stride=args.stride,
            max_frames=args.max_frames,
            short_side=args.size,
            bg_threshold=args.static_bg_threshold,
            bg_downscale_short_side=args.static_bg_downscale_short_side,
        )
    elif args.static_bg_clean:
        print(
            f"[INFO] Static background clean enabled: threshold={args.static_bg_threshold}, downscale_short_side={args.static_bg_downscale_short_side}"
        )
        frame_paths = extract_frames_with_static_bg_clean(
            video_path=video_path,
            image_dir=image_dir,
            stride=args.stride,
            max_frames=args.max_frames,
            short_side=args.size,
            bg_threshold=args.static_bg_threshold,
            bg_downscale_short_side=args.static_bg_downscale_short_side,
        )
    elif args.auto_crop:
        print(
            f"[INFO] Auto-crop enabled: margin={args.crop_margin}, threshold={args.crop_threshold}, downscale_short_side={args.crop_downscale_short_side}"
        )
        frame_paths = extract_frames_with_auto_crop(
            video_path=video_path,
            image_dir=image_dir,
            stride=args.stride,
            max_frames=args.max_frames,
            short_side=args.size,
            margin=args.crop_margin,
            threshold=args.crop_threshold,
            downscale_short_side=args.crop_downscale_short_side,
        )
    else:
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
        args.video_camera_phi,
        args.video_camera_radius,
    )
    write_transforms(
        os.path.join(out_dir, "transforms_test.json"),
        args.camera_angle_x,
        test_frames,
        args.video_camera_phi,
        args.video_camera_radius,
    )

    print("Prepared dataset:")
    print(f"  video: {video_path}")
    print(f"  out:   {out_dir}")
    print(f"  total frames: {len(frame_paths)}")
    print(f"  train/test:   {len(train_frames)}/{len(test_frames)}")
    print("Now you can run train.py with this output folder as -s.")


if __name__ == "__main__":
    main()
