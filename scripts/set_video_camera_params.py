#!/usr/bin/env python3
"""
快速修改 monocular_spin/nerf-synthetic 这类 transforms 的 video_cameras 轨道参数。

用途：
- 不重抽帧、不重训
- 直接改 transforms_train.json / transforms_test.json 里的
  video_camera_phi / video_camera_radius
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Set video camera params in transforms json files.")
    p.add_argument("--dataset-dir", "-d", required=True, help="dataset 根目录（包含 transforms_train.json）")
    p.add_argument("--phi", type=float, default=None, help="video_camera_phi（度），例如 -30.0")
    p.add_argument("--radius", type=float, default=None, help="video_camera_radius，越小越靠近模型")
    p.add_argument("--yes", action="store_true", help="不需要交互确认")
    args = p.parse_args()

    if args.phi is None and args.radius is None:
        raise SystemExit("请至少提供 --phi 或 --radius")

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_dir}")

    def update_one(pth: Path) -> None:
        if not pth.exists():
            print(f"[WARN] missing: {pth}")
            return
        with open(pth, "r", encoding="utf-8") as f:
            data = json.load(f)
        before = (data.get("video_camera_phi"), data.get("video_camera_radius"))
        if args.phi is not None:
            data["video_camera_phi"] = float(args.phi)
        if args.radius is not None:
            data["video_camera_radius"] = float(args.radius)
        after = (data.get("video_camera_phi"), data.get("video_camera_radius"))
        with open(pth, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"[UPDATED] {pth.name}: {before} -> {after}")

    train_json = dataset_dir / "transforms_train.json"
    test_json = dataset_dir / "transforms_test.json"

    update_one(train_json)
    update_one(test_json)

    print("[DONE] video_camera params updated.")


if __name__ == "__main__":
    main()

