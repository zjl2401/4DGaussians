#!/usr/bin/env python3
"""
从单条环绕视频抽取帧到 COLMAP 所需的目录结构，便于 4DGaussians 训练与渲染。

典型场景：相机约 180° 绕静止或慢动物体运动的一条视频。
依赖：系统 PATH 中有 ffmpeg（https://ffmpeg.org/）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional


def _which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def main():
    p = argparse.ArgumentParser(description="视频抽帧 → data/<场景>/input，供 convert.py + COLMAP 使用")
    p.add_argument("--video", "-v", required=True, type=str, help="输入视频路径（如 orbit.mp4）")
    p.add_argument(
        "--out",
        "-o",
        required=True,
        type=str,
        help="场景根目录（将创建子目录 input/，其内为 %06d.jpg）",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="抽帧帧率；不指定则导出全部帧（视频很长时建议设 2~5 降低 COLMAP 负担）",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="最多保留前 N 帧，0 表示不限制",
    )
    p.add_argument("--overwrite", action="store_true", help="若 input/ 已存在则先清空")
    args = p.parse_args()

    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        print("错误：未找到 ffmpeg，请先安装并加入 PATH。", file=sys.stderr)
        sys.exit(1)

    video = os.path.abspath(args.video)
    root = os.path.abspath(args.out)
    inp = os.path.join(root, "input")

    if not os.path.isfile(video):
        print(f"错误：找不到视频文件: {video}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(inp):
        if args.overwrite:
            shutil.rmtree(inp)
        else:
            print(f"错误：{inp} 已存在。请加 --overwrite 或换 --out。", file=sys.stderr)
            sys.exit(1)

    os.makedirs(inp, exist_ok=True)
    out_pattern = os.path.join(inp, "%06d.jpg")

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", video]
    if args.fps is not None and args.fps > 0:
        cmd += ["-vf", f"fps={args.fps}"]
    cmd += ["-q:v", "2", out_pattern]

    if args.max_frames and args.max_frames > 0:
        cmd.insert(-2, "-frames:v")
        cmd.insert(-2, str(args.max_frames))

    print("运行:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("ffmpeg 失败，退出码:", r.returncode, file=sys.stderr)
        sys.exit(r.returncode)

    n = len([f for f in os.listdir(inp) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if n < 10:
        print(f"警告：仅 {n} 张图，COLMAP/训练可能不稳定，建议增加 --fps 或换更长视频。", file=sys.stderr)

    readme = os.path.join(root, "CUSTOM_VIDEO_PIPELINE.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "4DGaussians 自定义单目环绕视频 — 后续步骤\n\n"
            "1) COLMAP（在项目根目录执行，需已安装 COLMAP）：\n"
            f"   python convert.py -s \"{root}\"\n\n"
            "2) 训练（建议全帧训练 + 平滑导出轨迹）：\n"
            f"   python train.py -s \"{root}\" -m output/<你的实验名> --no_eval --colmap_video_interp 160\n\n"
            "3) 渲染最终视频：\n"
            f"   python render.py -s \"{root}\" -m output/<你的实验名> --skip_train --skip_test\n\n"
            "说明：物体尽量慢动、光照稳定；环绕约 180° 且重叠足够时 COLMAP 更容易成功。\n"
        )
    print(f"完成：共写入 {n} 帧到 {inp}")
    print(f"已生成说明文件: {readme}")


if __name__ == "__main__":
    main()
