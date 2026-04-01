#!/usr/bin/env python3
"""
对 COLMAP orbit 路径逐帧渲染并打分，估计「主体在画面中心且较清晰」的时长占比，
并推荐一段方位角区间，便于生成「近距离 + 主体可见时间长」的旋转视频。

说明：不调用目标检测，仅用中心 ROI 的拉普拉斯方差、饱和度与过曝惩罚作启发式，
对转盘+白背景+彩色主体通常有效；结果需结合肉眼抽查。
"""
from __future__ import annotations

import argparse
import os
import sys

# WSL2：部分环境下 cuDNN 动态加载 libcuda.so 时默认找不到路径，需包含 /usr/lib/wsl/lib
_wsl_libcuda_dir = "/usr/lib/wsl/lib"
if sys.platform == "linux" and os.path.isdir(_wsl_libcuda_dir):
    _prev = os.environ.get("LD_LIBRARY_PATH", "")
    if _wsl_libcuda_dir not in _prev.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = (
            _wsl_libcuda_dir + (os.pathsep + _prev if _prev else "")
        )

import numpy as np
import torch
from argparse import ArgumentParser
import cv2
from mmengine.config import Config

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from arguments import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
from gaussian_renderer import GaussianModel, render
from scene import Scene
from utils.general_utils import safe_state
from utils.params_utils import merge_hparams


def _frame_score(rgb: torch.Tensor, center_margin: float = 0.15) -> float:
    """rgb: [3,H,W], 0~1。在 CPU 上用 OpenCV 算拉普拉斯方差，避免触发 cuDNN（WSL 下 libcudnn 找 libcuda 失败）。"""
    _, H, W = rgb.shape
    t = int(H * center_margin)
    l = int(W * center_margin)
    crop_t = rgb[:, t : H - t, l : W - l]
    crop = (
        crop_t.detach()
        .clamp(0, 1)
        .mul(255.0)
        .byte()
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(max(lap.var(), 1e-12))
    sat = float((crop.max(axis=2) - crop.min(axis=2)).mean() / 255.0)
    lum = float(crop.mean() / 255.0)
    blow = max(0.0, (lum - 0.88) / 0.12)
    return lap_var * (0.15 + sat) * (1.0 - min(1.0, blow))


def _normalize_minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.ones_like(x) * 0.5
    return (x - lo) / (hi - lo)


def _best_linear_arc_sum(scores: np.ndarray, k: int) -> tuple[int, float]:
    """在「不跨过 linspace 首尾」的前提下找连续 k 帧最高分弧段，便于一条 --azimuth_min/max 渲染。"""
    n = len(scores)
    if k >= n:
        return 0, float(scores.sum())
    c = np.concatenate([[0.0], np.cumsum(scores)])
    best_s, best_i = -1e30, 0
    for start in range(n - k + 1):
        tot = c[start + k] - c[start]
        if tot > best_s:
            best_s, best_i = tot, start
    return best_i, best_s


def main():
    parser = ArgumentParser(description="Orbit 视角主体可见性打分（启发式）")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--configs", type=str, default=None)
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--num_angles", type=int, default=72, help="环绕采样帧数（越多越细，越慢）")
    parser.add_argument(
        "--radius_scale",
        type=float,
        default=0.65,
        help="与 render 的 colmap_video_orbit_radius_scale 一致，建议用你想拉近的距离",
    )
    parser.add_argument("--azimuth_min", type=float, default=-180.0)
    parser.add_argument("--azimuth_max", type=float, default=180.0)
    parser.add_argument(
        "--arc_percent",
        type=float,
        default=50.0,
        help="推荐连续弧占整圈的比例（%%），在该长度内找总分最高的弧段",
    )
    parser.add_argument(
        "--good_threshold",
        type=float,
        default=0.5,
        help="归一化分数 >= 该值视为「主体可见较好」一帧，用于统计占比",
    )
    parser.add_argument("--save_csv", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    if args.configs:
        cfg = Config.fromfile(args.configs)
        args = merge_hparams(args, cfg)
    if getattr(args, "no_eval", False):
        args.eval = False
    safe_state(args.quiet)

    if not torch.cuda.is_available():
        print(
            "[ERR] 未检测到 CUDA（torch.cuda.is_available()=False）。\n"
            "本脚本需 GPU 渲染，与 render.py 相同；在 WSL 若报 libcuda.so / libcudnn 缺失，请：\n"
            "  1) Windows 安装支持 WSL2 的 NVIDIA 驱动，并在 WSL 内执行 nvidia-smi 能成功；或\n"
            "  2) 在 Windows 本机（非 WSL）的 CUDA 环境中运行同一命令。\n"
            "纯 CPU 无法跑通光栅化。"
        )
        sys.exit(1)

    args.colmap_video_mode = "orbit"
    args.colmap_video_orbit_frames = int(args.num_angles)
    args.colmap_video_orbit_radius_scale = float(args.radius_scale)
    args.colmap_video_interp = 0
    args.colmap_video_orbit_azimuth_min = float(args.azimuth_min)
    args.colmap_video_orbit_azimuth_max = float(args.azimuth_max)

    dataset = model.extract(args)
    hyper = hyperparam.extract(args)
    pipe = pipeline.extract(args)
    iteration = args.iteration

    bg = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if dataset.white_background
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    gaussians = GaussianModel(dataset.sh_degree, hyper)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    views = scene.getVideoCameras()
    n = len(views)
    if n == 0:
        print("[ERR] video 相机为空，请检查 colmap_video_mode=orbit 与数据路径。")
        sys.exit(1)

    angles = np.linspace(
        float(args.azimuth_min),
        float(args.azimuth_max),
        n,
        endpoint=False,
    )
    step = (float(args.azimuth_max) - float(args.azimuth_min)) / max(n, 1)

    scores = []
    with torch.no_grad():
        for i in tqdm_or_range(n, args.quiet):
            cam = views[i]
            out = render(cam, gaussians, pipe, bg, stage="fine", cam_type=scene.dataset_type)[
                "render"
            ]
            scores.append(_frame_score(out))

    scores = np.asarray(scores, dtype=np.float64)
    norm = _normalize_minmax(scores)
    good_ratio = float((norm >= float(args.good_threshold)).mean())

    pct = float(args.arc_percent)
    k = max(1, int(round(n * pct / 100.0)))
    k = min(k, n)
    best_start, best_sum = _best_linear_arc_sum(scores, k)
    az0 = float(angles[best_start])
    az1 = float(angles[best_start + k - 1] + step)

    print("\n========== Orbit 主体可见性（启发式）==========")
    print(f"采样帧数: {n}, radius_scale={args.radius_scale}, 方位角 [{args.azimuth_min},{args.azimuth_max})")
    print(f"归一化分数 >= {args.good_threshold} 的帧占比: {good_ratio*100:.1f}%  （越高表示整圈里「像样」的视角越多）")
    print(
        f"最佳连续弧（约 {pct:.0f}% 圆周，{k} 帧，不跨 -180/180 接缝）: "
        f"azimuth ∈ [{az0:.2f}, {az1:.2f}) （索引 {best_start}..{best_start + k - 1}）"
    )
    loaded_it = int(scene.loaded_iter)
    cfg_hint = f"--configs {args.configs} " if args.configs else ""
    print("\n建议渲染命令（保留近距离；按需保留 pcd_center_mode 等）：")
    print(
        f"  python render.py -s {dataset.source_path} -m {dataset.model_path} "
        f"--iteration {loaded_it} {cfg_hint}"
        f"--skip_train --skip_test "
        f"--colmap_video_mode orbit --colmap_video_orbit_radius_scale {args.radius_scale} "
        f"--colmap_video_orbit_azimuth_min {az0:.4f} --colmap_video_orbit_azimuth_max {az1:.4f} "
        f"--colmap_video_interp 480"
    )
    print("（按需补上 --configs / --colmap_video_orbit_pcd_center_mode 等；无 --configs 可删）\n")

    # get_combined_args 合并 cfg_args 时，默认 None 的 CLI 项不会写入 Namespace，需 getattr
    save_csv = getattr(args, "save_csv", None)
    if save_csv:
        import csv

        _dir = os.path.dirname(os.path.abspath(save_csv))
        if _dir:
            os.makedirs(_dir, exist_ok=True)
        with open(save_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "azimuth_deg", "raw_score", "norm_score"])
            for i in range(n):
                w.writerow([i, angles[i], scores[i], norm[i]])
        print(f"已写 CSV: {save_csv}")


def tqdm_or_range(n: int, quiet: bool):
    if quiet:
        return range(n)
    from tqdm import tqdm

    return tqdm(range(n), desc="Scoring orbit")


if __name__ == "__main__":
    main()
