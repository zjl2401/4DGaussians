#!/usr/bin/env python3
"""
通用“一键视频 -> 训练 -> 渲染”流水线。

给定输入视频，每次会生成一个与视频同名的 dataset/输出目录，并在该输出目录下渲染：
train/test/video（或由 --render-target 指定）。
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    print("[RUN]", " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd)
    if r.returncode != 0:
        sys.exit(r.returncode)

def _extract_config_iterations(config_py_path: str) -> Optional[int]:
    """
    从 arguments/dnerf/*.py 这种配置文件里，尝试读取 OptimizationParams['iterations']。
    这样可以避免 train.py 在“合并 config 之前”就处理 save_iterations 的坑。
    """
    try:
        import runpy

        data = runpy.run_path(config_py_path)
        opt = data.get("OptimizationParams", None)
        if isinstance(opt, dict) and "iterations" in opt:
            return int(opt["iterations"])
    except Exception:
        # 读取失败就交给用户显式指定 --save-iterations 或回退到 --iterations
        return None
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Video pipeline: prepare -> train -> render")
    p.add_argument("--video", "-v", required=True, help="输入视频路径，如 *.mp4")
    p.add_argument("--run-name", default=None, help="输出命名；默认取视频文件名(不含扩展名)")
    p.add_argument(
        "--mode",
        default="monocular_spin",
        choices=["monocular_spin"],
        help="数据准备/训练配置模式",
    )
    p.add_argument("--config", default="arguments/dnerf/monocular_spin_demo.py", help="训练/渲染用的配置文件")
    p.add_argument("--iterations", type=int, default=30000, help="训练迭代次数（也用于渲染 iteration）")
    p.add_argument("--save-iterations", type=int, default=None, help="保存高斯的迭代点；默认等于 --iterations")
    p.add_argument(
        "--render-iteration",
        type=int,
        default=-1,
        help="渲染使用的 iteration。默认 -1=自动取输出目录里最新的迭代（避免 config 覆盖导致找不到）。",
    )
    p.add_argument("--port", type=int, default=6017, help="训练 GUI 端口（影响网络 GUI 连接）")

    # dataset prepare (monocular_spin)
    p.add_argument("--max-frames", type=int, default=120, help="抽帧最大数量")
    p.add_argument("--stride", type=int, default=2, help="抽帧步长：每隔 N 帧取一帧")
    p.add_argument("--size", type=int, default=800, help="缩放：短边缩放到该值")
    p.add_argument("--train-ratio", type=float, default=0.9, help="train/test 划分比例")
    p.add_argument(
        "--video-radius",
        type=float,
        default=4.0,
        help="生成 video_cameras 的轨道半径（越小越靠近模型，可能更容易裁剪）。",
    )
    p.add_argument(
        "--video-phi",
        type=float,
        default=-30.0,
        help="生成 video_cameras 的轨道俯仰角(度)。默认 -30.0。",
    )
    p.add_argument(
        "--camera-angle-x",
        type=float,
        default=0.8575560450553894,
        help="prepare_monocular_spin_demo 的 camera_angle_x（影响视场 FOV，直接影响清晰度）。",
    )
    p.add_argument(
        "--auto-crop",
        action="store_true",
        help="自动前景裁剪（减少背景杂点对训练/渲染的干扰）。",
    )
    p.add_argument(
        "--crop-margin",
        type=float,
        default=0.15,
        help="自动裁剪的边缘留白比例（0~1）。",
    )
    p.add_argument(
        "--crop-threshold",
        type=int,
        default=25,
        help="自动裁剪差分阈值（越大越保守）。",
    )
    p.add_argument(
        "--crop-downscale-short-side",
        type=int,
        default=256,
        help="自动裁剪估计 bbox 的下采样短边大小。",
    )
    p.add_argument(
        "--static-bg-clean",
        action="store_true",
        help="静态背景清理：用帧中位数替换背景（适合相机不动+背景不动的转台场景）。",
    )
    p.add_argument(
        "--static-bg-threshold",
        type=int,
        default=25,
        help="静态背景清理阈值：差异灰度大于该值认为是前景。",
    )
    p.add_argument(
        "--static-bg-downscale-short-side",
        type=int,
        default=256,
        help="静态背景清理使用的下采样短边大小。",
    )
    p.add_argument(
        "--write-fg-masks",
        action="store_true",
        help="输出前景 mask（用于 mask-aware loss，转台/静态背景强烈推荐）。",
    )

    p.add_argument("--white-background", action="store_true", help="训练/渲染使用白色背景")
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Blender/单目：不划分 test，使用全部帧参与训练（通常更稳定）。",
    )
    p.add_argument(
        "--render-target",
        default="test",
        choices=["train", "test", "video", "all"],
        help="渲染输出类别",
    )
    p.add_argument(
        "--copy-mp4-to-desktop",
        action="store_true",
        help="把渲染得到的 video_rgb.mp4 复制到桌面（方便直接查看）",
    )
    p.add_argument(
        "--desktop-subdir",
        default=None,
        help="复制到桌面的子目录名；默认使用 run-name/渲染类别",
    )
    p.add_argument(
        "--desktop-path",
        default=None,
        help="显式指定 Windows 桌面路径（WSL 下建议用 /mnt/c/... 这种路径）",
    )
    p.add_argument("--dataset-root", default="data/monocular_spin", help="dataset 根目录")
    p.add_argument("--output-root", default="output/monocular_spin", help="模型输出根目录")
    p.add_argument("--overwrite", action="store_true", help="如 dataset/output 已存在则先删除重建")
    args = p.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    run_name = args.run_name or Path(video_path).stem
    dataset_dir = os.path.abspath(os.path.join(args.dataset_root, run_name))
    output_dir = os.path.abspath(os.path.join(args.output_root, run_name))

    if args.overwrite:
        if os.path.exists(dataset_dir):
            print("[INFO] Overwrite: removing dataset dir:", dataset_dir)
            shutil.rmtree(dataset_dir, ignore_errors=True)
        if os.path.exists(output_dir):
            print("[INFO] Overwrite: removing output dir:", output_dir)
            shutil.rmtree(output_dir, ignore_errors=True)

    config_path = os.path.join(repo_root, args.config)
    config_iters = _extract_config_iterations(config_path)
    # train.py 的 save/iterations 处理顺序会导致“config 覆盖后的 iterations 没被加入保存列表”
    # 所以默认情况下我们直接用 config 的 iterations 作为保存迭代。
    if args.save_iterations is not None:
        save_iters = args.save_iterations
    elif config_iters is not None:
        save_iters = config_iters
    else:
        save_iters = args.iterations

    # 1) prepare dataset
    if args.mode == "monocular_spin":
        prepare_script = os.path.join(repo_root, "scripts", "prepare_monocular_spin_demo.py")
        _run(
            [
                sys.executable,
                prepare_script,
                "--video",
                video_path,
                "--out",
                dataset_dir,
                "--max_frames",
                str(args.max_frames),
                "--stride",
                str(args.stride),
                "--size",
                str(args.size),
                "--train_ratio",
                str(args.train_ratio),
                "--video_camera_phi",
                str(args.video_phi),
                "--video_camera_radius",
                str(args.video_radius),
                "--camera_angle_x",
                str(args.camera_angle_x),
                *(["--auto-crop"] if args.auto_crop else []),
                "--crop-margin",
                str(args.crop_margin),
                "--crop-threshold",
                str(args.crop_threshold),
                "--crop-downscale-short-side",
                str(args.crop_downscale_short_side),
                *(["--static-bg-clean"] if args.static_bg_clean else []),
                "--static-bg-threshold",
                str(args.static_bg_threshold),
                "--static-bg-downscale-short-side",
                str(args.static_bg_downscale_short_side),
                *(["--write-fg-masks"] if args.write_fg_masks else []),
            ]
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    # 2) train
    train_script = os.path.join(repo_root, "train.py")
    train_cmd = [
        sys.executable,
        train_script,
        "-s",
        dataset_dir,
        "-m",
        output_dir,
        "--configs",
        config_path,
        "--iterations",
        str(args.iterations),
        "--save_iterations",
        str(save_iters),
        "--port",
        str(args.port),
    ]
    if args.white_background:
        train_cmd.append("--white_background")
    if args.no_eval:
        train_cmd.append("--no_eval")

    _run(train_cmd)

    # 3) render
    render_script = os.path.join(repo_root, "render.py")
    render_cmd = [
        sys.executable,
        render_script,
        "-s",
        dataset_dir,
        "-m",
        output_dir,
        "--iteration",
        str(args.render_iteration),
        "--configs",
        config_path,
    ]
    if args.no_eval:
        render_cmd.append("--no_eval")
    if args.render_target == "train":
        render_cmd += ["--skip_test", "--skip_video"]
    elif args.render_target == "test":
        render_cmd += ["--skip_train", "--skip_video"]
    elif args.render_target == "video":
        render_cmd += ["--skip_train", "--skip_test"]
    elif args.render_target == "all":
        pass
    else:
        raise ValueError(f"Unsupported --render-target: {args.render_target}")

    _run(render_cmd)

    # 4) output locations hint (+ optional copy)
    targets = ["train", "test", "video"] if args.render_target == "all" else [args.render_target]
    print("\n[DONE] Render done.")

    def _latest_mp4_for_target(t: str) -> Path | None:
        base = Path(output_dir) / t
        candidates = list(base.glob("ours_*/video_rgb.mp4"))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    desktop_dir: Path | None = None
    if args.copy_mp4_to_desktop:
        if args.desktop_path:
            desktop = Path(args.desktop_path)
        else:
            desktop = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop"
        if args.desktop_subdir:
            desktop_dir = desktop / args.desktop_subdir
        else:
            desktop_dir = desktop / run_name / ("all" if args.render_target == "all" else args.render_target)
        os.makedirs(desktop_dir, exist_ok=True)

    copied = 0
    for t in targets:
        mp4_path = _latest_mp4_for_target(t)
        if not mp4_path:
            print(f"[WARN] Missing mp4 for {t}: output dir={Path(output_dir)/t}")
            continue
        print(f"[RESULT] {t}: {mp4_path} (exists=True)")

        if desktop_dir:
            ours_dir = mp4_path.parent.name  # ours_<iter>
            dst = desktop_dir / f"{run_name}_{t}_{ours_dir}.mp4"
            shutil.copy2(str(mp4_path), str(dst))
            print(f"[COPIED] {dst}")
            copied += 1

    if desktop_dir and copied == 0:
        print("[WARN] No mp4 copied to desktop (check render-target/render-iteration).")


if __name__ == "__main__":
    main()

