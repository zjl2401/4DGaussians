#!/usr/bin/env python3
"""根据 sparse/0/points3D.ply 估计质心 / AABB 中心，辅助转盘场景居中与 orbit look-at 调参。"""
import argparse
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scene.dataset_readers import compute_colmap_pcd_center, fetchPly  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "colmap_root",
        type=str,
        help="COLMAP 数据根目录（含 sparse/0/points3D.ply）",
    )
    args = p.parse_args()
    ply = os.path.join(args.colmap_root, "sparse", "0", "points3D.ply")
    if not os.path.isfile(ply):
        print(f"[ERR] 未找到: {ply}")
        sys.exit(1)
    pcd = fetchPly(ply)
    pts = np.asarray(pcd.points, dtype=np.float64)
    mean_c = compute_colmap_pcd_center(pts, "mean")
    aabb_c = compute_colmap_pcd_center(pts, "aabb")
    mid_c = compute_colmap_pcd_center(pts, "mid")
    print(f"点数: {pts.shape[0]}")
    print(f"mean 中心: {mean_c}")
    print(f"aabb 中心: {aabb_c}")
    print(f"mid 中心:  {mid_c}")
    print(f"|mean-aabb|: {float(np.linalg.norm(mean_c - aabb_c)):.6f}")
    print()
    print("仅改渲染环绕目标（不重训）：")
    print(
        f'  --colmap_video_orbit_pcd_center_mode aabb --colmap_video_orbit_lookat_blend 1.0'
    )
    print()
    print("训练+渲染统一坐标（物体拉回原点附近，需重新训练）：")
    print(
        f"  --colmap_recenter_from_pcd --colmap_video_orbit_pcd_center_mode mid"
    )


if __name__ == "__main__":
    main()
