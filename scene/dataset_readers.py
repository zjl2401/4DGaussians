#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from scene.cameras import Camera

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from scene.hyper_loader import Load_hyper_data, format_hyper_data
import torchvision.transforms as transforms
import copy
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}

def _rotmat_to_quat_xyzw(R):
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-12)


def _quat_xyzw_to_rotmat(q):
    x, y, z, w = [float(v) for v in q]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _quat_slerp_xyzw(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = float(np.sum(q0 * q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_t0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / (sin_t0 + 1e-12)
    s1 = np.sin(theta) / (sin_t0 + 1e-12)
    out = s0 * q0 + s1 * q1
    return out / (np.linalg.norm(out) + 1e-12)


def interpolate_colmap_video_cameras(cam_infos, num_samples):
    """沿已排序的 COLMAP 相机轨迹球面线性插值位姿，用于更顺滑的导出视频。"""
    if num_samples <= 0 or len(cam_infos) < 2:
        return cam_infos
    n = len(cam_infos)
    out = []
    us = np.linspace(0, n - 1, num_samples)
    for uid, u in enumerate(us):
        i0 = int(np.floor(u))
        i1 = min(i0 + 1, n - 1)
        alpha = float(u - i0) if i1 > i0 else 0.0
        c0, c1 = cam_infos[i0], cam_infos[i1]
        q0 = _rotmat_to_quat_xyzw(c0.R)
        q1 = _rotmat_to_quat_xyzw(c1.R)
        q = _quat_slerp_xyzw(q0, q1, alpha)
        R = _quat_xyzw_to_rotmat(q)
        T = (1.0 - alpha) * np.asarray(c0.T, dtype=np.float64) + alpha * np.asarray(c1.T, dtype=np.float64)
        ir = int(np.round(np.clip(u, 0, n - 1)))
        cref = cam_infos[ir]
        t0 = float(getattr(c0, "time", 0.0))
        t1 = float(getattr(c1, "time", 0.0))
        tnorm = float((1.0 - alpha) * t0 + alpha * t1)
        tnorm = float(np.clip(tnorm, 0.0, 1.0))
        out.append(
            CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=cref.FovY,
                FovX=cref.FovX,
                image=cref.image,
                image_path=None,
                image_name=f"video_{uid:05d}",
                width=cref.width,
                height=cref.height,
                time=tnorm,
                mask=None,
            )
        )
    return out


def _c2w_to_colmap_rt(c2w):
    """与 generateCamerasFromTransforms 一致：c2w -> (R, T)。"""
    matrix = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
    R = -np.transpose(matrix[:3, :3])
    R[:, 0] = -R[:, 0]
    T = -matrix[:3, 3]
    return R, T


def _pose_spherical_c2w(theta_deg, phi_deg, radius):
    """与 generateCamerasFromTransforms 中 pose_spherical 一致（numpy），返回 4x4 c2w。"""
    t = np.deg2rad(float(theta_deg))
    p = np.deg2rad(float(phi_deg))
    trans_t = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, float(radius)], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    rot_phi = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(p), -np.sin(p), 0],
            [0, np.sin(p), np.cos(p), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    rot_theta = np.array(
        [
            [np.cos(t), 0, -np.sin(t), 0],
            [0, 1, 0, 0],
            [np.sin(t), 0, np.cos(t), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )
    b2c = np.array(
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    return b2c @ rot_theta @ rot_phi @ trans_t


def orbit_colmap_video_cameras(
    cam_infos,
    num_samples,
    phi_deg=-20.0,
    radius_scale=1.0,
    azimuth_min=-180.0,
    azimuth_max=180.0,
    time_mode="sweep",
    fixed_time=0.5,
    time_start=0.0,
    time_end=1.0,
    lookat_point=None,
    lookat_blend=1.0,
    lookat_nudge_frac=0.0,
    flip_y=False,
    roll_deg=0.0,
    no_align_ref_up=False,
    lookat_ox=0.0,
    lookat_oy=0.0,
    lookat_oz=0.0,
    pan_v0_frac=0.0,
    pan_v1_frac=0.0,
    azimuth_offset_deg=0.0,
    reverse=False,
    orbit_style="horizontal",
    disable_ref_up_projection=False,
):
    """环绕视频相机。

    orbit_style:
      - horizontal（默认）：在平均相机「上方向」与水平基 (v0,v1) 构成的锥面上采样，始终 look-at center，
        并可选将 up 与首帧相机 Y 对齐（no_align_ref_up / disable_ref_up_projection），适合 COLMAP 实拍。
      - spherical：pose_spherical（NeRF/Blender 约定）+ 平移 center；与合成数据一致，实拍易出现横倒/朝向怪。
    center 由训练相机均值 / lookat_point、nudge、pan、lookat_o* 得到；半径为相机到 center 的中位数 × radius_scale。
    """
    if len(cam_infos) == 0:
        return cam_infos
    if num_samples <= 0:
        num_samples = len(cam_infos)

    c2w_list = []
    cam_centers = []
    for cam in cam_infos:
        w2c = getWorld2View2(cam.R, cam.T)
        c2w = np.linalg.inv(w2c)
        c2w_list.append(c2w)
        cam_centers.append(c2w[:3, 3])
    cam_centers = np.asarray(cam_centers, dtype=np.float64)
    cam_mean = np.mean(cam_centers, axis=0)

    if lookat_point is not None:
        lp = np.asarray(lookat_point, dtype=np.float64).reshape(3)
        b = float(np.clip(lookat_blend, 0.0, 1.0))
        center = b * lp + (1.0 - b) * cam_mean
    else:
        center = cam_mean.copy()

    to_cams = cam_mean - center
    nt = np.linalg.norm(to_cams)
    if float(lookat_nudge_frac) != 0.0 and nt > 1e-6:
        dists0 = np.linalg.norm(cam_centers - center[None, :], axis=1)
        r0 = float(np.median(dists0)) if len(dists0) > 0 else 1.0
        center = center + float(lookat_nudge_frac) * r0 * (to_cams / nt)

    ups = np.stack([c[:3, 1] for c in c2w_list], axis=0)
    world_up = np.mean(ups, axis=0)
    nu = np.linalg.norm(world_up)
    if nu < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        world_up = world_up / nu

    aux = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    v0 = np.cross(world_up, aux)
    if np.linalg.norm(v0) < 1e-6:
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v0 = np.cross(world_up, aux)
    v0 = v0 / (np.linalg.norm(v0) + 1e-12)
    v1 = np.cross(world_up, v0)
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)

    dists_pre = np.linalg.norm(cam_centers - center[None, :], axis=1)
    r_pre = float(np.median(dists_pre)) if len(dists_pre) > 0 else 1.0
    center = (
        center
        + float(pan_v0_frac) * r_pre * v0
        + float(pan_v1_frac) * r_pre * v1
        + np.array(
            [float(lookat_ox), float(lookat_oy), float(lookat_oz)],
            dtype=np.float64,
        )
    )

    dists = np.linalg.norm(cam_centers - center[None, :], axis=1)
    base_radius = float(np.median(dists)) if len(dists) > 0 else 1.0
    radius = max(base_radius * float(radius_scale), 1e-3)

    # 让环绕的 0° 与首个训练相机方位对齐，避免“从空气开始转”。
    ref_vec = cam_centers[0] - center
    ref_vec = ref_vec - np.dot(ref_vec, world_up) * world_up
    nref = np.linalg.norm(ref_vec)
    if nref > 1e-6:
        v0 = ref_vec / nref
        v1 = np.cross(world_up, v0)
        nv1 = np.linalg.norm(v1)
        if nv1 > 1e-6:
            v1 = v1 / nv1
        else:
            v1 = np.cross(world_up, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            v1 = v1 / (np.linalg.norm(v1) + 1e-12)

    n = int(num_samples)
    theta_degs = np.linspace(
        float(azimuth_min), float(azimuth_max), n, endpoint=False
    ) + float(azimuth_offset_deg)
    if bool(reverse):
        theta_degs = np.flip(theta_degs)

    tm = str(time_mode).lower().strip()
    ft = float(np.clip(fixed_time, 0.0, 1.0))
    ts = float(np.clip(time_start, 0.0, 1.0))
    te = float(np.clip(time_end, 0.0, 1.0))
    if te < ts:
        ts, te = te, ts
    denom = max(n - 1, 1)
    rd = np.deg2rad(float(roll_deg))
    ref_cam = cam_infos[0]
    style = str(orbit_style).lower().strip()
    phi_rad = np.deg2rad(float(phi_deg))
    # horizontal：几何 up 与 COLMAP/渲染里图像 Y 常相反，默认等价于 flip_y；与 CLI flip_y 异或可再翻回
    is_spherical = style == "spherical"
    effective_flip_y = bool(flip_y) ^ (not is_spherical)

    out = []
    for uid, theta_deg in enumerate(theta_degs):
        if style == "spherical":
            c2w = _pose_spherical_c2w(float(theta_deg), float(phi_deg), radius)
            c2w = np.asarray(c2w, dtype=np.float64).copy()
            c2w[:3, 3] = c2w[:3, 3] + center

            if abs(rd) > 1e-8:
                cr, sr = np.cos(rd), np.sin(rd)
                r0, r1 = c2w[:3, 0].copy(), c2w[:3, 1].copy()
                c2w[:3, 0] = cr * r0 - sr * r1
                c2w[:3, 1] = sr * r0 + cr * r1
            if effective_flip_y:
                c2w[:3, 1] = -c2w[:3, 1]

            R, T = _c2w_to_colmap_rt(c2w)
        else:
            theta = np.deg2rad(float(theta_deg))
            offset = radius * (
                np.cos(phi_rad) * (np.cos(theta) * v0 + np.sin(theta) * v1)
                + np.sin(phi_rad) * world_up
            )
            eye = center + offset
            back = eye - center
            back = back / (np.linalg.norm(back) + 1e-12)
            right = np.cross(world_up, back)
            nr = np.linalg.norm(right)
            if nr < 1e-6:
                right = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), back)
                nr = np.linalg.norm(right)
            right = right / (nr + 1e-12)
            up = np.cross(back, right)
            up = up / (np.linalg.norm(up) + 1e-12)

            if not bool(no_align_ref_up):
                ref_up = c2w_list[0][:3, 1]
                if not bool(disable_ref_up_projection):
                    up_proj = ref_up - np.dot(ref_up, back) * back
                    nup = np.linalg.norm(up_proj)
                    if nup > 1e-6:
                        up_proj = up_proj / nup
                        if np.dot(up_proj, up) < 0.0:
                            up_proj = -up_proj
                        up = up_proj
                        right = np.cross(up, back)
                        right = right / (np.linalg.norm(right) + 1e-12)
                        up = np.cross(back, right)
                        up = up / (np.linalg.norm(up) + 1e-12)
                elif np.dot(up, ref_up) < 0.0:
                    up = -up
                    right = np.cross(up, back)
                    right = right / (np.linalg.norm(right) + 1e-12)
                    up = np.cross(back, right)
                    up = up / (np.linalg.norm(up) + 1e-12)
            if effective_flip_y:
                up = -up
                right = np.cross(up, back)
                right = right / (np.linalg.norm(right) + 1e-12)
                up = np.cross(back, right)
                up = up / (np.linalg.norm(up) + 1e-12)
            if abs(rd) > 1e-8:
                cr, sr = np.cos(rd), np.sin(rd)
                rn = cr * right - sr * up
                un = sr * right + cr * up
                right = rn / (np.linalg.norm(rn) + 1e-12)
                up = un / (np.linalg.norm(un) + 1e-12)

            c2w = np.eye(4, dtype=np.float64)
            c2w[:3, 0] = right
            c2w[:3, 1] = up
            c2w[:3, 2] = back
            c2w[:3, 3] = eye

            R, T = _c2w_to_colmap_rt(c2w)
        if tm == "fixed":
            tnorm = ft
        else:
            tnorm = float(ts + (te - ts) * (uid / denom))
            tnorm = float(np.clip(tnorm, 0.0, 1.0))

        out.append(
            CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovY=ref_cam.FovY,
                FovX=ref_cam.FovX,
                image=ref_cam.image,
                image_path=None,
                image_name=f"orbit_{uid:05d}",
                width=ref_cam.width,
                height=ref_cam.height,
                time=tnorm,
                mask=None,
            )
        )
    return out


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
        denom = max(len(cam_extrinsics) - 1, 1)
        # Optional foreground mask: dataset/masks/<stem>.png (same layout as Blender transforms)
        mask = None
        mask_dir = os.path.join(os.path.dirname(images_folder), "masks")
        if os.path.isdir(mask_dir):
            stem = Path(os.path.basename(extr.name)).stem
            img_suffix = Path(os.path.basename(extr.name)).suffix.lower() or ".png"
            for suf in (".png", ".jpg", ".jpeg", img_suffix):
                cand = os.path.join(mask_dir, stem + suf)
                if os.path.isfile(cand):
                    m = Image.open(cand).convert("L")
                    m = PILtoTorch(m, None).to(torch.float32)[0:1, :, :] / 255.0
                    if m.shape[-2:] != image.shape[-2:]:
                        m = torch.nn.functional.interpolate(
                            m.unsqueeze(0),
                            size=(image.shape[-2], image.shape[-1]),
                            mode="nearest",
                        ).squeeze(0)
                    mask = m
                    break
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time=float(idx / denom), mask=mask)  # 单目/视频：时间归一化到 [0,1]
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def compute_colmap_pcd_center(points_xyz, mode="mean"):
    """稀疏点云几何中心。

    mode:
      - mean: 质心
      - aabb: 轴对齐包围盒中心
      - mid: mean 与 aabb 的平均（转盘场景常更稳）
      - robust: 先做分位数去极值，再取均值（稀疏点云有离群点时更稳）
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.size == 0:
        return np.zeros(3, dtype=np.float64)
    m = str(mode).lower().strip()
    mean_c = np.mean(pts, axis=0)
    if m == "aabb":
        return (np.min(pts, axis=0) + np.max(pts, axis=0)) * 0.5
    if m in ("mid", "mean_aabb", "blend"):
        aabb_c = (np.min(pts, axis=0) + np.max(pts, axis=0)) * 0.5
        return 0.5 * (mean_c + aabb_c)
    if m in ("robust", "trimmed", "p10p90"):
        if pts.shape[0] < 32:
            return mean_c
        q10 = np.percentile(pts, 10.0, axis=0)
        q90 = np.percentile(pts, 90.0, axis=0)
        keep = np.logical_and(pts >= q10[None, :], pts <= q90[None, :]).all(axis=1)
        kept = pts[keep]
        if kept.shape[0] < max(16, int(0.05 * pts.shape[0])):
            return mean_c
        return np.mean(kept, axis=0)
    return mean_c


def shift_colmap_cameras_world_minus_delta(cam_infos, delta):
    """世界坐标平移 P' = P - delta。COLMAP 下 t' = R^T @ delta + t（与 readColmapCameras / getWorld2View2 一致）。"""
    delta = np.asarray(delta, dtype=np.float64).reshape(3)
    out = []
    for c in cam_infos:
        R = np.asarray(c.R, dtype=np.float64)
        T = np.asarray(c.T, dtype=np.float64).reshape(3)
        T_new = R.T @ delta + T
        out.append(c._replace(T=T_new))
    return out


def readColmapSceneInfo(
    path,
    images,
    eval,
    llffhold=8,
    video_interp_frames=0,
    video_mode="follow",
    video_orbit_frames=160,
    video_orbit_style="horizontal",
    video_orbit_phi=-20.0,
    video_orbit_radius_scale=1.0,
    video_orbit_time_mode="sweep",
    video_orbit_fixed_time=0.5,
    video_orbit_azimuth_min=-180.0,
    video_orbit_azimuth_max=180.0,
    video_orbit_use_pcd_center=True,
    video_orbit_lookat_blend=1.0,
    video_orbit_lookat_nudge_frac=0.0,
    video_orbit_time_start=0.0,
    video_orbit_time_end=1.0,
    video_orbit_flip_y=False,
    video_orbit_roll_deg=0.0,
    video_orbit_no_align_ref_up=False,
    video_orbit_lookat_ox=0.0,
    video_orbit_lookat_oy=0.0,
    video_orbit_lookat_oz=0.0,
    video_orbit_pan_v0_frac=0.0,
    video_orbit_pan_v1_frac=0.0,
    video_orbit_azimuth_offset_deg=0.0,
    video_orbit_reverse=False,
    video_orbit_disable_ref_up_projection=False,
    video_orbit_pcd_center_mode="robust",
    colmap_recenter_from_pcd=False,
):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # breakpoint()
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    center_mode = str(video_orbit_pcd_center_mode).lower().strip()
    if bool(colmap_recenter_from_pcd) and pcd is not None:
        pts0 = np.asarray(pcd.points, dtype=np.float64)
        delta = compute_colmap_pcd_center(pts0, center_mode)
        train_cam_infos = shift_colmap_cameras_world_minus_delta(train_cam_infos, delta)
        test_cam_infos = shift_colmap_cameras_world_minus_delta(test_cam_infos, delta)
        pts1 = pts0 - delta.reshape(1, 3)
        pcd = BasicPointCloud(
            points=pts1, colors=pcd.colors, normals=pcd.normals
        )
        print(
            f"[COLMAP] recentered scene from pcd (mode={center_mode}): "
            f"delta={np.array2string(delta, precision=4, separator=', ')}, |delta|={float(np.linalg.norm(delta)):.4f}"
        )

    nerf_normalization = getNerfppNorm(train_cam_infos)

    mode = str(video_mode).lower().strip()
    video_cam_infos = train_cam_infos
    pc_center_for_orbit = None
    if pcd is not None and bool(video_orbit_use_pcd_center):
        pc_center_for_orbit = compute_colmap_pcd_center(
            np.asarray(pcd.points), center_mode
        )

    if mode == "orbit":
        video_cam_infos = orbit_colmap_video_cameras(
            train_cam_infos,
            num_samples=int(video_orbit_frames),
            orbit_style=str(video_orbit_style),
            phi_deg=float(video_orbit_phi),
            radius_scale=float(video_orbit_radius_scale),
            azimuth_min=float(video_orbit_azimuth_min),
            azimuth_max=float(video_orbit_azimuth_max),
            time_mode=str(video_orbit_time_mode),
            fixed_time=float(video_orbit_fixed_time),
            time_start=float(video_orbit_time_start),
            time_end=float(video_orbit_time_end),
            lookat_point=pc_center_for_orbit,
            lookat_blend=float(video_orbit_lookat_blend),
            lookat_nudge_frac=float(video_orbit_lookat_nudge_frac),
            flip_y=bool(video_orbit_flip_y),
            roll_deg=float(video_orbit_roll_deg),
            no_align_ref_up=bool(video_orbit_no_align_ref_up),
            lookat_ox=float(video_orbit_lookat_ox),
            lookat_oy=float(video_orbit_lookat_oy),
            lookat_oz=float(video_orbit_lookat_oz),
            pan_v0_frac=float(video_orbit_pan_v0_frac),
            pan_v1_frac=float(video_orbit_pan_v1_frac),
            azimuth_offset_deg=float(video_orbit_azimuth_offset_deg),
            reverse=bool(video_orbit_reverse),
            disable_ref_up_projection=bool(video_orbit_disable_ref_up_projection),
        )
        print(
            f"[COLMAP video] mode=orbit, style={str(video_orbit_style)}, frames={len(video_cam_infos)}, "
            f"phi={float(video_orbit_phi):.1f}, radius_scale={float(video_orbit_radius_scale):.3f}, "
            f"time={str(video_orbit_time_mode)}"
            + (
                f" (fixed_t={float(video_orbit_fixed_time):.3f})"
                if str(video_orbit_time_mode).lower().strip() == "fixed"
                else f" (t={float(video_orbit_time_start):.3f}->{float(video_orbit_time_end):.3f})"
            )
            + f", azimuth=[{float(video_orbit_azimuth_min):.1f},{float(video_orbit_azimuth_max):.1f}]"
            + (
                f", lookat=pcd({center_mode})"
                if pc_center_for_orbit is not None
                else ", lookat=cams"
            )
            + (", reverse_azimuth" if bool(video_orbit_reverse) else "")
        )
        if int(video_interp_frames) > 0 and len(video_cam_infos) >= 2:
            n0 = len(video_cam_infos)
            video_cam_infos = interpolate_colmap_video_cameras(
                video_cam_infos, int(video_interp_frames)
            )
            print(
                f"[COLMAP video] orbit+slerp upsample: {n0} -> {len(video_cam_infos)} frames "
                f"(colmap_video_interp={int(video_interp_frames)})"
            )
    elif video_interp_frames > 0:
        video_cam_infos = interpolate_colmap_video_cameras(train_cam_infos, video_interp_frames)
        print(f"[COLMAP video] mode=follow+interp, frames={len(video_cam_infos)}")
    else:
        print(f"[COLMAP video] mode=follow, frames={len(video_cam_infos)}")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    video_phi = float(template_json.get("video_camera_phi", -30.0))
    video_radius = float(template_json.get("video_camera_radius", 4.0))

    # generate render poses and times
    render_poses = torch.stack(
        [pose_spherical(angle, video_phi, video_radius) for angle in np.linspace(-180, 180, 160 + 1)[:-1]],
        0,
    )
    render_times = torch.linspace(0, maxtime, render_poses.shape[0])
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        # Keep the extracted resolution (auto-crop may produce non-800 sizes).
        # Forced resize to 800x800 can blur fine details and harm training.
        image = PILtoTorch(image, None)
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos
def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = focal2fov(contents['fl_x'],contents['w'])
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = mapper[frame["time"]]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
            # Keep extracted resolution; avoid fixed 800x800 blur.
            image = PILtoTorch(image, None)
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx

            mask = None
            mask_path = os.path.join(path, "masks", Path(frame["file_path"]).name + extension)
            if os.path.exists(mask_path):
                m = Image.open(mask_path).convert("L")
                m = PILtoTorch(m, None).to(torch.float32)[0:1, :, :] / 255.0
                # ensure same H/W as image tensor
                if m.shape[-2:] != image.shape[-2:]:
                    m = torch.nn.functional.interpolate(
                        m.unsqueeze(0),
                        size=(image.shape[-2], image.shape[-1]),
                        mode="nearest",
                    ).squeeze(0)
                mask = m

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            time = time, mask=mask))
            
    return cam_infos
def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)  
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in test_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float
def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper)
    print("Generating Video Transforms")
    video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "fused.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)
        # xyz = -np.array(pcd.points)
        # pcd = pcd._replace(points=xyz)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info
def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras


def readHyperDataInfos(datadir,use_bg_points,eval):
    train_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split ="train")
    test_cam_infos = Load_hyper_data(datadir,0.5,use_bg_points,split="test")
    print("load finished")
    train_cam = format_hyper_data(train_cam_infos,"train")
    print("format finished")
    max_time = train_cam_infos.max_time
    video_cam_infos = copy.deepcopy(test_cam_infos)
    video_cam_infos.split="video"


    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    pcd = fetchPly(ply_path)
    xyz = np.array(pcd.points)

    pcd = pcd._replace(points=xyz)
    nerf_normalization = getNerfppNorm(train_cam)
    plot_camera_orientations(train_cam_infos, pcd.points)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )

    return scene_info
def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def add_points(pointsclouds, xyz_min, xyz_max):
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
    # breakpoint()
    # new_
def readdynerfInfo(datadir,use_bg_points,eval):
    # loading all the data follow hexplane format
    # ply_path = os.path.join(datadir, "points3D_dense.ply")
    ply_path = os.path.join(datadir, "points3D_downsample2.ply")
    from scene.neural_3D_dataset_NDC import Neural3D_NDC_Dataset
    train_dataset = Neural3D_NDC_Dataset(
    datadir,
    "train",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )    
    test_dataset = Neural3D_NDC_Dataset(
    datadir,
    "test",
    1.0,
    time_scale=1,
    scene_bbox_min=[-2.5, -2.0, -1.0],
    scene_bbox_max=[2.5, 2.0, 1.0],
    eval_index=0,
        )
    train_cam_infos = format_infos(train_dataset,"train")
    val_cam_infos = format_render_poses(test_dataset.val_poses,test_dataset)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # xyz = np.load
    pcd = fetchPly(ply_path)
    print("origin points,",pcd.points.shape[0])
    
    print("after points,",pcd.points.shape[0])

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_dataset,
                           test_cameras=test_dataset,
                           video_cameras=val_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=300
                           )
    return scene_info

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam
def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")
    # breakpoint()
def readPanopticmeta(datadir, json_path):
    with open(os.path.join(datadir,json_path)) as f:
        test_meta = json.load(f)
    w = test_meta['w']
    h = test_meta['h']
    max_time = len(test_meta['fn'])
    cam_infos = []
    for index in range(len(test_meta['fn'])):
        focals = test_meta['k'][index]
        w2cs = test_meta['w2c'][index]
        fns = test_meta['fn'][index]
        cam_ids = test_meta['cam_id'][index]

        time = index / len(test_meta['fn'])
        for focal, w2c, fn, cam in zip(focals, w2cs, fns, cam_ids):
            image_path = os.path.join(datadir,"ims")
            image_name=fn
            image = Image.open(os.path.join(datadir,"ims",fn))
            im_data = np.array(image.convert("RGBA"))
            im_data = PILtoTorch(im_data,None)[:3,:,:]
            camera = setup_camera(w, h, focal, w2c)
            cam_infos.append({
                "camera":camera,
                "time":time,
                "image":im_data})
            
    cam_centers = np.linalg.inv(test_meta['w2c'][0])[:, :3, 3]  # Get scene radius
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))
    return cam_infos, max_time, scene_radius 

def readPanopticSportsinfos(datadir):
    train_cam_infos, max_time, scene_radius = readPanopticmeta(datadir, "train_meta.json")
    test_cam_infos,_, _ = readPanopticmeta(datadir, "test_meta.json")
    nerf_normalization = {
        "radius":scene_radius,
        "translate":torch.tensor([0,0,0])
    }

    ply_path = os.path.join(datadir, "pointd3D.ply")

        # Since this data set has no colmap data, we start with random points
    plz_path = os.path.join(datadir, "init_pt_cld.npz")
    data = np.load(plz_path)["data"]
    xyz = data[:,:3]
    rgb = data[:,3:6]
    num_pts = xyz.shape[0]
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.ones((num_pts, 3)))
    storePly(ply_path, xyz, rgb)
    # pcd = fetchPly(ply_path)
    # breakpoint()
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time,
                           )
    return scene_info

def readMultipleViewinfos(datadir,llffhold=8):

    cameras_extrinsic_file = os.path.join(datadir, "sparse_/images.bin")
    cameras_intrinsic_file = os.path.join(datadir, "sparse_/cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    from scene.multipleview_dataset import multipleview_dataset
    train_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="train")
    test_cam_infos = multipleview_dataset(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, cam_folder=datadir,split="test")

    train_cam_infos_ = format_infos(train_cam_infos,"train")
    nerf_normalization = getNerfppNorm(train_cam_infos_)

    ply_path = os.path.join(datadir, "points3D_multipleview.ply")
    bin_path = os.path.join(datadir, "points3D_multipleview.bin")
    txt_path = os.path.join(datadir, "points3D_multipleview.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=test_cam_infos.video_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "dynerf" : readdynerfInfo,
    "nerfies": readHyperDataInfos,  # NeRFies & HyperNeRF dataset proposed by [https://github.com/google/hypernerf/releases/tag/v0.1]
    "PanopticSports" : readPanopticSportsinfos,
    "MultipleView": readMultipleViewinfos
}
