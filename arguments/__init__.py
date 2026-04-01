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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = True
        self.data_device = "cuda"
        self.eval = True
        self.render_process=False
        self.add_points=False
        self.extension=".png"
        self.llffhold=8
        # COLMAP：导出 video_rgb.mp4 时在轨迹上插值这么多视角（0=不额外插值）。
        # follow：在训练相机序列上 slerp；orbit：在环绕关键帧之间再加密，运动更顺滑（如 320/480）。
        self.colmap_video_interp = 0
        # COLMAP 导出视频相机模式：
        # - follow: 跟随训练相机轨迹（默认）
        # - orbit: 以场景中心为目标做环绕轨迹
        self.colmap_video_mode = "follow"
        # orbit 轨迹：horizontal=在 COLMAP 世界系里水平环绕并 look-at 中心（推荐，画面不易横倒）；
        # spherical=与 generateCamerasFromTransforms 相同的 NeRF/Blender 球形路径
        self.colmap_video_orbit_style = "horizontal"
        # COLMAP orbit 模式下的导出帧数
        self.colmap_video_orbit_frames = 160
        # COLMAP orbit 模式下的俯仰角（度，负数表示略俯视）
        self.colmap_video_orbit_phi = -20.0
        # COLMAP orbit 模式下的半径缩放（相对训练相机平均半径）
        self.colmap_video_orbit_radius_scale = 1.0
        # 动态场景：orbit 时时间轴。sweep=沿轨迹扫完整段动画；fixed=冻结在某一时刻（更像产品展示）
        self.colmap_video_orbit_time_mode = "sweep"
        self.colmap_video_orbit_fixed_time = 0.5
        # 水平环绕方位角（度），例如 -90~90 只做半圈
        self.colmap_video_orbit_azimuth_min = -180.0
        self.colmap_video_orbit_azimuth_max = 180.0
        # 导出 mp4 帧率（仅影响 video_rgb.mp4）
        self.colmap_video_fps = 30
        # orbit 的 look-at：优先用稀疏点云中心（比相机均值更贴物体）
        self.colmap_video_orbit_use_pcd_center = True
        self.colmap_video_orbit_lookat_blend = 1.0
        # 沿「点云中心 -> 相机平均位置」方向微调目标点，缓解物体在画面里偏后/裁切（约 0.05~0.2）
        self.colmap_video_orbit_lookat_nudge_frac = 0.0
        # sweep/ramp 时时间范围；fixed 仍用 colmap_video_orbit_fixed_time
        self.colmap_video_orbit_time_start = 0.0
        self.colmap_video_orbit_time_end = 1.0
        # horizontal 环绕默认已做一次竖直修正；若仍倒，加 --colmap_video_orbit_flip_y 可再翻回
        self.colmap_video_orbit_flip_y = False
        # 绕视线轴旋转（度），常见 180 可修正上下颠倒
        self.colmap_video_orbit_roll_deg = 0.0
        # 关闭与首帧相机 Y 轴对齐（若越对齐越错可设 True）
        self.colmap_video_orbit_no_align_ref_up = False
        # look-at 点在 COLMAP 世界系下的平移（按场景尺度试 ±0.5~5）
        self.colmap_video_orbit_lookat_ox = 0.0
        self.colmap_video_orbit_lookat_oy = 0.0
        self.colmap_video_orbit_lookat_oz = 0.0
        # 在环绕水平面内平移目标点（相对轨道半径的比例，约 ±0.05~0.2 修正左右偏移）
        self.colmap_video_orbit_pan_v0_frac = 0.0
        self.colmap_video_orbit_pan_v1_frac = 0.0
        # 整体旋转轨道起点（度），可微调「正面」朝向
        self.colmap_video_orbit_azimuth_offset_deg = 0.0
        # 环绕方向与默认相反时开启（帧序列上反转方位角采样顺序）
        self.colmap_video_orbit_reverse = False
        # 关闭则不用首帧相机 Y 投影（少数场景反而错时可设 True）
        self.colmap_video_orbit_disable_ref_up_projection = False
        # 环绕 look-at 用的点云中心：mean=质心；aabb=包围盒中心；mid=二者平均（转盘物体常更居中）
        self.colmap_video_orbit_pcd_center_mode = "robust"
        # True：按上式从点云算 delta，把所有相机与稀疏点云平移 P'=P-delta，使中心落原点附近（需训练/渲染同开，旧 checkpoint 不适用）
        self.colmap_recenter_from_pcd = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")
class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4 # useless
        self.defor_depth = 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.opacity_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.bounds = 1.6 
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.01 # TV loss of temporal grid
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,
                             'resolution': [64, 64, 64, 25]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }
        self.multires = [1, 2, 4, 8] # multi resolution of voxel grid
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds=False # cancel the deformation of Gaussians' scaling
        self.no_dr=False # cancel the deformation of Gaussians' rotations
        self.no_do=True # cancel the deformation of Gaussians' opacity
        self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False # useless
        self.grid_pe=0 # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp=False # useless
        self.apply_rotation=False # useless

        
        super().__init__(parser, "ModelHiddenParams")
        
class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.dataloader=False
        self.zerostamp_init=False
        self.custom_sampler=None
        self.iterations = 30_000
        self.coarse_iterations = 3000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 20_000
        self.deformation_lr_init = 0.00016
        self.deformation_lr_final = 0.000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.0016
        self.grid_lr_final = 0.00016

        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0
        # >0 时启用 LPIPS（显存+算力会增加），需 pip 安装 lpips
        self.lambda_lpips = 0
        # 变形场在 t=0 附近的刚性先验（抑制旋转视角下「崩解」）
        self.lambda_rigid_deform = 0.0
        self.weight_constraint_init= 1
        self.weight_constraint_after = 0.2
        self.weight_decay_iteration = 5000
        self.opacity_reset_interval = 3000
        self.densification_interval = 100
        # >0 且 iteration < early_prune_until_iter 时用更短的加密/剪枝周期（仅影响 densify/grow 的 modulo）
        self.densification_interval_early = 0
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold_coarse = 0.0002
        self.densify_grad_threshold_fine_init = 0.0002
        self.densify_grad_threshold_after = 0.0002
        self.pruning_from_iter = 500
        self.pruning_interval = 100
        # iteration < 该值时，不透明度剪枝阈值乘以 early_prune_opacity_boost（更大=更激进剔除低不透明度高斯）
        self.early_prune_until_iter = 5000
        self.early_prune_opacity_boost = 1.0
        self.opacity_threshold_coarse = 0.005
        self.opacity_threshold_fine_init = 0.005
        self.opacity_threshold_fine_after = 0.005
        # 世界空间尺度超过 extent * 该比例时剪枝（抑制过大高斯）；可与 colmap_turntable_clean 预设一起改小到 0.06~0.08
        self.prune_world_scale_extent_ratio = 0.1
        # 原逻辑仅当点数>200000 才走额外 prune；0 表示不限制（小场景也去低不透明/过大尺度）
        self.prune_extra_min_gaussians = 0
        self.batch_size=1
        self.add_point=False
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
