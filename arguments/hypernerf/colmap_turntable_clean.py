# COLMAP 转盘 / 单物体：减轻黑碎片、漂浮物与变形崩解的推荐超参（在 default 基础上偏「更干净、更稳」）
# 训练示例：
#   python train.py -s data/colmap/my_video_05 -m output/my_run --no_eval --configs arguments/hypernerf/colmap_turntable_clean.py
# 前景掩码：在数据根目录建 masks/，与 images 同主文件名（见 scene/dataset_readers 中 Optional foreground mask 注释），训练 L1 会走 masked_l1_loss

ModelParams = dict(
    sh_degree=3,
    # 环绕 look-at 用质心与 AABB 中点的平均，常比单纯 mean 更贴转盘物体
    colmap_video_orbit_pcd_center_mode="mid",
    # 若物体仍偏、愿重训：改为 True，并 train/render 均带同一 configs
    # colmap_recenter_from_pcd=True,
)

ModelHiddenParams = dict(
    plane_tv_weight=0.0004,
    time_smoothness_weight=0.0025,
    l1_time_planes=0.00015,
)

OptimizationParams = dict(
    iterations=16_000,
    batch_size=2,
    coarse_iterations=3000,
    densify_until_iter=10_000,
    densification_interval=100,
    densification_interval_early=50,
    early_prune_until_iter=5000,
    early_prune_opacity_boost=1.6,
    pruning_interval=100,
    prune_extra_min_gaussians=0,
    prune_world_scale_extent_ratio=0.07,
    opacity_reset_interval=8000,
    lambda_lpips=0.05,
    lambda_rigid_deform=0.0003,
    lambda_dssim=0.2,
)
