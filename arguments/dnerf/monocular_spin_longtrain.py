OptimizationParams = dict(
    coarse_iterations=1200,
    # Long-train profile for monocular spin data.
    iterations=12000,
    deformation_lr_init=0.00016,
    deformation_lr_final=0.000008,
    grid_lr_init=0.0012,
    grid_lr_final=0.00008,
    batch_size=1,
    densify_from_iter=200,
    # Keep densification active longer for better geometry coverage.
    densify_until_iter=10000,
    densification_interval=100,
    pruning_interval=200,
    opacity_reset_interval=3000,
    percent_dense=0.01,
    render_process=False,
)

ModelHiddenParams = dict(
    multires=[1, 2],
    defor_depth=1,
    net_width=64,
    plane_tv_weight=0.0001,
    time_smoothness_weight=0.01,
    l1_time_planes=0.0001,
    bounds=1.6,
)

