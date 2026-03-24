# WSL Ubuntu + RTX 4060: 自己拍摄单目视频跑 4DGaussians

本指南给出两种模式：

1. **fixed_camera**：相机基本不动、物体在动（比如转台/手持转物体）。
2. **moving_camera**：相机在动（物体可静止或缓慢运动）。

> 注意：单目下“相机和物体都大幅运动”是高难场景，重建/时序一致性会明显下降。建议先从慢运动、纹理丰富、光照稳定的视频开始。

## 0) 环境检查（WSL）

```bash
nvidia-smi
python -V
which colmap || echo "colmap not found"
which ffmpeg || echo "ffmpeg not found"
```

- `moving_camera` 模式依赖 `ffmpeg` 和 `colmap`。
- `fixed_camera` 模式只需要 Python 依赖，不需要 COLMAP。

## 1) 一键脚本（Linux/WSL）

脚本：`scripts/run_monocular_video_demo.sh`

### A. 相机不动 + 物体动

```bash
bash scripts/run_monocular_video_demo.sh \
  --mode fixed_camera \
  --video "/mnt/c/Users/未命名/Desktop/5ea426d43844789334bbc72c667b1cc9.mp4" \
  --run-name my_fixed_case
```

输出：

- 训练输出：`output/monocular_custom/my_fixed_case`
- 渲染视频：`output/monocular_custom/my_fixed_case/test/ours_*/video_rgb.mp4`

### B. 相机动（物体可静止或缓慢动）

```bash
bash scripts/run_monocular_video_demo.sh \
  --mode moving_camera \
  --video "/mnt/c/Users/未命名/Desktop/5ea426d43844789334bbc72c667b1cc9.mp4" \
  --run-name my_moving_case \
  --fps 3
```

输出：

- 训练输出：`output/monocular_custom/my_moving_case`
- 渲染视频：`output/monocular_custom/my_moving_case/video/ours_*/video_rgb.mp4`

## 2) 参数建议

- `fixed_camera`：`--max-frames 120 --stride 2 --size 800`（默认值）
- `moving_camera`：`--fps 2~5`（默认 3），视频越长可把 `fps` 调低以减轻 COLMAP 负担。
- 若显存不够，可先降低输入分辨率（采集时或预处理阶段）。

## 3) 拍摄建议（很重要）

- 尽量避免运动模糊（提高快门/光照）。
- 画面要有足够纹理，避免大面积纯色背景。
- 相机轨迹平滑，邻近帧要有重叠。
- 若“相机和物体都在动”，尽量让物体运动幅度小、速度慢。

## 4) 常见问题

1. **COLMAP 失败/点云很差**：降低 `--fps`、选择更清晰片段、确保纹理丰富和足够视差。  
2. **WSL 找不到 Windows 文件**：Windows `C:\...` 在 WSL 中对应 `/mnt/c/...`。  
3. **渲染没视频**：先确认训练是否完成，再检查 `output/<exp>/.../video_rgb.mp4` 路径。
