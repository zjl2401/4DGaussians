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

### 先进入仓库目录（非常关键）

如果你在 `~` 目录直接执行 `bash scripts/run_monocular_video_demo.sh`，会报：
`No such file or directory`。  
因为这个相对路径只在仓库根目录下有效。

```bash
cd ~/4DGaussians
# 如果你把仓库放在别处，请改成你自己的实际路径
```

也可以不 `cd`，直接用脚本绝对路径：

```bash
bash ~/4DGaussians/scripts/run_monocular_video_demo.sh --help
```

### A. 相机不动 + 物体动

```bash
bash scripts/run_monocular_video_demo.sh \
  --mode fixed_camera \
  --video "/mnt/c/Users/未命名/Desktop/5ea426d43844789334bbc72c667b1cc9.mp4" \
  --run-name my_fixed_case \
  --zoom-scale 1.25 \
  --highlight-foreground
```

输出：

- 训练输出：`output/monocular_custom/my_fixed_case`
- 渲染视频：`output/monocular_custom/my_fixed_case/test/ours_*/video_rgb.mp4`
- 拉近+前景高亮对比视频（自动导出到桌面）：`~/Desktop/*_compare.mp4`

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
- 可视化增强：`--zoom-scale 1.15~1.35`（中心拉近）+ `--highlight-foreground`（运动区域高亮，便于区分背景和主体）。
- 默认会把后处理视频复制到 `~/Desktop`，可用 `--desktop-dir <路径>` 改目录，或 `--no-export-desktop` 关闭。

## 3) 拍摄建议（很重要）

- 尽量避免运动模糊（提高快门/光照）。
- 画面要有足够纹理，避免大面积纯色背景。
- 相机轨迹平滑，邻近帧要有重叠。
- 若“相机和物体都在动”，尽量让物体运动幅度小、速度慢。

## 4) 常见问题

1. **`bash: scripts/run_monocular_video_demo.sh: No such file or directory`**  
   你当前不在仓库根目录。先执行 `cd ~/4DGaussians`，再运行脚本；或使用绝对路径执行脚本。  

2. **COLMAP 失败/点云很差**：降低 `--fps`、选择更清晰片段、确保纹理丰富和足够视差。  
3. **WSL 找不到 Windows 文件**：Windows `C:\...` 在 WSL 中对应 `/mnt/c/...`。  
4. **渲染没视频**：先确认训练是否完成，再检查 `output/<exp>/.../video_rgb.mp4` 路径。
