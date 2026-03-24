# Monocular Spin Demo

This demo targets a monocular video where the **camera is static** and the **object rotates**.

## 1) Prepare dataset from video

```bash
python scripts/prepare_monocular_spin_demo.py ^
  --video "C:\Users\36255\Documents\xwechat_files\wxid_84ssun42nr6w22_0763\msg\video\2026-03\5ea426d43844789334bbc72c667b1cc9.mp4" ^
  --out "data\monocular_spin\wechat_spin_demo" ^
  --max_frames 120 ^
  --stride 2 ^
  --size 800
```

This script will create:

- `images/*.png`
- `transforms_train.json`
- `transforms_test.json`

All frames share the same camera pose (`transform_matrix` identity), only `time` changes.

## 2) Train

```bash
python train.py ^
  -s "data\monocular_spin\wechat_spin_demo" ^
  --expname "monocular_spin/wechat_spin_demo" ^
  --configs arguments/dnerf/monocular_spin_demo.py ^
  --port 6017
```

## 3) Render

Use test split rendering to keep camera static in the output sequence:

```bash
python render.py ^
  --model_path "output/monocular_spin/wechat_spin_demo" ^
  --skip_train ^
  --skip_video ^
  --configs arguments/dnerf/monocular_spin_demo.py
```

Output video frames and mp4 will be under:

- `output/monocular_spin/wechat_spin_demo/test/ours_<iter>/renders`
- `output/monocular_spin/wechat_spin_demo/test/ours_<iter>/video_rgb.mp4`

## 4) One-click run (Windows, generic)

Script path:

- `scripts/run_monocular_spin_demo.bat`

Supported modes:

1. double-click script (uses built-in default video path)
2. drag and drop any `.mp4` onto the `.bat`
3. pass args in terminal

### Command examples

Use video file name as run name:

```bash
scripts\run_monocular_spin_demo.bat "D:\videos\spin_obj.mp4"
```

Custom run name:

```bash
scripts\run_monocular_spin_demo.bat "D:\videos\spin_obj.mp4" my_spin_demo
```

Outputs:

- dataset: `data/monocular_spin/<run_name>`
- model: `output/monocular_spin/<run_name>`
- rendered mp4: `output/monocular_spin/<run_name>/test/ours_<iter>/video_rgb.mp4`

