@echo off
setlocal enabledelayedexpansion

REM ========= Defaults =========
set "DEFAULT_VIDEO_PATH=C:\Users\36255\Documents\xwechat_files\wxid_84ssun42nr6w22_0763\msg\video\2026-03\5ea426d43844789334bbc72c667b1cc9.mp4"
set "DATASET_ROOT=data\monocular_spin"
set "EXP_ROOT=monocular_spin"
set "CONFIG_PATH=arguments/dnerf/monocular_spin_demo.py"
set "PORT=6017"

REM Quick demo defaults (change if needed)
set "MAX_FRAMES=120"
set "STRIDE=2"
set "IMG_SIZE=800"

REM ========= Input args =========
REM Usage:
REM   1) Drag and drop video onto this .bat
REM   2) run_monocular_spin_demo.bat "D:\xx\video.mp4"
REM   3) run_monocular_spin_demo.bat "D:\xx\video.mp4" demo_name
set "VIDEO_PATH=%~1"
set "RUN_NAME=%~2"

if "%VIDEO_PATH%"=="" set "VIDEO_PATH=%DEFAULT_VIDEO_PATH%"

if "%RUN_NAME%"=="" (
  for %%I in ("%VIDEO_PATH%") do set "RUN_NAME=%%~nI"
)

set "DATASET_DIR=%DATASET_ROOT%\%RUN_NAME%"
set "EXP_NAME=%EXP_ROOT%/%RUN_NAME%"

REM ========= Script =========
cd /d "%~dp0.."
echo [INFO] Workspace: %CD%
echo [INFO] Video: %VIDEO_PATH%
echo [INFO] Run name: %RUN_NAME%
echo [INFO] Dataset dir: %DATASET_DIR%
echo [INFO] Experiment: %EXP_NAME%

if not exist "%VIDEO_PATH%" (
  echo [ERROR] Video not found:
  echo         %VIDEO_PATH%
  echo.
  echo Usage examples:
  echo   scripts\run_monocular_spin_demo.bat "D:\videos\obj.mp4"
  echo   scripts\run_monocular_spin_demo.bat "D:\videos\obj.mp4" my_demo
  exit /b 1
)

echo [STEP 1/3] Prepare monocular spin dataset...
python scripts/prepare_monocular_spin_demo.py ^
  --video "%VIDEO_PATH%" ^
  --out "%DATASET_DIR%" ^
  --max_frames %MAX_FRAMES% ^
  --stride %STRIDE% ^
  --size %IMG_SIZE%
if errorlevel 1 (
  echo [ERROR] Dataset preparation failed.
  exit /b 1
)

echo [STEP 2/3] Train 4DGaussians...
python train.py ^
  -s "%DATASET_DIR%" ^
  --expname "%EXP_NAME%" ^
  --configs "%CONFIG_PATH%" ^
  --port %PORT%
if errorlevel 1 (
  echo [ERROR] Training failed.
  exit /b 1
)

echo [STEP 3/3] Render fixed-camera test sequence...
python render.py ^
  --model_path "output/%EXP_NAME%" ^
  --skip_train ^
  --skip_video ^
  --configs "%CONFIG_PATH%"
if errorlevel 1 (
  echo [ERROR] Rendering failed.
  exit /b 1
)

echo.
echo [DONE] Demo finished.
echo [RESULT] Check rendered video under:
echo         output/%EXP_NAME%/test/ours_*/video_rgb.mp4
echo [RESULT] Frame sequence folder:
echo         output/%EXP_NAME%/test/ours_*/renders
echo.
pause

