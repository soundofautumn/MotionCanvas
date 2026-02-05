@echo off
REM Windows batch script for training Wan 1.3B MotionCanvas model
REM Usage: train_1.3b.bat
REM 
REM Before running, please set the following variables:
REM   - DATA_PATH: Path to training dataset
REM   - OUTPUT_PATH: Path to save checkpoints
REM   - DIT_PATH: Path to DiT model
REM   - VAE_PATH: Path to VAE model
REM   - TEXT_ENCODER_PATH: Path to text encoder
REM   - IMAGE_ENCODER_PATH: Path to image encoder
REM   - TXT_PATH: Path to invalid data list (optional)
REM   - RESUME_CKPT_PATH: Path to resume checkpoint (optional)

REM Get the directory where this script is located (project root)
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Load configuration if config file exists
if exist train_1.3b_config.bat (
    call train_1.3b_config.bat
)

REM Set environment variables
set NUM_NODES=8
REM Set RANK, MASTER_ADDR, MASTER_PORT if not already set
if "%RANK%"=="" set RANK=0
if "%MASTER_ADDR%"=="" set MASTER_ADDR=localhost
if "%MASTER_PORT%"=="" set MASTER_PORT=29500

REM Check if required paths are set
if "%DATA_PATH%"=="" (
    echo ERROR: DATA_PATH is not set!
    echo Please set DATA_PATH, OUTPUT_PATH, DIT_PATH, VAE_PATH, TEXT_ENCODER_PATH, IMAGE_ENCODER_PATH before running.
    echo.
    echo You can either:
    echo   1. Set environment variables manually
    echo   2. Edit train_1.3b_config.bat and run: call train_1.3b_config.bat
    pause
    exit /b 1
)

REM Convert relative paths to absolute paths
if not "%DATA_PATH:~0,1%"=="\" if not "%DATA_PATH:~1,1%"==":" (
    set DATA_PATH=%SCRIPT_DIR%%DATA_PATH%
)
if not "%OUTPUT_PATH:~0,1%"=="\" if not "%OUTPUT_PATH:~1,1%"==":" (
    set OUTPUT_PATH=%SCRIPT_DIR%%OUTPUT_PATH%
)
if not "%DIT_PATH:~0,1%"=="\" if not "%DIT_PATH:~1,1%"==":" (
    set DIT_PATH=%SCRIPT_DIR%%DIT_PATH%
)
if not "%VAE_PATH:~0,1%"=="\" if not "%VAE_PATH:~1,1%"==":" (
    set VAE_PATH=%SCRIPT_DIR%%VAE_PATH%
)
if not "%TEXT_ENCODER_PATH:~0,1%"=="\" if not "%TEXT_ENCODER_PATH:~1,1%"==":" (
    set TEXT_ENCODER_PATH=%SCRIPT_DIR%%TEXT_ENCODER_PATH%
)
if not "%IMAGE_ENCODER_PATH:~0,1%"=="\" if not "%IMAGE_ENCODER_PATH:~1,1%"==":" (
    set IMAGE_ENCODER_PATH=%SCRIPT_DIR%%IMAGE_ENCODER_PATH%
)
if not "%TXT_PATH%"=="" (
    if not "%TXT_PATH:~0,1%"=="\" if not "%TXT_PATH:~1,1%"==":" (
        set TXT_PATH=%SCRIPT_DIR%%TXT_PATH%
    )
)
if not "%RESUME_CKPT_PATH%"=="" (
    if not "%RESUME_CKPT_PATH:~0,1%"=="\" if not "%RESUME_CKPT_PATH:~1,1%"==":" (
        set RESUME_CKPT_PATH=%SCRIPT_DIR%%RESUME_CKPT_PATH%
    )
)

REM Run training script
echo Starting training...
echo NUM_NODES=%NUM_NODES%
echo RANK=%RANK%
echo MASTER_ADDR=%MASTER_ADDR%
echo MASTER_PORT=%MASTER_PORT%
echo.

torchrun --nproc_per_node=8 --nnodes=%NUM_NODES% --node_rank=%RANK% --master_addr=%MASTER_ADDR% --master_port=%MASTER_PORT% examples\wanvideo\train_wan_1.3b_motioncanvas.py ^
    --task train ^
    --train_architecture full ^
    --dataset_path %DATA_PATH% ^
    --output_path %OUTPUT_PATH% ^
    --dit_path %DIT_PATH% ^
    --vae_path %VAE_PATH% ^
    --text_encoder_path %TEXT_ENCODER_PATH% ^
    --image_encoder_path %IMAGE_ENCODER_PATH% ^
    --invalid_data_path %TXT_PATH% ^
    --moving_noun_path data\filtered_ram_tag_list.txt ^
    --resume_from %RESUME_CKPT_PATH% ^
    --max_epochs 10000 ^
    --learning_rate 5e-5 ^
    --accumulate_grad_batches 1 ^
    --use_gradient_checkpointing ^
    --batch_size 1 ^
    --dataloader_num_workers 8 ^
    --every_n_train_steps 1000 ^
    --num_frames 49 ^
    --training_strategy deepspeed_stage_1 ^
    --num_nodes %NUM_NODES% ^
    --target_fps 15

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Training failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Training completed successfully!
pause
