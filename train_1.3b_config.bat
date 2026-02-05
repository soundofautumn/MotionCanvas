@echo off
REM Configuration file for training script
REM All paths are relative to the project root directory
REM Edit the paths below before running train_1.3b.bat

REM Get the directory where this script is located (project root)
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

REM Dataset and output paths (relative to project root)
set DATA_PATH=data\dataset
set OUTPUT_PATH=outputs\checkpoints

REM Model paths (relative to project root)
REM Wan 1.3B model paths - adjust these to match your actual model locations
set DIT_PATH=models\wan_1.3b\dit
set VAE_PATH=models\wan_1.3b\vae
set TEXT_ENCODER_PATH=models\wan_1.3b\text_encoder
set IMAGE_ENCODER_PATH=models\wan_1.3b\image_encoder

REM Optional paths (relative to project root)
set TXT_PATH=data\invalid_data.txt
set RESUME_CKPT_PATH=outputs\checkpoints\last.ckpt

REM Distributed training settings (for multi-node training)
REM set RANK=0
REM set MASTER_ADDR=localhost
REM set MASTER_PORT=29500

REM After setting the paths above, run:
REM train_1.3b.bat
