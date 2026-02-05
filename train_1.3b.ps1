# PowerShell script for training Wan 1.3B MotionCanvas model
# Usage: .\train_1.3b.ps1
# 
# Before running, please set the following variables:
#   - $env:DATA_PATH: Path to training dataset
#   - $env:OUTPUT_PATH: Path to save checkpoints
#   - $env:DIT_PATH: Path to DiT model
#   - $env:VAE_PATH: Path to VAE model
#   - $env:TEXT_ENCODER_PATH: Path to text encoder
#   - $env:IMAGE_ENCODER_PATH: Path to image encoder
#   - $env:TXT_PATH: Path to invalid data list (optional)
#   - $env:RESUME_CKPT_PATH: Path to resume checkpoint (optional)

# Get the directory where this script is located (project root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & .venv\Scripts\Activate.ps1
}

# Load configuration if config file exists
if (Test-Path "train_1.3b_config.ps1") {
    & .\train_1.3b_config.ps1
}

# Set environment variables
$env:NUM_NODES = 8
# Set RANK, MASTER_ADDR, MASTER_PORT if not already set
if (-not $env:RANK) { $env:RANK = 0 }
if (-not $env:MASTER_ADDR) { $env:MASTER_ADDR = "localhost" }
if (-not $env:MASTER_PORT) { $env:MASTER_PORT = "29500" }

# Check if required paths are set
if (-not $env:DATA_PATH) {
    Write-Host "ERROR: DATA_PATH is not set!" -ForegroundColor Red
    Write-Host "Please set DATA_PATH, OUTPUT_PATH, DIT_PATH, VAE_PATH, TEXT_ENCODER_PATH, IMAGE_ENCODER_PATH before running."
    Write-Host ""
    Write-Host "You can either:"
    Write-Host "  1. Set environment variables manually"
    Write-Host "  2. Edit train_1.3b_config.ps1 and run: .\train_1.3b_config.ps1"
    exit 1
}

# Convert relative paths to absolute paths
function ConvertToAbsolutePath {
    param($Path)
    if (-not $Path) { return $Path }
    if (-not [System.IO.Path]::IsPathRooted($Path)) {
        return Join-Path $ScriptDir $Path
    }
    return $Path
}

$env:DATA_PATH = ConvertToAbsolutePath $env:DATA_PATH
$env:OUTPUT_PATH = ConvertToAbsolutePath $env:OUTPUT_PATH
$env:DIT_PATH = ConvertToAbsolutePath $env:DIT_PATH
$env:VAE_PATH = ConvertToAbsolutePath $env:VAE_PATH
$env:TEXT_ENCODER_PATH = ConvertToAbsolutePath $env:TEXT_ENCODER_PATH
$env:IMAGE_ENCODER_PATH = ConvertToAbsolutePath $env:IMAGE_ENCODER_PATH
if ($env:TXT_PATH) {
    $env:TXT_PATH = ConvertToAbsolutePath $env:TXT_PATH
}
if ($env:RESUME_CKPT_PATH) {
    $env:RESUME_CKPT_PATH = ConvertToAbsolutePath $env:RESUME_CKPT_PATH
}

# Run training script
Write-Host "Starting training..." -ForegroundColor Green
Write-Host "NUM_NODES=$env:NUM_NODES"
Write-Host "RANK=$env:RANK"
Write-Host "MASTER_ADDR=$env:MASTER_ADDR"
Write-Host "MASTER_PORT=$env:MASTER_PORT"
Write-Host ""

torchrun --nproc_per_node=8 --nnodes=$env:NUM_NODES --node_rank=$env:RANK --master_addr=$env:MASTER_ADDR --master_port=$env:MASTER_PORT examples\wanvideo\train_wan_1.3b_motioncanvas.py `
    --task train `
    --train_architecture full `
    --dataset_path $env:DATA_PATH `
    --output_path $env:OUTPUT_PATH `
    --dit_path $env:DIT_PATH `
    --vae_path $env:VAE_PATH `
    --text_encoder_path $env:TEXT_ENCODER_PATH `
    --image_encoder_path $env:IMAGE_ENCODER_PATH `
    --invalid_data_path $env:TXT_PATH `
    --moving_noun_path data\filtered_ram_tag_list.txt `
    --resume_from $env:RESUME_CKPT_PATH `
    --max_epochs 10000 `
    --learning_rate 5e-5 `
    --accumulate_grad_batches 1 `
    --use_gradient_checkpointing `
    --batch_size 1 `
    --dataloader_num_workers 8 `
    --every_n_train_steps 1000 `
    --num_frames 49 `
    --training_strategy deepspeed_stage_1 `
    --num_nodes $env:NUM_NODES `
    --target_fps 15

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Training failed with error code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Training completed successfully!" -ForegroundColor Green
