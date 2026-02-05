# PowerShell configuration file for training script
# All paths are relative to the project root directory
# Edit the paths below before running train_1.3b.ps1

# Get the directory where this script is located (project root)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Dataset and output paths (relative to project root)
$env:DATA_PATH = "data\dataset"
$env:OUTPUT_PATH = "outputs\checkpoints"

# Model paths (relative to project root)
# Wan 1.3B model paths - adjust these to match your actual model locations
$env:DIT_PATH = "models\wan_1.3b\dit"
$env:VAE_PATH = "models\wan_1.3b\vae"
$env:TEXT_ENCODER_PATH = "models\wan_1.3b\text_encoder"
$env:IMAGE_ENCODER_PATH = "models\wan_1.3b\image_encoder"

# Optional paths (relative to project root)
$env:TXT_PATH = "data\invalid_data.txt"
$env:RESUME_CKPT_PATH = "outputs\checkpoints\last.ckpt"

Write-Host "Configuration loaded. Paths are relative to: $ScriptDir" -ForegroundColor Green
Write-Host "DATA_PATH: $env:DATA_PATH"
Write-Host "OUTPUT_PATH: $env:OUTPUT_PATH"
Write-Host "DIT_PATH: $env:DIT_PATH"
Write-Host "VAE_PATH: $env:VAE_PATH"
Write-Host "TEXT_ENCODER_PATH: $env:TEXT_ENCODER_PATH"
Write-Host "IMAGE_ENCODER_PATH: $env:IMAGE_ENCODER_PATH"
