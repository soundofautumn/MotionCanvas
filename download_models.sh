#!/bin/bash
# MotionCanvas 模型下载脚本
# 需要下载五类模型：
#   1. Wan2.1-Fun-1.3B-InP 基础模型 (~19GB)
#   2. Wan2.1 1.3B Motion Controller (~几百 MB)
#   3. Wan2.1 1.3B VACE (~数 GB)
#   4. MotionCanvas 预训练权重   (~3.1GB)
#   5. SAM (Segment Anything) checkpoint (~2.6GB)
#
# 使用方法:
#   bash download_models.sh              # 默认从 ModelScope 下载
#   bash download_models.sh --source hf  # 从 HuggingFace 下载
#
# 依赖: pip install modelscope   (ModelScope 下载)
#        pip install huggingface_hub  (HuggingFace 下载)

set -e

SOURCE="modelscope"
MODEL_DIR="/root/autodl-tmp/models"
COTRACKER_HUB_DIR="/root/autodl-tmp/torch_hub"

while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

mkdir -p "${MODEL_DIR}/wan_1.3b"
mkdir -p "${MODEL_DIR}/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1"
mkdir -p "${MODEL_DIR}/iic/VACE-Wan2.1-1.3B-Preview"
mkdir -p "${MODEL_DIR}/motioncanvas"
mkdir -p "${MODEL_DIR}/segment_anything"

echo "============================================"
echo " MotionCanvas 模型下载"
echo " 下载源: ${SOURCE}"
echo " 保存目录: ${MODEL_DIR}"
echo " CoTracker 缓存目录: ${COTRACKER_HUB_DIR}"
echo "============================================"

download_url() {
    URL="$1"
    OUT="$2"
    if [ -f "$OUT" ]; then
        echo "  ✓ 已存在: $OUT"
        return 0
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -O "$OUT" "$URL"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -o "$OUT" "$URL"
    else
        echo "错误: 需要 wget 或 curl 来下载: $URL"
        exit 1
    fi
}

# -----------------------------------------------
# 1. 下载 Wan2.1-Fun-1.3B-InP 基础模型
# -----------------------------------------------
echo ""
echo "[1/6] 下载 Wan2.1-Fun-1.3B-InP 基础模型 (~19GB) ..."

if [ "$SOURCE" = "modelscope" ]; then
    modelscope download \
        --model PAI/Wan2.1-Fun-1.3B-InP \
        --local_dir "${MODEL_DIR}/wan_1.3b"
elif [ "$SOURCE" = "hf" ]; then
    huggingface-cli download \
        alibaba-pai/Wan2.1-Fun-1.3B-InP \
        --local-dir "${MODEL_DIR}/wan_1.3b"
else
    echo "错误: 不支持的下载源 '${SOURCE}'，请使用 'modelscope' 或 'hf'"
    exit 1
fi

echo "[1/6] Wan2.1-Fun-1.3B-InP 下载完成！"

# -----------------------------------------------
# 2. 下载 Wan2.1 1.3B Motion Controller
# -----------------------------------------------
echo ""
echo "[2/6] 下载 Wan2.1 1.3B Motion Controller ..."

if [ "$SOURCE" = "modelscope" ]; then
    modelscope download \
        --model DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1 \
        --local_dir "${MODEL_DIR}/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1"
elif [ "$SOURCE" = "hf" ]; then
    echo "注意: Wan2.1 1.3B Motion Controller 目前按 ModelScope 路径下载，自动切换到 ModelScope..."
    modelscope download \
        --model DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1 \
        --local_dir "${MODEL_DIR}/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1"
fi

echo "[2/6] Wan2.1 1.3B Motion Controller 下载完成！"

# -----------------------------------------------
# 3. 下载 Wan2.1 1.3B VACE
# -----------------------------------------------
echo ""
echo "[3/6] 下载 Wan2.1 1.3B VACE ..."

if [ "$SOURCE" = "modelscope" ]; then
    modelscope download \
        --model iic/VACE-Wan2.1-1.3B-Preview \
        --local_dir "${MODEL_DIR}/iic/VACE-Wan2.1-1.3B-Preview"
elif [ "$SOURCE" = "hf" ]; then
    echo "注意: Wan2.1 1.3B VACE 目前按 ModelScope 路径下载，自动切换到 ModelScope..."
    modelscope download \
        --model iic/VACE-Wan2.1-1.3B-Preview \
        --local_dir "${MODEL_DIR}/iic/VACE-Wan2.1-1.3B-Preview"
fi

echo "[3/6] Wan2.1 1.3B VACE 下载完成！"

# -----------------------------------------------
# 4. 下载 MotionCanvas 预训练权重
# -----------------------------------------------
echo ""
echo "[4/6] 下载 MotionCanvas 预训练权重 (~3.1GB) ..."

if [ "$SOURCE" = "modelscope" ]; then
    modelscope download \
        --model doubiiu/MotionCanvas \
        --local_dir "${MODEL_DIR}/motioncanvas"
elif [ "$SOURCE" = "hf" ]; then
    echo "注意: MotionCanvas 权重仅在 ModelScope 提供，自动切换到 ModelScope 下载..."
    modelscope download \
        --model doubiiu/MotionCanvas \
        --local_dir "${MODEL_DIR}/motioncanvas"
fi

echo "[4/6] MotionCanvas 权重下载完成！"

# -----------------------------------------------
# 5. 预下载 CoTracker 到本地缓存
# -----------------------------------------------
echo ""
echo "[5/6] 预下载 CoTracker (torch.hub) ..."
mkdir -p "${COTRACKER_HUB_DIR}"
python - <<'PY'
import os
import torch

hub_dir = os.environ.get("COTRACKER_HUB_DIR", "/root/autodl-tmp/torch_hub")
os.makedirs(hub_dir, exist_ok=True)
torch.hub.set_dir(hub_dir)
torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", trust_repo=True)
print(f"CoTracker cached at {hub_dir}")
PY

echo "[5/6] CoTracker 预下载完成！"

# -----------------------------------------------
# 6. 下载 SAM (Segment Anything) checkpoint
# -----------------------------------------------
echo ""
echo "[6/6] 下载 SAM (Segment Anything) checkpoint (~2.6GB) ..."

SAM_CKPT_NAME="sam_vit_h_4b8939.pth"
SAM_CKPT_PATH="${MODEL_DIR}/segment_anything/${SAM_CKPT_NAME}"
SAM_CKPT_URL="https://dl.fbaipublicfiles.com/segment_anything/${SAM_CKPT_NAME}"

download_url "$SAM_CKPT_URL" "$SAM_CKPT_PATH"
echo "[6/6] SAM checkpoint 下载完成！"

# -----------------------------------------------
# 验证文件完整性
# -----------------------------------------------
echo ""
echo "============================================"
echo " 验证模型文件..."
echo "============================================"

MISSING=0

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo "  ✓ $1 (${SIZE})"
    else
        echo "  ✗ $1 [缺失]"
        MISSING=$((MISSING + 1))
    fi
}

echo "Wan2.1-Fun-1.3B-InP:"
check_file "${MODEL_DIR}/wan_1.3b/models_t5_umt5-xxl-enc-bf16.pth"
check_file "${MODEL_DIR}/wan_1.3b/Wan2.1_VAE.pth"
check_file "${MODEL_DIR}/wan_1.3b/diffusion_pytorch_model.safetensors"
check_file "${MODEL_DIR}/wan_1.3b/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"

echo ""
echo "Wan2.1 1.3B Motion Controller:"
check_file "${MODEL_DIR}/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1/model.safetensors"

echo ""
echo "Wan2.1 1.3B VACE:"
check_file "${MODEL_DIR}/iic/VACE-Wan2.1-1.3B-Preview/diffusion_pytorch_model.safetensors"
check_file "${MODEL_DIR}/iic/VACE-Wan2.1-1.3B-Preview/models_t5_umt5-xxl-enc-bf16.pth"
check_file "${MODEL_DIR}/iic/VACE-Wan2.1-1.3B-Preview/Wan2.1_VAE.pth"

echo ""
echo "MotionCanvas:"
check_file "${MODEL_DIR}/motioncanvas/model.pt"

echo ""
echo "SAM (Segment Anything):"
check_file "${MODEL_DIR}/segment_anything/sam_vit_h_4b8939.pth"

echo ""
echo "CoTracker (torch.hub):"
if [ -d "${COTRACKER_HUB_DIR}/facebookresearch_co-tracker_main" ] || [ -d "${COTRACKER_HUB_DIR}/hub/facebookresearch_co-tracker_main" ]; then
    echo "  ✓ ${COTRACKER_HUB_DIR}/facebookresearch_co-tracker_main"
else
    echo "  ✗ ${COTRACKER_HUB_DIR}/facebookresearch_co-tracker_main [缺失]"
    MISSING=$((MISSING + 1))
fi

echo ""
if [ $MISSING -eq 0 ]; then
    echo "所有模型文件已就绪！"
else
    echo "警告: 有 ${MISSING} 个文件缺失，请检查下载是否完整。"
    exit 1
fi

echo ""
echo "============================================"
echo " 下载完成！可以运行推理："
echo "   bash run.sh"
echo "============================================"
