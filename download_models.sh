#!/bin/bash
# MotionCanvas 模型下载脚本
# 需要下载六类模型：
#   1. Wan2.1-Fun-1.3B-InP 基础模型 (~19GB)
#   2. Wan2.1 1.3B Motion Controller (~几百 MB)
#   3. Wan2.1 1.3B VACE (~数 GB)
#   4. MotionCanvas 预训练权重   (~3.1GB)
#   5. GroundingDINO (Transformers) 权重（会下载较大的 .safetensors）
#   6. SAM (Segment Anything) checkpoint (~2.6GB)（从 ModelScope 下载）
#
# 使用方法:
#   bash download_models.sh              # 从 ModelScope 下载
#
# 依赖: pip install modelscope   (ModelScope 下载)

set -e

SOURCE="modelscope"
MODEL_DIR="/root/autodl-tmp/models"
COTRACKER_HUB_DIR="/root/autodl-tmp/torch_hub"

while [[ $# -gt 0 ]]; do
    case $1 in
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

if [ "$SOURCE" != "modelscope" ]; then
    echo "错误: 当前脚本仅支持 ModelScope 下载"
    exit 1
fi

mkdir -p "${MODEL_DIR}/wan_1.3b"
mkdir -p "${MODEL_DIR}/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1"
mkdir -p "${MODEL_DIR}/iic/VACE-Wan2.1-1.3B-Preview"
mkdir -p "${MODEL_DIR}/motioncanvas"
mkdir -p "${MODEL_DIR}/grounding_dino/GroundingDINO"
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
    MIN_BYTES="$3"  # optional

    # If file exists and looks complete, skip.
    if [ -f "$OUT" ]; then
        if [ -n "$MIN_BYTES" ]; then
            SIZE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
            if [ "$SIZE" -ge "$MIN_BYTES" ]; then
                echo "  ✓ 已存在且大小正常: $OUT ($(du -h "$OUT" | cut -f1))"
                return 0
            fi
            echo "  ! 检测到可能未下载完整: $OUT (当前 $(du -h "$OUT" | cut -f1))"
            echo "    尝试断点续传..."
        else
            echo "  ✓ 已存在: $OUT"
            return 0
        fi
    fi

    # Try resume download once; if still too small, delete and redownload once.
    for TRY in 1 2; do
        if command -v wget >/dev/null 2>&1; then
            wget -c -O "$OUT" "$URL"
        elif command -v curl >/dev/null 2>&1; then
            curl -L -C - -o "$OUT" "$URL"
        else
            echo "错误: 需要 wget 或 curl 来下载: $URL"
            exit 1
        fi

        if [ -n "$MIN_BYTES" ]; then
            SIZE=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
            if [ "$SIZE" -ge "$MIN_BYTES" ]; then
                return 0
            fi
            if [ "$TRY" -eq 1 ]; then
                echo "  ! 下载后体积仍偏小（$(du -h "$OUT" | cut -f1)），将删除并重试一次..."
                rm -f "$OUT"
            fi
        else
            return 0
        fi
    done

    echo "错误: 下载失败或文件不完整: $OUT"
    exit 1
}

# -----------------------------------------------
# 1. 下载 Wan2.1-Fun-1.3B-InP 基础模型
# -----------------------------------------------
echo ""
echo "[1/7] 下载 Wan2.1-Fun-1.3B-InP 基础模型 (~19GB) ..."

modelscope download \
    --model PAI/Wan2.1-Fun-1.3B-InP \
    --local_dir "${MODEL_DIR}/wan_1.3b"

echo "[1/7] Wan2.1-Fun-1.3B-InP 下载完成！"

# -----------------------------------------------
# 2. 下载 Wan2.1 1.3B Motion Controller
# -----------------------------------------------
echo ""
echo "[2/7] 下载 Wan2.1 1.3B Motion Controller ..."

modelscope download \
    --model DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1 \
    --local_dir "${MODEL_DIR}/DiffSynth-Studio/Wan2.1-1.3b-speedcontrol-v1"

echo "[2/7] Wan2.1 1.3B Motion Controller 下载完成！"

# -----------------------------------------------
# 3. 下载 Wan2.1 1.3B VACE
# -----------------------------------------------
echo ""
echo "[3/7] 下载 Wan2.1 1.3B VACE ..."

modelscope download \
    --model iic/VACE-Wan2.1-1.3B-Preview \
    --local_dir "${MODEL_DIR}/iic/VACE-Wan2.1-1.3B-Preview"

echo "[3/7] Wan2.1 1.3B VACE 下载完成！"

# -----------------------------------------------
# 4. 下载 MotionCanvas 预训练权重
# -----------------------------------------------
echo ""
echo "[4/7] 下载 MotionCanvas 预训练权重 (~3.1GB) ..."

modelscope download \
    --model doubiiu/MotionCanvas \
    --local_dir "${MODEL_DIR}/motioncanvas"

echo "[4/7] MotionCanvas 权重下载完成！"

# -----------------------------------------------
# 5. 预下载 GroundingDINO (Transformers) 权重到本地
# -----------------------------------------------
echo ""
echo "[5/7] 预下载 GroundingDINO (Transformers) 权重 ..."

GDINO_MS_MODEL="IDEA-Research/grounding-dino-base"
GDINO_DIR="${MODEL_DIR}/grounding_dino/GroundingDINO"
modelscope download \
    --model "${GDINO_MS_MODEL}" \
    --local_dir "$GDINO_DIR"

echo "[5/7] GroundingDINO 权重已缓存到本地：${GDINO_DIR}"

# -----------------------------------------------
# 6. 预下载 CoTracker 到本地缓存
# -----------------------------------------------
echo ""
echo "[6/7] 预下载 CoTracker (torch.hub) ..."
mkdir -p "${COTRACKER_HUB_DIR}"
export COTRACKER_HUB_DIR
# 优先使用项目 venv 中的 Python（若存在），否则用系统 Python
PYTHON_BIN="python"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi
$PYTHON_BIN - <<'PY'
import os
import torch

hub_dir = os.environ.get("COTRACKER_HUB_DIR", "/root/autodl-tmp/torch_hub")
os.makedirs(hub_dir, exist_ok=True)
torch.hub.set_dir(hub_dir)
torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline", trust_repo=True)
print(f"CoTracker cached at {hub_dir}")
PY

echo "[6/7] CoTracker 预下载完成！"

# -----------------------------------------------
# 7. 下载 SAM (Segment Anything) checkpoint
# -----------------------------------------------
echo ""
echo "[7/7] 下载 SAM (Segment Anything) checkpoint (~2.6GB) ..."

SAM_MS_MODEL="muse/sam_vit_h_4b8939"
modelscope download \
    --model "${SAM_MS_MODEL}" \
    --local_dir "${MODEL_DIR}/segment_anything"

SAM_CKPT_NAME="sam_vit_h_4b8939.pth"
SAM_CKPT_PATH="${MODEL_DIR}/segment_anything/${SAM_CKPT_NAME}"
echo "[7/7] SAM checkpoint 下载完成！"

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
echo "GroundingDINO (Transformers):"
GDINO_ANY=$(find "${MODEL_DIR}/grounding_dino/GroundingDINO" -maxdepth 5 -type f \( -name "*.safetensors" -o -name "pytorch_model.bin" -o -name "model.safetensors" \) 2>/dev/null | head -n 1)
if [ -n "${GDINO_ANY}" ]; then
    echo "  ✓ ${MODEL_DIR}/grounding_dino/GroundingDINO (found: $(basename "${GDINO_ANY}"))"
else
    echo "  ✗ ${MODEL_DIR}/grounding_dino/GroundingDINO [缺失或为空]"
    MISSING=$((MISSING + 1))
fi

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
