#!/bin/bash
# MotionCanvas 模型下载脚本
# 需要下载两个模型：
#   1. Wan2.1-Fun-1.3B-InP 基础模型 (~19GB)
#   2. MotionCanvas 预训练权重   (~3.1GB)
#
# 使用方法:
#   bash download_models.sh              # 默认从 ModelScope 下载
#   bash download_models.sh --source hf  # 从 HuggingFace 下载
#
# 依赖: pip install modelscope   (ModelScope 下载)
#        pip install huggingface_hub  (HuggingFace 下载)

set -e

SOURCE="modelscope"
MODEL_DIR="models"

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
mkdir -p "${MODEL_DIR}/motioncanvas"

echo "============================================"
echo " MotionCanvas 模型下载"
echo " 下载源: ${SOURCE}"
echo " 保存目录: ${MODEL_DIR}"
echo "============================================"

# -----------------------------------------------
# 1. 下载 Wan2.1-Fun-1.3B-InP 基础模型
# -----------------------------------------------
echo ""
echo "[1/2] 下载 Wan2.1-Fun-1.3B-InP 基础模型 (~19GB) ..."

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

echo "[1/2] Wan2.1-Fun-1.3B-InP 下载完成！"

# -----------------------------------------------
# 2. 下载 MotionCanvas 预训练权重
# -----------------------------------------------
echo ""
echo "[2/2] 下载 MotionCanvas 预训练权重 (~3.1GB) ..."

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

echo "[2/2] MotionCanvas 权重下载完成！"

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
echo "MotionCanvas:"
check_file "${MODEL_DIR}/motioncanvas/motioncanvas.pt"

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
