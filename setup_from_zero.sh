#!/bin/bash
# MotionCanvas 从零环境搭建脚本
# 步骤：安装 uv → 创建虚拟环境 → 安装项目依赖 → 下载模型
#
# 使用方法（在项目根目录执行）:
#   bash setup_from_zero.sh                    # 全流程
#   bash setup_from_zero.sh --skip-models      # 不下载模型
#   bash setup_from_zero.sh --source hf        # 模型从 HuggingFace 下载（Wan 用 HF，MotionCanvas 仍用 ModelScope）
#
# 依赖：curl, 系统 Python 3.8+（仅用于安装 uv）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SKIP_MODELS=false
SOURCE="modelscope"
PYTHON_VERSION=""   # 留空则用系统默认 Python

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "用法: $0 [--skip-models] [--source modelscope|hf] [--python 3.10]"
            exit 1
            ;;
    esac
done

echo "============================================"
echo " MotionCanvas 从零环境搭建"
echo " 项目目录: $SCRIPT_DIR"
echo "============================================"

# -----------------------------------------------
# 1. 安装 uv
# -----------------------------------------------
echo ""
echo "[1/4] 安装 uv ..."

if command -v uv &>/dev/null; then
    echo "  uv 已存在: $(uv --version)"
else
    echo "  正在从 astral.sh 安装 uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    if ! command -v uv &>/dev/null; then
        echo "  请将以下内容加入 ~/.bashrc 或当前 shell 后重试："
        echo "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
        exit 1
    fi
    echo "  安装完成: $(uv --version)"
fi

# -----------------------------------------------
# 2. 创建虚拟环境
# -----------------------------------------------
echo ""
echo "[2/4] 创建虚拟环境 (.venv) ..."

if [ -n "$PYTHON_VERSION" ]; then
    uv venv --python "$PYTHON_VERSION"
else
    uv venv
fi

# 激活虚拟环境（后续命令在同一 shell 中需用到）
source .venv/bin/activate

# -----------------------------------------------
# 3. 安装项目依赖（可编辑模式安装）
# -----------------------------------------------
echo ""
echo "[3/4] 安装项目依赖 (uv pip install -e .) ..."

uv pip install -e .

echo "  依赖安装完成。"
python -c "import torch; print('  PyTorch:', torch.__version__)"
python -c "import gradio; print('  Gradio:', gradio.__version__)"

# -----------------------------------------------
# 4. 下载模型
# -----------------------------------------------
echo ""
echo "[4/4] 下载模型 ..."

if [ "$SKIP_MODELS" = true ]; then
    echo "  已跳过（--skip-models）。稍后手动执行: bash download_models.sh"
else
    if [ ! -f "download_models.sh" ]; then
        echo "  错误: 未找到 download_models.sh"
        exit 1
    fi
    bash download_models.sh --source "$SOURCE"
fi

# -----------------------------------------------
# 完成
# -----------------------------------------------
echo ""
echo "============================================"
echo " 环境已就绪"
echo "============================================"
echo ""
echo "激活虚拟环境:"
echo "  source .venv/bin/activate"
echo ""
echo "运行 Gradio 界面:"
echo "  python apps/gradio/motioncanvas.py"
echo ""
echo "或直接使用 uv 运行（无需激活）:"
echo "  uv run python apps/gradio/motioncanvas.py"
echo ""
