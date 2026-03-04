#!/usr/bin/env bash
# 单机训练脚本（本机 1 张或多张 GPU）
# 使用前请设置下面的环境变量，或编辑本脚本中的默认路径

set -e
cd "$(dirname "$0")"

# ---------- 路径配置（请按需修改） ----------
DATA_PATH="${DATA_PATH:-data/dataset}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/checkpoints}"
DIT_PATH="${DIT_PATH:-models/wan_1.3b/dit}"
VAE_PATH="${VAE_PATH:-models/wan_1.3b/vae}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-models/wan_1.3b/text_encoder}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-models/wan_1.3b/image_encoder}"
# 可选
TXT_PATH="${TXT_PATH:-data/invalid_data.txt}"
MOVING_NOUN_PATH="${MOVING_NOUN_PATH:-.data/filtered_ram_tag_list.txt}"
RESUME_FROM="${RESUME_FROM:-}"

# 单机：使用本机 GPU 数量，默认 1
NUM_GPUS="${NUM_GPUS:-1}"

# 若有虚拟环境则激活
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# 保证能导入 examples/wanvideo 下的 dataset、logger_utils 等
export PYTHONPATH="${PWD}/examples/wanvideo:${PYTHONPATH:-}"

# 单节点训练
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT="${MASTER_PORT:-29500}"

CMD=(
  torchrun
  --nproc_per_node="$NUM_GPUS"
  --nnodes=1
  --node_rank=0
  --master_addr="$MASTER_ADDR"
  --master_port="$MASTER_PORT"
  examples/wanvideo/train_wan_1.3b_motioncanvas.py
  --task train
  --train_architecture full
  --dataset_path "$DATA_PATH"
  --output_path "$OUTPUT_PATH"
  --dit_path "$DIT_PATH"
  --vae_path "$VAE_PATH"
  --text_encoder_path "$TEXT_ENCODER_PATH"
  --image_encoder_path "$IMAGE_ENCODER_PATH"
  --max_epochs 10000
  --learning_rate 5e-5
  --accumulate_grad_batches 1
  --use_gradient_checkpointing
  --batch_size 1
  --dataloader_num_workers 8
  --every_n_train_steps 1000
  --num_frames 49
  --training_strategy deepspeed_stage_1
  --num_nodes 1
  --target_fps 15
)

# 可选参数：无效数据列表、moving noun 列表
[ -n "$TXT_PATH" ] && CMD+=(--invalid_data_path "$TXT_PATH")
[ -n "$MOVING_NOUN_PATH" ] && [ -f "$MOVING_NOUN_PATH" ] && CMD+=(--moving_noun_path "$MOVING_NOUN_PATH")

# 若指定了恢复路径则追加
[ -n "$RESUME_FROM" ] && CMD+=(--resume_from "$RESUME_FROM")

echo ">>> 单机训练: $NUM_GPUS GPU(s), 数据: $DATA_PATH, 输出: $OUTPUT_PATH"
"${CMD[@]}"
