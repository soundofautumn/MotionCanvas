#!/usr/bin/env bash
#
# 从 GPU 服务器拉取 ablation 实验结果到本地。
# 远端目录：ablation_results，本地目录：ablations
#
set -eu

REMOTE_DIR="ablation_results"
LOCAL_DIR="ablations"
SSH_HOST="motion_canvas_gpu"

echo ">>> 从 $SSH_HOST:~/$REMOTE_DIR/ 同步到本地 ./$LOCAL_DIR/ ..."
rsync -avz --progress "$SSH_HOST:~/MotionCanvas/$REMOTE_DIR/" "./$LOCAL_DIR/" \
  --exclude="bbox_mask.pt" \
  --exclude="track_video.pt" \
  --exclude="__pycache__/"

echo ""
echo "=== 完成: ./$LOCAL_DIR/ ==="
ls -d "$LOCAL_DIR"/*/ 2>/dev/null | sed 's/^/  /' || echo "  (空)"
