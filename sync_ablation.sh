#!/usr/bin/env bash
#
# 从 GPU 服务器拉取 ablation 实验结果到本地。
# 两端目录名一致（ablation_results），便于 git 管理。
#
set -eu

DIR="ablation_results"
SSH_HOST="motion_canvas_gpu"

echo ">>> 从 $SSH_HOST:~/$DIR/ 同步到本地 ./$DIR/ ..."
rsync -avz --progress "$SSH_HOST:~/MotionCanvas/$DIR/" "./$DIR/" \
  --exclude="bbox_mask.pt" \
  --exclude="track_video.pt" \
  --exclude="__pycache__/"

echo ""
echo "=== 完成: ./$DIR/ ==="
ls -d "$DIR"/*/ 2>/dev/null | sed 's/^/  /' || echo "  (空)"
