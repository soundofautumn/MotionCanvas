#!/usr/bin/env bash
#
# 从 GPU 服务器拉取 ablation 实验结果到本地并提交 git。
# 排除 bbox_mask.pt 和 track_video.pt（文件大且可从 JSON 重建）。
#
set -eu

SRC="motion_canvas_gpu:/root/MotionCanvas/ablations"
DST="./ablations"

echo ">>> 从服务器同步 ablations ..."
rsync -avz --progress "$SRC/" "$DST/" \
  --exclude="bbox_mask.pt" \
  --exclude="track_video.pt"

echo ""
echo ">>> 暂存到 git ..."
git add ablations/

# 双重保险：确保 .pt 不会被 git 追踪
git reset HEAD -- "ablations/**/bbox_mask.pt" "ablations/**/track_video.pt" 2>/dev/null || true

echo ""
echo ">>> 检查是否有变更待提交 ..."
if git diff --cached --quiet; then
  echo "    无变更，跳过提交"
else
  echo "    变更如下："
  git diff --cached --name-status
  echo ""
  read -p ">>> 是否提交？(y/N) " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    git commit -m "update ablation results"
    echo "    已提交"
  else
    echo "    已暂存未提交，手动执行: git commit"
  fi
fi

echo ""
echo "=== 完成 ==="
