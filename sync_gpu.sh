#!/usr/bin/env bash
#
# 从 Windows SSH config 中读取 motion_canvas_gpu 配置，
# 自动更新 git remote gpu 并推送同步。
# 自动处理远端 main 分支被 checkout 导致无法推送的问题。
#
set -eu

SSH_CONFIG="/mnt/c/Users/qjming/.ssh/config"
SSH_HOST="motion_canvas_gpu"
GIT_REMOTE="gpu"
REMOTE_REPO_PATH="~/MotionCanvas"

# ---------- 解析 SSH config ----------
parse_ssh_config() {
    local hostname="" user="" port=""
    local in_block=false

    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="$(echo "$line" | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')"
        [ -z "$line" ] && continue

        key="$(echo "$line" | awk '{print $1}')"
        val="$(echo "$line" | awk '{$1=""; print $0}' | sed 's/^[[:space:]]*//')"

        if echo "$key" | grep -iq '^host$'; then
            if $in_block; then
                break
            fi
            if [ "$val" = "$SSH_HOST" ]; then
                in_block=true
            fi
            continue
        fi

        if $in_block; then
            lkey="$(echo "$key" | tr '[:upper:]' '[:lower:]')"
            case "$lkey" in
                hostname) hostname="$val" ;;
                user)     user="$val" ;;
                port)     port="$val" ;;
            esac
        fi
    done < "$SSH_CONFIG"

    if [ -z "$hostname" ]; then
        echo "ERROR: 在 $SSH_CONFIG 中未找到 Host $SSH_HOST" >&2
        exit 1
    fi

    user="${user:-root}"
    port="${port:-22}"
    echo "$user $hostname $port"
}

# ---------- 主流程 ----------
echo ">>> 读取 SSH config: $SSH_CONFIG"
read -r RUSER HOST PORT <<< "$(parse_ssh_config)"

echo ">>> 解析到 $SSH_HOST:"
echo "    Host     = $HOST"
echo "    User     = $RUSER"
echo "    Port     = $PORT"
echo ""

# ---------- 同步 WSL 侧 SSH config（先写 config，再设 remote，这样 git 会用密钥）--------------
WSL_SSH_DIR="$HOME/.ssh"
WSL_SSH_CONFIG="$WSL_SSH_DIR/config"
mkdir -p "$WSL_SSH_DIR"
chmod 700 "$WSL_SSH_DIR"

BLOCK="Host $SSH_HOST
    HostName $HOST
    User $RUSER
    Port $PORT
    IdentityFile ~/.ssh/motion_canvas_gpu
    StrictHostKeyChecking no"

if [ -f "$WSL_SSH_CONFIG" ] && grep -q "Host $SSH_HOST" "$WSL_SSH_CONFIG"; then
    tmpfile="$(mktemp)"
    awk -v host="$SSH_HOST" '
        /^Host / { if ($2 == host) { skip=1; next } else { skip=0 } }
        skip && /^[[:space:]]/ { next }
        skip && /^[^[:space:]]/ { skip=0 }
        { print }
    ' "$WSL_SSH_CONFIG" > "$tmpfile"
    mv "$tmpfile" "$WSL_SSH_CONFIG"
    printf "\n%s\n" "$BLOCK" >> "$WSL_SSH_CONFIG"
    echo ">>> 已更新 WSL SSH config"
else
    printf "\n%s\n" "$BLOCK" >> "$WSL_SSH_CONFIG"
    echo ">>> 已写入 WSL SSH config"
fi
chmod 600 "$WSL_SSH_CONFIG"

# ---------- 更新 git remote：用 Host 别名，这样 git push 会走 SSH config 用密钥 ----------
# 若用 ssh://root@真实主机/path，git 不会匹配 Host 别名，会走密码认证
ALIAS_URL="${RUSER}@${SSH_HOST}:${REMOTE_REPO_PATH}"
CURRENT_URL="$(git remote get-url "$GIT_REMOTE" 2>/dev/null || true)"

if [ "$CURRENT_URL" = "$ALIAS_URL" ]; then
    echo ">>> remote '$GIT_REMOTE' 地址未变，无需更新"
else
    if [ -n "$CURRENT_URL" ]; then
        echo ">>> 更新 remote '$GIT_REMOTE'（使用别名以启用密钥）:"
        echo "    旧: $CURRENT_URL"
        echo "    新: $ALIAS_URL"
        git remote set-url "$GIT_REMOTE" "$ALIAS_URL"
    else
        echo ">>> 添加 remote '$GIT_REMOTE': $ALIAS_URL"
        git remote add "$GIT_REMOTE" "$ALIAS_URL"
    fi
fi

# ---------- 复制密钥 ----------
WIN_KEY="/mnt/c/Users/qjming/.ssh/motion_canvas_gpu"
WSL_KEY="$WSL_SSH_DIR/motion_canvas_gpu"
if [ -f "$WIN_KEY" ] && [ ! -f "$WSL_KEY" ]; then
    cp "$WIN_KEY" "$WSL_KEY"
    chmod 600 "$WSL_KEY"
    echo ">>> 已复制密钥到 WSL: $WSL_KEY"
elif [ -f "$WIN_KEY" ]; then
    cp "$WIN_KEY" "$WSL_KEY"
    chmod 600 "$WSL_KEY"
fi

# ---------- 配置远端仓库：不存在则创建并初始化，存在则清理暂存区 ----------
echo ""
echo ">>> 配置远端仓库 ..."
ssh -o StrictHostKeyChecking=no "$SSH_HOST" "bash -s" << 'REMOTE_SCRIPT'
REPO="$HOME/MotionCanvas"
if [ -d "$REPO/.git" ]; then
  cd "$REPO" && git config receive.denyCurrentBranch updateInstead
  git reset HEAD -- . 2>/dev/null; git checkout -- . 2>/dev/null
  echo "远端已就绪"
else
  mkdir -p "$REPO" && cd "$REPO" && git init
  git config receive.denyCurrentBranch updateInstead
  echo "已创建并初始化远端仓库"
fi
REMOTE_SCRIPT

# ---------- 推送 ----------
echo ""
BRANCH="$(git symbolic-ref --short HEAD)"
echo ">>> 推送 $BRANCH 到 $GIT_REMOTE ..."

if git push "$GIT_REMOTE" "$BRANCH" 2>&1; then
    echo ">>> 推送成功!"
else
    echo ""
    echo ">>> 普通 push 失败，尝试 force push ..."
    git push --force "$GIT_REMOTE" "$BRANCH" 2>&1
    echo ">>> 强制推送完成!"
fi

# ---------- 远端同步工作区：先切到推送的分支再 reset（新仓库 init 后默认在 master，推送完需切到 main）----------
echo ""
echo ">>> 远端同步工作区 ..."
ssh -o StrictHostKeyChecking=no "$SSH_HOST" \
    "cd ${REMOTE_REPO_PATH} && git checkout ${BRANCH} 2>/dev/null || true && (git rev-parse -q --verify HEAD >/dev/null && git reset --hard HEAD || true) && echo '远端工作区已同步'" 2>&1

echo ""
echo "=== 同步完成 ==="
