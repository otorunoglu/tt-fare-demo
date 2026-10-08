#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/equine-ai-monitor/apps/edge-monitor}"
BRANCH="${EDGE_MONITOR_BRANCH:-dev}"
REMOTE="${EDGE_MONITOR_REMOTE:-origin}"

# systemd services usually run with a minimal PATH and no interactive shell profile.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${HOME:-/home/edge-monitor}/.cargo/bin"

required_cmds=(git cargo sudo)
for cmd in "${required_cmds[@]}"; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "[edge-update] required command not found in PATH: $cmd" >&2
        echo "[edge-update] PATH=$PATH" >&2
        exit 127
    fi
done

echo "[edge-update] repo=$REPO_DIR remote=$REMOTE branch=$BRANCH"

cd "$REPO_DIR"

git fetch --prune "$REMOTE"

target_ref="$REMOTE/$BRANCH"
if ! git rev-parse --verify "$target_ref" >/dev/null 2>&1; then
    echo "[edge-update] missing ref: $target_ref" >&2
    exit 1
fi

target_commit="$(git rev-parse "$target_ref")"
current_commit="$(git rev-parse HEAD)"

if [[ "$current_commit" == "$target_commit" ]]; then
    echo "[edge-update] already at $target_ref ($target_commit)"
else
    echo "[edge-update] updating $current_commit -> $target_commit"
fi

# This assumes the checkout is dedicated to deployment. It intentionally avoids merge state.
git reset --hard "$target_ref"

cargo build --release -p webserver

sudo /bin/systemctl restart stable-edge-monitor-webserver.service

echo "[edge-update] completed successfully"