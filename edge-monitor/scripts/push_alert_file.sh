#!/bin/bash
# push_alert.sh  <stable>  <stall>  <absolute-path-to-file>
#
# Pushes a single alert clip to the DGX under:
#   /data/audio/alerts/<stable>/<stall>/
# The remote subfolders are created automatically on first push.
# The ml-backend alert watcher derives stable/stall from these two path levels,
# so the nested layout (not a concatenated name) is required.
#
# Example:
#   ./push_alert.sh stable03 stall03 /home/pi/alerts/clip_001.flac
#       -> root@DGX:/data/audio/alerts/stable03/stall03/clip_001.flac
 
set -euo pipefail
 
# === CONFIG ===
KEY="/home/edge-monitor/.ssh/alert_key"
REMOTE_USER="root"
REMOTE_IP="20.91.249.84"
REMOTE_PORT="50024"
REMOTE_BASE="/data/audio/alerts"
 
# === ARGS ===
if [ "$#" -ne 3 ]; then
    echo "usage: $0 <stable> <stall> <file>" >&2
    exit 2
fi
STABLE="$1"
STALL="$2"
FILE="$3"
 
# === VALIDATION ===
# stable/stall become part of a remote path, so allow only safe chars.
# This blocks "..", "/", spaces, and anything that could escape the folder.
SAFE='^[A-Za-z0-9_-]+$'
[[ "$STABLE" =~ $SAFE ]] || { echo "invalid stable: $STABLE" >&2; exit 2; }
[[ "$STALL"  =~ $SAFE ]] || { echo "invalid stall: $STALL"  >&2; exit 2; }
[ -f "$FILE" ] || { echo "no such file: $FILE" >&2; exit 1; }
 
REMOTE_DIR="${REMOTE_BASE}/${STABLE}/${STALL}"
 
# === PUSH ===
rsync -azq \
    --rsync-path="mkdir -p '${REMOTE_DIR}' && rsync" \
    -e "ssh -i ${KEY} -p ${REMOTE_PORT} -o StrictHostKeyChecking=accept-new" \
    "$FILE" \
    "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/"
 
echo "pushed $(basename "$FILE") -> ${REMOTE_DIR}/"
