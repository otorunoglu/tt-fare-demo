#!/bin/bash
# pull_weights.sh
# Pulls the ONNX model bundle from the server's /data/model_cache.
# The remote key is locked to `rrsync -ro /data/model_cache`, so the
# path below is relative to that folder.

set -euo pipefail

# === CONFIG ===
KEY="/home/edge-monitor/.ssh/weights_key"
REMOTE_USER="root"
REMOTE_IP="20.91.249.84"
REMOTE_PORT="50024"
REMOTE_FILE="stable_campion_onnx.zip"
DEST="/home/edge-monitor/equine-ai-monitor/apps/edge-monitor/data/models"

SSH="ssh -i ${KEY} -p ${REMOTE_PORT} -o StrictHostKeyChecking=accept-new"
REMOTE="${REMOTE_USER}@${REMOTE_IP}"

mkdir -p "$DEST"

# 1. Pull the zip to a temp name, then move into place atomically.
#    rsync -c compares by checksum, so an unchanged file transfers nothing.
TMP="${DEST}/.${REMOTE_FILE}.tmp"
trap 'rm -f "$TMP"' EXIT

rsync -az --checksum -e "$SSH" "${REMOTE}:${REMOTE_FILE}" "$TMP" \
    || { echo "failed to pull ${REMOTE_FILE}" >&2; exit 1; }

# 2. Integrity-check before installing (never swap in a truncated zip).
unzip -tq "$TMP" >/dev/null 2>&1 \
    || { echo "downloaded file is not a valid zip" >&2; exit 1; }

mv -f "$TMP" "${DEST}/${REMOTE_FILE}"

# 3. Extract into a fresh dir, then atomically swap the 'current' pointer,
#    so inference never sees a half-extracted model.  Remove this block if
#    you consume the zip directly.
STAGE=$(mktemp -d "${DEST}/.extract.XXXXXX")
unzip -q -o "${DEST}/${REMOTE_FILE}" -d "$STAGE"
rm -rf "${DEST}/current.new"
mv "$STAGE" "${DEST}/current.new"
rm -rf "${DEST}/current.old"
[ -e "${DEST}/current" ] && mv "${DEST}/current" "${DEST}/current.old"
mv "${DEST}/current.new" "${DEST}/current"
rm -rf "${DEST}/current.old"

echo "pulled ${REMOTE_FILE}; extracted to ${DEST}/current"
