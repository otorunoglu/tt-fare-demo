#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:?usage: publish_weights.sh <version-number>}"
[[ "$VERSION" =~ ^[0-9]+$ ]] || { echo "version must be digits only" >&2; exit 1; }

SRC_DIR="${WEIGHTS_SRC:-./out}"        # where the 3 champion files live
REMOTE="model_weights:weights"         # your rclone remote:path
NAME="champion_v${VERSION}"

cd "$SRC_DIR"
files=( "${NAME}_embedder.onnx" "${NAME}_classifier.onnx" "${NAME}_label_mapping.json" )
for f in "${files[@]}"; do
  [[ -f "$f" ]] || { echo "missing $f" >&2; exit 1; }
done

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
zipfile="${tmp}/${NAME}.zip"
zip -0 -j "$zipfile" "${files[@]}"     # -0 store (onnx won't compress), -j flat paths
unzip -t "$zipfile" >/dev/null         # verify before upload

rclone copy "$zipfile" "${REMOTE}/" --progress
echo "published ${NAME}.zip"