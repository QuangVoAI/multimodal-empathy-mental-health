#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/data/raw/avamerg"

mkdir -p "${TARGET_DIR}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: 'hf' CLI not found. Install it first with 'pip install -U huggingface_hub[cli]'." >&2
  exit 1
fi

echo "Downloading AvaMERG train split to ${TARGET_DIR}"
hf download ZhangHanXD/AvaMERG train.json --repo-type dataset --local-dir "${TARGET_DIR}"

echo "Done. You can optionally download audio/video archives later if you need them for multimodal expansion."
