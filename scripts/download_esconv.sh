#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/data/raw/esconv"
TMP_DIR="$(mktemp -d)"

mkdir -p "${TARGET_DIR}"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required to download ESConv." >&2
  exit 1
fi

echo "Cloning ESConv repository into temporary directory..."
git clone --depth=1 https://github.com/thu-coai/Emotional-Support-Conversation.git "${TMP_DIR}/esconv_repo" >/dev/null 2>&1

cp "${TMP_DIR}/esconv_repo/ESConv.json" "${TARGET_DIR}/ESConv.json"

echo "Saved ESConv.json to ${TARGET_DIR}"
