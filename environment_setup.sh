#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv310}"

echo "[1/6] Project root: ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python binary not found: ${PYTHON_BIN}"
    echo "Set it explicitly, for example:"
    echo "  PYTHON_BIN=python3.10 bash environment_setup.sh"
    exit 1
  fi
fi

echo "[2/6] Creating virtual environment at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"

echo "[3/6] Activating virtual environment"
source "${VENV_DIR}/bin/activate"

echo "[4/6] Upgrading pip"
python -m pip install --upgrade pip setuptools wheel

if python - <<'PY' >/dev/null 2>&1
import certifi
print(certifi.where())
PY
then
  export SSL_CERT_FILE="$(python - <<'PY'
import certifi
print(certifi.where())
PY
)"
  echo "[5/6] SSL_CERT_FILE set"
else
  echo "[5/6] certifi not yet available; continuing without SSL_CERT_FILE override"
fi

echo "[6/6] Installing project requirements"
pip install -r requirements.txt

echo "[extra] Registering local Jupyter kernel"
python -m ipykernel install --user --name multimodal-empathy-mental-health --display-name "Python (multimodal-empathy-mental-health)" >/dev/null 2>&1 || true

echo
echo "Environment setup complete."
echo "Next:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  hf auth login"
echo "  python scripts/train_sft.py --help"
