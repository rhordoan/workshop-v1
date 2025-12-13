#!/usr/bin/env bash
set -euo pipefail

# Workshop bootstrap:
# - creates a local venv (./.venv) if not present
# - installs Python deps from requirements.txt
# - prints a quick sanity check

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python3}"

if ! command -v "${PYTHON}" >/dev/null 2>&1; then
  echo "Missing ${PYTHON}. Install Python 3.10+ and re-run." >&2
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "==> Creating venv at ${ROOT_DIR}/.venv"
  "${PYTHON}" -m venv .venv
fi

echo "==> Activating venv"
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
python -m pip install --upgrade pip wheel setuptools

echo "==> Installing requirements (may take a few minutes)"
python -m pip install -r requirements.txt

echo "==> Ensuring Jupyter kernel for this venv is registered"
# ipykernel is sometimes not installed if the base image provides Jupyter separately.
python -m pip install -q ipykernel
python -m ipykernel install --user --name fico --display-name "Python (fico)" >/dev/null

echo
echo "==> Sanity check"
python - <<'PY'
import sys
print("python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
except Exception as e:
    print("torch: not available (ok for non-local-gen paths):", type(e).__name__, str(e)[:120])
PY

echo
echo "✅ Setup complete."
echo "Next:"
echo "  ./scripts/run_jupyter.sh"


