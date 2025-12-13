#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

PORT="${PORT:-8888}"
IP="${IP:-0.0.0.0}"

echo "==> Starting JupyterLab on ${IP}:${PORT}"
echo "    If running on Brev/remote, make sure the port is exposed."

exec python -m jupyterlab --ip "${IP}" --port "${PORT}" --no-browser


