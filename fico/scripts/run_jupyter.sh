#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -d ".venv" ]]; then
  echo "==> No venv found; running setup"
  ./scripts/setup_workshop.sh
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# If deps weren't installed for some reason, fix it now (keeps workshops smooth).
python - <<'PY' || { echo "==> Missing deps; re-running setup"; ./scripts/setup_workshop.sh; }
import numpy  # noqa: F401
import pandas  # noqa: F401
PY

PORT="${PORT:-8888}"
IP="${IP:-0.0.0.0}"

echo "==> Starting JupyterLab on ${IP}:${PORT}"
echo "    If running on Brev/remote, make sure the port is exposed."

exec python -m jupyterlab --ip "${IP}" --port "${PORT}" --no-browser


