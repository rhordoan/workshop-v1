#!/usr/bin/env bash
set -euo pipefail

# Workshop bootstrap:
# - creates a local venv (./.venv) if not present
# - installs Python deps from requirements.txt
# - registers a Jupyter kernel named "Python (fico)"
# - optionally prefetches key models to avoid first-run downloads

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python3}"
PREFETCH_MODELS="${PREFETCH_MODELS:-1}"
PREFETCH_STRICT="${PREFETCH_STRICT:-0}"

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
if [[ ! -f requirements.txt ]]; then
  echo "ERROR: requirements.txt not found under ${ROOT_DIR}" >&2
  exit 1
fi
python -m pip install -r requirements.txt

echo "==> Ensuring Jupyter kernel for this venv is registered"
python -m pip install -q ipykernel
python -m ipykernel install --user --name fico --display-name "Python (fico)" >/dev/null

if [[ "${PREFETCH_MODELS}" == "1" ]]; then
  echo
  echo "==> Prefetching workshop models (set PREFETCH_MODELS=0 to skip)"
  python - <<'PY' || {
    echo "⚠️ Model prefetch failed (non-fatal). You can still run notebooks; first-run may download." >&2
    if [[ "${PREFETCH_STRICT}" == "1" ]]; then
      echo "PREFETCH_STRICT=1 set; failing setup." >&2
      exit 1
    fi
  }
import os
print("prefetch: starting")
print("HF_HOME:", os.environ.get("HF_HOME"))
print("HUGGINGFACE_HUB_CACHE:", os.environ.get("HUGGINGFACE_HUB_CACHE"))
print("GENSIM_DATA_DIR:", os.environ.get("GENSIM_DATA_DIR"))

# 1) Sentence embeddings
from sentence_transformers import SentenceTransformer
st_id = "sentence-transformers/all-MiniLM-L6-v2"
print(f"prefetch: SentenceTransformer({st_id})")
st = SentenceTransformer(st_id, device="cpu")
_ = st.encode(["prefetch test"], normalize_embeddings=True, show_progress_bar=False)

# 2) GPT-2 demo model
from transformers import AutoTokenizer, AutoModelForCausalLM
gpt2_id = "gpt2"
print(f"prefetch: AutoTokenizer/AutoModelForCausalLM({gpt2_id})")
tok = AutoTokenizer.from_pretrained(gpt2_id)
mdl = AutoModelForCausalLM.from_pretrained(gpt2_id)
inputs = tok("prefetch test", return_tensors="pt")
_ = mdl(**inputs)

# 3) Word vectors for analogies (king - man + woman)
import gensim.downloader as api
glove_id = "glove-wiki-gigaword-50"
print(f"prefetch: gensim.downloader.load({glove_id})")
wv = api.load(glove_id)
_ = wv["king"]

print("prefetch: done")
PY
fi

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
