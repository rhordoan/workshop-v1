#!/usr/bin/env bash
set -euo pipefail

# Starts a local Ollama server (OpenAI-compatible) on:
#   http://localhost:11434
#
# It also pulls the workshop default model:
#   llama3.1:8b
#
# Usage:
#   ./scripts/start_ollama.sh
#
# Optional overrides:
#   export OLLAMA_PORT=11434
#   export OLLAMA_MODEL="llama3.1:8b"
#   export SKIP_PULL=1

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
require_cmd docker

OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
SKIP_PULL="${SKIP_PULL:-0}"

NAME="ollama"
VOLUME="ollama"

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "==> Removing existing container: ${NAME}"
  docker rm -f "${NAME}" >/dev/null
fi

echo "==> Starting Ollama on http://localhost:${OLLAMA_PORT}"
set +e
docker run -d --restart unless-stopped \
  --name "${NAME}" \
  --gpus all \
  -p "${OLLAMA_PORT}:11434" \
  -v "${VOLUME}:/root/.ollama" \
  ollama/ollama:latest >/dev/null
rc=$?
set -e
if [[ "${rc}" != "0" ]]; then
  echo "⚠️  GPU run failed; starting Ollama without --gpus all (CPU mode)." >&2
  docker run -d --restart unless-stopped \
    --name "${NAME}" \
    -p "${OLLAMA_PORT}:11434" \
    -v "${VOLUME}:/root/.ollama" \
    ollama/ollama:latest >/dev/null
fi

echo "==> Waiting for Ollama to respond..."
for _ in {1..60}; do
  if docker exec "${NAME}" ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if [[ "${SKIP_PULL}" == "1" ]]; then
  echo "==> SKIP_PULL=1; not pulling model."
else
  echo "==> Pulling model: ${OLLAMA_MODEL}"
  docker exec "${NAME}" ollama pull "${OLLAMA_MODEL}"
fi

cat <<EOF

✅ Ollama is running.

OpenAI-compatible endpoint:
  base_url = http://localhost:${OLLAMA_PORT}
  chat     = POST /v1/chat/completions
  model    = ${OLLAMA_MODEL}
EOF




