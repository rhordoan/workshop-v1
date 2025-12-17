#!/usr/bin/env bash
set -euo pipefail

# Starts the 3 services expected by this repo's notebook client:
#   - POST /v1/embeddings
#   - POST /v1/rerank
#   - POST /v1/chat/completions
#
# It runs three NIM containers (one per capability) and a small nginx gateway
# on http://localhost:8000 that routes each path to the correct backend.
#
# Requirements:
#   - docker installed and usable by the current user
#   - NVIDIA GPU + nvidia-container-toolkit working (for --gpus all)
#   - NGC API key exported as NGC_API_KEY (for pulling from nvcr.io)
#
# Usage:
#   export NGC_API_KEY="..."   # from https://ngc.nvidia.com/
#   ./scripts/start_nims.sh
#
# Optional overrides:
#   export NIM_EMBED_IMAGE="nvcr.io/nim/nvidia/nv-embedqa-mistral-7b-v2:1.0.1"
#   export NIM_RERANK_IMAGE="nvcr.io/nim/nvidia/nv-rerankqa-mistral-4b-v3:1.0.2"
#   export NIM_GEN_IMAGE="nvcr.io/nim/qwen/qwen-2.5-7b-instruct:latest"
#   export NIM_EMBED_MODEL="nvidia/nv-embedqa-mistral-7b-v2"
#   export NIM_RERANK_MODEL="nvidia/nv-rerankqa-mistral-4b-v3"
#   export NIM_GEN_MODEL="qwen/qwen-2.5-7b-instruct"
#   export NIM_CACHE_DIR="$HOME/.cache/nim"
#   export START_EMBED=1
#   export START_RERANK=1
#   export START_GEN=0
#   # Convenience for eval notebooks that only need a chat/judge model:
#   export START_JUDGE_ONLY=1  # implies START_EMBED=0 START_RERANK=0 START_GEN=1
#
# Notes:
# - Image names/tags vary by NIM release. If pulls fail, look up the exact
#   container names in the NGC NIM catalog and override *_IMAGE variables.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
require_cmd docker

NGC_API_KEY="${NGC_API_KEY:-}"
if [[ -z "${NGC_API_KEY}" ]]; then
  cat >&2 <<'EOF'
NGC_API_KEY is not set.

Create an API key in NGC and export it, then re-run:
  export NGC_API_KEY="..."
EOF
  exit 1
fi

NIM_EMBED_IMAGE="${NIM_EMBED_IMAGE:-nvcr.io/nim/nvidia/nv-embedqa-mistral-7b-v2:1.0.1}"
NIM_RERANK_IMAGE="${NIM_RERANK_IMAGE:-nvcr.io/nim/nvidia/nv-rerankqa-mistral-4b-v3:1.0.2}"
NIM_GEN_IMAGE="${NIM_GEN_IMAGE:-nvcr.io/nim/qwen/qwen-2.5-7b-instruct:latest}"

NIM_EMBED_MODEL="${NIM_EMBED_MODEL:-nvidia/nv-embedqa-mistral-7b-v2}"
NIM_RERANK_MODEL="${NIM_RERANK_MODEL:-nvidia/nv-rerankqa-mistral-4b-v3}"
NIM_GEN_MODEL="${NIM_GEN_MODEL:-qwen/qwen-2.5-7b-instruct}"

START_EMBED="${START_EMBED:-1}"
START_RERANK="${START_RERANK:-1}"
# IMPORTANT: Most workshop VMs (even A100 80GB) cannot co-run a heavy chat NIM plus
# the embedding + rerank NIMs with TensorRT profiles simultaneously without OOM.
# Start chat separately when needed.
START_GEN="${START_GEN:-0}"

# Convenience: for RAG eval notebooks, we often only need a chat model for judging.
START_JUDGE_ONLY="${START_JUDGE_ONLY:-0}"
if [[ "${START_JUDGE_ONLY}" == "1" ]]; then
  START_EMBED="0"
  START_RERANK="0"
  START_GEN="1"
fi

NIM_CACHE_DIR="${NIM_CACHE_DIR:-$HOME/.cache/nim}"
mkdir -p "${NIM_CACHE_DIR}"
chmod -R a+rwx "${NIM_CACHE_DIR}" || true

NETWORK="nim-net"
GATEWAY_NAME="nim-gateway"
EMBED_NAME="nim-embed"
RERANK_NAME="nim-rerank"
GEN_NAME="nim-gen"

echo "==> Docker login to nvcr.io (non-interactive)"
docker login nvcr.io -u '$oauthtoken' -p "${NGC_API_KEY}" >/dev/null

echo "==> Create docker network (${NETWORK}) if needed"
docker network inspect "${NETWORK}" >/dev/null 2>&1 || docker network create "${NETWORK}" >/dev/null

stop_if_running() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    echo "==> Removing existing container: ${name}"
    docker rm -f "${name}" >/dev/null
  fi
}

stop_if_running "${GATEWAY_NAME}"
stop_if_running "${EMBED_NAME}"
stop_if_running "${RERANK_NAME}"
stop_if_running "${GEN_NAME}"

echo "==> Pull images"
if [[ "${START_EMBED}" == "1" ]]; then docker pull "${NIM_EMBED_IMAGE}"; fi
if [[ "${START_RERANK}" == "1" ]]; then docker pull "${NIM_RERANK_IMAGE}"; fi
if [[ "${START_GEN}" == "1" ]]; then docker pull "${NIM_GEN_IMAGE}"; fi

COMMON_ENV=(
  # Some NIM images require one of these to accept terms non-interactively.
  -e "ACCEPT_EULA=Y"
  -e "NIM_ACCEPT_EULA=Y"
  # NIM pulls model artifacts/profiles from NGC at runtime; it needs an API key inside the container.
  -e "NGC_API_KEY=${NGC_API_KEY}"
  # Some NIM builds/tools look for this name instead.
  -e "NGC_CLI_API_KEY=${NGC_API_KEY}"
)

COMMON_RUN=(
  --detach
  --restart unless-stopped
  --network "${NETWORK}"
  --gpus all
  -v "${NIM_CACHE_DIR}:/opt/nim/.cache"
)

if [[ "${START_EMBED}" == "1" ]]; then
  echo "==> Start embedding NIM (${EMBED_NAME})"
  docker run "${COMMON_RUN[@]}" --name "${EMBED_NAME}" \
    "${COMMON_ENV[@]}" \
    -e "NIM_MODEL_NAME=${NIM_EMBED_MODEL}" \
    -e "NIM_MODEL=${NIM_EMBED_MODEL}" \
    -e "MODEL_NAME=${NIM_EMBED_MODEL}" \
    "${NIM_EMBED_IMAGE}" >/dev/null
else
  echo "==> Skipping embedding NIM (START_EMBED!=1)"
fi

if [[ "${START_RERANK}" == "1" ]]; then
  echo "==> Start rerank NIM (${RERANK_NAME})"
  docker run "${COMMON_RUN[@]}" --name "${RERANK_NAME}" \
    "${COMMON_ENV[@]}" \
    -e "NIM_MODEL_NAME=${NIM_RERANK_MODEL}" \
    -e "NIM_MODEL=${NIM_RERANK_MODEL}" \
    -e "MODEL_NAME=${NIM_RERANK_MODEL}" \
    "${NIM_RERANK_IMAGE}" >/dev/null
else
  echo "==> Skipping rerank NIM (START_RERANK!=1)"
fi

if [[ "${START_GEN}" == "1" ]]; then
  echo "==> Start generation/chat NIM (${GEN_NAME})"
  docker run "${COMMON_RUN[@]}" --name "${GEN_NAME}" \
    "${COMMON_ENV[@]}" \
    -e "NIM_MODEL_NAME=${NIM_GEN_MODEL}" \
    -e "NIM_MODEL=${NIM_GEN_MODEL}" \
    -e "MODEL_NAME=${NIM_GEN_MODEL}" \
    "${NIM_GEN_IMAGE}" >/dev/null
else
  echo "==> Skipping generation/chat NIM (START_GEN!=1)"
fi

echo "==> Write nginx gateway config"
GATEWAY_CONF_DIR="${NIM_CACHE_DIR}/gateway"
mkdir -p "${GATEWAY_CONF_DIR}"
cat > "${GATEWAY_CONF_DIR}/nginx.conf" <<'NGINX_HEAD'
worker_processes  1;
events { worker_connections 1024; }

http {
  # Conservative defaults for large responses
  client_max_body_size 50m;
  proxy_read_timeout 300s;
  proxy_send_timeout 300s;

  server {
    listen 8000;
NGINX_HEAD

if [[ "${START_EMBED}" == "1" ]]; then
  cat >> "${GATEWAY_CONF_DIR}/nginx.conf" <<'NGINX_EMBED'
    # Embeddings
    location /v1/embeddings {
      proxy_pass http://nim-embed:8000;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
NGINX_EMBED
else
  cat >> "${GATEWAY_CONF_DIR}/nginx.conf" <<'NGINX_EMBED_OFF'
    # Embeddings (disabled)
    location /v1/embeddings {
      return 503;
    }
NGINX_EMBED_OFF
fi

if [[ "${START_RERANK}" == "1" ]]; then
  cat >> "${GATEWAY_CONF_DIR}/nginx.conf" <<'NGINX_RERANK'
    # Rerank
    location /v1/rerank {
      # nv-rerank* NIMs commonly expose /v1/ranking. We map our stable /v1/rerank
      # gateway path to the underlying endpoint to keep notebooks consistent.
      proxy_pass http://nim-rerank:8000/v1/ranking;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Also expose the native reranker path (optional)
    location /v1/ranking {
      proxy_pass http://nim-rerank:8000/v1/ranking;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
NGINX_RERANK
else
  cat >> "${GATEWAY_CONF_DIR}/nginx.conf" <<'NGINX_RERANK_OFF'
    # Rerank (disabled)
    location /v1/rerank { return 503; }
    location /v1/ranking { return 503; }
NGINX_RERANK_OFF
fi

if [[ "${START_GEN}" == "1" ]]; then
  cat >> "${GATEWAY_CONF_DIR}/nginx.conf" <<'NGINX_CHAT'
    # Chat
    location /v1/chat/completions {
      proxy_pass http://nim-gen:8000;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
NGINX_CHAT
else
  cat >> "${GATEWAY_CONF_DIR}/nginx.conf" <<'NGINX_CHAT_OFF'
    # Chat
    #
    # NOTE: Do not reference an upstream that doesn't exist in docker DNS. If
    # nim-gen isn't started, nginx would fail to boot.
    location /v1/chat/completions {
      return 503;
    }
NGINX_CHAT_OFF
fi

cat >> "${GATEWAY_CONF_DIR}/nginx.conf" <<'NGINX_TAIL'

    location / {
      return 404;
    }
  }
}
NGINX_TAIL

echo "==> Start nginx gateway on localhost:8000"
docker run --detach --restart unless-stopped --name "${GATEWAY_NAME}" \
  --network "${NETWORK}" \
  -p 8000:8000 \
  -v "${GATEWAY_CONF_DIR}/nginx.conf:/etc/nginx/nginx.conf:ro" \
  nginx:alpine >/dev/null

cat <<EOF

✅ NIM gateway is up (routing via nginx): http://localhost:8000

Check logs:
  docker logs -f ${EMBED_NAME}   # if START_EMBED=1
  docker logs -f ${RERANK_NAME}  # if START_RERANK=1
  docker logs -f ${GEN_NAME}     # if START_GEN=1
  docker logs -f ${GATEWAY_NAME}

To stop everything:
  ${ROOT_DIR}/scripts/stop_nims.sh

Notebook defaults already use NIM_BASE_URL=http://localhost:8000.
EOF




