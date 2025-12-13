#!/usr/bin/env bash
set -euo pipefail

NETWORK="nim-net"
GATEWAY_NAME="nim-gateway"
EMBED_NAME="nim-embed"
RERANK_NAME="nim-rerank"
GEN_NAME="nim-gen"

rm_if_present() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    echo "==> Removing ${name}"
    docker rm -f "${name}" >/dev/null || true
  fi
}

rm_if_present "${GATEWAY_NAME}"
rm_if_present "${EMBED_NAME}"
rm_if_present "${RERANK_NAME}"
rm_if_present "${GEN_NAME}"

if docker network inspect "${NETWORK}" >/dev/null 2>&1; then
  echo "==> Leaving docker network ${NETWORK} intact (safe to reuse)."
fi

echo "✅ Stopped NIM containers."




