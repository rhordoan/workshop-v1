#!/usr/bin/env bash
set -euo pipefail

NETWORK="${OBS_NETWORK:-nim-net}"
JAEGER_NAME="${OBS_JAEGER_NAME:-obs-jaeger}"
COLLECTOR_NAME="${OBS_COLLECTOR_NAME:-obs-otel-collector}"

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
require_cmd docker

rm_if_present() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    echo "==> Removing ${name}"
    docker rm -f "${name}" >/dev/null || true
  fi
}

rm_if_present "${COLLECTOR_NAME}"
rm_if_present "${JAEGER_NAME}"

if docker network inspect "${NETWORK}" >/dev/null 2>&1; then
  echo "==> Leaving docker network ${NETWORK} intact (safe to reuse)."
fi

echo "✅ Stopped observability containers."


