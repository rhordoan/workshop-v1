#!/usr/bin/env bash
set -euo pipefail

# Stops the local Ollama docker container started by ./scripts/start_ollama.sh

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
require_cmd docker

NAME="ollama"

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "==> Stopping/removing container: ${NAME}"
  docker rm -f "${NAME}" >/dev/null
else
  echo "==> No container named ${NAME}."
fi




