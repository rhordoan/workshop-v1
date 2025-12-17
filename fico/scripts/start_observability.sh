#!/usr/bin/env bash
set -euo pipefail

# Starts a lightweight observability stack for the workshop:
#   - OpenTelemetry Collector (receives OTLP HTTP on :4318 and gRPC on :4317)
#   - Jaeger (UI on :16686)
#
# Notebook(s) can export spans to:
#   OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces
#
# Requirements:
#   - docker installed and usable by the current user
#
# Usage:
#   cd fico
#   chmod +x scripts/start_observability.sh scripts/stop_observability.sh
#   ./scripts/start_observability.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "Missing required command: $1" >&2; exit 1; }; }
require_cmd docker

NETWORK="${OBS_NETWORK:-nim-net}"
JAEGER_NAME="${OBS_JAEGER_NAME:-obs-jaeger}"
COLLECTOR_NAME="${OBS_COLLECTOR_NAME:-obs-otel-collector}"

JAEGER_IMAGE="${OBS_JAEGER_IMAGE:-jaegertracing/all-in-one:latest}"
COLLECTOR_IMAGE="${OBS_COLLECTOR_IMAGE:-otel/opentelemetry-collector-contrib:latest}"

OBS_CACHE_DIR="${OBS_CACHE_DIR:-$HOME/.cache/nim/observability}"
mkdir -p "${OBS_CACHE_DIR}"

stop_if_running() {
  local name="$1"
  if docker ps -a --format '{{.Names}}' | grep -qx "${name}"; then
    echo "==> Removing existing container: ${name}"
    docker rm -f "${name}" >/dev/null || true
  fi
}

echo "==> Create docker network (${NETWORK}) if needed"
docker network inspect "${NETWORK}" >/dev/null 2>&1 || docker network create "${NETWORK}" >/dev/null

stop_if_running "${COLLECTOR_NAME}"
stop_if_running "${JAEGER_NAME}"

echo "==> Write OpenTelemetry Collector config"
COLLECTOR_CONFIG="${OBS_CACHE_DIR}/otel-collector.yaml"
cat > "${COLLECTOR_CONFIG}" <<'YAML'
receivers:
  otlp:
    protocols:
      grpc:
      http:

processors:
  batch:

exporters:
  otlp/jaeger:
    endpoint: obs-jaeger:4317
    tls:
      insecure: true

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [batch]
      exporters: [otlp/jaeger]
YAML

echo "==> Start Jaeger (UI: http://localhost:16686)"
docker run --detach --restart unless-stopped --name "${JAEGER_NAME}" \
  --network "${NETWORK}" \
  -p 16686:16686 \
  -e COLLECTOR_OTLP_ENABLED=true \
  "${JAEGER_IMAGE}" >/dev/null

echo "==> Start OpenTelemetry Collector (OTLP: http://localhost:4318/v1/traces)"
docker run --detach --restart unless-stopped --name "${COLLECTOR_NAME}" \
  --network "${NETWORK}" \
  -p 4317:4317 \
  -p 4318:4318 \
  -v "${COLLECTOR_CONFIG}:/etc/otelcol-contrib/config.yaml:ro" \
  "${COLLECTOR_IMAGE}" \
  --config /etc/otelcol-contrib/config.yaml >/dev/null

cat <<EOF

✅ Observability stack is up.

- Jaeger UI:            http://localhost:16686
- OTLP HTTP endpoint:   http://localhost:4318/v1/traces

Environment variables (for notebooks):
  export OTEL_SERVICE_NAME="fico-rag-observability"
  export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318/v1/traces"

To stop:
  ${ROOT_DIR}/scripts/stop_observability.sh
EOF


