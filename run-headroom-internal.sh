#!/usr/bin/env bash
# Start the headroom-internal GPU container in the desired runtime state.
# Run on the GPU host (e.g. 100.77.242.54).
#
# Usage:
#   bash run-headroom-internal.sh                 # default settings
#   PORT=8788 NAME=headroom-internal-test bash run-headroom-internal.sh
#   GPUS=none bash run-headroom-internal.sh       # CPU-only (skip --gpus)
#
# Stops + removes any existing container with the same name first, then boots
# the new one. Idempotent.

set -euo pipefail

IMAGE="${IMAGE:-headroom-internal:gpu}"
NAME="${NAME:-headroom-internal-test}"
PORT="${PORT:-8788}"
DATA_VOLUME="${DATA_VOLUME:-headroom-internal-data}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
GPUS="${GPUS:-all}"   # set GPUS=none to disable GPU access

GPU_FLAG=()
if [ "$GPUS" != "none" ]; then
  if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    GPU_FLAG=(--gpus "$GPUS")
  else
    echo "warn: nvidia runtime not configured; starting without --gpus." >&2
    echo "      install nvidia-container-toolkit first (install-nvidia-toolkit.sh)." >&2
  fi
fi

echo ">>> stop+remove existing $NAME (if any)"
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo ">>> launch $NAME on :$PORT (image=$IMAGE, gpus=${GPU_FLAG[*]:-none})"
docker run -d \
  --name "$NAME" \
  --restart=unless-stopped \
  -p "${PORT}:8787" \
  -v "${DATA_VOLUME}:/home/nonroot/.headroom" \
  -e "HEADROOM_LOG_LEVEL=${LOG_LEVEL}" \
  "${GPU_FLAG[@]}" \
  "$IMAGE" \
  --host 0.0.0.0 --port 8787 \
  --memory \
  --code-graph \
  --log-messages

echo ">>> wait for /readyz"
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/readyz" >/dev/null 2>&1; then
    echo ">>> ready"
    break
  fi
  sleep 2
done

echo ">>> status"
docker ps --filter "name=^${NAME}$" --format "{{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ">>> banner highlights"
docker logs "$NAME" 2>&1 \
  | grep -iE "Memory:|Optimization:|Cache|Code Graph|Backend|extensions" \
  | head -10 || true

echo
echo ">>> probe"
echo "  curl http://127.0.0.1:${PORT}/readyz"
echo "  curl http://127.0.0.1:${PORT}/stats"
echo "  curl http://127.0.0.1:${PORT}/v1/retrieve/stats"
echo "  curl http://127.0.0.1:${PORT}/v1/toin/stats"
