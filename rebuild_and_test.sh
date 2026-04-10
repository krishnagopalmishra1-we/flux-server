#!/bin/bash
# Rebuild Docker container with updated code and run smoke tests
set -euo pipefail

echo "=== [1/6] Pre-build GPU state ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo "=== [2/6] Rebuild Docker image ==="
cd /opt/flux-server
sudo docker compose build --no-cache 2>&1 | tail -30
# Note: set -e + pipefail ensures build failure propagates even through the pipe.

echo "=== [3/6] Restart container ==="
sudo docker compose down
sudo docker compose up -d
echo "Waiting 30s for server to start..."
sleep 30

echo "=== [4/6] Wait for health ==="
H=""
for i in $(seq 1 24); do
  H=$(curl -sf http://localhost:8080/health 2>/dev/null || true)
  if [ -n "$H" ]; then
    echo "HEALTHY: $H"
    break
  fi
  echo "  Waiting... ($((i*5))s)"
  sleep 5
done
if [ -z "$H" ]; then
  echo "ERROR: Server never became healthy after 120s. Aborting." >&2
  exit 1
fi

echo "=== [5/6] Verify new code is running ==="
CNAME=$(sudo docker ps -q --filter name=flux-server 2>/dev/null | head -1)
if [ -n "$CNAME" ]; then
  sudo docker exec "$CNAME" head -5 /app/app/main.py
else
  echo "WARNING: No running flux-server container found" >&2
fi

echo "=== [6/6] Container logs (last 40 lines) ==="
sudo docker compose logs --tail=40
