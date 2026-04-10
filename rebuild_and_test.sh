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
CNAME=$(sudo docker compose ps -q flux-server 2>/dev/null | head -1)
if [ -z "$CNAME" ]; then
  echo "ERROR: No running flux-server container found — deploy may have failed." >&2
  exit 1
fi
# Verify the deployed code matches the local build by comparing the SHA-256 of main.py
# inside the container against the local copy.
EXPECTED=$(sha256sum /opt/flux-server/app/main.py | awk '{print $1}')
ACTUAL=$(sudo docker exec "$CNAME" sh -c "sha256sum /app/app/main.py" | awk '{print $1}')
if [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "ERROR: /app/app/main.py checksum mismatch — container may be running stale code." >&2
  echo "  expected: $EXPECTED" >&2
  echo "  actual:   $ACTUAL" >&2
  exit 1
fi
echo "Code verification passed (main.py SHA-256: $ACTUAL)"

echo "=== [6/6] Container logs (last 40 lines) ==="
sudo docker compose logs --tail=40
