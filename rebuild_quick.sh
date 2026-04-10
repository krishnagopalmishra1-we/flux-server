#!/bin/bash
# Quick rebuild (only copy gunicorn.conf and restart)
set -euo pipefail
cd /opt/flux-server

echo "=== Rebuild with fixed gunicorn.conf.py ==="
sudo docker compose build 2>&1 | tail -20
# Note: set -e + pipefail ensures build failure aborts the script even through the pipe.

echo "=== Restart ==="
sudo docker compose down
sudo docker compose up -d
echo "Waiting 45s for startup..."
sleep 45

echo "=== Wait for health ==="
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

echo "=== Container logs ==="
sudo docker compose logs --tail=60
