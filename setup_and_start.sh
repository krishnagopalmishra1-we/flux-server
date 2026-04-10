#!/bin/bash
# Setup, install deps, start server, then run smoke test
set -euo pipefail
cd /opt/flux-server || { echo "ERROR: Cannot enter /opt/flux-server" >&2; exit 1; }

echo "=== [1/5] Python environment ==="
python3 --version
which python3

echo "=== [2/5] Install/upgrade dependencies ==="
pip3 install -q --upgrade pip
pip3 install -q -r requirements.txt 2>&1 | tee pip-install.log | tail -20

echo "=== [3/5] Verify video_loras directory ==="
mkdir -p /opt/flux-server/video_loras
ls -la /opt/flux-server/video_loras/ 2>/dev/null || true

echo "=== [4/5] Check/restart server ==="
# Kill any existing server
pkill -f "gunicorn|uvicorn" 2>/dev/null || true
sleep 2

# Check if .env exists
if [ ! -f /opt/flux-server/.env ]; then
  echo "Creating .env from example..."
  cp /opt/flux-server/.env.example /opt/flux-server/.env 2>/dev/null || \
  cat > /opt/flux-server/.env <<'ENV'
HF_TOKEN=
CACHE_DIR=/mnt/hf-cache
OUTPUT_DIR=/mnt/outputs
ENABLE_VIDEO=true
ENABLE_MUSIC=true
ENABLE_ANIMATION=true
WAN_DEFAULT_VARIANT=14b
ENV
fi

# Validate HF_TOKEN — missing token causes silent failures at model download time
HF_TOKEN_VAL=$(grep -E '^HF_TOKEN=' /opt/flux-server/.env 2>/dev/null | cut -d= -f2- || true)
if [ -z "$HF_TOKEN_VAL" ]; then
  echo "WARNING: HF_TOKEN is not set in .env — model downloads from gated HuggingFace repos will fail." >&2
  echo "         Set HF_TOKEN in /opt/flux-server/.env before starting services." >&2
fi

# Ensure output dirs
mkdir -p /mnt/outputs/video /mnt/outputs/audio /mnt/outputs/animation /mnt/hf-cache || true

echo "=== Starting gunicorn with preload_app ==="
cd /opt/flux-server
nohup gunicorn app.main:app -c gunicorn.conf.py \
  --pid /tmp/gunicorn.pid \
  > /tmp/gunicorn.log 2>&1 &
echo "Gunicorn PID: $!"
sleep 5
echo "Server log tail:"
tail -30 /tmp/gunicorn.log

echo ""
echo "=== [5/5] Wait for health endpoint ==="
H=""
for i in $(seq 1 24); do
  H=$(curl -sf http://localhost:8080/health 2>/dev/null || true)
  if [ -n "$H" ]; then
    echo "Server healthy: $H"
    break
  fi
  echo "  Waiting... ($((i*5))s)"
  sleep 5
done
if [ -z "$H" ]; then
  echo "ERROR: Server never became healthy after 120s." >&2
  exit 1
fi
