#!/bin/bash
# VM operations: check env, install deps, restart server, run smoke test
# Run as root on VM

echo "=== ENVIRONMENT CHECK ==="
which python3.11 2>/dev/null && python3.11 --version || echo "python3.11 not in PATH"
which python3 2>/dev/null && python3 --version || echo "python3 not in PATH"
timeout 5 find /opt/venv /usr/local/lib /opt /home/*/.virtualenvs /home/*/venv -maxdepth 4 -name "site-packages" -type d 2>/dev/null | head -5

echo "=== FIND PIP ==="
find /usr/bin /usr/local/bin /opt /root -maxdepth 5 -name "pip3*" -type f 2>/dev/null | head -10
find /usr/bin /usr/local/bin /opt /root -maxdepth 5 -name "pip" -type f 2>/dev/null | head -10

echo "=== FIND GUNICORN ==="
which gunicorn 2>/dev/null || \
  find /usr/bin /usr/local/bin /opt /root -maxdepth 5 -name gunicorn -type f 2>/dev/null | head -5

echo "=== CURRENT PACKAGES (torch/diffusers) ==="
python3 -c "import torch; print('torch', torch.__version__)" 2>/dev/null || true
python3 -c "import diffusers; print('diffusers', diffusers.__version__)" 2>/dev/null || true
python3 -c "import sse_starlette; print('sse_starlette OK')" 2>/dev/null || echo "sse_starlette MISSING"
python3 -c "import peft; print('peft', peft.__version__)" 2>/dev/null || echo "peft MISSING"
python3 -c "import bitsandbytes; print('bitsandbytes', bitsandbytes.__version__)" 2>/dev/null || echo "bitsandbytes MISSING"

echo "=== GPU ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader
else
  echo "No NVIDIA GPU or nvidia-smi found"
fi

echo "=== SERVER STATUS ==="
curl -s --max-time 5 --connect-timeout 2 http://localhost:8080/health || echo "Server not responding"

echo "=== SERVER LOGS (last 50 lines) ==="
journalctl -u neural-studio --no-pager -n 50 2>/dev/null || \
  tail -50 /tmp/gunicorn.log 2>/dev/null || \
  echo "No log found"

echo "=== PYTHON PATH ==="
python3 -c "import sys; print('\n'.join(sys.path))" 2>/dev/null || echo "python3 not available"
