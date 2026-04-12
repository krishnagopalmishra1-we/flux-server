#!/bin/bash
# Download HunyuanVideo to SSD model cache (/app/model_cache inside container).
# ~40-50 GB download. Run inside container or with correct HF_HOME.
# Usage: bash /opt/flux-server/download_hunyuan.sh
set -uo pipefail

MODEL_ID="hunyuanvideo-community/HunyuanVideo"
CACHE_DIR="/app/model_cache"  # SSD volume — matches SSD_PRIORITY in model_manager.py

echo "[$(date '+%H:%M:%S')] Starting HunyuanVideo download to $CACHE_DIR"
echo "[$(date '+%H:%M:%S')] Model: $MODEL_ID"
df -h $CACHE_DIR 2>/dev/null || df -h /

sudo docker exec flux-server-flux-server-1 python3 - <<'PYEOF'
import os, sys
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

from huggingface_hub import snapshot_download

model_id = "hunyuanvideo-community/HunyuanVideo"
cache_dir = "/app/model_cache"

print(f"Downloading {model_id} to {cache_dir} ...")
print("This will take 30-60 min depending on bandwidth (~40-50 GB)")

try:
    path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        ignore_patterns=["*.gguf", "*.ggml"],  # skip GGUF quantized variants
    )
    print(f"\n[DONE] Downloaded to: {path}")
except Exception as e:
    print(f"\n[FAIL] Download error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

echo "[$(date '+%H:%M:%S')] Download complete."
df -h $CACHE_DIR 2>/dev/null || df -h /
