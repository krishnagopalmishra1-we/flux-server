#!/usr/bin/env python3
"""Download HunyuanVideo to SSD model cache (/app/model_cache).
Run inside container: python3 /tmp/download_hunyuan_inner.py
"""
import os, sys, time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

from huggingface_hub import snapshot_download

MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
CACHE_DIR = "/app/model_cache"

print(f"[{time.strftime('%H:%M:%S')}] Starting HunyuanVideo download")
print(f"[{time.strftime('%H:%M:%S')}] Model: {MODEL_ID}")
print(f"[{time.strftime('%H:%M:%S')}] Cache: {CACHE_DIR}")
print("Estimated: ~40 GB, 30-60 min depending on bandwidth")
sys.stdout.flush()

try:
    path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        ignore_patterns=["*.gguf", "*.ggml"],
    )
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE — downloaded to: {path}")
    sys.stdout.flush()
except Exception as e:
    print(f"\n[{time.strftime('%H:%M:%S')}] FAIL — {e}", file=sys.stderr)
    sys.exit(1)
