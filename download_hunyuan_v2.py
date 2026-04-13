#!/usr/bin/env python3
"""Download HunyuanVideo to SSD model cache (/app/model_cache).
v2: installs hf_transfer for fast downloads, logs per-file progress.
Run inside container: python3 /tmp/download_hunyuan_v2.py
"""
import os, sys, time, subprocess

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

# Install hf_transfer for Rust-based fast HTTP with better stall recovery
print(f"[{time.strftime('%H:%M:%S')}] Installing hf_transfer...")
sys.stdout.flush()
try:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "hf_transfer"],
        check=True, capture_output=True
    )
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    print(f"[{time.strftime('%H:%M:%S')}] hf_transfer enabled (fast Rust-based download)")
except Exception as e:
    print(f"[{time.strftime('%H:%M:%S')}] hf_transfer install failed ({e}), using standard download")
sys.stdout.flush()

from huggingface_hub import snapshot_download

MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
CACHE_DIR = "/app/model_cache"

print(f"[{time.strftime('%H:%M:%S')}] Starting HunyuanVideo download")
print(f"  model : {MODEL_ID}")
print(f"  cache : {CACHE_DIR}")
print(f"  size  : ~40 GB (transformer 15 shards + VAE + text encoder)")
sys.stdout.flush()

start = time.time()
try:
    path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        ignore_patterns=["*.gguf", "*.ggml"],
        max_workers=4,
    )
    elapsed = int(time.time() - start)
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {elapsed}s — {path}")
    sys.stdout.flush()
except Exception as e:
    elapsed = int(time.time() - start)
    print(f"\n[{time.strftime('%H:%M:%S')}] FAIL after {elapsed}s — {e}", file=sys.stderr)
    sys.exit(1)
