#!/usr/bin/env python3
"""Download HunyuanVideo — v3: standard requests, single worker, socket timeout.
No hf_transfer (its resume logic stalls on pre-existing incomplete blobs).
Run inside container: python3 /tmp/download_hunyuan_v3.py
"""
import os, sys, time, socket

# Standard requests only — no hf_transfer
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

# 5-min socket timeout so connections don't hang forever
socket.setdefaulttimeout(300)

from huggingface_hub import snapshot_download

MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
CACHE_DIR = "/app/model_cache"

print(f"[{time.strftime('%H:%M:%S')}] HunyuanVideo download v3 — standard requests, clean blobs")
print(f"  model : {MODEL_ID}")
print(f"  cache : {CACHE_DIR}")
print(f"  size  : ~40 GB  (15 transformer shards + VAE + text encoder)")
sys.stdout.flush()

start = time.time()
try:
    path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        ignore_patterns=["*.gguf", "*.ggml"],
        max_workers=1,    # single worker avoids concurrent blob lock issues
    )
    elapsed = int(time.time() - start)
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {elapsed}s ({elapsed//3600}h{(elapsed%3600)//60}m) — {path}")
    sys.stdout.flush()
except Exception as e:
    elapsed = int(time.time() - start)
    print(f"\n[{time.strftime('%H:%M:%S')}] FAIL after {elapsed}s — {e}", file=sys.stderr)
    sys.exit(1)
