#!/usr/bin/env python3
"""Download HunyuanVideo — v4.
Root cause of stall: hf_transfer pre-installed, enabled by default even without env var.
Fix: explicitly disable with "0", not just pop(). Use socket timeout as safety net.
Run inside container: python3 /tmp/download_hunyuan_v4.py
"""
import os, sys, time, socket

# EXPLICITLY disable hf_transfer — pop() is not enough; newer hf_hub auto-enables it
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
os.environ["HUGGINGFACE_HUB_VERBOSITY"] = "warning"

# 5-min socket timeout as safety net for stalled connections
socket.setdefaulttimeout(300)

from huggingface_hub import snapshot_download

MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
CACHE_DIR = "/app/model_cache"

print(f"[{time.strftime('%H:%M:%S')}] HunyuanVideo download v4")
print(f"  hf_transfer : DISABLED (explicit 0)")
print(f"  socket timeout: 300s")
print(f"  model : {MODEL_ID}")
print(f"  cache : {CACHE_DIR}")
print(f"  size  : ~40 GB (15 transformer shards + VAE + text encoder)")
sys.stdout.flush()

# Sanity check: confirm hf_transfer is disabled
try:
    import hf_transfer
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") != "0":
        print("WARNING: hf_transfer installed but env not set — forcing 0")
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    else:
        print(f"  hf_transfer pkg: installed but disabled")
except ImportError:
    print(f"  hf_transfer pkg: not installed")
sys.stdout.flush()

start = time.time()
try:
    path = snapshot_download(
        repo_id=MODEL_ID,
        cache_dir=CACHE_DIR,
        ignore_patterns=["*.gguf", "*.ggml"],
        max_workers=2,
    )
    elapsed = int(time.time() - start)
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {elapsed}s ({elapsed//3600}h{(elapsed%3600)//60}m) — {path}")
    sys.stdout.flush()
except Exception as e:
    elapsed = int(time.time() - start)
    print(f"\n[{time.strftime('%H:%M:%S')}] FAIL after {elapsed}s — {e}", file=sys.stderr)
    sys.exit(1)
