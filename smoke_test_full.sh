#!/bin/bash
# Full smoke test + peak-quality model test for Neural Creation Studio
# Run on VM: bash /tmp/smoke_test_full.sh 2>&1 | tee /tmp/smoke_results.txt
set -euo pipefail
BASE="http://localhost:8080"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
fail() { echo "[FAIL] $*"; exit 1; }

# ── 1. GPU Health ──────────────────────────────────────────────────────────────
log "=== PHASE 1: GPU & SYSTEM HEALTH ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu,driver_version \
  --format=csv,noheader
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    d = torch.cuda.get_device_properties(0)
    print(f'GPU: {d.name}  VRAM: {d.total_memory/1024**3:.1f} GB')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN: {torch.backends.cudnn.version()}')
    print(f'FlashAttention SDP: {torch.backends.cuda.flash_sdp_enabled()}')
    print(f'TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}')
    # Quick tensor op to confirm VRAM is writable
    t = torch.randn(4096, 4096, device='cuda')
    print(f'VRAM test tensor OK: {t.shape}')
    del t; torch.cuda.empty_cache()
"

# ── 2. Server Health ───────────────────────────────────────────────────────────
log "=== PHASE 2: SERVER HEALTH ==="
# Wait up to 60s for server to respond
for i in $(seq 1 12); do
  STATUS=$(curl -sf "$BASE/health" 2>/dev/null || true)
  if [ -n "$STATUS" ]; then
    log "Health: $STATUS"
    break
  fi
  log "Waiting for server... ($((i*5))s)"
  sleep 5
done
[ -z "$STATUS" ] && fail "Server not responding after 60s"

# Check queue stats
log "Queue stats: $(curl -sf "$BASE/api/queue" 2>/dev/null)"

# ── 3. Image generation — FLUX 1-dev max quality ──────────────────────────────
log "=== PHASE 3: IMAGE — FLUX 1-dev (1024x1024, 28 steps, guidance=3.5) ==="
T0=$(date +%s%N)
IMG_RESP=$(curl -sf -X POST "$BASE/api/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A photorealistic portrait of an astronaut on Mars, golden hour lighting, ultra detailed, 8k",
    "model_name": "FLUX.1-dev",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5
  }' 2>/dev/null)
T1=$(date +%s%N)
MS=$(( (T1-T0)/1000000 ))
log "FLUX 1-dev response: $(echo "$IMG_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(f"status={d.get(\"status\",\"?\")}, width={d.get(\"width\",\"?\")}") ' 2>/dev/null || echo "$IMG_RESP")"
log "FLUX 1-dev time: ${MS}ms"

# ── 4. Video generation — WAN T2V 14B, 480p, 81 frames (max quality) ──────────
log "=== PHASE 4: VIDEO — WAN T2V 14B (480p, 81 frames, 50 steps) ==="
VID_T0=$(date +%s%N)
VID_RESP=$(curl -sf -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cinematic timelapse of storm clouds rolling over a mountain range, golden light, photorealistic",
    "model_name": "wan-t2v-14b",
    "resolution": "480p",
    "num_frames": 81,
    "fps": 16,
    "guidance_scale": 7.5,
    "num_inference_steps": 50
  }' 2>/dev/null)
JOB_ID=$(echo "$VID_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("job_id",""))' 2>/dev/null || true)
log "WAN 14B T2V submitted: job_id=$JOB_ID"
log "Response: $VID_RESP"

if [ -n "$JOB_ID" ]; then
  log "Polling job $JOB_ID..."
  POLL_START=$(date +%s)
  while true; do
    JOB_STATUS=$(curl -sf "$BASE/api/jobs/$JOB_ID" 2>/dev/null || true)
    STATUS=$(echo "$JOB_STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("status",""))' 2>/dev/null || true)
    PROGRESS=$(echo "$JOB_STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("progress",0))' 2>/dev/null || true)
    NOW=$(date +%s)
    ELAPSED=$(( NOW - POLL_START ))
    log "  [${ELAPSED}s] WAN 14B status=$STATUS progress=${PROGRESS}%"
    if [[ "$STATUS" == "completed" ]]; then
      VID_T1=$(date +%s%N)
      VID_MS=$(( (VID_T1-VID_T0)/1000000 ))
      log "WAN 14B T2V COMPLETED in ${VID_MS}ms (${ELAPSED}s wall)"
      echo "$JOB_STATUS" | python3 -c '
import sys,json
d=json.load(sys.stdin)
r=d.get("result",{})
print(f"  frames={r.get(\"num_frames\",\"?\")}, duration={r.get(\"duration_seconds\",\"?\")}s")
print(f"  inference_time={r.get(\"inference_time_ms\",\"?\")}ms")
print(f"  video_url={r.get(\"video_url\",\"?\")}")
' 2>/dev/null || true
      break
    elif [[ "$STATUS" == "failed" ]]; then
      log "WAN 14B T2V FAILED: $(echo "$JOB_STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("error_message",""))' 2>/dev/null)"
      break
    elif [[ $ELAPSED -gt 1200 ]]; then
      log "WAN 14B T2V TIMEOUT after 1200s"
      break
    fi
    sleep 10
  done
fi

# ── 5. Video generation — WAN T2V 1.3B, 720p, 81 frames ──────────────────────
log "=== PHASE 5: VIDEO — WAN T2V 1.3B (720p, 81 frames, 50 steps) ==="
VID2_RESP=$(curl -sf -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A slow motion wave crashing on a tropical beach, crystal clear water, aerial shot",
    "model_name": "wan-t2v-1.3b",
    "resolution": "720p",
    "num_frames": 81,
    "fps": 16,
    "guidance_scale": 5.0,
    "num_inference_steps": 50
  }' 2>/dev/null)
JOB2_ID=$(echo "$VID2_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("job_id",""))' 2>/dev/null || true)
log "WAN 1.3B T2V submitted: job_id=$JOB2_ID"

if [ -n "$JOB2_ID" ]; then
  POLL_START=$(date +%s)
  while true; do
    JOB_STATUS=$(curl -sf "$BASE/api/jobs/$JOB2_ID" 2>/dev/null || true)
    STATUS=$(echo "$JOB_STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("status",""))' 2>/dev/null || true)
    PROGRESS=$(echo "$JOB_STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("progress",0))' 2>/dev/null || true)
    NOW=$(date +%s); ELAPSED=$(( NOW - POLL_START ))
    log "  [${ELAPSED}s] WAN 1.3B status=$STATUS progress=${PROGRESS}%"
    if [[ "$STATUS" == "completed" ]]; then
      log "WAN 1.3B T2V COMPLETED (${ELAPSED}s)"
      echo "$JOB_STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); r=d.get("result",{}); print(f"  inference_time={r.get(\"inference_time_ms\",\"?\")}ms, frames={r.get(\"num_frames\",\"?\")}") ' 2>/dev/null || true
      break
    elif [[ "$STATUS" == "failed" ]]; then
      log "WAN 1.3B T2V FAILED"
      break
    elif [[ $ELAPSED -gt 600 ]]; then
      log "WAN 1.3B T2V TIMEOUT"
      break
    fi
    sleep 10
  done
fi

# ── 6. Music generation ────────────────────────────────────────────────────────
log "=== PHASE 6: MUSIC — ACE-Step (60s, full quality) ==="
MUS_RESP=$(curl -sf -X POST "$BASE/api/music/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Epic cinematic orchestral score with swelling strings and dramatic percussion, Hollywood blockbuster style",
    "model_name": "ace-step",
    "duration_seconds": 60
  }' 2>/dev/null)
MUS_JOB=$(echo "$MUS_RESP" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("job_id",""))' 2>/dev/null || true)
log "Music job submitted: $MUS_JOB"
log "Response: $MUS_RESP"

if [ -n "$MUS_JOB" ]; then
  POLL_START=$(date +%s)
  while true; do
    JOB_STATUS=$(curl -sf "$BASE/api/jobs/$MUS_JOB" 2>/dev/null || true)
    STATUS=$(echo "$JOB_STATUS" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("status",""))' 2>/dev/null || true)
    NOW=$(date +%s); ELAPSED=$(( NOW - POLL_START ))
    log "  [${ELAPSED}s] Music status=$STATUS"
    if [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]]; then
      log "Music result: $JOB_STATUS"
      break
    elif [[ $ELAPSED -gt 300 ]]; then
      log "Music TIMEOUT"
      break
    fi
    sleep 10
  done
fi

# ── 7. VRAM profiling after load ───────────────────────────────────────────────
log "=== PHASE 7: VRAM STATE AFTER ALL TESTS ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader

# ── 8. Performance diagnostics ────────────────────────────────────────────────
log "=== PHASE 8: PERFORMANCE DIAGNOSTICS ==="
python3 - <<'PYEOF'
import torch

print("=== Attention backends ===")
print(f"  flash_sdp:        {torch.backends.cuda.flash_sdp_enabled()}")
print(f"  mem_efficient_sdp:{torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"  math_sdp:         {torch.backends.cuda.math_sdp_enabled()}")
print(f"  TF32 matmul:      {torch.backends.cuda.matmul.allow_tf32}")
print(f"  TF32 cuDNN:       {torch.backends.cudnn.allow_tf32}")

print("\n=== Compute capability ===")
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"  SM {cap[0]}.{cap[1]} (A100 = SM 8.0, needs >= 8.0 for BF16 native)")
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  Total VRAM:     {total:.1f} GB")
    print(f"  Reserved:       {reserved:.1f} GB")
    print(f"  Allocated:      {allocated:.1f} GB")
    print(f"  Free (approx):  {total - reserved:.1f} GB")

print("\n=== BF16 matmul benchmark (proxy for model throughput) ===")
if torch.cuda.is_available():
    # Warmup
    a = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
    b = torch.randn(4096, 4096, dtype=torch.bfloat16, device='cuda')
    for _ in range(3): torch.mm(a, b)
    torch.cuda.synchronize()

    import time
    start = time.perf_counter()
    for _ in range(20): torch.mm(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    # TFLOPS = 2 * N^3 * iters / time / 1e12
    tflops = 2 * 4096**3 * 20 / elapsed / 1e12
    gpu_name = torch.cuda.get_device_name(0).upper()
    peak_map = {"A100": 312, "H100": 989, "4090": 165, "3090": 142, "L4": 242}
    peak = next((v for k, v in peak_map.items() if k in gpu_name), 0)
    print(f"  BF16 matmul throughput: {tflops:.1f} TFLOPS  (GPU: {torch.cuda.get_device_name(0)})")
    if peak > 0:
        print(f"  Utilization: {tflops/peak*100:.1f}% of {peak} TFLOPS theoretical peak")
    else:
        print(f"  Theoretical peak unknown for this GPU")
    del a, b
    torch.cuda.empty_cache()

print("\n=== torch.compile availability ===")
try:
    import torch._dynamo
    backends = torch._dynamo.list_backends()
    print(f"  Available backends: {backends}")
    print("  inductor available:", "inductor" in backends)
except Exception as e:
    print(f"  Error: {e}")
PYEOF

log "=== SMOKE TEST COMPLETE ==="
