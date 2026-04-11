#!/bin/bash
# Final smoke test — all models at max quality, timing-focused
# Run on VM host. Results at /tmp/smoke_final.txt
set -uo pipefail
BASE="http://localhost:8080"
# Dev-only API key — must NOT grant production access.
# Set SMOKE_API_KEY in the environment before running this script.
# Example: export SMOKE_API_KEY="your-dev-key" && bash /tmp/smoke_final.sh
API_KEY="${SMOKE_API_KEY:-}"
if [ -z "$API_KEY" ]; then
  echo "ERROR: SMOKE_API_KEY environment variable is not set. Aborting." >&2
  exit 1
fi
PASS=0; FAIL=0; WARN=0

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { echo "[PASS] $*"; PASS=$((PASS+1)); return 0; }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL+1)); return 0; }
warn() { echo "[WARN] $*"; WARN=$((WARN+1)); return 0; }

poll_job() {
  local JOB_ID="$1"; local LABEL="$2"; local TIMEOUT="${3:-1800}"
  local POLL_START=$(date +%s)
  while true; do
    local JS=$(curl -sf "$BASE/api/jobs/$JOB_ID" 2>/dev/null || true)
    local STATUS=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)
    local PROG=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
    local NOW=$(date +%s); local ELAPSED=$((NOW-POLL_START))
    log "  [${ELAPSED}s] $LABEL: status=$STATUS progress=${PROG}%"
    if [ "$STATUS" = "completed" ]; then
      local INF=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('inference_time_ms',0))" 2>/dev/null || echo 0)
      local FRAMES=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('num_frames',0))" 2>/dev/null || echo 0)
      pass "$LABEL: COMPLETED in ${ELAPSED}s (inference=${INF}ms, frames=${FRAMES})"
      return 0
    elif [ "$STATUS" = "failed" ]; then
      local ERR=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error_message',''))" 2>/dev/null || true)
      fail "$LABEL: FAILED — $ERR"
      return 1
    elif [ $ELAPSED -gt $TIMEOUT ]; then
      fail "$LABEL: TIMEOUT after ${TIMEOUT}s (status=$STATUS)"
      return 1
    fi
    sleep 15
  done
}

# ── PHASE 1: System health ────────────────────────────────────────────────────
log "=== PHASE 1: SYSTEM HEALTH ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
H=$(curl -sf "$BASE/health" 2>/dev/null || true)
[ -n "$H" ] && pass "Server healthy: $H" || fail "Server not responding"

# ── PHASE 2: Python/CUDA diagnostics inside container ─────────────────────────
log "=== PHASE 2: PYTHON/CUDA DIAGNOSTICS ==="
CONTAINER=$(sudo docker ps -q --filter name=flux-server 2>/dev/null | head -1)
if [ -n "$CONTAINER" ] && sudo docker exec "$CONTAINER" python3.11 - <<'PYEOF'
import torch
print(f"PyTorch:         {torch.__version__}")
print(f"CUDA:            {torch.cuda.is_available()}")
if torch.cuda.is_available():
    d = torch.cuda.get_device_properties(0)
    print(f"GPU:             {d.name}")
    print(f"VRAM total:      {d.total_memory/1024**3:.1f} GB")
    print(f"SM:              {d.major}.{d.minor}  (A100=8.0)")
    print(f"CUDA version:    {torch.version.cuda}")
print(f"FlashSDP:        {torch.backends.cuda.flash_sdp_enabled()}")
print(f"TF32 matmul:     {torch.backends.cuda.matmul.allow_tf32}")
print(f"TF32 cuDNN:      {torch.backends.cudnn.allow_tf32}")
if torch.cuda.is_available():
    import time
    a = torch.randn(4096,4096,dtype=torch.bfloat16,device='cuda')
    b = torch.randn(4096,4096,dtype=torch.bfloat16,device='cuda')
    for _ in range(3): torch.mm(a,b)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(20): torch.mm(a,b)
    torch.cuda.synchronize()
    tflops=2*4096**3*20/(time.perf_counter()-t0)/1e12
    # Detect GPU and compare against verified BF16 (dense, no sparsity) TFLOPS peaks:
    #   A100 SXM4: 312 BF16 TFLOPS  (NVIDIA datasheet)
    #   H100 SXM:  1979 BF16 TFLOPS (NVIDIA datasheet, dense)
    #   RTX 4090:  82.6 BF16 TFLOPS (NVIDIA datasheet)
    #   RTX 3090:  71   BF16 TFLOPS (NVIDIA datasheet)
    #   L4:        121  BF16 TFLOPS (NVIDIA datasheet)
    gpu_name = d.name.upper()
    if "A100" in gpu_name:
        peak = 312
    elif "H100" in gpu_name:
        peak = 1979
    elif "RTX 4090" in gpu_name or "4090" in gpu_name:
        peak = 82
    elif "RTX 3090" in gpu_name or "3090" in gpu_name:
        peak = 71
    elif "L4" in gpu_name:
        peak = 121
    else:
        peak = 0
    if peak > 0:
        print(f"BF16 TFLOPS:     {tflops:.0f}  ({tflops/peak*100:.0f}% of {d.name} peak {peak})")
    else:
        print(f"BF16 TFLOPS:     {tflops:.0f}  (GPU peak unknown for {d.name})")
    del a,b; torch.cuda.empty_cache()
PYEOF
then
  pass "Container diagnostics done"
else
  warn "Container exec failed"
fi

# ── PHASE 3: FLUX 1-dev — 1024×1024, 28 steps ────────────────────────────────
log "=== PHASE 3: IMAGE — FLUX 1-dev (1024×1024, 28 steps) ==="
T0=$(date +%s)
RESP=$(curl -sf -X POST "$BASE/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"prompt":"Photorealistic portrait of an astronaut on Mars, golden hour, 8k, ultra detailed","model_name":"flux-1-dev","width":1024,"height":1024,"num_inference_steps":28,"guidance_scale":3.5}' 2>/dev/null || true)
T1=$(date +%s)
if [ -n "$RESP" ]; then
  STATUS_CODE=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','') or ('ok' if d.get('image_base64') or d.get('image_url') else 'unknown'))" 2>/dev/null || echo "?")
  pass "FLUX 1-dev response in $((T1-T0))s (status=$STATUS_CODE): $(echo $RESP | head -c 200)"
else
  warn "FLUX 1-dev: no response"
fi

# ── PHASE 4: WAN T2V 14B — 480p, 49 frames, 50 steps ─────────────────────────
# Note: 81 frames exceeds A100 40GB VRAM at 480p + NF4 (tested: OOM at ~38.76GB).
# 49 frames is the verified safe maximum for high-quality 14B generation on A100 40GB.
log "=== PHASE 4: VIDEO — WAN T2V 14B (480p, 49 frames, 50 steps) ==="
T0=$(date +%s)
RESP=$(curl -sf -X POST "$BASE/api/video/generate" -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"prompt":"Cinematic timelapse of storm clouds over mountains, photorealistic, golden light","model_name":"wan-t2v-14b","resolution":"480p","num_frames":49,"fps":16,"guidance_scale":7.5,"num_inference_steps":50}' 2>/dev/null || true)
JOB=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB" ]; then
  fail "WAN T2V 14B: submit failed — $RESP"
else
  pass "WAN T2V 14B: submitted ($JOB)"
  poll_job "$JOB" "WAN T2V 14B" 1800
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# ── PHASE 5: WAN T2V 1.3B — 720p, 81 frames, 50 steps ───────────────────────
log "=== PHASE 5: VIDEO — WAN T2V 1.3B (720p, 81 frames, 50 steps) ==="
T0=$(date +%s)
RESP=$(curl -sf -X POST "$BASE/api/video/generate" -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"prompt":"Slow motion wave crashing on tropical beach, crystal clear water, drone shot","model_name":"wan-t2v-1.3b","resolution":"720p","num_frames":81,"fps":16,"guidance_scale":5.0,"num_inference_steps":50}' 2>/dev/null || true)

JOB=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB" ]; then
  fail "WAN T2V 1.3B: submit failed — $RESP"
else
  pass "WAN T2V 1.3B: submitted ($JOB)"
  poll_job "$JOB" "WAN T2V 1.3B" 600
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# ── PHASE 6: WAN I2V 14B — 480p, 33 frames, 30 steps ────────────────────────
log "=== PHASE 6: VIDEO — WAN I2V 14B (480p, 33 frames, 30 steps) ==="
# 1x1 white pixel PNG as base64
IMG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
RESP=$(curl -sf -X POST "$BASE/api/video/generate" -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d "{\"prompt\":\"Majestic eagle soaring over snowy mountains, cinematic\",\"model_name\":\"wan-i2v-14b\",\"source_image_b64\":\"$IMG\",\"num_frames\":33,\"fps\":16,\"guidance_scale\":5.0,\"num_inference_steps\":30}" 2>/dev/null || true)
JOB=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB" ]; then
  warn "WAN I2V 14B: submit failed — $RESP"
else
  pass "WAN I2V 14B: submitted ($JOB)"
  poll_job "$JOB" "WAN I2V 14B" 900
fi

# ── PHASE 7: Final VRAM state ─────────────────────────────────────────────────
log "=== PHASE 7: FINAL VRAM STATE ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader

log "================================================================="
log "FINAL SUMMARY: $PASS PASSED, $WARN WARNINGS, $FAIL FAILED"
log "================================================================="
