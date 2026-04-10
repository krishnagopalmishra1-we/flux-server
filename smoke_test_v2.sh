#!/bin/bash
# Full smoke test — curl-based, runs on VM host against the Docker container
# Results: /tmp/smoke_results.txt
set -uo pipefail
BASE="http://localhost:8080"
PASS=0; FAIL=0; WARN=0

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { echo "[PASS] $*"; ((PASS++)); }
fail() { echo "[FAIL] $*"; ((FAIL++)); }
warn() { echo "[WARN] $*"; ((WARN++)); }

# ── PHASE 1: GPU & System ─────────────────────────────────────────────────────
log "=== PHASE 1: GPU & SYSTEM HEALTH ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu,driver_version \
  --format=csv,noheader,nounits | while IFS=',' read name total free util drv; do
  echo "  GPU:     $name"
  echo "  VRAM:    ${free}/${total} MiB free"
  echo "  Util:    ${util}%"
  echo "  Driver:  $drv"
done

# ── PHASE 2: Server Health ────────────────────────────────────────────────────
log "=== PHASE 2: SERVER HEALTH ==="
HEALTH=$(curl -sf "$BASE/health" 2>/dev/null)
if [ -n "$HEALTH" ]; then
  pass "Server health endpoint responds"
  echo "  $HEALTH"
  echo "$HEALTH" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'  GPU: {d.get(\"gpu_name\",\"?\")}')
print(f'  VRAM used: {d.get(\"vram_used_gb\",0):.2f} / {d.get(\"vram_total_gb\",0):.1f} GB')
print(f'  Model loaded: {d.get(\"model_loaded\",False)}')
" 2>/dev/null || true
else
  fail "Server not responding"
fi

QSTATS=$(curl -sf "$BASE/api/queue" 2>/dev/null || true)
[ -n "$QSTATS" ] && pass "Queue stats: $QSTATS" || warn "Queue stats endpoint not available"

# ── PHASE 3: Python perf diagnostics (inside Docker) ─────────────────────────
log "=== PHASE 3: PYTHON/CUDA DIAGNOSTICS (inside container) ==="
CONTAINER=$(sudo docker ps -q --filter name=flux-server 2>/dev/null | head -1)
if [ -n "$CONTAINER" ]; then
  sudo docker exec "$CONTAINER" python3.11 - <<'PYEOF'
import torch, sys

print("=== PyTorch / CUDA ===")
print(f"  torch:              {torch.__version__}")
print(f"  CUDA available:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    d = torch.cuda.get_device_properties(0)
    print(f"  GPU:                {d.name}")
    print(f"  VRAM total:         {d.total_memory/1024**3:.2f} GB")
    print(f"  SM:                 {d.major}.{d.minor}")
    print(f"  CUDA version:       {torch.version.cuda}")
    print(f"  cuDNN:              {torch.backends.cudnn.version()}")

print("\n=== Attention & Math flags ===")
print(f"  flash_sdp:          {torch.backends.cuda.flash_sdp_enabled()}")
print(f"  mem_efficient_sdp:  {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"  TF32 matmul:        {torch.backends.cuda.matmul.allow_tf32}")
print(f"  TF32 cuDNN:         {torch.backends.cudnn.allow_tf32}")

print("\n=== BF16 matmul benchmark (Tensor Core utilization) ===")
if torch.cuda.is_available():
    a = torch.randn(4096,4096,dtype=torch.bfloat16,device='cuda')
    b = torch.randn(4096,4096,dtype=torch.bfloat16,device='cuda')
    for _ in range(3): torch.mm(a,b)
    torch.cuda.synchronize()
    import time
    t0 = time.perf_counter()
    for _ in range(20): torch.mm(a,b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-t0
    tflops = 2*4096**3*20/elapsed/1e12
    gpu_name = torch.cuda.get_device_name(0).upper()
    peak_map = {"A100": 312, "H100": 989, "4090": 165, "3090": 142, "L4": 242}
    peak = next((v for k, v in peak_map.items() if k in gpu_name), 0)
    print(f"  TFLOPS:             {tflops:.1f}  (GPU: {torch.cuda.get_device_name(0)})")
    if peak > 0:
        print(f"  Tensor Core util:   {tflops/peak*100:.1f}% of {peak} TFLOPS peak")
    del a,b; torch.cuda.empty_cache()

print("\n=== torch.compile backends ===")
try:
    import torch._dynamo as dyn
    backends = dyn.list_backends()
    print(f"  Available:          {backends}")
    print(f"  inductor:           {'inductor' in backends}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Key packages ===")
for pkg in ['diffusers','transformers','bitsandbytes','peft','accelerate','sse_starlette']:
    try:
        m = __import__(pkg)
        print(f"  {pkg}: {getattr(m,'__version__','OK')}")
    except ImportError:
        print(f"  {pkg}: MISSING")
PYEOF
  if [ $? -eq 0 ]; then
    pass "Container diagnostics complete"
  else
    warn "Container diagnostics failed"
  fi
else
  warn "Could not find running container for diagnostics"
fi

# ── PHASE 4: FLUX 1-dev — 1024x1024, 28 steps ────────────────────────────────
log "=== PHASE 4: IMAGE — FLUX 1-dev (1024x1024, 28 steps, guidance=3.5) ==="
T0=$(date +%s)
RESP=$(curl -sf -X POST "$BASE/api/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A photorealistic portrait of an astronaut on Mars, golden hour lighting, 8k","model_name":"FLUX.1-dev","width":1024,"height":1024,"num_inference_steps":28,"guidance_scale":3.5}' 2>/dev/null || true)
T1=$(date +%s)
if [ -n "$RESP" ]; then
  STATUS=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || echo "?")
  ELAPSED=$((T1-T0))
  if echo "$STATUS" | grep -qE "success|ok|completed|image_url|base64"; then
    pass "FLUX 1-dev generated in ${ELAPSED}s"
  else
    warn "FLUX 1-dev response: $STATUS (${ELAPSED}s) — $RESP"
  fi
  echo "  Response: $(echo "$RESP" | head -c 300)"
else
  warn "FLUX 1-dev: no response (model may not be loaded yet)"
fi

# ── PHASE 5: WAN T2V 14B — 480p, 81 frames, 50 steps ─────────────────────────
log "=== PHASE 5: VIDEO — WAN T2V 14B (480p, 81 frames, 50 steps) ==="
T0=$(date +%s)
RESP=$(curl -sf -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Cinematic timelapse of storm clouds over mountains, golden light, photorealistic","model_name":"wan-t2v-14b","resolution":"480p","num_frames":81,"fps":16,"guidance_scale":7.5,"num_inference_steps":50}' 2>/dev/null || true)
JOB_ID=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB_ID" ]; then
  fail "WAN T2V 14B: failed to submit job. Response: $RESP"
else
  pass "WAN T2V 14B: job submitted ($JOB_ID)"
  log "  Polling (timeout: 900s)..."
  POLL_START=$(date +%s)
  while true; do
    JS=$(curl -sf "$BASE/api/jobs/$JOB_ID" 2>/dev/null || true)
    STATUS=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)
    PROG=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
    NOW=$(date +%s); ELAPSED=$((NOW-POLL_START))
    log "  [${ELAPSED}s] status=$STATUS progress=${PROG}%"
    if [ "$STATUS" = "completed" ]; then
      T1=$(date +%s); WALL=$((T1-T0))
      INF=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('inference_time_ms',0))" 2>/dev/null || echo 0)
      FRAMES=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('num_frames',0))" 2>/dev/null || echo 0)
      pass "WAN T2V 14B: COMPLETED in ${WALL}s (inference=${INF}ms, frames=$FRAMES)"
      echo "  Full result: $(echo "$JS" | python3 -c 'import sys,json; d=json.load(sys.stdin); r=d.get("result",{}); print(r)' 2>/dev/null || echo "$JS")"
      break
    elif [ "$STATUS" = "failed" ]; then
      ERR=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error_message',''))" 2>/dev/null || true)
      fail "WAN T2V 14B: FAILED — $ERR"
      break
    elif [ $ELAPSED -gt 900 ]; then
      fail "WAN T2V 14B: TIMEOUT after 900s (status=$STATUS)"
      break
    fi
    sleep 15
  done
fi

# ── PHASE 6: WAN T2V 1.3B — 720p, 81 frames, 50 steps ───────────────────────
log "=== PHASE 6: VIDEO — WAN T2V 1.3B (720p, 81 frames, 50 steps) ==="
T0=$(date +%s)
RESP=$(curl -sf -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Slow motion wave crashing on tropical beach, crystal clear water, aerial shot","model_name":"wan-t2v-1.3b","resolution":"720p","num_frames":81,"fps":16,"guidance_scale":5.0,"num_inference_steps":50}' 2>/dev/null || true)
JOB_ID2=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB_ID2" ]; then
  fail "WAN T2V 1.3B: failed to submit. Response: $RESP"
else
  pass "WAN T2V 1.3B: job submitted ($JOB_ID2)"
  POLL_START=$(date +%s)
  while true; do
    JS=$(curl -sf "$BASE/api/jobs/$JOB_ID2" 2>/dev/null || true)
    STATUS=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)
    PROG=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
    NOW=$(date +%s); ELAPSED=$((NOW-POLL_START))
    log "  [${ELAPSED}s] status=$STATUS progress=${PROG}%"
    if [ "$STATUS" = "completed" ]; then
      T1=$(date +%s)
      INF=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('inference_time_ms',0))" 2>/dev/null || echo 0)
      pass "WAN T2V 1.3B: COMPLETED in $((T1-T0))s (inference=${INF}ms)"
      break
    elif [ "$STATUS" = "failed" ]; then
      ERR=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error_message',''))" 2>/dev/null || true)
      fail "WAN T2V 1.3B: FAILED — $ERR"
      break
    elif [ $ELAPSED -gt 600 ]; then
      fail "WAN T2V 1.3B: TIMEOUT"
      break
    fi
    sleep 10
  done
fi

# ── PHASE 7: WAN I2V 14B — 480p, 33 frames ───────────────────────────────────
log "=== PHASE 7: VIDEO — WAN I2V 14B (480p, 33 frames, 30 steps) ==="
# 1x1 black pixel PNG encoded as base64 — minimal valid image for I2V model input
TEST_IMG_B64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
T0=$(date +%s)
RESP=$(curl -sf -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"A majestic eagle soaring over mountain peaks\",\"model_name\":\"wan-i2v-14b\",\"source_image_b64\":\"$TEST_IMG_B64\",\"num_frames\":33,\"fps\":16,\"guidance_scale\":5.0,\"num_inference_steps\":30}" 2>/dev/null || true)
JOB_ID3=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB_ID3" ]; then
  warn "WAN I2V 14B: job submit failed — $RESP"
else
  pass "WAN I2V 14B: job submitted ($JOB_ID3)"
  POLL_START=$(date +%s)
  while true; do
    JS=$(curl -sf "$BASE/api/jobs/$JOB_ID3" 2>/dev/null || true)
    STATUS=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)
    PROG=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
    NOW=$(date +%s); ELAPSED=$((NOW-POLL_START))
    log "  [${ELAPSED}s] I2V status=$STATUS progress=${PROG}%"
    if [ "$STATUS" = "completed" ]; then
      T1=$(date +%s)
      INF=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('inference_time_ms',0))" 2>/dev/null || echo 0)
      pass "WAN I2V 14B: COMPLETED in $((T1-T0))s (inference=${INF}ms)"
      break
    elif [ "$STATUS" = "failed" ]; then
      ERR=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error_message',''))" 2>/dev/null || true)
      warn "WAN I2V 14B: FAILED — $ERR"
      break
    elif [ $ELAPSED -gt 600 ]; then
      warn "WAN I2V 14B: TIMEOUT"
      break
    fi
    sleep 10
  done
fi

# ── PHASE 8: Music ────────────────────────────────────────────────────────────
log "=== PHASE 8: MUSIC — ACE-Step (60s) ==="
T0=$(date +%s)
RESP=$(curl -sf -X POST "$BASE/api/music/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Epic cinematic orchestral score, Hollywood blockbuster, swelling strings, dramatic percussion","model_name":"ace-step","duration_seconds":60}' 2>/dev/null || true)
JOB_ID4=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB_ID4" ]; then
  warn "Music: failed to submit — $RESP"
else
  pass "Music: job submitted ($JOB_ID4)"
  POLL_START=$(date +%s)
  while true; do
    JS=$(curl -sf "$BASE/api/jobs/$JOB_ID4" 2>/dev/null || true)
    STATUS=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)
    NOW=$(date +%s); ELAPSED=$((NOW-POLL_START))
    log "  [${ELAPSED}s] Music status=$STATUS"
    if [ "$STATUS" = "completed" ]; then
      T1=$(date +%s)
      pass "Music: COMPLETED in $((T1-T0))s"
      echo "  $JS" | head -c 200
      break
    elif [ "$STATUS" = "failed" ]; then
      ERR=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error_message',''))" 2>/dev/null || true)
      warn "Music: FAILED — $ERR"
      break
    elif [ $ELAPSED -gt 300 ]; then
      warn "Music: TIMEOUT"
      break
    fi
    sleep 10
  done
fi

# ── PHASE 9: Post-test VRAM ───────────────────────────────────────────────────
log "=== PHASE 9: VRAM AFTER ALL TESTS ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader
curl -sf "$BASE/health" 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'  VRAM used: {d.get(\"vram_used_gb\",0):.2f} / {d.get(\"vram_total_gb\",0):.1f} GB')
" 2>/dev/null || true

# ── Summary ───────────────────────────────────────────────────────────────────
log "==================================================================="
log "SMOKE TEST SUMMARY: $PASS PASSED, $WARN WARNINGS, $FAIL FAILED"
log "==================================================================="
[ $FAIL -eq 0 ] && log "ALL CRITICAL TESTS PASSED" || log "FAILURES DETECTED — see above"
