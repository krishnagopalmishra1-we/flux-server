#!/bin/bash
# Isolated re-test: WAN T2V 14B + WAN I2V 14B after OOM fix (commit 91736f2)
# Verifies flux_pipeline.model_manager.unload_all() correctly frees VRAM before 14B loads.
# Run: export SMOKE_API_KEY="your-key" && bash /tmp/test_wan_14b_isolated.sh
set -uo pipefail
BASE="http://localhost:8080"
API_KEY="${SMOKE_API_KEY:-}"
if [ -z "$API_KEY" ]; then
  echo "ERROR: SMOKE_API_KEY not set." >&2; exit 1
fi
PASS=0; FAIL=0

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { echo "[PASS] $*"; PASS=$((PASS+1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }

poll_job() {
  local JOB_ID="$1"; local LABEL="$2"; local TIMEOUT="${3:-4500}"
  local POLL_START=$(date +%s)
  local LAST_STATUS="" LAST_PROG=""
  while true; do
    local JS; JS=$(curl -s "$BASE/api/jobs/$JOB_ID" 2>/dev/null || true)
    local STATUS; STATUS=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)
    local PROG; PROG=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
    local NOW; NOW=$(date +%s); local ELAPSED=$((NOW-POLL_START))
    if [ "$STATUS" != "$LAST_STATUS" ] || [ "$PROG" != "$LAST_PROG" ]; then
      log "  [${ELAPSED}s] $LABEL: status=$STATUS progress=${PROG}%"
      LAST_STATUS="$STATUS"; LAST_PROG="$PROG"
    fi
    if [ "$STATUS" = "completed" ]; then
      local INF; INF=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('inference_time_ms',0))" 2>/dev/null || echo 0)
      local FRAMES; FRAMES=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('num_frames',0))" 2>/dev/null || echo 0)
      pass "$LABEL: COMPLETED in ${ELAPSED}s | inference=${INF}ms | frames=${FRAMES}"
      return 0
    elif [ "$STATUS" = "failed" ]; then
      local ERR; ERR=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error_message',''))" 2>/dev/null || true)
      fail "$LABEL: FAILED — $ERR"
      return 0
    elif [ $ELAPSED -gt $TIMEOUT ]; then
      fail "$LABEL: TIMEOUT after ${TIMEOUT}s (status=$STATUS)"
      return 0
    fi
    sleep 20
  done
}

log "=== ISOLATED 14B RETEST — OOM fix validation ==="
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader

# Abort if any jobs active
ACTIVE=$(curl -s "$BASE/api/jobs" 2>/dev/null | python3 -c "import sys,json; r=json.load(sys.stdin); jobs=r.get('jobs',r) if isinstance(r,dict) else r; print(len([j for j in jobs if j.get('status') in ('processing','queued')]))" 2>/dev/null || echo 0)
if [ "$ACTIVE" -gt 0 ]; then
  fail "Active jobs in queue ($ACTIVE). Drain before testing."; exit 1
fi

# ── TEST 1: FLUX image (confirm image gen still works after redeploy) ──────────
log "=== TEST 1: FLUX 1-dev image (warm-up + sanity check) ==="
RESP=$(curl -s -X POST "$BASE/generate" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"prompt":"A red apple on a wooden table","model_name":"flux-1-dev","width":512,"height":512,"num_inference_steps":20,"guidance_scale":3.5}' 2>/dev/null || true)
STATUS_CODE=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','') or ('ok' if d.get('image_base64') else 'fail'))" 2>/dev/null || echo "?")
if [ "$STATUS_CODE" = "completed" ] || [ "$STATUS_CODE" = "ok" ]; then
  pass "FLUX 1-dev: response ok (status=$STATUS_CODE)"
else
  fail "FLUX 1-dev: unexpected response (status=$STATUS_CODE) — $RESP"
fi

# ── TEST 2: WAN T2V 14B 480p/49fr (verifies OOM fix) ─────────────────────────
log "=== TEST 2: WAN T2V 14B 480p/49fr/50 steps ==="
log "Waiting 10s for VRAM to settle after FLUX..."
sleep 10
RESP=$(curl -s -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d '{"prompt":"Cinematic timelapse of storm clouds over mountains, photorealistic","model_name":"wan-t2v-14b","resolution":"480p","num_frames":49,"fps":16,"guidance_scale":7.5,"num_inference_steps":50}' 2>/dev/null || true)
JOB=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB" ]; then
  fail "WAN T2V 14B: submit failed — $RESP"
else
  pass "WAN T2V 14B: submitted ($JOB)"
  poll_job "$JOB" "WAN T2V 14B" 4500
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# ── TEST 3: WAN I2V 14B 480p/33fr (verifies OOM fix for I2V path too) ────────
log "=== TEST 3: WAN I2V 14B 480p/33fr/30 steps ==="
IMG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
RESP=$(curl -s -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d "{\"prompt\":\"Majestic eagle soaring over snowy mountains, cinematic\",\"model_name\":\"wan-i2v-14b\",\"source_image_b64\":\"$IMG\",\"num_frames\":33,\"fps\":16,\"guidance_scale\":5.0,\"num_inference_steps\":30}" 2>/dev/null || true)
JOB=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB" ]; then
  fail "WAN I2V 14B: submit failed — $RESP"
else
  pass "WAN I2V 14B: submitted ($JOB)"
  poll_job "$JOB" "WAN I2V 14B" 4500
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

log "================================================================="
log "RESULT: $PASS PASSED, $FAIL FAILED"
log "================================================================="
