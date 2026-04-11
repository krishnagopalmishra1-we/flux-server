#!/bin/bash
# WAN T2V 1.3B — max quality test
# Resolution: 720p (1280x720), 121 frames (~7.5s @ 16fps), 50 steps, guidance 7.5
# Run: export SMOKE_API_KEY="your-key" && bash /tmp/test_wan_1.3b_quality.sh
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
  local JOB_ID="$1"; local LABEL="$2"; local TIMEOUT="${3:-2400}"
  local POLL_START=$(date +%s)
  while true; do
    local JS; JS=$(curl -s "$BASE/api/jobs/$JOB_ID" 2>/dev/null || true)
    local STATUS; STATUS=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)
    local PROG; PROG=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
    local NOW; NOW=$(date +%s); local ELAPSED=$((NOW-POLL_START))
    log "  [${ELAPSED}s] $LABEL: status=$STATUS progress=${PROG}%"
    if [ "$STATUS" = "completed" ]; then
      local INF; INF=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('inference_time_ms',0))" 2>/dev/null || echo 0)
      local FRAMES; FRAMES=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('num_frames',0))" 2>/dev/null || echo 0)
      local VID; VID=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('result',{}); print(r.get('video_url','') or r.get('video_path',''))" 2>/dev/null || echo "")
      pass "$LABEL: COMPLETED in ${ELAPSED}s | inference=${INF}ms | frames=${FRAMES} | output=$VID"
      return 0
    elif [ "$STATUS" = "failed" ]; then
      local ERR; ERR=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('error_message',''))" 2>/dev/null || true)
      fail "$LABEL: FAILED — $ERR"
      return 0
    elif [ $ELAPSED -gt $TIMEOUT ]; then
      fail "$LABEL: TIMEOUT after ${TIMEOUT}s (status=$STATUS)"
      return 0
    fi
    sleep 15
  done
}

log "=== WAN T2V 1.3B — MAX QUALITY TEST ==="
log "Settings: 720p, 81 frames (API max), 50 steps, guidance=7.5"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader

log "--- Submitting job ---"
RESP=$(curl -sSf -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"prompt":"Cinematic aerial shot of a dense tropical rainforest at sunrise, morning mist rising through the canopy, golden rays of light piercing through the trees, birds in flight, ultra detailed, photorealistic, 8k, masterpiece","model_name":"wan-t2v-1.3b","resolution":"720p","num_frames":81,"fps":16,"guidance_scale":7.5,"num_inference_steps":50}' \
  2>/dev/null || true)

JOB=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB" ]; then
  fail "Submit failed — $RESP"
else
  pass "Submitted job: $JOB"
  poll_job "$JOB" "WAN T2V 1.3B 720p/81fr/50steps" 2400
fi

log "--- Final VRAM ---"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader
log "================================================================="
log "RESULT: $PASS PASSED, $FAIL FAILED"
log "================================================================="
