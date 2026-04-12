#!/bin/bash
# Test chunked long-video: WAN T2V 1.3B — 240 frames (~15s @ 16fps)
# Chunked sliding window: chunk=81, overlap=20 -> auto-routed to generate_long_video()
# Run: export SMOKE_API_KEY="your-key" && bash /tmp/test_chunked_240fr.sh
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
  local JOB_ID="$1"; local LABEL="$2"; local TIMEOUT="${3:-4800}"
  local POLL_START=$(date +%s)
  local LAST_STATUS=""
  local LAST_PROG=""
  while true; do
    local JS; JS=$(curl -s "$BASE/api/jobs/$JOB_ID" 2>/dev/null || true)
    local STATUS; STATUS=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)
    local PROG; PROG=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
    local NOW; NOW=$(date +%s); local ELAPSED=$((NOW-POLL_START))
    # Only log when status or progress changes
    if [ "$STATUS" != "$LAST_STATUS" ] || [ "$PROG" != "$LAST_PROG" ]; then
      log "  [${ELAPSED}s] $LABEL: status=$STATUS progress=${PROG}%"
      LAST_STATUS="$STATUS"
      LAST_PROG="$PROG"
    fi
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
    sleep 20
  done
}

log "=== CHUNKED GENERATION TEST: WAN T2V 1.3B — 240 frames (~15s) ==="
log "Settings: 720p, 240 frames, 50 steps, guidance=7.5"
log "Expected: auto-routed to generate_long_video() (chunks of 81, overlap=20)"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader

# Check no active jobs
ACTIVE=$(curl -s "$BASE/api/jobs" 2>/dev/null | python3 -c "import sys,json; jobs=[j for j in json.load(sys.stdin) if j.get('status') in ('processing','queued')]; print(len(jobs))" 2>/dev/null || echo 0)
if [ "$ACTIVE" -gt 0 ]; then
  fail "Active jobs in queue ($ACTIVE). Drain queue before testing."
  exit 1
fi

log "--- Submitting 240-frame job ---"
RESP=$(curl -s -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"prompt":"Cinematic aerial fly-through of dense tropical rainforest, morning mist rising through the canopy, golden rays of light, birds in flight, ultra detailed photorealistic 8k","model_name":"wan-t2v-1.3b","resolution":"720p","num_frames":240,"fps":16,"guidance_scale":7.5,"num_inference_steps":50}' \
  2>/dev/null || true)

JOB=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB" ]; then
  fail "Submit failed — $RESP"
  exit 1
fi
pass "Submitted job: $JOB"
poll_job "$JOB" "WAN T2V 1.3B 720p/240fr/50steps" 4800

log "--- Final VRAM ---"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader
log "================================================================="
log "RESULT: $PASS PASSED, $FAIL FAILED"
log "================================================================="
