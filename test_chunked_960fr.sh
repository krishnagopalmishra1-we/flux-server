#!/bin/bash
# Test chunked long-video: WAN T2V 1.3B — 960 frames (~60s @ 16fps)
# Chunked sliding window: chunk=81, overlap=20 -> auto-routed to generate_long_video()
# NOTE: ~44 min estimated inference at 50 steps. Run only after 240fr test passes.
# Run: export SMOKE_API_KEY="your-key" && bash /tmp/test_chunked_960fr.sh
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
  local JOB_ID="$1"; local LABEL="$2"; local TIMEOUT="${3:-9600}"
  local POLL_START=$(date +%s)
  local LAST_STATUS=""
  local LAST_PROG=""
  while true; do
    local JS; JS=$(curl -s "$BASE/api/jobs/$JOB_ID" 2>/dev/null || true)
    local STATUS; STATUS=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || true)
    local PROG; PROG=$(echo "$JS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('progress',0))" 2>/dev/null || true)
    local NOW; NOW=$(date +%s); local ELAPSED=$((NOW-POLL_START))
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
    sleep 30
  done
}

log "=== CHUNKED GENERATION TEST: WAN T2V 1.3B — 960 frames (~60s) ==="
log "Settings: 720p, 960 frames, 50 steps, guidance=7.5"
log "Expected: auto-routed to generate_long_video() (~44 min estimated)"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader

ACTIVE=$(curl -s "$BASE/api/jobs" 2>/dev/null | python3 -c "import sys,json; jobs=[j for j in json.load(sys.stdin) if j.get('status') in ('processing','queued')]; print(len(jobs))" 2>/dev/null || echo 0)
if [ "$ACTIVE" -gt 0 ]; then
  fail "Active jobs in queue ($ACTIVE). Drain queue before testing."
  exit 1
fi

log "--- Submitting 960-frame job ---"
RESP=$(curl -s -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"prompt":"Epic cinematic journey through ancient temple ruins, torchlight flickering on stone walls, exploring chamber after chamber, dust motes in beams of golden light, ultra detailed photorealistic 8k","model_name":"wan-t2v-1.3b","resolution":"720p","num_frames":960,"fps":16,"guidance_scale":7.5,"num_inference_steps":50}' \
  2>/dev/null || true)

JOB=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB" ]; then
  fail "Submit failed — $RESP"
  exit 1
fi
pass "Submitted job: $JOB"
poll_job "$JOB" "WAN T2V 1.3B 720p/960fr/50steps" 13200

log "--- Final VRAM ---"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader
log "================================================================="
log "RESULT: $PASS PASSED, $FAIL FAILED"
log "================================================================="
