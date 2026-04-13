#!/bin/bash
# HQ quality test: WAN T2V 14B — 5s (81fr) + 15s (240fr chunked)
# People-focused prompts, bright vivid ultra-photorealistic output
# Run: export SMOKE_API_KEY="your-key" && bash /tmp/test_hq_wan14b.sh
set -uo pipefail
BASE="http://localhost:8080"
API_KEY="${SMOKE_API_KEY:-}"
[ -z "$API_KEY" ] && { echo "ERROR: SMOKE_API_KEY not set." >&2; exit 1; }
PASS=0; FAIL=0

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { echo "[PASS] $*"; PASS=$((PASS+1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }

poll_job() {
  local JOB_ID="$1" LABEL="$2" TIMEOUT="${3:-5000}"
  local START=$(date +%s) LAST_S="" LAST_P=""
  while true; do
    local JS; JS=$(curl -s "$BASE/api/jobs/$JOB_ID" 2>/dev/null || true)
    local S; S=$(echo "$JS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || true)
    local P; P=$(echo "$JS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('progress',0))" 2>/dev/null || true)
    local E=$(( $(date +%s) - START ))
    [ "$S" != "$LAST_S" ] || [ "$P" != "$LAST_P" ] && { log "  [${E}s] $LABEL: $S ${P}%"; LAST_S="$S"; LAST_P="$P"; }
    if [ "$S" = "completed" ]; then
      local VID; VID=$(echo "$JS" | python3 -c "import sys,json; r=json.load(sys.stdin).get('result',{}); print(r.get('video_url','') or r.get('video_path',''))" 2>/dev/null || true)
      pass "$LABEL: DONE in ${E}s | output=$VID"
      return 0
    elif [ "$S" = "failed" ]; then
      local ERR; ERR=$(echo "$JS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error_message',''))" 2>/dev/null || true)
      fail "$LABEL: FAILED — $ERR"; return 0
    elif [ $E -gt $TIMEOUT ]; then
      fail "$LABEL: TIMEOUT after ${TIMEOUT}s"; return 0
    fi
    sleep 30
  done
}

check_queue() {
  local ACTIVE; ACTIVE=$(curl -s "$BASE/api/jobs" 2>/dev/null | python3 -c "import sys,json; print(len([j for j in json.load(sys.stdin) if j.get('status') in ('processing','queued')]))" 2>/dev/null || echo 0)
  [ "$ACTIVE" -gt 0 ] && { fail "Active jobs in queue ($ACTIVE). Drain first."; exit 1; }
}

log "=== HQ TEST: WAN T2V 14B ==="
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
check_queue

NEG="blurry, dark, low quality, dull, desaturated, empty, no people, cartoon, distorted, bad anatomy"

# ── TEST 1: 5 sec (81 frames @ 16fps) ─────────────────────────────────────────
log "--- TEST 1: WAN T2V 14B 720p / 81fr / 50 steps (5 sec) ---"
RESP=$(curl -s -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d "{\"prompt\":\"A confident young woman in a vibrant red dress walks through a sunlit city street, crowds of people strolling past shops, ultra-photorealistic 8K HDR, sharp facial details, warm golden hour light, vivid saturated colors, cinematic depth of field, professional cinematography\",\"negative_prompt\":\"$NEG\",\"model_name\":\"wan-t2v-14b\",\"resolution\":\"720p\",\"num_frames\":81,\"fps\":16,\"guidance_scale\":7.5,\"num_inference_steps\":50}" 2>/dev/null || true)
JOB=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null || true)
[ -z "$JOB" ] && { fail "TEST 1 submit failed — $RESP"; } || { pass "TEST 1 submitted: $JOB"; poll_job "$JOB" "WAN T2V 14B 720p/81fr/50s" 3600; }
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
sleep 10

# ── TEST 2: 15 sec (240 frames @ 16fps, chunked) ──────────────────────────────
log "--- TEST 2: WAN T2V 14B 720p / 240fr / 50 steps (15 sec, chunked) ---"
check_queue
RESP=$(curl -s -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d "{\"prompt\":\"Busy Tokyo Shibuya crossing at golden hour, hundreds of people crossing in all directions, neon signs glowing, ultra-photorealistic 8K HDR, sharp detailed faces and clothing, vivid neon colors, cinematic wide shot, dynamic crowd motion, professional film quality\",\"negative_prompt\":\"$NEG\",\"model_name\":\"wan-t2v-14b\",\"resolution\":\"720p\",\"num_frames\":240,\"fps\":16,\"guidance_scale\":7.5,\"num_inference_steps\":50}" 2>/dev/null || true)
JOB=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null || true)
[ -z "$JOB" ] && { fail "TEST 2 submit failed — $RESP"; } || { pass "TEST 2 submitted: $JOB"; poll_job "$JOB" "WAN T2V 14B 720p/240fr/50s chunked" 6000; }
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

log "================================================================="
log "RESULT: $PASS PASSED, $FAIL FAILED"
log "================================================================="
