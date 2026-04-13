#!/bin/bash
# HQ quality test: HunyuanVideo — 5s (129fr @ 24fps) + 15s (240fr @ 24fps)
# People-focused prompts, bright vivid ultra-photorealistic output
# Run: export SMOKE_API_KEY="your-key" && bash /tmp/test_hq_hunyuan.sh
set -uo pipefail
BASE="http://localhost:8080"
API_KEY="${SMOKE_API_KEY:-}"
[ -z "$API_KEY" ] && { echo "ERROR: SMOKE_API_KEY not set." >&2; exit 1; }
PASS=0; FAIL=0

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
pass() { echo "[PASS] $*"; PASS=$((PASS+1)); }
fail() { echo "[FAIL] $*"; FAIL=$((FAIL+1)); }

poll_job() {
  local JOB_ID="$1" LABEL="$2" TIMEOUT="${3:-4800}"
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
    sleep 20
  done
}

check_queue() {
  local ACTIVE; ACTIVE=$(curl -s "$BASE/api/jobs" 2>/dev/null | python3 -c "import sys,json; print(len([j for j in json.load(sys.stdin) if j.get('status') in ('processing','queued')]))" 2>/dev/null || echo 0)
  [ "$ACTIVE" -gt 0 ] && { fail "Active jobs in queue ($ACTIVE). Drain first."; exit 1; }
}

log "=== HQ TEST: HunyuanVideo ==="
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader
check_queue

NEG="blurry, dark, low quality, dull, desaturated, no people, cartoon, distorted, bad anatomy, empty scene"

# ── TEST 1: 5 sec (129 frames @ 24fps) ────────────────────────────────────────
log "--- TEST 1: HunyuanVideo 720p / 129fr / 50 steps (5 sec) ---"
RESP=$(curl -s -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d "{\"prompt\":\"A professional male athlete in bright athletic wear sprinting through a sunlit stadium track, ultra-photorealistic 8K HDR, sharp motion, bright stadium lights, vivid colors, intense determination on face, dramatic low angle shot, crystal clear detail, cinematic\",\"negative_prompt\":\"$NEG\",\"model_name\":\"hunyuan-video\",\"resolution\":\"720p\",\"num_frames\":129,\"fps\":24,\"guidance_scale\":6.0,\"num_inference_steps\":50}" 2>/dev/null || true)
JOB=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null || true)
[ -z "$JOB" ] && { fail "TEST 1 submit failed — $RESP"; } || { pass "TEST 1 submitted: $JOB"; poll_job "$JOB" "HunyuanVideo 720p/129fr/50s" 2400; }
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
sleep 10

# ── TEST 2: 15 sec (240 frames @ 24fps) ───────────────────────────────────────
log "--- TEST 2: HunyuanVideo 720p / 240fr / 50 steps (~10 sec) ---"
check_queue
RESP=$(curl -s -X POST "$BASE/api/video/generate" \
  -H "Content-Type: application/json" -H "X-API-Key: $API_KEY" \
  -d "{\"prompt\":\"Vibrant street market in Southeast Asia, vendors in colorful traditional clothing calling to shoppers, exotic fruits and spices displayed, bright tropical afternoon sunlight, ultra-photorealistic 8K HDR, sharp facial expressions, vivid saturated colors, lively crowd, cinematic documentary\",\"negative_prompt\":\"$NEG\",\"model_name\":\"hunyuan-video\",\"resolution\":\"720p\",\"num_frames\":240,\"fps\":24,\"guidance_scale\":6.0,\"num_inference_steps\":50}" 2>/dev/null || true)
JOB=$(echo "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null || true)
[ -z "$JOB" ] && { fail "TEST 2 submit failed — $RESP"; } || { pass "TEST 2 submitted: $JOB"; poll_job "$JOB" "HunyuanVideo 720p/240fr/50s" 4800; }
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

log "================================================================="
log "RESULT: $PASS PASSED, $FAIL FAILED"
log "================================================================="
