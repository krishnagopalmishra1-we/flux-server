#!/usr/bin/env bash
# launch.sh — Create a Vultr Bare Metal GPU server for Hyperforge AI.
#
# Usage:
#   export VULTR_API_KEY="your_key"
#   export SSH_KEY_ID="your_ssh_key_id"
#   ./launch.sh
#
# Optional env vars (with defaults):
#   REGION        ewr                          (ewr=NJ, atl=Atlanta)
#   PLAN_ID       vbm-112c-2048gb-8-a100-gpu   (8x A100 SXM 80GB, NVMe)
#   LABEL         hyperforge-gpu
#   DEPLOY_BRANCH codex/hyperforge-runtime-hardening-impl

set -euo pipefail

# ── Validate requirements ──────────────────────────────────────────────────────
: "${VULTR_API_KEY:?Set VULTR_API_KEY to your Vultr API key}"
: "${SSH_KEY_ID:?Set SSH_KEY_ID to your Vultr SSH key ID}"
HF_TOKEN="${HF_TOKEN:-}"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-codex/hyperforge-runtime-hardening-impl}"

# ── Config ────────────────────────────────────────────────────────────────────
REGION="${REGION:-ewr}"
PLAN_ID="${PLAN_ID:-vbm-112c-2048gb-8-a100-gpu}"
LABEL="${LABEL:-hyperforge-gpu}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "════════════════════════════════════════════════"
echo "  Hyperforge AI — Vultr Bare Metal Deployment"
echo "  Plan:    $PLAN_ID"
echo "  Region:  $REGION"
echo "  Label:   $LABEL"
echo "  Cost:    ~\$11.92/hr preemptible (billed per second)"
echo "════════════════════════════════════════════════"
echo ""

# ── Find Ubuntu 22.04 OS ID ───────────────────────────────────────────────────
echo "Finding Ubuntu 22.04 LTS OS ID..."
OS_ID=$(curl -sf "https://api.vultr.com/v2/os" \
  -H "Authorization: Bearer $VULTR_API_KEY" \
  | python3 -c "
import sys, json
oses = json.load(sys.stdin)['os']
match = [o for o in oses if 'Ubuntu 22.04' in o.get('name','') and o.get('arch','')=='x64']
if match: print(match[0]['id'])
" 2>/dev/null)

if [[ -z "$OS_ID" ]]; then
  echo "ERROR: Ubuntu 22.04 x64 not found."
  exit 1
fi
echo "  OS ID: $OS_ID (Ubuntu 22.04 LTS x64)"

# ── Upload bootstrap script ────────────────────────────────────────────────────
echo "Uploading bootstrap script..."
STARTUP_SCRIPT="#!/usr/bin/env bash
export HF_TOKEN='$HF_TOKEN'
export DEPLOY_BRANCH='$DEPLOY_BRANCH'
$(tail -n +2 "$SCRIPT_DIR/bootstrap.sh")"

STARTUP_B64=$(echo "$STARTUP_SCRIPT" | base64 -w0)

STARTUP_ID=$(curl -sf -X POST "https://api.vultr.com/v2/startup-scripts" \
  -H "Authorization: Bearer $VULTR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"hyperforge-bootstrap\",\"type\":\"boot\",\"script\":\"$STARTUP_B64\"}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['startup_script']['id'])")
echo "  Startup script ID: $STARTUP_ID"

# ── Create bare metal instance ─────────────────────────────────────────────────
echo ""
echo "Creating bare metal instance (this takes ~10 min to provision)..."
BM_RESP=$(curl -sf -X POST "https://api.vultr.com/v2/bare-metals" \
  -H "Authorization: Bearer $VULTR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"region\": \"$REGION\",
    \"plan\": \"$PLAN_ID\",
    \"os_id\": $OS_ID,
    \"label\": \"$LABEL\",
    \"sshkey_id\": [\"$SSH_KEY_ID\"],
    \"script_id\": \"$STARTUP_ID\",
    \"enable_ipv6\": false,
    \"tags\": [\"hyperforge\", \"gpu\", \"ai\"]
  }")

BM_ID=$(echo "$BM_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['bare_metal']['id'])" 2>/dev/null)
if [[ -z "$BM_ID" ]]; then
  echo "ERROR: Failed to create bare metal instance."
  echo "$BM_RESP"
  exit 1
fi
echo "  Bare metal ID: $BM_ID"

# ── Wait for IP ───────────────────────────────────────────────────────────────
echo "Waiting for instance to get an IP (~5-10 min)..."
PUBLIC_IP=""
for i in $(seq 1 72); do
  sleep 10
  STATUS_RESP=$(curl -sf "https://api.vultr.com/v2/bare-metals/$BM_ID" \
    -H "Authorization: Bearer $VULTR_API_KEY" || echo "{}")
  STATUS=$(echo "$STATUS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('bare_metal',{}).get('status',''))" 2>/dev/null)
  IP=$(echo "$STATUS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('bare_metal',{}).get('main_ip',''))" 2>/dev/null)
  echo "  [$((i*10))s] Status: $STATUS — IP: ${IP:-pending}"
  if [[ "$STATUS" == "active" && -n "$IP" && "$IP" != "0.0.0.0" ]]; then
    PUBLIC_IP="$IP"
    break
  fi
done

if [[ -z "$PUBLIC_IP" ]]; then
  echo "Instance did not get an IP in time. Check Vultr console."
  echo "  Bare metal ID: $BM_ID"
  exit 1
fi

# ── Save connection info ───────────────────────────────────────────────────────
cat > "$SCRIPT_DIR/.instance" <<EOF
BM_ID=$BM_ID
PUBLIC_IP=$PUBLIC_IP
REGION=$REGION
PLAN_ID=$PLAN_ID
STARTUP_ID=$STARTUP_ID
EOF

# ── Print next steps ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Bare Metal ID: $BM_ID"
echo "  Public IP:     $PUBLIC_IP"
echo "  Cost:          ~\$11.92/hr (billed per second)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Bootstrap running (~8 min after SSH opens). Monitor with:"
echo "  ssh root@$PUBLIC_IP 'tail -f /var/log/hyperforge-bootstrap.log'"
echo ""
echo "After bootstrap completes:"
echo "  1. SSH in:        ssh root@$PUBLIC_IP"
echo "  2. Copy .env:     scp /path/to/.env root@$PUBLIC_IP:/opt/flux-server/flux-server/.env"
echo "  3. Start service: cd /opt/flux-server/flux-server && sudo docker compose up --build -d"
echo ""
echo "Verify:"
echo "  curl http://$PUBLIC_IP:8080/health"
echo "  Open: http://$PUBLIC_IP:8080"
echo ""
echo "Stop instance (stops billing):"
echo "  curl -sf -X POST https://api.vultr.com/v2/bare-metals/$BM_ID/halt \\"
echo "    -H 'Authorization: Bearer \$VULTR_API_KEY'"
echo ""
echo "Saved connection info: $SCRIPT_DIR/.instance"
