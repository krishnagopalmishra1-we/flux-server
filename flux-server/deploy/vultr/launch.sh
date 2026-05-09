#!/usr/bin/env bash
# launch.sh — Create a Vultr Cloud GPU instance for Hyperforge AI.
#
# Prerequisites:
#   1. vultr-cli installed  (https://github.com/vultr/vultr-cli#installation)
#   2. VULTR_API_KEY exported
#   3. SSH key uploaded to Vultr and its ID noted
#
# Usage:
#   export VULTR_API_KEY="your_key"
#   export SSH_KEY_ID="your_ssh_key_id"
#   export HF_TOKEN="hf_your_token"
#   ./launch.sh
#
# Optional env vars (with defaults):
#   REGION            ewr          (ewr=NJ, lax=LA, sjc=SJ, ord=Chicago, fra=Frankfurt)
#   INSTANCE_LABEL    hyperforge-gpu
#   BLOCK_STORAGE_GB  500
#   PLAN_ID           (auto-selected from menu if not set)

set -euo pipefail

# ── Validate requirements ──────────────────────────────────────────────────────
: "${VULTR_API_KEY:?Set VULTR_API_KEY to your Vultr API key}"
: "${SSH_KEY_ID:?Set SSH_KEY_ID to your Vultr SSH key ID (vultr-cli ssh-key list)}"
: "${HF_TOKEN:?Set HF_TOKEN to your HuggingFace token (needed for model downloads)}"

if ! command -v vultr-cli &>/dev/null; then
  echo "ERROR: vultr-cli not found."
  echo ""
  echo "Install it:"
  echo "  Linux/WSL:"
  echo "    VER=\$(curl -s https://api.github.com/repos/vultr/vultr-cli/releases/latest | grep tag_name | cut -d'\"' -f4)"
  echo "    curl -fsSL https://github.com/vultr/vultr-cli/releases/download/\${VER}/vultr-cli_\${VER#v}_linux_amd64.tar.gz | tar -xz -C /usr/local/bin vultr-cli"
  echo "  macOS: brew install vultr/vultr-cli/vultr-cli"
  exit 1
fi

if ! vultr-cli account info &>/dev/null; then
  echo "ERROR: Cannot reach Vultr API. Check VULTR_API_KEY."
  exit 1
fi

# ── Config ────────────────────────────────────────────────────────────────────
REGION="${REGION:-ewr}"
INSTANCE_LABEL="${INSTANCE_LABEL:-hyperforge-gpu}"
BLOCK_STORAGE_GB="${BLOCK_STORAGE_GB:-500}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "════════════════════════════════════════════════"
echo "  Hyperforge AI — Vultr GPU Deployment"
echo "  Region:  $REGION"
echo "  Label:   $INSTANCE_LABEL"
echo "  Disk:    ${BLOCK_STORAGE_GB}GB block storage"
echo "════════════════════════════════════════════════"
echo ""

# ── Discover GPU plans ────────────────────────────────────────────────────────
echo "Fetching available GPU plans in region '$REGION'..."
echo ""

# Query plans via API directly (vultr-cli plans list output varies by version)
PLANS_JSON=$(curl -sf "https://api.vultr.com/v2/plans?type=vcg&per_page=100" \
  -H "Authorization: Bearer $VULTR_API_KEY") || {
  echo "ERROR: Could not fetch plans. Check VULTR_API_KEY and internet connection."
  exit 1
}

# Filter plans available in the selected region
REGION_PLANS=$(echo "$PLANS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
plans = data.get('plans', [])
region = '$REGION'
avail = [p for p in plans if region in p.get('locations', [])]
if not avail:
    # fallback: show all if region filter yields nothing
    avail = plans
for i, p in enumerate(avail):
    vcpu = p.get('vcpu_count', '?')
    ram  = p.get('ram', 0) // 1024
    disk = p.get('disk', '?')
    bw   = p.get('bandwidth', '?')
    cost = p.get('monthly_cost', '?')
    hr   = round(float(cost)/730, 3) if cost != '?' else '?'
    gpus = p.get('gpu_vram_mb', 0) // 1024
    gpu_type = p.get('gpu_type', '')
    print(f\"  [{i}] {p['id']:<35} | {gpu_type:<10} {gpus}GB VRAM | {vcpu}vCPU {ram}GB RAM | \${hr}/hr (\${cost}/mo)\")
" 2>/dev/null) || true

if [[ -z "$REGION_PLANS" ]]; then
  echo "No GPU plans found for region '$REGION'."
  echo "Try a different REGION (ewr, lax, sjc, ord, fra, nrt, blr)."
  echo ""
  echo "Available regions with GPU plans:"
  curl -sf "https://api.vultr.com/v2/regions" -H "Authorization: Bearer $VULTR_API_KEY" \
    | python3 -c "import sys,json; [print('  '+r['id']+' — '+r['city']+', '+r['country']) for r in json.load(sys.stdin)['regions']]" 2>/dev/null || true
  exit 1
fi

if [[ -n "${PLAN_ID:-}" ]]; then
  echo "Using preset PLAN_ID: $PLAN_ID"
else
  echo "Available GPU plans:"
  echo "$REGION_PLANS"
  echo ""
  echo -n "Select plan number (recommend A100 80GB): "
  read -r PLAN_NUM
  PLAN_ID=$(echo "$PLANS_JSON" | python3 -c "
import sys, json
data = json.load(sys.stdin)
plans = data.get('plans', [])
region = '$REGION'
avail = [p for p in plans if region in p.get('locations', [])]
if not avail: avail = plans
print(avail[int('$PLAN_NUM')]['id'])
" 2>/dev/null)
  echo "  Selected: $PLAN_ID"
fi
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
  echo "ERROR: Ubuntu 22.04 x64 not found. List available:"
  curl -sf "https://api.vultr.com/v2/os" -H "Authorization: Bearer $VULTR_API_KEY" \
    | python3 -c "import sys,json; [print(o['id'],o['name']) for o in json.load(sys.stdin)['os'] if 'Ubuntu' in o.get('name','')]" 2>/dev/null
  exit 1
fi
echo "  OS ID: $OS_ID (Ubuntu 22.04 LTS x64)"

# ── Encode startup script ──────────────────────────────────────────────────────
echo "Encoding bootstrap script..."
BOOTSTRAP_B64=$(base64 -w0 < "$SCRIPT_DIR/bootstrap.sh")

# Inject HF_TOKEN into the startup environment by prepending an export
STARTUP_SCRIPT="#!/usr/bin/env bash
export HF_TOKEN='$HF_TOKEN'
$(cat "$SCRIPT_DIR/bootstrap.sh" | tail -n +2)"

STARTUP_B64=$(echo "$STARTUP_SCRIPT" | base64 -w0)

# Upload startup script to Vultr
echo "Uploading startup script to Vultr..."
STARTUP_ID=$(curl -sf -X POST "https://api.vultr.com/v2/startup-scripts" \
  -H "Authorization: Bearer $VULTR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"hyperforge-bootstrap\",\"type\":\"boot\",\"script\":\"$STARTUP_B64\"}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['startup_script']['id'])")
echo "  Startup script ID: $STARTUP_ID"

# ── Create the instance ────────────────────────────────────────────────────────
echo ""
echo "Creating Vultr Cloud GPU instance..."
INSTANCE_RESP=$(curl -sf -X POST "https://api.vultr.com/v2/instances" \
  -H "Authorization: Bearer $VULTR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"region\": \"$REGION\",
    \"plan\": \"$PLAN_ID\",
    \"os_id\": $OS_ID,
    \"label\": \"$INSTANCE_LABEL\",
    \"sshkey_id\": [\"$SSH_KEY_ID\"],
    \"script_id\": \"$STARTUP_ID\",
    \"enable_ipv6\": false,
    \"backups\": \"disabled\",
    \"tags\": [\"hyperforge\", \"gpu\", \"ai\"]
  }")

INSTANCE_ID=$(echo "$INSTANCE_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['instance']['id'])" 2>/dev/null)
if [[ -z "$INSTANCE_ID" ]]; then
  echo "ERROR: Failed to create instance."
  echo "$INSTANCE_RESP"
  exit 1
fi
echo "  Instance ID: $INSTANCE_ID"

# ── Wait for instance to have an IP ───────────────────────────────────────────
echo "Waiting for instance to become active (~2-3 min)..."
PUBLIC_IP=""
for i in $(seq 1 36); do
  sleep 10
  STATUS_RESP=$(curl -sf "https://api.vultr.com/v2/instances/$INSTANCE_ID" \
    -H "Authorization: Bearer $VULTR_API_KEY" || echo "{}")
  STATUS=$(echo "$STATUS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('instance',{}).get('status',''))" 2>/dev/null)
  IP=$(echo "$STATUS_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('instance',{}).get('main_ip',''))" 2>/dev/null)
  echo -n "  [$((i*10))s] Status: $STATUS — IP: ${IP:-pending}  "
  if [[ "$STATUS" == "active" && -n "$IP" && "$IP" != "0.0.0.0" ]]; then
    PUBLIC_IP="$IP"
    echo ""
    break
  fi
  echo ""
done

if [[ -z "$PUBLIC_IP" ]]; then
  echo "Instance did not become active in time. Check Vultr console."
  echo "  Instance ID: $INSTANCE_ID"
  exit 1
fi

# ── Create block storage volume ────────────────────────────────────────────────
echo ""
echo "Creating ${BLOCK_STORAGE_GB}GB block storage volume..."
BLOCK_RESP=$(curl -sf -X POST "https://api.vultr.com/v2/blocks" \
  -H "Authorization: Bearer $VULTR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"region\": \"$REGION\",
    \"size_gb\": $BLOCK_STORAGE_GB,
    \"label\": \"hyperforge-models\",
    \"block_type\": \"storage_opt\"
  }")
BLOCK_ID=$(echo "$BLOCK_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['block']['id'])" 2>/dev/null)
echo "  Block volume ID: $BLOCK_ID"

# Attach block volume to instance
sleep 5
echo "Attaching block volume to instance..."
curl -sf -X POST "https://api.vultr.com/v2/blocks/$BLOCK_ID/attach" \
  -H "Authorization: Bearer $VULTR_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"instance_id\": \"$INSTANCE_ID\", \"live\": true}" > /dev/null
echo "  Volume attached."

# ── Save connection info ───────────────────────────────────────────────────────
cat > "$SCRIPT_DIR/.instance" <<EOF
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
REGION=$REGION
PLAN_ID=$PLAN_ID
BLOCK_ID=$BLOCK_ID
STARTUP_ID=$STARTUP_ID
EOF

# ── Print next steps ───────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Instance:   $INSTANCE_ID"
echo "  Public IP:  $PUBLIC_IP"
echo "  Block vol:  $BLOCK_ID (${BLOCK_STORAGE_GB}GB)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Bootstrap running (~8 min). Monitor with:"
echo "  ssh root@$PUBLIC_IP 'tail -f /var/log/hyperforge-bootstrap.log'"
echo ""
echo "After bootstrap completes:"
echo "  1. SSH in:         ssh root@$PUBLIC_IP"
echo "  2. Format disk:    sudo /opt/flux-server/flux-server/deploy/vultr/setup_storage.sh"
echo "  3. Verify .env:    nano /opt/flux-server/flux-server/.env"
echo "                     # HF_TOKEN should already be set if passed during launch"
echo "  4. Start service:  cd /opt/flux-server/flux-server && sudo docker compose up --build -d"
echo ""
echo "Verify:"
echo "  curl http://$PUBLIC_IP:8080/health"
echo "  Open: http://$PUBLIC_IP:8080"
echo ""
echo "Stop instance (billing pauses):"
echo "  vultr-cli instance stop $INSTANCE_ID"
echo ""
echo "Saved connection info: $SCRIPT_DIR/.instance"
