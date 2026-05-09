#!/usr/bin/env bash
# bootstrap.sh — Vultr user-data startup script for Hyperforge AI.
# Injected as startup script when the instance is created.
# Runs once as root on first boot.
#
# Log: /var/log/hyperforge-bootstrap.log

set -euo pipefail
exec >> /var/log/hyperforge-bootstrap.log 2>&1

echo "=== Hyperforge bootstrap started at $(date) ==="
echo "    Host: $(hostname) | Kernel: $(uname -r)"

# ── Wait for apt lock (cloud-init may be running) ─────────────────────────────
echo "Waiting for apt lock..."
for i in $(seq 1 20); do
  fuser /var/lib/dpkg/lock-frontend &>/dev/null || break
  sleep 5
done

# ── System packages ────────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y -qq \
  git curl wget unzip ca-certificates gnupg lsb-release \
  apt-transport-https software-properties-common

# ── Docker ─────────────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "Installing Docker..."
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable docker
  systemctl start docker
fi
echo "  Docker: $(docker --version)"

# ── NVIDIA drivers (Vultr GPU instances ship Ubuntu 22.04 with CUDA-capable kernel)
if ! command -v nvidia-smi &>/dev/null; then
  echo "Installing NVIDIA drivers..."
  add-apt-repository -y ppa:graphics-drivers/ppa 2>/dev/null || true
  apt-get update -qq
  apt-get install -y -qq ubuntu-drivers-common
  ubuntu-drivers install --gpgpu 2>/dev/null || apt-get install -y -qq nvidia-driver-550
fi
echo "  NVIDIA driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'pending reboot')"

# ── nvidia-container-toolkit ───────────────────────────────────────────────────
if ! dpkg -l nvidia-container-toolkit &>/dev/null; then
  echo "Installing nvidia-container-toolkit..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's|deb https://|deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://|g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
systemctl restart docker
echo "  nvidia-container-toolkit: ready"

# ── Clone repo ─────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/krishnagopalmishra1-we/flux-server.git"
APP_DIR="/opt/flux-server"
DEPLOY_BRANCH="${DEPLOY_BRANCH:-codex/hyperforge-runtime-hardening-impl}"

if [[ -d "$APP_DIR/.git" ]]; then
  echo "Repo exists, switching to $DEPLOY_BRANCH and pulling latest..."
  git -C "$APP_DIR" fetch --depth=1 origin "$DEPLOY_BRANCH"
  git -C "$APP_DIR" checkout -B "$DEPLOY_BRANCH" "origin/$DEPLOY_BRANCH"
else
  echo "Cloning $REPO_URL branch $DEPLOY_BRANCH..."
  git clone --depth=1 --branch "$DEPLOY_BRANCH" "$REPO_URL" "$APP_DIR"
fi

FLUX_DIR="$APP_DIR/flux-server"
cd "$FLUX_DIR"

# ── Activate Vultr compose config ──────────────────────────────────────────────
cp "$FLUX_DIR/deploy/vultr/docker-compose.vultr.yml" "$FLUX_DIR/docker-compose.yml"
echo "  docker-compose.yml: set to Vultr config"

# ── Create .env if missing ─────────────────────────────────────────────────────
ENV_FILE="$FLUX_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$FLUX_DIR/.env.example" "$ENV_FILE" 2>/dev/null || cat > "$ENV_FILE" <<'ENVEOF'
HF_TOKEN=
API_KEYS=
LORA_DIR=/mnt/hf-cache/loras
VIDEO_LORA_DIR=/mnt/hf-cache/video_loras
WAN_DEFAULT_VARIANT=1.3b
OUTPUT_DIR=/mnt/outputs
OUTPUT_TTL_HOURS=168
VIDEO_PARALLEL_BACKEND=auto
GPUS_PER_JOB=4
NUM_WORKERS=2
JOB_BACKEND=redis
REDIS_URL=redis://redis:6379/0
ENVEOF
  echo "  .env created — set HF_TOKEN before starting"
fi

# Inject HF_TOKEN if it was passed as an env var during bootstrap
if [[ -n "${HF_TOKEN:-}" ]]; then
  sed -i "s|HF_TOKEN=.*|HF_TOKEN=$HF_TOKEN|" "$ENV_FILE"
  echo "  HF_TOKEN injected into .env"
fi

# ── Pre-create mount points (block storage attached separately) ────────────────
mkdir -p /mnt/model-disk/{hf-cache,outputs/image,outputs/video,loras,video_loras}
mkdir -p "$FLUX_DIR/loras" "$FLUX_DIR/video_loras"

# ── Placeholder symlinks so container starts even before block storage ─────────
# Real data goes on block storage volume — see setup_storage.sh
ln -sfn /mnt/model-disk/hf-cache    /mnt/hf-cache    2>/dev/null || true
ln -sfn /mnt/model-disk/outputs      /mnt/outputs     2>/dev/null || true

echo ""
echo "=== Bootstrap complete at $(date) ==="
echo ""
echo "NEXT STEPS:"
echo "  1. Run setup_storage.sh to attach + format the block volume"
echo "  2. Edit .env: sudo nano $ENV_FILE"
echo "  3. Start service: cd $FLUX_DIR && sudo docker compose up --build -d"
echo ""
# Note: driver may require one reboot — service start is manual to avoid racing
