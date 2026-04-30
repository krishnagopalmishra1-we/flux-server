#!/usr/bin/env bash
# bootstrap.sh — EC2 user-data script for Hyperforge AI.
# Runs once as root on first boot. Clones the repo, sets up Docker/NVIDIA,
# and activates the AWS compose config. Does NOT mount the data disk — run
# setup_disks.sh manually after SSH-ing in.
#
# Log: /var/log/hyperforge-bootstrap.log

set -euo pipefail
exec >> /var/log/hyperforge-bootstrap.log 2>&1

echo "=== Hyperforge bootstrap started at $(date) ==="

# ── Ensure nvidia-container-toolkit (Deep Learning AMI usually has it) ─────────
if ! dpkg -l nvidia-container-toolkit &>/dev/null; then
  echo "Installing nvidia-container-toolkit..."
  DISTRO="ubuntu22.04"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL "https://nvidia.github.io/libnvidia-container/$DISTRO/libnvidia-container.list" \
    | sed 's|deb https://|deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://|g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
fi

# Configure Docker runtime and restart (idempotent)
nvidia-ctk runtime configure --runtime=docker 2>/dev/null || true
systemctl restart docker
echo "  nvidia-container-toolkit: ready"

# ── Ensure git ─────────────────────────────────────────────────────────────────
apt-get install -y -qq git 2>/dev/null || true

# ── Clone or update repo ───────────────────────────────────────────────────────
REPO_URL="https://github.com/krishnagopalmishra1-we/flux-server.git"
APP_DIR="/opt/flux-server"

if [[ -d "$APP_DIR/.git" ]]; then
  echo "Repo already cloned, pulling latest..."
  git -C "$APP_DIR" pull --ff-only
else
  echo "Cloning $REPO_URL..."
  git clone "$REPO_URL" "$APP_DIR"
fi

FLUX_DIR="$APP_DIR/flux-server"

# ── Activate AWS compose config ────────────────────────────────────────────────
# Copy AWS compose file over the default so 'docker compose up' just works.
cp "$FLUX_DIR/deploy/aws/docker-compose.aws.yml" "$FLUX_DIR/docker-compose.yml"
echo "  docker-compose.yml: set to AWS config"

# ── Create .env if missing ─────────────────────────────────────────────────────
ENV_FILE="$FLUX_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$FLUX_DIR/.env.example" "$ENV_FILE"
  echo "  .env created from .env.example — you MUST set HF_TOKEN before starting"
fi

# ── Create LoRA directories (need to exist before container starts) ────────────
mkdir -p "$FLUX_DIR/loras" "$FLUX_DIR/video_loras"

# ── Create placeholder model-disk dirs so container can start even without data ─
# Actual persistent data goes on the EBS disk (setup_disks.sh).
mkdir -p /mnt/model-disk/{hf-cache,model-cache-ssd,outputs/video,loras,video_loras}
chmod 777 /mnt/model-disk/outputs /mnt/model-disk/outputs/video

echo "=== Bootstrap complete at $(date) ==="
echo ""
echo "Next steps (after SSH-ing in):"
echo "  1. sudo $FLUX_DIR/deploy/aws/setup_disks.sh   # format + mount EBS data disk"
echo "  2. sudo nano $ENV_FILE                          # set HF_TOKEN=hf_..."
echo "  3. cd $FLUX_DIR && sudo docker compose up --build -d"
