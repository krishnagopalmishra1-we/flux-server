#!/usr/bin/env bash
# setup_disks.sh — Format and mount the EBS data disk on a new Hyperforge EC2 instance.
# Run once as root AFTER SSH-ing in: sudo /opt/flux-server/flux-server/deploy/aws/setup_disks.sh
#
# The data disk (attached as /dev/sdf, appears as an NVMe device on Nitro instances)
# is mounted at /mnt/model-disk and holds all model cache, outputs, and LoRAs.

set -euo pipefail

MOUNT_POINT="/mnt/model-disk"
LABEL="hyperforge-data"

echo "=== Hyperforge disk setup ==="
echo ""
echo "Current block devices:"
lsblk -d -o NAME,SIZE,FSTYPE,MOUNTPOINT
echo ""

# ── Identify the EBS data disk ─────────────────────────────────────────────────
# On Nitro-based instances (p4d, g5, etc.) EBS volumes appear as NVMe devices.
# The volume's serial number (from nvme id-ctrl) starts with "vol" + volume ID.
# Local NVMe instance store serials look like random hex strings.

find_ebs_data_device() {
  if ! command -v nvme &>/dev/null; then
    apt-get install -y -qq nvme-cli 2>/dev/null || true
  fi

  local root_dev
  root_dev=$(lsblk -n -o PKNAME "$(df --output=source / | tail -1)" 2>/dev/null | head -1 || echo "")

  for dev in /dev/nvme*n1; do
    [[ -e "$dev" ]] || continue
    [[ "$(basename "$dev")" == "$root_dev" ]] && continue  # skip root

    # Check if already mounted
    if lsblk -n -o MOUNTPOINT "$dev" 2>/dev/null | grep -q '/'; then
      continue
    fi

    # EBS volumes have a serial starting with "vol" (vol0abc123...)
    if command -v nvme &>/dev/null; then
      local serial
      serial=$(nvme id-ctrl "$dev" 2>/dev/null | grep -E "^sn\s" | awk '{print $3}' || echo "")
      if [[ "$serial" == vol* ]]; then
        echo "$dev"
        return 0
      fi
    fi
  done

  # Fallback: find largest unformatted, unmounted block device (not loop/sr)
  local candidate
  candidate=$(lsblk -d -b -n -o NAME,SIZE,FSTYPE,MOUNTPOINT \
    | awk '$3=="" && $4=="" && !/loop/ && !/sr/ && !/rom/' \
    | sort -k2 -rn \
    | awk 'NR==1 {print "/dev/"$1}')
  echo "$candidate"
}

DATA_DEV=$(find_ebs_data_device)

if [[ -z "$DATA_DEV" ]]; then
  echo "ERROR: Could not auto-detect data disk."
  echo "Run 'lsblk' to identify the correct device and format it manually:"
  echo "  sudo mkfs.ext4 -L $LABEL /dev/<device>"
  echo "  echo 'LABEL=$LABEL $MOUNT_POINT ext4 defaults,nofail 0 2' | sudo tee -a /etc/fstab"
  echo "  sudo mount -a"
  exit 1
fi

echo "Detected data disk: $DATA_DEV"
echo ""

# ── Safety check ──────────────────────────────────────────────────────────────
EXISTING_FS=$(lsblk -n -o FSTYPE "$DATA_DEV" 2>/dev/null | head -1 || echo "")
if [[ -n "$EXISTING_FS" ]]; then
  echo "WARNING: $DATA_DEV already has filesystem '$EXISTING_FS'."
  read -r -p "This looks like a previously used disk. Mount it as-is? [y/N] " confirm
  if [[ "$confirm" =~ ^[Yy]$ ]]; then
    echo "Mounting existing filesystem..."
    mkdir -p "$MOUNT_POINT"
    # Try by label first, then by device
    if ! mount -L "$LABEL" "$MOUNT_POINT" 2>/dev/null; then
      mount "$DATA_DEV" "$MOUNT_POINT"
    fi
    goto_create_dirs=true
  else
    echo "Aborted."
    exit 1
  fi
else
  read -r -p "Format $DATA_DEV as ext4 and mount at $MOUNT_POINT? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

  echo "Formatting $DATA_DEV..."
  mkfs.ext4 -L "$LABEL" "$DATA_DEV"

  mkdir -p "$MOUNT_POINT"
  echo "Mounting..."
  mount -L "$LABEL" "$MOUNT_POINT"
  goto_create_dirs=false
fi

# ── Add to fstab for auto-mount on reboot ─────────────────────────────────────
if ! grep -q "$LABEL" /etc/fstab; then
  echo "LABEL=$LABEL  $MOUNT_POINT  ext4  defaults,nofail  0  2" >> /etc/fstab
  echo "  Added to /etc/fstab"
fi

# ── Create directory structure ─────────────────────────────────────────────────
echo "Creating directories..."
mkdir -p \
  "$MOUNT_POINT/hf-cache" \
  "$MOUNT_POINT/hf-cache/loras" \
  "$MOUNT_POINT/hf-cache/video_loras" \
  "$MOUNT_POINT/model-cache-ssd" \
  "$MOUNT_POINT/outputs/video" \
  "$MOUNT_POINT/loras" \
  "$MOUNT_POINT/video_loras"

chmod 777 "$MOUNT_POINT/outputs" "$MOUNT_POINT/outputs/video"

echo ""
echo "=== Disk setup complete ==="
df -h "$MOUNT_POINT"
echo ""
echo "Next: set HF_TOKEN in /opt/flux-server/flux-server/.env, then:"
echo "  cd /opt/flux-server/flux-server && sudo docker compose up --build -d"
