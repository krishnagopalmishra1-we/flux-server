#!/usr/bin/env bash
# setup_storage.sh — Format and mount the Vultr block storage volume.
# Run manually after SSH-ing into the instance once bootstrap is done.
#
# The block volume appears as /dev/vdb (or similar) on Vultr instances.
# This script finds it, formats it (if new), and mounts at /mnt/model-disk.
#
# Idempotent: safe to run again if the volume is already mounted.

set -euo pipefail

MOUNT_POINT="/mnt/model-disk"
FSTAB_LABEL="hyperforge-data"

echo "=== Hyperforge storage setup ==="

# ── Find the attached block volume ────────────────────────────────────────────
# On Vultr, block volumes appear as /dev/vdb, /dev/vdc, etc. (not /dev/nvme)
# The root disk is /dev/vda. Block storage volume is the next one.
BLOCK_DEV=""
for dev in /dev/vdb /dev/vdc /dev/xvdb /dev/sdb; do
  if [[ -b "$dev" ]]; then
    BLOCK_DEV="$dev"
    break
  fi
done

if [[ -z "$BLOCK_DEV" ]]; then
  echo "ERROR: No block volume found."
  echo "Attached block devices:"
  lsblk -o NAME,SIZE,TYPE,MOUNTPOINT
  echo ""
  echo "Ensure the block storage volume is attached in the Vultr console"
  echo "or set BLOCK_DEV manually: BLOCK_DEV=/dev/vdb $0"
  exit 1
fi

BLOCK_DEV="${BLOCK_DEV:-/dev/vdb}"
echo "  Block device: $BLOCK_DEV ($(lsblk -no SIZE "$BLOCK_DEV"))"

# ── Check if already mounted ───────────────────────────────────────────────────
if mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
  echo "  Already mounted at $MOUNT_POINT — skipping format/mount."
  echo ""
  echo "Storage layout:"
  df -h "$MOUNT_POINT"
  ls -la "$MOUNT_POINT/"
  exit 0
fi

# ── Format if no filesystem present ───────────────────────────────────────────
EXISTING_FS=$(blkid -s TYPE -o value "$BLOCK_DEV" 2>/dev/null || true)
if [[ -z "$EXISTING_FS" ]]; then
  echo "  Formatting $BLOCK_DEV as ext4 with label '$FSTAB_LABEL'..."
  mkfs.ext4 -L "$FSTAB_LABEL" -F "$BLOCK_DEV"
  echo "  Format complete."
else
  echo "  Filesystem already exists: $EXISTING_FS — skipping format."
fi

# ── Create mount point and mount ───────────────────────────────────────────────
mkdir -p "$MOUNT_POINT"
mount "$BLOCK_DEV" "$MOUNT_POINT"
echo "  Mounted $BLOCK_DEV → $MOUNT_POINT"

# ── Add to fstab for persistence across reboots ────────────────────────────────
FSTAB_ENTRY="LABEL=$FSTAB_LABEL  $MOUNT_POINT  ext4  defaults,nofail  0  2"
if ! grep -qF "$FSTAB_LABEL" /etc/fstab; then
  echo "$FSTAB_ENTRY" >> /etc/fstab
  echo "  Added to /etc/fstab."
else
  echo "  Already in /etc/fstab."
fi

# ── Create directory structure ─────────────────────────────────────────────────
echo ""
echo "Creating model cache directory structure..."
mkdir -p "$MOUNT_POINT"/{hf-cache,outputs/image,outputs/video,loras,video_loras,redis}
chmod 777 "$MOUNT_POINT/outputs" "$MOUNT_POINT/outputs/image" "$MOUNT_POINT/outputs/video"
# Redis container runs as UID 999; needs write access to its data directory.
chmod 777 "$MOUNT_POINT/redis"
echo "  Directories created."

# ── Update symlinks ────────────────────────────────────────────────────────────
ln -sfn "$MOUNT_POINT/hf-cache" /mnt/hf-cache   2>/dev/null || true
ln -sfn "$MOUNT_POINT/outputs"  /mnt/outputs    2>/dev/null || true
echo "  Symlinks updated."

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Storage ready ==="
df -h "$MOUNT_POINT"
echo ""
echo "Directory layout:"
ls -la "$MOUNT_POINT/"
echo ""
echo "Next steps:"
echo "  1. Verify .env:    nano /opt/flux-server/flux-server/.env"
echo "  2. Start service:  cd /opt/flux-server/flux-server && sudo docker compose up --build -d"
echo "  3. Check GPU:      nvidia-smi"
