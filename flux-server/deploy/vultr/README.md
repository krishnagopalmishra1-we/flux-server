# Hyperforge AI — Vultr Cloud GPU Deployment

Deploy the full Hyperforge AI stack (image + video generation) on a Vultr Cloud GPU instance with a single A100 80GB or L40S GPU.

---

## Why Vultr Bare Metal

| | Vultr Bare Metal (8× A100 SXM) | GCP A100 40GB Spot | AWS p4de.24xlarge |
|---|---|---|---|
| GPU | **8× A100 SXM 80GB** (640GB VRAM) | A100 40GB | 8× A100 80GB |
| Price | ~$11.92/hr preemptible | ~$1.5–2/hr | ~$32/hr |
| Preemption | Yes (preemptible) | Yes | Yes (spot) |
| Disk | **4× NVMe SSD** | HDD (30 min loads) | gp3 EBS |
| WAN 14B load time | **~2 min** (NVMe) | ~30 min (HDD) | ~4 min |
| Plan ID | `vbm-112c-2048gb-8-a100-gpu` | — | — |

---

## Step 0 — Get a Vultr API Key

1. Log in at [console.vultr.com](https://console.vultr.com)
2. Go to **Account → API**
3. Click **Enable API** → copy the key
4. Set it in your shell:

```bash
export VULTR_API_KEY="your_api_key_here"
```

Add to `~/.bashrc` or `~/.zshrc` to persist.

---

## Step 1 — Install vultr-cli

```bash
# Linux / WSL
curl -fsSL https://github.com/vultr/vultr-cli/releases/latest/download/vultr-cli_linux_amd64.tar.gz \
  | tar -xz -C /usr/local/bin vultr-cli

# Verify
vultr-cli version
```

Or on macOS:
```bash
brew install vultr/vultr-cli/vultr-cli
```

---

## Step 2 — Add your SSH key to Vultr

```bash
# Generate key if you don't have one
ssh-keygen -t ed25519 -C "hyperforge" -f ~/.ssh/hyperforge_vultr

# Upload to Vultr
vultr-cli ssh-key create --name "hyperforge" \
  --key "$(cat ~/.ssh/hyperforge_vultr.pub)"

# Note the returned key ID
vultr-cli ssh-key list
```

---

## Step 3 — Launch the instance

```bash
cd flux-server/deploy/vultr

# Required
export VULTR_API_KEY="your_key"
export SSH_KEY_ID="your_ssh_key_id"        # from Step 2
export HF_TOKEN="hf_your_huggingface_token"

# Optional overrides
export REGION="ewr"                        # ewr=New Jersey, lax=LA, sjc=San Jose
export INSTANCE_LABEL="hyperforge-gpu"
export BLOCK_STORAGE_GB=500               # model cache disk size

./launch.sh
```

`launch.sh` will:
1. Query all available GPU plans in your chosen region
2. Print a selection menu — pick the A100 80GB plan
3. Create the instance with your SSH key and startup script
4. Create a persistent block storage volume for models
5. Print SSH and API connection info

---

## Step 4 — Wait for bootstrap (~8 min)

```bash
# Monitor bootstrap progress
ssh -i ~/.ssh/hyperforge_vultr root@<PUBLIC_IP> \
  'tail -f /var/log/hyperforge-bootstrap.log'
```

Bootstrap installs Docker, NVIDIA drivers, clones the repo, and starts building the container.

---

## Step 5 — Attach and format the model disk

```bash
ssh -i ~/.ssh/hyperforge_vultr root@<PUBLIC_IP>
sudo /opt/flux-server/flux-server/deploy/vultr/setup_storage.sh
```

This formats the attached block volume and mounts it at `/mnt/model-disk`.

---

## Step 6 — Set HF token and start the service

```bash
# Still inside the VM
sudo nano /opt/flux-server/flux-server/.env
# Add: HF_TOKEN=hf_your_token_here

cd /opt/flux-server/flux-server
sudo docker compose up --build -d

# Verify
curl http://localhost:8080/health
curl http://localhost:8080/models
```

---

## Step 7 — Verify from your machine

```bash
curl http://<PUBLIC_IP>:8080/health
# Open in browser: http://<PUBLIC_IP>:8080
```

---

## Daily Operations

```bash
# Stop instance (billing pauses)
vultr-cli instance stop <INSTANCE_ID>

# Start instance
vultr-cli instance start <INSTANCE_ID>

# SSH
ssh -i ~/.ssh/hyperforge_vultr root@<PUBLIC_IP>

# Deploy code update
ssh root@<PUBLIC_IP> "cd /opt/flux-server/flux-server && git pull && sudo docker compose up --build -d"

# Check GPU + container
ssh root@<PUBLIC_IP> "nvidia-smi && sudo docker ps"

# View logs
ssh root@<PUBLIC_IP> "sudo docker compose -f /opt/flux-server/flux-server/docker-compose.yml logs -f --tail=100"
```

---

## Disk Layout (after setup_storage.sh)

| Mount | Device | Size | Contents |
|-------|--------|------|----------|
| `/` | Local NVMe | 160GB | OS + Docker + container |
| `/mnt/model-disk` | Block storage | 500GB | HF model cache, outputs, LoRAs |

**Model path inside container**: `/mnt/hf-cache` → `/mnt/model-disk/hf-cache`

---

## Environment Variables (.env)

```env
HF_TOKEN=hf_your_token_here
API_KEYS=                          # blank = no auth required
LORA_DIR=/mnt/hf-cache/loras
VIDEO_LORA_DIR=/mnt/hf-cache/video_loras
WAN_DEFAULT_VARIANT=1.3b
OUTPUT_DIR=/mnt/outputs
OUTPUT_TTL_HOURS=168               # keep outputs 7 days
```

---

## Cost Estimate

| Scenario | Duration | Cost |
|----------|----------|------|
| Image generation session (2hr) | 2hr × $2.50 | $5 |
| WAN 14B full day | 24hr × $2.50 | $60 |
| Stopped instance | — | $0 (no compute charge; block storage ~$0.01/GB/hr) |

Vultr charges **per-second** with no minimum. Stop the instance when not generating.

---

## Troubleshooting

**Container not starting**
```bash
sudo docker compose -f /opt/flux-server/flux-server/docker-compose.yml logs
```

**GPU not detected**
```bash
nvidia-smi
# If missing: sudo reboot (driver needs one reboot after bootstrap)
```

**Model download stuck**
```bash
# Check HF_TOKEN is set
sudo docker exec flux-server-flux-server-1 env | grep HF_TOKEN
```

**Block storage not mounted**
```bash
lsblk
sudo /opt/flux-server/flux-server/deploy/vultr/setup_storage.sh
```
