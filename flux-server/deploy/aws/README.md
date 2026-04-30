# Hyperforge AI — AWS Deployment

Target: single p4d.24xlarge spot instance (1 of its 8 × A100 40 GB GPUs).
Persistent model storage on a 1 TB EBS gp3 volume (500 MB/s, models survive stops).

## Prerequisites

1. AWS CLI installed and configured (`aws configure`).
2. An EC2 key pair created in your target region.
   ```bash
   aws ec2 create-key-pair --key-name hyperforge --query 'KeyMaterial' \
     --output text --region us-east-1 > ~/.ssh/hyperforge.pem
   chmod 400 ~/.ssh/hyperforge.pem
   ```
3. p4d quota approved for your region (us-east-1 has best availability).
   Check: AWS Console → Service Quotas → EC2 → "Running On-Demand P instances".

---

## Step-by-step

### 1. Launch the instance (run from your laptop)

```bash
cd flux-server/deploy/aws
KEY_NAME=hyperforge ./launch.sh
```

This will:
- Find the latest Deep Learning AMI (Ubuntu 22.04, CUDA pre-installed)
- Create a `hyperforge-sg` security group (ports 22 + 8080)
- Launch a p4d.24xlarge **spot** instance (persistent, stops on interruption)
- Attach a 200 GB root EBS + 1 TB data EBS (models persist across stops)
- Run `bootstrap.sh` as user-data in the background

Optional overrides:
```bash
REGION=us-west-2 INSTANCE_TYPE=p4d.24xlarge KEY_NAME=my-key ./launch.sh
```

### 2. Wait for bootstrap (~5-10 min), then SSH in

```bash
ssh -i ~/.ssh/hyperforge.pem ubuntu@<PUBLIC_IP>

# Monitor bootstrap progress
tail -f /var/log/hyperforge-bootstrap.log
```

### 3. Format and mount the data disk (run once on the VM)

```bash
sudo /opt/flux-server/flux-server/deploy/aws/setup_disks.sh
```

This formats the 1 TB EBS volume as ext4, mounts it at `/mnt/model-disk`,
and creates the model cache / output directories.

### 4. Set your Hugging Face token

```bash
sudo nano /opt/flux-server/flux-server/.env
# Add / update:  HF_TOKEN=hf_your_token_here
```

### 5. Build and start the service

```bash
cd /opt/flux-server/flux-server
sudo docker compose up --build -d

# Watch logs
sudo docker compose logs -f --tail=50
```

### 6. Verify

```bash
curl http://localhost:8080/health
curl http://localhost:8080/models
# Or from your laptop using the public IP
curl http://<PUBLIC_IP>:8080/health
```

---

## Daily operations

```bash
# SSH in (IP saved by launch.sh)
ssh -i ~/.ssh/hyperforge.pem ubuntu@<PUBLIC_IP>

# Check running jobs
curl -s http://localhost:8080/api/jobs | python3 -c \
  "import sys,json; [print(j['job_id'][:8], j['status'], j.get('progress')) for j in json.load(sys.stdin)]"

# Tail server logs
sudo docker compose -C /opt/flux-server/flux-server logs -f

# Stop instance (billing stops, EBS data persists)
aws ec2 stop-instances --instance-ids <INSTANCE_ID> --region us-east-1

# Start instance again
aws ec2 start-instances --instance-ids <INSTANCE_ID> --region us-east-1

# After restart: public IP changes — get the new one
aws ec2 describe-instances --instance-ids <INSTANCE_ID> \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

The instance ID is saved to `deploy/aws/.instance` after `launch.sh` runs.

### Re-deploy after code changes

```bash
cd /opt/flux-server
git pull
cd flux-server
sudo docker compose up --build -d
```

---

## Storage layout

| Path (inside container) | Host path | Contents |
|---|---|---|
| `/mnt/hf-cache` | `/mnt/model-disk/hf-cache` | All HF model weights (WAN 14B, I2V 14B, etc.) |
| `/app/model_cache` | `/mnt/model-disk/model-cache-ssd` | Priority model cache (WAN 1.3B, FLUX) |
| `/mnt/outputs` | `/mnt/model-disk/outputs` | Generated videos and images |
| `/app/loras` | `flux-server/loras/` | Uploaded image LoRAs |
| `/app/video_loras` | `flux-server/video_loras/` | Uploaded video LoRAs |

All model data survives instance stop/start. Data is lost only if the EBS volume
is explicitly deleted (it has `DeleteOnTermination=false`).

**Model load times from EBS gp3 (500 MB/s vs GCP HDD 65 MB/s):**
| Model | AWS EBS | GCP HDD |
|---|---|---|
| WAN 14B (118 GB) | ~4 min | ~30 min |
| WAN 1.3B (27 GB) | ~1 min | ~7 min |

---

## Cost estimates (us-east-1, spot)

| Component | Rate | Notes |
|---|---|---|
| p4d.24xlarge spot | ~$9–12/hr | Varies; on-demand is ~$32/hr |
| EBS gp3 1 TB | ~$0.12/GB/mo | ~$120/mo just for storage |
| Data transfer | $0.09/GB | Outbound video downloads |

**Stop the instance when not in use** — EBS charges continue even when stopped,
but they are negligible (~$4/day for 1 TB) vs. the GPU cost (~$9/hr).

### Cheaper alternatives for testing / image-only workloads

| Instance | GPU | VRAM | Notes |
|---|---|---|---|
| `g5.xlarge` | A10G | 24 GB | Can run WAN 1.3B and FLUX. 14B NF4 needs 25.5 GB — tight. |
| `g5.2xlarge` | A10G | 24 GB | Same GPU, more CPU/RAM — better for long video. |
| `p3.2xlarge` | V100 | 16 GB | Only image models. No 14B video. |

---

## Troubleshooting

**Bootstrap didn't finish / service not running:**
```bash
cat /var/log/hyperforge-bootstrap.log
sudo docker compose -C /opt/flux-server/flux-server ps
sudo docker compose -C /opt/flux-server/flux-server logs
```

**GPU not visible in container:**
```bash
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
# If this fails: nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
```

**Data disk not mounted after instance restart:**
```bash
lsblk
sudo mount -a   # fstab entry was added by setup_disks.sh
```

**Out of disk space:**
```bash
df -h /mnt/model-disk
# Clean old outputs (container auto-TTL is OUTPUT_TTL_HOURS in .env)
sudo find /mnt/model-disk/outputs -name '*.mp4' -mtime +1 -delete
```

**p4d spot capacity not available:**
Try a different AZ by setting `SUBNET_ID` to a subnet in another availability zone:
```bash
aws ec2 describe-subnets --filters 'Name=default-for-az,Values=true' \
  --query 'Subnets[].{AZ:AvailabilityZone,ID:SubnetId}' --output table --region us-east-1
SUBNET_ID=subnet-xxx KEY_NAME=hyperforge ./launch.sh
```
