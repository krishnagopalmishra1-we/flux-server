#!/usr/bin/env bash
# launch.sh — Launch Hyperforge AI on an AWS p4d.24xlarge spot instance.
#
# Prerequisites:
#   aws configure   (or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
#   A key pair created in the target region (set KEY_NAME below)
#
# Usage:
#   KEY_NAME=my-keypair ./launch.sh
#   KEY_NAME=my-keypair REGION=us-west-2 INSTANCE_TYPE=p4d.24xlarge ./launch.sh

set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
: "${KEY_NAME:?Set KEY_NAME to your EC2 key pair name (without .pem extension)}"

# ── Configurable ──────────────────────────────────────────────────────────────
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-p4d.24xlarge}"
INSTANCE_NAME="hyperforge-gpu"
SG_NAME="hyperforge-sg"

# EBS data disk — model cache persists across spot interruptions (DeleteOnTermination=false)
# 1 TB gp3 with 500 MB/s throughput → WAN 14B (118 GB) loads in ~4 min vs 30 min on GCP HDD
DATA_DISK_GB=1000
DATA_DISK_THROUGHPUT=500
DATA_DISK_IOPS=6000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Hyperforge AI — AWS Deployment ==="
echo "    Region:  $REGION"
echo "    Type:    $INSTANCE_TYPE"
echo ""

# ── Validate AWS CLI ───────────────────────────────────────────────────────────
if ! aws sts get-caller-identity --region "$REGION" --output text &>/dev/null; then
  echo "ERROR: AWS CLI not configured or no credentials. Run 'aws configure' first."
  exit 1
fi

# ── Find latest Deep Learning AMI (Ubuntu 22.04 + CUDA + Docker preinstalled) ──
echo "Finding latest AWS Deep Learning AMI (Ubuntu 22.04)..."
AMI_ID=$(aws ec2 describe-images \
  --owners amazon \
  --filters \
    "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*" \
    "Name=state,Values=available" \
  --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
  --output text \
  --region "$REGION")

if [[ -z "$AMI_ID" || "$AMI_ID" == "None" ]]; then
  echo "ERROR: No Deep Learning AMI found in $REGION."
  echo "List available AMIs:"
  echo "  aws ec2 describe-images --owners amazon --filters 'Name=name,Values=Deep Learning*Ubuntu 22.04*' --query 'Images[].Name' --output text --region $REGION"
  exit 1
fi
echo "  AMI: $AMI_ID"

# ── Create security group (idempotent) ────────────────────────────────────────
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=$SG_NAME" \
  --query 'SecurityGroups[0].GroupId' \
  --output text \
  --region "$REGION" 2>/dev/null || true)

if [[ -z "$SG_ID" || "$SG_ID" == "None" ]]; then
  echo "Creating security group '$SG_NAME'..."
  SG_ID=$(aws ec2 create-security-group \
    --group-name "$SG_NAME" \
    --description "Hyperforge AI: SSH (22) + API (8080)" \
    --region "$REGION" \
    --query 'GroupId' --output text)

  # SSH access
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 \
    --region "$REGION" > /dev/null
  # API access
  aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" --protocol tcp --port 8080 --cidr 0.0.0.0/0 \
    --region "$REGION" > /dev/null
fi
echo "  Security group: $SG_ID"

# ── Pick default subnet (first AZ with p4d capacity — try use-east-1a) ────────
SUBNET_ID="${SUBNET_ID:-$(aws ec2 describe-subnets \
  --filters "Name=default-for-az,Values=true" \
  --query 'Subnets[0].SubnetId' \
  --output text --region "$REGION")}"
echo "  Subnet: $SUBNET_ID"

# ── Launch spot instance ───────────────────────────────────────────────────────
echo ""
echo "Launching $INSTANCE_TYPE spot instance..."
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --subnet-id "$SUBNET_ID" \
  --block-device-mappings "[
    {
      \"DeviceName\": \"/dev/sda1\",
      \"Ebs\": {
        \"VolumeSize\": 200,
        \"VolumeType\": \"gp3\",
        \"Throughput\": 500,
        \"DeleteOnTermination\": true
      }
    },
    {
      \"DeviceName\": \"/dev/sdf\",
      \"Ebs\": {
        \"VolumeSize\": $DATA_DISK_GB,
        \"VolumeType\": \"gp3\",
        \"Throughput\": $DATA_DISK_THROUGHPUT,
        \"Iops\": $DATA_DISK_IOPS,
        \"DeleteOnTermination\": false
      }
    }
  ]" \
  --instance-market-options '{
    "MarketType": "spot",
    "SpotOptions": {
      "SpotInstanceType": "persistent",
      "InstanceInterruptionBehavior": "stop"
    }
  }' \
  --user-data "file://$SCRIPT_DIR/bootstrap.sh" \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=$INSTANCE_NAME},{Key=Project,Value=hyperforge}]" \
    "ResourceType=volume,Tags=[{Key=Name,Value=hyperforge-data},{Key=Project,Value=hyperforge}]" \
  --region "$REGION" \
  --query 'Instances[0].InstanceId' \
  --output text)

echo "  Instance: $INSTANCE_ID"
echo ""
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text --region "$REGION")

# ── Save connection info ───────────────────────────────────────────────────────
cat > "$SCRIPT_DIR/.instance" <<EOF
INSTANCE_ID=$INSTANCE_ID
PUBLIC_IP=$PUBLIC_IP
REGION=$REGION
KEY_NAME=$KEY_NAME
INSTANCE_TYPE=$INSTANCE_TYPE
EOF

# ── Print next steps ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Instance running: $INSTANCE_ID"
echo "  Public IP:        $PUBLIC_IP"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Bootstrap running in background (~5-10 min). Monitor with:"
echo "  ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP 'tail -f /var/log/hyperforge-bootstrap.log'"
echo ""
echo "Once bootstrap finishes — complete setup:"
echo "  1. SSH in:        ssh -i ~/.ssh/$KEY_NAME.pem ubuntu@$PUBLIC_IP"
echo "  2. Mount disk:    sudo /opt/flux-server/flux-server/deploy/aws/setup_disks.sh"
echo "  3. Set HF token:  sudo nano /opt/flux-server/flux-server/.env"
echo "                    # Add: HF_TOKEN=hf_your_token_here"
echo "  4. Start service: cd /opt/flux-server/flux-server && sudo docker compose up -d"
echo ""
echo "Verify:"
echo "  curl http://$PUBLIC_IP:8080/health"
echo "  curl http://$PUBLIC_IP:8080/models"
echo ""
echo "Stop instance (saves billing):"
echo "  aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "Restart instance:"
echo "  aws ec2 start-instances --instance-ids $INSTANCE_ID --region $REGION"
