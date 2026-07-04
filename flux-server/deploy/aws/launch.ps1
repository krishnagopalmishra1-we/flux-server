# launch.ps1 — Launch Hyperforge AI on AWS EC2 (g5.2xlarge On-Demand) from Windows.
# Replicates launch.sh behavior natively for PowerShell.

# ── Configurable Parameters ──────────────────────────────────────────────────
$KeyName = $env:KEY_NAME
if (-not $KeyName) { $KeyName = "hyperforge" }

$Region = $env:REGION
if (-not $Region) { $Region = "us-east-1" }

# Target g5.2xlarge (1 × A10G 24GB, 8 vCPUs) to fit On-Demand G quota
$InstanceType = $env:INSTANCE_TYPE
if (-not $InstanceType) { $InstanceType = "g5.2xlarge" }

$InstanceName = "hyperforge-gpu"
$SgName = "hyperforge-sg"

$DataDiskGb = 1000
$DataDiskThroughput = 500
$DataDiskIops = 6000

Write-Host "=== Hyperforge AI - AWS Deployment (Windows) ==="
Write-Host "    Region:        $Region"
Write-Host "    Instance Type: $InstanceType (On-Demand)"
Write-Host "    Key Pair Name: $KeyName"
Write-Host ""

# ── Validate AWS CLI ──────────────────────────────────────────────────────────
$null = aws sts get-caller-identity --region $Region --output text 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "AWS CLI is not configured or credentials have expired. Please run 'aws configure' first."
    exit 1
}

# ── Ensure Key Pair and SSH Folder Setup ─────────────────────────────────────
$sshDir = Join-Path $env:USERPROFILE ".ssh"
if (-not (Test-Path $sshDir)) {
    $null = New-Item -ItemType Directory -Path $sshDir -Force
}
$pemPath = Join-Path $sshDir "$KeyName.pem"

Write-Host "Checking for existing AWS Key Pair '$KeyName'..."
$keyExists = aws ec2 describe-key-pairs --key-names $KeyName --region $Region --query 'KeyPairs[0].KeyName' --output text 2>$null

if ($LASTEXITCODE -eq 0 -and $keyExists -eq $KeyName) {
    Write-Host "  AWS Key Pair '$KeyName' already exists."
    if (-not (Test-Path $pemPath)) {
        Write-Warning "Local private key file '$pemPath' is missing but the key pair exists on AWS."
        Write-Warning "If you do not have the private key, you must delete the key pair on AWS and run this script again."
    }
} else {
    Write-Host "AWS Key Pair '$KeyName' not found. Creating it..."
    $keyMaterial = aws ec2 create-key-pair --key-name $KeyName --query 'KeyMaterial' --output text --region $Region
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create AWS Key Pair '$KeyName'."
        exit 1
    }
    $keyMaterial | Out-File -FilePath $pemPath -Encoding ascii
    Write-Host "  Private key saved to: $pemPath"
    
    # Restrict permissions (equivalent of chmod 400)
    Write-Host "  Restricting permissions on local private key..."
    # Disable inheritance and grant full access only to the current user
    $null = icacls $pemPath /inheritance:r /grant "${env:USERNAME}:F"
}

# ── Query Deep Learning AMI ──────────────────────────────────────────────────
Write-Host "Finding latest Deep Learning AMI (Ubuntu 22.04)..."
$amiId = aws ec2 describe-images `
  --owners amazon `
  --filters `
    "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch * (Ubuntu 22.04)*" `
    "Name=state,Values=available" `
  --query "sort_by(Images, &CreationDate)[-1].ImageId" `
  --output text `
  --region $Region

if ($LASTEXITCODE -ne 0 -or $amiId -eq "None" -or -not $amiId) {
    Write-Error "No Deep Learning AMI found in region $Region."
    exit 1
}
Write-Host "  AMI ID: $amiId"

# ── Create/Verify Security Group ──────────────────────────────────────────────
Write-Host "Checking Security Group '$SgName'..."
$sgId = aws ec2 describe-security-groups `
  --filters "Name=group-name,Values=$SgName" `
  --query 'SecurityGroups[0].GroupId' `
  --output text `
  --region $Region 2>$null

if ($LASTEXITCODE -ne 0 -or $sgId -eq "None" -or -not $sgId) {
    Write-Host "Creating security group '$SgName'..."
    $sgId = aws ec2 create-security-group `
      --group-name $SgName `
      --description "Hyperforge AI: SSH (22) + API (8080)" `
      --region $Region `
      --query 'GroupId' --output text
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create Security Group '$SgName'."
        exit 1
    }
      
    # Authorize SSH
    $null = aws ec2 authorize-security-group-ingress `
      --group-id $sgId --protocol tcp --port 22 --cidr 0.0.0.0/0 --region $Region
    # Authorize API
    $null = aws ec2 authorize-security-group-ingress `
      --group-id $sgId --protocol tcp --port 8080 --cidr 0.0.0.0/0 --region $Region
}
Write-Host "  Security Group: $sgId"

# ── Pick Subnet ──────────────────────────────────────────────────────────────
$subnetId = $env:SUBNET_ID
if (-not $subnetId) {
    $subnetId = aws ec2 describe-subnets `
      --filters "Name=default-for-az,Values=true" `
      --query 'Subnets[0].SubnetId' `
      --output text --region $Region
}
if ($LASTEXITCODE -ne 0 -or -not $subnetId) {
    Write-Error "Failed to identify subnet."
    exit 1
}
Write-Host "  Subnet ID: $subnetId"

# ── Launch On-Demand Instance ────────────────────────────────────────────────
Write-Host ""
Write-Host "Launching $InstanceType On-Demand instance..."

# Double quotes must be escaped with backslashes so they are preserved when passed to aws.exe
$blockDeviceMappings = @"
[
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
      \"VolumeSize\": $DataDiskGb,
      \"VolumeType\": \"gp3\",
      \"Throughput\": $DataDiskThroughput,
      \"Iops\": $DataDiskIops,
      \"DeleteOnTermination\": false
    }
  }
]
"@

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$bootstrapPath = Join-Path $scriptDir "bootstrap.sh"
# Convert to absolute path with forward slashes for AWS CLI file prefix
$bootstrapUrl = "file://" + $bootstrapPath.Replace("\", "/")

# Launch as On-Demand to utilize the active quota
$instanceId = aws ec2 run-instances `
  --image-id $amiId `
  --instance-type $InstanceType `
  --key-name $KeyName `
  --security-group-ids $sgId `
  --subnet-id $subnetId `
  --block-device-mappings $blockDeviceMappings `
  --user-data $bootstrapUrl `
  --tag-specifications `
    "ResourceType=instance,Tags=[{Key=Name,Value=$InstanceName},{Key=Project,Value=hyperforge}]" `
    "ResourceType=volume,Tags=[{Key=Name,Value=hyperforge-data},{Key=Project,Value=hyperforge}]" `
  --region $Region `
  --query 'Instances[0].InstanceId' `
  --output text

if ($LASTEXITCODE -ne 0 -or -not $instanceId) {
    Write-Error "Failed to launch EC2 instance."
    exit 1
}

Write-Host "  Instance ID: $instanceId"
Write-Host "Waiting for instance to enter 'running' state..."
$null = aws ec2 wait instance-running --instance-ids $instanceId --region $Region

$publicIp = aws ec2 describe-instances `
  --instance-ids $instanceId `
  --query 'Reservations[0].Instances[0].PublicIpAddress' `
  --output text --region $Region

# ── Save connection details ──────────────────────────────────────────────────
$infoContent = @"
INSTANCE_ID=$instanceId
PUBLIC_IP=$publicIp
REGION=$Region
KEY_NAME=$KeyName
INSTANCE_TYPE=$InstanceType
"@
$infoPath = Join-Path $scriptDir ".instance"
$infoContent | Out-File -FilePath $infoPath -Encoding utf8 -Force

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host "  Instance running: $instanceId"
Write-Host "  Public IP:        $publicIp"
Write-Host "═══════════════════════════════════════════════════════════"
Write-Host ""
Write-Host "Bootstrap is running in the background (~5-10 min). Monitor it with:"
Write-Host "  ssh -i $pemPath ubuntu@$publicIp 'tail -f /var/log/hyperforge-bootstrap.log'"
Write-Host ""
Write-Host "Once bootstrap finishes, SSH in and configure disks:"
Write-Host "  1. SSH in:        ssh -i $pemPath ubuntu@$publicIp"
Write-Host "  2. Mount disk:    sudo /opt/flux-server/flux-server/deploy/aws/setup_disks.sh"
Write-Host "  3. Set HF token:  sudo nano /opt/flux-server/flux-server/.env"
Write-Host "  4. Start service: cd /opt/flux-server/flux-server && sudo docker compose up -d"
Write-Host ""
Write-Host "Stop instance (saves billing):"
Write-Host "  aws ec2 stop-instances --instance-ids $instanceId --region $Region"
