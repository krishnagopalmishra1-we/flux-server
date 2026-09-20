# deploy_headless.ps1 - Automated Headless Colab Deployment for Windows
# Run this script from your local PowerShell to deploy the server to Colab Pro entirely in the background.

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "🚀 Hyperforge AI - Headless Colab CLI Automator" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

$colab_cmd = "$env:USERPROFILE\.local\bin\colab.exe"
$uv_cmd = "$env:USERPROFILE\.local\bin\uv.exe"

# 1. Install Google Colab CLI if missing
if (-not (Test-Path $colab_cmd)) {
    Write-Host "Google Colab CLI not found. Installing Python 3.12 and CLI via uv..." -ForegroundColor Yellow
    
    if (-not (Test-Path $uv_cmd)) {
        Write-Host "Installing uv package manager..."
        irm https://astral.sh/uv/install.ps1 | iex
    }
    
    & $uv_cmd tool install --python 3.12 git+https://github.com/googlecolab/google-colab-cli
    
    if (-not (Test-Path $colab_cmd)) {
        Write-Host "Failed to install Colab CLI." -ForegroundColor Red
        exit 1
    }
}

Write-Host "[OK] Colab CLI is ready." -ForegroundColor Green

# Authentication is handled automatically by colab new if the token is missing or expired,
# and it uses the cached ~/.colab-cli-oauth-config.json otherwise.

# 3. Provision GPU
Write-Host ""
Write-Host "Step 2: Provisioning Colab Pro GPU (A100)..." -ForegroundColor Yellow
Write-Host "Cleaning up old sessions to prevent multiple active instances..."
$sessions = & $colab_cmd sessions
foreach ($line in $sessions) {
    if ($line -match "^\[([a-zA-Z0-9]+)\]") {
        $sessionId = $matches[1]
        Write-Host "Stopping existing session $sessionId..."
        & $colab_cmd stop -s $sessionId
    }
}

Write-Host "This will consume Colab Compute Units."
try {
    & $colab_cmd new --gpu A100
} catch {
    Write-Host "Failed to provision GPU. Ensure you have Colab Compute Units available." -ForegroundColor Red
    exit 1
}

# 4. Execute Payload
Write-Host ""
Write-Host "Step 3: Preparing deployment payload..." -ForegroundColor Yellow

$HF_TOKEN = ""
if (Test-Path "flux-server\.env") {
    $envLines = Get-Content "flux-server\.env"
    foreach ($line in $envLines) {
        if ($line -match "^HF_TOKEN=(.*)") {
            $HF_TOKEN = $matches[1].Trim()
            Write-Host "Successfully loaded HF_TOKEN from flux-server\.env" -ForegroundColor Green
            break
        }
    }
}

if ([string]::IsNullOrWhiteSpace($HF_TOKEN)) {
    # Prompt the user for their HuggingFace token securely
    $HF_TOKEN = Read-Host -Prompt "Please paste your HuggingFace Token (HF_TOKEN) to download FLUX"
}
if ([string]::IsNullOrWhiteSpace($HF_TOKEN)) {
    Write-Host "HF_TOKEN is required to download the gated model. Exiting." -ForegroundColor Red
    exit 1
}

$tempPayload = "deploy_payload_temp.py"
$payloadContent = Get-Content "deploy_payload.py" -Raw
# Prepend the token injection to the script
$injectedContent = "import os`nos.environ['HF_TOKEN'] = '$HF_TOKEN'`n" + $payloadContent
Set-Content -Path $tempPayload -Value $injectedContent

Write-Host "Uploading and executing payload on Colab A100..." -ForegroundColor Yellow
& $colab_cmd exec -f $tempPayload

# Cleanup
Remove-Item -Path $tempPayload -ErrorAction SilentlyContinue
