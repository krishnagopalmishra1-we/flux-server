# deploy_headless.ps1 - Automated Headless Colab Deployment for Windows
# Run this script from your local PowerShell to deploy the server to Colab Pro entirely in the background.

Write-Host "=================================================" -ForegroundColor Cyan
Write-Host "🚀 Hyperforge AI - Headless Colab CLI Automator" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Install Google Colab CLI if missing
if (-not (Get-Command "colab" -ErrorAction SilentlyContinue)) {
    Write-Host "Google Colab CLI not found. Installing via pip..." -ForegroundColor Yellow
    pip install git+https://github.com/googlecolab/google-colab-cli
    
    if (-not (Get-Command "colab" -ErrorAction SilentlyContinue)) {
        Write-Host "Failed to install Colab CLI. Ensure Python/pip is in your PATH." -ForegroundColor Red
        exit 1
    }
}

Write-Host "[OK] Colab CLI is ready." -ForegroundColor Green

# 2. Authenticate
Write-Host ""
Write-Host "Step 1: Checking Authentication..." -ForegroundColor Yellow
Write-Host "If a browser window opens, please log into the Google Account associated with your Colab Pro."
try {
    colab auth login
} catch {
    Write-Host "Authentication failed or was cancelled." -ForegroundColor Red
    exit 1
}

# 3. Provision GPU
Write-Host ""
Write-Host "Step 2: Provisioning Colab Pro GPU (A100)..." -ForegroundColor Yellow
Write-Host "This will consume Colab Compute Units."
# We use --gpu A100 (or L4). If A100 is unavailable, Colab will automatically attempt fallback if configured.
try {
    colab new --gpu A100
} catch {
    Write-Host "Failed to provision GPU. Ensure you have Colab Compute Units available." -ForegroundColor Red
    exit 1
}

# 4. Execute Payload
Write-Host ""
Write-Host "Step 3: Uploading and executing deployment payload..." -ForegroundColor Yellow
# Send the local deploy_payload.sh to the Colab environment and run it
colab exec -f deploy_payload.sh
