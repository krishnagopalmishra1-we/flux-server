#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deployment script for Neural Creation Studio to GCP A100 VM
.DESCRIPTION
    Deploys the latest code, rebuilds Docker image, and starts the server
    Run this from d:\Flux_Lora\flux-server directory
#>

Set-Location "$PSScriptRoot/.."
$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "Deploying from: $ScriptPath" -ForegroundColor Cyan

# Step 1: Set GCP environment
Write-Host "`n[DEPLOY] Step 1: Loading GCP environment..." -ForegroundColor Yellow
& "$ScriptPath\deploy\gcp\set_env.ps1"

# Step 2: Verify VM is running
Write-Host "`n[DEPLOY] Step 2: Checking VM status..." -ForegroundColor Yellow
$vmStatus = gcloud compute instances describe $env:INSTANCE_NAME --zone=$env:ZONE --format="value(status)" 2>&1
if ($vmStatus -like "*ERROR*" -or $vmStatus -ne "RUNNING") {
    Write-Host "[DEPLOY] WARNING: VM may be down. Starting it..." -ForegroundColor Yellow
    gcloud compute instances start $env:INSTANCE_NAME --zone=$env:ZONE 2>&1
    Start-Sleep -Seconds 10
}

# Step 3: Copy updated app code
Write-Host "`n[DEPLOY] Step 3: Uploading app code..." -ForegroundColor Yellow
gcloud compute scp --recurse "app" "flux-a100-preemptible:/opt/flux-server" --zone="us-central1-a" 2>&1 | Select-Object -Last 20
if ($LASTEXITCODE -ne 0) {
    Write-Host "[DEPLOY] ERROR: SCP failed. Check network and VM status." -ForegroundColor Red
    exit 1
}

# Step 4: Rebuild and restart container
Write-Host "`n[DEPLOY] Step 4: Rebuilding Docker image and restarting..." -ForegroundColor Yellow
$restartCmd = @"
cd /opt/flux-server && \
sudo docker compose down 2>&1 && \
sudo docker compose up -d --build 2>&1 && \
sleep 5 && \
docker compose logs --tail=20
"@
gcloud compute ssh $env:INSTANCE_NAME --zone=$env:ZONE --command "$restartCmd" 2>&1 | Select-Object -Last 30

# Step 5: Wait for server to be ready
Write-Host "`n[DEPLOY] Step 5: Waiting for server startup (~30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

# Step 6: Check health
Write-Host "`n[DEPLOY] Step 6: Verifying server health..." -ForegroundColor Yellow
$ip = gcloud compute instances describe $env:INSTANCE_NAME --zone=$env:ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>&1
Write-Host "Server IP: $ip" -ForegroundColor Cyan

for ($attempt = 0; $attempt -lt 6; $attempt++) {
    try {
        $health = Invoke-RestMethod "http://${ip}:8080/health" -TimeoutSec 5 -ErrorAction Stop
        Write-Host "[DEPLOY] ✓ Server is healthy!" -ForegroundColor Green
        Write-Host "  GPU: $($health.gpu_name) | VRAM: $($health.vram_total_gb) GB" -ForegroundColor Green
        Write-Host "  Model Loaded: $($health.model_loaded)" -ForegroundColor Green
        exit 0
    } catch {
        Write-Host "[DEPLOY] Server not responding yet... waiting ($($attempt+1)/6)" -ForegroundColor Yellow
        Start-Sleep -Seconds 10
    }
}

Write-Host "[DEPLOY] ERROR: Server failed to start. Check logs on VM." -ForegroundColor Red
Write-Host "  Run: gcloud compute ssh flux-a100-preemptible --zone=us-central1-a --command='docker compose logs --tail=50'" -ForegroundColor Yellow
exit 1
