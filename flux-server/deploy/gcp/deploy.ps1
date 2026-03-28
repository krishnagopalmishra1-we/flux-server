if (-not $env:PROJECT_ID) {
  Write-Error "Run set_env.ps1 first in the same shell."
  exit 1
}

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $here "..\..")
$repoRoot = Get-Location

Write-Host "Step 1/3: Setting up infra"
& "$here\create_infra.ps1"

Write-Host "Step 2/3: Creating VM"
& "$here\create_vm.ps1"

Write-Host "Step 3/5: Waiting for VM SSH readiness"
Start-Sleep -Seconds 30

Write-Host "Step 4/5: Copying project to VM"
gcloud compute scp --recurse "$repoRoot\\*" "$env:INSTANCE_NAME`:/opt/flux-server" --zone=$env:ZONE

Write-Host "Step 5/5: Creating .env and starting containers"
gcloud compute ssh $env:INSTANCE_NAME --zone=$env:ZONE --command "cd /opt/flux-server; cp .env.example .env; sed -i 's|# CACHE_DIR=/mnt/hf-cache|CACHE_DIR=/mnt/hf-cache|g' .env; sed -i 's|# MODEL_ID=black-forest-labs/FLUX.1-dev|MODEL_ID=black-forest-labs/FLUX.1-dev|g' .env; echo 'PORT=8080' >> .env; echo 'WORKERS=1' >> .env; docker compose up -d --build"

gcloud compute instances list --filter="name=$env:INSTANCE_NAME" --zones=$env:ZONE

Write-Host "Deployment complete. SSH and add HF_TOKEN to /opt/flux-server/.env, then run: docker compose restart"
