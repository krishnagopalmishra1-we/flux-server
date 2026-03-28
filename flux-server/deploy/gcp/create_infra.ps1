if (-not $env:PROJECT_ID) {
  Write-Error "Run set_env.ps1 first in the same shell."
  exit 1
}

gcloud config set project $env:PROJECT_ID

gcloud services enable compute.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com logging.googleapis.com monitoring.googleapis.com

gcloud compute firewall-rules create flux-allow-http --allow tcp:80,tcp:443,tcp:8080 --target-tags=$env:NETWORK_TAG --source-ranges=0.0.0.0/0 2>$null

gcloud compute firewall-rules create flux-allow-health --allow tcp:8000 --target-tags=$env:NETWORK_TAG --source-ranges=0.0.0.0/0 2>$null

Write-Host "Infra setup complete."
