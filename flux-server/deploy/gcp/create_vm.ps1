if (-not $env:PROJECT_ID) {
  Write-Error "Run set_env.ps1 first in the same shell."
  exit 1
}

gcloud compute instances create $env:INSTANCE_NAME `
  --project=$env:PROJECT_ID `
  --zone=$env:ZONE `
  --machine-type=$env:MACHINE_TYPE `
  --maintenance-policy=TERMINATE `
  --provisioning-model=SPOT `
  --instance-termination-action=STOP `
  --image-family=$env:IMAGE_FAMILY `
  --image-project=$env:IMAGE_PROJECT `
  --boot-disk-size=${env:BOOT_DISK_GB}GB `
  --boot-disk-type=pd-ssd `
  --tags=$env:NETWORK_TAG `
  --metadata-from-file=startup-script=deploy/gcp/startup.sh

Write-Host "VM creation requested: $env:INSTANCE_NAME"
