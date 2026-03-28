param(
  [Parameter(Mandatory = $true)] [string]$LocalDatasetDir,
  [Parameter(Mandatory = $true)] [string]$Bucket,
  [Parameter(Mandatory = $false)] [string]$Prefix = "datasets"
)

if (-not (Test-Path $LocalDatasetDir)) {
  Write-Error "Directory not found: $LocalDatasetDir"
  exit 1
}

$target = "gs://$Bucket/$Prefix/"
Write-Host "Uploading $LocalDatasetDir to $target"
gsutil -m rsync -r $LocalDatasetDir $target
Write-Host "Upload complete"
