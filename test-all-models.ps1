#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Comprehensive model testing script for Neural Creation Studio
.DESCRIPTION
    Tests every generation model across all modalities (image, video, music, animation)
#>

param(
    [string]$ServerURL = "http://35.239.234.106:8080",
    [int]$TimeoutSec = 300
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Status {
    param([string]$Message, [string]$Status = "INFO")
    $color = @{
        "PASS" = "Green"
        "FAIL" = "Red"
        "INFO" = "Cyan"
        "WARN" = "Yellow"
    }[$Status]
    Write-Host "[$Status] $Message" -ForegroundColor $color
}

function Test-Endpoint {
    param([string]$Method, [string]$Endpoint, [object]$Body)
    
    Write-Status "Testing $Method $Endpoint..." "INFO"
    try {
        $params = @{
            Uri     = "$ServerURL$Endpoint"
            Method  = $Method
            Timeout = $TimeoutSec
        }
        if ($Body) {
            $params["Body"] = ($Body | ConvertTo-Json)
            $params["ContentType"] = "application/json"
        }
        
        $response = Invoke-RestMethod @params
        Write-Status "$Endpoint responded" "PASS"
        return $response
    } catch {
        Write-Status "$Endpoint failed: $_" "FAIL"
        return $null
    }
}

Write-Host "
╔═══════════════════════════════════════════════╗
║  Neural Creation Studio – Model Test Suite   ║
╚═══════════════════════════════════════════════╝
" -ForegroundColor Cyan

# Test 1: Health Check
Write-Status "═══ HEALTH CHECK ═══" "INFO"
$health = Test-Endpoint "GET" "/health"
if ($health) {
    Write-Status "GPU: $($health.gpu_name) | VRAM: $($health.vram_used_gb)/$($health.vram_total_gb) GB" "PASS"
}
Write-Host ""

# Test 2: List Available Models
Write-Status "═══ AVAILABLE MODELS ═══" "INFO"
$models = Test-Endpoint "GET" "/models"
if ($models) {
    foreach ($cat in $models.categories) {
        $catModels = $models.models | Where-Object { $_.category -eq $cat }
        Write-Host "  $cat: $($catModels.Count) models" -ForegroundColor Cyan
        $catModels | ForEach-Object { Write-Host "    - $($_.name)" }
    }
}
Write-Host ""

# Test 3: IMAGE GENERATION
Write-Status "═══ IMAGE GENERATION ═══" "INFO"
$imageModels = @("flux-1-dev", "sd3-medium", "sdxl")
foreach ($model in $imageModels) {
    Write-Status "Testing image model: $model" "INFO"
    $imageReq = @{
        prompt               = "a beautiful landscape"
        model_name           = $model
        width                = 1024
        height               = 1024
        num_inference_steps  = 20
        guidance_scale       = 3.5
        seed                 = 42
    }
    $result = Test-Endpoint "POST" "/generate-ui" $imageReq
    if ($result -and $result.image_base64) {
        Write-Status "$model: Generated ${$($result.image_base64.Length / 1024 / 1024).ToString('F2')}MB image in $($result.inference_time_ms)ms" "PASS"
    }
}
Write-Host ""

# Test 4: VIDEO GENERATION
Write-Status "═══ VIDEO GENERATION ═══" "INFO"
$videoModels = @("ltx-video", "wan-t2v-1.3b")
foreach ($model in $videoModels) {
    Write-Status "Testing video model: $model" "INFO"
    $videoReq = @{
        prompt              = "a sunset over mountains"
        model_name          = $model
        resolution          = "480p"
        num_frames          = 16
        fps                 = 16
        guidance_scale      = 5.0
        num_inference_steps = 20
        seed                = 42
    }
    $result = Test-Endpoint "POST" "/api/video/generate" $videoReq
    if ($result -and $result.job_id) {
        Write-Status "$model: Job submitted ($($result.job_id)) queue position: $($result.queue_position)" "PASS"
        
        # Wait for completion with polling
        $maxWait = 120
        $elapsed = 0
        while ($elapsed -lt $maxWait) {
            Start-Sleep -Seconds 5
            $jobStatus = Test-Endpoint "GET" "/api/jobs/$($result.job_id)"
            if ($jobStatus.status -eq "completed") {
                Write-Status "$model: Video generated in $($jobStatus.processing_time_ms)ms" "PASS"
                break
            } elseif ($jobStatus.status -eq "failed") {
                Write-Status "$model: Job failed - $($jobStatus.error_message)" "FAIL"
                break
            } else {
                Write-Status "$model: Processing... ($($jobStatus.progress)%)" "INFO"
            }
            $elapsed += 5
        }
    }
}
Write-Host ""

# Test 5: MUSIC GENERATION
Write-Status "═══ MUSIC GENERATION ═══" "INFO"
$musicModels = @("audioldm2")
foreach ($model in $musicModels) {
    Write-Status "Testing music model: $model" "INFO"
    $musicReq = @{
        prompt           = "peaceful ambient music"
        model_name       = $model
        duration_seconds = 10
        seed             = 42
    }
    $result = Test-Endpoint "POST" "/api/music/generate" $musicReq
    if ($result -and $result.job_id) {
        Write-Status "$model: Job submitted ($($result.job_id))" "PASS"
    }
}
Write-Host ""

# Test 6: QUEUE STATUS
Write-Status "═══ QUEUE STATUS ═══" "INFO"
$queueStatus = Test-Endpoint "GET" "/api/queue/status"
if ($queueStatus) {
    Write-Status "Queued: $($queueStatus.queued) | Processing: $($queueStatus.processing) | Done: $($queueStatus.completed) | Failed: $($queueStatus.failed)" "PASS"
}
Write-Host ""

Write-Status "═════════════════════════════════════════" "INFO"
Write-Status "Testing complete! Check results above." "INFO"
