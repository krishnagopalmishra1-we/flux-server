# NEURAL CREATION STUDIO — DIAGNOSTIC & RECOVERY GUIDE

## Current Status: DEPLOYMENT ISSUE
- **Server**: Unreachable at 35.239.234.106:8080
- **Last Known State**: Container restarted after code push
- **Likely Cause**: VM may be down, or container crashed during startup

---

## IMMEDIATE RECOVERY STEPS

### Step 1: Check VM Status (Windows PowerShell)
```powershell
Set-Location d:\Flux_Lora\flux-server

# Configure gcloud environment
. .\deploy\gcp\set_env.ps1

# Check if VM is running
gcloud compute instances describe $env:INSTANCE_NAME --zone=$env:ZONE --format="value(status)"

# If TERMINATED, start it:
gcloud compute instances start $env:INSTANCE_NAME --zone=$env:ZONE
Start-Sleep -Seconds 30
```

### Step 2: Check Container Status
```powershell
# SSH into VM and check Docker status
gcloud compute ssh $env:INSTANCE_NAME --zone=$env:ZONE --command="sudo docker compose logs --tail=100"

# If container is in error state, restart it:
gcloud compute ssh $env:INSTANCE_NAME --zone=$env:ZONE --command="cd /opt/flux-server && sudo docker compose down && sudo docker compose up -d --build"
```

### Step 3: Automated Deployment (Once Network Is Stable)
```powershell
# From d:\Flux_Lora directory:
.\deploy-now.ps1
```

---

## CODE CHANGES APPLIED

### Disabled Features (Temporary Debug Mode)
- **Wan 2.2 Background Pre-load**: DISABLED to improve startup reliability
  - Was timing out server, preventing health checks
  - Will be re-enabled after core models are verified working
  - No impact on functionality — models load on-demand

### Active Fixes
1. **Image Generation**: Non-blocking via `asyncio.to_thread()` + `_gpu_lock`
2. **GPU Serialization**: `_gpu_lock` prevents concurrent model loads (OOM protection)
3. **Video LoRA**: Schema support added, implementation temporarily disabled
4. **Error Handling**: Proper HTTPException on image/video endpoints
5. **CORS**: Fixed to include DELETE method for job cancellation

---

## TESTING ONCE SERVER IS BACK UP

### Automated Model Testing
```powershell
# From d:\Flux_Lora directory:
.\test-all-models.ps1

# This will test:
# - Image models: flux-1-dev, sd3-medium, sdxl
# - Video models: ltx-video, wan-t2v-1.3b
# - Music models: audioldm2
# - Queue system: status and polling
```

### Manual Testing by Model

#### Image Generation
```powershell
$ip = "35.239.234.106"
$payload = @{
    prompt = "a beautiful sunset"
    model_name = "flux-1-dev"
    width = 1024
    height = 1024
    num_inference_steps = 28
} | ConvertTo-Json

Invoke-RestMethod "http://${ip}:8080/generate-ui" -Method POST -Body $payload -ContentType "application/json"
```

#### Video Generation (LTX - fastest)
```powershell
$payload = @{
    prompt = "a cat running"
    model_name = "ltx-video"
    resolution = "480p"
    num_frames = 16
    num_inference_steps = 20
} | ConvertTo-Json

$job = Invoke-RestMethod "http://${ip}:8080/api/video/generate" -Method POST -Body $payload -ContentType "application/json"
Write-Host "Job ID: $($job.job_id)"

# Poll for completion:
1..30 | ForEach-Object {
    Start-Sleep -Seconds 2
    $status = Invoke-RestMethod "http://${ip}:8080/api/jobs/$($job.job_id)"
    Write-Host "Status: $($status.status) - Progress: $($status.progress)%"
    if ($status.status -in @("completed", "failed")) { break }
}
```

#### Music Generation
```powershell
$payload = @{
    prompt = "ambient electronic music"
    model_name = "audioldm2"
    duration_seconds = 10
} | ConvertTo-Json

$job = Invoke-RestMethod "http://${ip}:8080/api/music/generate" -Method POST -Body $payload -ContentType "application/json"
Write-Host "Job submitted: $($job.job_id)"
```

---

## KEY FILES MODIFIED IN THIS SESSION

| File | Change | Impact |
|------|--------|--------|
| main.py | Disabled Wan pre-load (line ~128) | Server startup faster, no background model loading |
| main.py | Image endpoints use `asyncio.to_thread` + `_gpu_lock` | Non-blocking image generation |
| schemas_v2.py | Added LoRA fields to VideoGenerateRequest | Ready for video LoRA (feature deferred) |
| video_pipeline.py | Added `_apply_lora()` + `_unload_lora()` (currently no-op) | Video LoRA support prepared but disabled |
| app.js | Added video LoRA UI fields | Frontend ready for video LoRA selection |
| deploy.ps1 | Added `loras/` dir creation | Ensures LoRA directory exists on VM |

---

## KNOWN ISSUES & WORKAROUNDS

### Issue: Wan 2.2 Download Timeout
- **Symptom**: `/api/video/generate` with `wan-t2v-14b` times out after 30+ min
- **Cause**: 14B model is 50GB+, single worker blocks gunicorn timeout
- **Workaround**: Use `ltx-video` or `wan-t2v-1.3b` for testing; Wan 14B pre-downloads to cache

### Issue: ACE-Step Incompatibility
- **Symptom**: Music with ACE-Step returns model loading error
- **Cause**: Upstream transformers/huggingface library mismatch
- **Workaround**: Use `audioldm2` or `musicgen` instead

### Issue: Stable Audio 403 Error
- **Symptom**: Music with Stable Audio returns 403 Forbidden
- **Cause**: User hasn't accepted HF terms for gated model
- **Fix**: Accept terms at https://huggingface.co/stabilityai/stable-audio-open-1.0 and set `HF_TOKEN` in .env

---

## RE-ENABLING FEATURES AFTER TESTING

### To Re-enable Wan 2.2 Pre-load (once core models work):
Edit `main.py` line ~128, replace:
```python
async def _preload_video_model_background():
    # TEMPORARILY DISABLED to debug server responsiveness issues
    logger.info("Video pre-load: disabled (debug mode)")
    return
```

With:
```python
async def _preload_video_model_background():
    """Background task to pre-load Wan 2.2 model while server handles requests."""
    await asyncio.sleep(2)  # Let server stabilize first
    try:
        async with _gpu_lock:
            logger.info("Background pre-load: loading Wan 2.2 model...")
            await asyncio.to_thread(video_pipeline.load_model, "wan-t2v-1.3b")
            logger.info("Wan 2.2 model pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Background pre-load failed (non-critical): {e}")
```

### To Enable Video LoRA (once Wan supports it):
Edit `video_pipeline.py` lines ~245-280 to restore full LoRA implementation:
1. Check if diffusers `WanPipeline` supports `load_lora_weights`
2. Uncomment full `_apply_lora()` method
3. Test with small LoRA file on video generation

---

##  NEXT STEPS FOR FULL STABILITY

1. **Get server online**: Execute Step 1-3 of recovery guide above
2. **Test all models**: Run `.\test-all-models.ps1` to identify remaining issues
3. **Fix per-model issues**: Address any model-specific errors from step 2
4. **Re-enable Wan pre-load**: Once core models stable, re-enable background loading
5. **Implement robust error recovery**: Add automatic fallback models
6. **Monitor logs**: Keep checking server logs for VRAM OOM or timeout issues

---

## SUPPORT COMMAND REFERENCE

```powershell
# Check server logs (last 50 lines)
gcloud compute ssh flux-a100-preemptible --zone=us-central1-a --command="docker compose logs --tail=50"

# View queue status
Invoke-RestMethod "http://35.239.234.106:8080/api/queue/status" | ConvertTo-Json

# Cancel a stalled job
Invoke-RestMethod "http://35.239.234.106:8080/api/jobs/{job_id}" -Method DELETE

# Check available models
Invoke-RestMethod "http://35.239.234.106:8080/models" | ConvertTo-Json

# Get current VM IP (in case it changed)
gcloud compute instances describe flux-a100-preemptible --zone=us-central1-a --format="value(networkInterfaces[0].accessConfigs[0].natIP)"
```

---

**Last Updated**: April 3, 2026  
**Status**: Awaiting Network/VM Recovery  
**Next Action**: Execute recovery steps above
