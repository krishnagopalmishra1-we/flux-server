# AGENT.md - Hyperforge AI Runtime Notes

> Last updated: 2026-04-23
> This file is the shortest reliable handoff for future agents touching `flux-server/`.

## Canonical App

Treat `flux-server/` as the production codebase. Root-level `app/` and older root Docker assets are legacy unless the user explicitly asks to revive them.

## Production Shape

- FastAPI backend in `app/main.py`
- Image pipeline in `app/pipeline.py`
- Video pipeline in `app/pipelines/video_pipeline.py`
- Shared GPU runtime coordinator in `app/runtime.py`
- In-memory jobs and SSE in `app/job_queue.py`
- Public Hyperforge AI frontend in `app/static/`

## Current Model Registry

### Image

| Key | Model ID | Notes |
|---|---|---|
| `flux-1-dev` | `black-forest-labs/FLUX.1-dev` | Runs in BF16 on A100. NF4 path was removed for stability after bitsandbytes load failures. |
| `sd3.5-large` | `stabilityai/stable-diffusion-3.5-large` | NF4 transformer path. |
| `realvisxl-v5` | `SG161222/RealVisXL_V5.0` | FP16 SDXL photoreal. |
| `juggernaut-xl` | `RunDiffusion/Juggernaut-XL-v9` | FP16 SDXL general-purpose. |

### Video

| Key | Model ID | Notes |
|---|---|---|
| `wan-t2v-1.3b` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Default video model and safest public default. |
| `wan-t2v-14b` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | NF4 dual-transformer path, stricter A100 limits. |
| `wan-i2v-14b` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | NF4 image-to-video path. |
| `hunyuan-video` | `hunyuanvideo-community/HunyuanVideo` | NF4 transformer, CPU-offloaded text encoder. |

Removed from the active surface:

- Music generation
- Animation generation
- Broken legacy tabs wired to non-existent backend endpoints

## Hard Runtime Constraints

- One A100 40GB GPU
- Limited disk
- No extra GPU
- No extra storage
- Keep only one heavy model resident in VRAM at a time

The code now reflects this:

- Image and video work serialize through a shared GPU coordinator.
- Video validation rejects settings outside per-model A100-safe limits.
- Disk-budget checks run before large LoRA writes.
- Output cleanup is part of normal runtime hygiene.

## LoRA Paths

Do not rely on relative `loras/` folders anymore. Use configured persistent paths:

- `LORA_DIR`
- `VIDEO_LORA_DIR`

Useful diagnostics:

- `GET /api/loras/diagnostics`
- `GET /loras?model_name=flux-1-dev`
- `GET /api/video/loras`

If uploaded LoRAs are "missing", the first thing to check is whether the VM bind mount and configured directory still point at the same persistent disk path.

## API Auth

- If `API_KEYS` is non-empty, image generation requires `X-API-Key`.
- `GET /api/auth/status` tells the frontend whether the field is required.
- For local testing, `API_KEYS` can be blank.

## Frontend Notes

The current UI is public-facing Hyperforge AI:

- Routes: `/image`, `/video`, `/queue`
- Predefined image and video style chips
- Sample prompt actions
- Internal metrics like VRAM are hidden from the public UI

Recent bug fixes:

- Restored prompt focus after rerenders
- Reduced polling-driven rerenders to avoid constant flicker
- Removed visible mojibake from user-facing strings
- Queue navigation now changes the URL instead of only switching internal state

## Files Worth Reading First

- `app/main.py`
- `app/runtime.py`
- `app/model_manager.py`
- `app/pipeline.py`
- `app/pipelines/video_pipeline.py`
- `app/static/app.js`
- `app/static/style.css`

## Safe Defaults To Preserve

### Image

- Default model: `flux-1-dev`
- Keep FLUX on BF16 unless there is a verified replacement for the failed NF4 image path

### Video

- Default model: `wan-t2v-1.3b`
- Default resolution: `480p`
- Default frames: `33`
- Default steps: `30`

These defaults are there to stay inside one A100 40GB plus limited disk, not because they are the theoretical best quality.

## Deployment Context

Recent production deployment target:

- Project: `flux-lora-gpu-project`
- Zone: `us-central1-a`
- Instance: `flux-a100-preemptible`
- App dir on VM: `/opt/flux-server`

## Verification Checklist

Before calling a change done, verify:

1. `/health`
2. `/models`
3. `/api/auth/status`
4. `/loras`
5. `/api/video/loras`
6. Image generation
7. Video generation
8. Queue polling and SSE
9. Cancellation
10. Output serving

## Things Not To Reintroduce Quietly

- Separate image and video GPU locks
- Public UI tabs for unsupported music or animation flows
- Relative LoRA directories as the only storage path
- Public-facing GPU diagnostics as core UI content
