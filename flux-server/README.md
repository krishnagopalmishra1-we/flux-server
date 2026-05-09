# Hyperforge AI

> Bright, public-facing image and video studio built on FastAPI, FLUX, Wan, HunyuanVideo, Redis, and optional xDiT multi-GPU WAN 14B inference.

## What Is Running

`flux-server/` is the active production app. It serves:

- Image generation at `POST /generate`
- Video generation at `POST /api/video/generate`
- Redis-backed job polling, queue state, cancellation, and SSE/poll progress under `/api/jobs/*`
- LoRA listing and upload endpoints for image and video adapters
- The Hyperforge AI web UI at `/`, `/image`, `/video`, `/queue`, and `/library`

The service supports two runtime shapes:

- Single-worker A100 testing, using the in-memory queue if `JOB_BACKEND=memory`
- Multi-worker 8x80GB testing, using `JOB_BACKEND=redis`, `GPUS_PER_JOB=4`, and two Gunicorn workers for two concurrent WAN 14B xDiT jobs

## Current Model Set

### Image

| Key | Backend model | Runtime notes |
|---|---|---|
| `flux-1-dev` | `black-forest-labs/FLUX.1-dev` | BF16 path on A100. This replaced the broken NF4 image path. |
| `sd3.5-large` | `stabilityai/stable-diffusion-3.5-large` | NF4 transformer path. |
| `realvisxl-v5` | `SG161222/RealVisXL_V5.0` | FP16 SDXL photoreal model. |
| `juggernaut-xl` | `RunDiffusion/Juggernaut-XL-v9` | FP16 SDXL general-purpose model. |

### Video

| Key | Backend model | Runtime notes |
|---|---|---|
| `wan-t2v-1.3b` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Default video model. Fastest, safest option on A100. Default output is `480p`, short clips. |
| `wan-t2v-14b` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | Prefers BF16 xDiT on 4x80GB GPU groups, falls back to NF4 diffusers elsewhere. |
| `wan-i2v-14b` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | NF4 image-to-video path. Requires `source_image_b64`. |
| `hunyuan-video` | `hunyuanvideo-community/HunyuanVideo` | 720p-capable path with NF4 transformer and CPU-offloaded text encoder. |

## Runtime Safety Rules

- Image and video pipelines share a single GPU runtime coordinator.
- Image and video generations do not run concurrently on GPU.
- Video validation is model-aware and rejects settings that are unsafe for A100 40GB.
- In Redis mode, queued jobs are claimed through one Redis sorted set so multiple workers cannot claim the same job.
- Processing job state, progress, ETA, cancellation flags, and terminal results are persisted to Redis.
- xDiT subprocesses use per-job localhost ports, process-group cleanup, a four-hour default timeout, and temp-directory cleanup.
- Long videos stream frame output to disk instead of retaining all encoded frames in RAM.
- Output files, temp files, uploads, and model cache usage are guarded for limited disk environments.

## LoRA Storage

Image and video LoRAs now use explicit persistent directories from config instead of fragile relative paths.

- `LORA_DIR=/mnt/hf-cache/loras`
- `VIDEO_LORA_DIR=/mnt/hf-cache/video_loras`
- `MAX_LORA_UPLOAD_MB=1536`

Useful endpoints:

- `GET /loras?model_name=flux-1-dev`
- `GET /api/video/loras`
- `GET /api/loras/diagnostics`
- `POST /loras/upload`
- `POST /api/video/loras/upload`

If LoRAs appear missing, check the diagnostics endpoint first and verify the mounted host path contains the uploaded `.safetensors` files.

## API Auth

Image generation can be protected with `API_KEYS`. If `API_KEYS` is set, requests must send `X-API-Key`. For local testing, leave `API_KEYS` blank.

Check current mode with:

```bash
curl http://localhost:8080/api/auth/status
```

## Quick Start

```bash
git clone https://github.com/krishnagopalmishra1-we/flux-server.git
cd flux-server/flux-server
cp .env.example .env
docker compose up --build -d
```

Minimum required env:

```env
HF_TOKEN=hf_your_token_here
API_KEYS=
LORA_DIR=/mnt/hf-cache/loras
VIDEO_LORA_DIR=/mnt/hf-cache/video_loras
WAN_DEFAULT_VARIANT=1.3b
JOB_BACKEND=redis
REDIS_URL=redis://redis:6379/0
```

## Vultr Deploy Branch

Vultr bootstrap deploys `DEPLOY_BRANCH`, defaulting to `codex/hyperforge-runtime-hardening-impl` until these runtime changes are merged to `main`. Override it when launching if needed:

```bash
DEPLOY_BRANCH=main ./deploy/vultr/launch.sh
```

Then verify:

```bash
curl http://localhost:8080/health
curl http://localhost:8080/models
curl http://localhost:8080/api/auth/status
```

## Public UI Notes

The web UI is branded as Hyperforge AI and is intentionally public-facing:

- No internal GPU or VRAM details in normal UI flows
- Routes: `/image`, `/video`, `/queue`, `/library`
- **Quality presets** on the image tab: Draft · Balanced · HQ · Ultra (one-click steps + CFG)
- **15s video preset** added alongside 1s, 2s, 3s, 5s
- **12 Unsplash sample images** across landscape, portrait, abstract, sci-fi, fashion categories
- **Library tab** — persistent grid/list gallery of all generated images and videos with filtering (All / Image / Video), download, delete, and lightbox preview
- Tab transition animations (fade + slide), shimmer skeleton loader
- Predefined image and video styles

Internal health details remain available through API endpoints for debugging.

## Core Endpoints

### Image

`POST /generate`

Main fields:

- `prompt`
- `model_name`
- `negative_prompt`
- `width`
- `height`
- `num_inference_steps`
- `guidance_scale`
- `seed`
- `lora_name`
- `lora_scale`

### Video

`POST /api/video/generate`

Main fields:

- `prompt`
- `model_name`
- `negative_prompt`
- `num_frames`
- `fps`
- `resolution`
- `num_inference_steps`
- `guidance_scale`
- `seed`
- `source_image_b64`
- `lora_name`
- `lora_scale`
- `chunk_size`
- `chunk_overlap`

Current safe defaults:

- `model_name=wan-t2v-1.3b`
- `resolution=480p`
- `num_frames=33`
- `num_inference_steps=30`

### Jobs and Queue

- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/stream`
- `DELETE /api/jobs/{job_id}`
- `GET /api/queue/status`

### Library

- `GET /api/library?type=all|image|video&limit=200` — paginated list of all saved generations, newest first
- `DELETE /api/library/{id}` — delete an item and its file from disk

Generated images are automatically saved to `outputs/image/` and indexed in `outputs/library_meta.json`. Videos were already persisted; they are now cross-referenced in the same index so both types appear in the Library tab.

## Repository Notes

- `flux-server/` is canonical.
- Root `app/`, root Docker assets, and older notes are legacy unless explicitly revived.
- Music and animation tabs were removed from the active UI because the matching backend flows do not exist.

## Deployment

The GCP deployment scripts live under `deploy/gcp/`. The production VM used during recent testing was:

- Project: `flux-lora-gpu-project`
- Zone: `us-central1-a`
- Instance: `flux-a100-preemptible`

## Development Sanity Checks

```bash
python -m py_compile app\config.py app\job_queue.py app\main.py app\model_manager.py app\output_store.py app\pipeline.py app\runtime.py app\schemas.py app\pipelines\video_pipeline.py
git diff --check
```
