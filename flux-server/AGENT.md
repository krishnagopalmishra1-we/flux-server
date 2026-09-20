# AGENT.md - Hyperforge AI Runtime Notes

> Last updated: 2026-07-04
> This file is the shortest reliable handoff for future agents touching `flux-server/`.

## Canonical App

Treat `flux-server/` as the production codebase. The app has been streamlined to support **image generation only** on a single A10G GPU. All video pipelines, video LoRAs, and job queueing logic (including Redis) have been completely removed.

## Production Shape

- FastAPI backend in `app/main.py`
- Image pipeline in `app/pipeline.py`
- Model manager in `app/model_manager.py` (lazy-loading, memory management)
- Shared GPU runtime coordinator in `app/runtime.py` (serializes `/generate` requests)
- Public Hyperforge AI frontend in `app/static/`

## Current Model Registry (Image Only)

| Key | Model ID | Description |
|---|---|---|
| `flux-1-dev` | `black-forest-labs/FLUX.1-dev` | FP8 transformer (torchao) + INT8 T5-XXL + CLIP/VAE BF16. Total ~18 GB VRAM. |
| `sd3.5-large` | `stabilityai/stable-diffusion-3.5-large` | NF4 transformer path. |
| `realvisxl-v5` | `SG161222/RealVisXL_V5.0` | FP16 SDXL photoreal. |
| `juggernaut-xl` | `RunDiffusion/Juggernaut-XL-v9` | FP16 SDXL versatile. |

## Runtime & Quantization Strategies

- **AWS g5.2xlarge**: 1 × NVIDIA A10G (24 GB VRAM).
- **FLUX.1-dev FP8 (torchao)**:
  - Transformer quantized to FP8 weight-only via `torchao`. Uses hardware tensor cores on Ampere architectures, accelerating inference by ~1.8× vs bitsandbytes NF4.
  - T5-XXL text encoder loaded in INT8 via bitsandbytes. Saves ~5 GB VRAM without affecting generation speed (only runs during the initial text encoding step).
  - VRAM footprint: ~18 GB total. Fits comfortably on the 24 GB A10G with ~6 GB headroom.
  - Override via environment variable: `FLUX_QUANTIZE=fp8|nf4|bf16`.

## LoRA Paths

- Configured persistent path: `LORA_DIR` (defaults to `/mnt/hf-cache/loras`).
- Uploads/lists via:
  - `GET /loras?model_name=flux-1-dev`
  - `POST /loras/upload`

## API Auth

- If `API_KEYS` is non-empty, image generation requires `X-API-Key`.
- `GET /api/auth/status` tells the frontend whether the field is required.
- For local testing, `API_KEYS` can be blank.

## Frontend Notes

- SPA UI routes: `/image`, `/library`
- Predefined image style chips.
- **Quality presets**: Draft (15st/cfg3.0) · Balanced (25st/cfg3.5) · HQ (35st/cfg5.0) · Ultra (50st/cfg7.0)
- **12 Unsplash sample images** across multiple visual categories.
- **Library tab**: persistent gallery of all generated images. Supports grid/list view, lightbox, download, delete.

## Output Store

`app/output_store.py` persists generated images to disk:
- Images land in `/mnt/outputs/image/`.
- `library_meta.json` in `/mnt/outputs/` tracks all generations with prompt, model, seed, and timestamp.
- Endpoints: `GET /api/library`, `DELETE /api/library/{id}`.

## Verification Checklist

1. `/health`
2. `/models`
3. `/api/auth/status`
4. `/loras`
5. `/generate` (Image generation)
6. Output serving
