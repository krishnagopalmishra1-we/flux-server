# Hyperforge AI

Image-generation only FastAPI server for FLUX, SD3.5, and SDXL-class text-to-image models.

The production app lives in `flux-server/` and runs synchronous image generation through `POST /generate`. Video generation, Redis-backed queues, music, animation, and multi-modal job infrastructure are intentionally not part of this codebase.

## What Runs

- FastAPI backend: `app/main.py`
- Image inference pipeline: `app/pipeline.py`
- Image model registry and lazy loading: `app/model_manager.py`
- Single GPU runtime lock: `app/runtime.py`
- Browser UI: `app/static/`
- Generated image library: `/mnt/outputs/image` plus `/mnt/outputs/library_meta.json`

## API

### Health

```bash
curl http://localhost:8080/health
```

### Models

```bash
curl http://localhost:8080/models
```

### Generate Image

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a red apple on a white table, photorealistic",
    "model_name": "flux-1-dev",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5
  }'
```

The response is synchronous and returns `image_base64`, `seed_used`, and `inference_time_ms`.

### LoRAs

```bash
curl http://localhost:8080/loras?model_name=flux-1-dev
```

Upload `.safetensors` adapters:

```bash
curl -X POST http://localhost:8080/loras/upload \
  -F "file=@my_adapter.safetensors"
```

## Current Model Registry

| Key | Model ID | Notes |
|---|---|---|
| `flux-1-dev` | `black-forest-labs/FLUX.1-dev` | Default. FP8 transformer via torchao, INT8 T5-XXL, BF16 CLIP/VAE. Tuned for A10G 24 GB. |
| `sd3.5-large` | `stabilityai/stable-diffusion-3.5-large` | NF4 transformer path on smaller GPUs; BF16 on large VRAM GPUs. |
| `realvisxl-v5` | `SG161222/RealVisXL_V5.0` | FP16 SDXL photoreal model. |
| `juggernaut-xl` | `RunDiffusion/Juggernaut-XL-v9` | FP16 SDXL general-purpose model. |

See `docs/model_audit_2026-09-19.md` for newer open model candidates and licensing notes.

## Target Hardware

The default deployment target is AWS `g5.2xlarge` with 1 x NVIDIA A10G 24 GB VRAM.

FLUX.1-dev is configured for the 24 GB card with:

- Transformer: FP8 weight-only quantization through `torchao`
- T5-XXL text encoder: INT8 through bitsandbytes
- CLIP and VAE: BF16
- Runtime override: `FLUX_QUANTIZE=fp8|nf4|bf16`

## Configuration

Copy `.env.example` to `.env` and set at least `HF_TOKEN` if you use gated Hugging Face models.

Important settings:

- `HF_TOKEN`
- `API_KEYS`
- `CACHE_DIR`
- `CACHE_DIR_SSD`
- `OUTPUT_DIR`
- `LORA_DIR`
- `FLUX_QUANTIZE`
- `OUTPUT_TTL_HOURS`

## Local Docker

```bash
cd flux-server
cp .env.example .env
docker compose up --build -d
docker compose logs -f --tail=50
```

Then open:

```text
http://localhost:8080
```

## AWS

AWS deployment helpers live in `deploy/aws/`.

```bash
cd flux-server/deploy/aws
./launch.sh
```

After SSH:

```bash
sudo /opt/flux-server/flux-server/deploy/aws/setup_disks.sh
cd /opt/flux-server/flux-server
sudo docker compose up --build -d
```

## Development Checks

```bash
python -m py_compile app/config.py app/main.py app/model_manager.py app/output_store.py app/pipeline.py app/runtime.py app/schemas.py app/security.py
python deploy/aws/smoke_test.py
git diff --check
```

## Architecture Guardrail

Keep this application image-only:

- No Redis or queue backend
- No `/api/jobs/*`
- No `/api/video/*`
- No video, music, or animation pipelines
- `/generate` remains synchronous and serialized by `gpu_runtime`
