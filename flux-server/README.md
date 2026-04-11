# Neural Creation Studio

> **High-performance AI generation platform for images and videos.**
> Powered by FLUX.1, Stable Diffusion 3.5, Wan 2.2, and HunyuanVideo on a single A100 GPU.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

### Image Generation
- **FLUX.1-dev** — State-of-the-art 12B parameter model with NF4 quantization
- **SD3.5-Large** — Flexible multi-modal prompt adhesion
- **RealVisXL V5** — Photorealistic portraits and product photography
- LoRA support with runtime upload

### Video Generation
- **Wan 2.2 T2V** — Text-to-video in 1.3B (fast) and 14B (cinematic quality) variants
- **Wan 2.2 I2V** — Image-to-video animation from a source image
- **HunyuanVideo** — 720p NF4-quantized video generation
- **Chunked long-video** — Generate 1-2 minute videos via sliding window with temporal blending
- LoRA support for Wan models

### Platform
- **Async job queue** with real-time SSE progress streaming
- **Job cancellation** for both queued and running jobs
- **ETA estimation** based on inference progress rate
- **Tiered OOM recovery** — reduce steps → reduce frames → CPU offload
- **Disk space guard** and automatic output cleanup
- **Admin API** for queue management
- **Built-in web UI** at `http://localhost:8080/`

---

## 🚀 Quick Start

### Prerequisites
- NVIDIA GPU with ≥24GB VRAM (A100 40GB recommended)
- Docker with NVIDIA Container Toolkit
- Hugging Face token ([get one free](https://huggingface.co/settings/tokens))

### 1. Clone and configure

```bash
git clone https://github.com/krishnagopalmishra1-we/flux-server.git
cd flux-server
cp .env.example .env
```

Edit `.env` and set your `HF_TOKEN`:
```env
HF_TOKEN=hf_your_token_here
API_KEYS=your-secret-key
ADMIN_API_KEY=your-admin-key
```

### 2. Launch

```bash
docker compose up --build -d
```

The server starts at `http://localhost:8080`. First model load takes 5-15 minutes (downloading weights).

### 3. Verify

```bash
curl http://localhost:8080/health
```

---

## 📡 API Reference

### Image Generation

```bash
POST /generate
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text prompt (1-2000 chars) |
| `model_name` | string | `flux-1-dev` | Model: `flux-1-dev`, `sd3.5-large`, `realvisxl-v5` |
| `negative_prompt` | string | `null` | Negative prompt |
| `width` | int | 1024 | Image width (256-2048, multiple of 8) |
| `height` | int | 1024 | Image height (256-2048, multiple of 8) |
| `num_inference_steps` | int | 28 | Denoising steps (1-50) |
| `guidance_scale` | float | 3.5 | Classifier-free guidance (0-20) |
| `seed` | int | `null` | Random seed (null = random) |
| `lora_name` | string | `null` | LoRA adapter name |
| `lora_scale` | float | 1.0 | LoRA strength (0-2) |

**Response:** JSON with `image_base64`, `seed_used`, `inference_time_ms`

### Video Generation

```bash
POST /api/video/generate
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | *required* | Text prompt |
| `model_name` | string | `wan-t2v-1.3b` | Model: `wan-t2v-1.3b`, `wan-t2v-14b`, `wan-i2v-14b`, `hunyuan-video` |
| `num_frames` | int | 33 | Frame count (16-1920) |
| `fps` | int | 16 | Frames per second (8-30) |
| `resolution` | string | `480p` | `480p` or `720p` |
| `num_inference_steps` | int | 30 | Denoising steps (10-100) |
| `guidance_scale` | float | 5.0 | CFG scale (0-20) |
| `source_image_b64` | string | `null` | Base64 image for I2V |
| `chunk_size` | int | 81 | Frames per chunk for long videos |
| `chunk_overlap` | int | 20 | Overlap frames between chunks |

**Response:** JSON with `job_id` — poll status via `/api/jobs/{job_id}`

### Job Management

```bash
GET  /api/jobs/{job_id}           # Poll job status + ETA
GET  /api/jobs/{job_id}/stream    # SSE real-time progress
DELETE /api/jobs/{job_id}         # Cancel queued or running job
GET  /api/jobs                    # List recent jobs
GET  /api/queue/status            # Queue statistics
POST /api/admin/queue/drain       # Cancel all jobs (requires Admin-Key header)
```

### Other Endpoints

```bash
GET  /health                      # System health, VRAM usage
GET  /models                      # List all available models
POST /loras/upload                # Upload LoRA adapter (.safetensors)
GET  /loras                       # List available LoRAs
```

---

## 🧠 Supported Models

| Model | Category | VRAM | Speed | Quality |
|-------|----------|------|-------|---------|
| `flux-1-dev` | Image | ~12GB (NF4) | 46s/image | ⭐⭐⭐⭐⭐ |
| `sd3.5-large` | Image | ~10GB (NF4) | 35s/image | ⭐⭐⭐⭐ |
| `realvisxl-v5` | Image | ~7GB | 25s/image | ⭐⭐⭐⭐ |
| `wan-t2v-1.3b` | Text→Video | ~10GB | ~2min/clip | ⭐⭐⭐ |
| `wan-t2v-14b` | Text→Video | ~18GB (NF4) | ~8min/clip | ⭐⭐⭐⭐⭐ |
| `wan-i2v-14b` | Image→Video | ~18GB (NF4) | ~8min/clip | ⭐⭐⭐⭐⭐ |
| `hunyuan-video` | Text→Video | ~15GB (NF4) | ~12min/clip | ⭐⭐⭐⭐ |

*Timings measured on A100 40GB at default settings.*

---

## 🎬 Long-Form Video Generation

Generate videos up to **2 minutes** using chunked sliding-window inference:

```bash
curl -X POST http://localhost:8080/api/video/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A drone shot flying over a tropical beach at golden hour",
    "model_name": "wan-t2v-1.3b",
    "num_frames": 480,
    "chunk_size": 81,
    "chunk_overlap": 20,
    "fps": 16
  }'
```

The system automatically:
1. Splits into overlapping 81-frame chunks
2. Generates each chunk sequentially
3. Blends overlapping regions with cosine weighting
4. Reports per-chunk progress via SSE

---

## ⚙️ Configuration

All settings are configured via environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face access token (required) |
| `API_KEYS` | — | Comma-separated API keys for auth |
| `ADMIN_API_KEY` | — | Admin key for queue drain endpoint |
| `CORS_ORIGINS` | `*` | Allowed CORS origins |
| `CACHE_DIR` | `/mnt/hf-cache` | HDD model cache |
| `CACHE_DIR_SSD` | `/app/model_cache` | SSD model cache (priority models) |
| `OUTPUT_DIR` | `/mnt/outputs` | Generated file storage |
| `OUTPUT_TTL_HOURS` | 24 | Auto-cleanup after N hours |
| `HF_OFFLINE` | `false` | Skip HF network checks after caching |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    FastAPI Server                     │
├──────────────┬──────────────┬────────────────────────┤
│ /generate    │ /api/video/* │ /api/jobs/*            │
│ (sync image) │ (async video)│ (status/cancel/SSE)    │
├──────────────┴──────┬───────┴────────────────────────┤
│              GPU Lock Layer                          │
│   _gpu_lock_image   │   _gpu_lock_video              │
├─────────────────────┴────────────────────────────────┤
│              Model Manager                           │
│   Lazy load · VRAM swap · NF4 quantization           │
│   SSD/HDD cache tiers · Single source of truth       │
├──────────────────────────────────────────────────────┤
│              Pipelines                               │
│   FluxPipeline · WanPipeline · HunyuanVideo          │
│   Chunked generation · Cosine blending · OOM recovery│
├──────────────────────────────────────────────────────┤
│              Job Queue                               │
│   Priority queue · Cancel flag · ETA calculation     │
│   Watchdog · SSE progress streaming                  │
├──────────────────────────────────────────────────────┤
│              Output Store                            │
│   Disk space guard · TTL cleanup · File serving      │
└──────────────────────────────────────────────────────┘
```

---

## 🧪 Testing

### Remote Smoke Test (all models)
```bash
python tools/remote_smoke_test.py --server http://<ip>:8080
```

### Targeted Test (specific models)
```bash
python tools/test_hq.py --server http://<ip>:8080
```

---

## 📋 Development

### Project Structure
```
app/
├── main.py              # FastAPI app, all endpoints, job handler
├── config.py            # Pydantic settings from environment
├── schemas_v2.py        # All API schemas (image + video + jobs)
├── model_manager.py     # Model registry, loading, VRAM management
├── pipeline_new.py      # Image generation pipeline
├── job_queue.py         # Async job queue + cancellation + ETA
├── output_store.py      # File storage + disk guard + cleanup
├── security.py          # API key auth + rate limiting
├── ui.py                # Built-in web UI template
└── pipelines/
    └── video_pipeline.py  # Video generation (T2V, I2V, chunked)
```

### Key Implementation Notes
- **Schemas**: All in `schemas_v2.py` — no separate schema files
- **Cache tiers**: Controlled by `MultiModelManager.get_cache_dir()` — single source of truth
- **GPU locks**: Separate for image (`_gpu_lock_image`) and video (`_gpu_lock_video`)
- **Job cancellation**: `cancel_flag` checked in step callback → raises `InterruptedError`
- **OOM**: Three-tier recovery — never jumps straight to CPU offload

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
