# 🤖 AGENT.md — Neural Creation Studio

> **Last Updated:** 2026-04-16
> This file is the primary context document for any AI agent working on this codebase.

---

## 🏗️ Architecture Overview

```
Client (Browser/API)
  ↓
FastAPI app (app/main.py)
  ├── /generate          → Image generation (sync, GPU-locked)
  ├── /api/video/generate → Video generation (async job queue)
  ├── /api/jobs/{id}      → Job status (Completed: video_url promoted to top)
  ├── /api/jobs/{id}/stream → SSE real-time progress
  ├── /api/admin/queue/drain → Admin: cancel all jobs
  └── /health             → System health check
  ↓
Job Queue (app/job_queue.py)
  → _handle_video_job() [thread executor]
  → Model-aware resolution cap (1.3b @ 480p)
  → video_pipeline.generate_*() [model load + inference]
  → output_store.save()
```

### Key Design Principles
- **Single GPU**: All models share one A100 40GB. Only one model loaded at a time.
- **Turn-Based VRAM**: Image and video have separate GPU locks (`_gpu_lock_image`, `_gpu_lock_video`), but each modality unloads the other before loading its own model.
- **Async Job Queue**: Video jobs are async (submitted → queued → processing → completed). Image jobs are synchronous.
- **Cancel Support**: Running video jobs can be cancelled via `DELETE /api/jobs/{id}`. The step callback checks `job.cancel_flag`.

---

## 📦 Model Registry (Current)

| Category | Model Key | Model ID | VRAM | Status |
|----------|-----------|----------|------|--------|
| **Image** | `flux-1-dev` | `black-forest-labs/FLUX.1-dev` (NF4) | ~12GB | ✅ Production |
| **Image** | `sd3.5-large` | `stabilityai/stable-diffusion-3.5-large` (NF4) | ~10GB | ✅ Production |
| **Image** | `realvisxl-v5` | `SG161222/RealVisXL_V5.0` | ~7GB | ✅ Production |
| **Video** | `wan-t2v-1.3b` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | ~10GB | ✅ Production (480p cap) |
| **Video** | `wan-t2v-14b` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` (NF4) | ~18GB | ✅ Production |
| **Video** | `wan-i2v-14b` | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` (NF4) | ~18GB | ✅ Production |
| **Video** | `hunyuan-video` | `hunyuanvideo-community/HunyuanVideo` (NF4) | ~15GB | ✅ Production |

### Removed Models
- `ltx-video` — Removed (untested, broken pipeline)
- `audioldm2`, `ace-step` — Removed (music feature deprecated)
- `echomimic`, `liveportrait` — Removed (animation feature deprecated)

---

## 📊 Recent Test Results (2026-04-13 Update)

Smoke tests conducted on a **GCP A100 (40GB)** instance.

| Test | Model | Params | Status | Duration |
| :--- | :--- | :--- | :--- | :--- |
| **Pixar 3D** | `wan-t2v-1.3b` | 720p/240fr/50st | **PASS** | 1290s (~21 min). Result: Distorted. |
| **WAN Optimization**| `wan-t2v-1.3b`| 480p/240fr/20st| **PASSED** | ~12 min inference (optimized overlap). |

### Performance Discovered
- **Resolution Impact**: 1.3b at 720p is ~5x slower than 480p due to quadratic spatial attention. 1.3b native is 480p; interpolation at 720p causes distortion.
- **Chunk Overlap Waste**: `_blend_overlap` uses EDGE_FADE=4; old 16-frame overlap wasted 12 inference frames per chunk. New default is **8**.
- **Guidance Scale**: WAN sweet spot is **7.0**. 5.0 causes flat contrast/shittier quality.
- **A100 VRAM**: Standby (1.3B) = 14.5 GB. Inference (14B NF4) = ~18-24 GB. **14B Chunking (240fr)** = **36.2 GB**.
- **GCP DNS**: Transient `compute.googleapis.com` resolution errors encountered; retry logic recommended.

---

## 📁 Project Structure

```
flux-server/
├── app/
│   ├── main.py              # FastAPI app, endpoints, job handler, lifespan
│   ├── config.py             # Pydantic settings (env-driven)
│   ├── schemas.py            # ALL request/response schemas (image + video + jobs)
│   ├── model_manager.py      # MultiModelManager — model registry, loading, VRAM mgmt
│   ├── pipeline.py           # Image generation pipeline (FluxInferencePipeline)
│   ├── job_queue.py          # Async job queue with priority, cancellation, ETA
│   ├── output_store.py       # File storage, cleanup cron, disk space guard
│   ├── security.py           # API key auth, rate limiting
│   ├── pipelines/
│   │   └── video_pipeline.py # Video generation — T2V, I2V, HunyuanVideo, chunked long-video
│   └── static/
│       ├── app.js            # Frontend JavaScript (vanilla)
│       └── style.css         # Frontend CSS
├── tools/                    # Smoke testing and debugging scripts
├── deploy/gcp/               # GCP deployment scripts
├── docs/                     # Technical docs (NF4 quantization notes)
├── Dockerfile                # Multi-stage Docker build
├── docker-compose.yml        # Production compose with GPU passthrough
├── gunicorn.conf.py          # ASGI server config
├── requirements.txt          # Python dependencies
├── .env.example              # Environment template
└── README.md                 # Project documentation
```

---

## ⚠️ Critical Knowledge

### Cache Tier System
- **SSD tier** (`cache_dir_ssd`): Fast models — `flux-1-dev`, `wan-t2v-1.3b`, `hunyuan-video`
- **HDD tier** (`cache_dir`): Large models — `wan-t2v-14b`, `wan-i2v-14b` (118GB each)
- Single source of truth: `MultiModelManager.get_cache_dir(model_name)`

### OOM Recovery (Tiered)
1. Reduce inference steps (min 20)
2. Reduce frame count (min 16, halved)
3. Sequential CPU offload (last resort, 10× slower)

### Chunked Long-Video Generation
- When `num_frames > chunk_size` (default **49**), auto-routes to `generate_long_video()`
- Sliding window with configurable overlap (default **8** frames)
- Cosine-weighted blending at chunk boundaries
- Default `num_inference_steps`: **20** (optimized for warm A100 performance)
- Default `guidance_scale`: **7.0**
- Supports up to 1920 frames (2 min @ 16fps)
- **Performance**: 240fr/480p/20 steps ≈ 12 min inference (warm).

### Schema Logic
- **JobStatusResponse**: Video result fields (`video_url`, `thumbnail_b64`, `inference_time_ms`, etc.) are automatically promoted to top-level from the `result` dict when job status is `completed`.
- **progress_pct**: Alias for `progress`, supported for polling scripts.

### BitsAndBytes Quantized Models
- **NEVER** call `.to(device)` on quantized pipelines — it crashes.
- Move individual non-quantized components instead. See `docs/nf4_quantization_error.md`.
- `device_map="balanced"` is **incompatible** with BnB — use explicit `.to(device)`.

---

## 🔧 Pending Work (for future agents)

### BLOCKED on Infrastructure (GCP)
- [ ] Attach 256GB NVMe SSD (`pd-ssd`) for 14B models → reduces load from 12min to 2min
- [ ] Create udev rule for persistent HDD readahead (32MB)

### Requires GPU Testing
- [ ] Validate `torch.compile(mode="reduce-overhead")` on A100
- [ ] Test NF4 quantization on VAE for 14B models (save 1-2GB VRAM)
- [ ] Confirm all pipelines use `torch.bfloat16` (not `float16` which causes NaN)
- [ ] Verify image jobs use `JobPriority.FAST` to bypass video queue

### Smoke Test Timeouts (GCP VM only) — CONFIRMED
- [ ] WAN T2V 14B: 1800s is insufficient for HQ → use 4500s.
- [ ] WAN T2V 1.3B: 600s is insufficient for chunked → use 2400s.
- [ ] WAN I2V 14B: 900s is insufficient for HQ → use 4500s.

### Known Issues
- [ ] **Model Selection**: API endpoints default to `wan-t2v-1.3b` when `model_name` is unspecified, which can lead to unexpected model usage in long-video jobs. Needs schema-level default refinement.

---

## 🚀 Quick Commands

```bash
# Local development
docker compose up --build

# Check health
curl http://localhost:8080/health

# Generate image
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat in space"}'

# Generate video (returns job_id)
curl -X POST http://localhost:8080/api/video/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "ocean waves at sunset", "model_name": "wan-t2v-1.3b"}'

# Check job status
curl http://localhost:8080/api/jobs/<job_id>

# Cancel job
curl -X DELETE http://localhost:8080/api/jobs/<job_id>

# Admin: drain all jobs
curl -X POST http://localhost:8080/api/admin/queue/drain \
  -H "Admin-Key: <your-admin-key>"
```
