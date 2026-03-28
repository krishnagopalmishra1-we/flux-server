# FLUX.1-dev Production Deployment on NVIDIA L4 GPU

## 💰 Licensing & Cost Audit — Everything is FREE

> [!NOTE]
> **All software in this plan is $0 cost.** The only expense is GCP cloud compute (~$0.25–$0.84/hr while running).

| Component | License | Payment Required? |
|---|---|---|
| **FLUX.1-dev** | FLUX.1 [dev] Non-Commercial License | ❌ **Free** — just accept license on Hugging Face (no payment, ever) |
| **FLUX.1-schnell** | Apache 2.0 | ❌ **Free** — fully open, even for commercial use |
| PyTorch | BSD-3-Clause | ❌ Free |
| Hugging Face Diffusers | Apache 2.0 | ❌ Free |
| Hugging Face Transformers | Apache 2.0 | ❌ Free |
| FastAPI | MIT | ❌ Free |
| Gunicorn / Uvicorn | MIT | ❌ Free |
| Docker (CE) | Apache 2.0 | ❌ Free |
| NVIDIA Container Toolkit | Apache 2.0 | ❌ Free |
| NVIDIA CUDA Runtime | NVIDIA EULA (free) | ❌ Free |
| Redis | BSD-3-Clause | ❌ Free |
| Gradio (optional UI) | Apache 2.0 | ❌ Free |
| xFormers | BSD-3-Clause | ❌ Free |
| All other Python deps | Open source | ❌ Free |

> [!IMPORTANT]
> **FLUX.1-dev** requires you to accept a free license on Hugging Face (click "Agree" on the model page). No credit card, no payment — just a free HF account. The license only restricts **commercial resale** of the model itself, not using it.

---

## My Suggested Improvements Over Your Original Prompt

> [!TIP]
> All improvements below use 100% free, open-source software:

| Area | Your Prompt | My Recommendation | Why |
|---|---|---|---|
| **Model** | FLUX.1-dev only | FLUX.1-dev + **FLUX.1-schnell** fallback | Schnell is 4-step distilled, 5× faster, Apache 2.0 licensed — perfect for low-latency or budget mode |
| **Precision** | `bfloat16` only | `bfloat16` + **torch.compile()** | L4 supports `bfloat16` natively; `torch.compile` adds 20-40% speedup after warmup |
| **Offloading** | `enable_model_cpu_offload()` | **Skip CPU offload** on L4 24GB | FLUX.1-dev fits in ~18GB at bf16. CPU offload adds 3-5s latency per image. Only enable for batch/high-res |
| **Server** | FastAPI only | FastAPI + **async queue** (Celery/Redis) | Image gen takes 10-30s — blocking the endpoint kills throughput |
| **Caching** | Optional section | **Mandatory** HF cache mount + prompt-hash dedup cache | Avoids re-downloading 24GB model on every cold start |
| **UI** | None | Add a **Gradio** or **Streamlit** admin panel | Real-time monitoring, test generation, queue visibility |
| **Container** | Single Dockerfile | **Multi-stage** build + NVIDIA Container Toolkit | Smaller image, faster pulls, security hardening |
| **Deployment** | GCE or Vertex AI | **Cloud Run GPU** (GA March 2025) | Serverless, scale-to-zero, pay-per-use, no VM management |
| **Monitoring** | Generic logging | **OpenTelemetry** + Cloud Monitoring custom metrics | Track gen latency p50/p95/p99, VRAM usage, queue depth |
| **Security** | API key + rate limit | API key + rate limit + **input sanitization + CORS + request size limits** | Prevent prompt injection, abuse, and DoS |

---

## 1. Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                     Google Cloud Platform                      │
│                                                               │
│  ┌─────────────┐    ┌──────────────────────────────────────┐  │
│  │  Cloud Load  │    │  GCE Instance (g2-standard-8)       │  │
│  │  Balancer    │───▶│  NVIDIA L4 24GB · 8 vCPU · 32GB RAM │  │
│  │  (HTTPS)     │    │                                      │  │
│  └─────────────┘    │  ┌──────────┐  ┌──────────────────┐  │  │
│                      │  │ Gunicorn │  │   Redis Queue    │  │  │
│  ┌─────────────┐    │  │ + Uvicorn│  │  (in-memory)     │  │  │
│  │ Cloud        │    │  │          │  └──────────────────┘  │  │
│  │ Monitoring   │◀───│  │ FastAPI  │                        │  │
│  │ + Logging    │    │  │ Server   │  ┌──────────────────┐  │  │
│  └─────────────┘    │  │          │──│  FLUX.1-dev      │  │  │
│                      │  │          │  │  Pipeline (GPU)  │  │  │
│  ┌─────────────┐    │  └──────────┘  └──────────────────┘  │  │
│  │ GCS Bucket  │    │                                      │  │
│  │ (outputs)   │◀───│  /mnt/hf-cache (persistent disk)    │  │
│  └─────────────┘    └──────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **GPU**: NVIDIA L4 (24GB VRAM, Ada Lovelace, 3rd-gen Tensor Cores)
  - FLUX.1-dev at `bfloat16` uses ~18GB VRAM → fits comfortably
  - No CPU offloading needed at default resolution (1024×1024)
- **Instance**: `g2-standard-8` (1× L4, 8 vCPU, 32GB RAM) — ~$0.84/hr on-demand, ~$0.25/hr spot
- **Queue**: Redis-backed async queue for non-blocking generation
- **Cache**: Persistent disk mounted at `/mnt/hf-cache` for model weights

---

## 2. Hugging Face Integration

### Model Download Strategy

```bash
# Pre-download during Docker build or instance startup
huggingface-cli download black-forest-labs/FLUX.1-dev \
  --local-dir /mnt/hf-cache/FLUX.1-dev \
  --local-dir-use-symlinks False
```

- **FLUX.1-dev** is a gated model — requires a Hugging Face token with accepted license
- Total download: ~24GB (transformer + VAE + text encoders)
- Cache to persistent disk to survive instance restarts

### Diffusers Pipeline

```python
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    cache_dir="/mnt/hf-cache",
)
pipe.to("cuda")
```

### VRAM-Saving Features (use selectively)

| Feature | VRAM Saved | Latency Cost | When to Use |
|---|---|---|---|
| `enable_attention_slicing()` | ~2-3GB | +10-15% | When generating >1024×1024 |
| `enable_vae_slicing()` | ~1-2GB | +5% | When batch decoding |
| `enable_model_cpu_offload()` | ~8-10GB | +3-5s/image | Only if running multiple models |
| `enable_sequential_cpu_offload()` | ~15GB | +10-15s/image | Only if GPU <16GB |
| `torch.compile(pipe.transformer)` | 0 | −20-40% (faster) | Always (after warmup) |

**Recommended L4 config** (no offloading needed):
```python
pipe.to("cuda")
pipe.enable_attention_slicing()      # Marginal cost, safety margin
pipe.enable_vae_slicing()            # Free for single images
# pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")  # Optional: adds warmup time
```

---

## 3. Complete FastAPI Server

### Project Structure

```
flux-server/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── pipeline.py          # Model loading & inference
│   ├── schemas.py           # Pydantic models
│   ├── security.py          # API key auth + rate limiting
│   ├── config.py            # Settings
│   └── queue.py             # Async job queue
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── gunicorn.conf.py
└── .env.example
```

### `app/config.py`

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Model
    model_id: str = "black-forest-labs/FLUX.1-dev"
    cache_dir: str = "/mnt/hf-cache"
    hf_token: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1  # 1 worker per GPU

    # Security
    api_keys: str = ""  # Comma-separated valid API keys
    rate_limit_per_minute: int = 10

    # Generation defaults
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 28
    default_guidance_scale: float = 3.5
    max_width: int = 2048
    max_height: int = 2048

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### `app/schemas.py`

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import uuid

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: Optional[str] = Field(None, max_length=1000)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    num_inference_steps: int = Field(28, ge=1, le=50)
    guidance_scale: float = Field(3.5, ge=0.0, le=20.0)
    seed: Optional[int] = None

    @field_validator("width", "height")
    @classmethod
    def must_be_multiple_of_8(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError("Must be a multiple of 8")
        return v

    @field_validator("prompt")
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        # Basic sanitization — strip control characters
        return "".join(c for c in v if c.isprintable())

class GenerateResponse(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "completed"
    image_base64: Optional[str] = None
    seed_used: int = 0
    inference_time_ms: float = 0.0

class HealthResponse(BaseModel):
    status: str = "healthy"
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    model_loaded: bool = False
```

### `app/security.py`

```python
from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from collections import defaultdict
import time
from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory rate limiter (use Redis in production cluster)
_request_timestamps: dict[str, list[float]] = defaultdict(list)

def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    settings = get_settings()
    valid_keys = [k.strip() for k in settings.api_keys.split(",") if k.strip()]

    if not valid_keys:
        return "anonymous"  # No keys configured = open access

    if not api_key or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key

def check_rate_limit(request: Request, api_key: str) -> None:
    settings = get_settings()
    now = time.time()
    window = 60.0  # 1 minute
    client_id = api_key or request.client.host

    # Purge old timestamps
    _request_timestamps[client_id] = [
        ts for ts in _request_timestamps[client_id] if now - ts < window
    ]

    if len(_request_timestamps[client_id]) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {settings.rate_limit_per_minute} requests/minute.",
            headers={"Retry-After": "60"},
        )

    _request_timestamps[client_id].append(now)
```

### `app/pipeline.py`

```python
import torch
import time
import base64
import io
import logging
from diffusers import FluxPipeline
from app.config import get_settings

logger = logging.getLogger(__name__)

class FluxInferencePipeline:
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        settings = get_settings()
        logger.info(f"Loading model: {settings.model_id}")

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "cache_dir": settings.cache_dir,
        }
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        self.pipe = FluxPipeline.from_pretrained(
            settings.model_id,
            **load_kwargs,
        )

        # Move to GPU
        self.pipe.to(self.device)

        # Enable memory optimizations
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()

        # Optional: torch.compile for 20-40% speedup (adds ~2min warmup)
        # self.pipe.transformer = torch.compile(
        #     self.pipe.transformer, mode="reduce-overhead"
        # )

        # Warmup pass
        logger.info("Running warmup inference...")
        with torch.no_grad():
            self.pipe(
                prompt="warmup",
                width=256,
                height=256,
                num_inference_steps=1,
                output_type="latent",
            )
        torch.cuda.empty_cache()
        logger.info("Model loaded and warmed up.")

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int | None = None,
    ) -> tuple[str, int, float]:
        """Returns (base64_png, seed_used, inference_time_ms)."""
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        start = time.perf_counter()

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        image = result.images[0]

        # Encode to base64 PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        torch.cuda.empty_cache()

        logger.info(
            f"Generated {width}x{height} in {elapsed_ms:.0f}ms "
            f"(steps={num_inference_steps}, seed={seed})"
        )
        return img_b64, seed, elapsed_ms

    @property
    def is_loaded(self) -> bool:
        return self.pipe is not None

    def gpu_info(self) -> dict:
        if not torch.cuda.is_available():
            return {"name": "N/A", "total_gb": 0, "used_gb": 0}
        return {
            "name": torch.cuda.get_device_name(0),
            "total_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2),
            "used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        }

# Singleton
flux_pipeline = FluxInferencePipeline()
```

### `app/main.py`

```python
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.security import verify_api_key, check_rate_limit
from app.pipeline import flux_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    flux_pipeline.load()
    yield
    # Cleanup
    del flux_pipeline.pipe

app = FastAPI(
    title="FLUX.1-dev Inference API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health():
    gpu = flux_pipeline.gpu_info()
    return HealthResponse(
        status="healthy" if flux_pipeline.is_loaded else "loading",
        gpu_name=gpu["name"],
        vram_total_gb=gpu["total_gb"],
        vram_used_gb=gpu["used_gb"],
        model_loaded=flux_pipeline.is_loaded,
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    check_rate_limit(request, api_key)

    img_b64, seed_used, elapsed_ms = flux_pipeline.generate(
        prompt=req.prompt,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
    )

    return GenerateResponse(
        status="completed",
        image_base64=img_b64,
        seed_used=seed_used,
        inference_time_ms=elapsed_ms,
    )
```

### `requirements.txt`

```
torch>=2.2.0
diffusers>=0.30.0
transformers>=4.40.0
accelerate>=0.30.0
safetensors>=0.4.0
sentencepiece>=0.2.0
protobuf>=4.25.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
gunicorn>=22.0.0
pydantic-settings>=2.2.0
python-multipart>=0.0.9
Pillow>=10.3.0
huggingface-hub>=0.23.0
```

### `gunicorn.conf.py`

```python
bind = "0.0.0.0:8080"
workers = 1           # 1 worker per GPU
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 300         # Image generation can take 30s+
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
```

---

## 4. Dockerfile (Production-Ready, Multi-Stage)

```dockerfile
# ============================================================
# Stage 1: Builder — install Python deps
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Runtime — slim production image
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -s /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/mnt/hf-cache" \
    TRANSFORMERS_CACHE="/mnt/hf-cache"

WORKDIR /app
COPY app/ ./app/
COPY gunicorn.conf.py .

# Run as non-root
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]
```

### `docker-compose.yml` (local dev)

```yaml
version: "3.9"
services:
  flux-server:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - hf-cache:/mnt/hf-cache
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - API_KEYS=${API_KEYS}
      - RATE_LIMIT_PER_MINUTE=10
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  hf-cache:
```

---

## 5. GCP Deployment Instructions

### Option A: GCE with NVIDIA L4 (Recommended for control)

```bash
# 1. Create persistent disk for model cache
gcloud compute disks create flux-model-cache \
    --size=100GB \
    --type=pd-ssd \
    --zone=us-central1-a

# 2. Create the VM
gcloud compute instances create flux-server \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-balanced \
    --image-family=common-cu124-ubuntu-2204 \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --disk="name=flux-model-cache,device-name=hf-cache,mode=rw" \
    --tags=flux-server \
    --scopes=cloud-platform

# 3. Firewall rule
gcloud compute firewall-rules create allow-flux-8080 \
    --allow=tcp:8080 \
    --target-tags=flux-server \
    --source-ranges=0.0.0.0/0  # Restrict to your IP in production

# 4. SSH in and deploy
gcloud compute ssh flux-server --zone=us-central1-a

# On the VM:
sudo mkfs.ext4 /dev/disk/by-id/google-hf-cache
sudo mkdir -p /mnt/hf-cache
sudo mount /dev/disk/by-id/google-hf-cache /mnt/hf-cache
sudo chmod 777 /mnt/hf-cache

# Install Docker + NVIDIA Container Toolkit
curl -fsSL https://get.docker.com | sudo sh
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Clone your repo and run
git clone <your-repo-url> && cd flux-server
echo "HF_TOKEN=hf_your_token_here" > .env
echo "API_KEYS=your-secret-key-1,your-secret-key-2" >> .env
sudo docker compose up -d
```

### Option B: Cloud Run with GPU (Serverless — scale to zero)

```bash
# Build and push to Artifact Registry
gcloud artifacts repositories create flux-repo \
    --repository-format=docker \
    --location=us-central1

gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/flux-repo/flux-server:v1

# Deploy to Cloud Run with L4 GPU
gcloud run deploy flux-server \
    --image us-central1-docker.pkg.dev/$PROJECT_ID/flux-repo/flux-server:v1 \
    --region us-central1 \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --cpu 8 \
    --memory 32Gi \
    --port 8080 \
    --min-instances 0 \
    --max-instances 3 \
    --timeout 300 \
    --no-allow-unauthenticated \
    --set-env-vars "HF_TOKEN=hf_xxx,API_KEYS=key1,RATE_LIMIT_PER_MINUTE=10"
```

> [!WARNING]
> Cloud Run GPU has cold starts of ~60-120s (model loading). Set `min-instances=1` if you need low latency.

### Autoscaling Strategy

| Metric | Target | Action |
|---|---|---|
| GPU utilization | >70% sustained 2min | Scale out +1 instance |
| Request queue depth | >5 pending | Scale out +1 instance |
| GPU utilization | <20% sustained 10min | Scale in −1 instance |
| No requests | 30min idle | Scale to 0 (Cloud Run) |

---

## 6. Security

### API Key Authentication
- Keys passed via `X-API-Key` header
- Stored as environment variable (comma-separated)
- For production: use **GCP Secret Manager** instead of env vars

```bash
# Create secret
echo -n "key1,key2,key3" | gcloud secrets create flux-api-keys --data-file=-

# Grant access to service account
gcloud secrets add-iam-policy-binding flux-api-keys \
    --member="serviceAccount:YOUR_SA@PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

### Rate Limiting
- In-memory sliding window (single instance)
- For multi-instance: use **Redis** or **Cloud Armor rate limiting**

### Additional Security Measures
- **Input sanitization**: Strip control characters from prompts (implemented in schemas)
- **Request size limits**: Max prompt 2000 chars, max resolution 2048×2048
- **CORS**: Restrict `allow_origins` to your frontend domain
- **Non-root container**: `USER appuser` in Dockerfile
- **Network**: Use VPC + Cloud Armor for DDoS protection

---

## 7. Performance Tuning for L4 (24GB VRAM)

### Benchmarks (estimated, 1024×1024, bf16)

| Steps | Without compile | With torch.compile | With compile + schnell |
|---|---|---|---|
| 28 | ~25s | ~18s | N/A |
| 4 (schnell only) | N/A | N/A | ~3s |

### Optimizations Checklist

- [x] `torch.bfloat16` — native L4 support, no precision loss vs fp16
- [x] `enable_attention_slicing()` — reduces peak VRAM with minimal latency cost
- [x] `enable_vae_slicing()` — reduces VAE decode memory
- [ ] `torch.compile(mode="reduce-overhead")` — 20-40% faster after 2-3 warmup runs
- [ ] `torch.cuda.set_per_process_memory_fraction(0.95)` — pre-allocate VRAM
- [ ] **xFormers** — alternative to attention slicing if installed
- [ ] **TensorRT** via `torch_tensorrt` — maximum throughput (complex setup)

### Resolution-VRAM Guide

| Resolution | VRAM Usage (bf16) | Feasible on L4? |
|---|---|---|
| 512×512 | ~12GB | ✅ Comfortable |
| 768×768 | ~15GB | ✅ Comfortable |
| 1024×1024 | ~18GB | ✅ Default |
| 1024×1536 | ~21GB | ⚠️ Tight — enable offload |
| 2048×2048 | ~28GB+ | ❌ Needs CPU offload |

---

## 8. Cost Optimization

### Instance Pricing (us-central1)

| Strategy | g2-standard-8 (1×L4) | Monthly (24/7) | Monthly (12hr/day) |
|---|---|---|---|
| On-demand | $0.84/hr | ~$610 | ~$305 |
| Spot/Preemptible | $0.25/hr | ~$182 | ~$91 |
| 1-year CUD | $0.59/hr | ~$428 | N/A (always-on) |
| 3-year CUD | $0.42/hr | ~$306 | N/A (always-on) |
| Cloud Run GPU | ~$0.84/hr active | Pay per request | Best for bursty |

### Cost Reduction Strategies

1. **Spot instances** for dev/staging (70% savings, may preempt)
2. **Scheduled start/stop** via Cloud Scheduler for non-24/7 workloads
3. **Cloud Run scale-to-zero** for bursty/low-volume workloads
4. **Use FLUX.1-schnell** for draft/preview quality (4 steps vs 28 = 7× cheaper per image)
5. **Prompt-hash caching**: Cache generated images by prompt+params hash (Redis or GCS)
6. **Right-size**: Monitor actual VRAM usage; `g2-standard-4` may suffice

### Scheduled Start/Stop Script

```bash
# Auto-stop at midnight, auto-start at 8am (UTC)
gcloud compute resource-policies create instance-schedule flux-schedule \
    --region=us-central1 \
    --vm-start-schedule="0 8 * * *" \
    --vm-stop-schedule="0 0 * * *" \
    --timezone=UTC

gcloud compute instances add-resource-policies flux-server \
    --resource-policies=flux-schedule \
    --zone=us-central1-a
```

---

## 9. Logging & Monitoring

### Structured Logging

```python
import json, logging

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record),
            "module": record.module,
        }
        if hasattr(record, "extra"):
            log.update(record.extra)
        return json.dumps(log)
```

### Custom Metrics to Track

| Metric | Type | Alert Threshold |
|---|---|---|
| `inference_latency_ms` | Distribution | p95 > 40s |
| `gpu_vram_used_pct` | Gauge | > 90% |
| `requests_per_minute` | Counter | > rate limit × 2 |
| `queue_depth` | Gauge | > 10 |
| `error_rate` | Ratio | > 5% |

### Cloud Monitoring Alert

```bash
gcloud monitoring policies create \
    --notification-channels=YOUR_CHANNEL \
    --display-name="FLUX High Latency" \
    --condition-display-name="p95 latency > 40s" \
    --condition-filter='metric.type="custom.googleapis.com/flux/inference_latency"' \
    --condition-threshold-value=40000 \
    --condition-threshold-duration=300s
```

---

## 10. Model Caching Strategy

### Layer 1: Persistent Disk Cache (Model Weights)

```bash
# Mount at /mnt/hf-cache, set HF_HOME=/mnt/hf-cache
# Survives instance restarts, ~100GB SSD
```

### Layer 2: Output Cache (Generated Images)

```python
import hashlib
from functools import lru_cache

def cache_key(prompt: str, width: int, height: int, steps: int,
              guidance: float, seed: int) -> str:
    raw = f"{prompt}|{width}|{height}|{steps}|{guidance}|{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()

# Check GCS/Redis before generating; store result after
```

### Layer 3: Pre-built Docker Image with Weights

```dockerfile
# For zero-cold-start: bake weights into Docker image
# Warning: image size ~30GB+, slow to push/pull
FROM flux-server:base AS with-weights
RUN huggingface-cli download black-forest-labs/FLUX.1-dev \
    --local-dir /mnt/hf-cache/FLUX.1-dev \
    --token $HF_TOKEN
```

---

## Verification Plan

### Automated Tests

```bash
# Build the Docker image
docker build -t flux-server:test .

# Run health check (no GPU needed for this test)
docker run --rm flux-server:test python -c "from app.config import get_settings; print(get_settings())"

# Run with GPU (requires NVIDIA runtime)
docker compose up -d
curl http://localhost:8080/health
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"prompt": "a cat sitting on a cloud", "num_inference_steps": 4, "width": 512, "height": 512}'
```

### Manual Verification

1. Deploy to a GCE `g2-standard-8` instance
2. Confirm `/health` returns `model_loaded: true`
3. Generate a test image and verify PNG output
4. Confirm rate limiting triggers after configured threshold
5. Monitor Cloud Logging for structured JSON logs
