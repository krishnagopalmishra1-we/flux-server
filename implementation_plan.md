# Hyperforge Implementation Plan
# Last updated: 2026-05-09

---

## COMPLETED

### ✅ Phase 0 — Baseline (GCP A100 40GB)
- WAN 2.1 T2V 1.3B: chunked 480p, 15s in ~12 min warm
- WAN 2.2 T2V/I2V 14B: NF4, 720p, 5s smoke test passed
- Chunked video with cosine-blend overlap
- HunyuanVideo: NF4 + CPU offload (partial download)

### ✅ Phase 1 — UI Enhancement (2026-05-09)
- Library tab with filter/grid/list/lightbox/download/delete
- Quality presets: Draft · Balanced · HQ · Ultra
- 15s video preset (240 frames), 12 sample images, tab animations
- Backend: image indexing in library_meta.json, GET/DELETE /api/library

### ✅ Phase 2 — Vultr 8× A100 80GB Foundation (2026-05-09)
- docker-compose: `count: all`, removed CUDA_VISIBLE_DEVICES=0, shm 16g
- gunicorn: auto GPU detection, post_fork assigns CUDA_VISIBLE_DEVICES per worker
- model_manager: VRAM-aware NF4 (bf16_min_vram_gb), SD3.5-Large → BF16 on 80GB
- video_pipeline: _gpu_total_gb(), NF4 conditional on VRAM < 90GB
- Result: 8 parallel workers, 1.2× faster per job (SXM bandwidth), 2min model load

---

## NEXT SESSION — Priority 1

### 🔲 Phase 3 — xDiT 4-GPU Sequence Parallelism for WAN 14B
**Goal**: 23 min → ~7 min per 15s video at full BF16 quality. 2 concurrent jobs.

**Why xDiT, not device_map="auto"**:
- device_map="auto" = pipeline parallelism (GPUs take turns) → no speedup
- xDiT Sequence Parallelism = all 4 GPUs compute every step simultaneously → ~3.5× speedup
- xDiT officially supports WAN 2.x models

**Speed targets (WAN 14B T2V, 720p, 240 frames)**:

| Config | Per step | Total inference | vs current |
|--------|----------|-----------------|------------|
| Current (GCP NF4 1 GPU) | ~10s | ~23 min | baseline |
| Vultr NF4 1 GPU (done) | ~8s | ~19 min | 1.2× |
| **Vultr BF16 4 GPU xDiT** | **~2.8s** | **~6.5 min** | **3.5×** |

**Concurrent jobs**: 8 GPUs ÷ 4 per job = 2 simultaneous jobs.

---

#### 3.1 Install xDiT in Docker image

Add to `flux-server/Dockerfile`:
```dockerfile
RUN pip install xdit
# or from source for latest WAN support:
# RUN pip install git+https://github.com/xdit-project/xDiT.git
```

Verify WAN pipeline is available:
```python
from xfuser import xFuserWanPipeline, xFuserArgs
```

#### 3.2 GPU group allocation in gunicorn

Change from "1 GPU per worker" to "4 GPUs per worker group":

In `gunicorn.conf.py`:
```python
GPUS_PER_JOB = int(os.environ.get("GPUS_PER_JOB", "4"))
_num_gpus = _detect_gpu_count()       # 8
workers = _num_gpus // GPUS_PER_JOB   # 2

def post_fork(server, worker):
    group = worker.age % workers       # 0 or 1
    start = group * GPUS_PER_JOB      # 0 or 4
    gpus = ",".join(str(i) for i in range(start, start + GPUS_PER_JOB))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus   # "0,1,2,3" or "4,5,6,7"
```

#### 3.3 xDiT WAN pipeline wrapper in video_pipeline.py

Replace `_load_wan_t2v` for 14B with xDiT parallel version:

```python
def _load_wan_t2v_xdit(self, model_id: str, settings, cache_dir: str) -> None:
    from xfuser import xFuserWanPipeline, xFuserArgs
    import torch.distributed as dist

    # Init process group if not already done (one process per GPU in the group)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    engine_args = xFuserArgs(
        model=model_id,
        tensor_parallel_degree=1,
        sequence_parallel_degree=torch.cuda.device_count(),  # 4
        use_cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,     # Full BF16 — no NF4 needed across 4 GPUs
    )
    self._pipe = xFuserWanPipeline.from_pretrained(model_id, **engine_args.to_dict())
    self._xdit_enabled = True
```

**Key detail**: xDiT uses `torch.distributed` with NCCL backend. Each worker process spawns 4 CUDA processes internally (one per GPU in its group). This is handled by xDiT's engine — not manual multiprocessing.

#### 3.4 Update generate_long_video() for xDiT

The xDiT pipeline has the same call signature as diffusers `WanPipeline`, so the existing chunked generation loop works unchanged. Only the pipe construction changes.

Fallback: if xDiT import fails or dist init fails, fall back to NF4 single-GPU path.

```python
def _load_wan_t2v(self, model_id: str, quantize: bool, settings, cache_dir: str):
    if not quantize and self._gpu_total_gb() >= 70 and torch.cuda.device_count() >= 2:
        try:
            return self._load_wan_t2v_xdit(model_id, settings, cache_dir)
        except ImportError:
            logger.warning("xDiT not installed — falling back to NF4 single GPU")
    # existing NF4 path ...
```

#### 3.5 gunicorn + torch.distributed compatibility

**Problem**: gunicorn forks workers, but torch.distributed needs to be initialized AFTER fork (not before). xDiT handles this internally, but the MASTER_ADDR/MASTER_PORT env vars must be set per worker group.

In `post_fork`:
```python
def post_fork(server, worker):
    group = worker.age % workers
    start = group * GPUS_PER_JOB
    gpus = ",".join(str(i) for i in range(start, start + GPUS_PER_JOB))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # Each worker group gets its own distributed rendezvous port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(29500 + group)  # 29500 for group 0, 29501 for group 1
    os.environ["WORLD_SIZE"] = str(GPUS_PER_JOB)
    os.environ["RANK"] = "0"   # master rank within this worker's xDiT engine
```

#### 3.6 Test plan

```bash
# Smoke test: 5s video on 4 GPUs, verify all 4 show utilisation
curl -X POST http://localhost:8080/api/video/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"test","model_name":"wan-t2v-14b","num_frames":81,"resolution":"720p"}'

# In another terminal, watch GPU utilisation during generation
watch -n1 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
# Expected: GPUs 0,1,2,3 all showing >80% utilisation simultaneously
```

---

## FUTURE — Lower Priority

### 🔲 Phase 4 — 60s video (960 frames)
- After Phase 3 passes 15s test
- 24 chunks × ~55s each ≈ 22 min total on 4-GPU xDiT
- Test with test_hq_wan14b.sh (update frame count)

### 🔲 Phase 5 — HunyuanVideo 4-GPU
- xDiT also supports HunyuanVideo (xFuserHunyuanVideoPipeline)
- Complete download first (~38GB remaining)
- Expected: 720p 129-frame video in ~4 min on 4 GPUs (was ~12 min single GPU)

### 🔲 Phase 6 — 1080p resolution
- WAN 14B BF16 across 4 GPUs has ~240GB free VRAM for activations
- 1080p latent: ~4× larger than 720p → needs profiling
- Potential: first model to generate true 1080p video in <15 min

---

## KEY REFERENCE NUMBERS

| Model | VRAM (NF4) | VRAM (BF16) | Load time (NVMe) |
|-------|-----------|------------|-----------------|
| WAN T2V 1.3B | ~4 GB | ~4 GB | <1 min |
| WAN T2V/I2V 14B | ~25.5 GB | ~67.5 GB | ~19s BF16 / ~7s NF4 |
| HunyuanVideo | ~9 GB (NF4+CPU offload) | N/A | ~3 min |
| FLUX.1-dev | ~24 GB BF16 | ~24 GB | ~7s |
| SD3.5-Large | ~25 GB BF16 | ~25 GB | ~7s |
