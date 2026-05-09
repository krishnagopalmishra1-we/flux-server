# CLAUDE.md — Neural Creation Studio
# Read this at the start of every session. It is the single source of truth.
# Last updated: 2026-05-09

---

## WHAT THIS PROJECT IS

A FastAPI video/image generation server targeting Vultr Bare Metal 8× A100 SXM 80GB.
Primary goal: generate 1–2 minute videos in ≤30 minutes without quality loss,
with 8 parallel workers for maximum GPU utilisation.

**Repo root**: `d:/Flux_Lora/`
**Server code**: `d:/Flux_Lora/flux-server/`

### GCP VM (legacy — use Vultr for production)
**GCP VM**: `flux-a100-preemptible`, zone `us-central1-a`, project `flux-lora-gpu-project`
**SSH**: `gcloud compute ssh krishnagopalmishra1-we@flux-a100-preemptible --zone=us-central1-a`
**Container**: `flux-server-flux-server-1`
**API**: `http://localhost:8080` (from inside VM)

### Vultr Bare Metal (active target)
**Plan**: `vbm-112c-2048gb-8-a100-gpu` — 8× A100 SXM 80 GB, NVMe, 112 vCPU, 2TB RAM
**Deploy script**: `flux-server/deploy/vultr/launch.sh`
**Bootstrap**: `flux-server/deploy/vultr/bootstrap.sh`
**Compose**: `flux-server/deploy/vultr/docker-compose.vultr.yml`

---

## CURRENT VM STATE

**GCP VM status**: RUNNING (as of end-of-session 2026-04-16)
**Stop GCP VM before exit**: `gcloud compute instances stop flux-a100-preemptible --zone=us-central1-a`

---

## DISK LAYOUT

| Disk | Mount | Size | Contents |
|------|-------|------|----------|
| Root (SSD) | `/` | 243GB | OS + Docker + SSD model cache |
| HDD | `/mnt/hf-cache-disk` | 492GB | Large 14B models |

**SSD model cache** (`/var/lib/docker/volumes/flux-server_model_cache_ssd/_data/` → `/app/model_cache`):
- `wan-t2v-1.3b` (27GB)
- `flux-1-dev`
- `hunyuan-video` (PARTIALLY downloaded — ~1.7GB / ~40GB as of 2026-04-13)

**HDD model cache** (`/mnt/hf-cache-disk/`):
- `wan-t2v-14b` (118GB) — loads in ~30 min from HDD
- `wan-i2v-14b` (118GB) — loads in ~18 min from HDD

---

## VULTR 8× A100 MULTI-GPU ARCHITECTURE
**Added: 2026-05-09**

### How parallel GPU utilisation works

| Layer | File | Change |
|-------|------|--------|
| Docker | `deploy/vultr/docker-compose.vultr.yml` | `count: all` (was `count: 1`), removed `CUDA_VISIBLE_DEVICES=0`, `shm_size: 16g` |
| Gunicorn | `gunicorn.conf.py` | `workers = nvidia-smi GPU count` (auto-detected), `post_fork` assigns each worker `CUDA_VISIBLE_DEVICES=N` |
| Model mgr | `app/model_manager.py` | NF4 now VRAM-aware: skipped for image models on ≥50 GB GPU; WAN 14B retains NF4 (BF16 ~78 GB is too tight for single 80 GB) |
| Video pipeline | `app/pipelines/video_pipeline.py` | WAN 14B NF4 conditional on `_gpu_total_gb() < 90` |

**Result**: 8 gunicorn workers, each pinned to one A100 80 GB, each serving one concurrent video/image job.
Expected throughput: **8× parallel jobs** vs previous 1.

### NF4 quantization decisions on 80 GB A100

| Model | BF16 size | NF4 on 80 GB? | Reason |
|-------|-----------|---------------|--------|
| WAN T2V / I2V 14B | ~67.5 GB model + ~12 GB activations ≈ 80 GB | **Yes** (NF4) | Too close to limit; NF4 = 25.5 GB, safe |
| HunyuanVideo | ~9 GB NF4 | **Yes** (NF4) | CPU-offload text encoder, NF4 already optimised |
| SD3.5-Large | ~25 GB BF16 | **No** (BF16) | `bf16_min_vram_gb=50` → auto BF16 on 80 GB |
| FLUX.1-dev | ~24 GB BF16 | **No** (BF16) | Already unquantized |
| WAN T2V 1.3B | ~4 GB BF16 | **No** (BF16) | Already unquantized |

### Env vars for tuning

| Var | Default | Purpose |
|-----|---------|---------|
| `NUM_WORKERS` | auto (GPU count) | Override worker count (e.g. `4` for debug) |
| `GPU_COUNT` | auto (nvidia-smi) | Override GPU detection (e.g. if nvidia-smi unavailable) |

### Vultr Quick Commands

```bash
# Deploy (first time)
export VULTR_API_KEY="..." SSH_KEY_ID="..."
bash flux-server/deploy/vultr/launch.sh

# SSH into instance (after launch.sh saves .instance file)
source flux-server/deploy/vultr/.instance
ssh root@$PUBLIC_IP

# After SSH: verify all 8 GPUs visible inside container
docker exec flux-server-flux-server-1 nvidia-smi -L

# Check workers and their GPU assignments (from container logs)
docker logs flux-server-flux-server-1 2>&1 | grep "gunicorn.*GPU"

# Rebuild and restart after code change
cd /opt/flux-server/flux-server && git pull && docker compose up --build -d

# Check active jobs on all workers
curl -s http://localhost:8080/api/jobs | python3 -c "import sys,json; [print(j['job_id'][:8], j['status'], j.get('model_name')) for j in json.load(sys.stdin)]"

# GPU utilisation across all 8 A100s
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

---

## Status: UI ENHANCED + LIBRARY ADDED
**Last Update: 2026-05-09**

### Previous (2026-04-18): Native HQ Deployed
- Removed 480p clamp, added cosine-wave blending → smooth 15s 720p on 1.3B

### New (2026-05-09): Hyperforge UI Improvement Pass
- **Library tab** (`/library`): persistent gallery of all generated images + videos with filter (All/Image/Video), grid/list view, lightbox, download, delete
- **Quality presets** on image tab: Draft · Balanced · HQ · Ultra (one-click steps + CFG)
- **15s video preset** (240 frames) added to duration strip
- **12 Unsplash sample images** (was 4) across diverse visual styles
- **Tab animations**: `pageIn` fade+slide transition, shimmer skeleton loader
- **Backend**: images now saved to `outputs/image/` + indexed in `outputs/library_meta.json`
- **API**: `GET /api/library`, `DELETE /api/library/{id}`
- Infrastructure: committed and pushed to `codex/hyperforge-runtime-hardening`

| Feature | Change | impact |
|------|--------|--------|
| **Resolution Cap**| `wan-t2v-1.3b` auto-capped at **480p** | **5x speedup** on 1.3b (avoids quadratic attention) |
| **Chunk Overlap** | Reduced **16 → 8** frames | **~25% total speedup** (removes redundant inference) |
| **Guidance Scale**| Default **5.0 → 7.0** | **Higher contrast**, clearer animation |
| **Steps** | Default **30 → 20** (for 1.3b) | **33% faster** warm inference |
| **Field Promotion**| `video_url` promoted to top-level | Fixes "video_url: None" in API response |

**Performance (WAN T2V 1.3B, 15s/240fr/480p):**
- Expected: **~12 min** warm (was 42 min)
- result: **Higher Quality** (7.0 guidance) and **Distortion-Free** (native resolution)

**Note**: PyTorch 2.5 native flash SDPA already enabled (`enable_flash_sdp(True)`). No separate flash_attn package needed.

---

## SMOKE TEST RESULTS SUMMARY

| Test | Config | Status | Time |
|------|--------|--------|------|
| WAN T2V 14B 5s | 81fr/50st/720p | **COMPLETED** | ~80 min (30min HDD load + 50min inf, HDD contention) |
| WAN T2V 14B 15s | 240fr/50st/81fr-chunks/720p | **ABORTED** @ 63.6% | Was 4-5 hours — wrong config |
| WAN T2V 14B HQ (smoke) | 49fr/50st/720p | **PASS** | 40 min (2395s total, 493s inf) |
| WAN I2V 14B HQ (smoke) | 33fr/50st/720p | **PASS** | 21 min (1283s total, 187s inf) |
| WAN T2V 1.3B chunked | 240fr/720p | **PASS** | 42.5 min |

**Test 1 output saved locally**: `d:/Flux_Lora/test_outputs/wan14b_hq_t2v_5s_81fr_720p.mp4` (3.5MB)

---

## PENDING WORK (next session)

### Priority 1 — Deploy Vultr 8× A100 instance
The multi-GPU code changes are DONE (2026-05-09). Need to provision and test:
```bash
# Provision instance (takes ~10 min)
export VULTR_API_KEY="..." SSH_KEY_ID="..."
bash flux-server/deploy/vultr/launch.sh

# After instance is up, run setup_storage.sh on it, copy .env, start service
source flux-server/deploy/vultr/.instance
ssh root@$PUBLIC_IP
```
Then verify all 8 GPUs are seen by the container and 8 gunicorn workers start.

### Priority 2 — Deploy UI + library changes (included in same deploy)
Includes: Library tab, quality presets, 15s video preset, 12 sample images, tab animations.
These are already committed on `codex/hyperforge-runtime-hardening`.

### Priority 3 — Re-run WAN T2V 14B 15s test on Vultr
Use updated test script: `test_hq_wan14b.sh` (49fr chunks, 20 steps).
On Vultr with NVMe block storage, WAN 14B load time should be ≤5 min vs 30 min on GCP HDD.

### Priority 4 — Complete HunyuanVideo download (~38GB remaining)
Use `download_hunyuan_v4.py`. Rules:
- Run ONLY during active inference (model in VRAM) — NOT during model load
- `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"` BEFORE any imports
- Delete `.incomplete` blobs before restart if stalled
- Run inside container: `docker exec container bash -c 'nohup python3 /tmp/download_hunyuan_v4.py > /tmp/log 2>&1 &'`

### Priority 5 — Parallel chunk generation across GPUs (future, ~2-3 days)
True parallel chunk inference using torch.multiprocessing or subprocess-per-GPU.
Not yet implemented — current state: 8 independent workers, each handles one job sequentially.

### RESOLVED (GCP) — Phase 1 NVMe SSD
Vultr NVMe block storage already solves the 30min model load issue on GCP.

### RESOLVED — Docker GPU lock
`count: all` + no `CUDA_VISIBLE_DEVICES` in compose. gunicorn post_fork assigns GPUs.

---

## KEY TECHNICAL FACTS

### WAN 14B VRAM (NF4 double-quantized)
- NF4 transformer: ~7 GB
- NF4 transformer_2 (WAN 2.2 only): ~7 GB
- UMT5 text encoder: ~10.5 GB
- VAE: ~1 GB
- **Total: ~25.5 GB** (fits A100 40GB with 14GB headroom)
- Full inference usage: ~36.2 GB

### WAN 14B Loading
- From HDD: ~30 min (118GB / ~65 MB/s)
- Model STAYS in VRAM between consecutive jobs (no idle TTL)
- Only unloads if: different model requested, OOM, container restart

### Inference Rate (A100, NF4, 720p)
- WAN 14B @ 49fr: 493s / 50 steps = **~10s per step** (total 8.2 min inference)
- WAN 14B @ 81fr: quadratic attention → ~22s per step (total ~55 min inference)
- At 20 steps, 49fr: **~3.3 min per chunk**

### Chunked Video Math (49fr chunks, 16fr overlap, 20 steps)
- step = 49 - 16 = 33 frames per chunk
- 240 frames → 7 chunks
- Expected: 7 × 3.3 min = **23 min inference** (model warm)

---

## CRITICAL RULES — DO NOT VIOLATE

### 1. Resource conflicts (HDD)
NEVER run model download and model loading simultaneously on the same HDD.
- WAN 14B loads FROM HDD → 30 min, 65 MB/s read
- Hunyuan downloads TO HDD → 7 MB/s write
- Simultaneous = 2-3× slowdown on BOTH = wasted session
- Safe: download DURING active inference (model fully in VRAM, HDD idle)

### 2. Check active jobs before submitting
```bash
curl -s http://localhost:8080/api/jobs | python3 -c "import sys,json; [print(j.get('job_id'), j.get('status'), j.get('model_name')) for j in json.load(sys.stdin)]"
```
poll_job script timeout only stops the script — server job keeps running.

### 3. Verify model cache before testing
```bash
du -sh /app/model_cache/models--<org>--<model>
find /app/model_cache/models--<org>--<model> -name '*.incomplete' | wc -l
```
Never trust plan docs for download status — always check filesystem.

### 4. hf_transfer must be explicitly disabled
```python
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # "0" not pop() — must be before imports
```
If RSS > 2GB = hf_transfer active. Kill, delete .incomplete blobs, restart.

### 5. nohup inside container, not outside
```bash
# WRONG — dies on SSH disconnect:
nohup sudo docker exec container python3 script.py &
# CORRECT — survives SSH disconnect:
sudo docker exec container bash -c 'nohup python3 /tmp/script.py > /tmp/log 2>&1 &'
```

### 6. Test simplest path first
Before debugging 14B model issues, always verify WAN 1.3B works end-to-end.

### 7. BitsAndBytes NF4 rules
- NEVER call `.to(device)` on a quantized pipeline — crashes
- NEVER use `device_map="balanced"` with BnB
- Both `transformer` and `transformer_2` must be NF4-quantized (WAN 2.2 has both)

### 8. torch.compile skipped for NF4
Expected behavior. NF4 custom CUDA ops are incompatible with compile graph capture.

---

## IMPORTANT FILES

| File | Purpose |
|------|---------|
| `d:/Flux_Lora/implementation_plan.md` | Full plan with DONE/PENDING status |
| `d:/Flux_Lora/flux-server/AGENT.md` | Server code context for agents |
| `d:/Flux_Lora/test_hq_wan14b.sh` | HQ test script (updated: 49fr/20 steps) |
| `d:/Flux_Lora/test_hq_hunyuan.sh` | HunyuanVideo test script |
| `d:/Flux_Lora/download_hunyuan_v4.py` | Working Hunyuan download script |
| `d:/Flux_Lora/test_outputs/` | Downloaded test videos |
| `flux-server/app/pipelines/video_pipeline.py` | All video generation logic |
| `flux-server/app/main.py` | API routing + job dispatch |
| `flux-server/app/model_manager.py` | Cache tier selection, model loading |

---

## GCP QUICK COMMANDS

```bash
# Start VM
gcloud compute instances start flux-a100-preemptible --zone=us-central1-a

# SSH
gcloud compute ssh krishnagopalmishra1-we@flux-a100-preemptible --zone=us-central1-a

# Check container + GPU
sudo docker ps | grep flux && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# Check active jobs
curl -s http://localhost:8080/api/jobs | python3 -c "import sys,json; [print(j.get('job_id')[:8], j.get('status'), j.get('progress')) for j in json.load(sys.stdin)]"

# Stop VM (saves billing)
gcloud compute instances stop flux-a100-preemptible --zone=us-central1-a

# Deploy after code changes
cd /opt/flux-server/flux-server && git pull && sudo docker compose up --build -d

# Copy file to VM
gcloud compute scp <local_file> krishnagopalmishra1-we@flux-a100-preemptible:/tmp/ --zone=us-central1-a

# Copy file from VM (need sudo cp to home first)
sudo cp /path/file.mp4 /home/krishnagopalmishra1-we/file.mp4
gcloud compute scp krishnagopalmishra1-we@flux-a100-preemptible:/home/krishnagopalmishra1-we/file.mp4 d:/Flux_Lora/test_outputs/ --zone=us-central1-a
```

---

## SESSION START CHECKLIST

Before doing anything:
1. Check VM status: `gcloud compute instances list --filter="name=flux-a100-preemptible"`
2. If starting VM: check for active jobs before submitting new ones
3. Check Hunyuan download status (if relevant): `du -sh /app/model_cache/models--hunyuanvideo-community--HunyuanVideo`
4. If testing WAN 14B: confirm model NOT loading during any HDD write operations
5. Use updated test scripts (49fr/20 steps) — old scripts (81fr/50 steps) caused 4-5 hr runs
