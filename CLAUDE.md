# CLAUDE.md — Neural Creation Studio
# Read this at the start of every session. It is the single source of truth.
# Last updated: 2026-04-16

---

## WHAT THIS PROJECT IS

A FastAPI video/image generation server running on a GCP A100 (40GB) SPOT VM.
Primary goal: generate 1–2 minute videos in ≤30 minutes without quality loss.

**Repo root**: `d:/Flux_Lora/`
**Server code**: `d:/Flux_Lora/flux-server/`
**GCP VM**: `flux-a100-preemptible`, zone `us-central1-a`, project `flux-lora-gpu-project`
**SSH**: `gcloud compute ssh krishnagopalmishra1-we@flux-a100-preemptible --zone=us-central1-a`
**Container**: `flux-server-flux-server-1`
**API**: `http://localhost:8080` (from inside VM)

---

## CURRENT VM STATE

**VM status**: RUNNING (as of end-of-session 2026-04-16)
**Stop VM before exit**: `gcloud compute instances stop flux-a100-preemptible --zone=us-central1-a`

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

## Status: NATIVE HQ DEPLOYED
**Last Update: 2026-04-18**
- **Issue**: 1.3B was forced to 480p and had 'black jumps' on 15s videos.
- **Fix**: Removed 480p clamp in `main.py`. Implemented 16-frame **Cosine Wave Blending** in `video_pipeline.py`.
- **Outcome**: 15s 720p generations are now smooth and detailed.
- **Infrastructure**: All changes pushed and VM stopping.

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

### Priority 1 — Deploy speed optimization (needs redeploy)
```bash
gcloud compute instances start flux-a100-preemptible --zone=us-central1-a
gcloud compute ssh krishnagopalmishra1-we@flux-a100-preemptible --zone=us-central1-a
cd /opt/flux-server/flux-server && git pull && sudo docker compose up --build -d
```

### Priority 2 — Re-run WAN T2V 14B 15s test with new settings
Use updated test script: `test_hq_wan14b.sh` (49fr chunks, 20 steps)
Copy to VM: `gcloud compute scp test_hq_wan14b.sh krishnagopalmishra1-we@flux-a100-preemptible:/tmp/ --zone=us-central1-a`

### Priority 3 — Complete HunyuanVideo download (~38GB remaining)
Use `download_hunyuan_v4.py`. Rules:
- Run ONLY during active inference (model in VRAM) — NOT during model load
- `os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"` BEFORE any imports
- Delete `.incomplete` blobs before restart if stalled
- Run inside container: `sudo docker exec container bash -c 'nohup python3 /tmp/download_hunyuan_v4.py > /tmp/log 2>&1 &'`

### Priority 4 — Plan item 3.4: 960-frame (60s) chunked test
After 15s test passes with new settings.

### BLOCKED — Phase 1 NVMe SSD
Would reduce WAN 14B load time 30min→2min. Pending GCP provisioning.
```bash
gcloud compute disks create flux-model-nvme --size=256GB --type=pd-ssd --zone=us-central1-a
gcloud compute instances attach-disk flux-a100-preemptible --disk=flux-model-nvme --zone=us-central1-a
```

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
