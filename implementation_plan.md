# NEURAL CREATION STUDIO — IMPLEMENTATION PLAN
# Target: 1–2 Minute Video in ≤30 Minutes Without Quality Loss
# Last Updated: 2026-04-12 (Post-Smoke-Test)
# Legend: DONE | IN PROGRESS | PENDING | BLOCKED (needs infra)

---

## PHASE 1 — Infrastructure

| # | Action | Status | Notes |
|---|--------|--------|-------|
| 1.1 | Attach 256GB NVMe pd-ssd to VM | BLOCKED | Pending GCP provisioning |
| 1.2 | Mount NVMe at /ssd/model_cache, add to /etc/fstab | BLOCKED | Depends on 1.1 |
| 1.3 | Add cache_dir_nvme to config.py | BLOCKED | Depends on 1.1 |
| 1.4 | Update model_manager.py NVME_PRIORITY for WAN 14B + I2V 14B | BLOCKED | Depends on 1.1 |
| 1.5 | Move WAN 14B + I2V 14B to NVMe (14B load: 11min -> 2min) | BLOCKED | Depends on 1.1 |
| 1.6 | Make HDD readahead persistent via udev rule | DONE | Applied 2026-04-11, udev rule deployed |

---

## PHASE 2 — Core Code Fixes

| # | Action | Status | Notes |
|---|--------|--------|-------|
| 2.1 | Delete pipeline.py (212 dead lines) | DONE | Removed in commit f887a52 |
| 2.2 | Delete smoke_test_full.sh, smoke_test_v2.sh | DONE | Deleted 2026-04-12 |
| 2.3 | Delete SMOKE_TEST_PROGRESS_2026-04-04.md | DONE | Deleted 2026-04-12 |
| 2.4 | main.py: remove flux_pipeline dead import + unload_all() | DONE | commit f887a52 |
| 2.5 | main.py: remove /datasets/*, /training/*, /generate-ui endpoints | DONE | commit f887a52 |
| 2.6 | model_manager.py: remove dead FLUX local transformer code (lines 374-388) | DONE | commit f887a52 |
| 2.7 | model_manager.py: remove incompatible device_map="balanced" | DONE | commit f887a52 |
| 2.8 | model_manager.py: get_cache_dir() as single source of truth for SSD/HDD | DONE | commit f887a52 |
| 2.9 | video_pipeline.py: remove duplicate SSD_PRIORITY | DONE | commit f887a52 |
| 2.10 | video_pipeline.py: remove LTX dead branch | DONE | commit f887a52 |
| 2.11 | video_pipeline.py: keepalive 30s -> 5s | DONE | commit f887a52 |
| 2.12 | video_pipeline.py: OOM recovery order: steps -> frames -> cpu_offload | DONE | commit f887a52 |
| 2.13 | job_queue.py: add cancel_flag to Job, cancel_job() works on PROCESSING | DONE | commit f887a52 |
| 2.14 | job_queue.py: decouple _gpu_lock_image / _gpu_lock_video | DONE | commit f887a52 |
| 2.15 | job_queue.py: add estimated_time_remaining to job status | DONE | commit f887a52 |
| 2.16 | schemas_v2.py: raise num_frames 81->1920, steps 50->100, add 540p, chunk fields | DONE | commit f887a52 |
| 2.17 | requirements.txt: remove librosa, soundfile, face-alignment (dead deps) | DONE | commit f887a52 |
| 2.18 | smoke_final.sh: fix timeouts (T2V 14B->4500s, 1.3B->2400s, I2V->4500s) | DONE | Fixed 2026-04-12 |
| 2.19 | output_store.py: add disk space guard before generation | DONE | commit f887a52 |
| 2.20 | main.py: periodic output cleanup cron on startup | DONE | commit f887a52 |

---

## PHASE 3 — Long-Video Chunked Generation

| # | Action | Status | Notes |
|---|--------|--------|-------|
| 3.1 | video_pipeline.py: generate_long_video() with sliding window + cosine blend | DONE | commit f887a52 |
| 3.2 | main.py: auto-route num_frames>81 to generate_long_video() | DONE | commit f887a52 |
| 3.3 | Test 240-frame (15s) chunked generation end-to-end | DONE | Verified with WAN 1.3B (~42 min) |
| 3.4 | Test 960-frame (60s) chunked generation | PENDING | VM stopped; planned for next session |

---

## PHASE 4 — Performance Optimizations

| # | Action | Status | Notes |
|---|--------|--------|-------|
| 4.1 | video_pipeline.py: torch.compile for Ampere+ GPU (A100) | DONE | commit f887a52 |
| 4.2 | video_pipeline.py: torch.inference_mode() wrapping inference | DONE | commit f887a52 |
| 4.3 | video_pipeline.py: VAE tiling + slicing for all models | DONE | commit f887a52 |

---

## PHASE 5 — Queue & API Hardening

| # | Action | Status | Notes |
|---|--------|--------|-------|
| 5.1 | job_queue.py: asyncio.PriorityQueue with image FAST-path | DONE | commit f887a52 |
| 5.2 | main.py: GET /api/jobs list endpoint | DONE | commit f887a52 |
| 5.3 | main.py: POST /api/admin/queue/drain | DONE | commit f887a52 |

---

## DEPLOYMENT

| # | Action | Status | Notes |
|---|--------|--------|-------|
| D1 | Make readahead persistent (udev rule /dev/sdb) | DONE | Applied 2026-04-11 |
| D2 | git pull + docker compose up --build on VM | DONE | Deployed 2026-04-12 |
| D2.5 | Fix NF4 pipe.to('cuda') crash (FLUX + SD3.5) | DONE | commit b4a6f71, redeployed |
| D2.6 | Fix video OOM: unload FLUX before video job starts | DONE | commit 91736f2 — needs redeploy |
| D3 | Re-run smoke test after deploy | DONE | Ph1-3 PASS, Ph4 OOM (fixed), Ph5 PASS |
| D3.5 | Isolated re-test: WAN T2V 14B + WAN I2V 14B | DONE | T2V 49fr / I2V 33fr PASS (A100) |
| D4 | Test chunked 240-frame generation | DONE | Passed with 1.3B (default behavior) |

---

## SMOKE TEST FINDINGS (2026-04-12)

- **WAN 14B HQ (T2V)**: 49 frames. Total 2395s (40 min). Inference: 493s (8.2 min).
- **WAN 14B HQ (I2V)**: 33 frames. Total 1283s (21 min). Inference: 187s (3.1 min).
- **Chunked 240fr**: 1.3B model. Total 2551s (42.5 min). 
- **Discovery**: `generate_long_video` defaults to `1.3b`. If `14b` is intended for long videos, it must be explicitly set in the API request.
- **Hunyuan DL**: ~65GB in SSD cache volume. DONE.

---

## BLOCKED — Waiting on GCP NVMe SSD

When NVMe is provisioned:
1. `gcloud compute disks create flux-model-nvme --zone=us-central1-a --type=pd-ssd --size=256GB`
2. `gcloud compute instances attach-disk flux-a100-preemptible --disk=flux-model-nvme --zone=us-central1-a`
3. SSH: `sudo mkfs.ext4 /dev/sdc && sudo mkdir /ssd && sudo mount /dev/sdc /ssd`
4. Add to /etc/fstab: `/dev/sdc /ssd ext4 defaults 0 2`
5. Update config.py: add `cache_dir_nvme: str = "/ssd/model_cache"`
6. Update model_manager.py: add NVME_PRIORITY = {"wan-t2v-14b", "wan-i2v-14b"}
7. Move models: `sudo mv /mnt/hf-cache/Wan-AI/Wan2.2-T2V-A14B-Diffusers /ssd/model_cache/`
8. Move models: `sudo mv /mnt/hf-cache/Wan-AI/Wan2.2-I2V-A14B-Diffusers /ssd/model_cache/`
9. Re-test: 14B load time target 11min -> 2min

---

## GENERATION TIME TARGETS

| Model | Video Length | Load (now) | Load (NVMe) | Inference | Total Now | Total NVMe |
|-------|-------------|-----------|------------|-----------|-----------|-----------|
| WAN 1.3B | 5 sec / 81fr | 8min cold / instant warm | same | ~11 min | 11 min warm | 11 min |
| WAN 1.3B | 60 sec / chunked | instant warm | same | ~22 min | 22 min | 22 min |
| WAN 1.3B | 120 sec / chunked | instant warm | same | ~44 min | 44 min | 44 min |
| WAN T2V 14B | 5 sec / 49fr | 30 min | 2 min | ~8 min | 40 min | 10 min |
| WAN T2V 14B | 60 sec / chunked | 30 min | 2 min | ~35 min | 65 min | 37 min |
| WAN I2V 14B | 5 sec / 33fr | 18 min | 2 min | ~3 min | 21 min | 5 min |
| WAN 1.3B | 15 sec / 240fr | ~2 min | ~2 min | ~40 min | 42 min | 42 min |

*Actual observed load times on HDD are significantly higher than initial estimates (18-30 mins vs 11 mins).*
