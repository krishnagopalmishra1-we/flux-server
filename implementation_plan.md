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
| 2.6 | model_manager.py: remove dead FLUX local transformer code | DONE | commit f887a52 |
| 2.7 | model_manager.py: remove incompatible device_map="balanced" | DONE | commit f887a52 |
| 2.8 | model_manager.py: get_cache_dir() as single source of truth | DONE | commit f887a52 |
| 2.9 | video_pipeline.py: remove duplicate SSD_PRIORITY | DONE | commit f887a52 |
| 2.10 | video_pipeline.py: remove LTX dead branch | DONE | commit f887a52 |
| 2.11 | video_pipeline.py: keepalive 30s -> 5s | DONE | commit f887a52 |
| 2.12 | video_pipeline.py: OOM order: steps -> frames -> cpu_offload | DONE | commit f887a52 |
| 2.13 | job_queue.py: add cancel_flag, cancel_job() processing | DONE | commit f887a52 |
| 2.14 | job_queue.py: decouple _gpu_lock_image / _gpu_lock_video | DONE | commit f887a52 |
| 2.15 | job_queue.py: add estimated_time_remaining to status | DONE | commit f887a52 |
| 2.16 | schemas_v2.py: raise frames 1920, steps 100, add chunk fields | DONE | commit f887a52 |
| 2.17 | requirements.txt: remove dead deps (librosa, etc) | DONE | commit f887a52 |
| 2.18 | smoke_final.sh: fix timeouts (14B->4500s, 1.3B->2400s) | DONE | Fixed 2026-04-12 |
| 2.19 | output_store.py: add disk space guard before generation | DONE | commit f887a52 |
| 2.20 | main.py: periodic output cleanup cron on startup | DONE | commit f887a52 |

---

## PHASE 3 — Long-Video Chunked Generation

| # | Action | Status | Notes |
|---|--------|--------|-------|
| 3.1 | video_pipeline.py: generate_long_video() with sliding window | DONE | commit f887a52 |
| 3.2 | main.py: auto-route num_frames>81 to generate_long_video() | DONE | commit f887a52 |
| 3.3 | Test 240-frame (15s) chunked generation | DONE | Verified with WAN 1.3B (~42 min) |
| 3.3b | Test 240-frame (15s) WAN 14B HQ chunked | IN PROGRESS | Job f131c0d2, 52.4% at 12:35 UTC 2026-04-13, ETA ~14:30 UTC |
| 3.4 | Test 960-frame (60s) chunked generation | PENDING | Next session after 3.3b completes |

---

## PHASE 6 — SECURITY HARDENING (PENDING)

| # | Action | Status | Notes |
|---|--------|--------|-------|
| 6.1 | .claude/settings.local.json: remove gcloud compute:* wildcard | PENDING | Replace with explicit subcommands |
| 6.2 | .claude/settings.local.json: remove PII/IP/User hardcoding | PENDING | Replace with ${FLUX_REMOTE_IP}/${FLUX_REMOTE_USER} |
| 6.3 | .claude/settings.local.json: remove StrictHostKeyChecking=no | PENDING | Enforce proper host key management |
| 6.4 | .claude/settings.local.json: remove sudo from docker ps | PENDING | Use non-root docker group access |
| 6.5 | .claude/settings.local.json: switch curl to https + timeout 10 | PENDING | Use https://${FLUX_REMOTE_IP} |

---

## DEPLOYMENT

| # | Action | Status | Notes |
|---|--------|--------|-------|
| D1 | Make readahead persistent (udev rule /dev/sdb) | DONE | Applied 2026-04-11 |
| D2 | git pull + docker compose up --build on VM | DONE | Deployed 2026-04-12 |
| D3 | Re-run smoke test after deploy | DONE | Ph1-3 PASS, Ph4 OOM (fixed), Ph5 PASS |
| D3.5 | Isolated re-test: WAN T2V 14B + WAN I2V 14B | DONE | T2V 49fr / I2V 33fr PASS (A100) |
| D4 | Test chunked 240-frame generation | DONE | Passed with 1.3B (default behavior) |

---

## SMOKE TEST FINDINGS (2026-04-12)

- **WAN 14B HQ (T2V)**: 49 frames. Total 2395s (40 min). Inference: 493s (8.2 min).
- **WAN 14B HQ (I2V)**: 33 frames. Total 1283s (21 min). Inference: 187s (3.1 min).
- **Chunked 240fr**: 1.3B model. Total 2551s (42.5 min). 
- **Discovery**: `generate_long_video` defaults to `1.3b`. If `14b` is intended for long videos, it must be explicitly set in the API request.
- **WAN 14B HQ T2V 5s (81fr, 2026-04-13)**: COMPLETED. Total ~80 min (incl. 30min HDD load). Output: /outputs/video/98c9edaa-806_1776076782.mp4 (3.5MB). Downloaded locally: test_outputs/wan14b_hq_t2v_5s_81fr_720p.mp4
- **WAN 14B HQ T2V 15s (240fr chunked, 2026-04-13)**: IN PROGRESS. Job f131c0d2, 52.4% at 12:35 UTC. ETA ~14:30 UTC. Pure GPU-bound (~220min for 4×81fr chunks).
- **Hunyuan DL**: ~1.7 GB / ~40 GB (v4 download restarted 12:36 after stall, hf_transfer disabled, writing to blob cache). Requires separate session to complete.
- **Download lesson**: hf_transfer stalls on resume of large .incomplete blobs. Fix: delete all .incomplete, restart fresh. v4 script confirmed working.

---

## BLOCKED — Waiting on GCP NVMe SSD

1. `gcloud compute disks create flux-model-nvme --size=256GB`
2. `gcloud compute instances attach-disk flux-a100-preemptible --disk=flux-model-nvme`
3. `sudo mkfs.ext4 /dev/sdc && sudo mount /dev/sdc /ssd`
4. Update `config.py` + `model_manager.py` for `cache_dir_nvme`.

---

## GENERATION TIME TARGETS (A100)

| Model | Intensity | Load (HDD Now) | Inference | Total |
|-------|-----------|----------------|-----------|-------|
| WAN 1.3B | Chunked (240fr)| ~2 min | ~40 min | 42 min |
| WAN T2V 14B | HQ (49fr) | ~30 min | ~8 min | 40 min |
| WAN I2V 14B | HQ (33fr) | ~18 min | ~3 min | 21 min |
