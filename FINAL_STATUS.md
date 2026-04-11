# NEURAL CREATION STUDIO — FINAL STATUS REPORT

**Last Updated**: April 11, 2026  
**Session**: Smoke Test, Quality Validation, Performance Tuning  
**Status**: 🟢 All Image Models Passing | 🟡 Video Models Working (load time optimization needed)

---

## LIVE SERVER

**VM**: `flux-a100-preemptible` — GCP `us-central1-a`  
**GPU**: NVIDIA A100-SXM4-40GB (39.49 GB VRAM)  
**Port**: 8080  
**Auth**: API key required (`X-API-Key` header)

---

## MODEL STATUS (Tested 2026-04-11)

### Image Models — All PASSING

| Model | Key | Resolution | Steps | Result |
|-------|-----|-----------|-------|--------|
| FLUX.1-dev | `flux-1-dev` | 1024x1024 | 28 | PASS (~46s) |
| SD3.5 Large | `sd3.5-large` | 768x768 | 28 | PASS |
| RealVisXL V5.0 | `realvisxl` | 512x512 | 20 | PASS |
| Juggernaut XL v9 | `juggernaut-xl` | 512x512 | 20 | PASS |

### Video Models — Working, Load Time Optimization Needed

| Model | Key | Cache | Load Time | Inference | Status |
|-------|-----|-------|-----------|-----------|--------|
| WAN T2V 1.3B | `wan-t2v-1.3b` | SSD (27GB) | ~8 min cold | ~11 min | PASS |
| WAN T2V 14B | `wan-t2v-14b` | HDD (118GB) | ~11 min* | ~30 min est. | Works, slow load |
| WAN I2V 14B | `wan-i2v-14b` | HDD (118GB) | ~11 min* | ~15 min est. | Works, slow load |

*With HDD readahead tuning applied (was ~14 min before).

### API Limits (enforced by validators)
- `num_frames`: max 81 = ~5s video at 16fps
- `num_inference_steps`: max 50 for WAN 1.3B
- Resolutions: `480p` (848x480), `720p` (1280x720), `540p` (544x960)

---

## SMOKE TEST RESULTS (2026-04-11)

| Phase | Test | Result | Notes |
|-------|------|--------|-------|
| 1 | System Health | PASS | A100 healthy |
| 2 | Python/CUDA Diagnostics | PASS | Flash SDP enabled, TF32 on |
| 3 | FLUX 1-dev 1024x1024/28 steps | PASS | 46s |
| 4 | WAN T2V 14B 480p/49fr/50 steps | TIMEOUT | Script limit 1800s too short; job completed after |
| 5 | WAN T2V 1.3B 720p/81fr/50 steps | TIMEOUT | Script limit 600s too short |
| 6 | WAN I2V 14B 480p/33fr/30 steps | TIMEOUT | Script limit 900s too short |
| 7 | Final VRAM | — | 2342 MiB used |

All 3 video TIMEOUTs are script timeout issues, NOT model failures. Jobs completed after script exited.

### Quality Test — WAN T2V 1.3B (2026-04-11)

Isolated test: 720p, 81 frames, 50 steps, guidance=7.5

- Result: PASS
- Inference time: 676s (~11 min)
- Output: 2.9MB MP4, H.264, ~5 seconds at 720p
- Quality: Acceptable (reviewed by user)

---

## PERFORMANCE TUNING APPLIED

### HDD Readahead Tuning
```bash
sudo blockdev --setra 65536 /dev/sdb   # 128KB -> 32MB
```
- Before: ~71s/shard for WAN 14B (12 shards = ~14 min total)
- After: ~55s/shard = ~11 min total (22% faster)
- WARNING: Not persistent across reboots. Re-apply after each VM start.

To make permanent:
```bash
echo 'ACTION=="add", KERNEL=="sdb", RUN+="/sbin/blockdev --setra 65536 /dev/sdb"' \
  | sudo tee /etc/udev/rules.d/60-hdd-readahead.rules
```

---

## DISK LAYOUT

| Disk | Mount | Size | Contents |
|------|-------|------|----------|
| SSD (sda) | / | 250GB | OS + Docker + SSD model cache |
| HDD (sdb) | /mnt/hf-cache-disk | 500GB | WAN 14B models |

SSD model cache (/app/model_cache):
- FLUX.1-dev: ~32GB
- WAN T2V 1.3B: ~27GB
- HunyuanVideo: ~20GB

HDD model cache (/mnt/hf-cache):
- WAN T2V 14B: ~118GB
- WAN I2V 14B: ~118GB

---

## NEXT STEPS (Priority Order)

1. Make HDD readahead persistent (udev rule above)
2. Add job cancel endpoint: DELETE /api/jobs/{job_id}
3. Raise API frame/step limits for higher quality testing
4. Update smoke_final.sh video timeouts (14B: 4500s, 1.3B: 2400s, I2V: 4500s)
5. Test WAN T2V 14B quality at 480p/49fr/50 steps (isolated, no queue)

---

## REMOVED FROM THIS PROJECT

- Music generation (AudioLDM2, MusicGen, ACE-Step, Stable Audio)
- Animation generation (EchoMimic, LiveAvatar)
- LTX Video
- HunyuanVideo (on SSD but not validated this session)

---

## KEY COMMANDS

```bash
# Start VM
gcloud compute instances start flux-a100-preemptible --zone=us-central1-a

# Get current IP (changes on each start)
gcloud compute instances describe flux-a100-preemptible --zone=us-central1-a \
  --format="value(networkInterfaces[0].accessConfigs[0].natIP)"

# SSH
gcloud compute ssh flux-a100-preemptible --zone=us-central1-a

# Re-apply readahead after VM start (do this every time)
sudo blockdev --setra 65536 /dev/sdb

# Check health
curl -s http://SERVER_IP:8080/health

# Submit WAN 1.3B job (max quality)
curl -s -X POST http://SERVER_IP:8080/api/video/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"prompt":"...","model_name":"wan-t2v-1.3b","resolution":"720p","num_frames":81,"fps":16,"guidance_scale":7.5,"num_inference_steps":50}'

# Download output video from VM
gcloud compute ssh flux-a100-preemptible --zone=us-central1-a \
  --command="sudo cp /var/lib/docker/volumes/flux-server_outputs/_data/video/JOB_ID.mp4 /tmp/out.mp4 && sudo chmod 644 /tmp/out.mp4"
gcloud compute scp flux-a100-preemptible:/tmp/out.mp4 ./output.mp4 --zone=us-central1-a
```
