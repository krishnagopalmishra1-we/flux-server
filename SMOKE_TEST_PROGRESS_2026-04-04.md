# Smoke Test Progress - 2026-04-04

## Run Context
- Server used during run: `http://136.114.162.63:8080` (ephemeral IP)
- VM: `flux-a100-preemptible` (`us-central1-a`)
- Smoke script: `flux-server/tools/remote_smoke_test.py`
- Log file: `flux-server/tools/remote_smoke_run.log`
- VM billing state after checkpoint: **TERMINATED**

## Confirmed Passed Today
- Health endpoint: PASS
- Model catalog load (13 models): PASS
- Image model `flux-1-dev`: PASS (attempt 1)
- Image model `sd3.5-large`: PASS (attempt 1)
- Image model `realvisxl-v5`: PASS (attempt 1)
- Image model `juggernaut-xl`: PASS (attempt 1)

## In Progress When Stopped
- Video model `wan-t2v-1.3b`: started, then intentionally stopped to save cost

## Remaining Models To Test
### Video
- `wan-t2v-1.3b` (restart from this model)
- `wan-t2v-14b`
- `wan-i2v-14b`
- `ltx-video`

### Music
- `ace-step`
- `audioldm2`
- `stable-audio`

### Animation
- `liveportrait`
- `echomimic`

## Resume Steps for Tomorrow
1. Start VM:
   - `gcloud compute instances start flux-a100-preemptible --zone=us-central1-a --project=flux-lora-gpu-project`
2. Get current external IP:
   - `gcloud compute instances describe flux-a100-preemptible --zone=us-central1-a --project=flux-lora-gpu-project --format="value(networkInterfaces[0].accessConfigs[0].natIP)"`
3. Wait for API health:
   - `Invoke-RestMethod "http://<NEW_IP>:8080/health"`
4. Continue smoke test:
   - `d:/Flux_Lora/.venv/Scripts/python.exe d:/Flux_Lora/flux-server/tools/remote_smoke_test.py --server http://<NEW_IP>:8080 --retries 2`

## Notes
- Long delays are expected for first-time heavy model loads (especially SD3.5 and Wan family).
- Preemptible VM restarts can interrupt long tests; checkpointing after each major block avoids lost progress.

## Carry-over TODO (Next Session)
- [ ] Start VM and fetch fresh external IP
- [ ] Verify `/health` and `/models` are reachable
- [ ] Resume smoke test from video block (start with `wan-t2v-1.3b`)
- [ ] Complete remaining video models (`wan-t2v-14b`, `wan-i2v-14b`, `ltx-video`)
- [ ] Complete music models (`ace-step`, `audioldm2`, `stable-audio`)
- [ ] Complete animation models (`liveportrait`, `echomimic`)
- [ ] Record per-model pass/fail with error details for failed models
- [ ] Stop VM after test to avoid billing

---

## Update - 2026-04-08 (Claude run checkpoint)

### Verified Results (from Claude output)
- `wan-t2v-1.3b`: PASS (~103s, job `cd2d515f`)
- `wan-t2v-14b`: PASS (job `8f912753`)
- `wan-i2v-14b`: FAIL (HuggingFace/cache download error during cold start)
- `ltx-video`: started (no final result captured before session limit)

### Interpretation
- The confirmed failure in captured logs is **not** on `wan-t2v-14b`; it passed.
- The failure happened on `wan-i2v-14b`, and appears transient network/cache related.

### Remaining From This Checkpoint
- [x] Re-run `wan-i2v-14b` — PASS (~331s, cache cold-start resolved on retry)
- [x] Complete `ltx-video` — PASS (~278s)
- [x] `ace-step` — PASS (~232s)
- [x] `audioldm2` — PASS (~110s)
- [x] `liveportrait` — PASS (~18s)
- [x] `echomimic` — PASS (~18s)
- [x] `stable-audio` — **FAIL** — 403 Client Error (HuggingFace gated model; not a transient issue)

---

## Final Smoke Test Summary (2026-04-08)

| Category  | Model            | Status | Notes                                  |
|-----------|------------------|--------|----------------------------------------|
| Image     | flux-1-dev       | PASS   | (prev session)                         |
| Image     | sd3.5-large      | PASS   | (prev session)                         |
| Image     | realvisxl-v5     | PASS   | (prev session)                         |
| Image     | juggernaut-xl    | PASS   | (prev session)                         |
| Video     | wan-t2v-1.3b     | PASS   | ~103s                                  |
| Video     | wan-t2v-14b      | PASS   | passes                                 |
| Video     | wan-i2v-14b      | PASS   | ~331s cold-start; HF error on attempt1 was transient |
| Video     | ltx-video        | PASS   | ~278s                                  |
| Music     | ace-step         | PASS   | ~232s                                  |
| Music     | audioldm2        | PASS   | ~110s                                  |
| Music     | stable-audio     | **FAIL** | 403 HuggingFace model access denied — gated model, requires accepted ToS |
| Animation | liveportrait     | PASS   | ~18s                                   |
| Animation | echomimic        | PASS   | ~18s                                   |

**Score: 12 PASS / 1 FAIL**

### Action Required
- `stable-audio` (`stabilityai/stable-audio-open-1.0`) requires HuggingFace gated access.
  - Option A: Accept the model ToS on HF under the GCP VM's token account
  - Option B: Replace model with `audioldm2` as the default audio fallback
  - Option C: Remove `stable-audio` from the catalog if not needed
