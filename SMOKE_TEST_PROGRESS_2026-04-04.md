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
