# CLAUDE.md - Hyperforge Session Handoff
# Read this at the start of every session. It is the single source of truth.
# Last updated: 2026-05-09 (session 2)

---

## WHAT THIS PROJECT IS

A FastAPI image/video generation server in `d:/Flux_Lora/flux-server/`.

Current implementation target:
- deploy-ready WAN 14B quality path on multi-GPU hosts
- keep `/api/video/generate` unchanged
- improve WAN 14B output quality first, not just throughput

Primary quality strategy:
- WAN T2V 14B should prefer BF16 xDiT on 4x80GB GPU groups
- fallback remains NF4 diffusers on unsupported environments

---

## CANONICAL CODEBASE

- Repo root: `d:/Flux_Lora/`
- Server code: `d:/Flux_Lora/flux-server/`
- Canonical backend entrypoint: `flux-server/app/main.py`
- Canonical video runtime: `flux-server/app/pipelines/video_pipeline.py`
- Canonical model registry: `flux-server/app/model_manager.py`
- Canonical video defaults: `flux-server/app/video_defaults.py`

Treat `flux-server/` as production code. Root-level legacy app assets are not the active surface.

---

## CURRENT BRANCH STATE

Latest implementation branch:
- `codex/hyperforge-runtime-hardening-impl`

Latest pushed commit from this session:
- `f3ba63c feat: add deploy-ready WAN 14B xDiT quality backend`

Note:
- Root `AGENTS.md` is currently untracked locally. Do not assume it is part of the repo state unless explicitly added.

---

## WAN 14B QUALITY PATH

### Preferred backend

WAN T2V 14B now prefers:
- BF16 xDiT on 4 visible GPUs with enough VRAM
- launched through `torchrun`
- invoked from the API runtime as a subprocess

Fallback:
- existing NF4 diffusers path

This is intentional. The server does not try to become a distributed xDiT rank process itself.

### Why this design

- keeps `/api/video/generate` unchanged
- keeps single-GPU environments working
- makes the branch deploy-ready without requiring the entire FastAPI stack to run under `torchrun`
- prioritizes BF16 quality for WAN 14B where the host supports it

---

## CURRENT VIDEO DEFAULTS

These defaults are now centralized in `flux-server/app/video_defaults.py` and surfaced through `/models`.

### WAN T2V 1.3B
- resolution: `480p`
- frames: `33`
- fps: `16`
- steps: `30`
- guidance: `5.0`
- chunking: `49` size / `8` overlap

### WAN T2V 14B
- resolution: `720p`
- frames: `49`
- fps: `16`
- steps: `32`
- guidance: `6.0`
- chunking: `49` size / `12` overlap
- preferred backend: `xdit-bf16`

### WAN I2V 14B
- resolution: `720p`
- frames: `49`
- fps: `16`
- steps: `32`
- guidance: `5.5`
- chunking: `49` size / `12` overlap

### HunyuanVideo
- resolution: `720p`
- frames: `129`
- fps: `24`
- steps: `50`
- guidance: `6.0`

---

## XDIT IMPLEMENTATION DETAILS

### Files added

- `flux-server/app/video_defaults.py`
- `flux-server/tools/wan14b_xdit_infer.py`

### Files changed for xDiT / quality alignment

- `flux-server/app/main.py`
- `flux-server/app/model_manager.py`
- `flux-server/app/pipelines/video_pipeline.py`
- `flux-server/app/schemas.py`
- `flux-server/app/static/app.js`
- `flux-server/gunicorn.conf.py`
- `flux-server/docker-compose.yml`
- `flux-server/deploy/aws/docker-compose.aws.yml`
- `flux-server/requirements.txt`
- `implementation_plan.md`
- `flux-server/AGENT.md`

### Dependency rule

Install `xfuser` from GitHub source, not PyPI `xfuser==0.4.5`.

Reason:
- the PyPI wheel inspected during this session did not contain the needed WAN pipeline support
- upstream source does contain WAN 2.2 support

Current requirement:
- `git+https://github.com/xdit-project/xDiT.git`

---

## GUNICORN / GPU GROUPING

`flux-server/gunicorn.conf.py` now groups GPUs by job instead of assigning one worker per single GPU.

Important env/config:
- `GPUS_PER_JOB=4`
- `VIDEO_PARALLEL_BACKEND=auto`

Behavior:
- on 8 GPUs: 2 workers
- worker 0 gets GPUs `0,1,2,3`
- worker 1 gets GPUs `4,5,6,7`

The worker exports `CUDA_VISIBLE_DEVICES` for its slice, and WAN 14B xDiT runs inside that slice via subprocess.

---

## DEPLOY-READY CONFIG

Updated but not deployed:
- `flux-server/docker-compose.yml`
- `flux-server/deploy/aws/docker-compose.aws.yml`

Important changes:
- `count: all`
- `shm_size: 16g`
- `GPUS_PER_JOB=4`
- `VIDEO_PARALLEL_BACKEND=auto`
- removed single-GPU pinning assumptions

No live deployment was performed in this session.

---

## VERIFICATION ALREADY DONE

Completed locally:
- Python compile checks for changed app files
- import checks for `app.main`
- schema default resolution checks for WAN/Hunyuan video requests
- gunicorn GPU grouping math checks for 1 GPU and 8 GPU scenarios

Observed local warning:
- diffusers falls back to PIL image processors because `torchvision` is not installed
- this did not block imports during verification

---

## VULTR DEPLOYMENT STATUS (2026-05-09)

Vultr bare metal A100 plan (`vbm-112c-2048gb-8-a100-gpu`) requires account-level
approval before the API accepts provisioning requests. Tested in both ewr and atl
regions — both return 403 "Please open a support request for access to this product."

**Status: Vultr support ticket submitted. Waiting for A100 bare metal access approval.**

SSH key already uploaded: `hyperforge` (UUID: `50134fc3-969b-43a1-8d47-fb7b4b5c3eae`)

Evaluated and rejected fallback:
- A16 Cloud GPU plans (available now, no approval needed) — rejected because A16 has
  16 GB GDDR6 per chip (~10x less memory bandwidth than A100 SXM). WAN 14B at 720p
  would take 25-40 min/video and may not fit without CPU offloading. Not worth it.

When approval arrives, run `launch.sh` with:
  VULTR_API_KEY, SSH_KEY_ID=50134fc3-969b-43a1-8d47-fb7b4b5c3eae,
  HF_TOKEN (from flux-server/.env), REGION=ewr

---

## REMAINING REAL-WORLD VALIDATION

Still not done:
1. Build the Docker image with the new `xfuser` source dependency.
2. Run standalone WAN 14B xDiT smoke:
   `flux-server/tools/wan14b_xdit_infer.py`
3. Verify 4-GPU utilization on the target host during that run.
4. Run one WAN 14B API job and confirm backend selection logs show `xdit-bf16`.
5. Run two concurrent WAN 14B API jobs and confirm workers split across GPU groups.

---

## IMPORTANT RULES

### 1. Keep quality defaults centralized

Do not reintroduce separate WAN defaults in:
- `main.py`
- `schemas.py`
- frontend state

Use `app/video_defaults.py` as the source of truth.

### 2. Do not silently collapse xDiT back into generic diffusers defaults

If xDiT is unavailable, fallback should be explicit in logs.

### 3. Do not break the API surface

`/api/video/generate` should stay stable while backend selection remains internal.

### 4. Do not assume PyPI xfuser is enough

WAN support was validated against upstream source, not the older wheel.

---

## FILES TO READ FIRST NEXT SESSION

1. `implementation_plan.md`
2. `flux-server/AGENT.md`
3. `flux-server/app/video_defaults.py`
4. `flux-server/app/pipelines/video_pipeline.py`
5. `flux-server/tools/wan14b_xdit_infer.py`
6. `flux-server/gunicorn.conf.py`

---

## SESSION START CHECKLIST

Before continuing this work:
1. Confirm branch is `codex/hyperforge-runtime-hardening-impl`
2. Check `git status` for unexpected drift
3. Read `implementation_plan.md` and `flux-server/AGENT.md`
4. If testing xDiT, verify target host has 4+ visible 80GB GPUs per job group
5. Build and smoke-test before any live deployment
