# 🤖 Neural Creation Studio | Intelligent AI Agent

Welcome to the **Neural Creation Studio**, a high-performance, multi-modal AI generation platform designed for a single A100 GPU environment. This repository is managed by an intelligent AI assistant (Agent) to ensure seamless deployment, testing, and optimization of state-of-the-art generative models.

## 🏗️ System Architecture

The project is built with a modular architecture that enables multiple heavy AI models (Image, Video, Music, Animation) to share a single GPU efficiently.

### 🧠 MultiModelManager
The heart of the system is the [MultiModelManager](file:///d:/Flux_Lora/flux-server/app/model_manager.py). It manages the "Turn-Based GPU Access":
- **Lazy Loading**: Models are only loaded into VRAM when a request arrives.
- **VRAM Swapping**: When switching between modalities (e.g., Image to Video), the manager automatically unloads the previous model to free up memory.
- **Quantization**: Support for NF4 and FP8 quantization allow large models (like FLUX and Wan 2.2) to run smoothly.

### 📥 Asynchronous Job Queue
For long-running tasks (Video, Music, Animation), the server uses an [internal job queue](file:///d:/Flux_Lora/flux-server/app/job_queue.py):
- **Status API**: Clients poll for job progress and results.
- **Memory Safety**: The queue ensures only one generation task runs on the GPU at a time to prevent OOM (Out of Memory) errors.

---

## 🎨 Capabilities & Model Registry

| Category | Model Name | Description | VRAM Req |
| :--- | :--- | :--- | :--- |
| **Image** | `flux-1-dev` | High quality, 28-step professional generation | 20GB |
| **Image** | `sd3.5-large` | Flexible, multi-modal prompt adhesion | 18GB |
| **Image** | `realvisxl-v5` | Photorealistic portrait & product photography | 16GB |
| **Video** | `wan-t2v-1.3b` | Fast, lightweight text-to-video | 10GB |
| **Video** | `wan-t2v-14b` | SOTA cinematic video generation (480p) | 35GB |
| **Music** | `audioldm2` | High-fidelity sound effects and background music | 5GB |
| **Music** | `ace-step` | Full songs with lyrics and vocals | 4GB |
| **Animation** | `liveportrait` | Fast talking-head animation from image + audio | 6GB |
| **Animation** | `echomimic` | SOTA audio-driven facial performance | 12GB |

---

## 🛠️ Developer & Testing Guide

To ensure high reliability, the system includes a suite of remote smoke testing tools.

### Running a Complete Smoke Test
Tests all image, video, and audio models sequentially.
```powershell
python tools/remote_smoke_test.py --server http://<ip-address>:8080
```

### Targeted Testing
If you want to skip high-VRAM video models and focus on lightweight modalities:
```powershell
python tools/test_audio_anim.py --server http://<ip-address>:8080
```

---

## 📊 Current Status (April 2026)

| Modality | Verified | Issues |
| :--- | :--- | :--- |
| **Image** | ✅ Fully Operational | None |
| **Music** | 🏗️ Partial | `audioldm2` works; `ace-step` needs path fixing. |
| **Video** | ⚠️ VRAM Sensitive | Video loading requires ~10min on first run. |
| **Animation**| 🏗️ Pending | Installation of `EchoMimic` dependencies verified. |

---

## 🚀 Pro-Tips for the Agent
- **VRAM Management**: If the server becomes unresponsive, check `/health`. If VRAM is at 100%, trigger an `unload_all()` by restarting the container.
- **Dataset Planning**: Use [dataset_plan.py](file:///d:/Flux_Lora/flux-server/app/dataset_plan.py) to generate optimal composition ratios for LoRA training.
