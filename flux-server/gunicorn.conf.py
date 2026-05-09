import os
import subprocess

bind = "0.0.0.0:8080"
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 3600        # 60 min: covers first-time model downloads (WAN 14B ~118GB)
keepalive = 5
accesslog = "-"       # Log to stdout (captured by Docker / Cloud Logging)
errorlog = "-"
loglevel = "info"

# preload_app MUST stay disabled: CUDA cannot be re-initialized after fork.
# Each worker initialises its own CUDA context after receiving a GPU assignment
# in post_fork. Enabling preload_app would share a single CUDA context across all
# workers and crash on the second fork.

# max_requests / max_requests_jitter are disabled: recycling a worker mid-job
# would terminate long-running video generation (30+ min). Memory fragmentation
# is acceptable given that each worker holds one model in VRAM indefinitely.


def _detect_gpu_count() -> int:
    """Return the number of CUDA GPUs visible to the container.

    Priority: GPU_COUNT env var → nvidia-smi → 1 (safe fallback).
    """
    env_override = os.environ.get("GPU_COUNT", "").strip()
    if env_override.isdigit() and int(env_override) > 0:
        return int(env_override)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL, timeout=5
        )
        count = sum(1 for line in out.strip().splitlines() if line.startswith("GPU "))
        return max(1, count)
    except Exception:
        return 1


_num_gpus = _detect_gpu_count()

# One worker per GPU for maximum parallel job throughput.
# Override via NUM_WORKERS env var if you want fewer (e.g. 4 for BF16 dual-GPU jobs).
workers = int(os.environ.get("NUM_WORKERS", str(_num_gpus)))


def post_fork(server, worker):
    """Assign each gunicorn worker an exclusive GPU.

    worker.age is an incrementing counter (0, 1, 2, ...) assigned by the Arbiter
    at fork time. Modulo _num_gpus gives a stable round-robin GPU assignment.
    If a worker is restarted, it may temporarily share a GPU — this is acceptable
    since VRAM is large enough for two NF4 WAN 14B instances on 80 GB.
    """
    gpu_idx = worker.age % _num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    # Force a clean CUDA context in this worker (no inherited state from master)
    try:
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(
        f"[gunicorn] worker {worker.age} (pid={os.getpid()}) → GPU {gpu_idx}",
        flush=True,
    )
