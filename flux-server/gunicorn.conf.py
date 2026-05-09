import os
import subprocess

bind = "0.0.0.0:8080"
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 3600
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"


def _detect_gpu_count() -> int:
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
_gpus_per_job = max(1, int(os.environ.get("GPUS_PER_JOB", "4")))
_group_count = max(1, _num_gpus // _gpus_per_job)

# Allow an explicit override, otherwise default to one worker per GPU group.
workers = int(os.environ.get("NUM_WORKERS", str(_group_count)))


def _gpu_slice_for_worker(worker_age: int) -> list[int]:
    if _num_gpus <= _gpus_per_job:
        return list(range(_num_gpus))
    group_index = worker_age % _group_count
    start = group_index * _gpus_per_job
    return list(range(start, min(start + _gpus_per_job, _num_gpus)))


def post_fork(server, worker):
    gpu_slice = _gpu_slice_for_worker(worker.age)
    visible = ",".join(str(idx) for idx in gpu_slice) if gpu_slice else ""
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["XDIT_GPU_GROUP_SIZE"] = str(len(gpu_slice) or 1)
    os.environ["VIDEO_PARALLEL_BACKEND"] = os.environ.get("VIDEO_PARALLEL_BACKEND", "auto")
    try:
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(
        f"[gunicorn] worker {worker.age} (pid={os.getpid()}) -> GPUs [{visible or 'cpu'}]",
        flush=True,
    )
