bind = "0.0.0.0:8080"
workers = 1           # 1 worker per GPU (all models share VRAM)
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 3600        # 60min: first-time model downloads (Wan 14B ~28GB) can be slow
keepalive = 5
accesslog = "-"       # Log to stdout (captured by Docker/Cloud Logging)
errorlog = "-"
loglevel = "info"
# preload_app = True  # DISABLED: CUDA cannot be re-initialized after fork (workers=1, no benefit anyway)
max_requests = 100    # Recycle worker after N requests to prevent memory fragmentation
max_requests_jitter = 20  # ±20 jitter so all workers don't recycle simultaneously
