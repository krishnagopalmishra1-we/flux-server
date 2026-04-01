bind = "0.0.0.0:8080"
workers = 1           # 1 worker per GPU (all models share VRAM)
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 3600        # 60min: first-time model downloads (Wan 14B ~28GB) can be slow
keepalive = 5
accesslog = "-"       # Log to stdout (captured by Docker/Cloud Logging)
errorlog = "-"
loglevel = "info"
