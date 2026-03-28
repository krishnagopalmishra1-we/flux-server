bind = "0.0.0.0:8080"
workers = 1           # 1 worker per GPU (FLUX uses all VRAM)
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 1200        # First startup downloads ~24GB model; after that images take 30s+
keepalive = 5
accesslog = "-"       # Log to stdout (captured by Docker/Cloud Logging)
errorlog = "-"
loglevel = "info"
