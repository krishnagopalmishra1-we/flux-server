bind = "0.0.0.0:8080"
workers = 1           # 1 worker per GPU (FLUX uses all VRAM)
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 1800        # 30 minutes — allows longest video generation runs (14B models)
keepalive = 5
accesslog = "-"       # Log to stdout (captured by Docker/Cloud Logging)
errorlog = "-"
loglevel = "info"
