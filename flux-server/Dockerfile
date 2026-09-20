FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    build-essential cmake pkg-config \
    libgl1 libglib2.0-0 libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -s /bin/bash appuser

WORKDIR /app

RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/mnt/hf-cache" \
    TRANSFORMERS_CACHE="/mnt/hf-cache"

COPY app/ ./app/
COPY tools/ ./tools/
COPY gunicorn.conf.py .

RUN mkdir -p loras /mnt/outputs/image /app/model_cache && \
    chown -R appuser:appuser loras /mnt/outputs /app/model_cache

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]
