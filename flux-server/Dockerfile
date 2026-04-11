# ============================================================
# Stage 1: Builder — install Python dependencies
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git \
    build-essential cmake pkg-config \
    ffmpeg libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libavfilter-dev libswscale-dev libswresample-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN python3.11 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Runtime — slim production image
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsndfile1 \
    libsm6 \
    libxext6 && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -s /bin/bash appuser

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/mnt/hf-cache" \
    TRANSFORMERS_CACHE="/mnt/hf-cache"

WORKDIR /app
COPY app/ ./app/
COPY gunicorn.conf.py .

# Create directories with correct permissions
RUN mkdir -p loras video_loras /mnt/outputs/video \
        /app/model_cache && \
    chown -R appuser:appuser loras video_loras /mnt/outputs /app/model_cache

# Run as non-root for security
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3.11 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]
