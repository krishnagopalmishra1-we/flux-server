import json
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import hashlib
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.schemas_v2 import (
    VideoGenerateRequest, VideoGenerateResponse,
    MusicGenerateRequest, MusicGenerateResponse,
    AnimationGenerateRequest, AnimationGenerateResponse,
    JobStatusResponse, QueueStatusResponse,
)
from app.security import verify_api_key, check_rate_limit
from app.pipeline_new import inference_pipeline as flux_pipeline
from app.model_manager import ModelCategory
from app.job_queue import job_queue, JobPriority, JobStatus
from app.output_store import output_store
from app.pipelines.video_pipeline import video_pipeline
from app.pipelines.music_pipeline import music_pipeline
from app.pipelines.animation_pipeline import animation_pipeline
from app.dataset_plan import get_dataset_plan, available_domains
from app.training_presets import get_lora_preset
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
#  JOB HANDLERS — Connect job queue to pipelines
# ═══════════════════════════════════════════════════

def get_user_id(request: Request) -> str:
    """Hash the client IP to a stable, non-PII identifier."""
    host = request.client.host if request.client else "unknown"
    return hashlib.sha256(f"app_salt_{host}".encode()).hexdigest()[:16]

async def _handle_video_job(job) -> dict:
    """Process a video generation job with real-time progress reporting."""
    payload = job.payload
    flux_pipeline.model_manager.unload_all()

    def _progress(pct: float):
        job_queue.set_progress(job.id, pct)

    if payload.get("source_image_b64"):
        return await video_pipeline.generate_image_to_video(
            source_image_b64=payload["source_image_b64"],
            prompt=payload.get("prompt", ""),
            model_name=job.model_name,
            num_frames=payload.get("num_frames", 33),
            fps=payload.get("fps", 16),
            guidance_scale=payload.get("guidance_scale", 5.0),
            num_inference_steps=payload.get("num_inference_steps", 30),
            seed=payload.get("seed"),
            lora_name=payload.get("lora_name"),
            lora_scale=payload.get("lora_scale", 1.0),
            job_id=job.id,
            progress_callback=_progress,
        )
    else:
        return await video_pipeline.generate_text_to_video(
            prompt=payload["prompt"],
            model_name=job.model_name,
            negative_prompt=payload.get("negative_prompt", ""),
            resolution=payload.get("resolution", "480p"),
            num_frames=payload.get("num_frames", 33),
            fps=payload.get("fps", 16),
            guidance_scale=payload.get("guidance_scale", 5.0),
            num_inference_steps=payload.get("num_inference_steps", 30),
            seed=payload.get("seed"),
            lora_name=payload.get("lora_name"),
            lora_scale=payload.get("lora_scale", 1.0),
            job_id=job.id,
            progress_callback=_progress,
        )


async def _handle_music_job(job) -> dict:
    """Process a music generation job."""
    flux_pipeline.model_manager.unload_all()
    payload = job.payload

    return await music_pipeline.generate_song(
        prompt=payload["prompt"],
        model_name=job.model_name,
        duration_seconds=payload.get("duration_seconds", 30),
        lyrics=payload.get("lyrics"),
        genre=payload.get("genre"),
        bpm=payload.get("bpm"),
        seed=payload.get("seed"),
        job_id=job.id,
    )


async def _handle_animation_job(job) -> dict:
    """Process an animation generation job."""
    flux_pipeline.model_manager.unload_all()
    payload = job.payload

    return await animation_pipeline.generate_talking_head(
        source_image_b64=payload["source_image_b64"],
        audio_b64=payload["audio_b64"],
        model_name=job.model_name,
        expression_scale=payload.get("expression_scale", 1.0),
        pose_style=payload.get("pose_style", 0),
        use_enhancer=payload.get("use_enhancer", False),
        job_id=job.id,
    )


# Lock that serialises GPU-heavy operations so image generation,
# video pre-loading, and queued jobs never fight for VRAM at the same time.
_gpu_lock = asyncio.Lock()


async def _preload_video_model_background():
    """Background task to pre-load Wan 2.2 model while server handles requests."""
    # TEMPORARILY DISABLED to debug server responsiveness issues
    # Re-enable once core models are working
    logger.info("Video pre-load: disabled (debug mode)")
    return


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, register job handlers, cleanup on shutdown."""
    logger.info("Starting Neural Creation Studio...")

    # Register job handlers for async generation
    job_queue.register_handler("video", _handle_video_job)
    job_queue.register_handler("music", _handle_music_job)
    job_queue.register_handler("animation", _handle_animation_job)

    # Share the GPU lock with the job queue so video/music/animation jobs
    # cannot run concurrently with image generation endpoints.
    job_queue.gpu_lock = _gpu_lock

    # Create output directories
    settings = get_settings()
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    for subdir in ["video", "audio", "animation"]:
        (Path(settings.output_dir) / subdir).mkdir(exist_ok=True)

    # Keep startup fast and responsive; launch background model pre-loading
    # while server immediately becomes ready to accept requests.
    logger.info("Server ready to accept requests (Wan 2.2 pre-loading in background)")
    preload_task = asyncio.create_task(_preload_video_model_background())
    yield
    # Cleanup on shutdown
    logger.info("Shutting down, releasing GPU memory...")
    preload_task.cancel()
    try:
        await preload_task
    except asyncio.CancelledError:
        pass
    flux_pipeline.model_manager.unload_all()
    video_pipeline.unload()
    music_pipeline.unload()
    animation_pipeline.unload()


app = FastAPI(
    title="Neural Creation Studio API",
    description=(
        "Multi-modal AI generation platform: Image, Video, Music, and Animation. "
        "Powered by FLUX, Wan 2.2, ACE-Step, AudioLDM2, EchoMimic, and more."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# CORS — restrict allow_origins to your frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check server health, GPU status, and model loading state."""
    gpu = flux_pipeline.gpu_info()
    return HealthResponse(
        status="healthy",
        gpu_name=gpu["name"],
        vram_total_gb=gpu["total_gb"],
        vram_used_gb=gpu["used_gb"],
        model_loaded=flux_pipeline.is_loaded,
        current_model=flux_pipeline.model_manager.current_model,
    )


@app.get("/models")
async def list_models(category: str = None):
    """List available generation models and metadata, optionally filtered by category."""
    cat_filter = ModelCategory(category) if category else None
    models = flux_pipeline.model_manager.list_models(cat_filter)
    return {
        "current_model": flux_pipeline.model_manager.current_model,
        "categories": flux_pipeline.model_manager.get_categories(),
        "models": [
            {
                "name": name,
                **flux_pipeline.model_manager.get_model_info(name),
                "summary": summary,
            }
            for name, summary in models.items()
        ],
    }


@app.get("/loras")
async def list_loras(model_name: str = "flux-1-dev"):
    """List LoRA files compatible with the selected model."""
    return {
        "model_name": model_name,
        "recommended_scale": flux_pipeline.get_recommended_lora_scale(model_name),
        "loras": flux_pipeline.get_compatible_loras(model_name),
    }


@app.post("/loras/upload")
async def upload_lora(file: UploadFile = File(...)):
    """Upload a .safetensors LoRA file to the loras/ directory."""
    if not file.filename.endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="Only .safetensors files are allowed.")

    # Sanitize filename — strip any path components
    safe_name = Path(file.filename).name
    if not safe_name or safe_name != file.filename.replace("\\", "/").split("/")[-1]:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    lora_dir = Path("loras")
    lora_dir.mkdir(exist_ok=True)
    dest = lora_dir / safe_name

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    dest.write_bytes(contents)
    logger.info("LoRA uploaded: %s (%d bytes)", safe_name, len(contents))
    return {"status": "uploaded", "filename": safe_name, "size_bytes": len(contents)}


@app.get("/datasets/domains")
async def dataset_domains():
    """List available dataset planning domains."""
    return {"domains": available_domains()}


@app.get("/datasets/plan/{domain}")
async def dataset_plan(domain: str):
    """Return recommended dataset composition for a domain."""
    return get_dataset_plan(domain)


@app.get("/training/preset/{style}")
async def training_preset(style: str):
    """Return LoRA training hyperparameter preset."""
    return get_lora_preset(style)


def _run_image_generation(req: GenerateRequest) -> tuple[str, int, float]:
    """Run image inference in a worker thread (CPU-bound / GPU-bound)."""
    # Unload any video/music/animation model to free VRAM for image model.
    video_pipeline.unload()
    music_pipeline.unload()
    animation_pipeline.unload()

    return flux_pipeline.generate(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        model_name=req.model_name,
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        lora_name=req.lora_name,
        lora_scale=req.lora_scale,
        use_refiner=req.use_refiner,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Generate an image from a text prompt."""
    check_rate_limit(request, api_key)

    try:
        async with _gpu_lock:
            img_b64, seed_used, elapsed_ms = await asyncio.to_thread(
                _run_image_generation, req
            )
    except RuntimeError as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during image generation")
        raise HTTPException(status_code=500, detail="Image generation failed. Check server logs.")

    return GenerateResponse(
        status="completed",
        image_base64=img_b64,
        seed_used=seed_used,
        inference_time_ms=elapsed_ms,
    )


@app.post("/generate-ui", response_model=GenerateResponse)
async def generate_ui(req: GenerateRequest, request: Request):
    """Browser-facing generation endpoint for the built-in UI."""
    check_rate_limit(request, "")

    try:
        async with _gpu_lock:
            img_b64, seed_used, elapsed_ms = await asyncio.to_thread(
                _run_image_generation, req
            )
    except RuntimeError as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during image generation")
        raise HTTPException(status_code=500, detail="Image generation failed. Check server logs.")

    return GenerateResponse(
        status="completed",
        image_base64=img_b64,
        seed_used=seed_used,
        inference_time_ms=elapsed_ms,
    )


# ═══════════════════════════════════════════════════
#  VIDEO GENERATION ENDPOINTS
# ═══════════════════════════════════════════════════


@app.post("/api/video/generate", response_model=VideoGenerateResponse)
async def generate_video(req: VideoGenerateRequest, request: Request):
    """Submit a video generation job (text-to-video or image-to-video)."""
    settings = get_settings()
    if not settings.enable_video:
        raise HTTPException(status_code=503, detail="Video generation is disabled.")

    check_rate_limit(request, "")

    try:
        job = await job_queue.submit(
            job_type="video",
            model_name=req.model_name,
            payload=req.model_dump(),
            priority=JobPriority.SLOW,
            user_id=get_user_id(request),
        )
    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))

    position = job_queue.get_queue_position(job.id)
    return VideoGenerateResponse(
        job_id=job.id,
        status=job.status.value,
        queue_position=position,
    )


# ═══════════════════════════════════════════════════
#  MUSIC GENERATION ENDPOINTS
# ═══════════════════════════════════════════════════


@app.post("/api/music/generate", response_model=MusicGenerateResponse)
async def generate_music(req: MusicGenerateRequest, request: Request):
    """Submit a music/song generation job."""
    settings = get_settings()
    if not settings.enable_music:
        raise HTTPException(status_code=503, detail="Music generation is disabled.")

    check_rate_limit(request, "")

    try:
        job = await job_queue.submit(
            job_type="music",
            model_name=req.model_name,
            payload=req.model_dump(),
            priority=JobPriority.FAST,
            user_id=get_user_id(request),
        )
    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))

    position = job_queue.get_queue_position(job.id)
    return MusicGenerateResponse(
        job_id=job.id,
        status=job.status.value,
        queue_position=position,
    )


# ═══════════════════════════════════════════════════
#  ANIMATION GENERATION ENDPOINTS
# ═══════════════════════════════════════════════════


@app.post("/api/animation/generate", response_model=AnimationGenerateResponse)
async def generate_animation(req: AnimationGenerateRequest, request: Request):
    """Submit an animation generation job (audio-driven talking head)."""
    settings = get_settings()
    if not settings.enable_animation:
        raise HTTPException(status_code=503, detail="Animation generation is disabled.")

    check_rate_limit(request, "")

    try:
        job = await job_queue.submit(
            job_type="animation",
            model_name=req.model_name,
            payload=req.model_dump(),
            priority=JobPriority.NORMAL,
            user_id=get_user_id(request),
        )
    except ValueError as e:
        raise HTTPException(status_code=429, detail=str(e))

    position = job_queue.get_queue_position(job.id)
    return AnimationGenerateResponse(
        job_id=job.id,
        status=job.status.value,
        queue_position=position,
    )


# ═══════════════════════════════════════════════════
#  JOB STATUS ENDPOINTS
# ═══════════════════════════════════════════════════


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the status of a generation job."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    position = job_queue.get_queue_position(job_id)
    return JobStatusResponse(
        job_id=job.id,
        job_type=job.job_type,
        model_name=job.model_name,
        status=job.status.value,
        progress=job.progress,
        result=job.result,
        error_message=job.error_message,
        queue_position=position,
        queue_time_ms=job.queue_time_ms,
        processing_time_ms=job.processing_time_ms,
    )


@app.get("/api/jobs")
async def list_jobs(request: Request, status: str = None, limit: int = 20):
    """List recent jobs for the current user."""
    user_id = get_user_id(request)
    status_filter = JobStatus(status) if status else None
    jobs = job_queue.list_jobs(user_id=user_id, status=status_filter, limit=limit)
    return {"jobs": jobs, "total": len(jobs)}


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a queued job."""
    if job_queue.cancel_job(job_id):
        return {"status": "cancelled", "job_id": job_id}
    raise HTTPException(
        status_code=400,
        detail="Job cannot be cancelled (not found or already processing)."
    )


@app.get("/api/queue/status", response_model=QueueStatusResponse)
async def queue_status():
    """Get current queue statistics."""
    stats = job_queue.queue_stats()
    return QueueStatusResponse(**stats)


# ═══════════════════════════════════════════════════
#  SSE — Real-time job progress streaming
# ═══════════════════════════════════════════════════

@app.get("/api/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str, request: Request):
    """
    Server-Sent Events endpoint for real-time job progress.

    Client connects once per job; receives JSON events:
      data: {"progress": 42.5, "status": "processing"}
      data: {"progress": 100.0, "status": "completed", "result": {...}}
    Connection closes automatically on job completion/failure.
    """
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    async def event_generator():
        q = job_queue.subscribe_progress(job_id)
        try:
            # Immediately send current state
            current = job_queue.get_job(job_id)
            if current:
                payload = {"progress": current.progress, "status": current.status.value}
                if current.status.value in ("completed", "failed", "cancelled"):
                    payload["result"] = current.result
                    payload["error"] = current.error_message
                yield f"data: {json.dumps(payload)}\n\n"

            # Stream updates until terminal state or client disconnect
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=1.0)
                    # Re-fetch job for result payload on completion
                    updated = job_queue.get_job(job_id)
                    if updated and updated.status.value in ("completed", "failed", "cancelled"):
                        event["result"] = updated.result
                        event["error"] = updated.error_message
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in ("completed", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    # Heartbeat — keeps connection alive through proxies
                    j = job_queue.get_job(job_id)
                    if j:
                        hb = {"progress": j.progress, "status": j.status.value}
                        yield f"data: {json.dumps(hb)}\n\n"
                        if j.status.value in ("completed", "failed", "cancelled"):
                            break
        finally:
            job_queue.unsubscribe_progress(job_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════
#  VIDEO LORA ENDPOINTS
# ═══════════════════════════════════════════════════

@app.get("/api/video/loras")
async def list_video_loras():
    """List LoRA files available for video models."""
    loras = video_pipeline.get_available_loras()
    return {"loras": loras, "count": len(loras)}


@app.post("/api/video/loras/upload")
async def upload_video_lora(file: UploadFile = File(...)):
    """Upload a .safetensors LoRA adapter for video models."""
    if not file.filename.endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="Only .safetensors files are allowed.")
    safe_name = Path(file.filename).name
    if (
        not safe_name
        or ".." in safe_name
        or safe_name != file.filename.replace("\\", "/").split("/")[-1]
    ):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    from app.pipelines.video_pipeline import VIDEO_LORA_DIR
    VIDEO_LORA_DIR.mkdir(exist_ok=True)
    dest = VIDEO_LORA_DIR / safe_name
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    dest.write_bytes(contents)
    logger.info("Video LoRA uploaded: %s (%d bytes)", safe_name, len(contents))
    return {"status": "uploaded", "filename": safe_name, "size_bytes": len(contents)}


# ═══════════════════════════════════════════════════
#  STATIC FILES & OUTPUT SERVING
# ═══════════════════════════════════════════════════


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount output directory for serving generated files
try:
    _settings = get_settings()
    _output_path = Path(_settings.output_dir)
    _output_path.mkdir(parents=True, exist_ok=True)
    for _sub in ["video", "audio", "animation"]:
        (_output_path / _sub).mkdir(exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(_output_path)), name="outputs")
except Exception as _e:
    logger.warning(f"Could not mount output directory: {_e}. Will be created at startup.")


@app.get("/", include_in_schema=False)
async def root_ui():
    return FileResponse(STATIC_DIR / "index.html")
