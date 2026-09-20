import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pathlib import Path

from app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.security import verify_api_key, check_rate_limit
from app.pipeline import inference_pipeline, get_lora_dir
from app.output_store import output_store
from app.runtime import gpu_runtime
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class ContentLengthLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()
        max_bytes = int(settings.max_request_body_mb * 1024 * 1024) if hasattr(settings, 'max_request_body_mb') else 25 * 1024 * 1024
        content_length = request.headers.get("content-length")
        try:
            request_bytes = int(content_length) if content_length else 0
        except ValueError:
            request_bytes = 0
        if request_bytes > max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request body too large."},
            )
        return await call_next(request)


def _validate_lora_upload(safe_name: str, contents: bytes) -> None:
    settings = get_settings()
    max_bytes = settings.max_lora_upload_mb * 1024 * 1024
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"LoRA upload is too large ({len(contents) / (1024 * 1024):.1f}MB). "
                f"Limit is {settings.max_lora_upload_mb}MB. Full checkpoints are not accepted."
            ),
        )
    if not safe_name.endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="Only .safetensors files are allowed.")


def _ensure_write_budget(path: Path, required_bytes: int, reserve_gb: float = 2.0) -> None:
    path.mkdir(parents=True, exist_ok=True)
    import shutil
    usage = shutil.disk_usage(str(path))
    reserve_bytes = int(reserve_gb * 1024 * 1024 * 1024)
    if usage.free - required_bytes < reserve_bytes:
        raise HTTPException(
            status_code=507,
            detail=(
                f"Insufficient disk space in {path}: keep at least {reserve_gb:.1f}GB free "
                "after upload."
            ),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create output directories and manage lifecycle."""
    logger.info("Starting Neural Creation Studio (Image Only)...")

    # Create output directories
    settings = get_settings()
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).joinpath("image").mkdir(exist_ok=True)
    get_lora_dir().mkdir(parents=True, exist_ok=True)

    # Schedule periodic output cleanup (runs every hour, deletes files older than TTL)
    cleanup_task = asyncio.create_task(_periodic_output_cleanup())

    logger.info("Server ready to accept image requests")
    yield
    # Cleanup on shutdown
    logger.info("Shutting down, releasing GPU memory...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    inference_pipeline.model_manager.unload_all()


async def _periodic_output_cleanup():
    """Background task that cleans up expired output files every hour."""
    while True:
        await asyncio.sleep(3600)  # Every hour
        try:
            deleted = output_store.cleanup_expired()
            if deleted:
                logger.info(f"Periodic cleanup: removed {deleted} expired output files")
        except Exception as e:
            logger.warning(f"Periodic output cleanup error: {e}")


app = FastAPI(
    title="Neural Creation Studio API",
    description="AI generation platform: Image generation powered by FLUX.",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS
_cors_settings = get_settings()
_cors_origins = (
    [o.strip() for o in _cors_settings.cors_origins.split(",")]
    if _cors_settings.cors_origins != "*"
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
app.add_middleware(ContentLengthLimitMiddleware)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check server health, GPU status, and model loading state."""
    gpu = inference_pipeline.gpu_info()
    return HealthResponse(
        status="healthy",
        gpu_name=gpu["name"],
        vram_total_gb=gpu["total_gb"],
        vram_used_gb=gpu["used_gb"],
        model_loaded=inference_pipeline.is_loaded,
        current_model=inference_pipeline.model_manager.current_model,
    )


@app.get("/api/auth/status")
async def auth_status():
    """Expose whether browser clients must send X-API-Key."""
    valid_keys = [k.strip() for k in get_settings().api_keys.split(",") if k.strip()]
    return {
        "api_key_required": bool(valid_keys),
        "configured_key_count": len(valid_keys),
        "header": "X-API-Key",
    }


@app.get("/models")
async def list_models():
    """List available generation models and metadata."""
    models = inference_pipeline.model_manager.list_models()
    return {
        "current_model": inference_pipeline.model_manager.current_model,
        "models": [
            {
                "name": name,
                **inference_pipeline.model_manager.get_model_info(name),
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
        "lora_dir": str(get_lora_dir()),
        "recommended_scale": inference_pipeline.get_recommended_lora_scale(model_name),
        "loras": inference_pipeline.get_compatible_loras(model_name),
    }


@app.post("/loras/upload")
async def upload_lora(file: UploadFile = File(...)):
    """Upload a .safetensors LoRA file to the loras/ directory."""
    if not file.filename.endswith(".safetensors"):
        raise HTTPException(status_code=400, detail="Only .safetensors files are allowed.")

    # Sanitize filename
    safe_name = Path(file.filename).name
    if not safe_name or safe_name != file.filename.replace("\\", "/").split("/")[-1]:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    lora_dir = get_lora_dir()
    lora_dir.mkdir(parents=True, exist_ok=True)
    dest = lora_dir / safe_name

    contents = await file.read()
    _validate_lora_upload(safe_name, contents)
    _ensure_write_budget(lora_dir, len(contents))

    dest.write_bytes(contents)
    logger.info("LoRA uploaded: %s (%d bytes)", safe_name, len(contents))
    return {"status": "uploaded", "filename": safe_name, "size_bytes": len(contents), "dir": str(lora_dir)}


def _run_image_generation(req: GenerateRequest) -> tuple[str, int, float]:
    """Run image inference in a worker thread."""
    return inference_pipeline.generate(
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
        async with gpu_runtime.claim():
            img_b64, seed_used, elapsed_ms = await asyncio.to_thread(
                _run_image_generation, req
            )
    except RuntimeError as e:
        logger.error(f"Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during image generation")
        raise HTTPException(status_code=500, detail="Image generation failed. Check server logs.")

    # Persist to disk and record in library (best-effort)
    try:
        import base64
        img_bytes = base64.b64decode(img_b64)
        rel_path = output_store.save_file(img_bytes, "image", ".png")
        output_store.record_entry(rel_path, "image", {
            "prompt": req.prompt,
            "model_name": req.model_name,
            "width": req.width,
            "height": req.height,
            "seed": seed_used,
            "steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
            "lora_name": req.lora_name or None,
            "inference_time_ms": elapsed_ms,
        })
    except Exception as _lib_err:
        logger.warning(f"Library save skipped: {_lib_err}")

    return GenerateResponse(
        status="completed",
        image_base64=img_b64,
        seed_used=seed_used,
        inference_time_ms=elapsed_ms,
    )


@app.get("/api/library")
async def get_library(limit: int = 200, offset: int = 0):
    """Return saved library items, newest first."""
    return output_store.list_library(type_filter="image", limit=limit, offset=offset)


@app.delete("/api/library/{item_id}")
async def delete_library_item(item_id: str):
    """Delete a library item and its file."""
    deleted = output_store.delete_entry(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Library item not found.")
    return {"status": "deleted", "id": item_id}


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount output directory for serving generated files
try:
    _settings = get_settings()
    _output_path = Path(_settings.output_dir)
    _output_path.mkdir(parents=True, exist_ok=True)
    (_output_path / "image").mkdir(exist_ok=True)
    app.mount("/outputs", StaticFiles(directory=str(_output_path)), name="outputs")
except Exception as _e:
    logger.warning(f"Could not mount output directory: {_e}. Will be created at startup.")


@app.get("/", include_in_schema=False)
async def root_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/{page_name:path}", include_in_schema=False)
async def routed_ui(page_name: str):
    """Serve the SPA shell for public UI routes."""
    if page_name in {"image", "library"}:
        return FileResponse(STATIC_DIR / "index.html")
    raise HTTPException(status_code=404, detail="Not found")
