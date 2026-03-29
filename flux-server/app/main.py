import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.security import verify_api_key, check_rate_limit
from app.pipeline_new import inference_pipeline as flux_pipeline
from app.dataset_plan import get_dataset_plan, available_domains
from app.training_presets import get_lora_preset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, cleanup on shutdown."""
    logger.info("Starting multi-model inference server...")
    flux_pipeline.load("flux-1-dev")
    logger.info("Server ready to accept requests!")
    yield
    # Cleanup on shutdown
    logger.info("Shutting down, releasing GPU memory...")
    flux_pipeline.model_manager.unload_all()


app = FastAPI(
    title="Multi-Model Image Generation API",
    description="Generate images using FLUX.1-dev, FLUX.1-schnell, SD3-Medium, and SDXL on NVIDIA A100",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — restrict allow_origins to your frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check server health, GPU status, and model loading state."""
    gpu = flux_pipeline.gpu_info()
    return HealthResponse(
        status="healthy" if flux_pipeline.is_loaded else "loading",
        gpu_name=gpu["name"],
        vram_total_gb=gpu["total_gb"],
        vram_used_gb=gpu["used_gb"],
        model_loaded=flux_pipeline.is_loaded,
        current_model=flux_pipeline.model_manager.current_model,
    )


@app.get("/models")
async def list_models():
    """List available generation models and metadata."""
    models = flux_pipeline.list_available_models()
    return {
        "current_model": flux_pipeline.model_manager.current_model,
        "models": [
            {
                "name": name,
                **flux_pipeline.get_model_info(name),
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


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Generate an image from a text prompt."""
    check_rate_limit(request, api_key)

    img_b64, seed_used, elapsed_ms = flux_pipeline.generate(
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

    img_b64, seed_used, elapsed_ms = flux_pipeline.generate(
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

    return GenerateResponse(
        status="completed",
        image_base64=img_b64,
        seed_used=seed_used,
        inference_time_ms=elapsed_ms,
    )


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def root_ui():
    return FileResponse(STATIC_DIR / "index.html")
