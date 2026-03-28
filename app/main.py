import logging
from contextlib import asynccontextmanager
import gradio as gr
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse
from app.security import verify_api_key, check_rate_limit
from app.pipeline import flux_pipeline
from app.ui import build_ui

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, cleanup on shutdown."""
    logger.info("Starting FLUX.1-dev inference server...")
    flux_pipeline.load()
    logger.info("Server ready to accept requests!")
    yield
    # Cleanup on shutdown
    logger.info("Shutting down, releasing GPU memory...")
    if flux_pipeline.pipe:
        del flux_pipeline.pipe


app = FastAPI(
    title="FLUX.1-dev Inference API",
    description="Generate images from text prompts using FLUX.1-dev on NVIDIA L4 GPU",
    version="1.0.0",
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
    )


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
        width=req.width,
        height=req.height,
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
        lora_name=req.lora_name,
        lora_scale=req.lora_scale,
    )

    return GenerateResponse(
        status="completed",
        image_base64=img_b64,
        seed_used=seed_used,
        inference_time_ms=elapsed_ms,
    )


# Mount Gradio UI at root — API endpoints remain at /health, /generate, /docs
gradio_app = build_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/")
