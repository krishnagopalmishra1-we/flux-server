"""
Request/Response schemas for the Neural Creation Studio API.

Covers image generation, video generation, job status, and queue management.
All schemas use Pydantic v2 for validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import uuid


# ═══════════════════════════════════════════════════
#  IMAGE SCHEMAS
# ═══════════════════════════════════════════════════


class GenerateRequest(BaseModel):
    """Request body for the /generate image endpoint."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    model_name: str = Field("flux-1-dev")
    negative_prompt: Optional[str] = Field(None, max_length=1000)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    num_inference_steps: int = Field(28, ge=1, le=50)
    guidance_scale: float = Field(3.5, ge=0.0, le=20.0)
    seed: Optional[int] = None
    lora_name: Optional[str] = None
    lora_scale: float = Field(1.0, ge=0.0, le=2.0)
    use_refiner: bool = False

    @field_validator("width", "height")
    @classmethod
    def must_be_multiple_of_8(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError("Must be a multiple of 8")
        return v

    @field_validator("prompt")
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        """Strip control characters to prevent prompt injection."""
        return "".join(c for c in v if c.isprintable())


class GenerateResponse(BaseModel):
    """Response body from the /generate image endpoint."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "completed"
    image_base64: Optional[str] = None
    seed_used: int = 0
    inference_time_ms: float = 0.0


class HealthResponse(BaseModel):
    """Response body from the /health endpoint."""
    status: str = "healthy"
    gpu_name: str = ""
    vram_total_gb: float = 0.0
    vram_used_gb: float = 0.0
    model_loaded: bool = False
    current_model: str = ""



# ═══════════════════════════════════════════════════
#  VIDEO SCHEMAS
# ═══════════════════════════════════════════════════


class VideoGenerateRequest(BaseModel):
    """Request body for video generation endpoints."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    model_name: str = Field("wan-t2v-1.3b")
    negative_prompt: Optional[str] = Field(None, max_length=1000)
    num_frames: int = Field(33, ge=16, le=1920)
    fps: int = Field(16, ge=8, le=30)
    resolution: str = Field("480p", pattern=r"^(480p|720p)$")
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    num_inference_steps: int = Field(30, ge=10, le=100)
    seed: Optional[int] = None
    # For Image-to-Video: base64-encoded source image
    source_image_b64: Optional[str] = None
    # LoRA support for Wan 2.2 models
    lora_name: Optional[str] = None
    lora_scale: float = Field(1.0, ge=0.0, le=2.0)
    # Chunked generation — for num_frames > chunk_size, generate in chunks
    chunk_size: int = Field(81, ge=16, le=81)
    chunk_overlap: int = Field(20, ge=0, le=30)

    @field_validator("prompt")
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        return "".join(c for c in v if c.isprintable())


class VideoGenerateResponse(BaseModel):
    """Response body for video generation."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "queued"
    video_url: Optional[str] = None
    thumbnail_b64: Optional[str] = None
    duration_seconds: float = 0.0
    inference_time_ms: float = 0.0
    queue_position: int = 0


# ═══════════════════════════════════════════════════
#  JOB STATUS SCHEMAS
# ═══════════════════════════════════════════════════


class JobStatusResponse(BaseModel):
    """Response body for job status polling."""
    job_id: str
    job_type: str = ""
    model_name: str = ""
    status: str = "queued"
    progress: float = 0.0
    result: Dict[str, Any] = {}
    error_message: str = ""
    queue_position: int = -1
    queue_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    estimated_seconds_remaining: Optional[int] = None


class QueueStatusResponse(BaseModel):
    """Response body for queue statistics."""
    queued: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    total: int = 0
    max_queue_size: int = 50


# ═══════════════════════════════════════════════════
#  MODELS LISTING SCHEMAS
# ═══════════════════════════════════════════════════


class ModelInfo(BaseModel):
    """Information about a single model."""
    name: str
    model_id: str
    category: str
    output_type: str
    description: str
    vram_needed_gb: float
    min_steps: int
    max_steps: int
    default_steps: int
    default_guidance_scale: float
    loaded: bool = False


class ModelsByCategory(BaseModel):
    """Models grouped by category."""
    category: str
    models: List[ModelInfo]
