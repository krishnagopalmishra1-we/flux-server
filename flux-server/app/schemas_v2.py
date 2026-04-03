"""
Request/Response schemas for all generation modalities.

Extends the original image schemas with video, music, animation,
and job status schemas. All schemas use Pydantic v2 for validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
import uuid


# ═══════════════════════════════════════════════════
#  VIDEO SCHEMAS
# ═══════════════════════════════════════════════════


class VideoGenerateRequest(BaseModel):
    """Request body for video generation endpoints."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    model_name: str = Field("ltx-video")
    negative_prompt: Optional[str] = Field(None, max_length=1000)
    num_frames: int = Field(33, ge=16, le=81)
    fps: int = Field(16, ge=8, le=30)
    resolution: str = Field("480p", pattern=r"^(480p|720p)$")
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    num_inference_steps: int = Field(30, ge=10, le=50)
    seed: Optional[int] = None
    # For Image-to-Video: base64-encoded source image
    source_image_b64: Optional[str] = None
    # LoRA support for Wan 2.2 models
    lora_name: Optional[str] = None
    lora_scale: float = Field(1.0, ge=0.0, le=2.0)

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
#  MUSIC SCHEMAS
# ═══════════════════════════════════════════════════


class MusicGenerateRequest(BaseModel):
    """Request body for music/song generation."""
    prompt: str = Field(..., min_length=1, max_length=2000)
    model_name: str = Field("audioldm2")
    duration_seconds: int = Field(30, ge=5, le=300)
    # ACE-Step specific: lyrics for vocal generation
    lyrics: Optional[str] = Field(None, max_length=5000)
    genre: Optional[str] = Field(None, max_length=200)
    bpm: Optional[int] = Field(None, ge=40, le=240)
    # MusicGen specific: base64 melody audio for conditioning
    melody_audio_b64: Optional[str] = None
    seed: Optional[int] = None

    @field_validator("prompt")
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        return "".join(c for c in v if c.isprintable())


class MusicGenerateResponse(BaseModel):
    """Response body for music generation."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "queued"
    audio_url: Optional[str] = None
    audio_b64: Optional[str] = None  # For short clips, can return inline
    duration_seconds: float = 0.0
    sample_rate: int = 0
    inference_time_ms: float = 0.0
    queue_position: int = 0


# ═══════════════════════════════════════════════════
#  ANIMATION SCHEMAS
# ═══════════════════════════════════════════════════


class AnimationGenerateRequest(BaseModel):
    """Request body for audio-driven animation."""
    model_name: str = Field("echomimic")
    # Source face image (base64 PNG/JPG)
    source_image_b64: str = Field(..., min_length=1)
    # Driving audio (base64 WAV/MP3)
    audio_b64: str = Field(..., min_length=1)
    expression_scale: float = Field(1.0, ge=0.1, le=3.0)
    pose_style: int = Field(0, ge=0, le=46)
    # Whether to apply face enhancement to the output
    use_enhancer: bool = False


class AnimationGenerateResponse(BaseModel):
    """Response body for animation generation."""
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
