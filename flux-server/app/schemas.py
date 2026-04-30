"""
Request/Response schemas for the Neural Creation Studio API.

Covers image generation, video generation, job status, and queue management.
All schemas use Pydantic v2 for validation.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Dict, Any
import uuid

from app.config import get_settings as _get_settings
_cfg = _get_settings()


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
    num_frames: int = Field(_cfg.default_video_frames, ge=16, le=480)
    fps: int = Field(_cfg.default_video_fps, ge=8, le=30)
    # A100-safe default: 480p short clips on Wan 1.3B. Higher tiers are
    # explicitly model-gated below to avoid wasting scarce GPU/disk budget.
    resolution: str = Field("480p", pattern=r"^(480p|540p|720p)$")
    # guidance_scale=5.0: WAN optimal for 1.3B. Higher values cause saturation artifacts.
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    # 30 steps: good quality/speed balance for all models.
    num_inference_steps: int = Field(30, ge=10, le=100)
    seed: Optional[int] = None
    # For Image-to-Video: base64-encoded source image
    source_image_b64: Optional[str] = None
    # LoRA support for Wan 2.2 models
    lora_name: Optional[str] = None
    lora_scale: float = Field(1.0, ge=0.0, le=2.0)
    # Chunked generation — for num_frames > chunk_size, generate in chunks.
    # chunk_size=49: avoids quadratic attention OOM at 81fr (AGENT.md §Speed Optimization).
    # chunk_overlap=8: _blend_overlap only fades EDGE_FADE=4 frames; 16 overlap was
    #   wasting 12 inference frames per chunk with no quality benefit.
    chunk_size: int = Field(49, ge=16, le=81)
    chunk_overlap: int = Field(8, ge=0, le=30)

    @field_validator("prompt")
    @classmethod
    def sanitize_prompt(cls, v: str) -> str:
        return "".join(c for c in v if c.isprintable())

    @model_validator(mode="after")
    def enforce_a100_limits(self) -> "VideoGenerateRequest":
        limits = {
            "wan-t2v-1.3b": {
                "resolutions": {"480p"},
                "max_frames": 240,
                "max_steps": 40,
                "max_chunk": 49,
                "max_overlap": 16,
            },
            "wan-t2v-14b": {
                "resolutions": {"480p", "540p", "720p"},
                "max_frames": 129,
                "max_steps": 50,
                "max_chunk": 49,
                "max_overlap": 16,
            },
            "wan-i2v-14b": {
                "resolutions": {"480p", "540p", "720p"},
                "max_frames": 81,
                "max_steps": 50,
                "max_chunk": 49,
                "max_overlap": 16,
            },
            "hunyuan-video": {
                "resolutions": {"540p", "720p"},
                "max_frames": 129,
                "max_steps": 60,
                "max_chunk": 49,
                "max_overlap": 16,
            },
        }
        cfg = limits.get(self.model_name)
        if cfg is None:
            raise ValueError(f"Unsupported video model: {self.model_name}")
        if self.source_image_b64 and self.model_name != "wan-i2v-14b":
            raise ValueError("Image-to-video requires model_name='wan-i2v-14b'.")
        if self.resolution not in cfg["resolutions"]:
            allowed = ", ".join(sorted(cfg["resolutions"]))
            raise ValueError(f"{self.model_name} supports resolution(s): {allowed}.")
        if self.num_frames > cfg["max_frames"]:
            raise ValueError(f"{self.model_name} is capped at {cfg['max_frames']} frames on A100 40GB.")
        if self.num_inference_steps > cfg["max_steps"]:
            raise ValueError(f"{self.model_name} is capped at {cfg['max_steps']} inference steps.")
        if self.chunk_size > cfg["max_chunk"]:
            raise ValueError(f"{self.model_name} is capped at chunk_size={cfg['max_chunk']}.")
        if self.chunk_overlap > cfg["max_overlap"]:
            raise ValueError(f"{self.model_name} is capped at chunk_overlap={cfg['max_overlap']}.")
        return self

    @model_validator(mode="after")
    def validate_video_request(self) -> "VideoGenerateRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.source_image_b64 and self.model_name != "wan-i2v-14b":
            raise ValueError(
                "source_image_b64 requires model_name='wan-i2v-14b'. "
                "Text-to-video models do not accept an input image."
            )
        return self


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
    """Response body for job status polling.

    On completion, video result fields are promoted to the top level
    (video_url, thumbnail_b64, etc.) for easy client consumption.
    The raw `result` dict is also included for backward compatibility.
    """
    job_id: str
    job_type: str = ""
    model_name: str = ""
    status: str = "queued"
    progress: float = 0.0
    # progress_pct mirrors `progress` — both names accepted by clients.
    progress_pct: float = 0.0
    result: Dict[str, Any] = {}
    error_message: str = ""
    queue_position: int = -1
    queue_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    estimated_seconds_remaining: Optional[int] = None
    # ── Promoted video result fields (populated from result on completion) ──
    video_url: Optional[str] = None
    thumbnail_b64: Optional[str] = None
    duration_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    seed_used: Optional[int] = None
    num_frames: Optional[int] = None
    chunks_generated: Optional[int] = None

    @model_validator(mode="after")
    def _promote_result_fields(self) -> "JobStatusResponse":
        """Populate top-level video fields from the result dict when the job
        is completed. This keeps the schema self-documenting while maintaining
        backward compatibility with clients that read result directly."""
        # Sync progress_pct alias
        self.progress_pct = self.progress
        if self.status == "completed" and self.result:
            r = self.result
            if self.video_url is None:
                self.video_url = r.get("video_url") or r.get("video_url")
            if self.thumbnail_b64 is None:
                self.thumbnail_b64 = r.get("thumbnail_b64")
            if self.duration_seconds is None:
                self.duration_seconds = r.get("duration_seconds")
            if self.inference_time_ms is None:
                self.inference_time_ms = r.get("inference_time_ms")
            if self.seed_used is None:
                self.seed_used = r.get("seed_used")
            if self.num_frames is None:
                self.num_frames = r.get("num_frames")
            if self.chunks_generated is None:
                self.chunks_generated = r.get("chunks_generated")
        return self


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
