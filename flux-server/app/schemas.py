from pydantic import BaseModel, Field, field_validator
from typing import Optional
import uuid


class GenerateRequest(BaseModel):
    """Request body for the /generate endpoint."""

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
    """Response body from the /generate endpoint."""

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
