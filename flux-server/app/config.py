from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Model
    model_id: str = "black-forest-labs/FLUX.1-dev"

    # Cache directories — split by disk tier for speed.
    # High-priority models (FLUX, WAN 1.3B, HunyuanVideo) go on SSD (/app/model_cache).
    # Low-priority models (WAN 14B, I2V 14B, LTX) fall back to HDD (/mnt/hf-cache).
    # Set cache_dir_ssd=/mnt/hf-cache to disable split caching (single disk).
    cache_dir: str = "/mnt/hf-cache"        # default / HDD fallback
    cache_dir_ssd: str = "/app/model_cache" # SSD — fast-path for priority models

    hf_token: str = ""
    sd3_hf_token: str = ""  # Separate token for SD3 gated models

    # Offline mode — set after first cache fill to skip HF network metadata checks.
    # Eliminates ~2-30s of network overhead per model load.
    hf_offline: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1  # 1 worker per GPU

    # Security
    api_keys: str = ""  # Comma-separated valid API keys
    rate_limit_per_minute: int = 10

    # Generation defaults (image)
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 28
    default_guidance_scale: float = 3.5
    max_width: int = 2048
    max_height: int = 2048

    # Output storage
    output_dir: str = "/mnt/outputs"
    output_ttl_hours: int = 24  # Auto-cleanup generated files after N hours

    # Job queue
    max_queue_size: int = 50
    max_jobs_per_user: int = 5

    # Feature flags — enable/disable modalities
    enable_video: bool = True
    enable_music: bool = True
    enable_animation: bool = True

    # Video defaults
    wan_default_variant: str = "14b"   # "1.3b" or "14b"
    default_video_fps: int = 16
    default_video_frames: int = 33

    # Music defaults
    default_music_duration: int = 30  # seconds

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
