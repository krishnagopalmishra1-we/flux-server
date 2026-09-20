from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Model
    model_id: str = "black-forest-labs/FLUX.1-dev"

    # Cache directories — split by disk tier for speed.
    # High-priority models (FLUX) go on SSD (/app/model_cache).
    # Other models fall back to /mnt/hf-cache.
    cache_dir: str = "/mnt/hf-cache"        # default / fallback
    cache_dir_ssd: str = "/app/model_cache" # SSD — fast-path for priority models

    hf_token: str = ""
    sd3_hf_token: str = ""  # Separate token for SD3/SD3.5 gated models

    # Offline mode — set after first cache fill to skip HF network metadata checks.
    # Eliminates network overhead per model load.
    hf_offline: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1  # 1 worker per GPU

    # Security
    api_keys: str = ""  # Comma-separated valid API keys
    rate_limit_per_minute: int = 10

    # CORS — restrict to your frontend domain in production.
    # Use "*" for dev (default). Comma-separated origins for production.
    cors_origins: str = "*"

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

    # LoRA storage. Keep these on persistent mounted storage so uploaded
    # adapters survive container rebuilds and are visible to list/load paths.
    lora_dir: str = "/mnt/hf-cache/loras"
    max_lora_upload_mb: int = 1536

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
