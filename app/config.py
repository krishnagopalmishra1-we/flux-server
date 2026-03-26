from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Model
    model_id: str = "black-forest-labs/FLUX.1-dev"
    cache_dir: str = "/mnt/hf-cache"
    hf_token: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1  # 1 worker per GPU

    # Security
    api_keys: str = ""  # Comma-separated valid API keys
    rate_limit_per_minute: int = 10

    # Generation defaults
    default_width: int = 1024
    default_height: int = 1024
    default_steps: int = 28
    default_guidance_scale: float = 3.5
    max_width: int = 2048
    max_height: int = 2048

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
