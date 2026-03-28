from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader
from collections import defaultdict
import time
from app.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory rate limiter (sufficient for single-instance deployment)
_request_timestamps: dict[str, list[float]] = defaultdict(list)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the API key from the X-API-Key header."""
    settings = get_settings()
    valid_keys = [k.strip() for k in settings.api_keys.split(",") if k.strip()]

    if not valid_keys:
        return "anonymous"  # No keys configured = open access

    if not api_key or api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key


def check_rate_limit(request: Request, api_key: str) -> None:
    """Sliding-window rate limiter (per API key or IP)."""
    settings = get_settings()
    now = time.time()
    window = 60.0  # 1 minute
    client_id = api_key or request.client.host

    # Purge old timestamps
    _request_timestamps[client_id] = [
        ts for ts in _request_timestamps[client_id] if now - ts < window
    ]

    if len(_request_timestamps[client_id]) >= settings.rate_limit_per_minute:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {settings.rate_limit_per_minute} requests/minute.",
            headers={"Retry-After": "60"},
        )

    _request_timestamps[client_id].append(now)
