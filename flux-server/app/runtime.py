"""Runtime coordination for the single-GPU deployment."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class GpuRuntimeCoordinator:
    """Serialize all GPU model load/unload/inference operations.

    The service is tuned for a single A10G 24GB image-generation deployment.
    Requests are serialized to avoid overlapping model loads and inference.
    """

    def __init__(self) -> None:
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def claim(self) -> AsyncIterator[None]:
        async with self.lock:
            yield


gpu_runtime = GpuRuntimeCoordinator()
