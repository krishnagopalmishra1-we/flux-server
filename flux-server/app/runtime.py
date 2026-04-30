"""Runtime coordination for the single-GPU deployment."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class GpuRuntimeCoordinator:
    """Serialize all GPU model load/unload/inference operations.

    The service runs on one A100 40GB. Image and video pipelines unload each
    other to stay within VRAM, so they must never overlap.
    """

    def __init__(self) -> None:
        # Single GPU lock — image and video paths each unload the other's model, so concurrent access causes OOM or CUDA assertion failures.
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def claim(self) -> AsyncIterator[None]:
        async with self.lock:
            yield


gpu_runtime = GpuRuntimeCoordinator()
