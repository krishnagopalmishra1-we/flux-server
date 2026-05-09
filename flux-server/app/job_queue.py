"""
Async job queue for multi-modal AI generation.

Manages job submission, priority ordering, status tracking, and result delivery.
Jobs are processed sequentially (single GPU) but accepted concurrently from
multiple users. Designed for FastAPI BackgroundTasks integration.
"""

import asyncio
import json
import logging
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Lifecycle states for a generation job."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class JobPriority(int, Enum):
    """Priority levels — lower number = processed first."""
    INSTANT = 0   # Image generation (<15s)
    FAST = 1      # Music generation (~30s)
    NORMAL = 2    # Animation (~1-2min)
    SLOW = 3      # Video generation (2-10min)


@dataclass
class Job:
    """Represents a single generation job."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    job_type: str = ""          # "image", "video", "music", "animation"
    model_name: str = ""
    status: JobStatus = JobStatus.QUEUED
    priority: JobPriority = JobPriority.NORMAL
    progress: float = 0.0       # 0-100%
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    user_id: str = "anonymous"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_updated_at: float = field(default_factory=time.time)
    # Cancellation flag — checked during inference step callbacks.
    cancel_flag: bool = False
    # Inference timing — set when the first inference step begins (after model load).
    inference_start_time: Optional[float] = None
    estimated_seconds_remaining: Optional[int] = None
    owner_id: Optional[str] = None
    cancel_checker: Optional[Callable[[], bool]] = field(default=None, repr=False, compare=False)

    @property
    def queue_time_ms(self) -> float:
        """Time spent in queue before processing started."""
        if self.started_at:
            return (self.started_at - self.created_at) * 1000
        return (time.time() - self.created_at) * 1000

    @property
    def processing_time_ms(self) -> float:
        """Time spent processing (generating)."""
        if not self.started_at:
            return 0.0
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Serialize job state for API responses."""
        return {
            "job_id": self.id,
            "job_type": self.job_type,
            "model_name": self.model_name,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "result": self.result,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "queue_time_ms": round(self.queue_time_ms, 0),
            "processing_time_ms": round(self.processing_time_ms, 0),
            "estimated_seconds_remaining": self.estimated_seconds_remaining,
            "owner_id": self.owner_id,
            "cancel_flag": self.cancel_flag,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "last_updated_at": self.last_updated_at,
            "priority": int(self.priority.value),
            "payload": self.payload,
        }

    def to_record(self) -> Dict[str, Any]:
        """Serialize full persistent job state, excluding local SSE/callback objects."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "model_name": self.model_name,
            "status": self.status.value,
            "priority": int(self.priority.value),
            "progress": self.progress,
            "payload": self.payload,
            "result": self.result,
            "error_message": self.error_message,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "last_updated_at": self.last_updated_at,
            "cancel_flag": self.cancel_flag,
            "inference_start_time": self.inference_start_time,
            "estimated_seconds_remaining": self.estimated_seconds_remaining,
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_record(cls, data: Dict[str, Any]) -> "Job":
        job = cls(
            id=data["id"],
            job_type=data.get("job_type", ""),
            model_name=data.get("model_name", ""),
            status=JobStatus(data.get("status", JobStatus.QUEUED.value)),
            priority=JobPriority(int(data.get("priority", JobPriority.NORMAL.value))),
            progress=float(data.get("progress", 0.0)),
            payload=data.get("payload") or {},
            result=data.get("result") or {},
            error_message=data.get("error_message", ""),
            user_id=data.get("user_id", "anonymous"),
            created_at=float(data.get("created_at") or time.time()),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            last_updated_at=float(data.get("last_updated_at") or time.time()),
            cancel_flag=bool(data.get("cancel_flag", False)),
            inference_start_time=data.get("inference_start_time"),
            estimated_seconds_remaining=data.get("estimated_seconds_remaining"),
            owner_id=data.get("owner_id"),
        )
        return job

    def cancel_requested(self) -> bool:
        if self.cancel_flag:
            return True
        if self.cancel_checker is not None:
            try:
                self.cancel_flag = bool(self.cancel_checker())
            except Exception:
                logger.warning("Cancel check failed for job %s", self.id[:8], exc_info=True)
        return self.cancel_flag


# Type alias for job handler functions
JobHandler = Callable[[Job], Dict[str, Any]]


class JobQueue:
    """
    Async job queue with priority ordering and sequential processing.

    Features:
    - Priority-based ordering (images first, video last)
    - Per-user job limits
    - Auto-cleanup of completed jobs after TTL
    - Thread-safe status tracking
    - Real-time progress callbacks for SSE streaming
    """

    def __init__(
        self,
        max_queue_size: int = 50,
        max_per_user: int = 5,
        result_ttl_seconds: float = 3600,
    ):
        self.max_queue_size = max_queue_size
        self.max_per_user = max_per_user
        self.result_ttl = result_ttl_seconds
        settings = get_settings()
        self.backend = (settings.job_backend or "memory").strip().lower()
        if self.backend not in {"memory", "redis"}:
            self.backend = "memory"
        self.redis_url = settings.redis_url
        self.watchdog_timeout = float(settings.job_watchdog_timeout_seconds)
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._redis = None
        self._redis_sync = None

        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
        self._handlers: Dict[str, JobHandler] = {}
        self._processing = False
        self._worker_task: Optional[asyncio.Task] = None
        self._watchdog_task: Optional[asyncio.Task] = None
        # Shared GPU lock — set by main.py after startup to serialize GPU access
        # with the image generation endpoints, preventing VRAM contention.
        self.gpu_lock: Optional[asyncio.Lock] = None
        # Progress listeners: job_id -> list of asyncio.Queue for SSE subscribers.
        # Protected by _listeners_lock because set_progress is called from worker threads.
        self._progress_listeners: Dict[str, List[asyncio.Queue]] = {}
        self._listeners_lock = threading.Lock()
        # Event loop reference — captured when the worker starts so worker threads
        # can schedule put_nowait via call_soon_threadsafe.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(
            f"JobQueue initialized (max_size={max_queue_size}, "
            f"max_per_user={max_per_user}, result_ttl={result_ttl_seconds}s, "
            f"backend={self.backend}, worker_id={self.worker_id})"
        )

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """Register a handler function for a specific job type."""
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    @property
    def is_redis_backend(self) -> bool:
        return self.backend == "redis"

    def _job_key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def _queue_score(self, job: Job) -> int:
        return int(job.priority.value) * 1_000_000_000_000_000 + int(job.created_at * 1_000_000_000)

    async def start(self) -> None:
        """Initialize backend resources and start worker/watchdog tasks."""
        self._loop = asyncio.get_running_loop()
        if self.is_redis_backend:
            await self._ensure_redis()
            await self.recover_stale_processing_jobs()
            self._ensure_worker()

    async def _ensure_redis(self):
        if not self.is_redis_backend:
            return None
        if self._redis is not None:
            return self._redis
        try:
            import redis
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise RuntimeError("JOB_BACKEND=redis requires the redis Python package.") from exc
        self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
        self._redis_sync = redis.from_url(self.redis_url, decode_responses=True)
        await self._redis.ping()
        logger.info("Redis job backend connected: %s", self.redis_url)
        return self._redis

    def _redis_ping_sync(self) -> None:
        if self.is_redis_backend and self._redis_sync is not None:
            self._redis_sync.ping()

    def _decode_job(self, raw: Optional[str]) -> Optional[Job]:
        if not raw:
            return None
        return Job.from_record(json.loads(raw))

    async def _redis_get_job(self, job_id: str) -> Optional[Job]:
        redis = await self._ensure_redis()
        return self._decode_job(await redis.get(self._job_key(job_id)))

    async def _redis_save_job(self, job: Job) -> None:
        redis = await self._ensure_redis()
        await redis.set(self._job_key(job.id), json.dumps(job.to_record()))
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EXPIRED):
            await redis.expire(self._job_key(job.id), int(self.result_ttl))

    def _redis_save_job_sync(self, job: Job) -> None:
        if self._redis_sync is None:
            return
        self._redis_sync.set(self._job_key(job.id), json.dumps(job.to_record()))
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EXPIRED):
            self._redis_sync.expire(self._job_key(job.id), int(self.result_ttl))

    def _redis_get_job_sync(self, job_id: str) -> Optional[Job]:
        if self._redis_sync is None:
            return None
        return self._decode_job(self._redis_sync.get(self._job_key(job_id)))

    def is_cancel_requested(self, job_id: str) -> bool:
        if self.is_redis_backend:
            job = self._redis_get_job_sync(job_id)
            return bool(job and job.cancel_flag)
        job = self._jobs.get(job_id)
        return bool(job and job.cancel_flag)

    async def recover_stale_processing_jobs(self) -> int:
        if not self.is_redis_backend:
            return 0
        redis = await self._ensure_redis()
        now = time.time()
        recovered = 0
        async for key in redis.scan_iter(match="job:*", count=200):
            job = self._decode_job(await redis.get(key))
            if not job or job.status != JobStatus.PROCESSING:
                continue
            if (now - job.last_updated_at) <= self.watchdog_timeout:
                continue
            job.status = JobStatus.FAILED
            job.cancel_flag = True
            job.error_message = "worker lost/stale processing job"
            job.completed_at = now
            job.last_updated_at = now
            await self._redis_save_job(job)
            recovered += 1
        if recovered:
            logger.warning("Marked %d stale Redis processing job(s) failed on startup", recovered)
        return recovered

    def set_progress(self, job_id: str, progress: float) -> None:
        """Update job progress and notify SSE listeners. Safe to call from worker threads."""
        clamped = min(100.0, max(0.0, progress))
        job = self._jobs.get(job_id)
        if self.is_redis_backend and job is None:
            job = self._redis_get_job_sync(job_id)
        if job:
            job.progress = clamped
            job.last_updated_at = time.time()
            # Compute ETA once enough progress has been made.
            if job.inference_start_time and clamped > 5:
                elapsed = time.time() - job.inference_start_time
                rate = clamped / elapsed  # % per second
                if rate > 0:
                    remaining = (100.0 - clamped) / rate
                    job.estimated_seconds_remaining = int(remaining)
            if self.is_redis_backend:
                self._redis_save_job_sync(job)
                self._jobs[job_id] = job

        # Schedule queue puts on the event loop — put_nowait is not thread-safe from
        # a worker thread, so we use call_soon_threadsafe to hand off to the loop.
        loop = self._loop
        if loop is None or not loop.is_running():
            return

        status = job.status.value if job else "processing"
        event = {"progress": clamped, "status": status}

        with self._listeners_lock:
            listeners = list(self._progress_listeners.get(job_id, []))

        def _put_all():
            for q in listeners:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        loop.call_soon_threadsafe(_put_all)

    def _notify_terminal(self, job: Job) -> None:
        """Notify SSE listeners that a job reached a terminal state."""
        event = {
            "progress": job.progress,
            "status": job.status.value,
            "result": job.result,
            "error": job.error_message,
        }
        with self._listeners_lock:
            listeners = list(self._progress_listeners.get(job.id, []))

        def _put_all():
            for q in listeners:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(_put_all)

    def subscribe_progress(self, job_id: str) -> asyncio.Queue:
        """Subscribe to real-time progress updates for a job. Returns an asyncio.Queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        with self._listeners_lock:
            if job_id not in self._progress_listeners:
                self._progress_listeners[job_id] = []
            self._progress_listeners[job_id].append(q)
        return q

    def unsubscribe_progress(self, job_id: str, q: asyncio.Queue) -> None:
        """Remove an SSE subscriber queue."""
        with self._listeners_lock:
            listeners = self._progress_listeners.get(job_id)
            if listeners and q in listeners:
                listeners.remove(q)
            if not listeners:
                self._progress_listeners.pop(job_id, None)

    async def submit(
        self,
        job_type: str,
        model_name: str,
        payload: Dict[str, Any],
        priority: JobPriority = JobPriority.NORMAL,
        user_id: str = "anonymous",
    ) -> Job:
        """
        Submit a new generation job to the queue.

        Returns the Job object with its assigned ID for status polling.
        Raises ValueError if queue is full or user has too many pending jobs.
        """
        if self.is_redis_backend:
            return await self._submit_redis(job_type, model_name, payload, priority, user_id)

        async with self._lock:
            # Check per-user limit
            user_pending = sum(
                1 for job in self._jobs.values()
                if job.user_id == user_id
                and job.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
            )
            if user_pending >= self.max_per_user:
                raise ValueError(
                    f"Too many pending jobs for user '{user_id}' "
                    f"({user_pending}/{self.max_per_user}). "
                    "Wait for current jobs to complete."
                )

            # Check handler exists
            if job_type not in self._handlers:
                raise ValueError(
                    f"No handler registered for job type: '{job_type}'. "
                    f"Available: {list(self._handlers.keys())}"
                )

            # Create job
            job = Job(
                job_type=job_type,
                model_name=model_name,
                payload=payload,
                priority=priority,
                user_id=user_id,
            )
            self._jobs[job.id] = job

            # Add to priority queue (lower priority number = processed first)
            try:
                self._queue.put_nowait((priority.value, job.created_at, job.id))
            except asyncio.QueueFull:
                del self._jobs[job.id]
                raise ValueError(
                    f"Job queue is full ({self.max_queue_size} jobs). "
                    "Please try again later."
                )

            logger.info(
                f"Job submitted: {job.id[:8]}... "
                f"(type={job_type}, model={model_name}, priority={priority.name}, user={user_id})"
            )

        # Auto-start worker if not running
        self._ensure_worker()
        return job

    async def _submit_redis(
        self,
        job_type: str,
        model_name: str,
        payload: Dict[str, Any],
        priority: JobPriority,
        user_id: str,
    ) -> Job:
        redis = await self._ensure_redis()
        if job_type not in self._handlers:
            raise ValueError(
                f"No handler registered for job type: '{job_type}'. "
                f"Available: {list(self._handlers.keys())}"
            )

        stats = await self._redis_queue_stats()
        if stats["queued"] >= self.max_queue_size:
            raise ValueError(f"Job queue is full ({self.max_queue_size} jobs). Please try again later.")

        pending_for_user = 0
        async for key in redis.scan_iter(match="job:*", count=200):
            job = self._decode_job(await redis.get(key))
            if (
                job
                and job.user_id == user_id
                and job.status in (JobStatus.QUEUED, JobStatus.PROCESSING)
            ):
                pending_for_user += 1
                if pending_for_user >= self.max_per_user:
                    raise ValueError(
                        f"Too many pending jobs for user '{user_id}' "
                        f"({pending_for_user}/{self.max_per_user}). "
                        "Wait for current jobs to complete."
                    )

        job = Job(
            job_type=job_type,
            model_name=model_name,
            payload=payload,
            priority=priority,
            user_id=user_id,
        )
        await self._redis_save_job(job)
        await redis.zadd("job_queue", {job.id: self._queue_score(job)})
        self._jobs[job.id] = job
        logger.info(
            "Redis job submitted: %s... (type=%s, model=%s, priority=%s, user=%s)",
            job.id[:8],
            job_type,
            model_name,
            priority.name,
            user_id,
        )
        self._ensure_worker()
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by its ID."""
        if self.is_redis_backend:
            job = self._redis_get_job_sync(job_id)
            if job:
                self._jobs[job_id] = job
            return job
        return self._jobs.get(job_id)

    def get_queue_position(self, job_id: str) -> int:
        """Get position of a job in the queue (0-indexed, -1 if not queued)."""
        job = self._jobs.get(job_id)
        if self.is_redis_backend:
            job = self.get_job(job_id)
        if not job or job.status != JobStatus.QUEUED:
            return -1

        if self.is_redis_backend and self._redis_sync is not None:
            rank = self._redis_sync.zrank("job_queue", job_id)
            return int(rank) if rank is not None else -1

        # Count jobs ahead in queue by priority + creation time
        position = 0
        for other_job in self._jobs.values():
            if other_job.id == job_id:
                continue
            if other_job.status != JobStatus.QUEUED:
                continue
            if (other_job.priority.value, other_job.created_at) < (job.priority.value, job.created_at):
                position += 1
        return position

    def list_jobs(
        self,
        user_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List jobs, optionally filtered by user and/or status."""
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        if self.is_redis_backend and self._redis_sync is not None:
            jobs = []
            for key in self._redis_sync.scan_iter(match="job:*", count=200):
                job = self._decode_job(self._redis_sync.get(key))
                if job:
                    jobs.append(job)
            jobs.sort(key=lambda j: j.created_at, reverse=True)
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        if status:
            jobs = [j for j in jobs if j.status == status]
        return [j.to_dict() for j in jobs[:limit]]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued or processing job. Returns True if cancelled."""
        if self.is_redis_backend:
            return self._cancel_job_redis_sync(job_id)
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status == JobStatus.QUEUED:
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            job.error_message = "Job cancelled before processing."
            self._notify_terminal(job)
            logger.info(f"Job cancelled (was queued): {job_id[:8]}...")
            return True
        if job.status == JobStatus.PROCESSING:
            # Set cancel flag — checked during inference step callbacks.
            job.cancel_flag = True
            logger.info(f"Job cancel requested (processing): {job_id[:8]}...")
            return True
        return False

    def _cancel_job_redis_sync(self, job_id: str) -> bool:
        if self._redis_sync is None:
            return False
        job = self._redis_get_job_sync(job_id)
        if not job:
            return False
        if job.status == JobStatus.QUEUED:
            self._redis_sync.zrem("job_queue", job_id)
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            job.last_updated_at = job.completed_at
            job.error_message = "Job cancelled before processing."
            job.cancel_flag = True
            self._redis_save_job_sync(job)
            self._notify_terminal(job)
            logger.info("Redis job cancelled (queued): %s...", job_id[:8])
            return True
        if job.status == JobStatus.PROCESSING:
            job.cancel_flag = True
            job.last_updated_at = time.time()
            self._redis_save_job_sync(job)
            self._jobs[job_id] = job
            logger.info("Redis job cancel requested (processing): %s...", job_id[:8])
            return True
        return False

    def queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        if self.is_redis_backend and self._redis_sync is not None:
            return self._redis_queue_stats_sync()

        queued = sum(1 for j in self._jobs.values() if j.status == JobStatus.QUEUED)
        processing = sum(1 for j in self._jobs.values() if j.status == JobStatus.PROCESSING)
        completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)

        return {
            "queued": queued,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "total": len(self._jobs),
            "max_queue_size": self.max_queue_size,
            "backend": self.backend,
            "worker_id": self.worker_id,
        }

    async def _redis_queue_stats(self) -> Dict[str, Any]:
        redis = await self._ensure_redis()
        counts = {status.value: 0 for status in JobStatus}
        total = 0
        async for key in redis.scan_iter(match="job:*", count=200):
            job = self._decode_job(await redis.get(key))
            if job:
                counts[job.status.value] = counts.get(job.status.value, 0) + 1
                total += 1
        return {
            "queued": counts.get("queued", 0),
            "processing": counts.get("processing", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "total": total,
            "max_queue_size": self.max_queue_size,
            "backend": self.backend,
            "worker_id": self.worker_id,
            "redis_url": self.redis_url,
        }

    def _redis_queue_stats_sync(self) -> Dict[str, Any]:
        counts = {status.value: 0 for status in JobStatus}
        total = 0
        for key in self._redis_sync.scan_iter(match="job:*", count=200):
            job = self._decode_job(self._redis_sync.get(key))
            if job:
                counts[job.status.value] = counts.get(job.status.value, 0) + 1
                total += 1
        return {
            "queued": counts.get("queued", 0),
            "processing": counts.get("processing", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "total": total,
            "max_queue_size": self.max_queue_size,
            "backend": self.backend,
            "worker_id": self.worker_id,
            "redis_url": self.redis_url,
        }

    def drain(self) -> Dict[str, int]:
        """Cancel queued jobs and flag processing jobs for cancellation."""
        cancelled = 0
        flagged = 0
        if self.is_redis_backend and self._redis_sync is not None:
            for key in self._redis_sync.scan_iter(match="job:*", count=200):
                job = self._decode_job(self._redis_sync.get(key))
                if not job:
                    continue
                if job.status == JobStatus.QUEUED:
                    self._redis_sync.zrem("job_queue", job.id)
                    job.status = JobStatus.CANCELLED
                    job.cancel_flag = True
                    job.completed_at = time.time()
                    job.last_updated_at = job.completed_at
                    job.error_message = "Job cancelled by admin queue drain."
                    self._redis_save_job_sync(job)
                    self._notify_terminal(job)
                    cancelled += 1
                elif job.status == JobStatus.PROCESSING:
                    job.cancel_flag = True
                    job.last_updated_at = time.time()
                    self._redis_save_job_sync(job)
                    flagged += 1
            return {"cancelled": cancelled, "flagged": flagged}

        for job in list(self._jobs.values()):
            if job.status == JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                job.error_message = "Job cancelled by admin queue drain."
                self._notify_terminal(job)
                cancelled += 1
            elif job.status == JobStatus.PROCESSING:
                job.cancel_flag = True
                flagged += 1
        return {"cancelled": cancelled, "flagged": flagged}

    def _ensure_worker(self) -> None:
        """Start the background worker if not running."""
        if self._worker_task is None or self._worker_task.done():
            # Capture the running event loop so worker threads can use call_soon_threadsafe.
            self._loop = asyncio.get_running_loop()
            self._worker_task = asyncio.create_task(self._worker_loop())
            if self._watchdog_task is None or self._watchdog_task.done():
                self._watchdog_task = asyncio.create_task(self._watchdog_loop())

    async def _watchdog_loop(self) -> None:
        """Monitor processing jobs for stalled pipelines."""
        while True:
            await asyncio.sleep(60.0)
            now = time.time()
            timeout_seconds = self.watchdog_timeout
            if self.is_redis_backend:
                await self.recover_stale_processing_jobs()
                continue
            for job in list(self._jobs.values()):
                if job.status == JobStatus.PROCESSING:
                    if (now - getattr(job, 'last_updated_at', job.started_at or now)) > timeout_seconds:
                        logger.error(f"Watchdog: Job {job.id} stalled! No progress for {timeout_seconds}s.")
                        job.status = JobStatus.FAILED
                        job.cancel_flag = True
                        logger.warning(f"Watchdog cancelled job {job.id} — cancel_flag set; handler result will be ignored.")
                        job.error_message = f"Job stalled (watchdog timeout after {timeout_seconds}s without progress)."
                        job.completed_at = now
                        
                        # Notify subscribers about failure
                        status_event = {"progress": job.progress, "status": "failed", "error": job.error_message}
                        with self._listeners_lock:
                            listeners = list(self._progress_listeners.get(job.id, []))
                        def _put_all():
                            for q in listeners:
                                try:
                                    q.put_nowait(status_event)
                                except asyncio.QueueFull:
                                    pass
                        if self._loop and self._loop.is_running():
                            self._loop.call_soon_threadsafe(_put_all)

    async def _worker_loop(self) -> None:
        """Background worker that processes jobs sequentially."""
        logger.info("Job queue worker started")
        while True:
            try:
                if self.is_redis_backend:
                    job_id = await self._claim_redis_job_id()
                    if not job_id:
                        self._cleanup_expired()
                        continue
                    job = await self._redis_get_job(job_id)
                    if not job:
                        continue
                    if job.status == JobStatus.CANCELLED or job.cancel_flag:
                        continue
                    await self._process_job(job)
                    self._cleanup_expired()
                    continue

                # Wait for next job (blocks until available)
                priority_value, created_at, job_id = await asyncio.wait_for(
                    self._queue.get(), timeout=30.0
                )
            except asyncio.TimeoutError:
                # Periodic cleanup
                self._cleanup_expired()
                continue
            except Exception as e:
                logger.error(f"Queue worker error: {e}")
                await asyncio.sleep(1)
                continue

            job = self._jobs.get(job_id)
            if not job:
                continue
            if job.status == JobStatus.CANCELLED:
                continue

            # Process the job
            await self._process_job(job)

            # Cleanup after each job
            self._cleanup_expired()

    async def _claim_redis_job_id(self) -> Optional[str]:
        redis = await self._ensure_redis()
        try:
            result = await redis.bzpopmin("job_queue", timeout=30)
        except Exception as exc:
            logger.error("Redis queue claim failed: %s", exc)
            await asyncio.sleep(1.0)
            return None
        if not result:
            return None
        _queue_name, job_id, _score = result
        return str(job_id)

    async def _process_job(self, job: Job) -> None:
        """Process a single job using its registered handler."""
        handler = self._handlers.get(job.job_type)
        if not handler:
            job.status = JobStatus.FAILED
            job.error_message = f"No handler for job type: {job.job_type}"
            return

        job.status = JobStatus.PROCESSING
        job.started_at = time.time()
        job.last_updated_at = time.time()
        job.owner_id = self.worker_id
        if self.is_redis_backend:
            job.cancel_checker = lambda job_id=job.id: self.is_cancel_requested(job_id)
            await self._redis_save_job(job)
            self._jobs[job.id] = job
        logger.info(f"Processing job: {job.id[:8]}... (type={job.job_type}, model={job.model_name})")

        try:
            # Run heavy generation handlers in a dedicated thread so the main
            # event loop can continue serving status/health polling requests.
            # Acquire the shared GPU lock (if set) to prevent concurrent VRAM
            # access with the image generation endpoints.
            if self.gpu_lock is not None:
                async with self.gpu_lock:
                    result = await asyncio.to_thread(self._run_handler_in_thread, handler, job)
            else:
                result = await asyncio.to_thread(self._run_handler_in_thread, handler, job)
            job.result = result
            if self.is_redis_backend:
                latest = await self._redis_get_job(job.id)
                if latest and latest.cancel_flag:
                    job.cancel_flag = True
                    job.status = JobStatus.CANCELLED
            if job.status in (JobStatus.FAILED, JobStatus.CANCELLED) or job.cancel_requested():
                logger.info(f"Job {job.id} handler returned but status is already {job.status} — skipping COMPLETED write.")
                if self.is_redis_backend:
                    job.completed_at = job.completed_at or time.time()
                    job.last_updated_at = job.completed_at
                    await self._redis_save_job(job)
                return
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = time.time()
            job.last_updated_at = job.completed_at
            if self.is_redis_backend:
                await self._redis_save_job(job)
            self._notify_terminal(job)
            logger.info(
                f"Job completed: {job.id[:8]}... "
                f"(processing_time={job.processing_time_ms:.0f}ms)"
            )
        except InterruptedError as e:
            job.status = JobStatus.CANCELLED
            job.error_message = str(e) or "Job cancelled by user request"
            job.completed_at = time.time()
            job.last_updated_at = job.completed_at
            if self.is_redis_backend:
                await self._redis_save_job(job)
            self._notify_terminal(job)
            logger.info(f"Job cancelled: {job.id[:8]}...")
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = time.time()
            job.last_updated_at = job.completed_at
            if self.is_redis_backend:
                await self._redis_save_job(job)
            self._notify_terminal(job)
            logger.exception(f"Job failed: {job.id[:8]}... — {e}")

    @staticmethod
    def _run_handler_in_thread(handler: JobHandler, job: Job) -> Dict[str, Any]:
        """Execute handler directly — already running in a worker thread off the event loop."""
        return handler(job)

    def _cleanup_expired(self) -> None:
        """Remove completed/failed jobs older than TTL and prune their listener entries."""
        now = time.time()
        expired_ids = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.EXPIRED)
            and job.completed_at
            and (now - job.completed_at) > self.result_ttl
        ]
        for job_id in expired_ids:
            del self._jobs[job_id]
        # Prune listener entries for expired jobs so the dict doesn't grow unboundedly.
        if expired_ids:
            with self._listeners_lock:
                for job_id in expired_ids:
                    self._progress_listeners.pop(job_id, None)
            logger.info(f"Cleaned up {len(expired_ids)} expired jobs")


# Module-level singleton — limits driven by config so .env overrides take effect.
_cfg = get_settings()
job_queue = JobQueue(
    max_queue_size=_cfg.max_queue_size,
    max_per_user=_cfg.max_jobs_per_user,
    result_ttl_seconds=_cfg.job_result_ttl_seconds,
)
