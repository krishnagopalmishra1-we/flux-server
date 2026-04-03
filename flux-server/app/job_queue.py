"""
Async job queue for multi-modal AI generation.

Manages job submission, priority ordering, status tracking, and result delivery.
Jobs are processed sequentially (single GPU) but accepted concurrently from
multiple users. Designed for FastAPI BackgroundTasks integration.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

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
        }


# Type alias for job handler functions
JobHandler = Callable[[Job], Coroutine[Any, Any, Dict[str, Any]]]


class JobQueue:
    """
    Async job queue with priority ordering and sequential processing.

    Features:
    - Priority-based ordering (images first, video last)
    - Per-user job limits
    - Auto-cleanup of completed jobs after TTL
    - Thread-safe status tracking
    """

    def __init__(
        self,
        max_queue_size: int = 50,
        max_per_user: int = 5,
        result_ttl_seconds: float = 3600,  # Keep results for 1 hour
    ):
        self.max_queue_size = max_queue_size
        self.max_per_user = max_per_user
        self.result_ttl = result_ttl_seconds

        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._jobs: Dict[str, Job] = {}  # All jobs (active + completed)
        self._lock = asyncio.Lock()
        self._handlers: Dict[str, JobHandler] = {}
        self._processing = False
        self._worker_task: Optional[asyncio.Task] = None

        logger.info(
            f"JobQueue initialized (max_size={max_queue_size}, "
            f"max_per_user={max_per_user}, result_ttl={result_ttl_seconds}s)"
        )

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """Register a handler function for a specific job type."""
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

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

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by its ID."""
        return self._jobs.get(job_id)

    def get_queue_position(self, job_id: str) -> int:
        """Get position of a job in the queue (0-indexed, -1 if not queued)."""
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.QUEUED:
            return -1

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
        if user_id:
            jobs = [j for j in jobs if j.user_id == user_id]
        if status:
            jobs = [j for j in jobs if j.status == status]
        return [j.to_dict() for j in jobs[:limit]]

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a queued job. Returns True if cancelled, False if not found or already processing."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status != JobStatus.QUEUED:
            return False
        job.status = JobStatus.CANCELLED
        logger.info(f"Job cancelled: {job_id[:8]}...")
        return True

    def queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
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
        }

    def _ensure_worker(self) -> None:
        """Start the background worker if not running."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def _worker_loop(self) -> None:
        """Background worker that processes jobs sequentially."""
        logger.info("Job queue worker started")
        while True:
            try:
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

    async def _process_job(self, job: Job) -> None:
        """Process a single job using its registered handler."""
        handler = self._handlers.get(job.job_type)
        if not handler:
            job.status = JobStatus.FAILED
            job.error_message = f"No handler for job type: {job.job_type}"
            return

        job.status = JobStatus.PROCESSING
        job.started_at = time.time()
        logger.info(f"Processing job: {job.id[:8]}... (type={job.job_type}, model={job.model_name})")

        try:
            # Run heavy generation handlers in a dedicated thread so the main
            # event loop can continue serving status/health polling requests.
            result = await asyncio.to_thread(self._run_handler_in_thread, handler, job)
            job.result = result
            job.status = JobStatus.COMPLETED
            job.progress = 100.0
            job.completed_at = time.time()
            logger.info(
                f"Job completed: {job.id[:8]}... "
                f"(processing_time={job.processing_time_ms:.0f}ms)"
            )
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = time.time()
            logger.exception(f"Job failed: {job.id[:8]}... — {e}")

    @staticmethod
    def _run_handler_in_thread(handler: JobHandler, job: Job) -> Dict[str, Any]:
        """Execute async handler in an isolated event loop inside a worker thread."""
        return asyncio.run(handler(job))

    def _cleanup_expired(self) -> None:
        """Remove completed/failed jobs older than TTL."""
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
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired jobs")


# Module-level singleton
job_queue = JobQueue()
