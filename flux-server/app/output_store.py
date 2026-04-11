"""
Output storage manager for generated files (video, audio, animation).

Manages the output directory, generates unique filenames, serves files
via FastAPI, and auto-cleans expired outputs.
"""

import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


class OutputStore:
    """
    Manages persistent file storage for generated outputs.

    Features:
    - Organized subdirectories by type (video, audio, animation)
    - Unique filename generation with job prefix
    - TTL-based auto-cleanup
    - Storage usage tracking
    """

    SUBDIRS = ("video", "audio", "animation")

    def __init__(self, base_dir: Optional[str] = None, ttl_hours: int = 24):
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.output_dir)
        self.ttl_seconds = ttl_hours * 3600

        # Create directory structure
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for subdir in self.SUBDIRS:
            (self.base_dir / subdir).mkdir(exist_ok=True)

        logger.info(
            f"OutputStore initialized: {self.base_dir} "
            f"(TTL={ttl_hours}h)"
        )

    def check_disk_space(self, min_gb: float = 10.0) -> None:
        """Raise RuntimeError if free disk space is below min_gb.

        Called before starting any generation to avoid writing corrupt
        partial outputs when the disk is nearly full.
        """
        usage = shutil.disk_usage(str(self.base_dir))
        free_gb = usage.free / 1e9
        if free_gb < min_gb:
            raise RuntimeError(
                f"Insufficient disk space: {free_gb:.1f}GB free, {min_gb}GB required. "
                f"Delete old outputs or expand storage."
            )

    def save_file(
        self,
        content: bytes,
        file_type: str,
        extension: str,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Save generated content to disk.

        Args:
            content: Raw file bytes
            file_type: Subdirectory name ("video", "audio", "animation")
            extension: File extension including dot (e.g. ".mp4", ".wav")
            job_id: Optional job ID prefix for the filename

        Returns:
            Relative path from base_dir (e.g. "video/abc123.mp4")
        """
        if file_type not in self.SUBDIRS:
            raise ValueError(f"Unknown file type: {file_type}. Valid: {self.SUBDIRS}")

        prefix = job_id[:12] if job_id else uuid.uuid4().hex[:12]
        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}{extension}"
        rel_path = f"{file_type}/{filename}"
        abs_path = self.base_dir / rel_path

        abs_path.write_bytes(content)
        size_mb = len(content) / (1024 * 1024)
        logger.info(f"Saved output: {rel_path} ({size_mb:.1f} MB)")
        return rel_path

    def save_file_from_path(
        self,
        source_path: str | Path,
        file_type: str,
        job_id: Optional[str] = None,
    ) -> str:
        """
        Move/copy a generated file from a temp location to the output store.

        Args:
            source_path: Absolute path to the source file
            file_type: Subdirectory name
            job_id: Optional job ID prefix

        Returns:
            Relative path from base_dir
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        extension = source.suffix
        prefix = job_id[:12] if job_id else uuid.uuid4().hex[:12]
        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}{extension}"
        rel_path = f"{file_type}/{filename}"
        dest = self.base_dir / rel_path

        shutil.move(str(source), str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"Stored output: {rel_path} ({size_mb:.1f} MB)")
        return rel_path

    def get_absolute_path(self, rel_path: str) -> Path:
        """Get absolute filesystem path for a relative output path."""
        abs_path = (self.base_dir / rel_path).resolve()
        # Security: ensure path is within base_dir
        if not str(abs_path).startswith(str(self.base_dir.resolve())):
            raise ValueError(f"Path traversal detected: {rel_path}")
        return abs_path

    def get_url(self, rel_path: str) -> str:
        """Get the URL path for serving a file."""
        return f"/outputs/{rel_path}"

    def file_exists(self, rel_path: str) -> bool:
        """Check if an output file exists."""
        return (self.base_dir / rel_path).exists()

    def delete_file(self, rel_path: str) -> bool:
        """Delete an output file. Returns True if deleted."""
        path = self.base_dir / rel_path
        if path.exists():
            path.unlink()
            logger.info(f"Deleted output: {rel_path}")
            return True
        return False

    def cleanup_expired(self) -> int:
        """
        Remove files older than the configured TTL.

        Returns:
            Number of files deleted
        """
        now = time.time()
        deleted = 0

        for subdir in self.SUBDIRS:
            dir_path = self.base_dir / subdir
            if not dir_path.exists():
                continue

            for file in dir_path.iterdir():
                if not file.is_file():
                    continue
                age = now - file.stat().st_mtime
                if age > self.ttl_seconds:
                    file.unlink()
                    deleted += 1

        if deleted:
            logger.info(f"Cleaned up {deleted} expired output files")
        return deleted

    def storage_stats(self) -> dict:
        """Get storage usage statistics."""
        stats = {"total_files": 0, "total_size_mb": 0.0, "by_type": {}}

        for subdir in self.SUBDIRS:
            dir_path = self.base_dir / subdir
            if not dir_path.exists():
                stats["by_type"][subdir] = {"files": 0, "size_mb": 0.0}
                continue

            files = list(dir_path.iterdir())
            size = sum(f.stat().st_size for f in files if f.is_file())
            count = sum(1 for f in files if f.is_file())

            stats["by_type"][subdir] = {
                "files": count,
                "size_mb": round(size / (1024 * 1024), 1),
            }
            stats["total_files"] += count
            stats["total_size_mb"] += size / (1024 * 1024)

        stats["total_size_mb"] = round(stats["total_size_mb"], 1)
        return stats

# Lazy singleton — avoid import-time filesystem operations
_output_store_instance: OutputStore | None = None


def get_output_store() -> OutputStore:
    """Get or create the OutputStore singleton."""
    global _output_store_instance
    if _output_store_instance is None:
        _output_store_instance = OutputStore()
    return _output_store_instance


# Backward-compatible module-level access (initialized on first use)
class _LazyOutputStore:
    """Proxy that lazily initializes the real OutputStore on first use."""
    def __getattr__(self, name):
        return getattr(get_output_store(), name)


output_store = _LazyOutputStore()
