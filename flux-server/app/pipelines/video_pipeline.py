"""
Video generation pipeline — Text-to-Video and Image-to-Video.

Supports:
- Wan 2.2 T2V (1.3B and 14B variants)
- Wan 2.2 I2V (Image-to-Video)
- LTX Video (fast drafts)

All models share the same interface: accept a prompt/settings, return an MP4 file path.
"""

import gc
import io
import time
import base64
import logging
import tempfile
import importlib
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from PIL import Image

from app.config import get_settings
from app.output_store import output_store

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
#  CUDA OPTIMIZATIONS — Faster math without quality loss
# ═══════════════════════════════════════════════════
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True  # Fast matrix multiply with TF32
    torch.backends.cudnn.allow_tf32 = True        # Fast cuDNN ops with TF32

# Resolution presets → actual pixel dimensions
RESOLUTION_MAP = {
    "480p": (480, 848),
    "720p": (720, 1280),
}


class VideoPipeline:
    """
    Manages video generation using Wan 2.2 and LTX Video models.

    Only one video model lives in VRAM at a time. The parent ModelManager
    handles unloading previous models before calling into this pipeline.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = None
        self._current_model = None

    def _resolve_pipeline_class(self, class_name: str, module_candidates: list[str]):
        """Resolve a diffusers pipeline class across versions/exports."""
        try:
            import diffusers
            cls = getattr(diffusers, class_name, None)
            if cls is not None:
                return cls
        except Exception:
            pass

        for module_name in module_candidates:
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name, None)
                if cls is not None:
                    return cls
            except Exception:
                continue

        raise ImportError(
            f"{class_name} is not available in this diffusers build. "
            "Upgrade diffusers to a Wan/LTX-capable version."
        )

    def load_model(self, model_name: str) -> None:
        """Load a video model into VRAM."""
        if self._current_model == model_name and self._pipe is not None:
            logger.info(f"Video model {model_name} already loaded")
            return

        self.unload()
        settings = get_settings()
        logger.info(f"Loading video model: {model_name}")

        try:
            if model_name == "wan-t2v-1.3b":
                self._load_wan_t2v(
                    model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                    quantize=False,
                    settings=settings,
                )
            elif model_name == "wan-t2v-14b":
                self._load_wan_t2v(
                    model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    quantize=True,
                    settings=settings,
                )
            elif model_name == "wan-i2v-14b":
                self._load_wan_i2v(
                    model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
                    settings=settings,
                )
            elif model_name == "ltx-video":
                self._load_ltx_video(
                    model_id="Lightricks/LTX-Video",
                    settings=settings,
                )
            else:
                raise ValueError(f"Unknown video model: {model_name}")

            self._current_model = model_name
            logger.info(f"✅ Video model {model_name} loaded")

        except Exception as e:
            logger.exception(f"Failed to load video model {model_name}: {e}")
            self.unload()
            raise

    def _load_wan_t2v(self, model_id: str, quantize: bool, settings) -> None:
        """Load Wan 2.2 Text-to-Video pipeline with optimizations."""
        WanPipeline = self._resolve_pipeline_class(
            "WanPipeline",
            [
                "diffusers.pipelines.wan.pipeline_wan",
                "diffusers.pipelines.wan",
            ],
        )

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "cache_dir": settings.cache_dir,
        }
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        if quantize:
            # Use CPU offload for 14B model to fit in 40GB
            self._pipe = WanPipeline.from_pretrained(model_id, **load_kwargs)
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe = WanPipeline.from_pretrained(model_id, **load_kwargs)
            self._pipe.to(self.device)

        # Memory-efficient attention optimizations
        if hasattr(self._pipe, "enable_vae_slicing"):
            self._pipe.enable_vae_slicing()
        if hasattr(self._pipe, "enable_vae_tiling"):
            self._pipe.enable_vae_tiling()
        if hasattr(self._pipe, "enable_attention_slicing"):
            self._pipe.enable_attention_slicing()
        if hasattr(self._pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
                logger.info("✓ xFormers memory efficient attention enabled")
            except Exception:
                pass  # xFormers optional

    def _load_wan_i2v(self, model_id: str, settings) -> None:
        """Load Wan 2.2 Image-to-Video pipeline with optimizations."""
        WanImageToVideoPipeline = self._resolve_pipeline_class(
            "WanImageToVideoPipeline",
            [
                "diffusers.pipelines.wan.pipeline_wan_i2v",
                "diffusers.pipelines.wan",
            ],
        )

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "cache_dir": settings.cache_dir,
        }
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        self._pipe = WanImageToVideoPipeline.from_pretrained(model_id, **load_kwargs)
        self._pipe.enable_model_cpu_offload()
        
        # Memory-efficient attention optimizations
        if hasattr(self._pipe, "enable_vae_slicing"):
            self._pipe.enable_vae_slicing()
        if hasattr(self._pipe, "enable_vae_tiling"):
            self._pipe.enable_vae_tiling()
        if hasattr(self._pipe, "enable_attention_slicing"):
            self._pipe.enable_attention_slicing()
        if hasattr(self._pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def _load_ltx_video(self, model_id: str, settings) -> None:
        """Load LTX Video pipeline with optimizations."""
        LTXPipeline = self._resolve_pipeline_class(
            "LTXPipeline",
            [
                "diffusers.pipelines.ltx.pipeline_ltx",
                "diffusers.pipelines.ltx",
            ],
        )

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "cache_dir": settings.cache_dir,
        }
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        self._pipe = LTXPipeline.from_pretrained(model_id, **load_kwargs)
        self._pipe.to(self.device)
        
        # Memory-efficient attention optimizations
        if hasattr(self._pipe, "enable_vae_slicing"):
            self._pipe.enable_vae_slicing()
        if hasattr(self._pipe, "enable_vae_tiling"):
            self._pipe.enable_vae_tiling()
        if hasattr(self._pipe, "enable_attention_slicing"):
            self._pipe.enable_attention_slicing()
        if hasattr(self._pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    def unload(self) -> None:
        """Unload current video model and free VRAM."""
        if self._pipe is not None:
            logger.info(f"Unloading video model: {self._current_model}")
            del self._pipe
            self._pipe = None
            self._current_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _apply_lora(self, lora_name: Optional[str], lora_scale: float) -> None:
        """LoRA support for video pipelines — currently skipped (diffusers support varies)."""
        # TODO: Implement when we confirm Wan/LTX support load_lora_weights
        pass

    def _unload_lora(self) -> None:
        """LoRA cleanup — currently no-op."""
        pass

    async def generate_text_to_video(
        self,
        prompt: str,
        model_name: str = "wan-t2v-1.3b",
        negative_prompt: str = "",
        resolution: str = "480p",
        num_frames: int = 33,
        fps: int = 16,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        lora_name: Optional[str] = None,
        lora_scale: float = 1.0,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a video from a text prompt.

        Returns dict with video_url, duration_seconds, inference_time_ms.
        """
        self.load_model(model_name)
        self._apply_lora(lora_name, lora_scale)

        height, width = RESOLUTION_MAP.get(resolution, (480, 848))
        # LTX-Video requires dimensions divisible by 32
        height = (height // 32) * 32
        width = (width // 32) * 32

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        logger.info(
            f"Generating T2V: {width}x{height}, {num_frames} frames, "
            f"model={model_name}, steps={num_inference_steps}"
        )

        start = time.perf_counter()

        with torch.no_grad():
            output = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                height=height,
                width=width,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        frames = output.frames[0]  # List of PIL images

        # Save as MP4
        video_path = self._save_video(frames, fps, job_id)

        # Generate thumbnail from first frame
        thumbnail_b64 = self._frame_to_b64(frames[0] if frames else None)

        self._unload_lora()
        torch.cuda.empty_cache()

        duration_seconds = len(frames) / fps
        logger.info(
            f"Video generated: {len(frames)} frames, {duration_seconds:.1f}s, "
            f"inference={elapsed_ms:.0f}ms"
        )

        return {
            "video_url": output_store.get_url(video_path),
            "thumbnail_b64": thumbnail_b64,
            "duration_seconds": round(duration_seconds, 2),
            "inference_time_ms": round(elapsed_ms, 0),
            "seed_used": seed,
            "num_frames": len(frames),
        }

    async def generate_image_to_video(
        self,
        source_image_b64: str,
        prompt: str = "",
        model_name: str = "wan-i2v-14b",
        num_frames: int = 33,
        fps: int = 16,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        lora_name: Optional[str] = None,
        lora_scale: float = 1.0,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a video from a source image + optional prompt.

        Returns dict with video_url, duration_seconds, inference_time_ms.
        """
        # Decode source image
        img_bytes = base64.b64decode(source_image_b64)
        source_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        self.load_model(model_name)
        self._apply_lora(lora_name, lora_scale)

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        logger.info(f"Generating I2V: {num_frames} frames, model={model_name}")

        start = time.perf_counter()

        with torch.no_grad():
            output = self._pipe(
                image=source_image,
                prompt=prompt or "",
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        frames = output.frames[0]

        video_path = self._save_video(frames, fps, job_id)
        thumbnail_b64 = self._frame_to_b64(frames[0] if frames else None)

        self._unload_lora()
        torch.cuda.empty_cache()

        duration_seconds = len(frames) / fps
        logger.info(
            f"I2V generated: {len(frames)} frames, {duration_seconds:.1f}s, "
            f"inference={elapsed_ms:.0f}ms"
        )

        return {
            "video_url": output_store.get_url(video_path),
            "thumbnail_b64": thumbnail_b64,
            "duration_seconds": round(duration_seconds, 2),
            "inference_time_ms": round(elapsed_ms, 0),
            "seed_used": seed,
            "num_frames": len(frames),
        }

    def _save_video(self, frames: list, fps: int, job_id: Optional[str]) -> str:
        """Encode frames as MP4 and save to output store."""
        import imageio

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        # Convert PIL images to numpy arrays for imageio
        import numpy as np
        np_frames = [np.array(frame) for frame in frames]

        writer = imageio.get_writer(tmp_path, fps=fps, codec="libx264", quality=8)
        for frame in np_frames:
            writer.append_data(frame)
        writer.close()

        # Move to output store
        rel_path = output_store.save_file_from_path(tmp_path, "video", job_id)
        return rel_path

    def _frame_to_b64(self, frame) -> Optional[str]:
        """Convert a PIL frame to base64 JPEG thumbnail."""
        if frame is None:
            return None
        thumb = frame.copy()
        thumb.thumbnail((320, 180))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=75)
        return base64.b64encode(buf.getvalue()).decode()

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None


# Module-level singleton
video_pipeline = VideoPipeline()
