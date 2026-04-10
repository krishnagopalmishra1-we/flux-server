"""
Video generation pipeline — Text-to-Video and Image-to-Video.

Supports:
- Wan 2.2 T2V (1.3B and 14B variants)
- Wan 2.2 I2V (Image-to-Video, 14B)
- LTX Video (fast drafts)

Optimization stack (applied in order):
1. NF4 double quantization for 14B transformer (bitsandbytes)
2. FlashAttention-2 via PyTorch 2.x scaled_dot_product_attention (no xFormers dep)
3. torch.compile(mode="reduce-overhead") on non-quantized models
4. TF32 global math acceleration
5. VAE slicing + tiling (decodes incrementally, avoids OOM on long videos)
6. LoRA hot-swap for WAN and LTX models
7. OOM auto-fallback to sequential_cpu_offload (preserves output quality)
"""

from __future__ import annotations

import gc
import io
import time
import base64
import logging
import tempfile
import importlib
from pathlib import Path
from typing import Optional, Dict, Any, Callable

import torch
from PIL import Image

from app.config import get_settings
from app.output_store import output_store

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════
#  GLOBAL CUDA ACCELERATION
# ═══════════════════════════════════════════════════
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable PyTorch 2.x native FlashAttention-2 / memory-efficient SDP kernels
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

RESOLUTION_MAP = {
    "480p":  (480, 848),
    "720p":  (720, 1280),
}

VIDEO_LORA_DIR = Path("video_loras")


class VideoPipeline:
    """
    Manages video generation for Wan 2.2 and LTX Video.

    One video model lives in VRAM at a time. LoRA adapters are hot-swappable
    without reloading the base model.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipe = None
        self._current_model: Optional[str] = None
        self._compiled = False
        self._loaded_lora: Optional[str] = None

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def _resolve_pipeline_class(self, class_name: str, module_candidates: list[str]):
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
            f"{class_name} not available. Upgrade diffusers to a Wan/LTX-capable version."
        )

    def _apply_memory_opts(self) -> None:
        """Apply all safe memory optimizations to self._pipe."""
        p = self._pipe
        if p is None:
            return
        if hasattr(p, "enable_vae_slicing"):
            p.enable_vae_slicing()
        if hasattr(p, "enable_vae_tiling"):
            p.enable_vae_tiling()
        if hasattr(p, "enable_attention_slicing"):
            p.enable_attention_slicing(1)
        # Attempt xFormers only when PyTorch 2.x FlashAttention-2 SDP is not active.
        if not torch.backends.cuda.flash_sdp_enabled():
            if hasattr(p, "enable_xformers_memory_efficient_attention"):
                try:
                    p.enable_xformers_memory_efficient_attention()
                    logger.info("  ✓ xFormers fallback attention enabled")
                except Exception:
                    pass

    def _try_compile_transformer(self) -> None:
        """Apply torch.compile to the transformer on non-quantized pipelines."""
        if self._compiled:
            return
        if self._pipe is None:
            return
        transformer = getattr(self._pipe, "transformer", None)
        if transformer is None:
            return
        # Do NOT compile quantized (BitsAndBytes) models — they use custom kernels
        # that are incompatible with torch.compile graph capture.
        if hasattr(transformer, "is_quantized") and transformer.is_quantized:
            logger.info("  ⚠ Skipping torch.compile — quantized transformer")
            return
        try:
            logger.info("  Compiling transformer with torch.compile(reduce-overhead)...")
            self._pipe.transformer = torch.compile(
                transformer,
                mode="reduce-overhead",
                fullgraph=False,
            )
            self._compiled = True
            logger.info("  ✓ torch.compile applied to transformer")
        except Exception as e:
            logger.warning(f"  torch.compile skipped: {e}")

    def _make_nf4_config(self):
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            raise ImportError(
                "transformers is required for NF4 quantization. "
                "Install it with: pip install transformers>=4.40"
            )
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # ──────────────────────────────────────────────
    #  Model Loading
    # ──────────────────────────────────────────────

    def load_model(self, model_name: str) -> None:
        if self._current_model == model_name and self._pipe is not None:
            return
        self.unload()
        settings = get_settings()
        logger.info(f"Loading video model: {model_name}")
        try:
            if model_name == "wan-t2v-1.3b":
                self._load_wan_t2v("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", quantize=False, settings=settings)
            elif model_name == "wan-t2v-14b":
                self._load_wan_t2v("Wan-AI/Wan2.2-T2V-A14B-Diffusers", quantize=True, settings=settings)
            elif model_name == "wan-i2v-14b":
                self._load_wan_i2v("Wan-AI/Wan2.2-I2V-A14B-Diffusers", settings=settings)
            elif model_name == "ltx-video":
                self._load_ltx_video("Lightricks/LTX-Video", settings=settings)
            else:
                raise ValueError(f"Unknown video model: {model_name}")
            self._current_model = model_name
            gpu = self._gpu_mb()
            logger.info(f"✅ {model_name} loaded | VRAM used: {gpu:.0f} MB")
        except Exception as e:
            logger.exception(f"Failed to load video model {model_name}: {e}")
            self.unload()
            raise

    def _load_wan_t2v(self, model_id: str, quantize: bool, settings) -> None:
        WanPipeline = self._resolve_pipeline_class(
            "WanPipeline",
            ["diffusers.pipelines.wan.pipeline_wan", "diffusers.pipelines.wan"],
        )
        load_kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": settings.cache_dir}
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        if quantize:
            logger.info(f"  Loading {model_id} transformer with NF4 double-quantization...")
            nf4 = self._make_nf4_config()
            try:
                from diffusers import WanTransformer3DModel
            except ImportError:
                from diffusers.models import WanTransformer3DModel
            transformer = WanTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer",
                quantization_config=nf4, torch_dtype=torch.bfloat16,
                cache_dir=settings.cache_dir,
                **({"token": settings.hf_token} if settings.hf_token else {}),
            )
            logger.info("  ✓ Transformer: NF4 quantized (~7 GB)")
            self._pipe = WanPipeline.from_pretrained(model_id, transformer=transformer, **load_kwargs)
            for name, component in self._pipe.components.items():
                if name != "transformer" and hasattr(component, "to"):
                    component.to(self.device)
            logger.info("  ✓ All non-transformer components on GPU")
        else:
            self._pipe = WanPipeline.from_pretrained(model_id, **load_kwargs)
            self._pipe.to(self.device)

        self._apply_memory_opts()
        if not quantize:
            self._try_compile_transformer()

    def _load_wan_i2v(self, model_id: str, settings) -> None:
        WanI2V = self._resolve_pipeline_class(
            "WanImageToVideoPipeline",
            ["diffusers.pipelines.wan.pipeline_wan_i2v", "diffusers.pipelines.wan"],
        )
        load_kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": settings.cache_dir}
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        logger.info(f"  Loading {model_id} transformer with NF4 double-quantization...")
        nf4 = self._make_nf4_config()
        try:
            from diffusers import WanTransformer3DModel
        except ImportError:
            from diffusers.models import WanTransformer3DModel
        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer",
            quantization_config=nf4, torch_dtype=torch.bfloat16,
            cache_dir=settings.cache_dir,
            **({"token": settings.hf_token} if settings.hf_token else {}),
        )
        self._pipe = WanI2V.from_pretrained(model_id, transformer=transformer, **load_kwargs)
        for name, component in self._pipe.components.items():
            if name != "transformer" and hasattr(component, "to"):
                component.to(self.device)
        self._apply_memory_opts()

    def _load_ltx_video(self, model_id: str, settings) -> None:
        LTXPipeline = self._resolve_pipeline_class(
            "LTXPipeline",
            ["diffusers.pipelines.ltx.pipeline_ltx", "diffusers.pipelines.ltx"],
        )
        load_kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": settings.cache_dir}
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token
        self._pipe = LTXPipeline.from_pretrained(model_id, **load_kwargs)
        self._pipe.to(self.device)
        self._apply_memory_opts()
        self._try_compile_transformer()

    # ──────────────────────────────────────────────
    #  LoRA Support
    # ──────────────────────────────────────────────

    def get_available_loras(self) -> list[str]:
        VIDEO_LORA_DIR.mkdir(exist_ok=True)
        return [f.name for f in VIDEO_LORA_DIR.glob("*.safetensors")]

    def _apply_lora(self, lora_name: Optional[str], lora_scale: float) -> None:
        if not lora_name or lora_name == "None":
            self._unload_lora()
            return
        if self._pipe is None:
            return
        if lora_name == self._loaded_lora:
            if hasattr(self._pipe, "set_adapters"):
                adapter = Path(lora_name).stem.replace(".", "_")
                scale = max(0.1, min(1.5, lora_scale))
                self._pipe.set_adapters(adapter, adapter_weights=[scale])
            return

        lora_path = VIDEO_LORA_DIR / lora_name
        if not lora_path.exists():
            logger.warning(f"Video LoRA not found: {lora_path} — skipping")
            return

        # Unload previous adapter first
        self._unload_lora()

        logger.info(f"Loading video LoRA: {lora_name}")
        try:
            adapter = Path(lora_name).stem.replace(".", "_")
            self._pipe.load_lora_weights(str(lora_path), adapter_name=adapter)
            if hasattr(self._pipe, "set_adapters"):
                scale = max(0.1, min(1.5, lora_scale))
                self._pipe.set_adapters(adapter, adapter_weights=[scale])
            self._loaded_lora = lora_name
            logger.info(f"  ✓ Video LoRA {lora_name!r} loaded (scale={lora_scale})")
        except Exception as e:
            logger.warning(f"  Video LoRA load failed: {e} — continuing without LoRA")
            self._loaded_lora = None

    def _unload_lora(self) -> None:
        if self._loaded_lora is None or self._pipe is None:
            return
        try:
            if hasattr(self._pipe, "unload_lora_weights"):
                self._pipe.unload_lora_weights()
            elif hasattr(self._pipe, "delete_adapters"):
                adapter = Path(self._loaded_lora).stem.replace(".", "_")
                self._pipe.delete_adapters(adapter)
            logger.info(f"  ✓ Video LoRA {self._loaded_lora!r} unloaded")
        except Exception as e:
            logger.warning(f"  Video LoRA unload warning: {e}")
        finally:
            self._loaded_lora = None

    # ──────────────────────────────────────────────
    #  Unload
    # ──────────────────────────────────────────────

    def unload(self) -> None:
        if self._pipe is None:
            return
        logger.info(f"Unloading video model: {self._current_model}")
        self._loaded_lora = None
        del self._pipe
        self._pipe = None
        self._current_model = None
        self._compiled = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # ──────────────────────────────────────────────
    #  Inference — Text to Video
    # ──────────────────────────────────────────────

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
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        # Emit 1% immediately so the UI shows activity even during model load/download
        if progress_callback:
            progress_callback(1.0)
        self.load_model(model_name)
        self._apply_lora(lora_name, lora_scale)

        height, width = RESOLUTION_MAP.get(resolution, (480, 848))
        height = (height // 32) * 32
        width  = (width  // 32) * 32

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        logger.info(
            f"T2V: {width}x{height}, {num_frames} frames, "
            f"model={model_name}, steps={num_inference_steps}"
        )
        if progress_callback:
            progress_callback(5.0)

        # Build diffusers step callback that feeds into job progress
        _step_cb = _make_step_callback(num_inference_steps, progress_callback, start=10, end=90)

        # For long videos (>49 frames), offload the VAE decoder to CPU to free ~2GB VRAM
        # during denoising — decode happens at the end so latency cost is minimal.
        if num_frames > 49 and hasattr(self._pipe, "vae") and hasattr(self._pipe.vae, "to"):
            self._pipe.vae.to("cpu")
            logger.info(f"T2V: VAE offloaded to CPU for {num_frames}-frame video")

        start = time.perf_counter()
        try:
            with torch.no_grad():
                output = self._pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt or None,
                    height=height, width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback_on_step_end=_step_cb,
                )
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during T2V — enabling sequential CPU offload and retrying...")
            torch.cuda.empty_cache()
            gc.collect()
            if hasattr(self._pipe, "enable_sequential_cpu_offload"):
                self._pipe.enable_sequential_cpu_offload()
            try:
                with torch.no_grad():
                    output = self._pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt or None,
                        height=height, width=width,
                        num_frames=num_frames,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        callback_on_step_end=_step_cb,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_err:
                # Retry also failed — CUDA context may be poisoned. Unload to reset state
                # so the next job starts clean (avoids "device-side assert triggered" cascade).
                logger.error(f"T2V OOM retry also failed: {retry_err}. Unloading pipeline to reset CUDA state.")
                self.unload()
                torch.cuda.empty_cache()
                gc.collect()
                raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        if progress_callback:
            progress_callback(92.0)

        raw_frames = output.frames[0]
        frames = list(raw_frames) if hasattr(raw_frames, "__iter__") else []

        video_path = self._save_video(frames, fps, job_id)
        thumbnail_b64 = self._frame_to_b64(frames[0] if frames else None)

        self._unload_lora()
        torch.cuda.empty_cache()

        num_frames_out = len(frames)
        duration_seconds = num_frames_out / fps
        logger.info(
            f"T2V complete: {num_frames_out} frames, {duration_seconds:.1f}s, "
            f"{elapsed_ms:.0f}ms"
        )
        if progress_callback:
            progress_callback(100.0)

        return {
            "video_url": output_store.get_url(video_path),
            "thumbnail_b64": thumbnail_b64,
            "duration_seconds": round(duration_seconds, 2),
            "inference_time_ms": round(elapsed_ms, 0),
            "seed_used": seed,
            "num_frames": num_frames_out,
        }

    # ──────────────────────────────────────────────
    #  Inference — Image to Video
    # ──────────────────────────────────────────────

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
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Dict[str, Any]:
        img_bytes = base64.b64decode(source_image_b64)
        source_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Emit 1% immediately so the UI shows activity even during model load/download
        if progress_callback:
            progress_callback(1.0)
        self.load_model(model_name)
        self._apply_lora(lora_name, lora_scale)

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        if progress_callback:
            progress_callback(5.0)
        _step_cb = _make_step_callback(num_inference_steps, progress_callback, start=10, end=90)

        logger.info(f"I2V: {num_frames} frames, model={model_name}, steps={num_inference_steps}")

        # For long videos (>49 frames), offload the VAE decoder to CPU to free ~2GB VRAM
        if num_frames > 49 and hasattr(self._pipe, "vae") and hasattr(self._pipe.vae, "to"):
            self._pipe.vae.to("cpu")
            logger.info(f"I2V: VAE offloaded to CPU for {num_frames}-frame video")

        start = time.perf_counter()

        try:
            with torch.no_grad():
                output = self._pipe(
                    image=source_image,
                    prompt=prompt or "",
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback_on_step_end=_step_cb,
                )
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during I2V — enabling sequential CPU offload and retrying...")
            torch.cuda.empty_cache()
            gc.collect()
            if hasattr(self._pipe, "enable_sequential_cpu_offload"):
                self._pipe.enable_sequential_cpu_offload()
            try:
                with torch.no_grad():
                    output = self._pipe(
                        image=source_image,
                        prompt=prompt or "",
                        num_frames=num_frames,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        callback_on_step_end=_step_cb,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_err:
                logger.error(f"I2V OOM retry also failed: {retry_err}. Unloading pipeline to reset CUDA state.")
                self.unload()
                torch.cuda.empty_cache()
                gc.collect()
                raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        if progress_callback:
            progress_callback(92.0)

        raw_frames = output.frames[0]
        frames = list(raw_frames) if hasattr(raw_frames, "__iter__") else []

        video_path = self._save_video(frames, fps, job_id)
        thumbnail_b64 = self._frame_to_b64(frames[0] if frames else None)

        self._unload_lora()
        torch.cuda.empty_cache()

        num_frames_out = len(frames)
        duration_seconds = num_frames_out / fps
        if progress_callback:
            progress_callback(100.0)

        return {
            "video_url": output_store.get_url(video_path),
            "thumbnail_b64": thumbnail_b64,
            "duration_seconds": round(duration_seconds, 2),
            "inference_time_ms": round(elapsed_ms, 0),
            "seed_used": seed,
            "num_frames": num_frames_out,
        }

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    def _save_video(self, frames: list, fps: int, job_id: Optional[str]) -> str:
        import imageio
        import numpy as np
        from PIL import Image as _PILImage

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        np_frames = []
        for frame in frames:
            if isinstance(frame, _PILImage.Image):
                np_frames.append(np.array(frame))
            else:
                arr = np.array(frame)
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                np_frames.append(arr)

        writer = imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            output_params=["-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"],
        )
        for frame in np_frames:
            writer.append_data(frame)
        writer.close()

        return output_store.save_file_from_path(tmp_path, "video", job_id)

    def _frame_to_b64(self, frame) -> Optional[str]:
        if frame is None:
            return None
        import numpy as np
        from PIL import Image as _PILImage
        if not isinstance(frame, _PILImage.Image):
            arr = np.array(frame)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            frame = _PILImage.fromarray(arr)
        thumb = frame.copy()
        thumb.thumbnail((480, 270))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=82)
        return base64.b64encode(buf.getvalue()).decode()

    def _gpu_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None


# ──────────────────────────────────────────────────────
#  Step callback factory — converts denoising steps to
#  job progress percentage for real-time SSE streaming.
# ──────────────────────────────────────────────────────

def _make_step_callback(
    total_steps: int,
    progress_callback: Optional[Callable[[float], None]],
    start: float = 10.0,
    end: float = 90.0,
) -> Optional[Callable]:
    if progress_callback is None or total_steps <= 0:
        return None

    def _cb(pipeline, step_index: int, timestep, callback_kwargs: dict):
        pct = start + (step_index / total_steps) * (end - start)
        try:
            progress_callback(round(pct, 1))
        except Exception:
            pass
        return callback_kwargs

    return _cb


# Module-level singleton
video_pipeline = VideoPipeline()
