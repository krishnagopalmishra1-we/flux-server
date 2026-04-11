"""
Video generation pipeline — Text-to-Video and Image-to-Video.

Supports:
- Wan 2.2 T2V (1.3B and 14B variants)
- Wan 2.2 I2V (Image-to-Video, 14B)
- HunyuanVideo (720p NF4-quantized)

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
    # Reserve 5% VRAM for CUDA overhead to reduce fragmentation-related OOMs.
    torch.cuda.set_per_process_memory_fraction(0.95)

RESOLUTION_MAP = {
    "480p":  (480, 848),
    "720p":  (720, 1280),
    "540p":  (544, 960),   # HunyuanVideo native 9:16
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
        # Do NOT call enable_attention_slicing — it disables FlashSDP batching.
        # PyTorch 2.x scaled_dot_product_attention handles memory efficiently on A100.
        if hasattr(p, "set_progress_bar_config"):
            p.set_progress_bar_config(disable=True)

    def _try_compile_transformer(self) -> None:
        """Apply torch.compile to the transformer on non-quantized pipelines.

        Requires a C compiler (gcc/cc) for the inductor backend. Silently skips
        if no compiler is available (common in minimal containers).
        """
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
            logger.info("  Skipping torch.compile — quantized transformer")
            return
        # torch.compile inductor backend requires gcc/cc. Skip gracefully if absent.
        import shutil
        if shutil.which("gcc") is None and shutil.which("cc") is None:
            logger.info("  Skipping torch.compile — no C compiler found (gcc/cc)")
            return
        try:
            logger.info("  Compiling transformer with torch.compile(reduce-overhead)...")
            self._pipe.transformer = torch.compile(
                transformer,
                mode="reduce-overhead",
                fullgraph=False,
            )
            self._compiled = True
            logger.info("  torch.compile applied to transformer")
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

    def load_model(self, model_name: str, progress_callback: Optional[Callable[[float], None]] = None) -> None:
        if self._current_model == model_name and self._pipe is not None:
            return

        import threading
        stop_event = threading.Event()
        _keepalive_pct = [2.0]  # mutable counter for incremental progress
        def keepalive():
            while not stop_event.wait(5.0):  # 5s interval (was 30s)
                if progress_callback:
                    progress_callback(_keepalive_pct[0])
                    _keepalive_pct[0] = min(_keepalive_pct[0] + 0.5, 9.0)  # increment up to 9%

        keepalive_thread = threading.Thread(target=keepalive)
        keepalive_thread.daemon = True
        keepalive_thread.start()

        try:
            self.unload()
            settings = get_settings()

            # Apply offline mode to skip HF network metadata round-trips.
            if settings.hf_offline:
                import os as _os
                _os.environ.setdefault("HF_HUB_OFFLINE", "1")
                _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            # Use model_manager as single source of truth for cache tier selection.
            from app.model_manager import MultiModelManager
            cache_dir = MultiModelManager().get_cache_dir(model_name)

            logger.info(f"Loading video model: {model_name} (cache={cache_dir})")
            try:
                if model_name == "wan-t2v-1.3b":
                    self._load_wan_t2v("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", quantize=False, settings=settings, cache_dir=cache_dir)
                elif model_name == "wan-t2v-14b":
                    # NF4 double-quantization: transformer ~7-8 GB vs ~28 GB BF16.
                    # Required on A100 40GB — BF16 14B OOMs at 49 frames (38.76 GB).
                    self._load_wan_t2v("Wan-AI/Wan2.2-T2V-A14B-Diffusers", quantize=True, settings=settings, cache_dir=cache_dir)
                elif model_name == "wan-i2v-14b":
                    self._load_wan_i2v("Wan-AI/Wan2.2-I2V-A14B-Diffusers", settings=settings, cache_dir=cache_dir)
                elif model_name == "hunyuan-video":
                    self._load_hunyuan_video(settings=settings, cache_dir=cache_dir)
                else:
                    raise ValueError(f"Unknown video model: {model_name}")
                self._current_model = model_name
                gpu = self._gpu_mb()
                logger.info(f"✅ {model_name} loaded | VRAM used: {gpu:.0f} MB")
            except Exception as e:
                logger.exception(f"Failed to load video model {model_name}: {e}")
                self.unload()
                raise
        finally:
            stop_event.set()
            keepalive_thread.join(timeout=1.0)

    def _load_wan_t2v(self, model_id: str, quantize: bool, settings, cache_dir: str) -> None:
        WanPipeline = self._resolve_pipeline_class(
            "WanPipeline",
            ["diffusers.pipelines.wan.pipeline_wan", "diffusers.pipelines.wan"],
        )
        load_kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": cache_dir}
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        if quantize:
            # NF4 double-quantization: ~7-8 GB vs ~28 GB BF16 — required for 14B on A100 40GB.
            # All non-transformer components go to GPU in BF16 for full tensor-core speed.
            logger.info(f"  Loading {model_id} transformer with NF4 double-quantization...")
            nf4 = self._make_nf4_config()
            try:
                from diffusers import WanTransformer3DModel
            except ImportError:
                from diffusers.models import WanTransformer3DModel
            transformer = WanTransformer3DModel.from_pretrained(
                model_id, subfolder="transformer",
                quantization_config=nf4, torch_dtype=torch.bfloat16,
                cache_dir=cache_dir,
                **({"token": settings.hf_token} if settings.hf_token else {}),
            )
            logger.info("  Transformer: NF4 quantized (~7 GB)")
            self._pipe = WanPipeline.from_pretrained(model_id, transformer=transformer, **load_kwargs)
            for name, component in self._pipe.components.items():
                if name != "transformer" and hasattr(component, "to"):
                    component.to(self.device)
            logger.info("  All non-transformer components on GPU")
        else:
            # 1.3B model: full BF16, fits easily in 40GB. torch.compile applicable.
            logger.info(f"  Loading {model_id} in BF16...")
            self._pipe = WanPipeline.from_pretrained(model_id, **load_kwargs)
            self._pipe.to(self.device)

        self._apply_memory_opts()
        if not quantize:
            self._try_compile_transformer()

    def _load_wan_i2v(self, model_id: str, settings, cache_dir: str) -> None:
        WanI2V = self._resolve_pipeline_class(
            "WanImageToVideoPipeline",
            ["diffusers.pipelines.wan.pipeline_wan_i2v", "diffusers.pipelines.wan"],
        )
        load_kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": cache_dir}
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        # NF4 double-quantization: same VRAM constraint as T2V 14B.
        logger.info(f"  Loading {model_id} transformer with NF4 double-quantization...")
        nf4 = self._make_nf4_config()
        try:
            from diffusers import WanTransformer3DModel
        except ImportError:
            from diffusers.models import WanTransformer3DModel
        transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer",
            quantization_config=nf4, torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            **({"token": settings.hf_token} if settings.hf_token else {}),
        )
        self._pipe = WanI2V.from_pretrained(model_id, transformer=transformer, **load_kwargs)
        for name, component in self._pipe.components.items():
            if name != "transformer" and hasattr(component, "to"):
                component.to(self.device)
        logger.info("  I2V NF4 quantized, all non-transformer components on GPU")
        self._apply_memory_opts()



    def _load_hunyuan_video(self, settings, cache_dir: str) -> None:
        """Load HunyuanVideo with NF4-quantized transformer (fits A100 40GB).

        Transformer: ~8 GB NF4 | VAE + text encoders: ~4 GB BF16 | Total: ~12 GB
        Allows 720p 129-frame generation (~36 GB peak including activations).
        """
        model_id = "hunyuanvideo-community/HunyuanVideo"
        HunyuanVideoPipeline = self._resolve_pipeline_class(
            "HunyuanVideoPipeline",
            ["diffusers.pipelines.hunyuan_video", "diffusers"],
        )
        try:
            from diffusers import HunyuanVideoTransformer3DModel
        except ImportError:
            try:
                from diffusers.models import HunyuanVideoTransformer3DModel
            except ImportError:
                raise ImportError(
                    "HunyuanVideoTransformer3DModel not found. "
                    "Upgrade diffusers: pip install diffusers>=0.32"
                )

        load_kwargs = {"torch_dtype": torch.bfloat16, "cache_dir": cache_dir}
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        logger.info("  Loading HunyuanVideo transformer with NF4 double-quantization...")
        nf4 = self._make_nf4_config()
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer",
            quantization_config=nf4, torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            **({"token": settings.hf_token} if settings.hf_token else {}),
        )
        logger.info("  Transformer: NF4 quantized (~8 GB)")
        self._pipe = HunyuanVideoPipeline.from_pretrained(
            model_id, transformer=transformer, **load_kwargs
        )
        for name, component in self._pipe.components.items():
            if name != "transformer" and hasattr(component, "to"):
                component.to(self.device)
        logger.info("  All non-transformer components on GPU")
        self._apply_memory_opts()
        logger.info("✅ HunyuanVideo loaded")

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
    #  Chunked Long-Video Generation
    # ──────────────────────────────────────────────

    async def generate_long_video(
        self,
        prompt: str,
        model_name: str = "wan-t2v-1.3b",
        negative_prompt: str = "",
        resolution: str = "480p",
        total_frames: int = 240,
        chunk_size: int = 81,
        chunk_overlap: int = 20,
        fps: int = 16,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
        lora_name: Optional[str] = None,
        lora_scale: float = 1.0,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        job=None,
    ) -> Dict[str, Any]:
        """Generate a long video by chunking into overlapping segments.

        Chunks are generated sequentially, each with `chunk_overlap` frames
        overlapping the previous chunk. Overlapping regions are blended
        using cosine weights for smooth temporal transitions.
        """
        if progress_callback:
            progress_callback(1.0)
        self.load_model(model_name, progress_callback)
        self._apply_lora(lora_name, lora_scale)

        height, width = RESOLUTION_MAP.get(resolution, (480, 848))
        height = (height // 32) * 32
        width  = (width  // 32) * 32

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        step = chunk_size - chunk_overlap
        num_chunks = max(1, (total_frames - chunk_overlap + step - 1) // step)

        logger.info(
            f"Long video: {total_frames} frames in {num_chunks} chunks "
            f"(chunk_size={chunk_size}, overlap={chunk_overlap}), model={model_name}"
        )

        if job is not None:
            job.inference_start_time = time.time()

        all_frames: list = []
        start_time = time.perf_counter()

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * step
            chunk_end = min(chunk_start + chunk_size, total_frames)
            chunk_frames = chunk_end - chunk_start

            # Per-chunk progress: map each chunk into its portion of 10-90%
            chunk_pct_start = 10.0 + (chunk_idx / num_chunks) * 80.0
            chunk_pct_end = 10.0 + ((chunk_idx + 1) / num_chunks) * 80.0
            _step_cb = _make_step_callback(
                num_inference_steps, progress_callback,
                start=chunk_pct_start, end=chunk_pct_end, job=job,
            )

            logger.info(f"  Chunk {chunk_idx + 1}/{num_chunks}: frames {chunk_start}-{chunk_end} ({chunk_frames} frames)")

            try:
                with torch.inference_mode():
                    output = self._pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt or None,
                        height=height, width=width,
                        num_frames=chunk_frames,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        callback_on_step_end=_step_cb,
                    )
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"OOM on chunk {chunk_idx + 1} — enabling CPU offload and retrying...")
                torch.cuda.empty_cache()
                gc.collect()
                if hasattr(self._pipe, "enable_sequential_cpu_offload"):
                    self._pipe.enable_sequential_cpu_offload()
                with torch.inference_mode():
                    output = self._pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt or None,
                        height=height, width=width,
                        num_frames=chunk_frames,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        callback_on_step_end=_step_cb,
                    )

            raw_frames = output.frames[0]
            chunk_frame_list = list(raw_frames) if hasattr(raw_frames, "__iter__") else []

            # Blend overlap region with previous chunk
            if all_frames and chunk_overlap > 0 and len(chunk_frame_list) > chunk_overlap:
                blended = _blend_overlap(all_frames[-chunk_overlap:], chunk_frame_list[:chunk_overlap])
                all_frames = all_frames[:-chunk_overlap] + blended + chunk_frame_list[chunk_overlap:]
            else:
                all_frames.extend(chunk_frame_list)

            torch.cuda.empty_cache()

        # Trim to exact requested length
        all_frames = all_frames[:total_frames]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if progress_callback:
            progress_callback(92.0)

        video_path = self._save_video(all_frames, fps, job_id)
        thumbnail_b64 = self._frame_to_b64(all_frames[0] if all_frames else None)

        self._unload_lora()
        torch.cuda.empty_cache()

        num_frames_out = len(all_frames)
        duration_seconds = num_frames_out / fps
        logger.info(
            f"Long video complete: {num_frames_out} frames, {duration_seconds:.1f}s, "
            f"{num_chunks} chunks, {elapsed_ms:.0f}ms"
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
            "chunks_generated": num_chunks,
        }

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
        job=None,
    ) -> Dict[str, Any]:
        # Emit 1% immediately so the UI shows activity even during model load/download
        if progress_callback:
            progress_callback(1.0)
        self.load_model(model_name, progress_callback)
        self._apply_lora(lora_name, lora_scale)

        height, width = RESOLUTION_MAP.get(resolution, (480, 848))
        height = (height // 32) * 32
        width  = (width  // 32) * 32

        generator = torch.Generator(device=self.device)
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

        # Mark inference start for ETA calculation
        if job is not None:
            job.inference_start_time = time.time()

        # Build diffusers step callback that feeds into job progress
        _step_cb = _make_step_callback(num_inference_steps, progress_callback, start=10, end=90, job=job)

        start = time.perf_counter()
        try:
            with torch.inference_mode():
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
            # Tiered OOM recovery: try reducing parameters before falling back to CPU offload.
            torch.cuda.empty_cache()
            gc.collect()
            output = None

            # Tier 1: reduce inference steps
            if num_inference_steps > 20:
                reduced_steps = max(20, num_inference_steps - 10)
                logger.warning(f"OOM during T2V — retrying with reduced steps ({num_inference_steps} → {reduced_steps})...")
                try:
                    with torch.inference_mode():
                        output = self._pipe(
                            prompt=prompt, negative_prompt=negative_prompt or None,
                            height=height, width=width, num_frames=num_frames,
                            guidance_scale=guidance_scale, num_inference_steps=reduced_steps,
                            generator=generator, callback_on_step_end=_step_cb,
                        )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()

            # Tier 2: reduce frame count
            if output is None and num_frames > 16:
                reduced_frames = max(16, num_frames // 2)
                logger.warning(f"OOM during T2V — retrying with reduced frames ({num_frames} → {reduced_frames})...")
                try:
                    with torch.inference_mode():
                        output = self._pipe(
                            prompt=prompt, negative_prompt=negative_prompt or None,
                            height=height, width=width, num_frames=reduced_frames,
                            guidance_scale=guidance_scale, num_inference_steps=max(20, num_inference_steps - 10),
                            generator=generator, callback_on_step_end=_step_cb,
                        )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()

            # Tier 3: last resort — CPU offload
            if output is None:
                logger.warning("OOM during T2V — enabling sequential CPU offload (last resort)...")
                if hasattr(self._pipe, "enable_sequential_cpu_offload"):
                    self._pipe.enable_sequential_cpu_offload()
                try:
                    with torch.inference_mode():
                        output = self._pipe(
                            prompt=prompt, negative_prompt=negative_prompt or None,
                            height=height, width=width, num_frames=num_frames,
                            guidance_scale=guidance_scale, num_inference_steps=num_inference_steps,
                            generator=generator, callback_on_step_end=_step_cb,
                        )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_err:
                    logger.error(f"T2V OOM all retries failed: {retry_err}. Unloading pipeline.")
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
        job=None,
    ) -> Dict[str, Any]:
        img_bytes = base64.b64decode(source_image_b64)
        source_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Emit 1% immediately so the UI shows activity even during model load/download
        if progress_callback:
            progress_callback(1.0)
        self.load_model(model_name, progress_callback)
        self._apply_lora(lora_name, lora_scale)

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        if progress_callback:
            progress_callback(5.0)

        # Mark inference start for ETA calculation
        if job is not None:
            job.inference_start_time = time.time()

        _step_cb = _make_step_callback(num_inference_steps, progress_callback, start=10, end=90, job=job)

        logger.info(f"I2V: {num_frames} frames, model={model_name}, steps={num_inference_steps}")

        start = time.perf_counter()

        try:
            with torch.inference_mode():
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
                with torch.inference_mode():
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
    #  Inference — HunyuanVideo
    # ──────────────────────────────────────────────

    async def generate_hunyuan_video(
        self,
        prompt: str,
        resolution: str = "720p",
        num_frames: int = 129,
        fps: int = 24,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
        job=None,
    ) -> Dict[str, Any]:
        if progress_callback:
            progress_callback(1.0)
        self.load_model("hunyuan-video", progress_callback)

        # HunyuanVideo uses (height, width) — native 9:16 at 720p is 720×1280
        height, width = RESOLUTION_MAP.get(resolution, (720, 1280))
        height = (height // 32) * 32
        width  = (width  // 32) * 32

        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        if progress_callback:
            progress_callback(5.0)

        # Mark inference start for ETA calculation
        if job is not None:
            job.inference_start_time = time.time()

        _step_cb = _make_step_callback(num_inference_steps, progress_callback, start=10, end=90, job=job)

        logger.info(
            f"HunyuanVideo: {width}x{height}, {num_frames} frames, steps={num_inference_steps}"
        )

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                output = self._pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    callback_on_step_end=_step_cb,
                )
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during HunyuanVideo — enabling sequential CPU offload and retrying...")
            torch.cuda.empty_cache()
            gc.collect()
            if hasattr(self._pipe, "enable_sequential_cpu_offload"):
                self._pipe.enable_sequential_cpu_offload()
            try:
                with torch.inference_mode():
                    output = self._pipe(
                        prompt=prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=generator,
                        callback_on_step_end=_step_cb,
                    )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_err:
                logger.error(f"HunyuanVideo OOM retry failed: {retry_err}. Resetting CUDA state.")
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
        torch.cuda.empty_cache()

        num_frames_out = len(frames)
        duration_seconds = num_frames_out / fps
        logger.info(f"HunyuanVideo complete: {num_frames_out} frames, {elapsed_ms:.0f}ms")
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
    job=None,
) -> Optional[Callable]:
    if progress_callback is None or total_steps <= 0:
        return None

    def _cb(pipeline, step_index: int, timestep, callback_kwargs: dict):
        # Check cancellation flag — raises to abort inference loop.
        if job is not None and getattr(job, 'cancel_flag', False):
            raise InterruptedError("Job cancelled by user request")
        pct = start + (step_index / total_steps) * (end - start)
        try:
            progress_callback(round(pct, 1))
        except Exception:
            pass
        return callback_kwargs

    return _cb


def _blend_overlap(tail_frames: list, head_frames: list) -> list:
    """Blend overlapping frames from two chunks using cosine weights.

    Args:
        tail_frames: Last N frames from the previous chunk.
        head_frames: First N frames from the current chunk.

    Returns:
        Blended frames (same length as inputs).
    """
    import numpy as np
    from PIL import Image as _PILImage

    n = min(len(tail_frames), len(head_frames))
    blended = []
    for i in range(n):
        # Cosine weight: 1.0 at start (all tail) → 0.0 at end (all head)
        w = 0.5 * (1.0 + np.cos(np.pi * i / max(n - 1, 1)))

        a = tail_frames[i]
        b = head_frames[i]

        # Convert to numpy arrays
        arr_a = np.array(a).astype(np.float32)
        arr_b = np.array(b).astype(np.float32)

        # Weighted blend
        arr_out = (w * arr_a + (1.0 - w) * arr_b).clip(0, 255).astype(np.uint8)

        # Return as PIL Image if inputs were PIL
        if isinstance(a, _PILImage.Image):
            blended.append(_PILImage.fromarray(arr_out))
        else:
            blended.append(arr_out)

    return blended


# Module-level singleton
video_pipeline = VideoPipeline()
