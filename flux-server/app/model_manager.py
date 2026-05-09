"""
Universal Model Manager for image and video generation.
Handles lazy-loading, VRAM management, and model switching.

Multi-GPU deployment (Vultr 8× A100 80GB):
  Each gunicorn worker sees exactly one GPU via CUDA_VISIBLE_DEVICES set in
  gunicorn's post_fork hook.  This manager therefore always addresses device 0
  (the worker's assigned GPU).  gpu_info() aggregates all GPUs visible to the
  process for diagnostic purposes.

NF4 quantization is applied adaptively: on a 40 GB A100 it is required for WAN
14B and SD3.5-Large; on an 80 GB A100 it is skipped for image models that fit
comfortably in BF16 (SD3.5-Large ~25 GB BF16 vs 80 GB available).  WAN 14B
still requires NF4 on a single 80 GB GPU because BF16 inference peaks at ~78 GB.
"""

import gc
import torch
import logging
from enum import Enum
from typing import Optional, Dict, List, Any
from pathlib import Path
from diffusers import (
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
)
from transformers import BitsAndBytesConfig as HFBitsAndBytesConfig
from app.config import get_settings
from app.video_defaults import get_video_model_defaults

logger = logging.getLogger(__name__)

# Set global CUDA performance flags at import time so they are active for every
# model loaded in this process, including the first one.
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("Global CUDA flags: FlashSDP=True, TF32=True")


class ModelCategory(str, Enum):
    """Categories of AI generation models."""
    IMAGE = "image"
    VIDEO = "video"


class OutputType(str, Enum):
    """Output types for generation results."""
    IMAGE_B64 = "image_b64"       # Base64-encoded PNG
    VIDEO_FILE = "video_file"     # MP4 file path


class ModelConfig:
    """Configuration for a specific model across all modalities."""

    def __init__(
        self,
        model_id: str,
        pipeline_class: Any = None,
        category: ModelCategory = ModelCategory.IMAGE,
        output_type: OutputType = OutputType.IMAGE_B64,
        quantize: bool = False,
        quantize_type: str = "nf4",
        # GPU VRAM (in GB) above which BF16 is safe for this model without quantization.
        # Set to 0.0 to always quantize when quantize=True (e.g. WAN 14B on single GPU).
        bf16_min_vram_gb: float = 0.0,
        transformer_file: str | None = None,
        variant: str | None = None,
        vram_free_gb: float = 1.0,
        description: str = "",
        # Inference parameters (primarily for image/video diffusion)
        min_steps: int = 1,
        max_steps: int = 50,
        default_steps: int = 28,
        default_guidance_scale: float = 3.5,
        # Pipeline module path for non-diffusers models
        pipeline_module: str | None = None,
        # Extra kwargs passed during model loading
        extra_load_kwargs: dict | None = None,
        # Video request defaults surfaced to the frontend
        default_resolution: str | None = None,
        default_num_frames: int | None = None,
        default_fps: int | None = None,
        default_chunk_size: int | None = None,
        default_chunk_overlap: int | None = None,
        preferred_backend: str | None = None,
    ):
        self.model_id = model_id
        self.pipeline_class = pipeline_class
        self.category = category
        self.output_type = output_type
        self.quantize = quantize
        self.quantize_type = quantize_type
        self.bf16_min_vram_gb = bf16_min_vram_gb
        self.transformer_file = transformer_file
        self.variant = variant
        self.vram_free_gb = vram_free_gb
        self.description = description
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.default_steps = default_steps
        self.default_guidance_scale = default_guidance_scale
        self.pipeline_module = pipeline_module
        self.extra_load_kwargs = extra_load_kwargs or {}
        self.default_resolution = default_resolution
        self.default_num_frames = default_num_frames
        self.default_fps = default_fps
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self.preferred_backend = preferred_backend


class MultiModelManager:
    """
    Manages multiple diffusion models with lazy-loading and memory optimization.
    
    Features:
    - Load models on-demand
    - Switch between models (2-3 sec latency)
    - Monitor VRAM usage
    - Unload unused models (LRU cache)
    - Fallback model support
    """
    
    # Available models configuration
    MODELS = {
        # ═══════════════════════════════════════════════
        #  IMAGE MODELS (existing — unchanged)
        # ═══════════════════════════════════════════════
        "flux-1-dev": ModelConfig(
            model_id="black-forest-labs/FLUX.1-dev",
            pipeline_class=FluxPipeline,
            category=ModelCategory.IMAGE,
            output_type=OutputType.IMAGE_B64,
            quantize=False,
            vram_free_gb=34.0,
            description="FLUX.1-dev: High quality BF16 path for A100 40GB testing",
            min_steps=4,
            max_steps=50,
            default_steps=28,
            default_guidance_scale=3.5,
        ),
        "sd3.5-large": ModelConfig(
            model_id="stabilityai/stable-diffusion-3.5-large",
            pipeline_class=StableDiffusion3Pipeline,
            category=ModelCategory.IMAGE,
            output_type=OutputType.IMAGE_B64,
            quantize=True,
            quantize_type="nf4",
            # BF16 footprint ~25 GB — skip NF4 on GPUs with ≥50 GB VRAM (e.g. A100 80 GB).
            bf16_min_vram_gb=50.0,
            vram_free_gb=18.0,
            description="SD3.5-Large: Multi-modal, flexible, top-tier quality",
            min_steps=20,
            max_steps=50,
            default_steps=28,
            default_guidance_scale=4.5,
        ),
        "realvisxl-v5": ModelConfig(
            model_id="SG161222/RealVisXL_V5.0",
            pipeline_class=StableDiffusionXLPipeline,
            category=ModelCategory.IMAGE,
            output_type=OutputType.IMAGE_B64,
            quantize=False,
            variant="fp16",
            vram_free_gb=16.0,
            description="RealVisXL V5: photorealistic SDXL model",
            min_steps=20,
            max_steps=50,
            default_steps=30,
            default_guidance_scale=7.0,
        ),
        "juggernaut-xl": ModelConfig(
            model_id="RunDiffusion/Juggernaut-XL-v9",
            pipeline_class=DiffusionPipeline,
            category=ModelCategory.IMAGE,
            output_type=OutputType.IMAGE_B64,
            quantize=False,
            variant="fp16",
            vram_free_gb=16.0,
            description="Juggernaut XL: versatile SDXL model",
            min_steps=20,
            max_steps=50,
            default_steps=30,
            default_guidance_scale=6.5,
        ),

        # ═══════════════════════════════════════════════
        #  VIDEO MODELS (Text-to-Video & Image-to-Video)
        # ═══════════════════════════════════════════════
        "wan-t2v-1.3b": ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            pipeline_class=None,  # Loaded via video_pipeline module
            category=ModelCategory.VIDEO,
            output_type=OutputType.VIDEO_FILE,
            pipeline_module="app.pipelines.video_pipeline",
            quantize=False,
            vram_free_gb=10.0,
            description="Wan 2.1 T2V 1.3B: fast text-to-video, lightweight model (Wan 2.1)",
            min_steps=20,
            max_steps=50,
            default_steps=get_video_model_defaults("wan-t2v-1.3b")["default_steps"],
            default_guidance_scale=get_video_model_defaults("wan-t2v-1.3b")["default_guidance_scale"],
            default_resolution=get_video_model_defaults("wan-t2v-1.3b")["default_resolution"],
            default_num_frames=get_video_model_defaults("wan-t2v-1.3b")["default_num_frames"],
            default_fps=get_video_model_defaults("wan-t2v-1.3b")["default_fps"],
            default_chunk_size=get_video_model_defaults("wan-t2v-1.3b")["default_chunk_size"],
            default_chunk_overlap=get_video_model_defaults("wan-t2v-1.3b")["default_chunk_overlap"],
            preferred_backend=get_video_model_defaults("wan-t2v-1.3b")["preferred_backend"],
        ),
        "wan-t2v-14b": ModelConfig(
            model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            pipeline_class=None,
            category=ModelCategory.VIDEO,
            output_type=OutputType.VIDEO_FILE,
            pipeline_module="app.pipelines.video_pipeline",
            quantize=True,
            quantize_type="nf4",
            # Dual-transformer NF4: ~7 GB each + UMT5 text encoder ~10.5 GB + VAE ~1 GB = ~25.5 GB
            # Peak during load (one shard in BF16 at a time): ~30 GB. Need 30 GB free to start.
            vram_free_gb=30.0,
            description="Wan 2.2 T2V 14B: high-fidelity cinematic video, prefers BF16 xDiT on 4x80GB and falls back to NF4",
            min_steps=20,
            max_steps=60,
            default_steps=get_video_model_defaults("wan-t2v-14b")["default_steps"],
            default_guidance_scale=get_video_model_defaults("wan-t2v-14b")["default_guidance_scale"],
            default_resolution=get_video_model_defaults("wan-t2v-14b")["default_resolution"],
            default_num_frames=get_video_model_defaults("wan-t2v-14b")["default_num_frames"],
            default_fps=get_video_model_defaults("wan-t2v-14b")["default_fps"],
            default_chunk_size=get_video_model_defaults("wan-t2v-14b")["default_chunk_size"],
            default_chunk_overlap=get_video_model_defaults("wan-t2v-14b")["default_chunk_overlap"],
            preferred_backend=get_video_model_defaults("wan-t2v-14b")["preferred_backend"],
        ),
        "wan-i2v-14b": ModelConfig(
            model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            pipeline_class=None,
            category=ModelCategory.VIDEO,
            output_type=OutputType.VIDEO_FILE,
            pipeline_module="app.pipelines.video_pipeline",
            quantize=True,
            quantize_type="nf4",
            # Same dual-transformer architecture as T2V 14B — same VRAM profile.
            vram_free_gb=30.0,
            description="Wan 2.2 I2V 14B: image-to-video animation, NF4, ~25.5 GB VRAM",
            min_steps=20,
            max_steps=60,
            default_steps=get_video_model_defaults("wan-i2v-14b")["default_steps"],
            default_guidance_scale=get_video_model_defaults("wan-i2v-14b")["default_guidance_scale"],
            default_resolution=get_video_model_defaults("wan-i2v-14b")["default_resolution"],
            default_num_frames=get_video_model_defaults("wan-i2v-14b")["default_num_frames"],
            default_fps=get_video_model_defaults("wan-i2v-14b")["default_fps"],
            default_chunk_size=get_video_model_defaults("wan-i2v-14b")["default_chunk_size"],
            default_chunk_overlap=get_video_model_defaults("wan-i2v-14b")["default_chunk_overlap"],
            preferred_backend=get_video_model_defaults("wan-i2v-14b")["preferred_backend"],
        ),

        "hunyuan-video": ModelConfig(
            model_id="hunyuanvideo-community/HunyuanVideo",
            pipeline_class=None,
            category=ModelCategory.VIDEO,
            output_type=OutputType.VIDEO_FILE,
            pipeline_module="app.pipelines.video_pipeline",
            quantize=True,
            quantize_type="nf4",
            # NF4 transformer ~8 GB + VAE ~1 GB = ~9 GB GPU steady state.
            # LLaMA-3-8B text encoder (~16 GB BF16) kept on CPU via enable_model_cpu_offload;
            # moves to GPU only during the text-encoding step, then back to CPU.
            vram_free_gb=10.0,
            description="HunyuanVideo: 720p text-to-video, NF4 transformer, ~9 GB VRAM (CPU-offload text encoder)",
            min_steps=20,
            max_steps=100,
            default_steps=get_video_model_defaults("hunyuan-video")["default_steps"],
            default_guidance_scale=get_video_model_defaults("hunyuan-video")["default_guidance_scale"],
            default_resolution=get_video_model_defaults("hunyuan-video")["default_resolution"],
            default_num_frames=get_video_model_defaults("hunyuan-video")["default_num_frames"],
            default_fps=get_video_model_defaults("hunyuan-video")["default_fps"],
            default_chunk_size=get_video_model_defaults("hunyuan-video")["default_chunk_size"],
            default_chunk_overlap=get_video_model_defaults("hunyuan-video")["default_chunk_overlap"],
            preferred_backend=get_video_model_defaults("hunyuan-video")["preferred_backend"],
        ),

    }
    
    def __init__(self, default_model: str = "flux-1-dev"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipelines: Dict[str, DiffusionPipeline] = {}
        self.current_model = default_model
        self.lru_cache: List[str] = []  # Track load order for unloading
        # One model per worker. In multi-GPU mode each gunicorn worker owns one GPU
        # (set via CUDA_VISIBLE_DEVICES in post_fork) so max_loaded=1 is correct.
        self.max_loaded = 1

        logger.info(
            f"MultiModelManager initialized (device={self.device}, "
            f"vram={self._get_vram_gb():.0f}GB)"
        )
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a model."""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        return self.MODELS[model_name]

    # Single source of truth for model → cache directory mapping.
    # SSD tier: fast-path for priority models with moderate disk footprint.
    # HDD tier: fallback for large models (WAN 14B = 118GB) or low-priority ones.
    SSD_PRIORITY = {"flux-1-dev", "wan-t2v-1.3b", "hunyuan-video"}

    @classmethod
    def get_cache_dir(cls, model_name: str) -> str:
        """Return the appropriate cache directory for a model based on disk tier."""
        settings = get_settings()
        if model_name in cls.SSD_PRIORITY:
            return settings.cache_dir_ssd
        return settings.cache_dir
    
    @staticmethod
    def _get_vram_gb() -> float:
        """Total VRAM in GB for the first visible CUDA device, or 0 if no GPU."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    def _should_quantize(self, config: ModelConfig) -> bool:
        """Return True if NF4 quantization should be applied for this model.

        Decision logic:
        - If config.quantize is False → never quantize.
        - If bf16_min_vram_gb is set and GPU VRAM meets the threshold → skip NF4,
          load in full BF16 (better quality, faster inference, no dequant overhead).
        - Otherwise → apply NF4 (required on smaller GPUs).

        WAN 14B BF16 peaks at ~78 GB on a single GPU — bf16_min_vram_gb is left at
        the default 0.0, so NF4 is always used for those models.
        """
        if not config.quantize:
            return False
        if config.bf16_min_vram_gb > 0.0:
            vram_gb = self._get_vram_gb()
            if vram_gb >= config.bf16_min_vram_gb:
                logger.info(
                    f"GPU VRAM {vram_gb:.0f}GB ≥ {config.bf16_min_vram_gb:.0f}GB threshold "
                    f"— loading in BF16 (skipping NF4)"
                )
                return False
        return True

    def list_models(self, category: ModelCategory | None = None) -> Dict[str, str]:
        """List available models with descriptions, optionally filtered by category."""
        return {
            name: config.description
            for name, config in self.MODELS.items()
            if category is None or config.category == category
        }

    def get_models_by_category(self, category: ModelCategory) -> Dict[str, "ModelConfig"]:
        """Get all model configs for a specific category."""
        return {
            name: config
            for name, config in self.MODELS.items()
            if config.category == category
        }

    def get_categories(self) -> List[str]:
        """Get all unique model categories."""
        return sorted(set(config.category.value for config in self.MODELS.values()))
    
    def gpu_info(self) -> Dict:
        """Get GPU memory info for the primary (worker-assigned) device.

        In multi-GPU gunicorn mode each worker has CUDA_VISIBLE_DEVICES set to a
        single GPU, so device 0 here is always that worker's exclusive GPU.
        The 'all_gpus' key lists every GPU visible to the process (useful for the
        health endpoint and diagnostics).
        """
        if not torch.cuda.is_available():
            return {
                "name": "CPU",
                "total_gb": 0,
                "used_gb": 0,
                "free_gb": 0,
                "device": "cpu",
                "gpu_count": 0,
            }

        # Primary device (index 0 = worker-assigned GPU)
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024 ** 3)
        used_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free_memory = total_memory - used_memory

        # Aggregate stats across all visible devices (diagnostic)
        n = torch.cuda.device_count()
        all_gpus = []
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            all_gpus.append({
                "index": i,
                "name": p.name,
                "total_gb": round(p.total_memory / (1024 ** 3), 1),
                "used_gb": round(torch.cuda.memory_allocated(i) / (1024 ** 3), 1),
            })

        return {
            "name": props.name,
            "total_gb": round(total_memory, 2),
            "used_gb": round(used_memory, 2),
            "free_gb": round(free_memory, 2),
            "device": self.device,
            "gpu_count": n,
            "all_gpus": all_gpus,
        }
    
    def is_loaded(self) -> bool:
        """Check if current model is loaded."""
        return self.current_model in self.pipelines
    
    def _unload_model(self, model_name: str) -> None:
        """Safely unload a model and release all GPU memory."""
        if model_name not in self.pipelines:
            return
        logger.info(f"Unloading {model_name} to free VRAM...")
        pipe = self.pipelines.pop(model_name)
        del pipe
        if model_name in self.lru_cache:
            self.lru_cache.remove(model_name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gpu = self.gpu_info()
        logger.info(f"Unloaded {model_name}. GPU: {gpu['used_gb']:.1f}GB / {gpu['total_gb']:.1f}GB")
    
    def load(self, model_name: str) -> None:
        """
        Load a model into GPU memory.
        Unloads all other models first to maximize free VRAM.
        """
        settings = get_settings()

        # Apply offline mode — eliminates HF network metadata round-trips on every load.
        if settings.hf_offline:
            import os as _os
            _os.environ.setdefault("HF_HUB_OFFLINE", "1")
            _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if model_name in self.pipelines:
            logger.info(f"Model {model_name} already loaded, skipping load")
            self.current_model = model_name
            return

        config = self.get_model_config(model_name)

        # Unload ALL other models to maximize free VRAM
        for old_model in list(self.pipelines.keys()):
            if old_model != model_name:
                self._unload_model(old_model)

        gpu_info = self.gpu_info()
        logger.info(f"Loading {model_name} (needs ~{config.vram_free_gb}GB free)...")
        logger.info(f"GPU: {gpu_info['name']} | Free: {gpu_info['free_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB")

        # Pre-load VRAM check — fail fast before committing to a long load that will OOM.
        if gpu_info['total_gb'] > 0 and gpu_info['free_gb'] < config.vram_free_gb:
            raise RuntimeError(
                f"Insufficient VRAM to load {model_name}: "
                f"need {config.vram_free_gb:.1f}GB free, "
                f"only {gpu_info['free_gb']:.1f}GB available. "
                f"Call unload_all() first or reduce concurrent model usage."
            )

        cache_dir = self.get_cache_dir(model_name)

        try:
            # Pick the right HF token for this model
            token = settings.hf_token
            if model_name in {"sd3-medium", "sd3.5-large"} and settings.sd3_hf_token:
                token = settings.sd3_hf_token

            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "cache_dir": cache_dir,
            }
            if token:
                load_kwargs["token"] = token
            if config.variant:
                load_kwargs["variant"] = config.variant
            
            use_nf4 = self._should_quantize(config)

            # FLUX models with quantization: load transformer separately
            if use_nf4 and config.pipeline_class == FluxPipeline:
                from diffusers import FluxTransformer2DModel
                nf4_config = HFBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info(f"Loading {model_name} transformer with NF4 quantization...")
                transformer = FluxTransformer2DModel.from_pretrained(
                    config.model_id,
                    subfolder="transformer",
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir,
                    token=token if token else None,
                )
                load_kwargs["transformer"] = transformer
                pipe = FluxPipeline.from_pretrained(config.model_id, **load_kwargs)
                # NF4-quantized transformer is already on CUDA via bitsandbytes.
                # pipe.to() is invalid on quantized pipelines — move non-quantized components only.
                for attr in ("vae", "text_encoder", "text_encoder_2"):
                    component = getattr(pipe, attr, None)
                    if component is not None and hasattr(component, "to"):
                        component.to(self.device)
            elif use_nf4 and config.pipeline_class == StableDiffusion3Pipeline:
                # SD3.5-Large NF4 path: only used on GPUs with <50 GB VRAM (e.g. A100 40 GB).
                # On 80 GB A100 _should_quantize() returns False → falls through to BF16 path.
                from diffusers import SD3Transformer2DModel
                nf4_config = HFBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info(f"Loading {model_name} transformer with NF4 quantization...")
                transformer = SD3Transformer2DModel.from_pretrained(
                    config.model_id,
                    subfolder="transformer",
                    quantization_config=nf4_config,
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir,
                    token=token if token else None,
                )
                load_kwargs["transformer"] = transformer
                pipe = StableDiffusion3Pipeline.from_pretrained(config.model_id, **load_kwargs)
                # NF4-quantized transformer is already on CUDA via bitsandbytes.
                # pipe.to() is invalid on quantized pipelines — move non-quantized components only.
                for attr in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
                    component = getattr(pipe, attr, None)
                    if component is not None and hasattr(component, "to"):
                        component.to(self.device)
            else:
                # Standard BF16 loading — used for all unquantized models and for
                # quantize=True models on high-VRAM GPUs (e.g. SD3.5 on 80 GB A100).
                logger.info(f"Loading {model_name} in BF16 from {config.model_id}...")
                pipe = config.pipeline_class.from_pretrained(config.model_id, **load_kwargs)
                logger.info(f"Moving {model_name} to {self.device}...")
                pipe.to(self.device)
            
            # Enable PyTorch 2.x FlashAttention-2 / SDP backends
            if torch.cuda.is_available():
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            # Enable memory optimizations for image models
            # Note: attention_slicing is intentionally omitted — FlashSDP on A100 handles
            # memory management natively and attention_slicing disables its batching.
            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
            if hasattr(pipe, 'enable_vae_tiling'):
                pipe.enable_vae_tiling()
            if hasattr(pipe, 'set_progress_bar_config'):
                pipe.set_progress_bar_config(disable=True)

            # Store and track
            self.pipelines[model_name] = pipe
            self.lru_cache.append(model_name)
            self.current_model = model_name

            gpu_info = self.gpu_info()
            vram_used = gpu_info['total_gb'] - gpu_info['free_gb']
            logger.info(f"✅ {model_name} loaded. Actual VRAM: {vram_used:.1f}GB used / {gpu_info['total_gb']:.1f}GB total ({gpu_info['free_gb']:.1f}GB free)")
            
        except Exception as e:
            logger.exception(f"Failed to load {model_name}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            err_text = str(e)
            if "gated" in err_text.lower() or "403" in err_text:
                raise RuntimeError(
                    f"Model '{model_name}' requires gated Hugging Face access. "
                    "Request access on its model page and set HF_TOKEN in environment."
                ) from e
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e
    
    def get_pipeline(self, model_name: str = None) -> DiffusionPipeline:
        """Get pipeline for a model, loading if necessary."""
        if model_name is None:
            model_name = self.current_model
        
        if model_name not in self.pipelines:
            self.load(model_name)
        
        return self.pipelines[model_name]
    
    def switch_model(self, model_name: str) -> None:
        """Switch to a different model."""
        if model_name == self.current_model and self.is_loaded():
            return
        
        logger.info(f"Switching model: {self.current_model} → {model_name}")
        self.load(model_name)
    
    def unload_all(self) -> None:
        """Unload all models and free VRAM."""
        for model_name in list(self.pipelines.keys()):
            self._unload_model(model_name)
        self.lru_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gpu = self.gpu_info()
        logger.info(f"All models unloaded. GPU: {gpu['used_gb']:.1f}GB / {gpu['total_gb']:.1f}GB")
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get detailed info about a model."""
        config = self.get_model_config(model_name)
        return {
            "name": model_name,
            "model_id": config.model_id,
            "category": config.category.value,
            "output_type": config.output_type.value,
            "description": config.description,
            "vram_needed_gb": config.vram_free_gb,
            "min_steps": config.min_steps,
            "max_steps": config.max_steps,
            "default_steps": config.default_steps,
            "default_guidance_scale": config.default_guidance_scale,
            "default_resolution": config.default_resolution,
            "default_num_frames": config.default_num_frames,
            "default_fps": config.default_fps,
            "default_chunk_size": config.default_chunk_size,
            "default_chunk_overlap": config.default_chunk_overlap,
            "preferred_backend": config.preferred_backend,
            "loaded": model_name in self.pipelines,
        }
