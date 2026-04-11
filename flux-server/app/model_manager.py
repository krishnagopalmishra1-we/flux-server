"""
Universal Model Manager for image and video generation.
Handles lazy-loading, VRAM management, and model switching on a single GPU.
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
    ):
        self.model_id = model_id
        self.pipeline_class = pipeline_class
        self.category = category
        self.output_type = output_type
        self.quantize = quantize
        self.quantize_type = quantize_type
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
            quantize=True,
            vram_free_gb=20.0,
            description="FLUX.1-dev: High quality, 4-28 steps, best results",
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
            default_steps=30,
            default_guidance_scale=5.0,
        ),
        "wan-t2v-14b": ModelConfig(
            model_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            pipeline_class=None,
            category=ModelCategory.VIDEO,
            output_type=OutputType.VIDEO_FILE,
            pipeline_module="app.pipelines.video_pipeline",
            quantize=True,
            quantize_type="fp8",
            vram_free_gb=35.0,
            description="Wan 2.2 T2V 14B: SOTA cinematic video, maximum quality",
            min_steps=20,
            max_steps=50,
            default_steps=30,
            default_guidance_scale=5.0,
        ),
        "wan-i2v-14b": ModelConfig(
            model_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            pipeline_class=None,
            category=ModelCategory.VIDEO,
            output_type=OutputType.VIDEO_FILE,
            pipeline_module="app.pipelines.video_pipeline",
            quantize=True,
            quantize_type="fp8",
            vram_free_gb=30.0,
            description="Wan 2.2 I2V 14B: Image-to-video animation, 480P",
            min_steps=20,
            max_steps=50,
            default_steps=30,
            default_guidance_scale=5.0,
        ),
        "ltx-video": ModelConfig(
            model_id="Lightricks/LTX-Video",
            pipeline_class=None,
            category=ModelCategory.VIDEO,
            output_type=OutputType.VIDEO_FILE,
            pipeline_module="app.pipelines.video_pipeline",
            quantize=False,
            vram_free_gb=12.0,
            description="LTX Video: Ultra-fast text-to-video drafts",
            min_steps=20,
            max_steps=50,
            default_steps=30,
            default_guidance_scale=3.0,
        ),
        "hunyuan-video": ModelConfig(
            model_id="hunyuanvideo-community/HunyuanVideo",
            pipeline_class=None,
            category=ModelCategory.VIDEO,
            output_type=OutputType.VIDEO_FILE,
            pipeline_module="app.pipelines.video_pipeline",
            quantize=True,
            quantize_type="nf4",
            vram_free_gb=36.0,
            description="HunyuanVideo: SOTA 720p text-to-video, NF4 quantized transformer",
            min_steps=20,
            max_steps=100,
            default_steps=50,
            default_guidance_scale=6.0,
        ),

    }
    
    def __init__(self, default_model: str = "flux-1-dev"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipelines: Dict[str, DiffusionPipeline] = {}
        self.current_model = default_model
        self.lru_cache: List[str] = []  # Track load order for unloading
        self.max_loaded = 1  # Only 1 model in VRAM at a time (40GB A100)
        
        logger.info(f"MultiModelManager initialized (device={self.device})")
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a model."""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        return self.MODELS[model_name]
    
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
        """Get GPU memory info."""
        if not torch.cuda.is_available():
            return {
                "name": "CPU",
                "total_gb": 0,
                "used_gb": 0,
                "free_gb": 0,
                "device": "cpu",
            }
        
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024**3)
        used_memory = torch.cuda.memory_allocated() / (1024**3)
        free_memory = total_memory - used_memory
        
        return {
            "name": props.name,
            "total_gb": round(total_memory, 2),
            "used_gb": round(used_memory, 2),
            "free_gb": round(free_memory, 2),
            "device": self.device,
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
        logger.info(f"Loading {model_name} (needs ~{config.vram_free_gb}GB)...")
        logger.info(f"GPU: {gpu_info['name']} | Free: {gpu_info['free_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB")

        # SSD-tier: 250GB SSD, 128GB free. Models that fit:
        #   FLUX.1-dev = 32GB, HunyuanVideo NF4 = ~20GB, WAN 1.3B = 27GB → ~79GB total.
        # WAN 14B = 118GB on disk → HDD (too large).
        SSD_PRIORITY = {"flux-1-dev", "hunyuan-video", "wan-t2v-1.3b"}
        cache_dir = (
            settings.cache_dir_ssd
            if model_name in SSD_PRIORITY
            else settings.cache_dir
        )

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
            
            # FLUX local transformer checkpoint
            if config.pipeline_class == FluxPipeline and config.transformer_file:
                from diffusers import FluxTransformer2DModel
                transformer_path = Path(config.transformer_file)
                if not transformer_path.exists():
                    raise RuntimeError(f"Local transformer file not found: {transformer_path}")
                logger.info(f"Loading local FLUX transformer from {transformer_path}...")
                transformer = FluxTransformer2DModel.from_single_file(
                    str(transformer_path),
                    torch_dtype=torch.bfloat16,
                )
                load_kwargs["transformer"] = transformer
                pipe = FluxPipeline.from_pretrained(config.model_id, **load_kwargs)
                logger.info(f"Moving {model_name} to {self.device}...")
                pipe.to(self.device)

            # FLUX models with quantization: load transformer separately
            elif config.quantize and config.pipeline_class == FluxPipeline:
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
                pipe = FluxPipeline.from_pretrained(config.model_id, device_map="balanced", **load_kwargs)
            elif config.quantize and config.pipeline_class == StableDiffusion3Pipeline:
                # SD3.5-Large NF4 path: quantize transformer while keeping pipeline API unchanged.
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
                pipe = StableDiffusion3Pipeline.from_pretrained(config.model_id, device_map="balanced", **load_kwargs)
            else:
                # Standard loading (no quantization)
                logger.info(f"Loading {model_name} from {config.model_id}...")
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
            logger.info(f"✅ {model_name} loaded successfully")
            logger.info(f"GPU: Used: {gpu_info['used_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB")
            
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
            "loaded": model_name in self.pipelines,
        }
