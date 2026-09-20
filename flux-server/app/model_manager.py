"""
Image Model Manager — image generation only.
Handles lazy-loading, VRAM management, and model switching.

GPU target: AWS g5.2xlarge (1 × NVIDIA A10G, 24 GB VRAM).

Quantization strategy for FLUX.1-dev:
  - Transformer: FP8 via torchao (float8_weight_only) → ~12 GB, ~1.8× faster than NF4.
  - T5-XXL text encoder: INT8 via bitsandbytes → ~4.7 GB (vs 9.5 GB in BF16).
  - CLIP text encoder + VAE: BF16 → ~1.5 GB.
  - Total: ~18 GB — fits on A10G 24 GB with 6 GB headroom.

  Override quantization at runtime with env var FLUX_QUANTIZE=fp8|nf4|bf16.
"""

import gc
import os as _os
import logging
import torch
from enum import Enum
from typing import Dict, List, Any, Optional

from diffusers import (
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionXLPipeline,
    DiffusionPipeline,
)
from transformers import BitsAndBytesConfig as HFBitsAndBytesConfig

from app.config import get_settings

logger = logging.getLogger(__name__)

# Enable PyTorch 2.x performance flags globally at import time.
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("Global CUDA flags: FlashSDP=True, TF32=True")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ModelCategory(str, Enum):
    IMAGE = "image"


class OutputType(str, Enum):
    IMAGE_B64 = "image_b64"


class ModelConfig:
    """Configuration for a single image-generation model."""

    def __init__(
        self,
        model_id: str,
        pipeline_class: Any,
        quantize: bool = False,
        quantize_type: str = "fp8",   # fp8 | nf4 | bf16
        bf16_min_vram_gb: float = 0.0,
        variant: Optional[str] = None,
        vram_free_gb: float = 1.0,
        description: str = "",
        min_steps: int = 1,
        max_steps: int = 50,
        default_steps: int = 28,
        default_guidance_scale: float = 3.5,
    ):
        self.model_id = model_id
        self.pipeline_class = pipeline_class
        self.category = ModelCategory.IMAGE
        self.output_type = OutputType.IMAGE_B64
        self.quantize = quantize
        self.quantize_type = quantize_type
        self.bf16_min_vram_gb = bf16_min_vram_gb
        self.variant = variant
        self.vram_free_gb = vram_free_gb
        self.description = description
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.default_steps = default_steps
        self.default_guidance_scale = default_guidance_scale


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

class MultiModelManager:
    """
    Lazy-loading manager for image diffusion models.

    - One model loaded at a time (max_loaded=1) to maximise free VRAM.
    - FP8 (torchao) is the default quantization for FLUX — ~1.8× faster
      inference vs NF4 on A10G's INT8 tensor cores, near-BF16 quality.
    - Runtime override: set FLUX_QUANTIZE=fp8|nf4|bf16 env var.
    """

    MODELS: Dict[str, ModelConfig] = {
        # ── FLUX.1-dev ────────────────────────────────────────────────────────
        # FP8 transformer (~12 GB) + T5-XXL INT8 (~4.7 GB) + CLIP/VAE (~1.5 GB)
        # = ~18 GB total. Fits on A10G 24 GB with 6 GB headroom.
        "flux-1-dev": ModelConfig(
            model_id="black-forest-labs/FLUX.1-dev",
            pipeline_class=FluxPipeline,
            quantize=True,
            quantize_type="fp8",
            bf16_min_vram_gb=0.0,  # A10G 24 GB < 35 GB needed for BF16 — always quantize
            vram_free_gb=20.0,
            description="FLUX.1-dev (Uncensored) — FP8 (torchao), ~18 GB VRAM, ~1.8× faster than NF4",
            min_steps=4,
            max_steps=50,
            default_steps=28,
            default_guidance_scale=3.5,
        ),

        # ── SD 3.5 Large ──────────────────────────────────────────────────────
        # NF4 on <50 GB GPU; BF16 on ≥50 GB GPU (e.g. A100 80 GB).
        "sd3.5-large": ModelConfig(
            model_id="stabilityai/stable-diffusion-3.5-large",
            pipeline_class=StableDiffusion3Pipeline,
            quantize=True,
            quantize_type="nf4",
            bf16_min_vram_gb=50.0,
            vram_free_gb=18.0,
            description="SD 3.5 Large — NF4, ~18 GB VRAM",
            min_steps=20,
            max_steps=50,
            default_steps=28,
            default_guidance_scale=4.5,
        ),

        # ── RealVisXL V5 ──────────────────────────────────────────────────────
        # SDXL-based photorealistic model. FP16. ~8 GB VRAM.
        "realvisxl-v5": ModelConfig(
            model_id="SG161222/RealVisXL_V5.0",
            pipeline_class=StableDiffusionXLPipeline,
            quantize=False,
            variant="fp16",
            vram_free_gb=10.0,
            description="RealVisXL V5 — FP16, ~8 GB VRAM, photorealistic SDXL",
            min_steps=20,
            max_steps=50,
            default_steps=30,
            default_guidance_scale=7.0,
        ),

        # ── Juggernaut XL v9 ─────────────────────────────────────────────────
        # SDXL-based versatile model. FP16. ~8 GB VRAM.
        "juggernaut-xl": ModelConfig(
            model_id="RunDiffusion/Juggernaut-XL-v9",
            pipeline_class=DiffusionPipeline,
            quantize=False,
            variant="fp16",
            vram_free_gb=10.0,
            description="Juggernaut XL v9 — FP16, ~8 GB VRAM, versatile SDXL",
            min_steps=20,
            max_steps=50,
            default_steps=30,
            default_guidance_scale=6.5,
        ),
    }

    # Models stored on the fast EBS/SSD path (all image models qualify).
    SSD_PRIORITY = {"flux-1-dev", "sd3.5-large", "realvisxl-v5", "juggernaut-xl"}

    def __init__(self, default_model: str = "flux-1-dev"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipelines: Dict[str, Any] = {}
        self.current_model = default_model
        self.lru_cache: List[str] = []
        self.max_loaded = 1
        logger.info(
            f"ImageModelManager initialized (device={self.device}, "
            f"vram={self._get_vram_gb():.0f} GB)"
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _get_vram_gb() -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    def gpu_info(self) -> Dict:
        if not torch.cuda.is_available():
            return {"name": "CPU", "total_gb": 0, "used_gb": 0, "free_gb": 0,
                    "device": "cpu", "gpu_count": 0}
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024 ** 3)
        used  = torch.cuda.memory_allocated(0) / (1024 ** 3)
        n = torch.cuda.device_count()
        all_gpus = [
            {
                "index": i,
                "name": torch.cuda.get_device_properties(i).name,
                "total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3), 1),
                "used_gb":  round(torch.cuda.memory_allocated(i) / (1024 ** 3), 1),
            }
            for i in range(n)
        ]
        return {
            "name": props.name,
            "total_gb": round(total, 2),
            "used_gb":  round(used, 2),
            "free_gb":  round(total - used, 2),
            "device":   self.device,
            "gpu_count": n,
            "all_gpus": all_gpus,
        }

    def get_model_config(self, model_name: str) -> ModelConfig:
        if model_name not in self.MODELS:
            raise ValueError(
                f"Unknown model: '{model_name}'. "
                f"Available: {list(self.MODELS.keys())}"
            )
        return self.MODELS[model_name]

    @classmethod
    def get_cache_dir(cls, model_name: str) -> str:
        settings = get_settings()
        if model_name in cls.SSD_PRIORITY:
            return settings.cache_dir_ssd
        return settings.cache_dir

    def _should_quantize(self, config: ModelConfig) -> bool:
        if not config.quantize:
            return False
        if config.bf16_min_vram_gb > 0.0 and self._get_vram_gb() >= config.bf16_min_vram_gb:
            logger.info(
                f"GPU {self._get_vram_gb():.0f} GB ≥ {config.bf16_min_vram_gb:.0f} GB "
                f"threshold — loading in BF16 (skipping quantization)"
            )
            return False
        return True

    def is_loaded(self) -> bool:
        return self.current_model in self.pipelines

    def list_models(self) -> Dict[str, str]:
        return {name: cfg.description for name, cfg in self.MODELS.items()}

    # ── Load / unload ────────────────────────────────────────────────────────

    def _unload_model(self, model_name: str) -> None:
        if model_name not in self.pipelines:
            return
        logger.info(f"Unloading {model_name}...")
        pipe = self.pipelines.pop(model_name)
        del pipe
        if model_name in self.lru_cache:
            self.lru_cache.remove(model_name)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        g = self.gpu_info()
        logger.info(f"Unloaded {model_name}. VRAM: {g['used_gb']:.1f}/{g['total_gb']:.1f} GB")

    def unload_all(self) -> None:
        for name in list(self.pipelines.keys()):
            self._unload_model(name)
        self.lru_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def load(self, model_name: str) -> None:
        """Load a model into GPU memory, unloading all others first."""
        settings = get_settings()

        if settings.hf_offline:
            _os.environ.setdefault("HF_HUB_OFFLINE", "1")
            _os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if model_name in self.pipelines:
            logger.info(f"{model_name} already loaded.")
            self.current_model = model_name
            return

        config = self.get_model_config(model_name)

        # Unload everything else to maximise free VRAM.
        for old in list(self.pipelines.keys()):
            self._unload_model(old)

        g = self.gpu_info()
        logger.info(
            f"Loading {model_name} (needs ~{config.vram_free_gb} GB free) | "
            f"GPU: {g['name']} | Free: {g['free_gb']:.1f}/{g['total_gb']:.1f} GB"
        )
        if g["total_gb"] > 0 and g["free_gb"] < config.vram_free_gb:
            raise RuntimeError(
                f"Insufficient VRAM for {model_name}: "
                f"need {config.vram_free_gb:.1f} GB, "
                f"only {g['free_gb']:.1f} GB free."
            )

        cache_dir = self.get_cache_dir(model_name)
        token = settings.hf_token
        if model_name in {"sd3-medium", "sd3.5-large"} and settings.sd3_hf_token:
            token = settings.sd3_hf_token

        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "cache_dir":   cache_dir,
        }
        if token:
            load_kwargs["token"] = token
        if config.variant:
            load_kwargs["variant"] = config.variant

        use_quantize = self._should_quantize(config)

        # Runtime override: FLUX_QUANTIZE=fp8|nf4|bf16
        flux_quant = _os.environ.get("FLUX_QUANTIZE", "").lower()
        if flux_quant and config.pipeline_class == FluxPipeline:
            if flux_quant == "bf16":
                use_quantize = False
            elif flux_quant in ("fp8", "nf4"):
                use_quantize = True
                config.quantize_type = flux_quant
            logger.info(f"FLUX_QUANTIZE override → {flux_quant}")

        try:
            pipe = None

            # ── FLUX FP8 path (torchao) ───────────────────────────────────────
            if use_quantize and config.pipeline_class == FluxPipeline and config.quantize_type == "fp8":
                try:
                    from torchao.quantization import quantize_, float8_weight_only
                    torchao_ok = True
                except ImportError:
                    logger.warning("torchao not installed — falling back to NF4 for FLUX")
                    torchao_ok = False

                if torchao_ok:
                    from diffusers import FluxTransformer2DModel
                    from transformers import T5EncoderModel

                    # T5-XXL in INT8: 9.5 GB → ~4.7 GB (text encoding is a one-time step, no speed hit)
                    logger.info(f"Loading {model_name}: T5-XXL text encoder → INT8")
                    text_encoder_2 = T5EncoderModel.from_pretrained(
                        config.model_id,
                        subfolder="text_encoder_2",
                        quantization_config=HFBitsAndBytesConfig(load_in_8bit=True),
                        torch_dtype=torch.bfloat16,
                        cache_dir=cache_dir,
                        token=token or None,
                    )

                    # Transformer in FP8: load BF16, quantize in-place, move to GPU
                    logger.info(f"Loading {model_name}: transformer BF16 → FP8 (torchao)")
                    transformer = FluxTransformer2DModel.from_pretrained(
                        config.model_id,
                        subfolder="transformer",
                        torch_dtype=torch.bfloat16,
                        cache_dir=cache_dir,
                        token=token or None,
                    )
                    quantize_(transformer, float8_weight_only())
                    transformer = transformer.to(self.device)

                    load_kwargs["transformer"]    = transformer
                    load_kwargs["text_encoder_2"] = text_encoder_2
                    pipe = FluxPipeline.from_pretrained(config.model_id, **load_kwargs)

                    # Move non-quantized components to GPU.
                    # text_encoder_2 is already on CUDA via bitsandbytes.
                    for attr in ("vae", "text_encoder"):
                        comp = getattr(pipe, attr, None)
                        if comp is not None:
                            comp.to(self.device)

                    logger.info(f"{model_name} FP8 load complete: transformer=FP8, T5-XXL=INT8")

            # ── FLUX NF4 path (bitsandbytes) — fallback / explicit override ────
            if pipe is None and use_quantize and config.pipeline_class == FluxPipeline:
                from diffusers import FluxTransformer2DModel
                nf4_cfg = HFBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info(f"Loading {model_name}: transformer → NF4")
                transformer = FluxTransformer2DModel.from_pretrained(
                    config.model_id,
                    subfolder="transformer",
                    quantization_config=nf4_cfg,
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir,
                    token=token or None,
                )
                load_kwargs["transformer"] = transformer
                pipe = FluxPipeline.from_pretrained(config.model_id, **load_kwargs)
                for attr in ("vae", "text_encoder", "text_encoder_2"):
                    comp = getattr(pipe, attr, None)
                    if comp is not None:
                        comp.to(self.device)

            # ── SD3.5 NF4 path ────────────────────────────────────────────────
            if pipe is None and use_quantize and config.pipeline_class == StableDiffusion3Pipeline:
                from diffusers import SD3Transformer2DModel
                nf4_cfg = HFBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                logger.info(f"Loading {model_name}: transformer → NF4")
                transformer = SD3Transformer2DModel.from_pretrained(
                    config.model_id,
                    subfolder="transformer",
                    quantization_config=nf4_cfg,
                    torch_dtype=torch.bfloat16,
                    cache_dir=cache_dir,
                    token=token or None,
                )
                load_kwargs["transformer"] = transformer
                pipe = StableDiffusion3Pipeline.from_pretrained(config.model_id, **load_kwargs)
                for attr in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
                    comp = getattr(pipe, attr, None)
                    if comp is not None:
                        comp.to(self.device)

            # ── Standard BF16 path (SDXL and other lightweight models) ─────────
            if pipe is None:
                logger.info(f"Loading {model_name} in BF16...")
                pipe = config.pipeline_class.from_pretrained(config.model_id, **load_kwargs)
                pipe.to(self.device)

            # ── Post-load optimisations ────────────────────────────────────────
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)
            
            # ── Uncensored Integration ─────────────────────────────────────────
            if model_name == "flux-1-dev":
                logger.info("Fusing Uncensored LoRA for FLUX.1-dev to ensure completely unrestricted generation...")
                try:
                    # Using a reliable community uncensored LoRA.
                    # It bypasses FLUX's implicit safety alignment while keeping its state-of-the-art prompt adherence.
                    pipe.load_lora_weights("shauray/flux.1-dev-uncensored", adapter_name="uncensored")
                    pipe.fuse_lora(adapter_names=["uncensored"])
                    logger.info("Uncensored LoRA fused successfully.")
                except Exception as lora_exc:
                    logger.warning(f"Failed to load uncensored LoRA: {lora_exc}")

            self.pipelines[model_name] = pipe
            self.lru_cache.append(model_name)
            self.current_model = model_name

            g = self.gpu_info()
            logger.info(
                f"✅ {model_name} loaded. "
                f"VRAM: {g['used_gb']:.1f}/{g['total_gb']:.1f} GB "
                f"({g['free_gb']:.1f} GB free)"
            )

        except Exception as exc:
            logger.exception(f"Failed to load {model_name}: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            err = str(exc)
            if "gated" in err.lower() or "403" in err:
                raise RuntimeError(
                    f"'{model_name}' requires gated HuggingFace access. "
                    "Request access on the model page and ensure HF_TOKEN is set."
                ) from exc
            raise RuntimeError(f"Failed to load '{model_name}': {exc}") from exc

    # ── Public API ───────────────────────────────────────────────────────────

    def get_pipeline(self, model_name: Optional[str] = None) -> Any:
        if model_name is None:
            model_name = self.current_model
        if model_name not in self.pipelines:
            self.load(model_name)
        return self.pipelines[model_name]

    def switch_model(self, model_name: str) -> None:
        if model_name == self.current_model and self.is_loaded():
            return
        logger.info(f"Switching model: {self.current_model} → {model_name}")
        self.load(model_name)

    def get_model_info(self, model_name: str) -> Dict:
        config = self.get_model_config(model_name)
        return {
            "name":                  model_name,
            "model_id":              config.model_id,
            "category":              config.category.value,
            "output_type":           config.output_type.value,
            "description":           config.description,
            "vram_needed_gb":        config.vram_free_gb,
            "min_steps":             config.min_steps,
            "max_steps":             config.max_steps,
            "default_steps":         config.default_steps,
            "default_guidance_scale": config.default_guidance_scale,
            "loaded":                model_name in self.pipelines,
        }

    def get_models_by_category(self, category: ModelCategory) -> Dict[str, ModelConfig]:
        return {
            name: cfg for name, cfg in self.MODELS.items()
            if cfg.category == category
        }
