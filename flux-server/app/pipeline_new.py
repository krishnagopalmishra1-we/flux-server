"""
Multi-Model Inference Pipeline with LoRA support.
Supports FLUX.1-dev, SD3.5-Large, SDXL and others.
"""

import torch
import time
import base64
import io
import logging
from pathlib import Path
from diffusers import FluxPipeline, StableDiffusionXLImg2ImgPipeline
from app.config import get_settings
from app.model_manager import MultiModelManager

logger = logging.getLogger(__name__)

AUTO_NEGATIVE_BASE = (
    "low quality, worst quality, blurry, out of focus, pixelated, noisy, "
    "jpeg artifacts, watermark, text, logo, signature, extra fingers, "
    "deformed face, bad anatomy"
)

AUTO_NEGATIVE_STYLE = {
    "Photorealistic": "cartoon, anime, cgi, overprocessed skin",
    "Anime": "photo, realistic skin texture, 3d render",
    "Oil Painting": "photo, cgi, flat lighting",
    "Watercolor": "photo, hard edges, oversharpened",
    "3D Render": "painting, sketch, watercolor",
}

AUTO_NEGATIVE_MODEL = {
    "realvisxl-v5": "plastic skin, doll-like face, unnatural eyes",
    "juggernaut-xl": "overexposed highlights, waxy skin, oversaturated",
    "sd3.5-large": "repeated text, typographic artifacts",
}


class InferencePipeline:
    """Manages multi-model diffusion pipeline lifecycle with LoRA support."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_manager = MultiModelManager(default_model="flux-1-dev")
        self.loaded_loras = {}  # Map of name -> adapter_name
        self.current_lora = None
        self.sdxl_refiner = None
        self._weight_type_cache = {}

    def _get_sdxl_refiner(self) -> StableDiffusionXLImg2ImgPipeline:
        """Load SDXL refiner lazily and keep it cached."""
        if self.sdxl_refiner is not None:
            return self.sdxl_refiner

        settings = get_settings()
        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "cache_dir": settings.cache_dir,
            "safety_checker": None,
            "requires_safety_checker": False,
            "feature_extractor": None,
        }
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        logger.info("Loading SDXL refiner...")
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            **load_kwargs,
        )
        refiner.to(self.device)
        if hasattr(refiner, "enable_attention_slicing"):
            refiner.enable_attention_slicing()
        if hasattr(refiner, "enable_vae_slicing"):
            refiner.enable_vae_slicing()
        self.sdxl_refiner = refiner
        logger.info("SDXL refiner loaded")
        return refiner

    def load(self, model_name: str = "flux-1-dev") -> None:
        """Load a model onto the GPU with optional torch.compile acceleration."""
        logger.info(f"Loading model: {model_name}")
        settings = get_settings()
        logger.info(f"Cache dir: {settings.cache_dir}")
        logger.info(f"HF token present: {bool(settings.hf_token)}")
        logger.info(f"Device: {self.device}")

        # Enable PyTorch 2.x FlashAttention-2 / SDP kernels globally
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)

        try:
            self.model_manager.load(model_name)
            pipe = self.model_manager.get_pipeline(model_name)

            # Attempt torch.compile on non-quantized transformers for ~20% speedup
            # after warm-up. Skip BnB quantized models (incompatible graph capture).
            transformer = getattr(pipe, "transformer", None)
            is_quantized = getattr(transformer, "is_quantized", False) if transformer else True
            if transformer is None or is_quantized:
                compilation_status = "skipped (quantized or no transformer)"
            else:
                try:
                    logger.info("  Applying torch.compile to image transformer...")
                    pipe.transformer = torch.compile(
                        transformer,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    compilation_status = "compiled"
                    logger.info("  ✓ torch.compile applied")
                except Exception as ce:
                    compilation_status = "compile failed"
                    logger.warning(f"  torch.compile skipped: {ce}")

            # Warmup pass — triggers kernel compilation and cache warm
            logger.info("Running warmup inference pass...")
            with torch.no_grad():
                pipe(
                    prompt="warmup",
                    width=256,
                    height=256,
                    num_inference_steps=1,
                    output_type="latent",
                )
            torch.cuda.empty_cache()
            logger.info(f"✅ {model_name} loaded ({compilation_status}), warmed up")
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise

    def get_available_loras(self) -> list[str]:
        """List all .safetensors LoRA files in the loras/ directory."""
        lora_dir = Path("loras")
        if not lora_dir.exists():
            return []
        return [f.name for f in lora_dir.glob("*.safetensors")]

    def _detect_lora_type(self, lora_path: Path) -> str:
        """Detect LoRA architecture type from key names."""
        if str(lora_path) in self._weight_type_cache:
            return self._weight_type_cache[str(lora_path)]

        # Very large safetensors are usually full model checkpoints, not LoRAs.
        try:
            if lora_path.exists() and lora_path.stat().st_size > 2 * 1024 * 1024 * 1024:
                self._weight_type_cache[str(lora_path)] = "full-checkpoint"
                return "full-checkpoint"
        except Exception:
            pass

        try:
            from safetensors.torch import load_file
            keys = list(load_file(str(lora_path), device="cpu").keys())
            if not keys:
                detected = "unknown"
                self._weight_type_cache[str(lora_path)] = detected
                return detected

            key_blob = " ".join(keys[:300]).lower()

            # Check for LoRA-style keys first (lora_A, lora_B, lora_down, lora_up)
            has_lora_keys = any(x in key_blob for x in ["lora_a", "lora_b", "lora_down", "lora_up", "lora_linear"])

            if has_lora_keys:
                # It's a LoRA — determine which architecture
                if any(x in key_blob for x in ["double_blocks", "single_blocks", "transformer"]):
                    detected = "flux"
                elif "lora_unet" in key_blob or "lora_te" in key_blob:
                    detected = "sdxl"
                elif "processor" in key_blob:
                    detected = "sd3"
                else:
                    detected = "flux"  # Default for unrecognized LoRA format
                self._weight_type_cache[str(lora_path)] = detected
                return detected

            # No LoRA keys — check if it's a full checkpoint
            if any(x in key_blob for x in ["double_blocks.", "single_blocks.", "time_text_embed", "x_embedder"]):
                detected = "full-checkpoint"
                self._weight_type_cache[str(lora_path)] = detected
                return detected

            sample = keys[0].lower()
            if "transformer" in sample or "single_transformer" in sample:
                detected = "flux"
                self._weight_type_cache[str(lora_path)] = detected
                return detected

            detected = "unknown"
            self._weight_type_cache[str(lora_path)] = detected
            return detected
        except Exception:
            detected = "unknown"
            self._weight_type_cache[str(lora_path)] = detected
            return detected

    def _model_lora_family(self, model_name: str) -> str:
        """Return LoRA family string for a model."""
        if model_name.startswith("flux-"):
            return "flux"
        if model_name.startswith("sd3"):
            return "sd3"
        return "sdxl"  # sdxl-turbo and others

    def get_model_family(self, model_name: str) -> str:
        """Public helper for UI/model routing."""
        return self._model_lora_family(model_name)

    def get_recommended_lora_scale(self, model_name: str) -> float:
        """Return safe default LoRA strength per model family."""
        family = self._model_lora_family(model_name)
        if family == "flux":
            return 0.85
        if family == "sd3":
            return 0.8
        return 0.75  # sdxl

    def get_compatible_loras(self, model_name: str) -> list[str]:
        """Return all available LoRA files."""
        return self.get_available_loras()

    def pick_model_for_lora(self, lora_name: str, current_model: str) -> str:
        """Pick best model automatically for a LoRA file."""
        if not lora_name or lora_name == "None":
            return current_model

        lora_type = self._detect_lora_type(Path("loras") / lora_name)
        if lora_type == "full-checkpoint":
            return current_model
        if lora_type == "unknown":
            return current_model

        model_order = {
            "flux": ["flux-1-dev"],
            "sdxl": ["realvisxl-v5", "juggernaut-xl"],
            "sd3": ["sd3.5-large"],
        }
        for model_name in model_order.get(lora_type, []):
            if model_name in self.model_manager.MODELS:
                return model_name
        return current_model

    def get_auto_negative_prompt(self, model_name: str, style: str | None = None) -> str:
        """Build a safe default negative prompt based on model/style."""
        chunks = [AUTO_NEGATIVE_BASE]
        model_specific = AUTO_NEGATIVE_MODEL.get(model_name)
        if model_specific:
            chunks.append(model_specific)
        if style and style in AUTO_NEGATIVE_STYLE:
            chunks.append(AUTO_NEGATIVE_STYLE[style])
        return ", ".join(chunks)

    def load_lora(self, lora_name: str, model_name: str = None) -> None:
        """Load a LoRA file into the current pipeline with compatibility check."""
        if lora_name == "None" or not lora_name:
            self.current_lora = None
            return

        if lora_name in self.loaded_loras:
            self.current_lora = lora_name
            return

        lora_path = Path("loras") / lora_name
        if not lora_path.exists():
            raise RuntimeError(f"LoRA file not found: {lora_path}")

        # Compatibility check
        lora_type = self._detect_lora_type(lora_path)
        if lora_type == "full-checkpoint":
            raise RuntimeError(
                f"'{lora_name}' is a full model checkpoint, not a LoRA adapter. "
                f"Full model checkpoints cannot be loaded as LoRA adapters."
            )

        model_family = self._model_lora_family(model_name or self.model_manager.current_model)
        if lora_type not in ("unknown", model_family):
            logger.warning(
                f"LoRA '{lora_name}' detected as {lora_type.upper()} but loading on "
                f"{model_name} ({model_family.upper()}). May be incompatible."
            )

        logger.info(f"Loading LoRA: {lora_name} (type={lora_type})")
        try:
            pipe = self.model_manager.get_pipeline(model_name)
            adapter_name = Path(lora_name).stem.replace(".", "_")
            pipe.load_lora_weights(
                str(lora_path),
                adapter_name=adapter_name,
            )
            self.loaded_loras[lora_name] = adapter_name
            self.current_lora = lora_name
            logger.info(f"Successfully loaded LoRA adapter: {adapter_name}")
        except Exception as e:
            logger.exception(f"Failed to load LoRA {lora_name}: {e}")
            raise RuntimeError(
                f"LoRA '{lora_name}' failed to load on {model_name}: {e}. "
                f"This LoRA may be incompatible with the selected model."
            )

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        model_name: str = "flux-1-dev",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int | None = None,
        lora_name: str | None = None,
        lora_scale: float = 1.0,
        use_refiner: bool = False,
        style: str | None = None,
    ) -> tuple[str, int, float]:
        """
        Generate an image from a text prompt.

        Returns:
            tuple of (base64_encoded_png, seed_used, inference_time_ms)
        """
        # Switch to requested model (clears loaded LoRAs if model changes)
        previous_model = self.model_manager.current_model
        if model_name != self.model_manager.current_model:
            self.loaded_loras.clear()
            self.current_lora = None
        try:
            self.model_manager.switch_model(model_name)
        except Exception as switch_error:
            recovery_note = ""
            if previous_model and previous_model != model_name:
                try:
                    self.model_manager.switch_model(previous_model)
                    recovery_note = f" Reverted to previous model '{previous_model}'."
                except Exception:
                    recovery_note = " Could not recover previous model automatically."
            raise RuntimeError(
                f"Unable to switch to model '{model_name}': {switch_error}.{recovery_note}"
            ) from switch_error

        pipe = self.model_manager.get_pipeline(model_name)
        model_cfg = self.model_manager.get_model_config(model_name)

        # Enforce model-safe generation parameters.
        num_inference_steps = max(model_cfg.min_steps, min(num_inference_steps, model_cfg.max_steps))

        # Auto-fill negative prompt for non-FLUX models when user leaves it blank.
        effective_negative_prompt = (negative_prompt or "").strip()
        if not effective_negative_prompt and model_cfg.pipeline_class != FluxPipeline:
            effective_negative_prompt = self.get_auto_negative_prompt(model_name, style)
        
        # Generator on GPU
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        start = time.perf_counter()

        # Handle LoRA switching
        if lora_name and lora_name != "None":
            self.load_lora(lora_name, model_name)
            adapter_name = self.loaded_loras.get(lora_name)
            if adapter_name and hasattr(pipe, 'set_adapters'):
                # Keep LoRA strength in a safe range to reduce artifact/failure risk.
                recommended = self.get_recommended_lora_scale(model_name)
                effective_scale = lora_scale if lora_scale > 0 else recommended
                effective_scale = max(0.3, min(1.2, effective_scale))
                pipe.set_adapters(adapter_name, adapter_weights=[effective_scale])
        else:
            # Disable LoRA if active
            if hasattr(pipe, 'disable_lora'):
                try:
                    pipe.disable_lora()
                except Exception:
                    pass  # No adapters loaded yet

        # Build generation kwargs
        gen_kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        # Only pass negative_prompt to models that support it (not FLUX)
        if effective_negative_prompt and model_cfg.pipeline_class != FluxPipeline:
            gen_kwargs["negative_prompt"] = effective_negative_prompt

        with torch.no_grad():
            if use_refiner and model_name in {"realvisxl-v5", "juggernaut-xl"}:
                # Two-stage SDXL: base produces latents, refiner polishes details.
                high_noise_frac = 0.8
                base_result = pipe(
                    **gen_kwargs,
                    output_type="latent",
                    denoising_end=high_noise_frac,
                )
                refiner = self._get_sdxl_refiner()
                refiner_steps = max(10, min(30, num_inference_steps))
                refiner_kwargs = {
                    "prompt": prompt,
                    "image": base_result.images,
                    "num_inference_steps": refiner_steps,
                    "denoising_start": high_noise_frac,
                    "guidance_scale": max(1.0, guidance_scale),
                    "generator": generator,
                }
                if effective_negative_prompt:
                    refiner_kwargs["negative_prompt"] = effective_negative_prompt
                result = refiner(**refiner_kwargs)
            else:
                result = pipe(**gen_kwargs)

        elapsed_ms = (time.perf_counter() - start) * 1000
        image = result.images[0]

        # Encode to base64 PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        torch.cuda.empty_cache()

        logger.info(
            f"Generated {width}x{height} in {elapsed_ms:.0f}ms "
            f"(steps={num_inference_steps}, seed={seed}, model={model_name})"
        )
        return img_b64, seed, elapsed_ms

    @property
    def is_loaded(self) -> bool:
        """Check if the current model is loaded."""
        return self.model_manager.is_loaded()
    
    def list_available_models(self) -> dict:
        """List available image models (backward-compatible for Gradio UI)."""
        from app.model_manager import ModelCategory
        return self.model_manager.list_models(ModelCategory.IMAGE)
    
    def get_model_info(self, model_name: str = None) -> dict:
        """Get info about a model."""
        if model_name is None:
            model_name = self.model_manager.current_model
        return self.model_manager.get_model_info(model_name)

    def gpu_info(self) -> dict:
        """Get GPU name, total VRAM, and used VRAM."""
        return self.model_manager.gpu_info()


# Singleton — one pipeline per process (1 process per GPU)
inference_pipeline = InferencePipeline()
