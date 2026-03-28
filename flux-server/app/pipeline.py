import torch
import time
import base64
import io
import logging
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import BitsAndBytesConfig as TransformersBnBConfig
from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
import os
from pathlib import Path
from app.config import get_settings

logger = logging.getLogger(__name__)


class FluxInferencePipeline:
    """Manages the FLUX.1-dev diffusion pipeline lifecycle."""

    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded_loras = {}  # Map of name -> adapter_name
        self.current_lora = None

    def load(self) -> None:
        """Load the FLUX.1-dev model onto the GPU."""
        settings = get_settings()
        logger.info(f"Loading model: {settings.model_id}")
        logger.info(f"Cache dir: {settings.cache_dir}")
        logger.info(f"HF token present: {bool(settings.hf_token)}")
        logger.info(f"Device: {self.device}")

        try:
            # NF4 quantization config — compresses transformer from ~24GB to ~8GB
            # This lets FLUX.1-dev fit entirely on L4 GPU (22GB usable) without CPU offload
            nf4_config = DiffusersBnBConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            logger.info("Loading transformer with NF4 quantization...")
            transformer = FluxTransformer2DModel.from_pretrained(
                settings.model_id,
                subfolder="transformer",
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16,
                cache_dir=settings.cache_dir,
                token=settings.hf_token if settings.hf_token else None,
            )
            logger.info("Transformer loaded at 4-bit. Loading full pipeline...")

            load_kwargs = {
                "torch_dtype": torch.bfloat16,
                "cache_dir": settings.cache_dir,
                "transformer": transformer,
                "safety_checker": None,
                "requires_safety_checker": False,
                "feature_extractor": None,
            }
            if settings.hf_token:
                load_kwargs["token"] = settings.hf_token

            self.pipe = FluxPipeline.from_pretrained(
                settings.model_id,
                **load_kwargs,
            )
            logger.info("Pipeline loaded, moving to GPU...")

            self.pipe.to(self.device)
            logger.info("Model on GPU")

            # Memory optimizations
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

            # Warmup pass — ensures CUDA kernels are compiled
            logger.info("Running warmup inference...")
            with torch.no_grad():
                self.pipe(
                    prompt="warmup",
                    width=256,
                    height=256,
                    num_inference_steps=1,
                    output_type="latent",
                )
            torch.cuda.empty_cache()
            logger.info("Model loaded and warmed up successfully!")
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise

    def get_available_loras(self) -> list[str]:
        """List all .safetensors LoRA files in the loras/ directory."""
        lora_dir = Path("loras")
        if not lora_dir.exists():
            return []
        return [f.name for f in lora_dir.glob("*.safetensors")]

    def load_lora(self, lora_name: str) -> None:
        """Load a LoRA file into the pipeline."""
        if lora_name == "None" or not lora_name:
            self.current_lora = None
            return

        if lora_name in self.loaded_loras:
            self.current_lora = lora_name
            return

        lora_path = Path("loras") / lora_name
        if not lora_path.exists():
            logger.error(f"LoRA file not found: {lora_path}")
            return

        logger.info(f"Loading LoRA: {lora_name}")
        try:
            # We use the filename (without extension) as the adapter name
            adapter_name = Path(lora_name).stem
            self.pipe.load_lora_weights(
                str(lora_path),
                adapter_name=adapter_name,
            )
            self.loaded_loras[lora_name] = adapter_name
            self.current_lora = lora_name
            logger.info(f"Successfully loaded LoRA adapter: {adapter_name}")
        except Exception as e:
            logger.exception(f"Failed to load LoRA {lora_name}: {e}")
            raise

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        seed: int | None = None,
        lora_name: str | None = None,
        lora_scale: float = 1.0,
    ) -> tuple[str, int, float]:
        """
        Generate an image from a text prompt.

        Returns:
            tuple of (base64_encoded_png, seed_used, inference_time_ms)
        """
        # Generator on GPU — model is fully on GPU now with NF4 quantization
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        start = time.perf_counter()

        # Handle LoRA switching
        if lora_name and lora_name != "None":
            self.load_lora(lora_name)
            adapter_name = self.loaded_loras.get(lora_name)
            if adapter_name:
                self.pipe.set_adapters(adapter_name, adapter_weights=[lora_scale])
        else:
            # Disable LoRA if active
            if self.loaded_loras:
                self.pipe.disable_lora()

        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        image = result.images[0]

        # Encode to base64 PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        torch.cuda.empty_cache()

        logger.info(
            f"Generated {width}x{height} in {elapsed_ms:.0f}ms "
            f"(steps={num_inference_steps}, seed={seed})"
        )
        return img_b64, seed, elapsed_ms

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.pipe is not None

    def gpu_info(self) -> dict:
        """Get GPU name, total VRAM, and used VRAM."""
        if not torch.cuda.is_available():
            return {"name": "N/A", "total_gb": 0, "used_gb": 0}
        props = torch.cuda.get_device_properties(0)
        total_bytes = getattr(props, 'total_global_mem', None) or getattr(props, 'total_mem', 0)
        return {
            "name": torch.cuda.get_device_name(0),
            "total_gb": round(total_bytes / 1e9, 2),
            "used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        }


# Singleton — one pipeline per process (1 process per GPU)
flux_pipeline = FluxInferencePipeline()
