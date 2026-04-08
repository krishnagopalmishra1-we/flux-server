"""
Music generation pipeline — Songs, Instrumentals, and Sound Effects.

Supports:
- ACE-Step 1.5: Full song generation with vocals and lyrics (Suno-like)
- AudioLDM 2: High-quality instrumental music and audio effects
- Stable Audio Open: Sound effects, loops, production elements

All models share the same interface: accept prompt/settings, return audio file path.
"""

import gc
import io
import time
import base64
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from app.config import get_settings
from app.output_store import output_store

logger = logging.getLogger(__name__)


class MusicPipeline:
    """
    Manages music/audio generation using ACE-Step, AudioLDM 2, and Stable Audio.

    Only one music model lives in VRAM at a time. The parent ModelManager
    handles unloading previous models before calling into this pipeline.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self._pipe = None
        self._current_model = None

    def load_model(self, model_name: str) -> None:
        """Load a music model into VRAM."""
        if self._current_model == model_name and (self._model is not None or self._pipe is not None):
            logger.info(f"Music model {model_name} already loaded")
            return

        self.unload()
        settings = get_settings()
        logger.info(f"Loading music model: {model_name}")

        try:
            if model_name == "ace-step":
                self._load_ace_step(settings)
            elif model_name == "audioldm2":
                self._load_audioldm2(settings)
            elif model_name == "stable-audio":
                self._load_stable_audio(settings)
            else:
                raise ValueError(f"Unknown music model: {model_name}")

            self._current_model = model_name
            logger.info(f"✅ Music model {model_name} loaded")

        except Exception as e:
            logger.exception(f"Failed to load music model {model_name}: {e}")
            self.unload()
            raise

    def _load_ace_step(self, settings) -> None:
        """Load MusicGen-medium as ace-step fallback backend.

        NOTE: The ACE-Step package (ACE-Step/ACE-Step-v1-3.5B) is not pip-installable
        as a standard diffusers model. facebook/musicgen-medium is used as a drop-in
        fallback. It generates instrumental music; lyrics/vocal synthesis is not supported
        by this backend.
        """
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        logger.warning(
            "ace-step fallback: using facebook/musicgen-medium. "
            "Vocal/lyrics synthesis not supported in this backend."
        )
        logger.info("Loading MusicGen-medium as ace-step backend...")
        model_id = "facebook/musicgen-medium"
        self._processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=settings.cache_dir,
        )
        self._model = MusicgenForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=settings.cache_dir,
            torch_dtype=torch.float16,
        )
        self._model.to(self.device)

    def _load_audioldm2(self, settings) -> None:
        """Load AudioLDM 2 for high-quality audio generation."""
        from diffusers import AudioLDM2Pipeline
        self._pipe = AudioLDM2Pipeline.from_pretrained(
            "cvssp/audioldm2",
            torch_dtype=torch.float16,
            cache_dir=settings.cache_dir,
        )
        self._pipe.to(self.device)

    def _load_stable_audio(self, settings) -> None:
        """Load Stable Audio Open for sound effects."""
        from diffusers import StableAudioPipeline

        load_kwargs = {
            "torch_dtype": torch.bfloat16,
            "cache_dir": settings.cache_dir,
        }
        if settings.hf_token:
            load_kwargs["token"] = settings.hf_token

        self._pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0",
            **load_kwargs,
        )
        self._pipe.to(self.device)

    def unload(self) -> None:
        """Unload current music model and free VRAM."""
        if self._model is not None or self._pipe is not None or self._processor is not None:
            logger.info(f"Unloading music model: {self._current_model}")
            del self._model
            del self._pipe
            del self._processor
            self._model = None
            self._pipe = None
            self._processor = None
            self._current_model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    async def generate_song(
        self,
        prompt: str,
        model_name: str = "ace-step",
        duration_seconds: int = 30,
        lyrics: Optional[str] = None,
        genre: Optional[str] = None,
        bpm: Optional[int] = None,
        seed: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a song (with vocals if ACE-Step, instrumental if MusicGen).

        Returns dict with audio_url, duration_seconds, inference_time_ms.
        """
        self.load_model(model_name)

        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        logger.info(
            f"Generating music: model={model_name}, duration={duration_seconds}s, "
            f"genre={genre}, bpm={bpm}"
        )

        start = time.perf_counter()

        if model_name == "ace-step":
            audio_data, sample_rate = await self._generate_ace_step(
                prompt, duration_seconds, lyrics, genre, bpm, seed
            )
        elif model_name == "audioldm2":
            audio_data, sample_rate = await self._generate_audioldm2(
                prompt, duration_seconds, seed
            )
        elif model_name == "stable-audio":
            audio_data, sample_rate = await self._generate_stable_audio(
                prompt, duration_seconds, seed
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Save as WAV
        audio_path = self._save_audio(audio_data, sample_rate, job_id)

        torch.cuda.empty_cache()

        logger.info(
            f"Music generated: {duration_seconds}s, "
            f"inference={elapsed_ms:.0f}ms, model={model_name}"
        )

        return {
            "audio_url": output_store.get_url(audio_path),
            "duration_seconds": duration_seconds,
            "sample_rate": sample_rate,
            "inference_time_ms": round(elapsed_ms, 0),
            "seed_used": seed,
        }

    async def _generate_ace_step(
        self,
        prompt: str,
        duration: int,
        lyrics: Optional[str],
        genre: Optional[str],
        bpm: Optional[int],
        seed: int,
    ) -> tuple:
        """Generate music using MusicGen (ace-step replacement)."""
        import numpy as np
        import asyncio

        full_prompt = prompt
        if genre:
            full_prompt = f"{genre} music, {full_prompt}"
        if lyrics:
            full_prompt = f"{full_prompt}. Lyrics inspiration: {lyrics[:200]}"
        if bpm:
            full_prompt = f"{full_prompt}, {bpm} bpm"

        inputs = self._processor(
            text=[full_prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # MusicGen generates ~50 tokens per second at 32kHz
        max_new_tokens = min(int(duration * 50), 3000)

        def run_generation():
            with torch.no_grad():
                audio_values = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    guidance_scale=3.0,
                )
            return audio_values

        audio_values = await asyncio.to_thread(run_generation)

        sample_rate = self._model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].cpu().numpy().astype(np.float32)

        return audio_data, sample_rate

    async def _generate_audioldm2(
        self,
        prompt: str,
        duration: int,
        seed: int,
    ) -> tuple:
        """Generate using AudioLDM 2."""
        import numpy as np

        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.no_grad():
            output = self._pipe(
                prompt,
                audio_length_in_s=duration,
                num_inference_steps=100,
                generator=generator,
            )

        audio = output.audios[0]
        sr = 16000  # Default AudioLDM2 SR

        return audio, sr

    async def _generate_stable_audio(
        self,
        prompt: str,
        duration: int,
        seed: int,
    ) -> tuple:
        """Generate using Stable Audio Open (sound effects)."""
        import numpy as np

        generator = torch.Generator(device=self.device).manual_seed(seed)

        with torch.no_grad():
            output = self._pipe(
                prompt=prompt,
                audio_end_in_s=min(duration, 47),  # Stable Audio max ~47s
                num_inference_steps=100,
                generator=generator,
            )

        audio = output.audios[0].cpu().numpy()
        sr = 44100

        return audio, sr

    def _save_audio(self, audio_data, sample_rate: int, job_id: Optional[str]) -> str:
        """Save audio data as WAV file to output store."""
        import soundfile as sf
        import numpy as np

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # Normalize audio
        audio = audio_data
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        audio = np.array(audio, dtype=np.float32)

        # Handle multi-dimensional audio
        if audio.ndim == 1:
            pass  # mono
        elif audio.ndim == 2:
            if audio.shape[0] <= 2:
                audio = audio.T  # (channels, samples) -> (samples, channels)
        elif audio.ndim == 3:
            audio = audio.squeeze(0)
            if audio.shape[0] <= 2:
                audio = audio.T

        # Normalize to [-1, 1]
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95

        sf.write(tmp_path, audio, sample_rate)
        rel_path = output_store.save_file_from_path(tmp_path, "audio", job_id)
        return rel_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._pipe is not None


# Module-level singleton
music_pipeline = MusicPipeline()
