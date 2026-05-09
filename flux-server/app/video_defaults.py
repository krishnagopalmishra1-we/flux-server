from __future__ import annotations

from copy import deepcopy
from typing import Any


VIDEO_MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "wan-t2v-1.3b": {
        "default_resolution": "480p",
        "default_num_frames": 33,
        "default_fps": 16,
        "default_guidance_scale": 5.0,
        "default_steps": 30,
        "default_chunk_size": 49,
        "default_chunk_overlap": 8,
        "allowed_resolutions": ("480p",),
        "max_frames": 240,
        "max_steps": 40,
        "max_chunk_size": 49,
        "max_chunk_overlap": 16,
        "preferred_backend": "diffusers-bf16",
    },
    "wan-t2v-14b": {
        "default_resolution": "720p",
        "default_num_frames": 49,
        "default_fps": 16,
        "default_guidance_scale": 6.0,
        "default_steps": 32,
        "default_chunk_size": 49,
        "default_chunk_overlap": 12,
        "allowed_resolutions": ("480p", "540p", "720p"),
        "max_frames": 240,
        "max_steps": 60,
        "max_chunk_size": 81,
        "max_chunk_overlap": 24,
        "preferred_backend": "xdit-bf16",
        "xdit_min_gpu_count": 4,
        "xdit_min_vram_gb": 70.0,
    },
    "wan-i2v-14b": {
        "default_resolution": "720p",
        "default_num_frames": 49,
        "default_fps": 16,
        "default_guidance_scale": 5.5,
        "default_steps": 32,
        "default_chunk_size": 49,
        "default_chunk_overlap": 12,
        "allowed_resolutions": ("480p", "540p", "720p"),
        "max_frames": 81,
        "max_steps": 60,
        "max_chunk_size": 81,
        "max_chunk_overlap": 24,
        "preferred_backend": "diffusers-nf4",
    },
    "hunyuan-video": {
        "default_resolution": "720p",
        "default_num_frames": 129,
        "default_fps": 24,
        "default_guidance_scale": 6.0,
        "default_steps": 50,
        "default_chunk_size": 49,
        "default_chunk_overlap": 8,
        "allowed_resolutions": ("540p", "720p"),
        "max_frames": 129,
        "max_steps": 60,
        "max_chunk_size": 49,
        "max_chunk_overlap": 16,
        "preferred_backend": "diffusers-nf4",
    },
}


def get_video_model_defaults(model_name: str) -> dict[str, Any]:
    defaults = VIDEO_MODEL_DEFAULTS.get(model_name)
    if defaults is None:
        raise ValueError(f"Unsupported video model: {model_name}")
    return deepcopy(defaults)


def apply_video_defaults(model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    defaults = get_video_model_defaults(model_name)
    merged = deepcopy(payload)
    for key, default_key in (
        ("resolution", "default_resolution"),
        ("num_frames", "default_num_frames"),
        ("fps", "default_fps"),
        ("guidance_scale", "default_guidance_scale"),
        ("num_inference_steps", "default_steps"),
        ("chunk_size", "default_chunk_size"),
        ("chunk_overlap", "default_chunk_overlap"),
    ):
        if merged.get(key) is None:
            merged[key] = defaults[default_key]
    return merged
