"""Training preset templates for LoRA jobs."""

from typing import Dict


def get_lora_preset(style: str) -> Dict:
    key = style.strip().lower()

    base = {
        "resolution": 512,
        "batch_size": 4,
        "rank": 16,
        "alpha": 32,
        "optimizer": "adamw8bit",
        "lr_scheduler": "cosine",
        "checkpoint_every": 100,
        "validation_split": 0.2,
    }

    if key == "realism":
        return {
            **base,
            "style": "realism",
            "learning_rate": 1e-4,
            "max_train_steps": 800,
            "caption_dropout": 0.05,
            "notes": "Best for natural-light photography and product/lifestyle realism.",
        }

    if key == "anime":
        return {
            **base,
            "style": "anime",
            "learning_rate": 1.5e-4,
            "max_train_steps": 1000,
            "caption_dropout": 0.1,
            "notes": "Use stronger style tags and keep composition captions concise.",
        }

    if key in {"face", "portrait"}:
        return {
            **base,
            "style": "face",
            "learning_rate": 8e-5,
            "max_train_steps": 700,
            "caption_dropout": 0.03,
            "notes": "Use strict face alignment and identity-safe policies.",
        }

    raise ValueError("Unsupported style. Use realism, anime, or face")
