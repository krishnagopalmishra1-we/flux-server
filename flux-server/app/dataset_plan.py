"""Dataset planning presets for realism/anime LoRA preparation."""

from typing import Dict, List


REALISM_STACK = {
    "name": "realism-v1",
    "total_images": 40000,
    "sources": [
        {"dataset": "LAION-Aesthetics", "count": 15000, "purpose": "photorealism and composition"},
        {"dataset": "OpenImages", "count": 10000, "purpose": "subject diversity"},
        {"dataset": "COCO", "count": 8000, "purpose": "scene grounding"},
        {"dataset": "Flickr Creative Commons", "count": 5000, "purpose": "natural variation"},
        {"dataset": "Unsplash Lite", "count": 2000, "purpose": "high-quality natural lighting"},
    ],
}


ANIME_STACK = {
    "name": "anime-v1",
    "total_images": 25000,
    "sources": [
        {"dataset": "Community Anime Datasets", "count": 18000, "purpose": "style identity"},
        {"dataset": "LAION-Aesthetics (anime tags)", "count": 7000, "purpose": "composition and polish"},
    ],
}


FACE_ADDON = {
    "name": "face-addon-v1",
    "total_images": 20000,
    "sources": [
        {"dataset": "CelebA", "count": 20000, "purpose": "face structure and realism"},
    ],
}


def get_dataset_plan(domain: str) -> Dict:
    key = domain.strip().lower()
    if key == "realism":
        return REALISM_STACK
    if key == "anime":
        return ANIME_STACK
    if key in {"face", "faces", "celeba"}:
        return FACE_ADDON
    raise ValueError("Unsupported domain. Use realism, anime, or face")


def available_domains() -> List[str]:
    return ["realism", "anime", "face"]
