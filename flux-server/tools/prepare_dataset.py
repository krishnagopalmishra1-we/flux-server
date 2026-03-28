import argparse
import csv
import hashlib
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm


def md5_bytes(data: bytes) -> str:
    h = hashlib.md5()
    h.update(data)
    return h.hexdigest()


def safe_filename(idx: int, source: str, ext: str = ".jpg") -> str:
    clean = "".join(c.lower() if c.isalnum() else "-" for c in source)[:40].strip("-")
    return f"img-{idx:07d}-{clean}{ext}"


def fetch_image(url: str, timeout: int = 20) -> bytes:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def preprocess_image(raw: bytes, size: int) -> Image.Image:
    img = Image.open(BytesIO(raw)).convert("RGB")
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def save_split_manifest(rows: List[Dict], out_dir: Path, val_ratio: float) -> None:
    total = len(rows)
    val_count = int(total * val_ratio)
    train_rows = rows[val_count:]
    val_rows = rows[:val_count]

    train_csv = out_dir / "train_manifest.csv"
    val_csv = out_dir / "val_manifest.csv"

    with train_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(train_rows)

    with val_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(val_rows)



def run(manifest_csv: str, out_dir: str, size: int, val_ratio: float, max_images: int) -> Tuple[int, int]:
    out = Path(out_dir)
    images_dir = out / "images"
    out.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_csv)
    required_cols = {"image_url", "caption", "source_dataset", "license"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in manifest: {missing}")

    if max_images > 0:
        df = df.head(max_images)

    seen_hashes = set()
    accepted_rows: List[Dict] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing dataset"):
        url = str(row.get("image_url", "")).strip()
        if not url:
            continue

        try:
            raw = fetch_image(url)
            digest = md5_bytes(raw)
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)

            img = preprocess_image(raw, size)
            name = safe_filename(idx + 1, str(row.get("source_dataset", "source")))
            rel_path = f"images/{name}"
            abs_path = out / rel_path
            img.save(abs_path, format="JPEG", quality=92, optimize=True)

            accepted_rows.append(
                {
                    "image_path": rel_path,
                    "caption": str(row.get("caption", "")).strip(),
                    "source_dataset": str(row.get("source_dataset", "")).strip(),
                    "license": str(row.get("license", "")).strip(),
                    "author": str(row.get("author", "")).strip(),
                    "source_page": str(row.get("source_page", "")).strip(),
                }
            )
        except Exception:
            continue

    if not accepted_rows:
        raise RuntimeError("No valid images prepared. Check network/manifest URLs.")

    all_csv = out / "all_manifest.csv"
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=accepted_rows[0].keys())
        w.writeheader()
        w.writerows(accepted_rows)

    save_split_manifest(accepted_rows, out, val_ratio)
    return len(df), len(accepted_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LoRA dataset from manifest CSV")
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--size", type=int, default=512, help="Square resize dimension")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap (0 = all)")

    args = parser.parse_args()
    total, kept = run(args.manifest, args.out, args.size, args.val_ratio, args.max_images)
    print(f"Done. Input rows: {total}, prepared images: {kept}")
