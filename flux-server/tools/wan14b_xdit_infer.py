from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.pipelines.video_pipeline import RESOLUTION_MAP, _blend_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WAN 14B xDiT BF16 inference helper")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--resolution", choices=sorted(RESOLUTION_MAP.keys()), default="720p")
    parser.add_argument("--total-frames", type=int, required=True)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--num-inference-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=49)
    parser.add_argument("--chunk-overlap", type=int, default=12)
    parser.add_argument("--output", required=True)
    parser.add_argument("--result-json", required=True)
    return parser.parse_args()


def save_video(frames: list, fps: int, output_path: Path) -> None:
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        output_params=["-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"],
        macro_block_size=None,
    )
    try:
        for frame in frames:
            arr = np.asarray(frame)
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            writer.append_data(arr)
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    result_path = Path(args.result_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    # Trigger model registration side effects before importing the runner.
    import xfuser.model_executor.models.runner_models  # noqa: F401
    from xfuser.runner import xFuserModelRunner
    from xfuser.core.utils.runner_utils import is_last_process

    height, width = RESOLUTION_MAP[args.resolution]
    height = (height // 32) * 32
    width = (width // 32) * 32

    runner = xFuserModelRunner(
        {
            "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "ulysses_degree": max(torch.cuda.device_count(), 1),
            "ring_degree": 1,
            "data_parallel_degree": 1,
            "tensor_parallel_degree": 1,
            "output_directory": str(output_path.parent),
            "num_iterations": 1,
            "warmup_calls": 0,
            "height": height,
            "width": width,
            "num_frames": args.chunk_size,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
        }
    )

    base_input = runner.preprocess_args(
        {
            "height": height,
            "width": width,
            "num_frames": args.chunk_size,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
        }
    )

    init_start = time.perf_counter()
    runner.initialize(base_input)
    init_elapsed_ms = (time.perf_counter() - init_start) * 1000

    step = args.chunk_size - args.chunk_overlap
    num_chunks = max(1, math.ceil((args.total_frames - args.chunk_overlap) / step))
    all_frames: list = []
    chunk_timings: list[float] = []

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * step
        requested_end = min(chunk_start + args.chunk_size, args.total_frames)
        requested_frames = requested_end - chunk_start
        chunk_frames = args.chunk_size if chunk_idx < num_chunks - 1 else max(16, requested_frames)
        chunk_frames = min(chunk_frames, args.chunk_size)

        chunk_args = dict(base_input)
        chunk_args["num_frames"] = chunk_frames
        chunk_args["seed"] = args.seed + chunk_idx

        output, timings = runner.run(chunk_args)
        chunk_timings.extend(timings)
        raw_frames = output.videos[0]
        chunk_frame_list = list(raw_frames) if hasattr(raw_frames, "__iter__") else []

        if all_frames and args.chunk_overlap > 0 and len(chunk_frame_list) > args.chunk_overlap:
            blended = _blend_overlap(all_frames[-args.chunk_overlap:], chunk_frame_list[:args.chunk_overlap])
            all_frames = all_frames[:-args.chunk_overlap] + blended + chunk_frame_list[args.chunk_overlap:]
        else:
            all_frames.extend(chunk_frame_list)

    all_frames = all_frames[: args.total_frames]

    if is_last_process():
        save_video(all_frames, args.fps, output_path)
        total_inference_ms = sum(chunk_timings) * 1000
        result_path.write_text(
            json.dumps(
                {
                    "backend": "xdit-bf16",
                    "resolution": args.resolution,
                    "num_frames": len(all_frames),
                    "duration_seconds": len(all_frames) / max(args.fps, 1),
                    "inference_time_ms": round(total_inference_ms, 0),
                    "model_init_time_ms": round(init_elapsed_ms, 0),
                    "seed_used": args.seed,
                    "chunks_generated": num_chunks,
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"output": str(output_path), "result_json": str(result_path)}), flush=True)

    runner.cleanup()


if __name__ == "__main__":
    main()
