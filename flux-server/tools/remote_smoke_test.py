import argparse
import base64
import io
import json
import struct
import time
import urllib.error
import urllib.request
import wave


def log(message: str) -> None:
    print(message, flush=True)


def http_json(method: str, url: str, payload: dict | None = None, timeout: int = 120):
    headers = {"Content-Type": "application/json"}
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            return response.status, json.loads(text)
    except urllib.error.HTTPError as err:
        err_text = err.read().decode("utf-8", errors="replace")
        try:
            err_body = json.loads(err_text)
        except json.JSONDecodeError:
            err_body = {"detail": err_text}
        return err.code, err_body
    except Exception as err:
        return 0, {"detail": str(err)}


def wait_for_job(base_url: str, job_id: str, timeout_sec: int = 2400, poll_sec: int = 10):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        status, body = http_json("GET", f"{base_url}/api/jobs/{job_id}", timeout=120)
        if status != 200:
            return False, {"status": "failed", "error": body}
        state = body.get("status", "")
        if state == "completed":
            return True, body
        if state == "failed":
            return False, body
        time.sleep(poll_sec)
    return False, {"status": "timeout", "error_message": f"Timed out waiting for {job_id}"}


def make_silence_wav_b64(duration_sec: float = 1.0, sample_rate: int = 16000) -> str:
    frame_count = max(1, int(duration_sec * sample_rate))
    pcm = struct.pack("<" + "h" * frame_count, *([0] * frame_count))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def safe_steps(model: dict, floor: int, ceil: int) -> int:
    model_default = int(model.get("default_steps", floor))
    model_min = int(model.get("min_steps", floor))
    model_max = int(model.get("max_steps", ceil))
    candidate = max(model_min, min(model_default, ceil))
    return max(floor, min(candidate, model_max))


def run_all(base_url: str, retries: int = 2):
    summary = {
        "server": base_url,
        "health": None,
        "totals": {"pass": 0, "fail": 0},
        "results": [],
    }

    status, health = http_json("GET", f"{base_url}/health", timeout=120)
    if status != 200:
        raise RuntimeError(f"Health check failed: {status} {health}")
    summary["health"] = health
    log(f"[health] PASS gpu={health.get('gpu_name','')} vram={health.get('vram_used_gb',0)}/{health.get('vram_total_gb',0)}")

    status, models_payload = http_json("GET", f"{base_url}/models", timeout=120)
    if status != 200:
        raise RuntimeError(f"Model listing failed: {status} {models_payload}")
    log(f"[models] loaded {len(models_payload.get('models', []))} models")

    models = models_payload.get("models", [])
    by_cat = {"image": [], "video": [], "music": [], "animation": []}
    for model in models:
        cat = model.get("category")
        if cat in by_cat:
            by_cat[cat].append(model)

    fallback_png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8B6R8AAAAASUVORK5CYII="
    )
    silence_wav_b64 = make_silence_wav_b64(duration_sec=1.0)

    first_image_b64 = None

    for model in by_cat["image"]:
        model_name = model["name"]
        log(f"[image] testing {model_name}")
        passed = False
        last_error = "unknown"
        for attempt in range(1, retries + 1):
            req = {
                "prompt": f"cinematic portrait for smoke test {model_name}",
                "model_name": model_name,
                "width": 512,
                "height": 512,
                "num_inference_steps": safe_steps(model, floor=4, ceil=28),
                "guidance_scale": float(model.get("default_guidance_scale", 5.0)),
                "seed": 42,
            }
            code, body = http_json("POST", f"{base_url}/generate-ui", req, timeout=2400)
            image_b64 = body.get("image_base64") if isinstance(body, dict) else None
            if code == 200 and image_b64:
                passed = True
                if not first_image_b64:
                    first_image_b64 = image_b64
                summary["results"].append(
                    {
                        "category": "image",
                        "model": model_name,
                        "status": "PASS",
                        "attempt": attempt,
                        "details": {
                            "image_bytes_est": int(len(image_b64) * 0.75),
                            "inference_time_ms": body.get("inference_time_ms", 0),
                        },
                    }
                )
                summary["totals"]["pass"] += 1
                log(f"[image] PASS {model_name} attempt={attempt}")
                break
            last_error = str(body)
        if not passed:
            log(f"[image] FAIL {model_name} error={last_error[:300]}")
            summary["results"].append(
                {
                    "category": "image",
                    "model": model_name,
                    "status": "FAIL",
                    "details": {"error": last_error},
                }
            )
            summary["totals"]["fail"] += 1

    for model in by_cat["video"]:
        model_name = model["name"]
        log(f"[video] testing {model_name}")
        passed = False
        last_error = "unknown"
        for attempt in range(1, retries + 1):
            req = {
                "prompt": f"short smoke test clip for {model_name}",
                "model_name": model_name,
                "resolution": "480p",
                "num_frames": 16,
                "fps": 16,
                "guidance_scale": float(model.get("default_guidance_scale", 5.0)),
                "num_inference_steps": safe_steps(model, floor=10, ceil=20),
                "seed": 42,
            }
            if "i2v" in model_name.lower():
                req["source_image_b64"] = first_image_b64 or fallback_png_b64
            code, body = http_json("POST", f"{base_url}/api/video/generate", req, timeout=180)
            job_id = body.get("job_id") if isinstance(body, dict) else None
            if code != 200 or not job_id:
                last_error = str(body)
                continue
            ok, job_result = wait_for_job(base_url, job_id, timeout_sec=3600, poll_sec=15)
            if ok:
                passed = True
                summary["results"].append(
                    {
                        "category": "video",
                        "model": model_name,
                        "status": "PASS",
                        "attempt": attempt,
                        "details": {
                            "job_id": job_id,
                            "processing_time_ms": job_result.get("processing_time_ms", 0),
                            "result": job_result.get("result", {}),
                        },
                    }
                )
                summary["totals"]["pass"] += 1
                log(f"[video] PASS {model_name} attempt={attempt} job_id={job_id}")
                break
            last_error = str(job_result)
        if not passed:
            log(f"[video] FAIL {model_name} error={last_error[:300]}")
            summary["results"].append(
                {
                    "category": "video",
                    "model": model_name,
                    "status": "FAIL",
                    "details": {"error": last_error},
                }
            )
            summary["totals"]["fail"] += 1

    first_audio_b64 = silence_wav_b64

    for model in by_cat["music"]:
        model_name = model["name"]
        log(f"[music] testing {model_name}")
        passed = False
        last_error = "unknown"
        for attempt in range(1, retries + 1):
            req = {
                "prompt": f"short ambient smoke test for {model_name}",
                "model_name": model_name,
                "duration_seconds": 6,
                "seed": 42,
            }
            if model_name == "ace-step":
                req["lyrics"] = "la la la"
                req["genre"] = "ambient"
                req["bpm"] = 100
            code, body = http_json("POST", f"{base_url}/api/music/generate", req, timeout=180)
            job_id = body.get("job_id") if isinstance(body, dict) else None
            if code != 200 or not job_id:
                last_error = str(body)
                continue
            ok, job_result = wait_for_job(base_url, job_id, timeout_sec=2400, poll_sec=10)
            if ok:
                result_obj = job_result.get("result", {})
                if result_obj.get("audio_b64"):
                    first_audio_b64 = result_obj["audio_b64"]
                passed = True
                summary["results"].append(
                    {
                        "category": "music",
                        "model": model_name,
                        "status": "PASS",
                        "attempt": attempt,
                        "details": {
                            "job_id": job_id,
                            "processing_time_ms": job_result.get("processing_time_ms", 0),
                            "result": result_obj,
                        },
                    }
                )
                summary["totals"]["pass"] += 1
                log(f"[music] PASS {model_name} attempt={attempt} job_id={job_id}")
                break
            last_error = str(job_result)
        if not passed:
            log(f"[music] FAIL {model_name} error={last_error[:300]}")
            summary["results"].append(
                {
                    "category": "music",
                    "model": model_name,
                    "status": "FAIL",
                    "details": {"error": last_error},
                }
            )
            summary["totals"]["fail"] += 1

    for model in by_cat["animation"]:
        model_name = model["name"]
        log(f"[animation] testing {model_name}")
        passed = False
        last_error = "unknown"
        for attempt in range(1, retries + 1):
            req = {
                "model_name": model_name,
                "source_image_b64": first_image_b64 or fallback_png_b64,
                "audio_b64": first_audio_b64,
                "expression_scale": 1.0,
                "pose_style": 0,
                "use_enhancer": False,
            }
            code, body = http_json("POST", f"{base_url}/api/animation/generate", req, timeout=180)
            job_id = body.get("job_id") if isinstance(body, dict) else None
            if code != 200 or not job_id:
                last_error = str(body)
                continue
            ok, job_result = wait_for_job(base_url, job_id, timeout_sec=2400, poll_sec=10)
            if ok:
                passed = True
                summary["results"].append(
                    {
                        "category": "animation",
                        "model": model_name,
                        "status": "PASS",
                        "attempt": attempt,
                        "details": {
                            "job_id": job_id,
                            "processing_time_ms": job_result.get("processing_time_ms", 0),
                            "result": job_result.get("result", {}),
                        },
                    }
                )
                summary["totals"]["pass"] += 1
                log(f"[animation] PASS {model_name} attempt={attempt} job_id={job_id}")
                break
            last_error = str(job_result)
        if not passed:
            log(f"[animation] FAIL {model_name} error={last_error[:300]}")
            summary["results"].append(
                {
                    "category": "animation",
                    "model": model_name,
                    "status": "FAIL",
                    "details": {"error": last_error},
                }
            )
            summary["totals"]["fail"] += 1

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run complete remote smoke test across all listed models.")
    parser.add_argument("--server", default="http://127.0.0.1:8080", help="Base URL, e.g. http://34.42.126.83:8080")
    parser.add_argument("--retries", type=int, default=2, help="Retries per model")
    args = parser.parse_args()

    log(f"Starting smoke test against {args.server.rstrip('/')} with retries={max(1, args.retries)}")
    report = run_all(args.server.rstrip("/"), retries=max(1, args.retries))
    print(json.dumps(report, ensure_ascii=True, indent=2))

    if report["totals"]["fail"] > 0:
        log("Smoke test finished with failures.")
        raise SystemExit(2)
    log("Smoke test finished with all models passing.")


if __name__ == "__main__":
    main()