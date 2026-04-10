import argparse
import base64
import io
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import wave

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def http_json(method: str, url: str, payload: dict | None = None, timeout: int = 120):
    headers = {"Content-Type": "application/json"}
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url=url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        err_text = err.read().decode("utf-8", errors="replace")
        try:
            err_body = json.loads(err_text)
        except json.JSONDecodeError:
            err_body = {"detail": err_text}
        return err.code, err_body
    except Exception as err:
        return 0, {"detail": str(err)}

def wait_for_job(base_url: str, job_id: str, timeout_sec: int = 3600, poll_sec: int = 15):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        status, body = http_json("GET", f"{base_url}/api/jobs/{job_id}", timeout=120)
        if status != 200:
            return False, {"status": "failed", "error": body}
        state = body.get("status", "")
        elapsed = body.get("processing_time_ms", 0)
        log(f"  polling job {job_id}: state={state} elapsed={elapsed:.0f}ms")
        if state == "completed":
            return True, body
        if state == "failed":
            return False, body
        time.sleep(poll_sec)
    return False, {"status": "timeout", "error": f"Timed out after {timeout_sec}s"}

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

FALLBACK_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8B6R8AAAAASUVORK5CYII="
)

SILENCE_WAV_B64 = make_silence_wav_b64(duration_sec=1.0)

def test_video(base_url: str, model_name: str, is_i2v: bool, retries: int) -> dict:
    log(f"[video] >> {model_name} (HQ)")
    for attempt in range(1, retries + 1):
        req = {
            "prompt": f"hyper-realistic 8k cinematic footage of a futuristic city skyline at night with flying cars, neon lights reflecting on wet streets, masterpiece — {model_name}",
            "model_name": model_name,
            "resolution": "720p",
            "num_frames": 49,
            "fps": 16,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "seed": 42,
        }
        if is_i2v:
            req["source_image_b64"] = FALLBACK_PNG_B64

        code, body = http_json("POST", f"{base_url}/api/video/generate", req, timeout=180)
        job_id = body.get("job_id") if isinstance(body, dict) else None
        if code != 200 or not job_id:
            log(f"[video] FAIL {model_name} attempt={attempt} submit_error={str(body)[:200]}")
            continue

        log(f"[video] submitted job_id={job_id} (attempt {attempt})")
        ok, result = wait_for_job(base_url, job_id, timeout_sec=1800, poll_sec=20)
        if ok:
            log(f"[video] PASS OK {model_name} attempt={attempt}")
            return {"category": "video", "model": model_name, "status": "PASS",
                    "attempt": attempt, "details": result.get("result", {})}
        log(f"[video] job failed: {str(result)[:200]}")

    log(f"[video] FAIL FAIL {model_name} — all {retries} attempts exhausted")
    return {"category": "video", "model": model_name, "status": "FAIL",
            "details": {"error": "all retries exhausted"}}

def test_music(base_url: str, model_name: str, retries: int) -> dict:
    log(f"[music] >> {model_name} (HQ)")
    for attempt in range(1, retries + 1):
        req = {
            "prompt": f"high quality orchestral soundtrack with cinematic strings, brass, and deep percussion, emotional climax — {model_name}",
            "model_name": model_name,
            "duration_seconds": 30,
            "seed": 42,
        }
        if model_name == "ace-step":
            req["lyrics"] = "We rise above the skies tonight"
            req["genre"] = "orchestral pop"
            req["bpm"] = 120

        code, body = http_json("POST", f"{base_url}/api/music/generate", req, timeout=180)
        job_id = body.get("job_id") if isinstance(body, dict) else None
        if code != 200 or not job_id:
            log(f"[music] FAIL {model_name} attempt={attempt} submit_error={str(body)[:200]}")
            continue

        log(f"[music] submitted job_id={job_id} (attempt {attempt})")
        ok, result = wait_for_job(base_url, job_id, timeout_sec=2400, poll_sec=10)
        if ok:
            log(f"[music] PASS OK {model_name} attempt={attempt}")
            return {"category": "music", "model": model_name, "status": "PASS",
                    "attempt": attempt, "details": result.get("result", {})}
        log(f"[music] job failed: {str(result)[:200]}")

    log(f"[music] FAIL FAIL {model_name} — all {retries} attempts exhausted")
    return {"category": "music", "model": model_name, "status": "FAIL",
            "details": {"error": "all retries exhausted"}}

def test_animation(base_url: str, model_name: str, retries: int) -> dict:
    log(f"[animation] >> {model_name} (HQ)")
    for attempt in range(1, retries + 1):
        req = {
            "model_name": model_name,
            "source_image_b64": FALLBACK_PNG_B64,
            "audio_b64": SILENCE_WAV_B64,
            "expression_scale": 1.5,
            "pose_style": 2,
            "use_enhancer": True,
        }

        code, body = http_json("POST", f"{base_url}/api/animation/generate", req, timeout=180)
        job_id = body.get("job_id") if isinstance(body, dict) else None
        if code != 200 or not job_id:
            log(f"[animation] FAIL {model_name} attempt={attempt} submit_error={str(body)[:200]}")
            continue

        log(f"[animation] submitted job_id={job_id} (attempt {attempt})")
        ok, result = wait_for_job(base_url, job_id, timeout_sec=2400, poll_sec=10)
        if ok:
            log(f"[animation] PASS OK {model_name} attempt={attempt}")
            return {"category": "animation", "model": model_name, "status": "PASS",
                    "attempt": attempt, "details": result.get("result", {})}
        log(f"[animation] job failed: {str(result)[:200]}")

    log(f"[animation] FAIL FAIL {model_name} — all {retries} attempts exhausted")
    return {"category": "animation", "model": model_name, "status": "FAIL",
            "details": {"error": "all retries exhausted"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8080")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()
    base = args.server.rstrip("/")

    log(f"High-Quality Smoke Test -> {base}")

    code, health = http_json("GET", f"{base}/health", timeout=60)
    if code != 200:
        raise RuntimeError(f"Health check failed: {code} {health}")
    log(f"Health OK — gpu={health.get('gpu_name','')}  vram={health.get('vram_used_gb',0)}/{health.get('vram_total_gb',0)} GB")

    results = []
    totals = {"pass": 0, "fail": 0}

    def record(r):
        results.append(r)
        if r["status"] == "PASS":
            totals["pass"] += 1
        else:
            totals["fail"] += 1

    log("=" * 55)
    log("VIDEO MODELS")
    log("=" * 55)
    # Testing all models
    record(test_video(base, "wan-t2v-1.3b",  is_i2v=False, retries=args.retries))
    record(test_video(base, "wan-t2v-14b",   is_i2v=False, retries=args.retries))
    record(test_video(base, "wan-i2v-14b",   is_i2v=True,  retries=args.retries))
    record(test_video(base, "ltx-video",     is_i2v=False, retries=args.retries))

    log("=" * 55)
    log("MUSIC MODELS")
    log("=" * 55)
    record(test_music(base, "ace-step",     retries=args.retries))
    record(test_music(base, "audioldm2",    retries=args.retries))
    record(test_music(base, "stable-audio", retries=args.retries))

    log("=" * 55)
    log("ANIMATION MODELS")
    log("=" * 55)
    record(test_animation(base, "liveportrait", retries=args.retries))
    record(test_animation(base, "echomimic",    retries=args.retries))

    log("=" * 55)
    log(f"DONE — PASS: {totals['pass']}  FAIL: {totals['fail']}")
    log("=" * 55)
    print(json.dumps({"server": base, "totals": totals, "results": results}, indent=2))
    sys.exit(1 if totals["fail"] > 0 else 0)

if __name__ == "__main__":
    main()
