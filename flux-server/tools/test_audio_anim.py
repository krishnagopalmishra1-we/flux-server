"""
Targeted smoke test for Music and Animation models:
  MUSIC:     ace-step, audioldm2, stable-audio
  ANIMATION: liveportrait, echomimic

Usage:
    python tools/test_audio_anim.py --server http://104.197.170.202:8080
"""

import argparse
import base64
import io
import json
import struct
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


def wait_for_job(base_url: str, job_id: str, timeout_sec: int = 2400, poll_sec: int = 15):
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
    return False, {"status": "timeout", "error_message": f"Timed out after {timeout_sec}s"}


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


def test_music(base_url: str, model_name: str, retries: int) -> dict:
    log(f"[music] >> {model_name}")
    for attempt in range(1, retries + 1):
        req = {
            "prompt": f"calm ambient test track for {model_name}",
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
            log(f"[music] FAIL {model_name} attempt={attempt} error={str(body)[:200]}")
            continue

        log(f"[music] submitted job_id={job_id}")
        ok, result = wait_for_job(base_url, job_id)
        if ok:
            log(f"[music] PASS OK {model_name}")
            return {"category": "music", "model": model_name, "status": "PASS", "details": result.get("result", {})}
        log(f"[music] job failed: {str(result)[:200]}")

    return {"category": "music", "model": model_name, "status": "FAIL", "details": {"error": "all retries failed"}}


def test_animation(base_url: str, model_name: str, retries: int) -> dict:
    log(f"[animation] >> {model_name}")
    for attempt in range(1, retries + 1):
        req = {
            "model_name": model_name,
            "source_image_b64": FALLBACK_PNG_B64,
            "audio_b64": SILENCE_WAV_B64,
            "expression_scale": 1.0,
            "pose_style": 0,
            "use_enhancer": False,
        }

        code, body = http_json("POST", f"{base_url}/api/animation/generate", req, timeout=180)
        job_id = body.get("job_id") if isinstance(body, dict) else None
        if code != 200 or not job_id:
            log(f"[animation] FAIL {model_name} attempt={attempt} error={str(body)[:200]}")
            continue

        log(f"[animation] submitted job_id={job_id}")
        ok, result = wait_for_job(base_url, job_id)
        if ok:
            log(f"[animation] PASS OK {model_name}")
            return {"category": "animation", "model": model_name, "status": "PASS", "details": result.get("result", {})}
        log(f"[animation] job failed: {str(result)[:200]}")

    return {"category": "animation", "model": model_name, "status": "FAIL", "details": {"error": "all retries failed"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://104.197.170.202:8080")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    base = args.server.rstrip("/")
    retries = max(1, args.retries)

    log(f"Starting Audio/Animation test -> {base}")
    
    results = []
    
    # Music
    results.append(test_music(base, "ace-step", retries))
    results.append(test_music(base, "audioldm2", retries))
    results.append(test_music(base, "stable-audio", retries))
    
    # Animation
    results.append(test_animation(base, "liveportrait", retries))
    results.append(test_animation(base, "echomimic", retries))
    
    passed = len([r for r in results if r["status"] == "PASS"])
    log(f"DONE - PASS: {passed} / {len(results)}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
