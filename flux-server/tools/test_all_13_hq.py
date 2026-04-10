import argparse
import base64
import io
import json
import struct
import time
import os
import urllib.error
import urllib.request
import wave
import subprocess

def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def http_json(method: str, url: str, payload: dict | None = None, timeout: int = 120):
    api_key = os.environ.get("FLUX_API_KEY")
    if not api_key:
        raise RuntimeError("FLUX_API_KEY environment variable is not set")
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }
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

def download_file(url, local_path):
    try:
        log(f"  Downloading result -> {local_path}")
        urllib.request.urlretrieve(url, local_path)
        return True
    except Exception as e:
        log(f"  Download failed: {e}")
        return False

def wait_for_job(base_url: str, job_id: str, timeout_sec: int = 3600, poll_sec: int = 30):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        status, body = http_json("GET", f"{base_url}/api/jobs/{job_id}", timeout=120)
        if status != 200:
            return False, {"status": "failed", "error": body}
        state = body.get("status", "")
        elapsed = body.get("processing_time_ms", 0)
        progress = body.get("progress", 0)
        log(f"  polling job {job_id}: state={state} progress={progress}% elapsed={elapsed:.0f}ms")
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

# Models to test
MODELS_IMAGE = ["flux-1-dev", "sd3.5-large", "realvisxl-v5", "juggernaut-xl"]
MODELS_VIDEO = ["wan-t2v-1.3b", "wan-t2v-14b", "wan-i2v-14b", "ltx-video"]
MODELS_MUSIC = ["ace-step", "audioldm2", "stable-audio"]
MODELS_ANIMATION = ["liveportrait", "echomimic"]

OUTPUT_DIR = "smoketest_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_image(base_url, model_name):
    log(f"[IMAGE] >> {model_name} (HQ)")
    req = {
        "prompt": f"A masterpiece oil painting of a futuristic hyper-detailed cosmic garden, vibrant colors, 8k resolution, cinematic lighting -- {model_name}",
        "model_name": model_name,
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 7.0,
        "seed": 42
    }
    code, body = http_json("POST", f"{base_url}/generate", req, timeout=900)
    if code == 200:
        img_b64 = body.get("image_base64")
        if img_b64:
            path = os.path.join(OUTPUT_DIR, f"{model_name}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(img_b64))
            log(f"[IMAGE] PASS OK {model_name} (Saved: {path})")
            return {"status": "PASS", "details": "Generated and saved"}
        log(f"[IMAGE] FAIL {model_name}: 200 OK but no image_base64 in response: {body}")
        return {"status": "FAIL", "details": "No image returned"}
    else:
        log(f"[IMAGE] FAIL {model_name}: {body}")
        return {"status": "FAIL", "error": body}

def test_video(base_url, model_name):
    log(f"[VIDEO] >> {model_name} (HQ)")
    is_i2v = "i2v" in model_name
    req = {
        "prompt": f"Hyper-realistic cinematic drone shot of a volcanic eruption at night, lava flowing into the ocean, steam rising, 4k textures, slow motion -- {model_name}",
        "model_name": model_name,
        "resolution": "720p",
        "num_frames": 49,
        "fps": 16,
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "seed": 42
    }
    if is_i2v:
        req["source_image_b64"] = FALLBACK_PNG_B64
    
    code, body = http_json("POST", f"{base_url}/api/video/generate", req, timeout=180)
    job_id = body.get("job_id") if isinstance(body, dict) else None
    if code != 200 or not job_id:
        log(f"[VIDEO] FAIL {model_name} submit error: {body}")
        return {"status": "FAIL", "error": body}
    
    log(f"[VIDEO] job_id={job_id} submitted. Waiting...")
    ok, result = wait_for_job(base_url, job_id, timeout_sec=5400)
    if ok:
        res_data = result.get("result", {})
        url = res_data.get("video_url")
        if url:
            path = os.path.join(OUTPUT_DIR, f"{model_name}_{job_id}.mp4")
            download_file(url, path)
        log(f"[VIDEO] PASS OK {model_name}")
        return {"status": "PASS", "details": res_data}
    else:
        log(f"[VIDEO] FAIL {model_name}: {result}")
        return {"status": "FAIL", "error": result}

def test_music(base_url, model_name):
    log(f"[MUSIC] >> {model_name} (HQ)")
    req = {
        "prompt": "Elevated cinematic orchestral masterpiece, triumphant horns, sweeping violins, deep percussion, emotional climax, high fidelity",
        "model_name": model_name,
        "duration_seconds": 30,
        "seed": 42
    }
    if model_name == "ace-step":
        req["lyrics"] = "Glory is found in the heart of the storm"
        req["genre"] = "Epic Orchestral"
        req["bpm"] = 120
        
    code, body = http_json("POST", f"{base_url}/api/music/generate", req, timeout=180)
    job_id = body.get("job_id") if isinstance(body, dict) else None
    if code != 200 or not job_id:
        log(f"[MUSIC] FAIL {model_name} submit error: {body}")
        return {"status": "FAIL", "error": body}
    
    log(f"[MUSIC] job_id={job_id} submitted. Waiting...")
    ok, result = wait_for_job(base_url, job_id, timeout_sec=2400)
    if ok:
        res_data = result.get("result", {})
        url = res_data.get("audio_url")
        if url:
            path = os.path.join(OUTPUT_DIR, f"{model_name}_{job_id}.wav")
            download_file(url, path)
        log(f"[MUSIC] PASS OK {model_name}")
        return {"status": "PASS", "details": res_data}
    else:
        log(f"[MUSIC] FAIL {model_name}: {result}")
        return {"status": "FAIL", "error": result}

def test_animation(base_url, model_name):
    log(f"[ANIMATION] >> {model_name} (HQ)")
    req = {
        "model_name": model_name,
        "source_image_b64": FALLBACK_PNG_B64,
        "audio_b64": SILENCE_WAV_B64,
        "expression_scale": 1.5,
        "pose_style": 2,
        "use_enhancer": True
    }
    code, body = http_json("POST", f"{base_url}/api/animation/generate", req, timeout=180)
    job_id = body.get("job_id") if isinstance(body, dict) else None
    if code != 200 or not job_id:
        log(f"[ANIMATION] FAIL {model_name} submit error: {body}")
        return {"status": "FAIL", "error": body}
    
    log(f"[ANIMATION] job_id={job_id} submitted. Waiting...")
    ok, result = wait_for_job(base_url, job_id, timeout_sec=2400)
    if ok:
        res_data = result.get("result", {})
        url = res_data.get("video_url")
        if url:
            path = os.path.join(OUTPUT_DIR, f"{model_name}_{job_id}.mp4")
            download_file(url, path)
        log(f"[ANIMATION] PASS OK {model_name}")
        return {"status": "PASS", "details": res_data}
    else:
        log(f"[ANIMATION] FAIL {model_name}: {result}")
        return {"status": "FAIL", "error": result}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="Base URL of the Neural Creation Studio server (e.g. http://localhost:8080)")
    parser.add_argument("--shutdown", action="store_true", help="Shut down VM after testing")
    args = parser.parse_args()
    base = args.server.rstrip("/")
    
    log(f"Starting FINAL 13-MODEL HQ SMOKE TEST -> {base}")
    
    report = {"summary": {}, "results": {}}
    
    categories = [
        ("IMAGE", MODELS_IMAGE, test_image),
        ("VIDEO", MODELS_VIDEO, test_video),
        ("MUSIC", MODELS_MUSIC, test_music),
        ("ANIMATION", MODELS_ANIMATION, test_animation)
    ]
    
    try:
        for cat_name, models, test_fn in categories:
            log("="*60)
            log(f" Testing {cat_name} Models")
            log("="*60)
            for model in models:
                try:
                    res = test_fn(base, model)
                    report["results"][model] = res
                except Exception as e:
                    log(f"CRITICAL ERROR testing {model}: {e}")
                    report["results"][model] = {"status": "ERROR", "error": str(e)}
    finally:
        # Generate final summary
        passed = [m for m, r in report["results"].items() if r.get("status") == "PASS"]
        failed = [m for m, r in report["results"].items() if r.get("status") in ("FAIL", "ERROR")]
        
        report["summary"] = {
            "total": len(report["results"]),
            "passed_count": len(passed),
            "failed_count": len(failed),
            "passed": passed,
            "failed": failed
        }
        
        log("="*60)
        log(" TEST COMPLETE")
        log(f" Passed: {len(passed)} | Failed: {len(failed)}")
        log("="*60)
        
        with open("full_smoke_test_report.json", "w") as f:
            json.dump(report, f, indent=2)
        log("Detailed report saved to full_smoke_test_report.json")

        if args.shutdown:
            log("INITIATING GCP VM SHUTDOWN (flux-a100-preemptible)...")
            try:
                subprocess.run(["gcloud", "compute", "instances", "stop", "flux-a100-preemptible", "--zone=us-central1-a", "--quiet"], check=True)
                log("✓ Shutdown command sent successfully.")
            except Exception as e:
                log(f"⚠ Shutdown command failed: {e}")

if __name__ == "__main__":
    main()
