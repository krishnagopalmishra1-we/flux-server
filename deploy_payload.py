import os
import subprocess
import time
import re
import sys

print("=" * 60)
print("🚀 Hyperforge AI - Headless Deployment")
print("=" * 60)

# ── Step 1: Clone the codebase ──────────────────────────────
print("\n[1/5] Downloading latest codebase...")
subprocess.run("rm -rf /content/hyperforge", shell=True)
subprocess.run("git clone https://github.com/krishnagopalmishra1-we/flux-server.git /content/hyperforge", shell=True, check=True)
os.chdir("/content/hyperforge/flux-server")

# ── Step 2: Install ONLY the missing packages ───────────────
# Colab A100 (Python 3.13, CUDA 12.8) already has:
#   torch 2.11, diffusers 0.40, transformers 5.16, accelerate 1.14,
#   safetensors 0.8, peft 0.20, sentencepiece 0.2.2, fastapi 0.141,
#   uvicorn 0.52, Pillow 11.3, numpy 2.1, scipy 1.16, einops 0.8,
#   huggingface_hub 1.29
#
# Only these are missing (verified via live diagnostic):
MISSING_PACKAGES = [
    "pydantic-settings>=2.4.0",
    "protobuf>=4.25.0",
    "python-multipart>=0.0.9",
]

print("[2/5] Installing missing packages...")
for pkg in MISSING_PACKAGES:
    print(f"  Installing {pkg}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", pkg],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  FAILED: {pkg}")
        print(result.stderr[-2000:])
        sys.exit(1)
    print(f"  ✅ {pkg}")

print("All dependencies ready.")

# ── Step 3: Cloudflare tunnel ───────────────────────────────
print("\n[3/5] Establishing secure tunnel...")
subprocess.run("wget -q -c -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", shell=True)
subprocess.run("chmod +x cloudflared-linux-amd64", shell=True)
subprocess.Popen(
    ["./cloudflared-linux-amd64", "tunnel", "--url", "http://localhost:8080"],
    stdout=open("/content/cloudflared.log", "w"),
    stderr=subprocess.STDOUT,
)

# ── Step 4: Start the server ────────────────────────────────
print("\n[4/5] Starting FastAPI server (FLUX in BF16 on A100)...")
if "HF_TOKEN" not in os.environ:
    print("WARNING: HF_TOKEN not set! Model downloads may fail.")

# A100 has 40-80 GB VRAM — run in full BF16, no quantization needed.
os.environ["FLUX_QUANTIZE"] = "bf16"

subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"],
    stdout=open("/content/server.log", "w"),
    stderr=subprocess.STDOUT,
)

# ── Step 5: Extract the public URL ──────────────────────────
print("\n[5/5] Waiting for Cloudflare tunnel URL...")
time.sleep(15)

try:
    with open("/content/cloudflared.log", "r") as f:
        log_text = f.read()
        url_match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", log_text)
        if url_match:
            print("\n\033[92m" + "=" * 70)
            print("🎉 SUCCESS! Your Image Studio is live on Colab A100.")
            print("URL: \033[94m\033[1m" + url_match.group(0) + "\033[0m")
            print("\033[92m" + "=" * 70 + "\033[0m\n")
        else:
            print("Tunnel not ready yet. Logs:")
            print(log_text[-2000:])
except Exception as e:
    print(f"Error reading tunnel logs: {e}")

print("Streaming server logs (Ctrl+C to disconnect, server keeps running):")
sys.stdout.flush()
subprocess.run("tail -f /content/server.log", shell=True)
