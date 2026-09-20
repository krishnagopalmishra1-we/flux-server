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

# ── Step 3: Verify app imports BEFORE starting server ───────
print("\n[3/5] Verifying application imports...")
sys.path.insert(0, "/content/hyperforge/flux-server")
os.environ["FLUX_QUANTIZE"] = "bf16"

import_tests = [
    "from app.config import get_settings",
    "from app.schemas import GenerateRequest",
    "from app.security import verify_api_key",
    "from app.runtime import gpu_runtime",
    "from app.output_store import output_store",
]
for test in import_tests:
    try:
        exec(test)
        print(f"  ✅ {test}")
    except Exception as e:
        print(f"  ❌ {test}")
        print(f"     Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Test the heavy imports separately with full traceback
try:
    from app.model_manager import MultiModelManager
    print("  ✅ from app.model_manager import MultiModelManager")
except Exception as e:
    print(f"  ❌ from app.model_manager import MultiModelManager")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from app.pipeline import inference_pipeline
    print("  ✅ from app.pipeline import inference_pipeline")
except Exception as e:
    print(f"  ❌ from app.pipeline import inference_pipeline")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from app.main import app
    print("  ✅ from app.main import app")
except Exception as e:
    print(f"  ❌ from app.main import app")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("All imports verified!")

# ── Step 4: Cloudflare tunnel ───────────────────────────────
print("\n[4/5] Establishing secure tunnel...")
subprocess.run("wget -q -c -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", shell=True)
subprocess.run("chmod +x cloudflared-linux-amd64", shell=True)
subprocess.Popen(
    ["./cloudflared-linux-amd64", "tunnel", "--url", "http://localhost:8080"],
    stdout=open("/content/cloudflared.log", "w"),
    stderr=subprocess.STDOUT,
)

# ── Step 5: Start server and verify it boots ────────────────
print("\n[5/5] Starting FastAPI server (FLUX in BF16 on A100)...")
if "HF_TOKEN" not in os.environ:
    print("WARNING: HF_TOKEN not set! Model downloads may fail.")

server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"],
    stdout=open("/content/server.log", "w"),
    stderr=subprocess.STDOUT,
)

# Wait and check if server started successfully
print("Waiting for server to start...")
for i in range(30):
    time.sleep(2)
    # Check if process died
    if server_proc.poll() is not None:
        print(f"\n❌ Server process DIED with exit code {server_proc.returncode}")
        print("=== SERVER LOG ===")
        with open("/content/server.log", "r") as f:
            print(f.read())
        sys.exit(1)
    
    # Check if port is listening
    port_check = subprocess.run("ss -tlnp | grep 8080", shell=True, capture_output=True, text=True)
    if "8080" in port_check.stdout:
        print(f"Server is listening on port 8080 (after {(i+1)*2}s)")
        break
    print(f"  ...waiting ({(i+1)*2}s)")
else:
    print("\n⚠️ Server still starting after 60s. Printing logs so far:")
    with open("/content/server.log", "r") as f:
        print(f.read())

# Extract the Cloudflare URL
time.sleep(5)
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
            print("Could not find tunnel URL. Tunnel logs:")
            print(log_text[-2000:])
except Exception as e:
    print(f"Error reading tunnel logs: {e}")

# Print server logs (NOT tail -f, just dump what we have)
print("\n=== SERVER LOG ===")
with open("/content/server.log", "r") as f:
    print(f.read()[-5000:])

print("\n✅ Deployment complete. Server running in background.")
