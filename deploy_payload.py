import os
import subprocess
import time
import re
import sys

print("==================================================")
print("🚀 Hyperforge AI - Headless Deployment Initializing")
print("==================================================")

print("[1/4] Downloading latest codebase...")
subprocess.run("rm -rf /content/hyperforge", shell=True)
subprocess.run("git clone https://github.com/krishnagopalmishra1-we/flux-server.git /content/hyperforge", shell=True)
os.chdir("/content/hyperforge/flux-server")

print("[2/4] Installing dependencies (this takes a minute)...")
subprocess.run("pip install -r requirements.txt", shell=True, check=True)
subprocess.run("pip install fastapi uvicorn pydantic-settings python-multipart", shell=True, check=True)

print("[3/4] Establishing secure tunnel...")
subprocess.run("wget -q -c -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", shell=True)
subprocess.run("chmod +x cloudflared-linux-amd64", shell=True)
subprocess.Popen(["./cloudflared-linux-amd64", "tunnel", "--url", "http://localhost:8080"], stdout=open("/content/cloudflared.log", "w"), stderr=subprocess.STDOUT)

print("[4/4] Starting FastAPI Server & Uncensored FLUX...")
if 'HF_TOKEN' not in os.environ:
    print("WARNING: HF_TOKEN not injected! Model downloads may fail.")

# Start server in background
subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"], 
    stdout=open('/content/server.log', 'w'), 
    stderr=subprocess.STDOUT
)

print("Waiting for Cloudflare Tunnel to assign a public URL...")
time.sleep(15)

try:
    with open("/content/cloudflared.log", "r") as f:
        log_text = f.read()
        url_match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", log_text)
        if url_match:
            print("\n\033[92m" + "="*70)
            print("🎉 SUCCESS! Your Uncensored Image Studio is running remotely on Colab A100.")
            print("Click this link to open the UI: \033[94m\033[1m" + url_match.group(0) + "\033[0m")
            print("\033[92m" + "="*70 + "\033[0m\n")
        else:
            print("Could not find the URL. Tunnel Logs:")
            print(log_text)
except Exception as e:
    print(f"Error reading logs: {e}")

print("Streaming live server logs (Press Ctrl+C to disconnect from logs, server will keep running):")
sys.stdout.flush()
subprocess.run("tail -f /content/server.log", shell=True)
