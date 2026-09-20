#!/bin/bash
# Headless deployment payload executed remotely by Google Colab CLI

echo "=================================================="
echo "🚀 Hyperforge AI - Headless Deployment Initializing"
echo "=================================================="

echo "[1/4] Downloading latest codebase..."
rm -rf /content/hyperforge
git clone https://github.com/krishnagopalmishra1-we/flux-server.git /content/hyperforge
cd /content/hyperforge/flux-server

echo "[2/4] Installing dependencies (this takes a minute)..."
pip install -r requirements.txt > /dev/null 2>&1
pip install fastapi uvicorn > /dev/null 2>&1

echo "[3/4] Establishing secure tunnel..."
wget -q -c -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
nohup ./cloudflared-linux-amd64 tunnel --url http://localhost:8080 > /content/cloudflared.log 2>&1 &

echo "[4/4] Starting FastAPI Server & Uncensored FLUX..."
cat << 'EOF' > start_server.py
import os
import time
import re
import subprocess

try:
    from google.colab import userdata
    # Grab the HF token from Colab Secrets safely
    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
except Exception as e:
    print("WARNING: HF_TOKEN not found in Colab Secrets. Model downloads may fail.")

# Start server in background
subprocess.Popen(
    ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"], 
    stdout=open('/content/server.log', 'w'), 
    stderr=subprocess.STDOUT
)

print("Waiting for Cloudflare Tunnel to assign a public URL...")
time.sleep(10)

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
EOF

python start_server.py

echo "Streaming live server logs (Press Ctrl+C to disconnect from logs, server will keep running):"
tail -f /content/server.log
