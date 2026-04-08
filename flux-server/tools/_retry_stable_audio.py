"""
Retry stable-audio only
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from test_pending import test_music, log, http_json
import json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://35.232.245.245:8080")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    base = args.server.rstrip("/")
    retries = max(1, args.retries)

    code, health = http_json("GET", f"{base}/health", timeout=60)
    if code != 200:
        raise RuntimeError(f"Health check failed: {code} {health}")
    log(f"Health OK — gpu={health.get('gpu_name','')}  vram={health.get('vram_used_gb',0)}/{health.get('vram_total_gb',0)} GB")

    log("=" * 55)
    log("MUSIC - stable-audio RETRY")
    log("=" * 55)
    result = test_music(base, "stable-audio", retries=retries)
    
    log("=" * 55)
    if result["status"] == "PASS":
        log("✓ PASS")
    else:
        log("✗ FAIL")
    log("=" * 55)
    print(json.dumps(result, indent=2))

    if result["status"] != "PASS":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
