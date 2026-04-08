"""
Smoke test resume from wan-i2v-14b onward (2026-04-08).
Already confirmed PASS: wan-t2v-1.3b, wan-t2v-14b
Remaining: wan-i2v-14b, ltx-video, ace-step, audioldm2, stable-audio, liveportrait, echomimic
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from test_pending import test_video, test_music, test_animation, log, http_json
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

    results = []
    totals = {"pass": 0, "fail": 0}

    def record(r):
        results.append(r)
        if r["status"] == "PASS":
            totals["pass"] += 1
        else:
            totals["fail"] += 1

    log("=" * 55)
    log("VIDEO (remaining)")
    log("=" * 55)
    record(test_video(base, "wan-i2v-14b", is_i2v=True,  retries=retries))
    record(test_video(base, "ltx-video",   is_i2v=False, retries=retries))

    log("=" * 55)
    log("MUSIC")
    log("=" * 55)
    record(test_music(base, "ace-step",     retries=retries))
    record(test_music(base, "audioldm2",    retries=retries))
    record(test_music(base, "stable-audio", retries=retries))

    log("=" * 55)
    log("ANIMATION")
    log("=" * 55)
    record(test_animation(base, "liveportrait", retries=retries))
    record(test_animation(base, "echomimic",    retries=retries))

    log("=" * 55)
    log(f"DONE — PASS: {totals['pass']}  FAIL: {totals['fail']}")
    log("=" * 55)
    print(json.dumps({"server": base, "totals": totals, "results": results}, indent=2))

    if totals["fail"] > 0:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
