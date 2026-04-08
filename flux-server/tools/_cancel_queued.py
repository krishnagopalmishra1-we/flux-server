import urllib.request, json, sys

server = sys.argv[1] if len(sys.argv) > 1 else "http://35.232.245.245:8080"

with urllib.request.urlopen(server + "/api/jobs", timeout=15) as r:
    jobs = json.loads(r.read().decode())["jobs"]

for j in jobs:
    if j["status"] != "queued":
        continue
    jid = j["job_id"]
    model = j["model_name"]
    req = urllib.request.Request(f"{server}/api/jobs/{jid}", method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            result = json.loads(r.read().decode())
            print(f"CANCELLED  {model:20} {jid[:8]}")
    except Exception as e:
        print(f"SKIP       {model:20} {jid[:8]}  {e}")

print("Done.")
