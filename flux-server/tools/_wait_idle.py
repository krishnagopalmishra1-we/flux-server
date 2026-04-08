import urllib.request, json, time

SERVER = "http://35.232.245.245:8080"

def fetch_jobs():
    with urllib.request.urlopen(SERVER + "/api/jobs", timeout=15) as r:
        return json.loads(r.read().decode())["jobs"]

def print_jobs(jobs):
    for j in jobs:
        proc = round(j["processing_time_ms"]/1000)
        queue = round(j["queue_time_ms"]/60000, 1)
        print(f'{j["status"]:12} {j["model_name"]:20} proc={proc}s  queue={queue}min')
    print(f"Total: {len(jobs)}")

print("Waiting for queue to go idle...")
for tick in range(300):
    jobs = fetch_jobs()
    active = [j for j in jobs if j["status"] == "processing"]
    if not active:
        print("Queue idle.")
        print_jobs(jobs)
        break
    names = [j["model_name"] for j in active]
    proc = round(active[0]["processing_time_ms"]/1000)
    print(f"[{tick*10}s] processing: {names}  proc={proc}s")
    time.sleep(10)
