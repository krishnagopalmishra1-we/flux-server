import urllib.request, json
with urllib.request.urlopen("http://35.232.245.245:8080/api/jobs", timeout=15) as r:
    jobs = json.loads(r.read().decode())["jobs"]
for j in jobs:
    proc = round(j["processing_time_ms"]/1000)
    queue = round(j["queue_time_ms"]/60000, 1)
    print(f'{j["status"]:12} {j["model_name"]:20} proc={proc}s queue={queue}min')
print(f"Total: {len(jobs)}")
