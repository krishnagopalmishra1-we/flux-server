import os
import json
import codecs
import hashlib

files = [
    "job_detail.json", "jobs_status.json", "jobs_status2.json", 
    "jobs_status3.json", "jobs_status4.json", "jobs_status5.json", 
    "jobs_status6.json"
]

def hash_ip(ip):
    if not ip or ip == "anonymous": return "anonymous"
    return hashlib.sha256(f"app_salt_{ip}".encode()).hexdigest()[:16]

directory = r"d:\Flux_Lora\flux-server"

for f in files:
    path = os.path.join(directory, f)
    if not os.path.exists(path):
        continue
    
    # Read ignoring BOM
    with codecs.open(path, 'r', 'utf-8-sig') as f_in:
        try:
            data = json.load(f_in)
        except json.JSONDecodeError:
            continue
            
    modified = False
    
    if 'jobs' in data:
        for job in data['jobs']:
            if 'user_id' in job and str(job['user_id']).count('.') == 3:
                job['user_id'] = hash_ip(job['user_id'])
                modified = True
            
            # Fix progress inconsistency
            if job.get('status') == 'processing' and job.get('progress') == 0.0 and job.get('processing_time_ms', 0) > 60000:
                job['progress'] = min(99.0, job['processing_time_ms'] / 20000.0)
                modified = True
    else:
        if 'user_id' in data and str(data['user_id']).count('.') == 3:
            data['user_id'] = hash_ip(data['user_id'])
            modified = True
            
        if data.get('status') == 'processing' and data.get('progress') == 0.0 and data.get('processing_time_ms', 0) > 60000:
            data['progress'] = min(99.0, data['processing_time_ms'] / 20000.0)
            modified = True
            
    # Write out WITHOUT BOM unconditionally
    with open(path, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, indent=4)
        
print("Fixtures fixed.")
