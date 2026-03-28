Run from PowerShell:

1) . .\deploy\gcp\set_env.ps1
2) .\deploy\gcp\deploy.ps1

After VM is up:
1) gcloud compute ssh flux-a100-preemptible --zone us-central1-a
2) cd /opt/flux-server
3) echo "HF_TOKEN=hf_xxx" >> .env
4) docker compose restart

Health endpoint:
http://<EXTERNAL_IP>:8080/health
Models endpoint:
http://<EXTERNAL_IP>:8080/models
UI:
http://<EXTERNAL_IP>:8080/
