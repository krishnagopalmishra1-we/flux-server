#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install -y docker.io docker-compose-plugin git ubuntu-drivers-common
ubuntu-drivers install || true
apt-get install -y nvidia-container-toolkit || true
nvidia-ctk runtime configure --runtime=docker || true
systemctl enable docker
systemctl start docker

mkdir -p /mnt/hf-cache
chown -R 1000:1000 /mnt/hf-cache || true

mkdir -p /opt/flux-server
chown -R $USER:$USER /opt/flux-server || true
