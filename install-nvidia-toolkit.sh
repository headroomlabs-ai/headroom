#!/usr/bin/env bash
# Install nvidia-container-toolkit on the host so Docker can pass --gpus all
# into containers. Run this on the Linux GPU host (e.g. 100.77.242.54), NOT
# inside the headroom-internal container. Requires sudo.
#
# Usage:
#   scp install-nvidia-toolkit.sh bauke@100.77.242.54:~
#   ssh bauke@100.77.242.54 'bash ~/install-nvidia-toolkit.sh'
#
# Or paste the block below directly into an interactive ssh session.

set -euo pipefail

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Smoke test the toolkit with NVIDIA's stock CUDA image.
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
