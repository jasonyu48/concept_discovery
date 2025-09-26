# TD-MPC2 Docker Setup Guide

## Ultra-Simple Start ⚡

```bash
# 1. Clone repository  
git clone <repository-url>
cd concept_discovery

# 2. Build Docker image (one-time setup)
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) \
            -f Dockerfile-simple -t tdmpc2:simple .

# 3. Create alias (one-time setup)
echo 'alias tdmpc2="docker run --rm -it --gpus all --user $(id -u):$(id -g) \
      -e PYTHONUNBUFFERED=1 -v $(pwd):/workspace -w /workspace tdmpc2:simple python"' >> ~/.bashrc && \
source ~/.bashrc

# 4. Run training with ultra-simple commands (always picks up current code/YAMLs)
tdmpc2 train.py task=counting5 steps=1000
```

That's it! Now you can use `tdmpc2` like a native command!

## Ultra-Simple Commands

### With Alias (Recommended)
```bash
# Default training  
tdmpc2 train.py

# Quick test
tdmpc2 train.py task=counting5 steps=1000

# Long training
tdmpc2 train.py task=cheetah-run steps=300000

# State-based training  
tdmpc2 train.py task=walker-walk obs=state steps=50000

# Evaluation
tdmpc2 evaluate.py checkpoint=logs/counting5/555/.../model.pt
```

### Manual Docker Commands (If No Alias)
```bash
# GPU training
docker run --rm --gpus all --user $(id -u):$(id -g) -v $(pwd):/workspace -w /workspace tdmpc2:simple python train.py

# CPU training  
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace -w /workspace tdmpc2:simple python train.py task=counting5 obs=state
```

## Prerequisites

### All Users
- Docker installed ([installation guide](https://docs.docker.com/get-docker/))
- Git for cloning the repository

### GPU Users (Additional Requirements)
- NVIDIA GPU with compute capability 3.5+
- NVIDIA drivers (version 450.80.02+)
- NVIDIA Container Toolkit

#### Installing NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
# Add NVIDIA's GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add repository
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt update && sudo apt install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Windows WSL2:**
- Install WSL2 with Ubuntu
- Follow Ubuntu instructions above
- Ensure WSL2 has GPU passthrough enabled

## Usage Examples

### Training Tasks
```bash
# Default configuration  
tdmpc2 train.py

# Quick test (counting5 environment)
tdmpc2 train.py task=counting5 steps=1000

# Longer training (Cheetah Run)  
tdmpc2 train.py task=cheetah-run steps=300000

# State-based training (CPU-friendly)
tdmpc2 train.py task=walker-walk obs=state steps=50000

# Custom parameters
tdmpc2 train.py task=cartpole-balance obs=rgb steps=10000 seed=42
```

### Evaluation
```bash
# Evaluate trained models
tdmpc2 evaluate.py checkpoint=logs/counting5/555/.../model.pt
```

## Troubleshooting

### GPU Not Detected
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.4-runtime-ubuntu22.04 nvidia-smi

# If this fails, check:
# 1. NVIDIA drivers installed
# 2. NVIDIA Container Toolkit installed
# 3. Docker daemon restarted after toolkit install
```

### Out of Memory
```bash
# Reduce batch size for smaller GPUs
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
  tdmpc2:simple python train.py task=counting5 batch_size=128 obs=rgb steps=10000
```
