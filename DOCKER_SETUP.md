# TD-MPC2 Docker Setup Guide

## Ultra-Simple Start ⚡

```bash
# 1. Clone repository  
git clone <repository-url>
cd concept_discovery

# 2. Build Docker image (one-time setup)
docker build -f Dockerfile-simple -t tdmpc2:simple .

# 3. Create alias (one-time setup) 
echo 'alias tdmpc2="docker run --rm -it --gpus all -e PYTHONUNBUFFERED=1 -v \$(pwd)/tdmpc2/logs:/workspace/tdmpc2/logs tdmpc2:simple python"' >> ~/.bashrc
source ~/.bashrc

# 4. Run training with ultra-simple commands
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
docker run --gpus all -v $(pwd)/tdmpc2/logs:/workspace/tdmpc2/logs tdmpc2:simple python train.py

# CPU training  
docker run -v $(pwd)/tdmpc2/logs:/workspace/tdmpc2/logs tdmpc2:simple python train.py task=counting5 obs=state
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
docker run --rm --gpus all -v $(pwd):/workspace -w /workspace/tdmpc2 \
  tdmpc2:latest python train.py task=counting5 batch_size=128 obs=rgb steps=10000
```

### Slow Performance on CPU
```bash
# Use state observations instead of RGB for CPU training
docker run --rm -v $(pwd):/workspace -w /workspace/tdmpc2 \
  tdmpc2:latest python train.py task=counting5 obs=state steps=10000
```

## Features

✅ **GPU Acceleration**: Automatic RTX/GTX GPU detection and usage  
✅ **RGB Observations**: 64x64 visual observations with software rendering  
✅ **Cross-Platform**: Works on Linux, WSL2, and macOS (CPU mode)  
✅ **Reproducible**: Identical results across different machines  
✅ **Complete Environment**: All dependencies pre-installed  

## Performance Notes

- **GPU Training**: ~10x faster than CPU, supports RGB observations
- **CPU Training**: Slower but works everywhere, recommended with state observations
- **Memory Usage**: ~2-4GB GPU memory for typical tasks
- **Storage**: Docker image ~3GB, logs vary by training length

## Supported Tasks

- `counting5` - Simple counting task (10 steps)
- `cheetah-run` - Continuous control locomotion
- `walker-walk` - Humanoid walking
- `cartpole-balance` - Classic control
- And more from dm_control suite

## Configuration Options

Common parameters:
- `task=<task_name>` - Environment to train on
- `obs=rgb|state` - Observation type (RGB images vs state vectors)
- `steps=<number>` - Training steps
- `seed=<number>` - Random seed for reproducibility

## Output

Training logs and models are saved to:
```
logs/<task>/<seed>/<experiment_name>/
├── eval.csv          # Evaluation metrics
├── train.csv         # Training metrics  
├── model.pt          # Trained model
└── encoding_monitor/ # Representation analysis
```

Access logs from host system after training completes. 