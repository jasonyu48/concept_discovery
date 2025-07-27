# TD-MPC2 Configuration Guide

## Ultra-Simple Workflow

1. **Build Docker image** (one-time): `docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -f Dockerfile-simple -t tdmpc2:simple .`
2. **Create alias** (one-time): `echo 'alias tdmpc2="docker run --rm -it --gpus all --user $(id -u):$(id -g) -e PYTHONUNBUFFERED=1 -v $(pwd):/workspace -w /workspace tdmpc2:simple python"' >> ~/.bashrc && source ~/.bashrc`
3. **Run training**: `tdmpc2 train.py task=counting5 steps=1000`
4. **Check results**: Logs are saved to `tdmpc2/logs/` in your repo (owned by you, no sudo needed)`

That's it! 🎉 Now you can use `tdmpc2` like any other command!

> **Note**: Override any parameter directly on the command line: `tdmpc2 train.py task=cheetah-run steps=50000`

## Key Configuration Options

Edit `tdmpc2/concept_discovery.yaml` to customize your experiment:

### Environment Settings
```yaml
# Choose your task
task: counting5              # counting5, cheetah-run, walker-walk, etc.

# Observation type  
obs: rgb                     # rgb (images) or state (vectors)

# Episode settings
episodic: true               # true for episodic tasks
```

### Training Parameters  
```yaml
# Training duration
steps: 10_000               # Number of training steps

# Learning settings
lr: 3e-4                    # Learning rate
batch_size: 256             # Batch size (reduce if GPU memory issues)

# Evaluation
eval_freq: 3_000           # Evaluate every N steps
eval_episodes: 10          # Number of evaluation episodes
```

### Model Architecture
```yaml
# Model size (affects capacity)
model_size: 5              # Options: 1, 5, 19, 48, 317 (larger = more parameters)
```

### Common Experiment Configurations

#### Fast Testing
```yaml
task: counting5
obs: rgb
steps: 1_000
eval_freq: 500
```

#### Full Training  
```yaml
task: cheetah-run
obs: rgb
steps: 300_000
eval_freq: 10_000
```

#### State-Based Training (CPU-friendly)
```yaml
task: walker-walk
obs: state
steps: 50_000
batch_size: 512
```

## Usage Examples

### Default Training
```bash
# Uses built-in concept_discovery.yaml configuration
tdmpc2 train.py
```

### Override Specific Parameters
```bash
# Override task and steps
tdmpc2 train.py task=cheetah-run steps=50000

# Override multiple parameters  
tdmpc2 train.py task=walker-walk obs=state steps=100000

# Different seeds for reproducibility
tdmpc2 train.py task=counting5 seed=42
```

### CPU Training
```bash
# For CPU-only, modify the alias or use manual command:
docker run --rm --user $(id -u):$(id -g) -v $(pwd):/workspace -w /workspace tdmpc2:simple python train.py task=counting5 obs=state
```