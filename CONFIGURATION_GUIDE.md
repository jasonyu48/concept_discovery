# TD-MPC2 Configuration Guide

## Ultra-Simple Workflow

1. **Build Docker image** (one-time): `docker build -f Dockerfile-simple -t tdmpc2:simple .`
2. **Create alias** (one-time): `echo 'alias tdmpc2="docker run --rm -it --gpus all -e PYTHONUNBUFFERED=1 -v \$(pwd)/tdmpc2/logs:/workspace/tdmpc2/logs tdmpc2:simple python"' >> ~/.bashrc && source ~/.bashrc`
3. **Run training**: `tdmpc2 train.py task=counting5 steps=1000`
4. **Check results**: Logs are saved to `tdmpc2/logs/` on your host system

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
docker run --rm -v $(pwd)/tdmpc2/logs:/workspace/tdmpc2/logs tdmpc2:simple python train.py task=counting5 obs=state
```

### Evaluation
```bash
# Evaluate trained models
tdmpc2 evaluate.py checkpoint=logs/counting5/555/.../model.pt
```

## Available Tasks

| Task | Description | Episode Length | Difficulty |
|------|-------------|----------------|------------|
| `counting5` | Simple counting | 10 steps | Easy |
| `cartpole-balance` | Balance pole | 1000 steps | Easy |
| `cheetah-run` | Locomotion | 500 steps | Medium |
| `walker-walk` | Humanoid walking | 500 steps | Medium |
| `hopper-hop` | Single-leg hopping | 500 steps | Hard |

## Output Structure

Results are saved to:
```
tdmpc2/logs/<task>/<seed>/<experiment_name>/
├── eval.csv                 # Evaluation metrics over time
├── train.csv               # Training metrics  
├── model.pt                # Trained model checkpoint
├── config.yaml             # Experiment configuration
└── encoding_monitor/       # Representation analysis
    ├── monitoring_curves.png
    └── monitoring_data.json
```

## Configuration Methods

### Method 1: Command Line Overrides (Recommended)
```bash
# Override any parameter directly with the alias
tdmpc2 train.py task=cheetah-run steps=100000 batch_size=128
tdmpc2 train.py task=walker-walk obs=state lr=1e-4
tdmpc2 train.py task=counting5 model_size=19
```

### Method 2: Rebuild with Custom Config (Rarely Needed)
```bash
# 1. Modify tdmpc2/concept_discovery.yaml
# 2. Rebuild the image
docker build -f Dockerfile-simple -t tdmpc2:simple .
# 3. Run with new defaults
tdmpc2 train.py
```

> **Tip**: Method 1 (command line overrides) is usually better than rebuilding!

## Tips

- **Start small**: Use `counting5` for testing, then scale to harder tasks
- **GPU memory**: Reduce `batch_size` if you get out-of-memory errors
- **Observation type**: Use `obs=rgb` for vision research, `obs=state` for faster training
- **Reproducibility**: Set `seed: 42` for reproducible results
- **Command line is easiest**: Override parameters directly instead of rebuilding the image

## Troubleshooting

**Training too slow?**
- Use `obs=state` instead of `obs=rgb`
- Reduce `batch_size`
- Use smaller `model_size`

**Want faster evaluation?**
- Increase `eval_freq` (evaluate less often)
- Reduce `eval_episodes` 

## Deleting Logs

Docker creates log files as root, so you need to change ownership before deleting:

```bash
# Change ownership back to your user
sudo chown -R $USER:$USER $HOME/concept_discovery/tdmpc2/logs/

# Then you can delete any specific run
rm -rf $HOME/concept_discovery/tdmpc2/logs/[task_name]/[seed_number]/

# Examples:
rm -rf $HOME/concept_discovery/tdmpc2/logs/counting5/10/
rm -rf $HOME/concept_discovery/tdmpc2/logs/counting5/
```