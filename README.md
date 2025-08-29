## Why and How Auxiliary Tasks Improve JEPA Representations

This repository extends TD-MPC2 for research on concept discovery under pixel observations. It focuses on world-model representation learning and analyzing the separability/interpretability of learned “concepts.” On top of TD-MPC2’s scalable world model and MPC planning, we add encoding-space monitoring, collapse prevention, visualization, and analysis utilities to study the process of concept discovery systematically.

What’s included:
- Single-task online training (primarily on a synthetic counting environment)
- Encoding-space/dimensionality monitoring and visualization
- Decoder reconstruction with GIF/JPG exports
- Optional representation collapse prevention (random-phenomenon predictor)

Note: This codebase reuses TD-MPC2’s structure for core training, environments, and algorithms. Defaults, training flow, and monitoring are customized for concept discovery experiments.

### Paper (paste here)
- Title: <paste the official title from Knowledge_Discovery.pdf>
- Abstract: <paste abstract>
- Keywords: <optional>

If you want me to auto-insert the title/abstract from the PDF, share the text or allow me to extract it.

---

## Setup and Installation

We recommend Conda (or Docker) for dependencies:

```bash
conda env create -f docker/environment.yaml
conda activate tdmpc2
```

Notes:
- `docker/environment.yaml` pins PyTorch 2.6 nightly and TorchRL/TensorDict nightly (CUDA 12.4 by default).
- CPU-only runs the counting environment and training but will be slow (GPU ≥ 8GB recommended).

Optional Docker (simple):
```bash
docker build -f Dockerfile-simple -t tdmpc2:simple .
# See CONFIGURATION_GUIDE.md for a convenient alias and usage examples
```

---

## Quickstart

The root `train.py` is a wrapper that calls `tdmpc2/train.py` (Hydra configs live under `tdmpc2/`). The default training config is `concept_discovery_random.yaml`.

```bash
# Option 1: run the wrapper at the repo root (recommended)
python train.py task=counting4 model_size=5

# Option 2: call the real entrypoint directly
python tdmpc2/train.py task=counting4 model_size=5
```

Common arguments (all overridable from CLI):
- `task`: e.g., `countingN` (such as `counting4`).
- `model_size`: capacity hyperparameter set, one of `{1, 5, 19, 48, 317}`.
- `steps`: training steps (default 300k).
- `obs`: observation type, `rgb` for concept discovery.

Outputs:
- Logs under `tdmpc2/logs/<task>/<seed>/<exp_name>` (shared by Hydra and code).
- Periodic eval to `eval.csv`; training snapshot rows to `train.csv`.
- Encoding-space artifacts under `encoding_monitor/`.

### Evaluation

Use `tdmpc2/evaluate.py`. The script’s default config name is `config`, but this repo provides `tdmpc2/tdmpc2.yaml`. Select it via Hydra:

```bash
python tdmpc2/evaluate.py --config-name tdmpc2 \
  task=counting4 checkpoint=/path/to/agent.pt save_video=true
```

Notes:
- Single-task models don’t need explicit `model_size` (default is 5). Set `checkpoint` to your saved weights.

---

## Concept Discovery: Features and Components

- Encoding-space monitoring (`tdmpc2/simple_encoding_space_monitor.py`)
  - Tracks and plots: average pairwise distance in latent space, minimum encoder-Jacobian rank (optional), RankMe (optional; requires `reptrix`), and clustering accuracy (available on the counting env).
  - Periodically writes `monitoring_curves.png` and `monitoring_data.json`.
  - Generates decoder comparison GIFs/JPGs under `decoder_gifs/` at the end of training.

- Collapse prevention
  - Enable via `cfg.collapse_prevention=true`. A frozen random function `_random_fn` and a predictor head `_collapse_pred` are trained with an MSE objective to discourage representation collapse.
  - Random net options: `linear` or `transformer` (see `tdmpc2/common/world_model.py`).

- Reward modeling and optional “current reward” head
  - Standard head: `reward(z, a)` (two-hot discrete regression).
  - Optional current-reward head: `reward_current(z)` trained on environment-provided `reward_pre` (action-independent).

- Planning and policy
  - Retains TD-MPC2’s latent-space MPPI planning (`mpc=true`).
  - Optional training-time “random action selection” to reduce MPPI compute (`random_action_selection=true`).

---

## Counting Environment (CountingObjectsEnv)

Location: `tdmpc2/envs/counting.py`
- Observation: `64×64` RGB (C,H,W). Object positions vary every step; shape/color stay fixed within an episode.
- Action:
  - Continuous (default): scalar in [−1,1] with thresholding to decrement/increment/no-op.
  - Discrete: one-hot (2-action or 3-action: remove/[no-op]/add).
- Episode termination: on target match or step limit (`episode_length`).
- Reward: sparse or dense (`reward_mode`).

Quick test:
```bash
python train.py task=counting4 steps=10000 obs=rgb
```

---

## Configurations and Common Switches

Key configs under `tdmpc2/`:
- `concept_discovery_random.yaml` (default): random action selection + collapse prevention.
- `concept_discovery_rewardonly.yaml`: reward-only supervision; enables `current_reward`.
- `concept_discovery_P_JEPA.yaml`: current reward + JEPA-style stop-gradient control.
- `concept_discovery_dense.yaml`: dense reward version.
- `concept_discovery_no_phenomenon.yaml`: collapse prevention disabled (ablation).

Important hyperparameters (examples):
- Training: `steps, batch_size, lr, eval_freq, seed`
- Task/obs: `task, obs, episodic, discrete_action, two_actions, reward_mode`
- Architecture: `encoder_arch, latent_dim, num_q, dynamics_arch (mlp/iresnet)`
- Collapse prevention: `collapse_prevention, collapse_prevention_network, collapse_prevention_dim, collapse_prevention_coef`
- Current reward: `current_reward, grad_from_current_R`
- Planning: `mpc, iterations, num_samples, horizon, temperature`
- Monitoring: `monitor_freq, dim_monitor_steps, monitor_encoding_space, monitor_cluster_acc, monitor_jacobian_rank, monitor_rankme`

All keys are Hydra-overridable via `key=value` on the CLI.

---

## Visualization and Analysis Scripts

Additional analysis utilities:
- `tdmpc2/plot_encoding_metrics.py`
  - Aggregates estimated latent dimensionality and RankMe across experiments and plots steps–cluster_acc curves.
  - Example:
    ```bash
    python tdmpc2/plot_encoding_metrics.py --seed 2022 --task counting4 \
      --results_root /path/to/tdmpc2/logs
    ```

Monitoring artifacts live under: `tdmpc2/logs/<task>/<seed>/<exp_name>/encoding_monitor/`.

---

## Repro Tips

Configs correspond to ablations in the paper (match to your sections as needed):
- Random actions + collapse prevention: `concept_discovery_random.yaml`
- Reward-only (with current-reward head): `concept_discovery_rewardonly.yaml`
- Current reward + JEPA control: `concept_discovery_P_JEPA.yaml`
- Dense reward: `concept_discovery_dense.yaml`
- No collapse prevention (ablation): `concept_discovery_no_phenomenon.yaml`

Example:
```bash
python train.py --config-name concept_discovery_random task=counting4 model_size=5
```

Note: `tdmpc2/train.py` defaults to `--config-name concept_discovery_random`, but you can set it explicitly.

---

## FAQ

- Config for evaluation
  - `evaluate.py` defaults to config name `config`. This repo provides `tdmpc2.yaml`. Use: `--config-name tdmpc2`.

- RankMe dependency
  - RankMe monitoring requires the `reptrix` package and a saved observation tensor (you can enable `save_obs_for_rankme` and set `rankme_obs_path`). If not installed or missing, RankMe is skipped gracefully.

- Logs directory
  - The program and Hydra share the same layout: `tdmpc2/logs/<task>/<seed>/<exp_name>`. The root `train.py` is just a wrapper and does not change paths.

---

## Acknowledgments and Citation

This project builds on TD-MPC2. If this repository or the TD-MPC2 components are useful for your work, please also cite TD-MPC2:

```text
Hansen, N., Su, H., & Wang, X. TD-MPC2: Scalable, Robust World Models for Continuous Control. ICLR 2024.
```

And include your own paper citation here:

```text
<Your paper BibTeX / citation entry>
```

For the original project and more background, see the TD-MPC2 website (`https://www.tdmpc2.com`).

---

## License

This project is released under the MIT License (see `LICENSE`). Third-party dependencies are subject to their respective licenses.


