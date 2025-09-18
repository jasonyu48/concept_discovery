## Why and How Auxiliary Tasks Improve JEPA Representations (P-JEPA)

This repository contains the reference code for the paper “Why and How Auxiliary Tasks Improve JEPA Representations.” It implements a Joint-Embedding Predictive Architecture with an auxiliary regression head (P-JEPA) and provides a counting environment to reproduce the qualitative/quantitative findings in the paper.

At a glance:
- P-JEPA jointly trains an encoder E, latent dynamics T, and an auxiliary head P on top of latent states.
- Theory: in deterministic MDPs, if both the latent-transition consistency loss and the auxiliary loss reach zero, non‑equivalent observations cannot collapse to the same representation (No Unhealthy Representation Collapse). The auxiliary target determines which distinctions the encoder must preserve.
- Practice: in a counting environment, P-JEPA learns nine distinct latent clusters (for counts 0–8) when the auxiliary target is reward. Alternative auxiliaries (e.g., fixed random function) change which distinctions are preserved.


## Setup

Please follow one of the two supported paths:

- Docker (recommended, reproducible): see `DOCKER_SETUP.md` for a fully containerized workflow.
- Conda (local environment): see `environment_setup.txt`.


## Repository Layout (key files)

- `tdmpc2/train.py`: Hydra entrypoint (default config: `concept_discovery_P-JEPA`).
- `tdmpc2/concept_discovery_*.yaml`: experiment configs used in the paper (P-JEPA, random auxiliary, reward-only, etc.).
- `tdmpc2/envs/counting.py`: 64×64 RGB counting environment used in the experiments.
- `tdmpc2/simple_encoding_space_monitor.py`: produces monitoring curves, t-SNE plots, and decoder reconstructions.
- `tdmpc2/tdmpc2.py`, `tdmpc2/common/*`, `tdmpc2/trainer/*`: agent, training loop, and utilities derived from TD‑MPC2.


## Quickstart (Counting Environment)

All commands below run online training in the counting environment. Hydra organizes outputs under `tdmpc2/logs/${task}/${seed}/${exp_name}`. A protective guard refuses to overwrite an experiment directory if it already contains `eval.csv`; change `exp_name` to start a new run.

General pattern:

```bash
python tdmpc2/train.py --config-name <config_yaml_basename>
```

Notes:
- `task=counting4` in the configs sets the target count n=4; change to `countingk` to target another count.
- `model_size` must be one of `[1, 5, 19, 48, 317]`. We used `5` in our runs.
- We only support online training. There are offline training code inherited from the TDMPC2 implementation, but they may not be compatible with the rest of the code.


### 1) P-JEPA with reward auxiliary (paper Fig. 1 first row)

Produces nine distinct clusters (counts 0–8); reconstructions discard shape/color/position.

```bash
python tdmpc2/train.py --config-name concept_discovery_P_JEPA
```


### 2) P-JEPA with random auxiliary (paper Fig. 1 second row)

Uses a fixed 256‑D random function as the auxiliary. Prevents most collapse but the representation space does not organize by count.

```bash
python tdmpc2/train.py --config-name concept_discovery_random
```


### 3) Reward‑only gradients to encoder (paper Fig. 1 third row)

Encoder only receives reward loss gradients (no latent‑dynamics gradients).

```bash
python tdmpc2/train.py --config-name concept_discovery_rewardonly
```


### Optional: Dense reward ablation

```bash
python tdmpc2/train.py --config-name concept_discovery_dense
```


## Outputs and Monitoring

Under `tdmpc2/logs/${task}/${seed}/${exp_name}` you will find:

- CSV logs: `train.csv`, `eval.csv` (training loss and evaluation reward).
- `encoding_monitor/` which is produced by `SimpleEncodingSpaceMonitor` class:
  - `monitoring_curves.png`: encoding-space size, cluster accuracy, etc.
  - `tsne_clusters.png` and `tsne_clusters_best.png`: tSNE plots of encoding space.
  - `decoder_gifs/decoder_cmp_*_original_observation.jpg` and `..._reconstruction.jpg`: observations from the environment and reconstructions
  - `models/latest_checkpoint.pt` and `models/best_checkpoint.pt` saved periodically.

Tip: If you see a FileExistsError about `eval.csv`, change `exp_name` in your config or delete the old directory.


## Reproducing Paper Figures

Run the three runs above. Each run will produce a `baseline_encodings_best.pt`. Then pass them to `tdmpc2/plot.py` to generate the figures.


## Troubleshooting

- Overwrite protection: change `exp_name` if a previous run wrote `eval.csv` in the same work dir.
- CUDA OOM during monitoring: reduce `monitor_batch_size` in the config.
- WandB disabled by default: set `enable_wandb=true` and fill `wandb_project`, `wandb_entity` if you want remote logging.


## Citation

To preserve double‑blind review, author information is intentionally omitted. A formal citation will be added after the review process.


## Acknowledgements

This codebase builds on the TD‑MPC2 implementation and follows its logging and agent structure, adapted for the counting environment and auxiliary‑task analysis in the paper.


