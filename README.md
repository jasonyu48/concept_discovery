## Why and How Auxiliary Tasks Improve JEPA Representations (P-JEPA)

This repository contains the reference code for the paper “Why and How Auxiliary Tasks Improve JEPA Representations.” It implements a practical Joint-Embedding Predictive Architecture with an auxiliary regression head (P-JEPA) and provides a simple counting environment to reproduce the qualitative/quantitative findings in the paper.

At a glance:
- P-JEPA jointly trains an encoder E, latent dynamics T, and an auxiliary head P on top of latent states.
- Theory: in deterministic MDPs, if both the latent-transition consistency loss and the auxiliary loss reach zero, non‑equivalent observations cannot collapse to the same representation (No Unhealthy Representation Collapse). The auxiliary target determines which distinctions the encoder must preserve.
- Practice: in a counting environment, P-JEPA learns nine distinct latent clusters (for counts 0–8) when the auxiliary target is reward. Alternative auxiliaries (e.g., fixed random function) change which distinctions are preserved.


## Setup

Please follow one of the two supported paths:

- Docker (recommended, reproducible): see `DOCKER_SETUP.md` for a fully containerized workflow.
- Conda (local environment): see `environment_setup.txt`.


## Repository Layout (key files)

- `tdmpc2/train.py`: Hydra entrypoint (default config: `concept_discovery_random`).
- `tdmpc2/concept_discovery_*.yaml`: experiment configs used in the paper (P-JEPA, random auxiliary, reward-only, dense-reward, etc.).
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
- Online runs do not use `data_dir`. Offline training is only for multi‑task datasets (`mt30`/`mt80`).


### 1) P-JEPA with reward auxiliary (paper Fig. 1 first row)

Produces nine distinct clusters (counts 0–8); reconstructions discard shape/color/position.

```bash
python tdmpc2/train.py --config-name concept_discovery_P_JEPA
```


### 2) P-JEPA with random auxiliary (paper Fig. 1 second row)

Uses a fixed 256‑D random function as the auxiliary. Prevents most collapse but does not organize by count.

```bash
python tdmpc2/train.py --config-name concept_discovery_random
```


### 3) Reward‑only gradients to encoder (paper Fig. 1 third row)

Encoder only receives reward loss gradients (no latent‑dynamics gradients). Leads to coarse separation.

```bash
python tdmpc2/train.py --config-name concept_discovery_rewardonly
```


### Optional: Dense reward ablation

```bash
python tdmpc2/train.py --config-name concept_discovery_dense model_size=5
```


## Outputs and Monitoring

Under `tdmpc2/logs/${task}/${seed}/${exp_name}` you will find:

- CSV logs: `train.csv`, `eval.csv` (episode reward/success, etc.).
- `encoding_monitor/` from `SimpleEncodingSpaceMonitor`:
  - `monitoring_curves.png`: encoding-space size, Jacobian rank, RankMe, decoder loss, cluster accuracy.
  - `tsne_clusters.png` and `tsne_clusters_best.png` (if labels available; counting env only).
  - `decoder_gifs/decoder_cmp_*_original_observation.jpg` and `..._reconstruction.jpg` (counting env) or GIFs in non-counting envs.
  - `models/latest_checkpoint.pt` and `models/best_checkpoint.pt` saved periodically.

Tip: If you see a FileExistsError about `eval.csv`, change `exp_name` in your config or delete the old directory.


## Reproducing Paper Figures

The three runs above correspond to Fig. 1 rows (a), (b), and (c) respectively. After each run reaches its best cluster compactness (as tracked by the monitor), use the artifacts in `encoding_monitor/` for:
- PCA/2D viz: t-SNE plots are saved automatically; PCA and heatmap can be produced using tdmpc2/plot.py.
- Decoder comparisons: compare `decoder_cmp_*_original_observation.jpg` vs `..._reconstruction.jpg`.


## Troubleshooting

- Overwrite protection: change `exp_name` if a previous run wrote `eval.csv` in the same work dir.
- CUDA OOM during monitoring: reduce `monitor_batch_size` in the config.
- WandB disabled by default: set `enable_wandb=true` and fill `wandb_project`, `wandb_entity` if you want remote logging.


## Citation

To preserve double‑blind review, author information is intentionally omitted. A formal citation will be added after the review process.


## Acknowledgements

This codebase builds on the TD‑MPC2 implementation and follows its logging and agent structure, adapted for the counting environment and auxiliary‑task analysis in the paper.


