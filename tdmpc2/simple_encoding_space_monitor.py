# Simplified Encoder Space Monitoring System
# Author: Based on your requirements
# Purpose: Monitor encoding space size and changes over time (simplified version)

import torch
import torch.nn.functional as F
from torch.func import jacrev, vmap
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import time
import imageio
import json
import pandas as pd
import matplotlib.cm as cm
from PIL import Image

# RankMe metric
try:
    from reptrix import rankme
except ImportError:
    rankme = None  # Will raise later if used without installation

class SimpleEncodingSpaceMonitor:
    """
    Simplified encoder monitoring - only tracks encoding space size
    """
    
    def __init__(
        self, 
        cfg, 
        encoder: torch.nn.Module,
        env,
        device: str = "cuda",
        save_dir: Optional[str] = None,
        agent = None,
        buffer = None,
    ):
        self.cfg = cfg
        self.encoder = encoder
        self.env = env
        self.device = device
        self.agent = agent
        self.buffer = buffer  # replay buffer reference (may be None for offline use)
        self.save_dir = Path(save_dir) if save_dir else Path("simple_encoding_logs")
        self.save_dir.mkdir(exist_ok=True)
        # Store top and first-layer encodings for baseline snapshots
        self.first_layer_baseline_encodings = None
        # Commutative encoders monitoring caches (E0 and two selected Ei)
        self.e0_baseline = None
        self.ei_baselines = None
        self.comm_indices = getattr(self.cfg, 'comm_indices_for_monitor', None)
        # Initialize commutative monitoring attributes to avoid attribute errors
        self.e0_baseline = None
        self.ei_baselines = None
        self.comm_indices = getattr(self.cfg, 'comm_indices_for_monitor', None)
        
        # Unified batch size for all monitoring-related computations (encoding, distances, RankMe, etc.)
        self.monitor_batch_size = getattr(self.cfg, 'monitor_batch_size', 1024)  # <------------ try to decrease this if not enough memory
        
        # Monitoring configuration
        self.num_seed_obs = 256  # Number of diverse seed observations
        self.monitor_freq = self.cfg.get('monitor_freq', 2000)  # Monitor every N steps
        # Dimension monitoring frequency & sample size
        self.dim_monitor_steps = getattr(self.cfg, 'dim_monitor_steps', 5000)
        self.enable_dim_monitor = True
        self.seeds = list(range(42, 42 + self.num_seed_obs))  # Fixed seeds for reproducibility
        
        # Metric enable/disable flags (default all enabled)
        self.enable_encoding_space = getattr(self.cfg, 'monitor_encoding_space', True)
        self.enable_jacobian_rank = getattr(self.cfg, 'monitor_jacobian_rank', True) 
        self.enable_rankme = getattr(self.cfg, 'monitor_rankme', True)
        self.enable_eval_reward = getattr(self.cfg, 'monitor_eval_reward', True)
        
        # Simple storage for monitoring data
        self.monitoring_data = {
            'steps': [],
            'timestamp': []
        }
        
        # Pre-allocate lists for dimension monitoring if enabled
        if self.enable_dim_monitor:
            self.monitoring_data['avg_dis_z'] = []
            self.monitoring_data['min_dis_z'] = []
            self.monitoring_data['avg_dis_p'] = []  # distances in Q-average space
            self.monitoring_data['min_dis_p'] = []
            self.monitoring_data['est_dim_avg'] = []
            self.monitoring_data['est_dim_min'] = []
            self.monitoring_data['est_dim_p_avg'] = []
            self.monitoring_data['est_dim_p_min'] = []
        
        # -----------------------------------------------------------------
        # Pre-sample probe actions (fixed throughout run) for q-distance
        # -----------------------------------------------------------------
        self.M_actions = getattr(self.cfg, 'dim_probe_actions', 64)
        torch.manual_seed(1234)
        probe_actions = []
        for _ in range(self.M_actions):
            a = self.env.rand_act()
            if isinstance(a, torch.Tensor):
                a = a.clone().detach().cpu().float()
            else:
                a = torch.from_numpy(a).float()
            probe_actions.append(a)
        self._probe_actions = torch.stack(probe_actions, dim=0)  # (M, A)
        torch.manual_seed(self.cfg.seed)
        
        # Add metric storage based on enabled flags
        if self.enable_encoding_space:
            self.monitoring_data['encoding_space_size'] = []
            self.monitoring_data['encoding_space_size_first'] = []
        if self.enable_jacobian_rank:
            self.monitoring_data['min_jacobian_rank'] = []
        if self.enable_rankme:
            self.monitoring_data['rankme'] = []
        self.monitoring_data['lipschitz_K'] = []
        self.enable_decoder_loss = getattr(self.cfg, 'enable_decoder', True) and self.cfg.obs == 'rgb'
        if self.enable_decoder_loss:
            self.monitoring_data['decoder_loss'] = []

        # Clustering accuracy (optional, only for counting envs where labels are known)
        self.enable_cluster_acc = getattr(self.cfg, 'monitor_cluster_acc', True)
        if self.enable_cluster_acc:
            self.monitoring_data['cluster_acc'] = []              # top layer (backward-compat)
            self.monitoring_data['cluster_acc_first'] = []        # layer 1
            self.monitoring_data['cluster_acc_mid'] = []          # middle layer
            self.monitoring_data['cluster_acc_top'] = []          # top layer
            # Track the best cluster accuracy encountered so far for saving the best t-SNE plot
            self.best_cluster_acc = float('-inf')
            self.best_cluster_acc_first = float('-inf')
        
        # Storage for per-count action distributions (for continuous 1-D case)
        self.monitoring_data['action_dist_plot_saved'] = []
        
        # -------------------------------------------------------------
        # RankMe setup (needed for baseline observation sampling)
        # -------------------------------------------------------------
        self.rankme_samples = getattr(self.cfg, 'rankme_samples', 30000)
        default_rankme_path = f"/scratch//obs_data/{getattr(self.cfg, 'task', 'unknown')}/obs/observations.pt"
        self.rankme_obs_path = Path(getattr(self.cfg, 'rankme_obs_path', default_rankme_path))
        self._rankme_observations = None  # Lazy loaded
        
        # Sample baseline observations from saved data
        self.baseline_observations = None
        self.baseline_encodings = None ###
        self.baseline_labels = None  # ground-truth object counts when available
        if hasattr(self.env, 'count'):
            self._generate_baseline_observations_fallback()
        else:
            self._sample_baseline_observations()
        
        print(f"🔍 SimpleEncodingSpaceMonitor initialized:")
        print(f"   - Monitoring frequency: every {self.monitor_freq} steps")
        print(f"   - Seed observations: {self.num_seed_obs}")
        print(f"   - Save directory: {self.save_dir}")
        
        # Compute initial baseline encodings
        self._update_baseline_encodings()
        
        # Compute and store initial metrics based on enabled flags
        self.monitoring_data['steps'].append(0)
        self.monitoring_data['timestamp'].append(time.time())
        
        if self.enable_encoding_space:
            initial_space_size = self.pairwise_distance(self.baseline_encodings).mean().item()
            print(f"📏 Initial encoding space size: {initial_space_size:.3e}")
            self.monitoring_data['encoding_space_size'].append(initial_space_size)
            # First-layer encoding space size
            # Use cached first-layer encodings when available
            if self.first_layer_baseline_encodings is not None:
                initial_space_size_first = self.pairwise_distance(self.first_layer_baseline_encodings).mean().item()
                self.monitoring_data['encoding_space_size_first'].append(initial_space_size_first)
            else:
                self.monitoring_data['encoding_space_size_first'].append(None)

        if self.enable_jacobian_rank:
            initial_min_rank = self.compute_min_jacobian_rank(self.baseline_observations)
            print(f"🔢 Initial min rank of encoder Jacobian: {initial_min_rank}")
            self.monitoring_data['min_jacobian_rank'].append(initial_min_rank)

        if self.enable_rankme:
            if rankme is not None and self.rankme_obs_path.exists():
                try:
                    initial_rankme = self.compute_rankme_metric_for_model(self.encoder)
                    print(f"🔢 Initial RankMe: {initial_rankme:.2f}")
                    self.monitoring_data['rankme'].append(initial_rankme)
                except Exception as e:
                    print(f"⚠️ Failed to compute initial RankMe: {e}")
                    self.monitoring_data['rankme'].append(None)
            else:
                print("⚠️ RankMe observations not found or reptrix not installed; RankMe metric disabled.")
                self.monitoring_data['rankme'].append(None)

        # Initial decoder evaluation loss
        if self.enable_decoder_loss:
            try:
                init_dec_loss = self._compute_decoder_eval_loss()
                if init_dec_loss is not None:
                    print(f"📉 Initial decoder evaluation loss: {init_dec_loss:.6f}")
                else:
                    print("📉 Initial decoder evaluation loss: N/A")
                self.monitoring_data['decoder_loss'].append(init_dec_loss)
            except Exception as e:
                print(f"⚠️ Failed to compute initial decoder evaluation loss: {e}")
                self.monitoring_data['decoder_loss'].append(None)
        
        # Initial cluster accuracy (if labels are available)
        if self.enable_cluster_acc:
            if self.baseline_labels is not None:
                try:
                    # Top encoder accuracy (uses self.baseline_encodings which is top-level)
                    initial_cluster_acc = self._compute_cluster_accuracy(self.baseline_encodings, self.baseline_labels)
                    print(f"🎯 Initial cluster accuracy (top): {initial_cluster_acc*100:.2f}%")
                    self.monitoring_data['cluster_acc'].append(initial_cluster_acc)
                    # Placeholders for per-layer series to keep list lengths aligned
                    self.monitoring_data['cluster_acc_first'].append(None)
                    self.monitoring_data['cluster_acc_mid'].append(None)
                    self.monitoring_data['cluster_acc_top'].append(initial_cluster_acc)
                except Exception as e:
                    print(f"⚠️ Failed to compute initial cluster accuracy: {e}")
                    self.monitoring_data['cluster_acc'].append(None)
                    self.monitoring_data['cluster_acc_first'].append(None)
                    self.monitoring_data['cluster_acc_mid'].append(None)
                    self.monitoring_data['cluster_acc_top'].append(None)
            else:
                print("🎯 Initial cluster accuracy: N/A (no labels available)")
                self.monitoring_data['cluster_acc'].append(None)
                self.monitoring_data['cluster_acc_first'].append(None)
                self.monitoring_data['cluster_acc_mid'].append(None)
                self.monitoring_data['cluster_acc_top'].append(None)
        
        # --- Collapse prevention: measure initial random function magnitude ---
        if getattr(self.cfg, 'collapse_prevention', False) and hasattr(self.agent.model, '_random_fn') and self.agent.model._random_fn is not None:
            with torch.no_grad():
                obs = self.baseline_observations.to(self.device)
                random_out = self.agent.model._random_fn(obs)
                initial_rand_mag = random_out.abs().mean().item()
                random_dist = self.pairwise_distance(random_out).min().item()
                random_rank = self.compute_min_jacobian_rank(self.baseline_observations, model=self.agent.model._random_fn)
                
                # Compute initial RankMe for random function if enabled
                initial_random_rankme = None
                if self.enable_rankme and rankme is not None and self.rankme_obs_path.exists():
                    try:
                        initial_random_rankme = self.compute_rankme_metric_for_model(self.agent.model._random_fn)
                        print(f"🎲 Initial random function RankMe: {initial_random_rankme:.2f}")
                    except Exception as e:
                        print(f"⚠️ Failed to compute initial random function RankMe: {e}")
                
                self.monitoring_data.setdefault('random_fn_magnitude', []).append(initial_rand_mag)
                self.monitoring_data.setdefault('random_fn_min_distance', []).append(random_dist)
                self.monitoring_data.setdefault('random_fn_min_rank', []).append(random_rank)
                self.monitoring_data.setdefault('random_fn_rankme', []).append(initial_random_rankme)
                print(f"🎲 Initial random function |output| mean: {initial_rand_mag:.6f}")
                print(f"🎲 Initial random function min pairwise distance: {random_dist:.6f}")
                print(f"🎲 Initial min rank of random function Jacobian: {random_rank}")
        else:
            initial_rand_mag = None
        
        # Pairwise distance computation parameters (memory-friendly)
        self.pd_max_samples = getattr(self.cfg, 'dim_pd_max_samples', 1000000)
        
    def _sample_baseline_observations(self):
        """Sample baseline observations from saved observation data using cfg.seed"""
        print("🌱 Sampling baseline observations from saved data...")
        
        try:
            # Save RNG states to avoid affecting global randomness
            torch_cpu_state = torch.get_rng_state()
            torch_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            np_state = np.random.get_state()

            # Load saved observations
            if not self.rankme_obs_path.exists():
                raise FileNotFoundError(f"Saved observations not found: {self.rankme_obs_path}")
            
            obs_tensor = torch.load(self.rankme_obs_path, map_location='cpu')
            print(f"   Loaded saved observations: {obs_tensor.shape}")
            
            # Set seed for reproducible sampling
            torch.manual_seed(43)
            np.random.seed(43)
            
            # Sample 64 observations
            total_obs = obs_tensor.shape[0]
            if total_obs < self.num_seed_obs:
                print(f"⚠️ Warning: Only {total_obs} saved observations available, using all")
                indices = torch.arange(total_obs)
            else:
                indices = torch.randperm(total_obs)[:self.num_seed_obs]
            
            sampled_obs = obs_tensor[indices].to(self.device).float()
            self.baseline_observations = sampled_obs
            # Try to infer labels from env if it has 'count' attribute
            if hasattr(self.env, 'count'):
                # We assume env.count holds the current count at sampling time
                # However, saved observations lack labels; so default to None.
                self.baseline_labels = None
            
            print(f"✅ Sampled {len(indices)} baseline observations")
            print(f"   Final shape: {self.baseline_observations.shape}")

            # Restore original RNG states so training randomness is unchanged
            torch.set_rng_state(torch_cpu_state)
            if torch_cuda_state is not None:
                torch.cuda.set_rng_state_all(torch_cuda_state)
            np.random.set_state(np_state)
            
        except Exception as e:
            print(f"⚠️ Failed to sample from saved observations: {e}")
            print("   Falling back to environment-based observation generation...")
            self._generate_baseline_observations_fallback()
    
    def _generate_baseline_observations_fallback(self):
        """Fallback method to generate observations from environment if saved data unavailable"""
        print("🌱 Generating baseline observations from environment (fallback)...")
        
        observations = []
        labels = []
        
        with torch.no_grad():
            for i, seed in enumerate(self.seeds):
                try:
                    # Reset environment with different seed
                    if hasattr(self.env, 'seed'):
                        self.env.seed(seed)
                    
                    # Reset and take a few steps
                    obs = self.env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]  # Handle new gym API
                    
                    # Take 3-5 random actions to get diverse frames
                    num_actions = np.random.randint(3, 6)
                    for step in range(num_actions):
                        action = self.env.action_space.sample()
                        action = torch.tensor(action)
                        obs, _, _, _ = self.env.step(action)
                    
                    observations.append(obs)
                    # Record ground truth count if env provides attribute
                    if hasattr(self.env, 'count'):
                        labels.append(int(self.env.count))
                        
                except Exception as e:
                    print(f"⚠️ Warning: Failed to generate observation with seed {seed}: {e}")
                    continue
        
        # Convert to tensor
        tensor_observations = []
        for obs in observations:
            try:
                if isinstance(obs, torch.Tensor):
                    tensor_obs = obs.to(self.device).float()
                elif isinstance(obs, np.ndarray):
                    tensor_obs = torch.from_numpy(obs.copy()).to(self.device).float()
                else:
                    tensor_obs = torch.from_numpy(np.array(obs)).to(self.device).float()
                tensor_observations.append(tensor_obs)
            except Exception as e:
                print(f"   Warning: Failed to convert observation: {e}")
                continue
        
        if len(tensor_observations) == 0:
            raise RuntimeError("No valid observations generated!")
            
        self.baseline_observations = torch.stack(tensor_observations)
        if labels:
            self.baseline_labels = torch.tensor(labels)
        print(f"✅ Generated {len(observations)} fallback baseline observations")
        print(f"   Final shape: {self.baseline_observations.shape}")
        
        
    def _update_baseline_encodings(self):
        """Update baseline encodings.
        Computes top and first-layer encodings in one pass using encode_all_layers when available.
        """
        with torch.no_grad():
            if self.cfg.multitask:
                raise NotImplementedError("monitor not implemented for multi-task encoders")
            # Prefer encode_all_layers from the agent model to get first/top at once
            if hasattr(self.agent, 'model') and hasattr(self.agent.model, 'encode_all_layers'):
                print(f"🔍 Updating baseline encodings (first & top) via encode_all_layers")
                z_layers = self.agent.model.encode_all_layers(self.baseline_observations.to(self.device), task=None)
                self._last_z_layers = z_layers
                self.first_layer_baseline_encodings = z_layers[0]
                self.baseline_encodings = z_layers[-1]
                # Also refresh commutative encoder baselines on every update, if configured
                if hasattr(self.agent.model, 'encode_comm_all') and int(getattr(self.agent.model, 'num_commutative_encoders', 0)) > 0:
                    e0 = self.agent.model.encode_e0(self.baseline_observations.to(self.device), task=None)
                    z_comm = self.agent.model.encode_comm_all(self.baseline_observations.to(self.device), task=None)
                    N = len(z_comm)
                    if self.comm_indices is None or not isinstance(self.comm_indices, (list, tuple)) or len(self.comm_indices) != 2:
                        idx_a, idx_b = 0, max(N-1, 0)
                    else:
                        idx_a = max(0, min(N-1, int(self.comm_indices[0])))
                        idx_b = max(0, min(N-1, int(self.comm_indices[1])))
                    self.e0_baseline = e0
                    self.ei_baselines = [z_comm[idx_a], z_comm[idx_b]]
            else:
                # Fallback: use provided encoder (top) only
                print(f"🔍 Updating baseline encodings with provided encoder (top only)")
                self.baseline_encodings = self.encoder(self.baseline_observations)
                self._last_z_layers = None
                self.e0_baseline = None
                self.ei_baselines = None
    
    def pairwise_distance(self, encodings: torch.Tensor) -> torch.Tensor:
        """Compute average pairwise distance in encoding space"""
        # Efficient pairwise distance computation
        # ||e_i - e_j||^2 = ||e_i||^2 + ||e_j||^2 - 2*e_i·e_j
        G = torch.mm(encodings, encodings.T)
        squared_norms = G.diag()
        
        # Distance matrix
        D2 = squared_norms.unsqueeze(1) + squared_norms.unsqueeze(0) - 2 * G
        
        # Extract upper triangle (avoid diagonal and duplicates)
        triu_indices = torch.triu_indices(D2.shape[0], D2.shape[1], offset=1)
        distances = torch.sqrt(torch.clamp(D2[triu_indices[0], triu_indices[1]], min=0))
        
        return distances
    
    def compute_min_jacobian_rank(self, observations: torch.Tensor, model: Optional[torch.nn.Module] = None) -> int:
        """Compute the minimum rank of the Jacobian dF/dx over a batch.

        Args:
            observations: input batch (B, ...)
            model: network to analyse. If None, defaults to the agent's encoder
                   (the same behaviour as before).
        Returns:
            Minimum matrix rank across the batch.
        """
        target_model = model if model is not None else self.encoder

        # Ensure the tensor has gradients enabled
        obs = observations.detach().clone().to(self.device).requires_grad_(True)

        # Define a helper that returns output for a *single* sample
        if model is None and hasattr(self.encoder, 'encode'):
            # original path for agent encoder (multi-task case)
            def single_forward(x_single: torch.Tensor):
                task = torch.zeros(1, self.cfg.task_dim, device=self.device)
                return self.encoder.encode(x_single.unsqueeze(0), task).squeeze(0)
        else:
            def single_forward(x_single: torch.Tensor):
                return target_model(x_single.unsqueeze(0)).squeeze(0)

        jac_single = jacrev(single_forward)
        # Compute Jacobians in smaller chunks to avoid GPU OOM
        batch_size_jac = 8  # number of samples per sub-batch  <------------ try to decrease this if not enough memory
        min_rank_val = float('inf')

        for start in range(0, obs.shape[0], batch_size_jac):
            obs_b = obs[start:start + batch_size_jac]
            # vmap over the smaller batch
            jac_b = vmap(jac_single, randomness="same")(obs_b)
            jac_b = jac_b.flatten(start_dim=2)

            # Compute matrix ranks for this sub-batch
            ranks_b = torch.linalg.matrix_rank(jac_b)
            batch_min = ranks_b.min().item()
            if batch_min < min_rank_val:
                min_rank_val = batch_min

        return int(min_rank_val)
    
    # -------------------------------------------------------------
    # Decoder evaluation helper
    # -------------------------------------------------------------
    def _compute_decoder_eval_loss(self) -> Optional[float]:
        """Compute the decoder reconstruction MSE on the baseline observations."""
        if not self.enable_decoder_loss:
            return None

        decoder = getattr(self.agent.model, '_decoder', None)
        if decoder is None:
            return None

        obs = self.baseline_observations.to(self.device)
        z = self.baseline_encodings.to(self.device)

        prev_mode = decoder.training
        decoder.eval()

        batch_size = self.monitor_batch_size
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for start in range(0, z.shape[0], batch_size):
                z_b = z[start:start + batch_size]
                pred = decoder(z_b)
                pred = pred.reshape_as(obs[start:start + batch_size])
                target = (obs[start:start + batch_size].float() / 255.0 - 0.5) * 2
                batch_loss = F.mse_loss(pred, target, reduction='mean').item()
                total_loss += batch_loss * z_b.size(0)
                total_samples += z_b.size(0)

        if prev_mode:
            decoder.train()

        if total_samples == 0:
            return float('nan')
        return total_loss / total_samples
    
    def monitor_step(self, step: int) -> Dict[str, float]:
        """
        Perform monitoring at given training step - simplified version
        
        Returns:
            Dictionary with encoding space size only
        """
        # Determine whether to run monitoring at this step
        if (step % self.monitor_freq != 0) and (step % self.dim_monitor_steps != 0):
            # Neither encoding nor dimension monitoring is scheduled
            return {}
        
        self.agent.model.eval()
        self.encoder.eval()
        
        # Update encodings with current encoder
        self._update_baseline_encodings()
        
        # Store results based on enabled metrics
        metrics = {
            'step': step,
        }
        # Track computed cluster accuracy for later use (e.g. in t-SNE plotting)
        cluster_acc_computed = None
        
        # Update monitoring data
        self.monitoring_data['steps'].append(step)
        self.monitoring_data['timestamp'].append(time.time())
        
        # Compute and store enabled metrics
        if self.enable_encoding_space:
            space_size = self.pairwise_distance(self.baseline_encodings).mean().item()
            metrics['encoding_space_size'] = space_size
            self.monitoring_data['encoding_space_size'].append(space_size)
            # First-layer encoding space size for this step
            if self.first_layer_baseline_encodings is not None:
                space_size_first = self.pairwise_distance(self.first_layer_baseline_encodings).mean().item()
            else:
                space_size_first = None
            metrics['encoding_space_size_first'] = space_size_first
            self.monitoring_data['encoding_space_size_first'].append(space_size_first)
            print(f"   Encoding space size: {space_size:.6f}")

        # Clustering accuracy metric (requires labels)
        if self.enable_cluster_acc and self.baseline_labels is not None:
            try:
                # Compute per-layer encodings for first, middle, and top layers
                # Use cached layer encodings from _update_baseline_encodings when available
                if hasattr(self, '_last_z_layers') and self._last_z_layers is not None:
                    z_layers = self._last_z_layers
                else:
                    with torch.no_grad():
                        z_layers = self.agent.model.encode_all_layers(self.baseline_observations.to(self.device), task=None)
                L = len(z_layers)
                idx_first = 0
                idx_mid = L // 2
                idx_top = L - 1
                acc_first = self._compute_cluster_accuracy(z_layers[idx_first], self.baseline_labels)
                acc_mid = self._compute_cluster_accuracy(z_layers[idx_mid], self.baseline_labels)
                acc_top = self._compute_cluster_accuracy(z_layers[idx_top], self.baseline_labels)
                # For backward-compat, keep 'cluster_acc' as top layer
                metrics['cluster_acc'] = acc_top
                self.monitoring_data['cluster_acc'].append(acc_top)
                # Store detailed series
                metrics['cluster_acc_first'] = acc_first
                metrics['cluster_acc_mid'] = acc_mid
                metrics['cluster_acc_top'] = acc_top
                self.monitoring_data['cluster_acc_first'].append(acc_first)
                self.monitoring_data['cluster_acc_mid'].append(acc_mid)
                self.monitoring_data['cluster_acc_top'].append(acc_top)
                print(f"   Cluster accuracy (first/mid/top): {acc_first*100:.2f}% / {acc_mid*100:.2f}% / {acc_top*100:.2f}%")
                cluster_acc_computed = acc_top
            except Exception as e:
                print(f"⚠️ Failed to compute per-layer cluster accuracy: {e}")
                self.monitoring_data['cluster_acc'].append(None)
                self.monitoring_data['cluster_acc_first'].append(None)
                self.monitoring_data['cluster_acc_mid'].append(None)
                self.monitoring_data['cluster_acc_top'].append(None)
            # Commutative encoders: e0 vs two E_i (use cached baselines from _update_baseline_encodings)
            if (
                hasattr(self.agent.model, 'num_commutative_encoders') and
                int(getattr(self.agent.model, 'num_commutative_encoders', 0)) > 0 and
                self.e0_baseline is not None and self.ei_baselines is not None and len(self.ei_baselines) == 2
            ):
                try:
                    acc_e0 = self._compute_cluster_accuracy(self.e0_baseline, self.baseline_labels)
                    acc_ei_a = self._compute_cluster_accuracy(self.ei_baselines[0], self.baseline_labels)
                    acc_ei_b = self._compute_cluster_accuracy(self.ei_baselines[1], self.baseline_labels)
                except Exception as e:
                    print(f"⚠️ Failed to compute commutative cluster accuracies: {e}")
                    acc_e0 = acc_ei_a = acc_ei_b = None
                self.monitoring_data.setdefault('cluster_acc_e0', []).append(acc_e0)
                self.monitoring_data.setdefault('cluster_acc_ei_a', []).append(acc_ei_a)
                self.monitoring_data.setdefault('cluster_acc_ei_b', []).append(acc_ei_b)
            else:
                self.monitoring_data.setdefault('cluster_acc_e0', []).append(None)
                self.monitoring_data.setdefault('cluster_acc_ei_a', []).append(None)
                self.monitoring_data.setdefault('cluster_acc_ei_b', []).append(None)
        elif self.enable_cluster_acc:
            self.monitoring_data['cluster_acc'].append(None)
            self.monitoring_data['cluster_acc_first'].append(None)
            self.monitoring_data['cluster_acc_mid'].append(None)
            self.monitoring_data['cluster_acc_top'].append(None)
            self.monitoring_data.setdefault('cluster_acc_e0', []).append(None)
            self.monitoring_data.setdefault('cluster_acc_ei_a', []).append(None)
            self.monitoring_data.setdefault('cluster_acc_ei_b', []).append(None)
        
        # Commutative encoding space sizes (record at monitor_step time)
        if self.enable_encoding_space:
            if self.e0_baseline is not None and self.ei_baselines is not None and len(self.ei_baselines) == 2:
                try:
                    ess_e0 = self.pairwise_distance(self.e0_baseline).mean().item()
                    ess_ei_a = self.pairwise_distance(self.ei_baselines[0]).mean().item()
                    ess_ei_b = self.pairwise_distance(self.ei_baselines[1]).mean().item()
                except Exception as e:
                    ess_e0 = ess_ei_a = ess_ei_b = float('nan')
                self.monitoring_data.setdefault('encoding_space_size_e0', []).append(ess_e0)
                self.monitoring_data.setdefault('encoding_space_size_ei_a', []).append(ess_ei_a)
                self.monitoring_data.setdefault('encoding_space_size_ei_b', []).append(ess_ei_b)
            else:
                self.monitoring_data.setdefault('encoding_space_size_e0', []).append(None)
                self.monitoring_data.setdefault('encoding_space_size_ei_a', []).append(None)
                self.monitoring_data.setdefault('encoding_space_size_ei_b', []).append(None)

        if self.enable_jacobian_rank:
            min_rank = self.compute_min_jacobian_rank(self.baseline_observations)
            metrics['min_jacobian_rank'] = float(min_rank)
            self.monitoring_data['min_jacobian_rank'].append(min_rank)
            print(f"   Minimum Jacobian rank: {float(min_rank)}")
            
        if self.enable_rankme:
            rankme_metric = None
            if rankme is not None and self.rankme_obs_path.exists():
                try:
                    rankme_metric = self.compute_rankme_metric_for_model(self.encoder)
                    print(f"   RankMe: {rankme_metric:.2f}")
                except Exception as e:
                    print(f"⚠️ Failed to compute RankMe: {e}")
            metrics['rankme'] = rankme_metric
            self.monitoring_data['rankme'].append(rankme_metric)
        
        # Decoder evaluation loss
        if self.enable_decoder_loss:
            if step % self.monitor_freq == 0:
                dec_loss_val = self._compute_decoder_eval_loss()
                metrics['decoder_loss'] = dec_loss_val
                self.monitoring_data['decoder_loss'].append(dec_loss_val)
                if dec_loss_val is not None:
                    print(f"   Decoder eval loss: {dec_loss_val:.6f}")
                else:
                    print("   Decoder eval loss: N/A")
            else:
                # Maintain alignment of monitoring_data lists
                self.monitoring_data['decoder_loss'].append(None)
        
        # -------------------------------------------------------------
        # Dimension monitoring (pairwise latent distance & estimated dim)
        # -------------------------------------------------------------
        if (
            self.enable_dim_monitor
            and getattr(self.cfg, 'monitor_bounds', True)
            and self.buffer is not None
            and (step % self.dim_monitor_steps == 0)
        ):
            dim_metrics = self.compute_full_dim_metrics()
            if dim_metrics:
                metrics.update(dim_metrics)
        
        # Capture previous best before any plotting might update it
        prev_best_cluster_acc = getattr(self, 'best_cluster_acc', float('-inf'))

        # Auto-save periodically and generate updated plots
        if step % (self.monitor_freq * 1) == 0:   # <------------
            self.save_monitoring_data()
            # Generate updated encoding space curve plot
            try:
                self.plot_monitoring_curves(save_plot=True)
                print(f"   ✅ Updated encoding space curve saved!")
            except Exception as e:
                print(f"   ⚠️ Failed to generate encoding space curve: {e}")

            # --- NEW: t-SNE cluster visualisations (top and first layers, if labels available) ---
            if self.enable_cluster_acc and self.baseline_labels is not None:
                try:
                    # Top-layer t-SNE (existing)
                    self._plot_tsne_clusters(save_path=self.save_dir / "tsne_clusters.png", cluster_acc=cluster_acc_computed)
                    # First-layer t-SNE using cached first-layer encodings, with unified function
                    if self.first_layer_baseline_encodings is not None:
                        try:
                            acc_first = self._compute_cluster_accuracy(self.first_layer_baseline_encodings, self.baseline_labels)
                        except Exception:
                            acc_first = None
                        self._plot_tsne_clusters(
                            save_path=self.save_dir / "tsne_clusters_first.png",
                            cluster_acc=acc_first,
                            enc=self.first_layer_baseline_encodings,
                            labels=self.baseline_labels,
                            best_attr_name='best_cluster_acc_first',
                            best_suffix='tsne_clusters_first_best.png',
                        )
                    print("   📐 t-SNE cluster plots updated (top & first if available)!")
                except Exception as e:
                    print(f"   ⚠️ Failed to generate t-SNE plots: {e}")
            
            # --- NEW: Action distribution heatmap when cluster accuracy is high ---
            try:
                if self._should_plot_action_distribution(cluster_acc_computed):
                    self._plot_action_distribution_heatmap(step)
                    self.monitoring_data['action_dist_plot_saved'].append(step)
            except Exception as e:
                print(f"   ⚠️ Failed to generate action distribution heatmap: {e}")
        
        # Save model periodically to same path (every 5 monitoring steps)
        if step % (self.monitor_freq * 5) == 0:
            self._save_model_checkpoint(step)

        # If cluster accuracy improves over previous best, save artifacts immediately
        if cluster_acc_computed is not None:
            try:
                best_cmp = float(cluster_acc_computed)
                if best_cmp > prev_best_cluster_acc:
                    self._on_new_best_cluster_acc(step, best_cmp)
            except Exception as e:
                print(f"   ⚠️ Failed to save artifacts on new best cluster acc: {e}")

        self.agent.model.train()
        self.encoder.train()

        return metrics

    # -------------------------------------------------------------
    # Helper: cluster accuracy for known labels
    # -------------------------------------------------------------
    def _compute_cluster_accuracy(self, encodings: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute simple nearest-centroid classification accuracy."""
        # Move to CPU for numpy ops
        enc = encodings.detach().cpu().numpy()
        labs = labels.detach().cpu().numpy()
        unique_labels = np.unique(labs)
        # Compute centroid per label
        centroids = {l: enc[labs == l].mean(axis=0) for l in unique_labels}
        # Predict label by nearest centroid
        pred = []
        for vec in enc:
            dists = {l: np.linalg.norm(vec - c) for l, c in centroids.items()}
            pred.append(min(dists, key=dists.get))
        pred = np.array(pred)
        acc = (pred == labs).mean()
        return float(acc)
    
    def save_monitoring_data(self):
        """Save monitoring data"""
        json_path = self.save_dir / "monitoring_data.json"
        with open(json_path, "w") as fj:
            json.dump({k: self._to_serializable(v) for k, v in self.monitoring_data.items()}, fj, indent=2)
        print(f"💾 Saved monitoring data to {json_path}")

    def _to_serializable(self, obj):
            """Convert obj to a JSON-serialisable form."""
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            if isinstance(obj, (list, tuple)):
                return [self._to_serializable(o) for o in obj]
            if isinstance(obj, dict):
                return {k: self._to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj.tolist()
            # Fallback: string representation
            return str(obj)
    
    def _save_model_checkpoint(self, step: int):
        """Save model checkpoint to same path (overwrites previous)"""
        try:
            # Create models directory if it doesn't exist
            models_dir = self.save_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Save to fixed path (overwrites previous checkpoint)
            checkpoint_path = models_dir / "latest_checkpoint.pt"
            
            self.agent.save(checkpoint_path)
            print(f"   ✅ Full agent checkpoint saved to {checkpoint_path}")

        except Exception as e:
            print(f"   ⚠️ Failed to save model checkpoint: {e}")

    def _save_best_model_checkpoint(self, step: int):
        """Save/overwrite a BEST checkpoint at a fixed path for cluster accuracy best."""
        try:
            models_dir = self.save_dir / "models"
            models_dir.mkdir(exist_ok=True)
            checkpoint_path = models_dir / "best_checkpoint.pt"
            self.agent.save(checkpoint_path)
            print(f"   🏅 Saved BEST agent checkpoint to {checkpoint_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save BEST model checkpoint: {e}")

    def _save_baseline_encodings_snapshot(self, step: int):
        """Save/overwrite current baseline encodings tensor at a fixed path."""
        try:
            enc_dir = self.save_dir / "baseline_encodings"
            enc_dir.mkdir(exist_ok=True)
            out_path = enc_dir / "baseline_encodings_best.pt"
            payload = {
                'step': int(step),
                'timestamp': float(time.time()),
                'encodings': self.baseline_encodings.detach().cpu(),
                'labels': self.baseline_labels.detach().cpu() if self.baseline_labels is not None else None,
            }
            torch.save(payload, out_path)
            print(f"   💾 Saved baseline encodings snapshot to {out_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save baseline encodings snapshot: {e}")

    def _on_new_best_cluster_acc(self, step: int, acc: float):
        """Handle actions when a new best cluster accuracy is achieved."""
        print(f"   🏆 New BEST cluster accuracy {acc*100:.2f}% at step {step} — saving artifacts...")
        # Save decoder GIFs (if decoder enabled)
        try:
            # Use fixed suffix 'best' so new best overwrites previous best GIFs
            self.save_decoder_gifs(num_gifs=getattr(self.cfg, 'num_best_gifs', 5), step="best")
        except Exception as e:
            print(f"   ⚠️ Failed to save decoder GIFs on best: {e}")

        # Save baseline encodings snapshot
        self._save_baseline_encodings_snapshot(step)

        # Save a best-tagged model checkpoint
        self._save_best_model_checkpoint(step)

        # Update in-memory best
        self.best_cluster_acc = float(acc)

    
    def plot_monitoring_curves(self, save_plot: bool = True):
        """Generate plots: encoding size, min rank, RankMe, eval reward, cluster accuracy, train losses."""
        if not self.monitoring_data['steps']:
            print("No monitoring data to plot")
            return None
        
        plt.figure(figsize=(14, 10))

        steps = np.array(self.monitoring_data['steps'])
        
        # Attempt to preload training losses for inclusion in main grid
        train_losses_df = None
        include_train_losses = False
        try:
            train_csv = f"{self.cfg.work_dir}/train.csv"
            df_train_try = pd.read_csv(train_csv)
            if {'step', 'reward_loss', 'consistency_loss'}.issubset(df_train_try.columns):
                train_losses_df = df_train_try
                include_train_losses = True
        except Exception:
            include_train_losses = False

        # Count enabled metrics to determine subplot layout
        enabled_metrics = [
            self.enable_encoding_space,
            self.enable_jacobian_rank,
            self.enable_rankme,
            self.enable_eval_reward,
            self.enable_decoder_loss,
            self.enable_cluster_acc,
        ]
        num_enabled = sum(enabled_metrics)

        # Also account for additional subplots rendered below (beyond the base metrics)
        additional_plots = 0
        if self.enable_encoding_space and 'encoding_space_size_first' in self.monitoring_data:
            additional_plots += 1
        if self.enable_cluster_acc and 'cluster_acc_e0' in self.monitoring_data:
            additional_plots += 1
        if self.enable_encoding_space and 'encoding_space_size_e0' in self.monitoring_data:
            additional_plots += 1
        if include_train_losses:
            additional_plots += 1

        total_plots = num_enabled + additional_plots

        if total_plots == 0:
            print("No metrics enabled for plotting")
            return None
            
        # Determine subplot layout sized for all attempted subplots
        if total_plots <= 1:
            rows, cols = 1, 1
        elif total_plots <= 2:
            rows, cols = 1, 2
        elif total_plots <= 4:
            rows, cols = 2, 2
        elif total_plots <= 6:
            rows, cols = 2, 3
        else:
            # Up to 9 plots expected (includes extra comparison/commutative panels)
            rows, cols = 3, 3
        
        subplot_idx = 1
        
        # Subplot 1: encoding space size
        if self.enable_encoding_space and 'encoding_space_size' in self.monitoring_data:
            space_sizes = np.array(self.monitoring_data['encoding_space_size'])
            ax = plt.subplot(rows, cols, subplot_idx)
            ax.plot(steps, space_sizes, 'b-o', linewidth=2, markersize=4)
            ax.set_title('Encoding Space Size', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Avg. Pairwise Distance')
            ax.set_ylim(0, 1.5)
            ax.grid(True, alpha=0.3)
            subplot_idx += 1

        # Subplot 2: min Jacobian rank
        if self.enable_jacobian_rank and 'min_jacobian_rank' in self.monitoring_data:
            ranks = np.array(self.monitoring_data['min_jacobian_rank'])
            ax = plt.subplot(rows, cols, subplot_idx)
            ax.plot(steps, ranks, 'g-o', linewidth=2, markersize=4)
            ax.set_title('Minimum Jacobian Rank', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Rank')
            ax.grid(True, alpha=0.3)
            subplot_idx += 1

        # Subplot 3: RankMe
        if self.enable_rankme and 'rankme' in self.monitoring_data:
            rankmes = np.array([m if m is not None else np.nan for m in self.monitoring_data['rankme']])
            if not np.all(np.isnan(rankmes)):
                ax = plt.subplot(rows, cols, subplot_idx)
                ax.plot(steps, rankmes, 'm-o', linewidth=2, markersize=4)
                ax.set_title('RankMe', fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('RankMe')
                ax.grid(True, alpha=0.3)
            subplot_idx += 1

        # Subplot 4: Eval reward (from CSV)
        if self.enable_eval_reward:
            try:
                eval_csv = f"{self.cfg.work_dir}/eval.csv"
                eval_df = pd.read_csv(eval_csv)

                ax = plt.subplot(rows, cols, subplot_idx)
                ax.plot(eval_df['step'], eval_df['episode_reward'], 'c-')
                ax.set_title('Eval Reward', fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Reward')
                ax.grid(True, alpha=0.3)
                subplot_idx += 1
            except Exception as e:
                print(f"⚠️ Could not load eval CSV: {e}")

        # Subplot 5: Decoder evaluation loss (log scale)
        if self.enable_decoder_loss and 'decoder_loss' in self.monitoring_data:
            dec_losses = np.array([l if l is not None else np.nan for l in self.monitoring_data['decoder_loss']])
            if not np.all(np.isnan(dec_losses)):
                ax = plt.subplot(rows, cols, subplot_idx)
                ax.plot(steps, dec_losses, 'r-o', linewidth=2, markersize=4)
                ax.set_title('Decoder Eval Loss', fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('MSE')
                ax.set_ylim(0, 0.06)
                ax.grid(True, alpha=0.3)
                subplot_idx += 1

        # Subplot 6: Cluster accuracy (first/mid/top)
        if self.enable_cluster_acc and 'cluster_acc' in self.monitoring_data:
            acc_top = np.array([acc if acc is not None else np.nan for acc in self.monitoring_data['cluster_acc_top']])
            acc_first = np.array([acc if acc is not None else np.nan for acc in self.monitoring_data['cluster_acc_first']])
            acc_mid = np.array([acc if acc is not None else np.nan for acc in self.monitoring_data['cluster_acc_mid']])
            if (not np.all(np.isnan(acc_top))) or (not np.all(np.isnan(acc_first))) or (not np.all(np.isnan(acc_mid))):
                ax = plt.subplot(rows, cols, subplot_idx)
                if not np.all(np.isnan(acc_first)):
                    ax.plot(steps, acc_first * 100, label='first', color='tab:blue', linewidth=2)
                if not np.all(np.isnan(acc_mid)):
                    ax.plot(steps, acc_mid * 100, label='mid', color='tab:green', linewidth=2)
                if not np.all(np.isnan(acc_top)):
                    ax.plot(steps, acc_top * 100, label='top', color='tab:orange', linewidth=2)
                ax.set_title('Cluster Accuracy (first/mid/top)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Accuracy (%)')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                ax.legend()
            subplot_idx += 1

        # Additional subplot: encoding space size (first vs top)
        if self.enable_encoding_space and 'encoding_space_size_first' in self.monitoring_data:
            # Align lengths with steps (pad with NaN if necessary)
            steps_arr = steps
            ess_top_list = self.monitoring_data['encoding_space_size']
            ess_first_list = self.monitoring_data['encoding_space_size_first']
            # pad to same length as steps
            if len(ess_top_list) < len(steps_arr):
                ess_top_list = ess_top_list + [None] * (len(steps_arr) - len(ess_top_list))
            if len(ess_first_list) < len(steps_arr):
                ess_first_list = ess_first_list + [None] * (len(steps_arr) - len(ess_first_list))
            ess_top = np.array([v if v is not None else np.nan for v in ess_top_list])
            ess_first = np.array([v if v is not None else np.nan for v in ess_first_list])
            if (not np.all(np.isnan(ess_top))) or (not np.all(np.isnan(ess_first))):
                ax = plt.subplot(rows, cols, subplot_idx)
                if not np.all(np.isnan(ess_first)):
                    ax.plot(steps_arr, ess_first, label='first', color='tab:blue', linewidth=2)
                if not np.all(np.isnan(ess_top)):
                    ax.plot(steps_arr, ess_top, label='top', color='tab:orange', linewidth=2)
                ax.set_title('Encoding Space Size (first vs top)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Avg. Pairwise Distance')
                ax.set_ylim(0, 1.5)
                ax.grid(True, alpha=0.3)
                ax.legend()
            subplot_idx += 1

        # Subplot: Commutative encoders cluster accuracy (E0 vs Ei_a/Ei_b)
        if self.enable_cluster_acc and 'cluster_acc_e0' in self.monitoring_data:
            steps_arr = steps
            def _pad(vs):
                return vs + [None] * (len(steps_arr) - len(vs)) if len(vs) < len(steps_arr) else vs
            e0 = np.array([v if v is not None else np.nan for v in _pad(self.monitoring_data['cluster_acc_e0'])])
            ei_a = np.array([v if v is not None else np.nan for v in _pad(self.monitoring_data['cluster_acc_ei_a'])])
            ei_b = np.array([v if v is not None else np.nan for v in _pad(self.monitoring_data['cluster_acc_ei_b'])])
            if (not np.all(np.isnan(e0))) or (not np.all(np.isnan(ei_a))) or (not np.all(np.isnan(ei_b))):
                ax = plt.subplot(rows, cols, subplot_idx)
                if not np.all(np.isnan(e0)):
                    ax.plot(steps_arr, e0 * 100, label='E0', linewidth=2)
                if not np.all(np.isnan(ei_a)):
                    ax.plot(steps_arr, ei_a * 100, label='Ei_a', linewidth=2)
                if not np.all(np.isnan(ei_b)):
                    ax.plot(steps_arr, ei_b * 100, label='Ei_b', linewidth=2)
                ax.set_title('Commutative Cluster Accuracy (E0 vs Ei)', fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Accuracy (%)')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                ax.legend()
            subplot_idx += 1

        # Subplot: Commutative encoders encoding space size (E0 vs Ei_a/Ei_b)
        if self.enable_encoding_space and 'encoding_space_size_e0' in self.monitoring_data:
            steps_arr = steps
            def _pad2(vs):
                return vs + [None] * (len(steps_arr) - len(vs)) if len(vs) < len(steps_arr) else vs
            e0s = np.array([v if v is not None else np.nan for v in _pad2(self.monitoring_data.get('encoding_space_size_e0', []))])
            eas = np.array([v if v is not None else np.nan for v in _pad2(self.monitoring_data.get('encoding_space_size_ei_a', []))])
            ebs = np.array([v if v is not None else np.nan for v in _pad2(self.monitoring_data.get('encoding_space_size_ei_b', []))])
            if (not np.all(np.isnan(e0s))) or (not np.all(np.isnan(eas))) or (not np.all(np.isnan(ebs))):
                ax = plt.subplot(rows, cols, subplot_idx)
                if not np.all(np.isnan(e0s)):
                    ax.plot(steps_arr, e0s, label='E0', linewidth=2)
                if not np.all(np.isnan(eas)):
                    ax.plot(steps_arr, eas, label='Ei_a', linewidth=2)
                if not np.all(np.isnan(ebs)):
                    ax.plot(steps_arr, ebs, label='Ei_b', linewidth=2)
                ax.set_title('Commutative Encoding Space Size', fontsize=12, fontweight='bold')
                ax.set_xlabel('Training Steps')
                ax.set_ylabel('Avg. Pairwise Distance')
                ax.grid(True, alpha=0.3)
                ax.legend()
            subplot_idx += 1

        # Additional subplot: training losses in main grid (reward_loss, consistency_loss, L_eq)
        if include_train_losses and train_losses_df is not None:
            ax = plt.subplot(rows, cols, subplot_idx)
            ax.plot(train_losses_df['step'], train_losses_df['reward_loss'], label='reward_loss', color='tab:blue')
            ax.plot(train_losses_df['step'], train_losses_df['consistency_loss'], label='consistency_loss', color='tab:red')
            if 'L_eq' in train_losses_df.columns:
                ax.plot(train_losses_df['step'], train_losses_df['L_eq'], label='L_eq', color='tab:purple')
            ax.set_title('Training Losses', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.set_ylim(1e-5, 2.0)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend()
            subplot_idx += 1

        plt.tight_layout()

        if save_plot:
            plot_path = self.save_dir / "monitoring_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved monitoring curves to {plot_path}")

        return plt.gcf()

    def _should_plot_action_distribution(self, cluster_acc: Optional[float]) -> bool:
        """Whether to plot action distribution heatmap.
        Conditions:
        - Continuous action, 1-D
        - cluster_acc >= 0.97 if available (else, skip)
        - Buffer available with actions and obs_type
        """
        if getattr(self.cfg, 'discrete_action', False):
            return False
        if getattr(self.cfg, 'action_dim', 1) != 1:
            return False
        if cluster_acc is None:
            return False
        if isinstance(cluster_acc, torch.Tensor):
            cluster_acc = float(cluster_acc.item())
        return cluster_acc >= 2 and self.buffer is not None and self.buffer.num_eps > 0 # <------------ disabled heatmap

    def _plot_action_distribution_heatmap(self, step: int):
        """Generate a heatmap of action distributions for up to 11 observation types (counts).
        Uses actions and obs_type from the replay buffer. Assumes continuous 1-D action.
        Also saves the underlying histogram data as JSON alongside the figure.
        """
        # Access internal storage
        storage = self.buffer._buffer._storage  # type: ignore
        total_steps = self.buffer._steps_in_buffer  # type: ignore
        td_all = storage[:total_steps]
        actions = td_all.get('action', None)  # shape (N, action_dim)
        obs_type = td_all.get('obs_type', None)  # shape (N,)
        if actions is None or obs_type is None:
            print("   ⚠️ Buffer does not contain actions or obs_type; skipping heatmap.")
            return
        # Flatten
        if actions.ndim > 2:
            actions = actions.view(-1, actions.shape[-1])
        if obs_type.ndim > 1:
            obs_type = obs_type.view(-1)
        # Keep only valid obs_type (>=0)
        mask_valid = (obs_type >= 0)
        actions = actions[mask_valid]
        obs_type = obs_type[mask_valid]
        if actions.numel() == 0:
            print("   ⚠️ No valid obs_type entries to plot action distribution.")
            return
        # Build bins for the 1-D action in [-1,1]
        num_bins = 50
        bin_edges = torch.linspace(-1.0, 1.0, steps=num_bins+1, device=actions.device)
        # Determine which 11 types to show: counts 0..10 (clip to observed range)
        unique_types = torch.unique(obs_type).to(torch.int64)
        # Prefer 0..10 if available, otherwise take smallest 11
        preferred = torch.arange(0, 11, device=obs_type.device, dtype=torch.int64)
        types_to_show = preferred[torch.isin(preferred, unique_types)]
        if types_to_show.numel() < 11:
            remaining = unique_types[~torch.isin(unique_types, preferred)]
            remaining = remaining.sort().values[:max(0, 11 - types_to_show.numel())]
            types_to_show = torch.cat([types_to_show, remaining], dim=0)
        # Build histogram per type
        heat = []
        labels = []
        totals = []
        for t in types_to_show.tolist():
            sel = (obs_type == t)
            if sel.sum() == 0:
                hist = torch.zeros(num_bins, device=actions.device)
                total = 0
            else:
                vals = actions[sel, 0]
                hist = torch.histc(vals, bins=num_bins, min=-1.0, max=1.0)
                hist = hist / hist.sum().clamp(min=1.0)
                total = int(sel.sum().item())
            heat.append(hist)
            labels.append(str(t))
            totals.append(total)
        if not heat:
            print("   ⚠️ Nothing to plot for action distribution heatmap.")
            return
        heat = torch.stack(heat, dim=0).cpu().numpy()  # (K, num_bins)
        bin_edges_np = bin_edges.detach().cpu().numpy()
        # Plot heatmap
        plt.figure(figsize=(12, max(4, 0.5*len(labels))))
        ax = plt.gca()
        # Map x-axis to actual action values using extent
        im = ax.imshow(
            heat,
            aspect='auto',
            cmap='viridis',
            origin='lower',
            extent=[bin_edges_np[0], bin_edges_np[-1], -0.5, len(labels)-0.5]
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        # Set x ticks to actual action values
        xticks = np.linspace(-1.0, 1.0, num=11)
        ax.set_xticks(xticks)
        ax.set_xlabel('Action value')
        ax.set_ylabel('Object count')
        cbar = plt.colorbar(im)
        cbar.set_label('Probability')
        title = f"Action distribution by object count (step {step})"
        plt.title(title)
        out_path = self.save_dir / f"action_dist_heatmap_step_{step}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"   ✅ Saved action distribution heatmap to {out_path}")

        # Save the histogram data used for plotting as JSON
        json_path = self.save_dir / f"action_dist_heatmap_step_{step}.json"
        data_to_save = {
            'step': int(step),
            'counts': labels,
            'num_bins': int(num_bins),
            'bin_edges': bin_edges_np.tolist(),
            'hist': heat.tolist(),
            'total_samples_per_count': totals,
        }
        try:
            with open(json_path, 'w') as f:
                json.dump(data_to_save, f, indent=2)
            print(f"   💾 Saved heatmap data to {json_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save heatmap data JSON: {e}")

    # -------------------------------------------------------------
    # RankMe computation
    # -------------------------------------------------------------
    def _get_rankme_observations(self) -> torch.Tensor:
        """Lazy-load observations for RankMe from file and cache the tensor."""
        if self._rankme_observations is not None:
            return self._rankme_observations

        if not self.rankme_obs_path.exists():
            raise FileNotFoundError(f"RankMe observation file not found: {self.rankme_obs_path}")

        obs_tensor = torch.load(self.rankme_obs_path, map_location='cpu')  # shape (N, C, H, W)
        if obs_tensor.ndim < 3:
            raise ValueError(f"Unexpected observation tensor shape: {obs_tensor.shape}")

        self._rankme_observations = obs_tensor.float()  # Keep on CPU to avoid extra GPU mem
        print(f"📥 Loaded RankMe observations: {self._rankme_observations.shape}")
        return self._rankme_observations
   
    def compute_rankme_metric_for_model(self, model: torch.nn.Module) -> float:
        """Compute RankMe over a subset of saved observations using a specific model."""

        obs_tensor = self._get_rankme_observations()
        num_samples = min(self.rankme_samples, obs_tensor.shape[0])
        # Random but deterministic sample (seed fixed)
        idx = torch.randperm(obs_tensor.shape[0], device='cpu')[:num_samples]
        obs_sample = obs_tensor[idx].to(self.device)  # Move once to GPU

        # Encode in large batches on GPU using specified model
        encodings = []
        with torch.no_grad():
            for i in range(0, num_samples, self.monitor_batch_size):
                batch = obs_sample[i:i+self.monitor_batch_size]
                if self.cfg.multitask:
                    raise NotImplementedError("do not support multi-task encoders")
                enc = model(batch)
                encodings.append(enc.cpu())
        encodings = torch.cat(encodings, dim=0)

        metric_val = rankme.get_rankme(encodings)
        return float(metric_val)
    
    def generate_simple_report(self) -> str:
        """Generate a simple text report focusing on encoding space"""
        if not self.monitoring_data['steps']:
            return "No monitoring data available."
        
        step = self.monitoring_data['steps'][-1]
        
        # Get values only for enabled metrics
        space_size = self.monitoring_data['encoding_space_size'][-1] if 'encoding_space_size' in self.monitoring_data else None
        min_rank = self.monitoring_data['min_jacobian_rank'][-1] if 'min_jacobian_rank' in self.monitoring_data else None
        rankme_val = self.monitoring_data['rankme'][-1] if 'rankme' in self.monitoring_data else None
        
        # Calculate trend
        if space_size is not None and len(self.monitoring_data['encoding_space_size']) > 1:
            initial_size = self.monitoring_data['encoding_space_size'][0]
            change_percent = ((space_size - initial_size) / initial_size) * 100
            trend = "Increasing" if change_percent > 5 else "Decreasing" if change_percent < -5 else "Stable"
        else:
            change_percent = 0
            trend = "N/A"
        
        report = f"""
📊 SIMPLE ENCODING SPACE REPORT - Step {step}
{'='*50}
{f"Current Encoding Space Size: {space_size:.6f}" if space_size is not None else "Encoding Space Size: Not monitored"}
{f"Minimum Jacobian Rank: {float(min_rank)}" if min_rank is not None else "Minimum Jacobian Rank: Not monitored"}
{f"RankMe: {rankme_val:.2f}" if rankme_val is not None else "RankMe: Not monitored"}
{f"Change from start: {change_percent:+.2f}%" if space_size is not None else ""}
{f"Trend: {trend}" if space_size is not None else ""}

📈 Data Points Collected: {len(self.monitoring_data['steps'])}
🔄 Monitoring Frequency: every {self.monitor_freq} steps
💾 Data saved to: {self.save_dir}

📝 Interpretation:
  • Encoding Space Size: Higher = more diverse, lower = potential collapse
  • Jacobian Rank: Higher = more expressive representations
  • RankMe: Higher = better representation quality (max = latent_dim)
  • Stable trends generally indicate healthy learning
"""
        return report
   

    # -------------------------------------------------------------
    # Public API: save multiple decoder GIFs after training
    # -------------------------------------------------------------
    def save_decoder_gifs(self, num_gifs = 5, step = None):
        """Generate visual comparisons between original and decoder outputs.

        - In counting environments (env has attribute 'count'), save JPG images with a vertical
          separator between original and reconstruction instead of animated GIFs.
        - Otherwise, save animated GIFs as before.

        Args:
            num_gifs: number of different observations to visualise (default 5).
            step:     optional step label used in the filename.
        """
        if not getattr(self.cfg, 'enable_decoder', False):
            print("⚠️ Decoder disabled – skipping GIF generation.")
            return None

        dec = getattr(getattr(self.agent, 'model', None), '_decoder', None)
        if dec is None:
            print("⚠️ Decoder module not found – skipping GIF generation.")
            return None

        base_dir = Path(self.cfg.work_dir) if hasattr(self.cfg, 'work_dir') else self.save_dir
        gif_dir = base_dir / "decoder_gifs"
        gif_dir.mkdir(exist_ok=True)

        # Ensure we have encodings up-to-date
        self._update_baseline_encodings()

        enc_all = self.baseline_encodings.to(self.device)
        obs_all = self.baseline_observations.detach().cpu()

        n = min(num_gifs, obs_all.shape[0])

        # Decoder to eval, restore afterwards
        prev_mode = dec.training
        dec.eval()
        with torch.no_grad():
            recon_all = dec(enc_all).detach().cpu()
        dec.train(prev_mode)

        is_counting_env = hasattr(self.env, 'count')
        for idx in range(n):
            obs0 = obs_all[idx]
            recon0 = recon_all[idx]
            C, H, W = obs0.shape
            num_frames = C // 3

            if is_counting_env:
                # In counting env, save two separate JPGs: original and reconstruction
                f = 0  # use the first frame for static visualization
                orig = obs0[f*3:(f+1)*3].numpy().astype(np.float32)
                rec = recon0[f*3:(f+1)*3].numpy().astype(np.float32)
                orig_img = np.transpose(orig, (1, 2, 0))
                rec_img = np.transpose(rec, (1, 2, 0))
                # Original: no normalization; map to uint8 based on value range
                if orig_img.dtype != np.uint8:
                    if float(orig_img.max()) <= 1.0 + 1e-6:
                        orig_uint8 = np.clip(orig_img * 255.0, 0, 255).astype(np.uint8)
                    else:
                        orig_uint8 = np.clip(orig_img, 0, 255).astype(np.uint8)
                else:
                    orig_uint8 = orig_img
                # Reconstruction: map from [-1,1] -> [0,1]
                rec_img = (rec_img + 1.0) / 2.0
                rec_img = np.clip(rec_img, 0.0, 1.0)
                if getattr(self.cfg, 'better_decoder_image', False):
                    rmin, rmax = rec_img.min(), rec_img.max()
                    if rmax > rmin:
                        rec_img = (rec_img - rmin) / (rmax - rmin + 1e-8)
                rec_uint8 = (rec_img * 255.0).astype(np.uint8)
                # Scale if requested
                scale = getattr(self.cfg, 'gif_scale', 4)
                if scale and scale > 1:
                    o_pil = Image.fromarray(orig_uint8)
                    o_pil = o_pil.resize((o_pil.width * scale, o_pil.height * scale), resample=Image.NEAREST)
                    orig_uint8 = np.array(o_pil)
                    r_pil = Image.fromarray(rec_uint8)
                    r_pil = r_pil.resize((r_pil.width * scale, r_pil.height * scale), resample=Image.NEAREST)
                    rec_uint8 = np.array(r_pil)
                suffix = f"{step}" if step is not None else "final"
                jpg_path_orig = gif_dir / f"decoder_cmp_{idx}_{suffix}_original_observation.jpg"
                jpg_path_rec = gif_dir / f"decoder_cmp_{idx}_{suffix}_reconstruction.jpg"
                try:
                    Image.fromarray(orig_uint8).save(str(jpg_path_orig), format='JPEG', quality=95)
                    Image.fromarray(rec_uint8).save(str(jpg_path_rec), format='JPEG', quality=95)
                    print(f"   🖼️  Saved decoder images → {jpg_path_orig} and {jpg_path_rec}")
                except Exception as e:
                    print(f"⚠️ Failed to save JPGs {jpg_path_orig} / {jpg_path_rec}: {e}")
            else:
                # Non-counting env: save animated GIF with all frames
                frames = []
                for f in range(num_frames):
                    orig = obs0[f*3:(f+1)*3].numpy().astype(np.float32)
                    rec = recon0[f*3:(f+1)*3].numpy().astype(np.float32)
                    orig_img = np.transpose(orig, (1, 2, 0))
                    rec_img = np.transpose(rec, (1, 2, 0))
                    # Original: no normalization; map to uint8
                    if orig_img.dtype != np.uint8:
                        if float(orig_img.max()) <= 1.0 + 1e-6:
                            orig_uint8 = np.clip(orig_img * 255.0, 0, 255).astype(np.uint8)
                        else:
                            orig_uint8 = np.clip(orig_img, 0, 255).astype(np.uint8)
                    else:
                        orig_uint8 = orig_img
                    # Reconstruction: [-1,1] -> [0,1] -> uint8
                    rec_img = (rec_img + 1.0) / 2.0
                    rec_img = np.clip(rec_img, 0.0, 1.0)
                    if getattr(self.cfg, 'better_decoder_image', False):
                        rmin, rmax = rec_img.min(), rec_img.max()
                        if rmax > rmin:
                            rec_img = (rec_img - rmin) / (rmax - rmin + 1e-8)
                    rec_uint8 = (rec_img * 255.0).astype(np.uint8)
                    # Optional scale
                    scale = getattr(self.cfg, 'gif_scale', 4)
                    if scale and scale > 1:
                        o_pil = Image.fromarray(orig_uint8)
                        o_pil = o_pil.resize((o_pil.width * scale, o_pil.height * scale), resample=Image.NEAREST)
                        orig_uint8 = np.array(o_pil)
                        r_pil = Image.fromarray(rec_uint8)
                        r_pil = r_pil.resize((r_pil.width * scale, r_pil.height * scale), resample=Image.NEAREST)
                        rec_uint8 = np.array(r_pil)
                    # Combine with vertical separator
                    sep_w = 3
                    separator = np.zeros((orig_uint8.shape[0], sep_w, 3), dtype=np.uint8)
                    comb_uint8 = np.concatenate([orig_uint8, separator, rec_uint8], axis=1)
                    frames.append(comb_uint8)

                suffix = f"{step}" if step is not None else "final"
                gif_path = gif_dir / f"decoder_cmp_{idx}_{suffix}.gif"
                try:
                    imageio.mimsave(str(gif_path), frames, fps=2)
                    print(f"   🎞️  Saved decoder GIF → {gif_path}")
                except Exception as e:
                    print(f"⚠️ Failed to save GIF {gif_path}: {e}")

        return gif_dir

    def compute_full_dim_metrics(self) -> Dict[str, float]:
        """Compute dimension metrics using *all* observations currently stored in the replay buffer."""
        if self.buffer is None or self.buffer.num_eps == 0:
            print("⚠️ No data in buffer – skipping full-dataset dimension analysis.")
            return {}


        # Attempt to access internal storage directly (TorchRL LazyTensorStorage)
        storage = self.buffer._buffer._storage  # type: ignore
        total_steps = self.buffer._steps_in_buffer  # type: ignore
        td_all = storage[:total_steps]  # TensorDict slice containing all stored steps
        obs_tensor = td_all.get('obs', None)
        if obs_tensor is None:
            print("⚠️ Buffer storage does not contain 'obs' field – cannot compute dimension metrics.")
            return {}

        # Flatten to (N, C, H, W)
        if obs_tensor.ndim > 4:
            obs_tensor = obs_tensor.view(-1, *obs_tensor.shape[-3:])

        # -------------------------------------------------
        # 0.5  Determine which steps are *visible* to the Q-function
        # -------------------------------------------------
        if self.buffer._q_sample_ratio != 1.0:
            print("⚠️ Buffer is not fully visible to the Q-function, getting visible steps from buffer")
            episode_ids = td_all.get('episode', None)
            episode_ids_cpu = episode_ids.view(-1).to('cpu')
            visible_mask = torch.tensor([
                self.buffer._episode_visible.get(int(ep_id), False) for ep_id in episode_ids_cpu
            ], dtype=torch.bool)
            visible_idx = visible_mask.nonzero(as_tuple=False).view(-1)
        else:
            visible_idx = torch.arange(obs_tensor.shape[0])

        # -------------------------------------------------
        # 1.5  Encode observations in manageable batches
        # -------------------------------------------------
        encode_batch = self.monitor_batch_size
        enc_list = []
        with torch.no_grad():
            for start in range(0, obs_tensor.shape[0], encode_batch):
                batch_obs = obs_tensor[start:start+encode_batch].to(self.device)
                enc = self.agent.model.encode(batch_obs, task=None)
                enc_list.append(enc.cpu())  # move to CPU to free GPU mem quickly


        z = torch.cat(enc_list, dim=0).to(self.device)

        N = z.shape[0]
        if N < 2:
            print("⚠️ Not enough observations for full-dataset analysis.")
            return {}

        # -------------------------------------------------
        # 3. Build stacked q outputs P(s) ONLY for *visible* samples
        # -------------------------------------------------
        if visible_idx.numel() >= 1:
            z_vis = z[visible_idx.to(self.device)]
        else:
            z_vis = z.new_empty((0, z.shape[1]))  # empty tensor

        probe_actions = self._probe_actions.to(self.device)  # (M, A)
        num_bins = getattr(self.cfg, 'num_bins', 101)

        q_list = []  # will collect on GPU then CPU

        batch_q = self.monitor_batch_size  # compute q in chunks
        with torch.no_grad():
            for start in range(0, z_vis.shape[0], batch_q):
                z_chunk = z_vis[start:start+batch_q]  # (B, latent_dim)
                Bc = z_chunk.shape[0]
                if Bc == 0:
                    break
                # Expand for M actions
                z_rep = z_chunk.unsqueeze(1).repeat(1, self.M_actions, 1)
                a_rep = probe_actions.unsqueeze(0).repeat(Bc, 1, 1)
                z_flat = z_rep.reshape(-1, z.shape[1])
                a_flat = a_rep.reshape(-1, a_rep.shape[-1])

                q_out = self.agent.model.Q(z_flat, a_flat, task=None, return_type='all', detach=True)
                q_out = q_out.mean(0)  # (B*M, num_bins) – average over ensemble
                q_out = q_out.view(Bc, self.M_actions, num_bins).mean(1)  # (B, num_bins) average over actions
                q_list.append(q_out)

        if len(q_list) > 0:
            P = torch.cat(q_list, dim=0)  # (N_vis, num_bins)
        else:
            P = torch.empty((0, num_bins), device=self.device)

        # -------------------------------------------------
        # 4. Compute pairwise distances – z (all) and P (visible only)
        # -------------------------------------------------
        avg_dis, min_dis = self._pairwise_distance_stats_large(
            z, batch_size=self.monitor_batch_size, max_samples=self.pd_max_samples)

        if P.shape[0] >= 2:
            avg_dis_p, min_dis_p = self._pairwise_distance_stats_large(
                P, batch_size=self.monitor_batch_size, max_samples=self.pd_max_samples)
        else:
            avg_dis_p, min_dis_p = float('nan'), float('nan')

        # -------------------------------------------------
        # 5. Radius R and dimensionality estimate (z)
        # -------------------------------------------------
        R = float(z.norm(dim=1).max().item())
        eps = 1e-10
        est_dim_avg = est_dim_min = float(-114514)
        try:
            est_dim_avg = float(np.log(N) / np.log(1 + 2 * R / (avg_dis + eps))) if avg_dis > eps else float('nan')
        except ZeroDivisionError:
            est_dim_avg = float('nan')
        try:
            est_dim_min = float(np.log(N) / np.log(1 + 2 * R / (min_dis + eps))) if min_dis > eps else float('nan')
        except ZeroDivisionError:
            est_dim_min = float('nan')

        # -------------------------------------------------
        # 5b. Lipschitz constant estimation (subset of z)
        # -------------------------------------------------
        if self.cfg.num_q == 1:
            lipschitz_supported = True
        else:
            lipschitz_supported = False
        if lipschitz_supported:
            K_est = self._estimate_lipschitz(z)
            self.monitoring_data['lipschitz_K'].append(K_est)
            print(f"🧮 Estimated Lipschitz Constant: {K_est:.4f}")
        else:
            K_est = float('nan')
            self.monitoring_data['lipschitz_K'].append(K_est)
            print("🧮 Lipschitz constant estimation not supported for multi-Q model")

        # -------------------------------------------------
        # 5c. Dimension estimates based on P-space distances & Lipschitz K
        # -------------------------------------------------
        N_vis = int(z_vis.shape[0])
        est_dim_p_avg = est_dim_p_min = float(-114514)
        try:
            est_dim_p_avg = float(np.log(N_vis) / np.log(1 + 2 * R * K_est / (avg_dis_p + eps))) if avg_dis_p > eps else float('nan')
        except:
            est_dim_p_avg = float('nan')
        try:
            est_dim_p_min = float(np.log(N_vis) / np.log(1 + 2 * R * K_est / (min_dis_p + eps))) if min_dis_p > eps else float('nan')
        except:
            est_dim_p_min = float('nan')

        # -------------------------------------------------
        # 6. Record and return
        # -------------------------------------------------
        self.monitoring_data['avg_dis_z'].append(avg_dis)
        self.monitoring_data['min_dis_z'].append(min_dis)
        self.monitoring_data['avg_dis_p'].append(avg_dis_p)
        self.monitoring_data['min_dis_p'].append(min_dis_p)
        self.monitoring_data['est_dim_avg'].append(est_dim_avg)
        self.monitoring_data['est_dim_min'].append(est_dim_min)
        self.monitoring_data['est_dim_p_avg'].append(est_dim_p_avg)
        self.monitoring_data['est_dim_p_min'].append(est_dim_p_min)

        print(f"📏 FULL DATASET Δ_z(avg): {avg_dis:.2e}, Δ_z(min): {min_dis:.2e}, R: {R:.2e}\n     → est_dim(avg): {est_dim_avg:.2f}, est_dim(min): {est_dim_min:.2f}")
        print(f"📊 P-space distances: Δ_p(avg): {avg_dis_p:.2e}, Δ_p(min): {min_dis_p:.2e}\n     → est_dim(avg): {est_dim_p_avg:.2f}, est_dim(min): {est_dim_p_min:.2f}")

        # Return dictionary (prefixed with _full to avoid confusion)
        return {
            'avg_dis_z_full': avg_dis,
            'min_dis_z_full': min_dis,
            'avg_dis_p_full': avg_dis_p,
            'min_dis_p_full': min_dis_p,
            'est_dim_avg_full': est_dim_avg,
            'est_dim_min_full': est_dim_min,
            'lipschitz_K_full': K_est,
            'est_dim_p_avg_full': est_dim_p_avg,
            'est_dim_p_min_full': est_dim_p_min,
        }

    # -------------------------------------------------------------
    # Memory-efficient pairwise distance stats for large datasets
    # -------------------------------------------------------------
    def _pairwise_distance_stats_large(self, X: torch.Tensor, batch_size: int = 1024, max_samples: int = 1000000):
        """Compute average & minimum pairwise L2 distance of rows in X without allocating the full NxN matrix.

        Args:
            X:          Tensor of shape (N, D)
            batch_size: Chunk size for distance computation.
            max_samples:If N > max_samples, randomly subsample to this many points first.

        Returns:
            (avg_distance, min_distance) tuple as floats.
        """
        if X.ndim > 2:
            X = X.view(X.shape[0], -1)
        N = X.shape[0]
        device = X.device

        # Optional subsampling to keep computation reasonable
        if N > max_samples:
            idx = torch.randperm(N, device=device)[:max_samples]
            X = X[idx]
            N = max_samples

        total_sum = 0.0
        total_pairs = 0
        min_dist = float('inf')

        for i in range(0, N, batch_size):
            Xi = X[i:min(i + batch_size, N)]
            for j in range(i, N, batch_size):
                Xj = X[j:min(j + batch_size, N)]
                dist_block = torch.cdist(Xi, Xj)  # (bi, bj)

                if i == j:
                    # Use upper triangle without the diagonal
                    triu_idx = torch.triu_indices(dist_block.shape[0], dist_block.shape[1], offset=1, device=device)
                    dvals = dist_block[triu_idx[0], triu_idx[1]]
                else:
                    dvals = dist_block.flatten()

                if dvals.numel() == 0:
                    continue
                total_sum += dvals.sum().item()
                total_pairs += dvals.numel()
                block_min = dvals.min().item()
                if block_min < min_dist:
                    min_dist = block_min

        avg_dist = total_sum / total_pairs if total_pairs > 0 else float('nan')
        return avg_dist, min_dist

    # -------------------------------------------------------------
    # Lipschitz constant estimation helpers
    # -------------------------------------------------------------
    def _estimate_lipschitz(self, z_tensor: torch.Tensor, L: int = 16384) -> float:
        """Estimate Lipschitz constant K_theta (w.r.t latent z).

        Finds the maximum spectral norm of the Jacobian of the stacked q outputs
        (over fixed probe actions) with respect to z across a subset of samples.
        """
        assert self.cfg.num_q == 1, "Lipschitz constant estimation only supported for single-Q model"
        assert self.cfg.num_bins >= 2, "num_bins must be at least 2"

        N = z_tensor.shape[0]
        if N == 0:
            return float('nan')
        idx = torch.randperm(N, device=z_tensor.device)[:min(L, N)]
        z_subset = z_tensor[idx]

        probe_actions = self._probe_actions.to(z_tensor.device)

        # Define single-sample function
        def q_stack(z_single: torch.Tensor):
            z_rep = z_single.repeat(self.M_actions, 1)
            a_rep = probe_actions
            q_out = self.agent.model.Q(z_rep, a_rep, task=None, return_type='all', detach=False)
            q_out = q_out.mean(0)  # (M, num_bins) – average over ensemble
            q_out = q_out.mean(0)  # average over actions → (num_bins)
            return q_out  # (num_bins)

        # jacrev then vmap to compute singular max quickly
        jac_fn = jacrev(q_stack)

        def single_spectral(z_single: torch.Tensor):
            J = jac_fn(z_single)
            # J shape: (output_dim, latent_dim)
            svals = torch.linalg.svdvals(J)
            return svals.max()

        # Compute spectral norms in manageable chunks to lower memory footprint
        batch_size = 32  # process this many latent samples at a time         <------------ try to decrease this if not enough memory
        max_spec_val = float('-inf')

        for start in range(0, z_subset.shape[0], batch_size):
            z_batch = z_subset[start:start + batch_size]
            # Vectorised evaluation within the current batch
            spec_vals = vmap(single_spectral, randomness="same")(z_batch)
            batch_max = spec_vals.max().item()
            if batch_max > max_spec_val:
                max_spec_val = batch_max

        K_est = float(max_spec_val)
        return K_est

    # -------------------------------------------------------------
    # Helper: t-SNE cluster visualisation
    # -------------------------------------------------------------

    def _plot_tsne_clusters(self, save_path,
                            cluster_acc: Optional[float] = None,
                            enc: Optional[torch.Tensor] = None,
                            labels: Optional[torch.Tensor] = None,
                            best_attr_name: Optional[str] = None,
                            best_suffix: Optional[str] = None):
        """Generate a 2-D t-SNE plot.
        - If enc/labels are None: use self.baseline_encodings/self.baseline_labels and track best via self.best_cluster_acc.
        - If enc/labels are provided: plot for that layer; best tracking uses best_attr_name/best_suffix when provided.
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError as _e:
            print("⚠️ scikit-learn not installed; cannot generate t-SNE plot.")
            return

        if enc is None or labels is None:
            if self.baseline_labels is None:
                print("⚠️ No labels available for t-SNE clustering plot.")
                return
            z = self.baseline_encodings.detach().cpu().numpy()
            labs = self.baseline_labels.detach().cpu().numpy()
            # Default best tracking for top layer
            best_attr_name = 'best_cluster_acc' if best_attr_name is None else best_attr_name
            best_suffix = 'tsne_clusters_best.png' if best_suffix is None else best_suffix
        else:
            z = enc.detach().cpu().numpy()
            labs = labels.detach().cpu().numpy()
            if best_attr_name is None:
                best_attr_name = 'best_cluster_acc'
            if best_suffix is None:
                best_suffix = 'tsne_clusters_best.png'

        if cluster_acc is None:
            cluster_acc = float('nan')

        tsne = TSNE(n_components=2, init="random", learning_rate="auto", perplexity=30, n_iter=1000)
        z_2d = tsne.fit_transform(z)

        plt.figure(figsize=(6, 5))
        unique_labels = sorted(set(labs))
        num_classes = len(unique_labels)

        # Choose a *categorical* palette with visually distinct colours.
        if num_classes <= 10:
            base_cmap = plt.get_cmap("tab10")
        elif num_classes <= 20:
            base_cmap = plt.get_cmap("tab20")
        else:
            # For many classes fall back to repeating tab20 palette but shift hue
            base_cmap = plt.get_cmap("tab20")

        colors = [base_cmap(i % base_cmap.N) for i in range(num_classes)]

        for cls, col in zip(unique_labels, colors):
            idx = labs == cls
            plt.scatter(z_2d[idx, 0], z_2d[idx, 1], s=10, alpha=0.85, label=str(cls), color=col)
        plt.legend(title="Label")
        # Show cluster accuracy (if available) in title
        if not np.isnan(cluster_acc):
            plt.title(f"t-SNE Latent Space (Acc: {cluster_acc*100:.2f}%)")
        else:
            plt.title("t-SNE Latent Space")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)

        # Save best-so-far plot (highest cluster accuracy) for selected layer
        prev_best = getattr(self, best_attr_name, float('-inf'))
        if not np.isnan(cluster_acc) and cluster_acc > prev_best:
            setattr(self, best_attr_name, cluster_acc)
            best_path = Path(save_path).with_name(best_suffix)
            plt.savefig(best_path, dpi=300)
            print(f"🔥 New best cluster accuracy {cluster_acc*100:.2f}% → saved to {best_path}")

        plt.close()

# Integration function for easy use in training loop
def create_simple_encoding_monitor(cfg, encoder, env, save_dir=None, agent=None, buffer=None):
    """Factory function to create simple encoding monitor"""
    return SimpleEncodingSpaceMonitor(
        cfg=cfg,
        encoder=encoder,
        env=env,
        device=cfg.get('device', 'cuda'),
        save_dir=save_dir or f"simple_encoding_logs_{cfg.exp_name}",
        agent=agent,
        buffer=buffer
    )
