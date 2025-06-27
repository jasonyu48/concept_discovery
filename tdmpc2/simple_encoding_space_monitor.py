# Simplified Encoder Space Monitoring System
# Author: Based on your requirements
# Purpose: Monitor encoding space size and changes over time (simplified version)

import torch
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
        agent = None
    ):
        self.cfg = cfg
        self.encoder = encoder
        self.env = env
        self.device = device
        self.agent = agent
        self.save_dir = Path(save_dir) if save_dir else Path("simple_encoding_logs")
        self.save_dir.mkdir(exist_ok=True)
        
        # Monitoring configuration
        self.num_seed_obs = 64  # Number of diverse seed observations
        self.monitor_freq = self.cfg.get('monitor_freq', 2000)  # Monitor every N steps
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
        
        # Add metric storage based on enabled flags
        if self.enable_encoding_space:
            self.monitoring_data['encoding_space_size'] = []
        if self.enable_jacobian_rank:
            self.monitoring_data['min_jacobian_rank'] = []
        if self.enable_rankme:
            self.monitoring_data['rankme'] = []
        
        # -------------------------------------------------------------
        # RankMe setup (needed for baseline observation sampling)
        # -------------------------------------------------------------
        self.rankme_samples = getattr(self.cfg, 'rankme_samples', 30000)
        self.rankme_batch_size = 4096
        default_rankme_path = f"/scratch/tshu2/jyu197/obs_data/{getattr(self.cfg, 'task', 'unknown')}/obs/observations.pt"
        self.rankme_obs_path = Path(getattr(self.cfg, 'rankme_obs_path', default_rankme_path))
        self._rankme_observations = None  # Lazy loaded
        
        # Sample baseline observations from saved data
        self.baseline_observations = None
        self.baseline_encodings = None
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
            print(f"📏 Initial encoding space size: {initial_space_size:.6f}")
            self.monitoring_data['encoding_space_size'].append(initial_space_size)

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
        
    def _sample_baseline_observations(self):
        """Sample baseline observations from saved observation data using cfg.seed"""
        print("🌱 Sampling baseline observations from saved data...")
        
        try:
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
            
            print(f"✅ Sampled {len(indices)} baseline observations")
            print(f"   Final shape: {self.baseline_observations.shape}")
            
            # Compute initial baseline encodings
            self._update_baseline_encodings()

            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            
        except Exception as e:
            print(f"⚠️ Failed to sample from saved observations: {e}")
            print("   Falling back to environment-based observation generation...")
            self._generate_baseline_observations_fallback()
    
    def _generate_baseline_observations_fallback(self):
        """Fallback method to generate observations from environment if saved data unavailable"""
        print("🌱 Generating baseline observations from environment (fallback)...")
        
        observations = []
        
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
        print(f"✅ Generated {len(observations)} fallback baseline observations")
        print(f"   Final shape: {self.baseline_observations.shape}")
        
        # Compute initial baseline encodings
        self._update_baseline_encodings()
        
    def _update_baseline_encodings(self):
        """Update baseline encodings with current encoder state"""
        with torch.no_grad():
            # Handle both single-task and multi-task encoders
            if self.cfg.multitask:
                raise NotImplementedError("monitor not implemented for multi-task encoders")
            else:
                # Single-task encoder (direct call)
                print(f"🔍 Updating baseline encodings with single-task encoder")
                self.baseline_encodings = self.encoder(self.baseline_observations)
    
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
        # vmap maps jac_single over the batch dimension of obs yielding
        # shape (B, E_dim, *input_shape)
        jac_batch = vmap(jac_single, randomness="same")(obs)
        jac_batch = jac_batch.flatten(start_dim=2)

        # Compute rank for each sample (batched matrix_rank supported by PyTorch)
        ranks = torch.linalg.matrix_rank(jac_batch)
        return int(ranks.min().item())
    
    def monitor_step(self, step: int) -> Dict[str, float]:
        """
        Perform monitoring at given training step - simplified version
        
        Returns:
            Dictionary with encoding space size only
        """
        if step % self.monitor_freq != 0:
            return {}
        
        # start_time = time.time()
        
        # Update encodings with current encoder
        self._update_baseline_encodings()
        
        # Store results based on enabled metrics
        metrics = {
            'step': step,
        }
        
        # Update monitoring data
        self.monitoring_data['steps'].append(step)
        self.monitoring_data['timestamp'].append(time.time())
        
        # Compute and store enabled metrics
        if self.enable_encoding_space:
            space_size = self.pairwise_distance(self.baseline_encodings).mean().item()
            metrics['encoding_space_size'] = space_size
            self.monitoring_data['encoding_space_size'].append(space_size)
            print(f"   Encoding space size: {space_size:.6f}")
            
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
        
        # Auto-save periodically and generate updated plots
        if step % (self.monitor_freq * 5) == 0:
            self.save_monitoring_data()
            # Generate updated encoding space curve plot
            try:
                self.plot_monitoring_curves(save_plot=True)
                print(f"   ✅ Updated encoding space curve saved!")
            except Exception as e:
                print(f"   ⚠️ Failed to generate encoding space curve: {e}")
        
        # Save model periodically to same path (every 5 monitoring steps)
        if step % (self.monitor_freq * 5) == 0:
            self._save_model_checkpoint(step)
        
        return metrics
    
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

    
    def plot_monitoring_curves(self, save_plot: bool = True):
        """Generate a 2x2 panel plot of metrics: encoding size, min rank, RankMe, eval reward."""
        if not self.monitoring_data['steps']:
            print("No monitoring data to plot")
            return None
        
        plt.figure(figsize=(14, 10))

        steps = np.array(self.monitoring_data['steps'])
        
        # Count enabled metrics to determine subplot layout
        enabled_metrics = [self.enable_encoding_space, self.enable_jacobian_rank, 
                          self.enable_rankme, self.enable_eval_reward]
        num_enabled = sum(enabled_metrics)
        
        if num_enabled == 0:
            print("No metrics enabled for plotting")
            return None
            
        # Determine subplot layout
        if num_enabled == 1:
            rows, cols = 1, 1
        elif num_enabled == 2:
            rows, cols = 1, 2
        elif num_enabled <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3  # fallback
        
        subplot_idx = 1
        
        # Subplot 1: encoding space size
        if self.enable_encoding_space and 'encoding_space_size' in self.monitoring_data:
            space_sizes = np.array(self.monitoring_data['encoding_space_size'])
            ax = plt.subplot(rows, cols, subplot_idx)
            ax.plot(steps, space_sizes, 'b-o', linewidth=2, markersize=4)
            ax.set_title('Encoding Space Size', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Avg. Pairwise Distance')
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
            eval_csv = f"{self.cfg.work_dir}/eval.csv"
            eval_df = pd.read_csv(eval_csv)

            ax = plt.subplot(rows, cols, subplot_idx)
            ax.plot(eval_df['step'], eval_df['episode_reward'], 'c-')
            ax.set_title('Eval Reward', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plot_path = self.save_dir / "monitoring_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved monitoring curves to {plot_path}")

        return plt.gcf()

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
            for i in range(0, num_samples, self.rankme_batch_size):
                batch = obs_sample[i:i+self.rankme_batch_size]
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

# Integration function for easy use in training loop
def create_simple_encoding_monitor(cfg, encoder, env, save_dir=None, agent=None):
    """Factory function to create simple encoding monitor"""
    return SimpleEncodingSpaceMonitor(
        cfg=cfg,
        encoder=encoder,
        env=env,
        device=cfg.get('device', 'cuda'),
        save_dir=save_dir or f"simple_encoding_logs_{cfg.exp_name}",
        agent=agent
    )
