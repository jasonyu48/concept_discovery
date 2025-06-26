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
        
        # Simple storage for monitoring data
        self.monitoring_data = {
            'steps': [],
            'encoding_space_size': [],  # Only this metric
            'min_jacobian_rank': [],    # New: track minimum Jacobian rank
            'rankme': [],               # New
            'timestamp': []
        }
        
        # Generate baseline diverse observations with consecutive frames
        self.baseline_observations = None
        self.baseline_encodings = None
        self.observation_frames = []  # Store frames for GIF generation
        self._generate_baseline_observations()
        
        print(f"🔍 SimpleEncodingSpaceMonitor initialized:")
        print(f"   - Monitoring frequency: every {self.monitor_freq} steps")
        print(f"   - Seed observations: {self.num_seed_obs}")
        print(f"   - Save directory: {self.save_dir}")
        
        # -------------------------------------------------------------
        # RankMe setup
        # -------------------------------------------------------------
        self.rankme_samples = getattr(self.cfg, 'rankme_samples', 30000)
        self.rankme_batch_size = 4096
        default_rankme_path = f"/scratch/tshu2/jyu197/obs_data/{getattr(self.cfg, 'task', 'unknown')}/obs/observations.pt"
        self.rankme_obs_path = Path(getattr(self.cfg, 'rankme_obs_path', default_rankme_path))
        self._rankme_observations = None  # Lazy loaded
        
        # Calculate and save initial encoding space size
        initial_space_size = self.pairwise_distance(self.baseline_encodings).mean().item()
        print(f"📏 Initial encoding space size: {initial_space_size:.6f}")

        # Calculate initial minimum Jacobian rank
        initial_min_rank = self.compute_min_jacobian_rank(self.baseline_observations)
        print(f"🔢 Initial min rank of encoder Jacobian: {initial_min_rank}")

        # Compute initial RankMe
        if rankme is not None and self.rankme_obs_path.exists():
            try:
                initial_rankme = self.compute_rankme_metric()
                print(f"🔢 Initial RankMe: {initial_rankme:.2f}")
            except Exception as e:
                print(f"⚠️ Failed to compute initial RankMe: {e}")
                initial_rankme = None
        else:
            print("⚠️ RankMe observations not found or reptrix not installed; RankMe metric disabled for now.")
            initial_rankme = None

        # Store initial measurement in monitoring data
        self.monitoring_data['steps'].append(0)
        self.monitoring_data['encoding_space_size'].append(initial_space_size)
        self.monitoring_data['min_jacobian_rank'].append(initial_min_rank)
        self.monitoring_data['rankme'].append(initial_rankme)
        self.monitoring_data['timestamp'].append(time.time())
        
        # --- Collapse prevention: measure initial random function magnitude ---
        if getattr(self.cfg, 'collapse_prevention', False) and hasattr(self.agent.model, '_random_fn') and self.agent.model._random_fn is not None:
            with torch.no_grad():
                obs = self.baseline_observations.to(self.device)
                random_out = self.agent.model._random_fn(obs)
                initial_rand_mag = random_out.abs().mean().item()
                random_dist = self.pairwise_distance(random_out).min().item()
                random_rank = self.compute_min_jacobian_rank(self.baseline_observations, model=self.agent.model._random_fn)
                self.monitoring_data.setdefault('random_fn_magnitude', []).append(initial_rand_mag)
                self.monitoring_data.setdefault('random_fn_min_distance', []).append(random_dist)
                self.monitoring_data.setdefault('random_fn_min_rank', []).append(random_rank)
                print(f"🎲 Initial random function |output| mean: {initial_rand_mag:.6f}")
                print(f"🎲 Initial random function min pairwise distance: {random_dist:.6f}")
                print(f"🎲 Initial min rank of random function Jacobian: {random_rank}")
        else:
            initial_rand_mag = None
        # -------------------------------------------------------------
        
        # Generate GIFs immediately after creating baseline observations
        print("🎬 Rendering baseline observation GIFs...")
        self.save_observation_gifs(max_gifs=5)
        
        return self.baseline_observations
    
    def _generate_baseline_observations(self):
        """Generate diverse baseline observations using different seeds with consecutive frames"""
        print("🌱 Generating baseline diverse observations with consecutive frames...")
        
        observations = []
        observation_frames = []  # Store individual frames for GIF generation
        
        with torch.no_grad():
            for i, seed in enumerate(self.seeds):
                try:
                    # Reset environment with different seed
                    if hasattr(self.env, 'seed'):
                        self.env.seed(seed)
                    
                    # Reset and clear frame buffer to start fresh
                    obs = self.env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]  # Handle new gym API
                    
                    # Generate 3-5 random actions to get consecutive frames
                    num_actions = np.random.randint(3, 6)  # Random between 3-5 actions
                    frame_sequence = []
                    
                    for step in range(num_actions):
                        action = self.env.action_space.sample()
                        # convert action to tensor
                        action = torch.tensor(action)
                        obs, _, _, _ = self.env.step(action)
                        
                        # Store individual frames for visualization (if RGB environment)
                        if hasattr(self.env, 'render') and step >= num_actions - 3:
                            try:
                                frame = self.env.render()
                                if frame is not None:
                                    frame_sequence.append(frame)
                            except:
                                pass  # Skip if render fails
                    
                    # Use the final observation (which contains consecutive 3 frames)
                    observations.append(obs)
                    observation_frames.append(frame_sequence)
                        
                except Exception as e:
                    print(f"⚠️ Warning: Failed to generate consecutive frames with seed {seed}: {e}")
                    # Fallback: reset and take a few steps
                    try:
                        obs = self.env.reset()
                        if isinstance(obs, tuple):
                            obs = obs[0]
                        for _ in range(3):  # Take 3 steps minimum
                            action = self.env.action_space.sample()
                            obs, _, _, _ = self.env.step(action)
                        observations.append(obs)
                        observation_frames.append([])  # Empty frame sequence for fallback
                    except Exception as e2:
                        print(f"❌ Complete failure for seed {seed}: {e2}")
                        continue
        
        # Store frames for potential GIF generation
        self.observation_frames = observation_frames
        
        # Convert to tensor with proper handling
        tensor_observations = []
        for obs in observations:
            try:
                if isinstance(obs, torch.Tensor):
                    # Already a tensor, just move to device
                    tensor_obs = obs.to(self.device).float()
                elif isinstance(obs, np.ndarray):
                    # Convert numpy array to tensor
                    tensor_obs = torch.from_numpy(obs.copy()).to(self.device).float()
                else:
                    # Convert other types to numpy first, then to tensor
                    tensor_obs = torch.from_numpy(np.array(obs)).to(self.device).float()
                
                tensor_observations.append(tensor_obs)
            except Exception as e:
                print(f"   Warning: Failed to convert observation to tensor: {e}")
                continue
        
        if len(tensor_observations) == 0:
            raise RuntimeError("No valid observations generated!")
            
        self.baseline_observations = torch.stack(tensor_observations)
        
        print(f"✅ Generated {len(observations)} diverse observations with consecutive frames")
        print(f"   Final shape: {self.baseline_observations.shape}")
        print(f"   Each observation contains 3 consecutive frames from random actions")
        
        # Compute initial baseline encodings and return
        self._update_baseline_encodings()
        return self.baseline_observations
    
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
        
        # Compute only the encoding space size
        space_size = self.pairwise_distance(self.baseline_encodings).mean().item()
        # Compute minimum Jacobian rank
        min_rank = self.compute_min_jacobian_rank(self.baseline_observations)
        
        # Compute RankMe metric if available
        rankme_metric = None
        if rankme is not None and self.rankme_obs_path.exists():
            try:
                rankme_metric = self.compute_rankme_metric()
            except Exception as e:
                print(f"⚠️ Failed to compute RankMe: {e}")
        
        # Store results (simplified)
        metrics = {
            'step': step,
            'encoding_space_size': space_size,
            'min_jacobian_rank': float(min_rank),  # cast to float for downstream reductions
            'rankme': rankme_metric,
        }
        
        # Update monitoring data
        self.monitoring_data['steps'].append(step)
        self.monitoring_data['encoding_space_size'].append(space_size)
        self.monitoring_data['min_jacobian_rank'].append(min_rank)
        self.monitoring_data['rankme'].append(rankme_metric)
        self.monitoring_data['timestamp'].append(time.time())
        
        # Print summary
        print(f"   Encoding space size: {space_size:.6f}")
        print(f"   Minimum Jacobian rank: {float(min_rank)}")
        if rankme_metric is not None:
            print(f"   RankMe: {rankme_metric:.2f}")
        # print(f"   Monitoring took {metrics['monitoring_time']:.2f}s")
        
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
        space_sizes = np.array(self.monitoring_data['encoding_space_size'])
        ranks = np.array(self.monitoring_data['min_jacobian_rank'])
        rankmes = np.array([m if m is not None else np.nan for m in self.monitoring_data['rankme']])

        # Subplot 1: encoding space size
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(steps, space_sizes, 'b-o', linewidth=2, markersize=4)
        ax1.set_title('Encoding Space Size', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Avg. Pairwise Distance')
        ax1.grid(True, alpha=0.3)

        # Subplot 2: min Jacobian rank
        ax2 = plt.subplot(2, 2, 2)
        ax2.plot(steps, ranks, 'g-o', linewidth=2, markersize=4)
        ax2.set_title('Minimum Jacobian Rank', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Rank')
        ax2.grid(True, alpha=0.3)

        # Subplot 3: RankMe
        if not np.all(np.isnan(rankmes)):
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(steps, rankmes, 'm-o', linewidth=2, markersize=4)
            ax3.set_title('RankMe', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('RankMe')
            ax3.grid(True, alpha=0.3)

        # Subplot 4: Eval reward (from CSV)
        eval_csv = Path(self.cfg.work_dir) / 'eval.csv' if hasattr(self.cfg, 'work_dir') else None
        if eval_csv is not None and eval_csv.exists():
            try:
                import pandas as pd
                eval_df = pd.read_csv(eval_csv)
                if 'step' in eval_df.columns and 'reward' in eval_df.columns:
                    ax4 = plt.subplot(2, 2, 4)
                    ax4.plot(eval_df['step'], eval_df['reward'], 'c-')
                    ax4.set_title('Eval Reward', fontsize=12, fontweight='bold')
                    ax4.set_xlabel('Training Steps')
                    ax4.set_ylabel('Reward')
                    ax4.grid(True, alpha=0.3)
            except Exception as e:
                print(f"⚠️ Failed to load eval.csv for plotting: {e}")

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

    def compute_rankme_metric(self) -> float:
        """Compute RankMe over a subset of saved observations."""
        if rankme is None:
            raise RuntimeError("reptrix not installed – cannot compute RankMe")

        obs_tensor = self._get_rankme_observations()
        num_samples = min(self.rankme_samples, obs_tensor.shape[0])
        # Random but deterministic sample (seed fixed)
        idx = torch.randperm(obs_tensor.shape[0], device='cpu')[:num_samples]
        obs_sample = obs_tensor[idx].to(self.device)  # Move once to GPU

        # Encode in large batches on GPU
        encodings = []
        with torch.no_grad():
            for i in range(0, num_samples, self.rankme_batch_size):
                batch = obs_sample[i:i+self.rankme_batch_size]
                if self.cfg.multitask:
                    raise NotImplementedError("RankMe not implemented for multi-task encoders")
                enc = self.encoder(batch)
                encodings.append(enc.cpu())
        encodings = torch.cat(encodings, dim=0)

        metric_val = rankme.get_rankme(encodings)
        return float(metric_val)
    
    def generate_simple_report(self) -> str:
        """Generate a simple text report focusing on encoding space"""
        if not self.monitoring_data['steps']:
            return "No monitoring data available."
        
        step = self.monitoring_data['steps'][-1]
        space_size = self.monitoring_data['encoding_space_size'][-1]
        min_rank = self.monitoring_data['min_jacobian_rank'][-1]
        
        # Calculate trend
        if len(self.monitoring_data['encoding_space_size']) > 1:
            initial_size = self.monitoring_data['encoding_space_size'][0]
            change_percent = ((space_size - initial_size) / initial_size) * 100
            trend = "Increasing" if change_percent > 5 else "Decreasing" if change_percent < -5 else "Stable"
        else:
            change_percent = 0
            trend = "N/A"
        
        report = f"""
📊 SIMPLE ENCODING SPACE REPORT - Step {step}
{'='*50}
Current Encoding Space Size: {space_size:.6f}
Minimum Jacobian Rank: {float(min_rank)}
Change from start: {change_percent:+.2f}%
Trend: {trend}

📈 Data Points Collected: {len(self.monitoring_data['steps'])}
🔄 Monitoring Frequency: every {self.monitor_freq} steps
💾 Data saved to: {self.save_dir}

📝 Interpretation:
  • Higher values = More diverse encodings
  • Lower values = More similar encodings (potential collapse)
  • Stable trend = Healthy encoding space
  • Decreasing trend = Potential collapse risk
"""
        return report
    
    def save_observation_gifs(self, max_gifs: int = 5):
        """Save GIFs of the first few observation sequences for visualization"""
        if not hasattr(self, 'observation_frames') or not self.observation_frames:
            print("⚠️ No observation frames available for GIF generation")
            return
        
        print(f"🎬 Saving observation GIFs...")
        gif_dir = self.save_dir / "observation_gifs"
        gif_dir.mkdir(exist_ok=True)
        
        saved_gifs = 0
        for i, frame_sequence in enumerate(self.observation_frames[:max_gifs]):
            if len(frame_sequence) >= 3:  # Only save if we have enough frames
                try:
                    gif_path = gif_dir / f"consecutive_frames_{i:03d}_seed_{self.seeds[i]}.gif"
                    
                    # Convert frames to proper format for imageio
                    gif_frames = []
                    for frame in frame_sequence:
                        if isinstance(frame, np.ndarray):
                            # Ensure frame is in uint8 format
                            if frame.dtype != np.uint8:
                                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
                            gif_frames.append(frame)
                    
                    # Save GIF with slower frame rate for better visualization
                    imageio.mimsave(gif_path, gif_frames, duration=0.5, loop=0)
                    saved_gifs += 1
                        
                except Exception as e:
                    print(f"   Failed to save GIF {i}: {e}")
        
        print(f"✅ Saved {saved_gifs} GIFs to {gif_dir}")

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
