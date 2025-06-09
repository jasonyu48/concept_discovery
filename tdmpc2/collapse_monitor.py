# Encoder Collapse Detection and Monitoring System
# Author: Based on exist_check.py and your requirements
# Purpose: Detect encoder collapse using diverse seed observations and covariance matrix rank

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time

class CollapseMonitor:
    """
    Monitor encoder collapse using multiple metrics:
    1. Covariance matrix rank (primary indicator)
    2. Encoding space size (average pairwise distance)
    3. Singular value distribution
    4. Orthogonality violation score
    """
    
    def __init__(
        self, 
        cfg, 
        encoder: torch.nn.Module,
        env,
        device: str = "cuda",
        save_dir: Optional[str] = None
    ):
        self.cfg = cfg
        self.encoder = encoder
        self.env = env
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("collapse_logs")
        self.save_dir.mkdir(exist_ok=True)
        
        # Monitoring configuration
        self.num_seed_obs = 64  # Number of diverse seed observations
        self.monitor_freq = self.cfg.get('exist_check_freq', 2000)  # Monitor every N steps
        self.seeds = list(range(42, 42 + self.num_seed_obs))  # Fixed seeds for reproducibility
        
        # Storage for monitoring data
        self.monitoring_data = {
            'steps': [],
            'covariance_rank': [],
            'effective_rank': [],  # More robust than hard rank
            'encoding_space_size': [],
            'singular_values': [],
            'condition_number': [],
            'collapse_score': [],  # Combined collapse metric
            'timestamp': []
        }
        
        # Generate baseline diverse observations
        self.baseline_observations = None
        self.baseline_encodings = None
        self._generate_baseline_observations()
        
        print(f"🔍 CollapseMonitor initialized:")
        print(f"   - Monitoring frequency: every {self.monitor_freq} steps")
        print(f"   - Seed observations: {self.num_seed_obs}")
        print(f"   - Save directory: {self.save_dir}")
    
    def _generate_baseline_observations(self):
        """Generate diverse baseline observations using different seeds"""
        print("🌱 Generating baseline diverse observations...")
        
        observations = []
        with torch.no_grad():
            for seed in self.seeds:
                try:
                    # Reset environment with different seed
                    if hasattr(self.env, 'seed'):
                        self.env.seed(seed)
                    obs = self.env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]  # Handle new gym API
                    observations.append(obs)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to reset with seed {seed}: {e}")
                    # Fallback: use random action steps
                    obs = self.env.reset()
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    for _ in range(seed % 10):  # Some randomness
                        action = self.env.action_space.sample()
                        obs, _, _, _ = self.env.step(action)
                    observations.append(obs)
        
        # Convert to tensor
        self.baseline_observations = torch.stack([
            torch.from_numpy(obs) if isinstance(obs, np.ndarray) else obs 
            for obs in observations
        ]).to(self.device).float()
        
        print(f" Generated {len(observations)} diverse observations with shape {self.baseline_observations.shape}")
        
        # Compute initial baseline encodings
        self._update_baseline_encodings()
        return self.baseline_observations
    
    def _update_baseline_encodings(self):
        """Update baseline encodings with current encoder state"""
        with torch.no_grad():
            # Handle both single-task and multi-task encoders
            if hasattr(self.encoder, 'encode'):
                # Multi-task encoder
                task = torch.zeros(self.baseline_observations.shape[0], self.cfg.task_dim).to(self.device)
                self.baseline_encodings = self.encoder.encode(self.baseline_observations, task)
            else:
                # Single-task encoder (direct call)
                self.baseline_encodings = self.encoder(self.baseline_observations)
    
    def compute_covariance_rank(self, encodings: torch.Tensor, threshold: float = 1e-6) -> Tuple[int, float, torch.Tensor]:
        """
        Compute covariance matrix rank and related metrics
        
        Returns:
            hard_rank: Number of singular values > threshold
            effective_rank: Sum of normalized singular values (more stable)
            singular_values: All singular values
        """
        # Center the encodings
        centered = encodings - encodings.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov_matrix = torch.mm(centered.T, centered) / (encodings.shape[0] - 1)
        
        # Compute singular values
        try:
            singular_values = torch.linalg.svdvals(cov_matrix)
        except:
            # Fallback for numerical issues
            U, S, V = torch.svd(cov_matrix)
            singular_values = S
        
        # Hard rank (number of significant singular values)
        hard_rank = (singular_values > threshold).sum().item()
        
        # Effective rank (more robust measure)
        normalized_sv = singular_values / singular_values.sum()
        effective_rank = torch.exp(-torch.sum(normalized_sv * torch.log(normalized_sv + 1e-12))).item()
        
        return hard_rank, effective_rank, singular_values
    
    def compute_encoding_space_size(self, encodings: torch.Tensor) -> float:
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
        
        return distances.mean().item()
    
    def compute_collapse_score(self, effective_rank: float, space_size: float, condition_num: float) -> float:
        """
        Compute combined collapse score (0 = collapsed, 1 = healthy)
        """
        expected_rank = min(self.baseline_encodings.shape[0], self.baseline_encodings.shape[1])
        
        # Normalize metrics
        rank_score = effective_rank / expected_rank
        size_score = min(space_size / 10.0, 1.0)  # Normalize by expected size
        condition_score = 1.0 / (1.0 + np.log10(max(condition_num, 1.0)))
        
        # Weighted combination
        collapse_score = 0.5 * rank_score + 0.3 * size_score + 0.2 * condition_score
        return min(collapse_score, 1.0)
    
    def monitor_step(self, step: int) -> Dict[str, float]:
        """
        Perform monitoring at given training step
        
        Returns:
            Dictionary with all computed metrics
        """
        if step % self.monitor_freq != 0:
            return {}
        
        print(f"🔍 Monitoring encoder collapse at step {step}...")
        start_time = time.time()
        
        # Update encodings with current encoder
        self._update_baseline_encodings()
        
        # Compute all metrics
        hard_rank, effective_rank, singular_values = self.compute_covariance_rank(self.baseline_encodings)
        space_size = self.compute_encoding_space_size(self.baseline_encodings)
        condition_num = (singular_values.max() / (singular_values.min() + 1e-12)).item()
        collapse_score = self.compute_collapse_score(effective_rank, space_size, condition_num)
        
        # Store results
        metrics = {
            'step': step,
            'covariance_rank': hard_rank,
            'effective_rank': effective_rank,
            'encoding_space_size': space_size,
            'condition_number': condition_num,
            'collapse_score': collapse_score,
            'max_singular_value': singular_values.max().item(),
            'min_singular_value': singular_values.min().item(),
            'monitoring_time': time.time() - start_time
        }
        
        # Update monitoring data
        self.monitoring_data['steps'].append(step)
        self.monitoring_data['covariance_rank'].append(hard_rank)
        self.monitoring_data['effective_rank'].append(effective_rank)
        self.monitoring_data['encoding_space_size'].append(space_size)
        self.monitoring_data['singular_values'].append(singular_values.cpu().numpy())
        self.monitoring_data['condition_number'].append(condition_num)
        self.monitoring_data['collapse_score'].append(collapse_score)
        self.monitoring_data['timestamp'].append(time.time())
        
        # Print summary
        print(f"   Covariance rank: {hard_rank}/{self.baseline_encodings.shape[1]} (effective: {effective_rank:.2f})")
        print(f"   Encoding space size: {space_size:.4f}")
        print(f"   Condition number: {condition_num:.2e}")
        print(f"   Collapse score: {collapse_score:.4f} (1.0=healthy, 0.0=collapsed)")
        print(f"    Monitoring took {metrics['monitoring_time']:.2f}s")
        
        # Check for collapse warning
        if collapse_score < 0.3:
            print(f"  WARNING: Potential encoder collapse detected! (score: {collapse_score:.4f})")
        elif collapse_score < 0.5:
            print(f" CAUTION: Encoder health declining (score: {collapse_score:.4f})")
        
        # Auto-save periodically
        if step % (self.monitor_freq * 5) == 0:
            self.save_monitoring_data()
        
        return metrics
    
    def save_monitoring_data(self):
        """Save monitoring data to disk"""
        save_path = self.save_dir / "monitoring_data.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(self.monitoring_data, f)
        print(f" Saved monitoring data to {save_path}")
    
    def plot_monitoring_results(self, save_plot: bool = True) -> plt.Figure:
        """Create comprehensive monitoring plots"""
        if not self.monitoring_data['steps']:
            print("No monitoring data to plot")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        steps = np.array(self.monitoring_data['steps'])
        
        # 1. Covariance rank over time
        axes[0, 0].plot(steps, self.monitoring_data['covariance_rank'], 'b-o', label='Hard Rank')
        axes[0, 0].plot(steps, self.monitoring_data['effective_rank'], 'r-s', label='Effective Rank')
        axes[0, 0].set_title('Covariance Matrix Rank')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Rank')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Encoding space size
        axes[0, 1].plot(steps, self.monitoring_data['encoding_space_size'], 'g-o')
        axes[0, 1].set_title('Encoding Space Size')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Average Pairwise Distance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Condition number (log scale)
        axes[0, 2].semilogy(steps, self.monitoring_data['condition_number'], 'purple', marker='o')
        axes[0, 2].set_title('Condition Number (Log Scale)')
        axes[0, 2].set_xlabel('Training Steps')
        axes[0, 2].set_ylabel('Condition Number')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Collapse score
        axes[1, 0].plot(steps, self.monitoring_data['collapse_score'], 'red', marker='o', linewidth=2)
        axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', label='Caution threshold')
        axes[1, 0].axhline(y=0.3, color='red', linestyle='--', label='Collapse threshold')
        axes[1, 0].set_title('Collapse Score (Higher = Healthier)')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Collapse Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1.1)
        
        # 5. Singular value distribution (latest)
        if self.monitoring_data['singular_values']:
            latest_sv = self.monitoring_data['singular_values'][-1]
            axes[1, 1].semilogy(range(len(latest_sv)), latest_sv, 'b-o')
            axes[1, 1].set_title('Latest Singular Values')
            axes[1, 1].set_xlabel('Index')
            axes[1, 1].set_ylabel('Singular Value (Log Scale)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Singular value heatmap over time
        if len(self.monitoring_data['singular_values']) > 1:
            sv_matrix = np.array([sv[:20] for sv in self.monitoring_data['singular_values']])  # Top 20 SVs
            im = axes[1, 2].imshow(sv_matrix.T, aspect='auto', cmap='viridis')
            axes[1, 2].set_title('Singular Values Over Time')
            axes[1, 2].set_xlabel('Monitoring Step')
            axes[1, 2].set_ylabel('Singular Value Index')
            plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.save_dir / f"monitoring_plots_step_{steps[-1]}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f" Saved monitoring plots to {plot_path}")
        
        return fig
    
    def generate_report(self) -> str:
        """Generate a text report of current encoder health"""
        if not self.monitoring_data['steps']:
            return "No monitoring data available."
        
        latest_idx = -1
        step = self.monitoring_data['steps'][latest_idx]
        collapse_score = self.monitoring_data['collapse_score'][latest_idx]
        effective_rank = self.monitoring_data['effective_rank'][latest_idx]
        space_size = self.monitoring_data['encoding_space_size'][latest_idx]
        condition_num = self.monitoring_data['condition_number'][latest_idx]
        
        # Health assessment
        if collapse_score >= 0.7:
            health_status = " HEALTHY"
        elif collapse_score >= 0.5:
            health_status = " MODERATE"
        elif collapse_score >= 0.3:
            health_status = " AT RISK"
        else:
            health_status = " COLLAPSED"
        
        report = f"""
 ENCODER HEALTH REPORT - Step {step}
{'='*50}
Overall Status: {health_status}
Collapse Score: {collapse_score:.4f}/1.0

 Detailed Metrics:
  • Effective Rank: {effective_rank:.2f}/{self.baseline_encodings.shape[1]}
  • Encoding Space Size: {space_size:.4f}
  • Condition Number: {condition_num:.2e}

 Trend Analysis:
  • Steps monitored: {len(self.monitoring_data['steps'])}
  • Monitoring frequency: every {self.monitor_freq} steps
  • Data saved to: {self.save_dir}

 Interpretation:
  • Collapse Score > 0.7: Encoder is healthy
  • Collapse Score 0.5-0.7: Monitor closely
  • Collapse Score 0.3-0.5: At risk of collapse
  • Collapse Score < 0.3: Likely collapsed
"""
        return report

# Integration function for easy use in training loop
def create_collapse_monitor(cfg, encoder, env, save_dir=None) -> CollapseMonitor:
    """Factory function to create collapse monitor"""
    return CollapseMonitor(
        cfg=cfg,
        encoder=encoder,
        env=env,
        device=cfg.get('device', 'cuda'),
        save_dir=save_dir or f"collapse_logs_{cfg.exp_name}"
    )
