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
            'encoding_values': [],      # Store actual encoding values for analysis
            'min_jacobian_rank': [],    # New: track minimum Jacobian rank
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
        
        # Compute initial baseline encodings
        self._update_baseline_encodings()
        
        # Calculate and save initial encoding space size
        initial_space_size = self.compute_encoding_space_size(self.baseline_encodings)
        print(f"📏 Initial encoding space size: {initial_space_size:.6f}")

        # Calculate initial minimum Jacobian rank
        initial_min_rank = self.compute_min_jacobian_rank(self.baseline_observations)
        print(f"🔢 Initial minimum Jacobian rank: {initial_min_rank}")

        # Store initial measurement in monitoring data
        self.monitoring_data['steps'].append(0)
        self.monitoring_data['encoding_space_size'].append(initial_space_size)
        self.monitoring_data['encoding_values'].append(self.baseline_encodings.cpu().numpy().copy())
        self.monitoring_data['min_jacobian_rank'].append(initial_min_rank)
        self.monitoring_data['timestamp'].append(time.time())
        
        # Generate GIFs immediately after creating baseline observations
        print("🎬 Rendering baseline observation GIFs...")
        self.save_observation_gifs(max_gifs=5)
        
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
    
    def compute_encoding_space_size(self, encodings: torch.Tensor) -> float:
        """Compute average pairwise distance in encoding space (our main metric)"""
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
    
    def compute_min_jacobian_rank(self, observations: torch.Tensor) -> int:
        """Compute the minimum rank of the encoder Jacobian dE/dx over a batch.
        """
        # Ensure the tensor has gradients enabled
        obs = observations.detach().clone().to(self.device).requires_grad_(True)

        # Define a helper that returns encoder output for a *single* sample
        if hasattr(self.encoder, 'encode'):
            def single_forward(x_single: torch.Tensor):
                task = torch.zeros(1, self.cfg.task_dim, device=self.device)
                # Add batch dim expected by encode, then squeeze back
                return self.encoder.encode(x_single.unsqueeze(0), task).squeeze(0)
        else:
            def single_forward(x_single: torch.Tensor):
                return self.encoder(x_single.unsqueeze(0)).squeeze(0)


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
        space_size = self.compute_encoding_space_size(self.baseline_encodings)
        # Compute minimum Jacobian rank
        min_rank = self.compute_min_jacobian_rank(self.baseline_observations)
        
        # Store results (simplified)
        metrics = {
            'step': step,
            'encoding_space_size': space_size,
            'min_jacobian_rank': float(min_rank),  # cast to float for downstream reductions
        }
        
        # Update monitoring data
        self.monitoring_data['steps'].append(step)
        self.monitoring_data['encoding_space_size'].append(space_size)
        self.monitoring_data['encoding_values'].append(self.baseline_encodings.cpu().numpy().copy())
        self.monitoring_data['min_jacobian_rank'].append(min_rank)
        self.monitoring_data['timestamp'].append(time.time())
        
        # Print summary
        print(f"   Encoding space size: {space_size:.6f}")
        print(f"   Minimum Jacobian rank: {float(min_rank)}")
        # print(f"   Monitoring took {metrics['monitoring_time']:.2f}s")
        
        # Auto-save periodically and generate updated plots
        if step % (self.monitor_freq * 5) == 0:
            self.save_monitoring_data()
            # Generate updated encoding space curve plot
            try:
                self.plot_encoding_space_curve(save_plot=True)
                print(f"   ✅ Updated encoding space curve saved!")
            except Exception as e:
                print(f"   ⚠️ Failed to generate encoding space curve: {e}")
        
        # Save model periodically to same path (every 5 monitoring steps)
        if step % (self.monitor_freq * 5) == 0:
            self._save_model_checkpoint(step)
        
        return metrics
    
    def save_monitoring_data(self):
        """Save monitoring data to disk"""
        save_path = self.save_dir / "simple_monitoring_data.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(self.monitoring_data, f)
        print(f"💾 Saved monitoring data to {save_path}")
    
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

    
    def plot_encoding_space_curve(self, save_plot: bool = True):
        """Create simple plot showing encoding space size over time"""
        if not self.monitoring_data['steps']:
            print("No monitoring data to plot")
            return None
        
        plt.figure(figsize=(12, 6))
        steps = np.array(self.monitoring_data['steps'])
        space_sizes = np.array(self.monitoring_data['encoding_space_size'])
        ranks = np.array(self.monitoring_data['min_jacobian_rank'])
        
        # Main plot: encoding space size
        plt.subplot(1, 2, 1)
        plt.plot(steps, space_sizes, 'b-o', linewidth=2, markersize=4)
        plt.title('Encoding Space Size Over Training', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps')
        plt.ylabel('Avg. Pairwise Distance')
        plt.grid(True, alpha=0.3)
        
        # Statistics annotation
        if len(space_sizes) > 1:
            initial_size = space_sizes[0]
            current_size = space_sizes[-1]
            change_percent = ((current_size - initial_size) / initial_size) * 100
            plt.text(0.02, 0.98,
                     f'Initial: {initial_size:.4f}\nCurrent: {current_size:.4f}\nChange: {change_percent:+.1f}%',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Right subplot: minimum Jacobian rank curve
        plt.subplot(1, 2, 2)
        plt.plot(steps, ranks, 'g-o', linewidth=2, markersize=4)
        plt.title('Minimum Jacobian Rank Over Training', fontsize=14, fontweight='bold')
        plt.xlabel('Training Steps')
        plt.ylabel('Rank')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.save_dir / "encoding_space_curve.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Saved encoding space curve to {plot_path}")
        
        return plt.gcf()
    
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
