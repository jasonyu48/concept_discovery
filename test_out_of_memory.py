import argparse, torch, numpy as np
from pathlib import Path
from types import SimpleNamespace
from tdmpc2.simple_encoding_space_monitor import SimpleEncodingSpaceMonitor

# Helper config class that supports both attribute and .get access
class Cfg(SimpleNamespace):
    """Simple config namespace that also mimics dict.get"""

    def get(self, key, default=None):
        return getattr(self, key, default)

# --- dummy env ----------------------------------------------------
class DummyActionSpace:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def sample(self):
        return np.random.randn(self.action_dim).astype(np.float32)

class DummyEnv:
    def __init__(self, action_dim):
        self.action_dim = action_dim
        self.action_space = DummyActionSpace(action_dim)

    def rand_act(self):
        return torch.randn(self.action_dim)

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        # Return random observation (C,H,W)
        return np.random.randn(9, 64, 64).astype(np.float32)

    def step(self, action):
        # Return next_obs, reward, done, info placeholder
        next_obs = np.random.randn(9, 64, 64).astype(np.float32)
        reward = 0.0
        done = False
        info = {}
        return next_obs, reward, done, info

# --- dummy model / agent -----------------------------------------
class DummyModel(torch.nn.Module):
    def __init__(self, obs_dim, latent_dim=256, num_bins=101):
        super().__init__()
        self.fc = torch.nn.Linear(obs_dim, latent_dim)
        self.latent_dim = latent_dim
        self.num_bins   = num_bins
    # encoder used by monitor
    def encode(self, obs, task=None):
        flat = obs.view(obs.size(0), -1)
        return self.fc(flat)
    # Q-function used for Lipschitz & P-space
    def Q(self, z_flat, a_flat, task=None, return_type="all", detach=True):
        #  shape expected by monitor: (ensemble, B, num_bins)
        B = z_flat.shape[0]
        return torch.randn(1, B, self.num_bins, device=z_flat.device)
    # Alias forward to encode so model(x) works
    def forward(self, obs):
        return self.encode(obs)

class DummyAgent:
    def __init__(self, model): self.model = model
    def save(self, path): torch.save(self.model.state_dict(), path)

# --- ultra-light dummy buffer ------------------------------------
class DummyStorage:
    def __init__(self, obs): self._obs = obs
    def __getitem__(self, sl):
        return {"obs": self._obs[sl]}

class DummyBuffer:
    def __init__(self, obs):
        self._steps_in_buffer = obs.shape[0]
        dummy = SimpleNamespace()
        dummy._storage = DummyStorage(obs)
        self._buffer = dummy
        self.num_eps = 1

# -----------------------------------------------------------------
def main(args):
    torch.manual_seed(0); np.random.seed(0)

    C, H, W = 9, 64, 64
    N        = args.n_steps
    device   = torch.device(args.device)

    # Allocate all observations on the *CPU* to mimic default buffer choice
    obs = torch.randn(N, C, H, W)

    # Create a small dummy observation file for baseline sampling
    rankme_path = Path("dummy_rankme_obs.pt")
    if not rankme_path.exists():
        torch.save(torch.randn(1000, 9, 64, 64), rankme_path)

    # ----- build monitor --------------------------------------------------
    cfg = Cfg(
        monitor_freq          = 2000,
        dim_monitor_steps     = 5000,
        dim_encode_batch_size = 1024,
        dim_pd_batch_size     = 1024,
        dim_pd_max_samples    = 1000000,
        action_dim            = 6,
        num_q                 = 1,
        num_bins              = 101,
        multitask             = False,
        task_dim              = 0,
        seed                  = 0,
        work_dir              = ".",
        exp_name              = "mem_test",
        enable_decoder        = False,
        device                = args.device,
        rankme_obs_path       = str(rankme_path),
        task                  = "dummy",
    )

    env    = DummyEnv(cfg.action_dim)
    model  = DummyModel(obs_dim=C*H*W).to(device)
    agent  = DummyAgent(model)
    buffer = DummyBuffer(obs)

    monitor = SimpleEncodingSpaceMonitor(
        cfg     = cfg,
        encoder = model,          # same object in this dummy setup
        env     = env,
        agent   = agent,
        buffer  = buffer,
        device  = args.device,
        save_dir= Path("tmp_monitor"),
    )

    torch.cuda.reset_peak_memory_stats(device)
    print("↪ running compute_full_dim_metrics ...")
    monitor.compute_full_dim_metrics()
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"✅ peak GPU memory: {peak:.2f} GB")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_steps", type=int, default=100_000,
                   help="number of steps in dummy buffer")
    p.add_argument("--device",  type=str, default="cuda",
                   help="cuda / cpu")
    main(p.parse_args())