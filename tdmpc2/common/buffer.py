import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from collections import deque


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
		self._capacity = min(cfg.buffer_size, cfg.steps)
		self._sampler = SliceSampler(
			num_slices=self.cfg.batch_size,
			end_key=None,
			traj_key='episode',
			truncated_key=None,
			strict_length=True,
			cache_values=cfg.multitask,
		)
		self._batch_size = cfg.batch_size * (cfg.horizon+1)
		self._num_eps = 0
		
		# Q-function sampling mask parameters (per-episode visibility)
		self._q_sample_ratio = self.cfg.q_sample_ratio
		# Fixed-size book-keeping controlled by capacity; see _register_episode
		# Each entry: (episode_id:int, remaining_steps:int, visible:bool)
		self._episode_queue = deque()  # oldest → newest
		self._episode_visible = {}              # episode_id -> bool
		self._steps_in_buffer = 0               # number of env steps currently resident
		self._q_mask_episodes = 0               # number of *visible* resident episodes
		# NOTE: we no longer store a growing tensor _q_mask. Visibility is queried via dict.

		# ---------------------------------------------------------------------
		# Observation collection for RankMe / representation analysis
		# ---------------------------------------------------------------------
		# Users can enable this by setting `save_obs_for_rankme=True` in cfg.
		# The monitor will save the *first* `obs_save_max_samples` observations
		# that enter the replay buffer (across *all* timesteps and episodes).
		# When the quota is reached, they are written to disk once and freed
		# from memory to minimise overhead during training.
		self._obs_collect_enabled = getattr(cfg, 'save_obs_for_rankme', False)
		self._obs_save_limit = int(getattr(cfg, 'obs_save_max_samples', 30000))
		if self._obs_collect_enabled:
			from pathlib import Path
			# Default save directory as requested: /scratch//obs_data/{task}/{exp_name}
			default_dir = f"/scratch//obs_data/{getattr(cfg, 'task', 'unknown')}/{cfg.exp_name}"
			_obs_dir = Path(getattr(cfg, 'obs_save_dir', default_dir))
			_obs_dir.mkdir(parents=True, exist_ok=True)
			self._obs_save_path = _obs_dir / "observations.pt"
			self._obs_buffer = []  # Temporarily holds collected observations
			self._obs_saved = 0
			print(f"[Buffer] Observation collection ENABLED – will save the first {self._obs_save_limit:,} observations to {self._obs_save_path}")
		else:
			self._obs_save_path = None

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes EVER inserted into the buffer."""
		return self._num_eps

	@property
	def resident_eps(self):
		"""Return the number of episodes whose steps are still stored in the buffer."""
		return len(self._episode_queue)

	def _reserve_buffer(self, storage):
		"""
		Reserve a buffer with the given storage.
		"""
		return ReplayBuffer(
			storage=storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=0,
			batch_size=self._batch_size,
		)

	def _init(self, tds):
		"""Initialize the replay buffer. Use the first episode to estimate storage requirements."""
		print(f'Buffer capacity: {self._capacity:,}')
		
		# Check if CUDA is available before accessing GPU memory info
		if torch.cuda.is_available():
			mem_free, _ = torch.cuda.mem_get_info()
			bytes_per_step = sum([
					(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
					else sum([x.numel()*x.element_size() for x in v.values()])) \
				for v in tds.values()
			]) / len(tds)
			total_bytes = bytes_per_step*self._capacity
			print(f'Storage required: {total_bytes/1e9:.2f} GB')
			# Heuristic: decide whether to use CUDA or CPU memory
			storage_device = 'cuda:0' if 5*total_bytes < mem_free else 'cpu'
		else:
			# CUDA not available, use CPU memory
			bytes_per_step = sum([
					(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
					else sum([x.numel()*x.element_size() for x in v.values()])) \
				for v in tds.values()
			]) / len(tds)
			total_bytes = bytes_per_step*self._capacity
			print(f'Storage required: {total_bytes/1e9:.2f} GB')
			storage_device = 'cpu'
			print('CUDA not available, using CPU memory for storage.')
		
		print(f'Using {storage_device.upper()} memory for storage.')
		self._storage_device = torch.device(storage_device)
		return self._reserve_buffer(
			LazyTensorStorage(self._capacity, device=self._storage_device)
		)

	def load(self, td):
		"""
		Load a batch of episodes into the buffer. This is useful for loading data from disk,
		and is more efficient than adding episodes one by one.
		"""
		# td has shape (num_eps, T, ...)
		num_new_eps = len(td)
		episode_lengths = td['reward'].shape[1]
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		# Register each episode before flattening
		for i in range(num_new_eps):
			self._register_episode(self._num_eps + i, episode_lengths)
		# Flatten and write to storage
		td_flat = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td_flat)
		self._num_eps += num_new_eps
		
		# Observation collection when bulk loading data
		if self._obs_collect_enabled and self._obs_saved < self._obs_save_limit:
			self._collect_observations(td_flat.get('obs', None))
		
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		ep_id = self._num_eps
		# Episode length (timesteps)
		ep_len = td['reward'].shape[0]
		td['episode'] = torch.full_like(td['reward'], ep_id, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		
		# Register episode for visibility bookkeeping
		self._register_episode(ep_id, ep_len)
		
		# -------------------------------------------------------------
		# Observation collection (per-episode) – BEFORE any early exit.
		# -------------------------------------------------------------
		if self._obs_collect_enabled and self._obs_saved < self._obs_save_limit:
			self._collect_observations(td.get('obs', None))
		
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", "reward_pre", strict=False).to(self._device, non_blocking=True)
		obs = td.get('obs').contiguous()
		action = td.get('action')[1:].contiguous()
		reward = td.get('reward')[1:].unsqueeze(-1).contiguous()
		terminated = td.get('terminated', None)
		if terminated is not None:
			terminated = td.get('terminated')[1:].unsqueeze(-1).contiguous()
		else:
			terminated = torch.zeros_like(reward)
		task = td.get('task', None)
		if task is not None:
			task = task[0].contiguous()
		# Optional observation type for analysis (e.g., object count per step)
		obs_type = td.get('obs_type', None)
		if obs_type is not None:
			obs_type = obs_type[1:].contiguous()  # align with (T) after shift
		# Optional pre-action reward target
		reward_pre = td.get('reward_pre', None)
		if reward_pre is not None:
			reward_pre = reward_pre[1:].unsqueeze(-1).contiguous()
		return obs, action, reward, terminated, task, obs_type, reward_pre

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		
		# Extract episode IDs before processing
		episode_ids = td.get('episode')[0].contiguous()  # Get episode IDs for each sample
		
		# Process the batch normally
		obs, action, reward, terminated, task, obs_type, reward_pre = self._prepare_batch(td)
		
		# Generate Q-function mask for this batch
		q_mask = self.get_q_mask_for_batch(episode_ids)
		
		return obs, action, reward, terminated, task, q_mask, obs_type, reward_pre

	def get_q_mask_for_batch(self, episode_ids):
		"""
		Get a mask indicating which samples in the current batch should be used for Q-function training.
		
		Args:
			episode_ids: Tensor of episode IDs for each sample in the batch
		
		Returns:
			torch.Tensor: Boolean mask of shape (batch_size,) indicating which samples to use for Q-function
		"""
		if self._q_sample_ratio >= 1.0:
			# Use all samples
			return torch.ones(len(episode_ids), dtype=torch.bool, device=self._device)
		
		batch_mask = torch.tensor([self._episode_visible.get(int(ep_id), False) for ep_id in episode_ids],
								 dtype=torch.bool, device=self._device)
		return batch_mask


	@property
	def q_visible_episodes(self):
		"""Return the number of episodes visible to Q-function training."""
		return self._q_mask_episodes

	# -----------------------------------------------------------------
	# Observation collection helpers
	# -----------------------------------------------------------------
	def _collect_observations(self, obs_tensor):
		"""Collect observations until the preset limit is reached.

		Args:
			obs_tensor (torch.Tensor or None): Tensor of observations with shape
				(..., *obs_shape). If None, nothing is done.
		"""
		if obs_tensor is None:
			return
		if self._obs_saved >= self._obs_save_limit:
			return  # Already finished

		obs_cpu = obs_tensor.detach().cpu()

		# Pixel observations typically have at least 4 dims (T,B,C,H,W) or (T,C,H,W).
		# We want to flatten the leading dims (time, batch) but keep (C,H,W).
		if obs_cpu.shape[-3:] in [(9, 64, 64), (3, 64, 64)]:
			obs_cpu = obs_cpu.reshape(-1, *obs_cpu.shape[-3:])  # (-1, C, H, W)
		else:
			raise ValueError(f"Unknown observation shape: {obs_cpu.shape}")

		# Determine how many we still need
		remaining = self._obs_save_limit - self._obs_saved
		obs_to_add = obs_cpu[:remaining]
		self._obs_buffer.append(obs_to_add)
		self._obs_saved += obs_to_add.shape[0]

		if self._obs_saved >= self._obs_save_limit:
			self._flush_observations_to_disk()

	def _flush_observations_to_disk(self):
		"""Save the collected observations to disk as a single torch file."""
		if not self._obs_buffer or self._obs_save_path is None:
			return
		import torch
		obs_cat = torch.cat(self._obs_buffer, dim=0)
		try:
			torch.save(obs_cat, self._obs_save_path)
			print(f"[Buffer] ✅ Saved {obs_cat.shape[0]:,} observations to {self._obs_save_path}")
		except Exception as e:
			print(f"[Buffer] ⚠️ Failed to save observations: {e}")
		finally:
			# Free memory
			self._obs_buffer = []

	def _register_episode(self, episode_id: int, length: int):
		"""Step-exact bookkeeping of episode residency and visibility.

		Args:
		    episode_id: Global, monotonically increasing ID.
		    length:     Number of environment steps of this episode.

		The method emulates the wrap-around effect of the underlying circular
		storage:
		* ``self._steps_in_buffer`` tracks how many *time-steps* are currently
		  resident.
		* When adding *length* new steps, we compute how many old steps will be
		  overwritten (``overwrite = max(0, steps_in_buffer + length − capacity)``)
		  and deduct them from the front of ``self._episode_queue`` *partially if
		  necessary*.
		* We keep an entry for an episode until **all** of its remaining steps have
		  been overwritten.  Only then do we drop its visibility flag.
		"""
		# ------------------------------------------------------------------
		# 1. Evict (possibly partially) the oldest episodes to make room
		# ------------------------------------------------------------------
		overwrite = max(0, self._steps_in_buffer + length - self._capacity)

		while overwrite > 0 and self._episode_queue:
			old = self._episode_queue[0]  # peek, do not pop yet
			old_id, old_rem_steps, old_vis = old
			if old_rem_steps <= overwrite:
				# Entire old episode will be removed
				self._episode_queue.popleft()
				overwrite -= old_rem_steps
				self._steps_in_buffer -= old_rem_steps
				if old_vis:
					self._q_mask_episodes -= 1
				self._episode_visible.pop(old_id, None)
			else:
				# Partially overwrite this episode; adjust its remaining length
				new_rem = old_rem_steps - overwrite
				self._episode_queue.popleft()
				self._episode_queue.appendleft((old_id, new_rem, old_vis))
				self._steps_in_buffer -= overwrite
				overwrite = 0

		# ------------------------------------------------------------------
		# 2. Decide visibility for the new episode
		# ------------------------------------------------------------------
		# We want the *running* fraction of visible episodes to match the
		# target q_sample_ratio as closely as possible.  Instead of a
		# Bernoulli coin-flip (which introduces random drift), use a
		# deterministic rule: make the new episode visible **iff** the
		# current number of visible episodes is below the rounded target
		# count after adding this episode.

		if self._q_sample_ratio >= 1.0:
			visible = True
		else:
			total_eps_after_insert = len(self._episode_visible) + 1  # +1 for the incoming episode
			target_visible = round(total_eps_after_insert * self._q_sample_ratio)
			# current visible BEFORE insertion
			current_visible = self._q_mask_episodes
			# Decide deterministically to hit the target as closely as possible
			visible = current_visible < target_visible

		# ------------------------------------------------------------------
		# 3. Insert metadata for the new episode
		# ------------------------------------------------------------------
		self._episode_queue.append((episode_id, length, visible))
		self._steps_in_buffer += length

		self._episode_visible[episode_id] = visible
		if visible:
			self._q_mask_episodes += 1

		# Sanity check: should never exceed capacity
		assert self._steps_in_buffer <= self._capacity, "Bookkeeping error: steps exceed capacity"
		return visible
