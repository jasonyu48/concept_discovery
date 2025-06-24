import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


class Buffer():
	"""
	Replay buffer for TD-MPC2 training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(self, cfg):
		self.cfg = cfg
		self._device = torch.device('cuda:0')
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
		
		# Q-function sampling mask parameters
		self._q_sample_ratio = getattr(cfg, 'q_sample_ratio', 1.0)
		self._q_mask = None
		self._q_mask_episodes = 0

	@property
	def capacity(self):
		"""Return the capacity of the buffer."""
		return self._capacity

	@property
	def num_eps(self):
		"""Return the number of episodes in the buffer."""
		return self._num_eps

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
		mem_free, _ = torch.cuda.mem_get_info()
		bytes_per_step = sum([
				(v.numel()*v.element_size() if not isinstance(v, TensorDict) \
				else sum([x.numel()*x.element_size() for x in v.values()])) \
			for v in tds.values()
		]) / len(tds)
		total_bytes = bytes_per_step*self._capacity
		print(f'Storage required: {total_bytes/1e9:.2f} GB')
		# Heuristic: decide whether to use CUDA or CPU memory
		storage_device = 'cuda:0' if 2.5*total_bytes < mem_free else 'cpu'
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
		num_new_eps = len(td)
		episode_idx = torch.arange(self._num_eps, self._num_eps+num_new_eps, dtype=torch.int64)
		td['episode'] = episode_idx.unsqueeze(-1).expand(-1, td['reward'].shape[1])
		if self._num_eps == 0:
			self._buffer = self._init(td[0])
		td = td.reshape(td.shape[0]*td.shape[1])
		self._buffer.extend(td)
		self._num_eps += num_new_eps
		
		# Update Q-function mask when new episodes are added
		self._update_q_mask_on_new_episodes()
		
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
		
		# Update Q-function mask when new episode is added
		self._update_q_mask_on_new_episodes()
		
		return self._num_eps

	def _prepare_batch(self, td):
		"""
		Prepare a sampled batch for training (post-processing).
		Expects `td` to be a TensorDict with batch size TxB.
		"""
		td = td.select("obs", "action", "reward", "terminated", "task", strict=False).to(self._device, non_blocking=True)
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
		return obs, action, reward, terminated, task

	def sample(self):
		"""Sample a batch of subsequences from the buffer."""
		td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
		
		# Extract episode IDs before processing
		episode_ids = td.get('episode')[0].contiguous()  # Get episode IDs for each sample
		
		# Process the batch normally
		obs, action, reward, terminated, task = self._prepare_batch(td)
		
		# Generate Q-function mask for this batch
		q_mask = self.get_q_mask_for_batch(episode_ids)
		
		return obs, action, reward, terminated, task, q_mask

	def get_q_mask_for_batch(self, episode_ids):
		"""
		Get a mask indicating which samples in the current batch should be used for Q-function training.
		
		Args:
			episode_ids: Tensor of episode IDs for each sample in the batch
		
		Returns:
			torch.Tensor: Boolean mask of shape (batch_size,) indicating which samples to use for Q-function
		"""
		if self._q_sample_ratio >= 1.0 or self._q_mask is None:
			# Use all samples
			return torch.ones(len(episode_ids), dtype=torch.bool, device=self._device)
		
		# Create mask based on episode visibility
		batch_mask = torch.zeros(len(episode_ids), dtype=torch.bool, device=self._device)
		
		# For each sample in batch, check if its episode is visible to Q-function
		for i, ep_id in enumerate(episode_ids):
			if ep_id < len(self._q_mask) and self._q_mask[ep_id]:
				batch_mask[i] = True
		
		return batch_mask

	def _update_q_mask_on_new_episodes(self):
		"""Update the mask for Q-function training when new episodes are added.
		Keeps existing visible episodes unchanged, only assigns visibility to new episodes.
		Allows for slight imprecision in ratio when total episodes is odd.
		"""
		if self._num_eps == 0:
			self._q_mask = None
			self._q_mask_episodes = 0
			return
		
		if self._q_sample_ratio >= 1.0:
			self._q_mask = torch.ones(self._num_eps, dtype=torch.bool)
			self._q_mask_episodes = self._num_eps
			return
		
		old_num_eps = len(self._q_mask) if self._q_mask is not None else 0
		
		if old_num_eps == 0:
			# First time creating mask - initial setup
			# Use round() instead of int() to handle odd numbers better
			num_visible = round(self._num_eps * self._q_sample_ratio)
			self._q_mask = torch.zeros(self._num_eps, dtype=torch.bool)
			if num_visible > 0:
				perm = torch.randperm(self._num_eps)
				visible_indices = perm[:num_visible]
				self._q_mask[visible_indices] = True
			self._q_mask_episodes = num_visible
			
		elif self._num_eps > old_num_eps:
			# New episodes added - extend mask but keep old visible episodes unchanged
			new_episodes_count = self._num_eps - old_num_eps
			old_mask = self._q_mask.clone()
			current_visible = old_mask.sum().item()
			
			# Calculate target visible episodes with rounding for better handling of odd numbers
			target_visible = round(self._num_eps * self._q_sample_ratio)
			
			# How many of the new episodes should be visible
			new_visible_needed = max(0, target_visible - current_visible)
			new_visible = min(new_visible_needed, new_episodes_count)
			
			# Create mask for new episodes (randomly select)
			new_mask_part = torch.zeros(new_episodes_count, dtype=torch.bool)
			if new_visible > 0:
				new_perm = torch.randperm(new_episodes_count)
				new_mask_part[new_perm[:new_visible]] = True
			
			# Combine old and new masks
			self._q_mask = torch.cat([old_mask, new_mask_part])
			self._q_mask_episodes = self._q_mask.sum().item()

	@property
	def q_visible_episodes(self):
		"""Return the number of episodes visible to Q-function training."""
		return self._q_mask_episodes if self._q_mask is not None else self._num_eps
