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
		self._q_mask_episodes = 0  # Track which episodes are visible to Q-function

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
		return self._num_eps

	def add(self, td):
		"""Add an episode to the buffer."""
		td['episode'] = torch.full_like(td['reward'], self._num_eps, dtype=torch.int64)
		if self._num_eps == 0:
			self._buffer = self._init(td)
		self._buffer.extend(td)
		self._num_eps += 1
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
		return self._prepare_batch(td)

	def sample_for_q(self):
		"""Sample a batch of subsequences for Q-function training with optional masking."""
		if self._q_sample_ratio >= 1.0 or self._q_mask is None or self._num_eps == 0:
			# Use regular sampling if no masking needed
			return self.sample()
		
		# Get visible episode indices
		visible_episodes = torch.where(self._q_mask)[0]
		
		if len(visible_episodes) == 0:
			# Fallback to regular sampling if no episodes are visible
			print("Warning: No episodes visible to Q-function, using regular sampling")
			return self.sample()
		
		# Try to sample a batch that contains data from visible episodes
		max_attempts = 10
		best_td = None
		best_visible_count = 0
		
		for attempt in range(max_attempts):
			td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
			
			# Check episode IDs in the sampled batch
			episode_ids = td.get('episode')[0]  # Shape: (batch_size,)
			
			# Count how many samples belong to visible episodes
			visible_count = 0
			for visible_ep in visible_episodes:
				visible_count += (episode_ids == visible_ep).sum().item()
			
			# Keep the batch with most visible samples
			if visible_count > best_visible_count:
				best_td = td
				best_visible_count = visible_count
			
			# If we found a batch with reasonable number of visible samples, use it
			if visible_count >= self.cfg.batch_size * 0.1:  # At least 10% from visible episodes
				break
		
		if best_td is None:
			# Fallback to regular sampling
			print("Warning: Could not sample from visible episodes, using regular sampling")
			return self.sample()
		
		# Use the best batch we found
		visible_ratio = best_visible_count / self.cfg.batch_size
		#if best_visible_count > 0:
			#print(f"Q-sampling: {best_visible_count}/{self.cfg.batch_size} samples from visible episodes ({visible_ratio:.2f})")
		
		return self._prepare_batch(best_td)

	def update_q_mask(self):
		"""Update mask while keeping existing visible episodes fixed."""
		if self._num_eps == 0:
			self._q_mask = None
			self._q_mask_episodes = 0
			return
		
		if self._q_sample_ratio >= 1.0:
			self._q_mask = torch.ones(self._num_eps, dtype=torch.bool)
			self._q_mask_episodes = self._num_eps
		else:
			old_num_eps = len(self._q_mask) if self._q_mask is not None else 0
			
			if old_num_eps == 0:
				# 第一次创建mask
				num_visible = max(1, int(self._num_eps * self._q_sample_ratio))
				perm = torch.randperm(self._num_eps)
				visible_indices = perm[:num_visible]
				self._q_mask = torch.zeros(self._num_eps, dtype=torch.bool)
				self._q_mask[visible_indices] = True
				self._q_mask_episodes = num_visible
				print(f"Q-function mask created: {self._q_mask_episodes}/{self._num_eps} episodes visible")
				
			elif self._num_eps > old_num_eps:
				# 有新episode添加，扩展mask但保持旧的可见episode不变
				new_episodes_count = self._num_eps - old_num_eps
				old_mask = self._q_mask.clone()
				
				# 计算目标可见episode数量
				target_visible = max(1, int(self._num_eps * self._q_sample_ratio))
				current_visible = old_mask.sum().item()
				
				# 新episode中需要多少个可见
				new_visible_needed = max(0, target_visible - current_visible)
				new_visible = min(new_visible_needed, new_episodes_count)
				
				# 为新episode创建mask（随机选择）
				new_mask_part = torch.zeros(new_episodes_count, dtype=torch.bool)
				if new_visible > 0:
					new_perm = torch.randperm(new_episodes_count)
					new_mask_part[new_perm[:new_visible]] = True
				
				# 合并旧mask和新mask
				self._q_mask = torch.cat([old_mask, new_mask_part])
				self._q_mask_episodes = self._q_mask.sum().item()
				
				print(f"Q-function mask extended: {self._q_mask_episodes}/{self._num_eps} episodes visible "
					  f"(kept {current_visible} old + added {new_visible} new)")
			else:
				# 如果episode数量没变，保持mask不变，只更新计数
				self._q_mask_episodes = self._q_mask.sum().item()
				print(f"Q-function mask unchanged: {self._q_mask_episodes}/{self._num_eps} episodes visible")

	@property
	def q_visible_episodes(self):
		"""Return the number of episodes visible to Q-function training."""
		return self._q_mask_episodes if self._q_mask is not None else self._num_eps
