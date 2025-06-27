from copy import deepcopy

import torch
import torch.nn as nn

from common import layers, math, init
from common import full_rank_layers
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class RandomPatchTransformer(nn.Module):
	"""Random Transformer Encoder on image patches.
	Takes input images of shape (B, C, 64, 64) and returns a representation
	of dimension ``d_model``. All parameters are frozen (requires_grad=False).
	"""
	def __init__(self, cfg: any, in_channels: int, patch_size: int, d_model: int):
		super().__init__()
		self.cfg = cfg
		self.patch_size = patch_size
		self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
		self.in_proj = nn.Linear(in_channels * patch_size * patch_size, d_model)
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=d_model,
			nhead=1,
			dim_feedforward=d_model,
			dropout=0.0,
			activation=nn.Mish(inplace=False),
			norm_first=False,
			batch_first=True,
		)
		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=None)
		self.scale_factor = cfg.scale_factor

		# Freeze parameters
		for p in self.parameters():
			p.requires_grad = False
		self.eval()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Forward pass.
		Args:
			x: Tensor of shape (B, C, 64, 64)
		Returns:
			Tensor of shape (B, d_model)
		"""
		B, C, H, W = x.shape
		assert H == 64 and W == 64, "RandomPatchTransformer expects 64x64 input size"
		x = self.unfold(x).transpose(1, 2)  # (B, N_patches, patch_dim)
		x = self.in_proj(x)  # (B, N_patches, d_model)
		x = self.encoder(x)  # (B, N_patches, d_model)
		x = x.sum(dim=1) * self.scale_factor
		return x


class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.multitask:
			self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
			self.register_buffer("_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim))
			for i in range(len(cfg.tasks)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		if cfg.simnorm:
			self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		else:
			self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim)
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		if cfg.full_rank:
			self._pi = full_rank_layers.full_rank_mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		else:
			self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		# Collapse prevention modules
		if getattr(cfg, 'collapse_prevention', False):
			self._collapse_pred = layers.mlp(cfg.latent_dim, 2*[cfg.mlp_dim], cfg.collapse_prevention_dim)
			in_channels = cfg.obs_shape['rgb'][0] if 'rgb' in cfg.obs_shape else cfg.obs_shape['state'][0]
			self._random_fn = RandomPatchTransformer(cfg,in_channels, patch_size=8, d_model=cfg.collapse_prevention_dim)
		else:
			self._collapse_pred = None
			self._random_fn = None
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		# We do this strange assignment to avoid having duplicated tensors in the state-dict -- working on a better API for this
		delattr(self._detach_Qs, "params")
		self._detach_Qs.__dict__["params"] = self._detach_Qs_params
		delattr(self._target_Qs, "params")
		self._target_Qs.__dict__["params"] = self._target_Qs_params

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Termination', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		# add a comparison of the number of parameters of the frozen transformer vs encoder and _collapse_pred
		if self.cfg.collapse_prevention:
			frozen_params = sum(p.numel() for p in self._random_fn.parameters())
			encoder_params = sum(p.numel() for p in self._encoder.parameters())
			collapse_params = sum(p.numel() for p in self._collapse_pred.parameters())
			repr += f"\nFrozen transformer params: {frozen_params:,}"
			repr += f"\nEncoder + collapse_pred params: {encoder_params + collapse_params:,}"
			if encoder_params + collapse_params < frozen_params:
				repr += f"\n❌ encoder + collapse_pred is not expressive enough"
			else:
				repr += f"\n✅ encoder + collapse_pred is expressive enough"
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def task_emb(self, x, task):
		"""
		Continuous task embedding for multi-task experiments.
		Retrieves the task embedding for a given task ID `task`
		and concatenates it to the input `x`.
		"""
		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)
		emb = self._task_emb(task.long())
		if x.ndim == 3:
			emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
		elif emb.shape[0] == 1:
			emb = emb.repeat(x.shape[0], 1)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			# Original (inefficient) implementation iterates over the first axis:
			#   torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
			# That runs the encoder `obs.shape[0]` times.  Instead, merge the first
			# dimension into the batch dimension so we can perform a single forward
			# pass and then reshape the output back to the original structure.
			#
			# Expected input shape: (T, B, C, H, W)  – here T can be any leading
			# dimension (e.g. temporal horizon).  The conv encoder expects (N, C, H, W).
			#
			# 1. Flatten the first two dims → (T*B, C, H, W)
			leading_dims = obs.shape[:2]  # (T, B) or similar
			flat_obs = obs.reshape(-1, *obs.shape[-3:])
			# 2. Single forward pass through the RGB encoder
			flat_enc = self._encoder[self.cfg.obs](flat_obs)
			# 3. Restore the original leading dimensions → (T, B, latent_dim)
			enc = flat_enc.reshape(*leading_dims, -1)
			return enc
		return self._encoder[self.cfg.obs](obs)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def termination(self, z, task, unnormalized=False):
		"""
		Predicts termination signal.
		"""
		assert task is None
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		if unnormalized:
			return self._termination(z)
		return torch.sigmoid(self._termination(z))
		

	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		if self.cfg.multitask:
			z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		if self.cfg.multitask: # Mask out unused action dimensions
			mean = mean * self._action_masks[task]
			log_std = log_std * self._action_masks[task]
			eps = eps * self._action_masks[task]
			action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
		else: # No masking
			action_dims = None

		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"action_prob": 1.,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		if self.cfg.multitask:
			z = self.task_emb(z, task)

		z = torch.cat([z, a], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
