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

class RandomLinear(nn.Module):
	def __init__(self, cfg, in_channels, out_channels):
		super().__init__()
		self.proj = nn.Linear(in_channels, out_channels)
		self.scale_factor = cfg.scale_factor

		for p in self.parameters():
			p.requires_grad = False
		self.eval()
		
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x is of shape (*, 9, 64, 64)
		# we want to flatten the last three dimensions
		x = x.reshape(-1, 9*64*64)
		return self.proj(x)*self.scale_factor


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
		# --- Transition / Dynamics model selection ---
		dyn_arch = getattr(cfg, "dynamics_arch", "iresnet")  # default to iresnet
		# --- Select dynamics architecture ---
		mlp_dims_dyn = getattr(cfg, "dyn_dims", cfg.mlp_dim)  # can be int or list
		if dyn_arch == "iresnet":
			C = getattr(cfg, "iresnet_C", 2.0)
			simple_dyn = getattr(cfg, "simple_dynamics", False)
			in_dim = cfg.latent_dim + cfg.action_dim + (cfg.task_dim if cfg.multitask else 0)
			self._dynamics = layers.IResNetTransition(
				in_dim=in_dim,
				z_dim=cfg.latent_dim,
				mlp_dims=mlp_dims_dyn,
				C=C,
				simple=simple_dyn,
			)
		elif dyn_arch == "mlp":
			in_dim_full = cfg.latent_dim + cfg.action_dim + cfg.task_dim
			if cfg.simnorm:
				self._dynamics = layers.mlp(
					in_dim_full,
					mlp_dims_dyn,
					cfg.latent_dim,
					act=layers.SimNorm(cfg),
				)
			else:
				self._dynamics = layers.mlp(
					in_dim_full,
					mlp_dims_dyn,
					cfg.latent_dim,
				)
		else:
			raise ValueError(f"Unsupported dynamics_arch '{dyn_arch}'.")
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		# Optional current reward head: predicts reward from latent only (no action)
		if getattr(cfg, 'current_reward', False):
			self._reward_current = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		else:
			self._reward_current = None
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		if cfg.full_rank:
			self._pi = full_rank_layers.full_rank_mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		else:
			self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
		# Initialize optional decoder for pixel reconstruction
		if getattr(cfg, 'enable_decoder', True):
			self._decoder = layers.dec(cfg)
		else:
			self._decoder = None
		# Collapse prevention modules
		if getattr(cfg, 'collapse_prevention', False):
			self._collapse_pred = layers.mlp(cfg.latent_dim, 2*[cfg.mlp_dim], cfg.collapse_prevention_dim)
			in_channels = cfg.obs_shape['rgb'][0] if 'rgb' in cfg.obs_shape else cfg.obs_shape['state'][0]
			if cfg.collapse_prevention_network == 'linear':
				self._random_fn = RandomLinear(cfg, 9*64*64, cfg.collapse_prevention_dim)
			elif cfg.collapse_prevention_network == 'transformer':
				self._random_fn = RandomPatchTransformer(cfg, in_channels, patch_size=8, d_model=cfg.collapse_prevention_dim)
			else:
				raise ValueError(f"Invalid collapse prevention network: {cfg.collapse_prevention_network}")
		else:
			self._collapse_pred = None
			self._random_fn = None
		
		# Apply standard weight initialization to all modules
		self.apply(init.weight_init)
		
		# Apply collapsed initialization to RGBMLPEncoder if requested
		if getattr(cfg, "collapsed_encoder_init", False):
			self._apply_collapsed_encoder_init()
		
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])
		if self._reward_current is not None:
			init.zero_([self._reward_current[-1].weight])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def _apply_collapsed_encoder_init(self):
		"""Apply collapsed initialization specifically to RGBMLPEncoder instances."""
		from common.layers import RGBMLPEncoder
		
		for name, module in self.named_modules():
			if isinstance(module, RGBMLPEncoder):
				# Zero out all linear layer weights and biases
				for mlp_module in module.mlp:
					if isinstance(mlp_module, nn.Linear):
						nn.init.normal_(mlp_module.weight, mean=0.0, std=1e-3)
						nn.init.constant_(mlp_module.bias, 0.0)
				
				# Set final bias to 0.1 for non-trivial downstream logits
				linear_layers = [m for m in module.mlp if isinstance(m, nn.Linear)]
				if linear_layers:
					nn.init.constant_(linear_layers[-1].bias, 0.1)

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
		module_objs = [self._encoder, self._dynamics, self._reward, self._termination, self._pi, self._Qs]
		for name, m in zip(modules, module_objs):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{name}: {m}\n"
		# Optionally include current-reward head in the printed architecture
		if getattr(self.cfg, 'current_reward', False) and getattr(self, '_reward_current', None) is not None:
			repr += f"Current reward: {self._reward_current}\n"
		# Total params
		repr += "Learnable parameters: {:,}".format(self.total_params)
		# Per-module parameter breakdown
		def count_params(module):
			return 0 if module is None else sum(p.numel() for p in module.parameters() if p.requires_grad)
		param_breakdown = []
		param_breakdown.append(("Encoder", count_params(self._encoder)))
		param_breakdown.append(("Dynamics", count_params(self._dynamics)))
		param_breakdown.append(("Reward", count_params(self._reward)))
		if getattr(self.cfg, 'current_reward', False) and getattr(self, '_reward_current', None) is not None:
			param_breakdown.append(("Current reward", count_params(self._reward_current)))
		if self.cfg.episodic:
			param_breakdown.append(("Termination", count_params(self._termination)))
		param_breakdown.append(("Policy prior", count_params(self._pi)))
		param_breakdown.append(("Q-functions", count_params(self._Qs)))
		if getattr(self.cfg, 'multitask', False):
			param_breakdown.append(("Task embedding", count_params(self._task_emb)))
		if getattr(self.cfg, 'enable_decoder', True) and getattr(self, '_decoder', None) is not None:
			param_breakdown.append(("Decoder", count_params(self._decoder)))
		if getattr(self.cfg, 'collapse_prevention', False) and getattr(self, '_collapse_pred', None) is not None:
			param_breakdown.append(("Collapse prevention", count_params(self._collapse_pred)))
		repr += "\nParameter breakdown:"
		for name, n in param_breakdown:
			repr += f"\n{name}: {n:,}"
		# add a comparison of the number of parameters of the frozen transformer vs encoder and _collapse_pred
		if self.cfg.collapse_prevention:
			frozen_params = sum(p.numel() for p in self._random_fn.parameters())
			encoder_params = sum(p.numel() for p in self._encoder.parameters())
			collapse_params = sum(p.numel() for p in self._collapse_pred.parameters())
			repr += f"\nFrozen {self.cfg.collapse_prevention_network} params: {frozen_params:,}"
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
	
	def reward_current(self, z, task):
		"""
		Predicts current reward from latent state only (no action input).
		"""
		if not getattr(self.cfg, 'current_reward', False) or self._reward_current is None:
			raise AttributeError('current_reward head is disabled')
		if self.cfg.multitask:
			z = self.task_emb(z, task)
		return self._reward_current(z)
	
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
