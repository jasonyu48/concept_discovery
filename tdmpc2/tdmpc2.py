import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for servers
import matplotlib.pyplot as plt

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from common.layers import api_model_conversion
from tensordict import TensorDict


####decoder related imports###
from pathlib import Path    
####end of decoder related imports###

class TDMPC2(torch.nn.Module):
	"""
	TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
		self.model = WorldModel(cfg).to(self.device)

		# Build parameter groups for the main optimizer (optionally includes decoder)
		param_groups = []
		# --- Encoders ---
		L_layers = int(getattr(self.model, 'num_jepa_layers', 1))
		# Base encoder (layer 0) – LR scaled by enc_lr_scale / L_layers
		param_groups.append({'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale / L_layers})
		# Higher MLP encoders (layers 1..L-1) – per-layer scaling: enc_lr_scale * 1/(L-l)
		if hasattr(self.model, '_enc_layers') and len(self.model._enc_layers) > 0:
			for idx, enc_l in enumerate(self.model._enc_layers, start=1):
				scale = 1 / max(L_layers - idx, 1)
				param_groups.append({'params': enc_l.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale * scale})
		# --- Dynamics (per layer)
		if hasattr(self.model, '_dyn_layers') and len(self.model._dyn_layers) > 0:
			for dyn_l in self.model._dyn_layers:
				param_groups.append({'params': dyn_l.parameters(), 'lr': self.cfg.lr})
		# --- Other heads ---
		param_groups.append({'params': self.model._reward.parameters()})
		param_groups.append({'params': self.model._termination.parameters() if self.cfg.episodic else []})
		param_groups.append({'params': self.model._Qs.parameters()})
		param_groups.append({'params': self.model._task_emb.parameters() if self.cfg.multitask else []})
		param_groups.append({'params': self.model._collapse_pred.parameters() if getattr(self.cfg, 'collapse_prevention', False) else []})
		# Optional current reward head parameters
		if getattr(self.cfg, 'current_reward', False):
			# Include per-layer current reward heads when present (no LR scaling per user spec)
			if hasattr(self.model, '_reward_current_layers') and self.model._reward_current_layers is not None:
				param_groups.append({'params': self.model._reward_current_layers.parameters()})
			elif getattr(self.model, '_reward_current', None) is not None:
				param_groups.append({'params': self.model._reward_current.parameters()})
		if getattr(self.cfg, 'enable_decoder', True):
			param_groups.append({'params': self.model._decoder.parameters()})
		# Use capturable=True only on CUDA devices for performance
		capturable = torch.cuda.is_available()
		self.optim = torch.optim.Adam(param_groups, lr=self.cfg.lr, capturable=capturable)

		# --- Optional separate optimizer for L_eq (commutative encoders & E_0) ---
		self.eq_optim = None
		N_comm = int(getattr(self.model, 'num_commutative_encoders', 0))
		if N_comm > 0 and hasattr(self.model, '_comm_encoders') and self.model._comm_encoders is not None:
			# E_0 encoder params (all parts that contribute to encode_e0)
			params_e0 = list(self.model._encoder.parameters()) + list(self.model._enc_layers.parameters())
			# Scale only encoders for this optimizer
			group_e0 = {'params': params_e0, 'lr': self.cfg.lr * self.cfg.enc_lr_scale}
			group_comm_enc = {'params': list(self.model._comm_encoders.parameters()), 'lr': self.cfg.lr * self.cfg.enc_lr_scale}
			# Scale commutative maps c_i by enc_lr_scale as requested
			groups = [group_e0, group_comm_enc]
			if getattr(self.model, '_comm_maps', None) is not None:
				group_comm_maps = {'params': list(self.model._comm_maps.parameters()), 'lr': self.cfg.lr * self.cfg.enc_lr_scale}
				groups.append(group_comm_maps)
			# Use base lr for comm dynamics and current-reward heads
			params_other = []
			if getattr(self.model, '_comm_dyns', None) is not None:
				params_other += list(self.model._comm_dyns.parameters())
			if getattr(self.model, '_reward_current_comm', None) is not None:
				params_other += list(self.model._reward_current_comm.parameters())
			if len(params_other) > 0:
				group_other = {'params': params_other, 'lr': self.cfg.lr}
				groups.append(group_other)
			self.eq_optim = torch.optim.Adam(groups, lr=self.cfg.lr, capturable=capturable)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=capturable)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device=self.device
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		print('Episode length:', cfg.episode_length)
		print('Discount factor:', self.discount)
		self._prev_mean = torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device))
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")
		
		# Initialize encoding space monitor
		self.encoding_monitor = None  # Will be set by trainer after env is available

	@property
	def plan(self):
		_plan_val = getattr(self, "_plan_val", None)
		if _plan_val is not None:
			return _plan_val
		if self.cfg.compile:
			plan = torch.compile(self._plan, mode="reduce-overhead")
		else:
			plan = self._plan
		self._plan_val = plan
		return self._plan_val

	def _get_discount(self, episode_length):
		"""
		Returns discount factor for a given episode length.
		Simple heuristic that scales discount linearly with episode length.
		Default values should work well for most tasks, but can be changed as needed.

		Args:
			episode_length (int): Length of the episode. Assumes episodes are of fixed length.

		Returns:
			float: Discount factor for the task.
		"""
		frac = episode_length/self.cfg.discount_denom
		return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str or dict): Filepath or state dict to load.
		"""
		if isinstance(fp, dict):
			state_dict = fp
		else:
			state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)
		return

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			return self.plan(obs, t0=t0, eval_mode=eval_mode, task=task).cpu()
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		if eval_mode:
			action = info["mean"]
		return action[0].cpu()

	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G, discount = 0, 1
		termination = torch.zeros(self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
			z = self.model.next(z, actions[t], task)
			G = G + discount * (1-termination) * reward
			discount_update = self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
			discount = discount * discount_update
			if self.cfg.episodic:
				termination = torch.clip(termination + (self.model.termination(z, task) > 0.5).float(), max=1.)
		action, _ = self.model.pi(z, task)
		return G + discount * (1-termination) * self.model.Q(z, action, task, return_type='avg')

	@torch.no_grad()
	def _plan(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Fast path: optionally use uniformly random actions during training
		if getattr(self.cfg, 'random_action_selection', False) and not eval_mode:
			# Initialize repetition memory lazily to keep changes local to this method
			if not hasattr(self, '_rand_repeat_remaining'):
				self._rand_repeat_remaining = 0
				self._rand_cached_action = None

			# If we have remaining repeats, return the cached action (re-masked per task if needed)
			if self._rand_repeat_remaining > 0 and self._rand_cached_action is not None:
				self._rand_repeat_remaining -= 1
				a_cached = self._rand_cached_action
				if self.cfg.multitask:
					return a_cached * self.model._action_masks[task]
				return a_cached

			# Otherwise, sample a new action and set it to repeat for the next 4 calls
			# Counting task: provide task-specific random sampling
			if 'counting' in self.cfg.task and getattr(self.cfg, 'discrete_action', False):
				# Sample one-hot over actions; if two_actions → no no-op
				if getattr(self.cfg, 'two_actions', False):
					probs = torch.tensor([0.5, 0.5], device=self.device)
					idx = torch.multinomial(probs, num_samples=1).item()
				else:
					probs = torch.tensor([1/3, 1/3, 1/3], device=self.device)
					idx = torch.multinomial(probs, num_samples=1).item()
				a_unmasked = torch.zeros(self.cfg.action_dim, device=self.device)
				a_unmasked[idx] = 1.0
			else:
				# Directly sample a random action in [-1, 1] without any MPPI compute
				a_unmasked = torch.empty(self.cfg.action_dim, device=self.device).uniform_(-1.0, 1.0)

			# Cache and set repeat counter (next 4 actions will repeat this sample)
			self._rand_cached_action = a_unmasked.detach().clone()
			self._rand_repeat_remaining = 4

			# Apply task-specific action mask, if any
			if self.cfg.multitask:
				return a_unmasked * self.model._action_masks[task]
			return a_unmasked

		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t], _ = self.model.pi(_z, task)
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1], _ = self.model.pi(_z, task)

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1)) # this is correct
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean.copy_(mean)
		return a.clamp(-1, 1)

	def update_pi(self, zs, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		action, info = self.model.pi(zs, task)
		qs = self.model.Q(zs, action, task, return_type='avg', detach=True)
		self.scale.update(qs[0])
		qs = self.scale(qs)

		# Loss is a weighted sum of Q-values
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1,2)) * rho).mean()
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)

		info = TensorDict({
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, terminated, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			terminated (torch.Tensor): Termination signal at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
		return reward + discount * (1-terminated) * self.model.Q(next_z, action, task, return_type='min', target=True)


	def _update(self, obs, action, reward, terminated, q_mask=None, task=None, step=None, pretrain_step=-1, reward_pre=None):
		# Prepare for update
		self.model.train()
		# zero existing gradients once per iteration
		self.optim.zero_grad(set_to_none=True)
		self.pi_optim.zero_grad(set_to_none=True)

		# ------------------------------------------------------------------
		# Encode observations through ALL layers once
		# enc_all: list of length L, each tensor shape (T+1, B, D)
		# enc_obs: top-layer latents (T+1, B, D) used for control/decoder/etc.
		# ------------------------------------------------------------------
		enc_all = self.model.encode_all_layers(obs, task)
		enc_obs = enc_all[-1]
		# Note: commutative encoder latents are only computed in the separate L_eq pass

		# ------------------------------------------------------------------
		# Pairwise shrink loss: encourage smaller pairwise distances between
		# encoding vectors *across the batch dimension* (per-time-step).
		# Efficient batched computation without explicit (B×B×D) tensor.
		# ------------------------------------------------------------------
		shrink_loss = torch.tensor(0.0, device=self.device)
		if getattr(self.cfg, "shrink_coef", 0.0) > 0 and enc_obs.shape[1] > 1:
			# enc_obs shape: (T+1, B, D)
			B = enc_obs.shape[1]
			# Compute squared norms per vector: (T+1, B)
			sq = enc_obs.pow(2).sum(dim=-1)
			# Batched Gram matrices: (T+1, B, B)
			gram = torch.bmm(enc_obs, enc_obs.transpose(1, 2))
			# Pairwise squared distances
			dist2 = sq.unsqueeze(2) + sq.unsqueeze(1) - 2 * gram
			# Exclude diagonal elements (distance=0 with itself)
			diag_sum = torch.diagonal(dist2, dim1=1, dim2=2).sum(dim=-1)
			dist2_sum = dist2.sum(dim=(1, 2)) - diag_sum
			mean_pairwise = dist2_sum / (B * (B - 1))
			shrink_loss = mean_pairwise.mean()

		# ------------------------------------------------------------------
		# Decoder reconstruction loss (detached) – no optimiser step yet
		# ------------------------------------------------------------------
		dec_loss = torch.tensor(0.0, device=self.device)
		if self.cfg.obs == 'rgb' and getattr(self.cfg, "enable_decoder", True):
			# Target images: normalize entire observation tensor to [-1,1]
			rgb_norm = (obs.to(self.device).float() / 255.0 - 0.5) * 2  # (T+1, B, C, H, W)
			z_detached = enc_obs.detach()
			pred_rgb = self.model._decoder(z_detached.reshape(-1, z_detached.shape[-1]))
			pred_rgb = pred_rgb.reshape_as(rgb_norm)
			dec_loss = F.mse_loss(pred_rgb, rgb_norm)

		# ------------------------------------------------------------------
		# Compute TD targets using pre-computed TOP-layer encodings
		# ------------------------------------------------------------------
		if not self.cfg.grad_from_dynamics:
			self.cfg.JEPA_sg = True
		if self.cfg.JEPA_sg:
			next_z = enc_obs[1:].detach() 
			with torch.no_grad():
				td_targets = self._td_target(next_z, reward, terminated, task)
		else:
			next_z = enc_obs[1:] # actual z at t+1
			with torch.no_grad():
				td_targets = self._td_target(next_z, reward, terminated, task)

		# ------------------------------------------------------------------
		# Consistency loss: original single-layer (flag) vs. new per-layer formulation
		# Also construct the TOP-layer rollout for control path accordingly
		# ------------------------------------------------------------------
		L_layers = getattr(self.model, 'num_jepa_layers', 1)
		if getattr(self.cfg, 'original_tdmpc2_implementation', False):
			# Original TD-MPC2: single-layer consistency on top latent and rollout
			zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
			z = enc_obs[0]  # uses gradients
			zs[0] = z
			consistency_loss = 0
			for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
				if not self.cfg.grad_from_dynamics:
					z_cons = self.model.next(z.detach(), _action, task)
				z = self.model.next(z, _action, task) # predicted z at t+1
				if self.cfg.grad_from_dynamics:
					z_cons = z
				consistency_loss = consistency_loss + F.mse_loss(z_cons, _next_z) * self.cfg.rho**t
				zs[t+1] = z
			consistency_loss = consistency_loss / self.cfg.horizon
		else:
			# New: per-layer P-JEPA consistency; separate top-layer rollout for control
			consistency_loss = 0.0
			for l in range(L_layers):
				z_seq = enc_all[l]                  # (T+1, B, D)
				for t, _action in enumerate(action.unbind(0)):
					z_t = z_seq[t]
					z_tp1_target = z_seq[t+1].detach() if self.cfg.JEPA_sg else z_seq[t+1]
					if not self.cfg.grad_from_dynamics:
						z_pred = self.model.next_layer(z_t.detach(), _action, task, layer_idx=l)
					else:
						z_pred = self.model.next_layer(z_t, _action, task, layer_idx=l)
					consistency_loss = consistency_loss + F.mse_loss(z_pred, z_tp1_target) * self.cfg.rho**t
			# Normalise by horizon
			consistency_loss = consistency_loss / (self.cfg.horizon)
			# Use encoded top-layer sequence directly (no rollout) for control/actor-critic path
			zs = enc_obs

		# Predictions
		_zs = zs[:-1]
		if self.cfg.grad_from_Q:
			_zs_q = _zs
		else:
			_zs_q = _zs.detach()
		qs = self.model.Q(_zs_q, action, task, return_type='all')
		if self.cfg.grad_from_R:
			_zs_r = _zs
		else:
			_zs_r = _zs.detach()
		reward_preds = self.model.reward(_zs_r, action, task)
		# Optional current reward head (from latent only)
		reward_current_preds = None
		if getattr(self.cfg, 'current_reward', False):
			if getattr(self.cfg, 'original_tdmpc2_implementation', False):
				# Original: use TOP-layer rollout sequence for current reward head
				if getattr(self.cfg, 'grad_from_current_R', False):
					_zs_rc = _zs  # allow gradients
				else:
					_zs_rc = _zs.detach()
				reward_current_preds = self.model.reward_current(_zs_rc, task)
			else:
				# Multi-layer: per-layer current reward predictions from encoded sequences
				# Build list length L, each tensor (T, B, num_bins)
				reward_current_preds = []
				for l in range(L_layers):
					z_seq = enc_all[l][:-1]
					z_in = z_seq if getattr(self.cfg, 'grad_from_current_R', False) else z_seq.detach()
					pred = self.model.reward_current_layer(z_in, task, layer_idx=l)
					reward_current_preds.append(pred)
		if self.cfg.episodic:
			if self.cfg.original_tdmpc2_implementation:
				termination_pred = self.model.termination(zs[1:], task, unnormalized=True)
			else:
				termination_pred = self.model.termination(zs[1:].detach(), task, unnormalized=True)

		# Compute losses
		# Split reward loss into post-action (environment) and current (phenomenon) components
		reward_post_loss = torch.tensor(0.0, device=self.device)
		reward_curr_loss = torch.tensor(0.0, device=self.device)
		value_loss = 0
		
		# Use buffer-provided pre-action reward targets when current_reward is enabled
		preaction_targets = reward_pre if getattr(self.cfg, 'current_reward', False) else None
		for t, (rew_pred_unbind, rew_unbind, td_target_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			# Post-action (environment) reward head
			reward_post_loss = reward_post_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			# Current reward head loss (pre-action target)
			if reward_current_preds is not None and preaction_targets is not None:
				pre_target_unbind = preaction_targets[t]
				# Two cases: multi-layer list or single tensor (original path)
				if isinstance(reward_current_preds, list):
					layer_loss = 0.0
					for l in range(L_layers):
						rew_curr_unbind_l = reward_current_preds[l][t]
						layer_loss = layer_loss + math.soft_ce(rew_curr_unbind_l, pre_target_unbind, self.cfg).mean()
					reward_curr_loss = reward_curr_loss + (layer_loss) * self.cfg.rho**t
				else:
					rew_curr_unbind = reward_current_preds[t]
					reward_curr_loss = reward_curr_loss + math.soft_ce(rew_curr_unbind, pre_target_unbind, self.cfg).mean() * self.cfg.rho**t
			
			# Apply Q-function mask to value loss if available
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				q_loss = math.soft_ce(qs_unbind_unbind, td_target_unbind, self.cfg)
				if q_mask is not None:
					# Apply mask to Q-function loss
					q_loss = q_loss * q_mask.float()
					# Normalize by number of visible samples to maintain loss scale
					q_loss = q_loss.sum() / q_mask.sum().float().clamp(min=1.0)
				else:
					q_loss = q_loss.mean()
				value_loss = value_loss + q_loss * self.cfg.rho**t

		# Normalise reward components by horizon
		reward_post_loss = reward_post_loss / self.cfg.horizon
		reward_curr_loss = reward_curr_loss / self.cfg.horizon
		# Total reward loss used for optimisation (unchanged behaviour)
		reward_loss = reward_post_loss + reward_curr_loss
		if self.cfg.episodic:
			termination_loss = F.binary_cross_entropy_with_logits(termination_pred, terminated)
		else:
			termination_loss = 0.

		# Collapse prevention loss using random function predictor
		collapse_prevention_coef = self.cfg.collapse_prevention_coef
		if self.cfg.collapse_prevention and collapse_prevention_coef > 0:
			# Flatten time and batch dimensions to use the entire sequence
			T_seq, B = obs.shape[0], obs.shape[1]
			obs_flat = obs.view(-1, *obs.shape[2:]).to(dtype=torch.float32)  # ((T)*B, ...)
			zs_flat = enc_obs.view(-1, enc_obs.shape[-1])        # Use encoded latents instead of rollout
			random_target = self.model._random_fn(obs_flat).detach()  # ((T)*B, D)
			pred_target = self.model._collapse_pred(zs_flat)          # ((T)*B, D)
			# Apply Q-sampling mask across batch dimension for every timestep
			if q_mask is not None:
				mask = q_mask.unsqueeze(0).expand(T_seq, B).reshape(-1)
				pred_target = pred_target[mask]
				random_target = random_target[mask]
			collapse_loss = F.mse_loss(pred_target, random_target) / T_seq
		else:
			collapse_loss = torch.tensor(0.0, device=self.device)

		value_loss = value_loss / (self.cfg.horizon * self.cfg.num_q)
		
		# Compute policy loss
		if self.cfg.grad_from_policy:
			zs_for_pi = zs
		else:
			zs_for_pi = zs.detach()
		
		action_pi, info_pi = self.model.pi(zs_for_pi, task)
		qs_pi = self.model.Q(zs_for_pi, action_pi, task, return_type='avg', detach=True)
		self.scale.update(qs_pi[0])
		qs_pi = self.scale(qs_pi)
		rho = torch.pow(self.cfg.rho, torch.arange(len(qs_pi), device=self.device))
		pi_loss = (-(self.cfg.entropy_coef * info_pi["scaled_entropy"] + qs_pi).mean(dim=(1,2)) * rho).mean()

		# Combine all losses (EXCLUDING L_eq; it is optimized with a separate optimizer below)
		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.termination_coef * termination_loss +
			self.cfg.value_coef * value_loss +
			self.cfg.pi_coef * pi_loss +  # Add policy loss to total
			collapse_prevention_coef * collapse_loss +
			self.cfg.shrink_coef * shrink_loss +
			dec_loss  # reconstruction component
		)

		# Backward + step for main optimizer
		total_loss.backward()
		
		# Update both model and policy
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		
		self.optim.step()
		# Separate forward/backward/step for L_eq with its own optimizer
		l_eq_scalar = None
		if self.eq_optim is not None and int(getattr(self.model, 'num_commutative_encoders', 0)) > 0:
			self.eq_optim.zero_grad(set_to_none=True)
			# Recompute encodings for eq pass to get a fresh graph
			z0_eq = self.model.encode_e0(obs, task)[:-1]  # (T, B, D)
			z_comm_all_eq = self.model.encode_comm_all(obs, task)  # list of (T+1, B, D)
			T_steps = z0_eq.shape[0]
			# 1) Alignment loss: encourage all mapped encoders to agree pairwise in E0-space
			c_list_eq = self.model.comm_map_all([z[:-1] for z in z_comm_all_eq])  # list of (T, B, D)
			Z_align = [z0_eq] + c_list_eq  # length = 1 + N_comm
			L_eq = torch.tensor(0.0, device=self.device)
			n_src = len(Z_align)
			n_pairs = n_src * (n_src - 1) // 2
			for i in range(n_src):
				for j in range(i+1, n_src):
					L_eq = L_eq + F.mse_loss(Z_align[i], Z_align[j])
			if n_pairs > 0:
				L_eq = L_eq / n_pairs
			# 2) Commutative consistency loss per E_i with its own dynamics next_comm
			L_cons_comm = torch.tensor(0.0, device=self.device)
			if T_steps >= 1:
				for idx, z_seq_i in enumerate(z_comm_all_eq):
					# z_seq_i: (T+1, B, D)
					for t, _action in enumerate(action.unbind(0)):
						z_t = z_seq_i[t]
						z_tp1_target = z_seq_i[t+1]
						z_pred = self.model.next_comm(z_t, _action, task, idx=idx)
						L_cons_comm = L_cons_comm + F.mse_loss(z_pred, z_tp1_target) * self.cfg.rho**t
				# Normalise by horizon
				L_cons_comm = L_cons_comm / max(self.cfg.horizon, 1)
			# 3) Commutative current-reward loss per E_i (optional)
			L_curr_comm = torch.tensor(0.0, device=self.device)
			if getattr(self.cfg, 'current_reward', False):
				preaction_targets = reward_pre
				if preaction_targets is not None:
					for idx, z_seq_i in enumerate(z_comm_all_eq):
						for t in range(T_steps):
							z_it = z_seq_i[t]
							pred_logits = self.model.reward_current_comm(z_it, task, idx=idx)
							L_curr_comm = L_curr_comm + math.soft_ce(pred_logits, preaction_targets[t], self.cfg).mean() * self.cfg.rho**t
					# Normalise by horizon
					L_curr_comm = L_curr_comm / max(self.cfg.horizon, 1)
			# Combine and step
			total_eq = (
				getattr(self.cfg, 'L_eq_coef', 1.0) * L_eq +
				self.cfg.consistency_coef * L_cons_comm +
				self.cfg.reward_coef * L_curr_comm
			)
			total_eq.backward()
			self.eq_optim.step()
			l_eq_scalar = float(L_eq.detach().item())
		self.pi_optim.step()
		
		self.optim.zero_grad(set_to_none=True)
		self.pi_optim.zero_grad(set_to_none=True)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			# Log ONLY the pre-action (current) reward loss under the key 'reward_loss'
			"reward_loss": reward_curr_loss,
			# Expose components for debugging/analysis
			"reward_curr_loss": reward_curr_loss,
			"reward_post_loss": reward_post_loss,
			"reward_total_loss": reward_loss,
			"value_loss": value_loss,
			"termination_loss": termination_loss,
			"total_loss": total_loss,
			"collapse_loss": collapse_loss,
			"shrink_loss": shrink_loss,
			"grad_norm": grad_norm,
			"pi_loss": pi_loss,
			"pi_grad_norm": pi_grad_norm,
			"pi_entropy": info_pi["entropy"],
			"pi_scaled_entropy": info_pi["scaled_entropy"],
			"pi_scale": self.scale.value,
		})
		# Attach L_eq scalar for logging if computed
		if l_eq_scalar is not None:
			info.update({"L_eq": torch.tensor(l_eq_scalar, device=self.device)})
		if self.cfg.episodic:
			info.update(math.termination_statistics(torch.sigmoid(termination_pred[-1]), terminated[-1]))
		
		# Monitor encoding space if available (simplified)
		if hasattr(self, 'encoding_monitor') and self.encoding_monitor is not None and (pretrain_step < 0 or pretrain_step+1 == self.cfg.seed_steps):
			encoding_metrics = self.encoding_monitor.monitor_step(step)
			if encoding_metrics:  # Only add if monitoring was performed
				info.update({
					f"encoding_{k}": v for k, v in encoding_metrics.items()
					if k not in ['step']  # Avoid duplicate step info
				})
		
		return info.detach().mean()

	def update(self, buffer, step, pretrain_step=-1):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		# Now returns obs_type and reward_pre for analysis as well
		obs, action, reward, terminated, task, q_mask, obs_type, reward_pre = buffer.sample()
		
		# Now, `obs` is passed directly to _update without modification.
		kwargs = {}
		if task is not None:
			kwargs["task"] = task
		torch.compiler.cudagraph_mark_step_begin()
		
		# Run main update
		update_info = self._update(obs, action, reward, terminated, q_mask=q_mask, **kwargs, step=step, pretrain_step=pretrain_step, reward_pre=reward_pre)
		
		# Compute visibility percentage
		resident = buffer.resident_eps
		visible_percent = 100.0 * buffer.q_visible_episodes / max(resident, 1)
		if step % self.cfg.monitor_freq == 0 and pretrain_step < 1:
			print(f"Q-visible episodes: {buffer.q_visible_episodes}/{resident} ({visible_percent:.1f}%) -> sample_ratio: {self.cfg.q_sample_ratio}")
		# Log metric
		update_info["q_visible_percent"] = torch.tensor(visible_percent, device=self.device)

		# Append reward/consistency (and L_eq if available) to train.csv every monitor_freq steps
		if step is not None and (int(step) % getattr(self.cfg, 'monitor_freq', 1000) == 0):
			try:
				work_dir = getattr(self.cfg, 'work_dir', '.')
				train_csv = Path(work_dir) / 'train.csv'
				if not train_csv.exists():
					train_csv.parent.mkdir(parents=True, exist_ok=True)
					with open(train_csv, 'w') as f:
						f.write('step,reward_loss,consistency_loss,L_eq\n')
				rl = float(update_info["reward_loss"].detach().cpu().item()) if hasattr(update_info["reward_loss"], 'item') else float(update_info["reward_loss"]) 
				cl = float(update_info["consistency_loss"].detach().cpu().item()) if hasattr(update_info["consistency_loss"], 'item') else float(update_info["consistency_loss"]) 
				leq = float(update_info.get("L_eq", torch.tensor(float('nan'))).detach().cpu().item()) if isinstance(update_info.get("L_eq", None), torch.Tensor) else (float(update_info["L_eq"]) if ("L_eq" in update_info and update_info["L_eq"] is not None) else float('nan'))
				with open(train_csv, 'a') as f:
					f.write(f'{int(step)},{rl:.8f},{cl:.8f},{leq:.8f}\n')
			except Exception as e:
				print(f"⚠️ Failed to write train.csv: {e}")

		return update_info