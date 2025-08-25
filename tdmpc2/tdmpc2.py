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
		param_groups = [
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._termination.parameters() if self.cfg.episodic else []},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
			{'params': self.model._collapse_pred.parameters() if getattr(self.cfg, 'collapse_prevention', False) else []}
		]
		# Optional current reward head parameters
		if getattr(self.cfg, 'current_reward', False) and getattr(self.model, '_reward_current', None) is not None:
			param_groups.append({'params': self.model._reward_current.parameters()})
		if getattr(self.cfg, 'enable_decoder', True):
			param_groups.append({'params': self.model._decoder.parameters()})
		# Use capturable=True only on CUDA devices for performance
		capturable = torch.cuda.is_available()
		self.optim = torch.optim.Adam(param_groups, lr=self.cfg.lr, capturable=capturable)
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
		# Encode ALL observations *once* (shape: (T+1, B, latent_dim))
		# ------------------------------------------------------------------
		enc_obs = self.model.encode(obs, task)

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
		# Compute TD targets using pre-computed encodings
		# ------------------------------------------------------------------
		if self.cfg.JEPA_sg:
			next_z = enc_obs[1:].detach()
			with torch.no_grad():
				td_targets = self._td_target(next_z, reward, terminated, task)
		else:
			next_z = enc_obs[1:]
			with torch.no_grad():
				td_targets = self._td_target(next_z, reward, terminated, task)

		# ------------------------------------------------------------------
		# Latent rollout starting from encoded obs[0] (undetached)
		# ------------------------------------------------------------------
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = enc_obs[0]  # uses gradients
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
			z = self.model.next(z, _action, task)
			# Consistency loss gradient control:
			# - When grad_from_dynamics is True: allow gradients to flow through the
			#   predicted latent (current) and optionally into the encoder for _next_z
			#   unless JEPA_sg is enabled (which always stop-grads next encodings).
			# - When grad_from_dynamics is False: block gradients from the consistency
			#   loss to both the current (predicted) latent and the t+1 encoder output.
			if self.cfg.grad_from_dynamics:
				z_cons = z
				next_z_cons = _next_z if not self.cfg.JEPA_sg else _next_z.detach()
			else:
				z_cons = z.detach()
				next_z_cons = _next_z.detach()
			consistency_loss = consistency_loss + F.mse_loss(z_cons, next_z_cons) * self.cfg.rho**t
			zs[t+1] = z

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
			if getattr(self.cfg, 'grad_from_current_R', False):
				_zs_rc = _zs  # allow gradients
			else:
				_zs_rc = _zs.detach()
			reward_current_preds = self.model.reward_current(_zs_rc, task)
		if self.cfg.episodic:
			if self.cfg.original_tdmpc2_implementation:
				termination_pred = self.model.termination(zs[1:], task, unnormalized=True)
			else:
				termination_pred = self.model.termination(zs[1:].detach(), task, unnormalized=True)

		# Compute losses
		reward_loss, value_loss = 0, 0
		
		# Use buffer-provided pre-action reward targets when current_reward is enabled
		preaction_targets = reward_pre if getattr(self.cfg, 'current_reward', False) else None
		for t, (rew_pred_unbind, rew_unbind, td_target_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho**t
			# Current reward head loss (pre-action target)
			if reward_current_preds is not None and preaction_targets is not None:
				rew_curr_unbind = reward_current_preds[t]
				pre_target_unbind = preaction_targets[t]
				reward_loss = reward_loss + math.soft_ce(rew_curr_unbind, pre_target_unbind, self.cfg).mean() * self.cfg.rho**t
			
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

		consistency_loss = consistency_loss / self.cfg.horizon
		reward_loss = reward_loss / self.cfg.horizon
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

		# Combine all losses
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

		# Single backward pass for all losses
		total_loss.backward()
		
		# Update both model and policy
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		
		self.optim.step()
		self.pi_optim.step()
		
		self.optim.zero_grad(set_to_none=True)
		self.pi_optim.zero_grad(set_to_none=True)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Return training statistics
		self.model.eval()
		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
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

		return update_info