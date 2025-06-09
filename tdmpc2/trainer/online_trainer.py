from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer
from tqdm import tqdm
from collapse_monitor import CollapseMonitor


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		elapsed_time = time() - self._start_time
		return dict(
			step=self._step,
			episode=self._ep_idx,
			elapsed_time=elapsed_time,
			steps_per_second=self._step / elapsed_time
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes, ep_lengths = [], [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			ep_lengths.append(t)
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
			episode_length= np.nanmean(ep_lengths),
		)

	def to_td(self, obs, action=None, reward=None, terminated=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		if terminated is None:
			terminated = torch.tensor(float('nan'))
		td = TensorDict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			terminated=terminated.unsqueeze(0),
		batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		
		# Initialize collapse monitor after env is available
		if not hasattr(self.agent, 'collapse_monitor') or self.agent.collapse_monitor is None:
			print("🔍 Initializing encoder collapse monitor...")
			try:
				self.agent.collapse_monitor = CollapseMonitor(
					cfg=self.cfg,
					encoder=self.agent.model._encoder[self.cfg.obs],
					env=self.env,
					device=str(self.agent.device),
					save_dir=f"collapse_logs_{self.cfg.exp_name}"
				)
				print("✅ Collapse monitor initialized successfully!")
			except Exception as e:
				print(f"⚠️ Failed to initialize collapse monitor: {e}")
				print("   Continuing training without collapse monitoring...")
				self.agent.collapse_monitor = None
		
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					if info['terminated'] and not self.cfg.episodic:
						raise ValueError('Termination detected but you are not in episodic mode. ' \
						'Set `episodic=true` to enable support for terminations.')
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
						episode_length=len(self._tds),
						episode_terminated=info['terminated'])
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, info['terminated']))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					for _ in tqdm(range(num_updates), desc='Pretraining agent on seed data...'):
						_train_metrics = self.agent.update(self.buffer, self._step)
				else:
					num_updates = 1
					_train_metrics = self.agent.update(self.buffer, self._step)
				train_metrics.update(_train_metrics)

			self._step += 1

		# Generate final collapse monitoring report
		if hasattr(self.agent, 'collapse_monitor') and self.agent.collapse_monitor is not None:
			print("\n" + "="*60)
			print("📊 FINAL ENCODER COLLAPSE ANALYSIS")
			print("="*60)
			try:
				# Generate final monitoring plots
				self.agent.collapse_monitor.plot_monitoring_results(save_plot=True)
				
				# Print final report
				report = self.agent.collapse_monitor.generate_report()
				print(report)
				
				# Save final monitoring data
				self.agent.collapse_monitor.save_monitoring_data()
				
			except Exception as e:
				print(f"⚠️ Error generating final collapse report: {e}")
			print("="*60)

		self.logger.finish(self.agent)
