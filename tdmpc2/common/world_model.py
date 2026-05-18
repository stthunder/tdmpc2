from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import layers, math, init
from common.mam_ode_world_model import MamODEDynamics
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


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
			if cfg.get('world_model', 'tdmpc2') == 'mam_ode':
				self.register_buffer("_obs_masks", torch.zeros(len(cfg.tasks), cfg.obs_shape[cfg.obs][0]))
				for i in range(len(cfg.tasks)):
					self._obs_masks[i, :cfg.obs_shapes[i]] = 1.
		if cfg.get('world_model', 'tdmpc2') == 'mam_ode':
			obs_dim = cfg.obs_shape[cfg.obs][0]
			stats_shape = (len(cfg.tasks), obs_dim) if cfg.multitask else (obs_dim,)
			self.register_buffer("_obs_mean", torch.zeros(stats_shape))
			self.register_buffer("_obs_std", torch.ones(stats_shape))
			reward_stats_shape = (len(cfg.tasks), 1) if cfg.multitask else (1,)
			self.register_buffer("_reward_mean", torch.zeros(reward_stats_shape))
			self.register_buffer("_reward_std", torch.ones(reward_stats_shape))
		self._encoder = layers.enc(cfg)
		if cfg.get('world_model', 'tdmpc2') == 'mam_ode':
			self._dynamics = MamODEDynamics(cfg)
		else:
			self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._termination = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 1) if cfg.episodic else None
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.Ensemble([layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout) for _ in range(cfg.num_q)])
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

	@property
	def is_mam_ode(self):
		return self.cfg.get('world_model', 'tdmpc2') == 'mam_ode'

	def set_obs_stats(self, mean, std):
		"""
		Set observation normalization statistics for MamODE model-only training.
		Statistics are computed from the train split only.
		"""
		self._obs_mean.copy_(mean.to(self._obs_mean.device))
		self._obs_std.copy_(std.to(self._obs_std.device).clamp_min(1e-6))

	def set_reward_stats(self, mean, std):
		"""
		Set per-task reward normalization statistics for MamODE reward loss.
		The reward model still predicts raw reward values.
		"""
		std_min = float(self.cfg.get("reward_std_min", 0.1))
		mean = torch.nan_to_num(mean.to(self._reward_mean.device), nan=0.0, posinf=0.0, neginf=0.0)
		std = torch.nan_to_num(std.to(self._reward_std.device), nan=1.0, posinf=1.0, neginf=1.0)
		self._reward_mean.copy_(mean)
		self._reward_std.copy_(std.clamp_min(std_min))

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
		if self.is_mam_ode:
			obs = self.normalize_obs(obs, task)
		if self.cfg.multitask:
			obs = self.task_emb(obs, task)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)

	def mask_obs(self, obs, task):
		"""
		Mask padded observation dimensions for multi-task MamODE training.
		"""
		if not self.cfg.multitask or not hasattr(self, "_obs_masks"):
			return obs
		mask = self._obs_masks[task]
		while mask.ndim < obs.ndim:
			mask = mask.unsqueeze(0)
		return obs * mask

	def _obs_stats(self, obs, task):
		if not self.cfg.multitask:
			mean, std = self._obs_mean, self._obs_std
		else:
			mean, std = self._obs_mean[task], self._obs_std[task]
		while mean.ndim < obs.ndim:
			mean = mean.unsqueeze(0)
			std = std.unsqueeze(0)
		return mean, std

	def _reward_stats(self, reward, task):
		if not self.cfg.multitask:
			mean, std = self._reward_mean, self._reward_std
		else:
			mean, std = self._reward_mean[task], self._reward_std[task]
		while mean.ndim < reward.ndim:
			mean = mean.unsqueeze(0)
			std = std.unsqueeze(0)
		return mean, std

	def normalize_obs(self, obs, task):
		"""
		Normalize raw observations using train-split statistics.
		"""
		if not self.is_mam_ode:
			return obs
		mean, std = self._obs_stats(obs, task)
		return self.mask_obs((obs - mean) / std, task)

	def denormalize_obs(self, obs, task):
		"""
		Convert normalized observations back to raw state units.
		"""
		if not self.is_mam_ode:
			return obs
		mean, std = self._obs_stats(obs, task)
		return self.mask_obs(obs * std + mean, task)

	def normalize_reward(self, reward, task):
		"""
		Normalize raw rewards using train-split per-task statistics.
		"""
		if not self.is_mam_ode:
			return reward
		mean, std = self._reward_stats(reward, task)
		std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(
			float(self.cfg.get("reward_std_min", 0.1))
		)
		return (reward - mean) / std

	def obs_loss(self, pred_norm, target, task):
		"""
		MSE between normalized prediction and normalized target.
		"""
		return F.mse_loss(self.mask_obs(pred_norm, task), self.normalize_obs(target, task))

	def reward_loss(self, pred, target, task):
		"""
		MSE between modeled reward and dataset reward in normalized reward space.
		"""
		return F.mse_loss(self.normalize_reward(pred, task), self.normalize_reward(target, task))

	def next_obs_norm(self, z, a, task):
		"""
		Predict the next padded observation in normalized state space.
		Only used by MamODE dynamics training/evaluation.
		"""
		if self.cfg.multitask:
			a = a * self._action_masks[task]
			z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self.mask_obs(self._dynamics(z), task)

	def init_dynamics_history(self, obs_hist, action_hist, task):
		"""
		Initialize MamODE dynamics from a history window.

		Args:
			obs_hist: (O, B, obs_dim), raw observations.
			action_hist: (O-1, B, action_dim), actions before current obs.
			task: (B,) task ids or None.
		"""
		z_hist = self.encode(obs_hist, task)
		if self.cfg.multitask:
			action_hist = action_hist * self._action_masks[task].unsqueeze(0)
			task_emb = self._task_emb(task.long())
		else:
			task_emb = None
		return self._dynamics.init_history(z_hist, action_hist, task_emb)

	def step_obs_norm(self, dyn_state, a, task):
		"""
		Predict one normalized observation using a history-initialized MamODE state.
		"""
		if self.cfg.multitask:
			a = a * self._action_masks[task]
		x_norm, reward, dyn_state = self._dynamics.step(dyn_state, a)
		return self.mask_obs(x_norm, task), reward, dyn_state

	def next_obs(self, z, a, task):
		"""
		Predict the next padded observation in raw state units.
		"""
		return self.denormalize_obs(self.next_obs_norm(z, a, task), task)

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		if self.is_mam_ode:
			return self.encode(self.next_obs(z, a, task), task)
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
