import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MamODEDynamics(nn.Module):
	"""
	MamODE-style observation dynamics block for the TD-MPC2 world model.

	The surrounding WorldModel still owns padding, task embeddings, reward,
	policy, and Q functions. This module predicts the next padded state x
	from the current latent state and action. WorldModel.next re-encodes this
	x prediction when a latent z_next is needed for planning.

		z' = z + dt * (diag(a) * z + B(a) * action)
		x' = C(z')

	Inputs follow WorldModel.next ordering:
		[z, task_embedding, action] for multitask
		[z, action] for single-task
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.latent_dim = cfg.latent_dim
		self.action_dim = cfg.action_dim
		self.task_dim = cfg.task_dim
		self.obs_dim = cfg.obs_shape[cfg.obs][0]
		self.prune_dim = cfg.get("mam_prune_dim", 8)
		self.hidden_dim = cfg.get("mam_hidden_dim", 128)
		self.ode_substeps = cfg.get("mam_ode_substeps", 1)

		context_dim = self.latent_dim + self.task_dim
		self.context = nn.Sequential(
			nn.Linear(context_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.prune_dim), nn.Tanh(),
		)
		self.ab_init = nn.Sequential(
			nn.Linear(context_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, 2 * self.prune_dim),
		)
		self.a_ode = nn.Sequential(
			nn.Linear(2 * self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.prune_dim), nn.Tanh(),
		)
		self.b_ode = nn.Sequential(
			nn.Linear(2 * self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.prune_dim), nn.Tanh(),
		)
		self.diag_head = nn.Sequential(
			nn.Linear(self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.latent_dim),
		)
		self.control_head = nn.Sequential(
			nn.Linear(self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.latent_dim * self.action_dim),
		)
		self.obs_head = nn.Sequential(
			nn.Linear(self.latent_dim + self.task_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.obs_dim),
		)
		self.log_dt = nn.Parameter(torch.tensor(math.log(math.exp(cfg.get("mam_dt", 0.1)) - 1.0)))

	def _split(self, x):
		z = x[..., :self.latent_dim]
		task_start = self.latent_dim
		task_end = task_start + self.task_dim
		if self.task_dim:
			task = x[..., task_start:task_end]
			context = torch.cat([z, task], dim=-1)
		else:
			context = z
		action = x[..., task_end:task_end + self.action_dim]
		return z, context, action

	def forward(self, x):
		leading_shape = x.shape[:-1]
		x = x.reshape(-1, x.shape[-1])
		z, context_input, action = self._split(x)
		context = self.context(context_input)
		a, b = self.ab_init(context_input).chunk(2, dim=-1)
		dt = F.softplus(self.log_dt) / max(self.ode_substeps, 1)

		for _ in range(max(self.ode_substeps, 1)):
			ode_input_a = torch.cat([a, context], dim=-1)
			ode_input_b = torch.cat([b, context], dim=-1)
			da = self.a_ode(ode_input_a)
			db = self.b_ode(ode_input_b)
			diag = self.diag_head(a)
			control = self.control_head(a).view(-1, self.latent_dim, self.action_dim)
			dz = diag * z + torch.einsum("bij,bj->bi", control, action)
			a = a + dt * da
			b = b + dt * db
			z = z + dt * dz

		if self.task_dim:
			x = self.obs_head(torch.cat([z, context_input[:, self.latent_dim:]], dim=-1))
		else:
			x = self.obs_head(z)
		return x.reshape(*leading_shape, self.obs_dim)
