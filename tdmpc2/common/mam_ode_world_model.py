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
		self.prune_dim = cfg.get("mam_prune_dim", 32)
		self.hidden_dim = cfg.get("mam_hidden_dim", 128)
		self.ode_substeps = cfg.get("mam_ode_substeps", 1)

		context_dim = self.latent_dim + self.task_dim
		history_dim = self.latent_dim + self.task_dim + self.action_dim
		self.history_conv = nn.Conv1d(
			history_dim,
			history_dim,
			kernel_size=cfg.get("mam_history_kernel", 3),
			padding=cfg.get("mam_history_kernel", 3) - 1,
			groups=history_dim,
		)
		self.history_context = nn.Sequential(
			nn.Linear(history_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.prune_dim), nn.Tanh(),
		)
		self.history_ab = nn.Sequential(
			nn.Linear(history_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, 5 * self.prune_dim),
		)
		self.context = nn.Sequential(
			nn.Linear(context_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.prune_dim), nn.Tanh(),
		)
		self.ab_init = nn.Sequential(
			nn.Linear(context_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, 5 * self.prune_dim),
		)
		self.a_ode = nn.Sequential(
			nn.Linear(2 * self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.prune_dim), nn.Tanh(),
		)
		self.b_ode = nn.Sequential(
			nn.Linear(2 * self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.prune_dim), nn.Tanh(),
		)
		self.c_ode = nn.Sequential(
			nn.Linear(2 * self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.prune_dim), nn.Tanh(),
		)
		self.d_ode = nn.Sequential(
			nn.Linear(2 * self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.prune_dim), nn.Tanh(),
		)
		self.e_ode = nn.Sequential(
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
		self.reward_c_head = nn.Sequential(
			nn.Linear(self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.latent_dim),nn.ReLU()
		)
		self.reward_d_head = nn.Sequential(
			nn.Linear(self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.latent_dim),
		)
		self.reward_e_head = nn.Sequential(
			nn.Linear(self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, 1),
		)
		self.terminal_c_head = nn.Sequential(
			nn.Linear(self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.latent_dim), nn.ReLU()
		)
		self.terminal_d_head = nn.Sequential(
			nn.Linear(self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, self.latent_dim),
		)
		self.terminal_e_head = nn.Sequential(
			nn.Linear(self.prune_dim, self.hidden_dim), nn.SiLU(),
			nn.Linear(self.hidden_dim, 1),
		)
		self.log_dt = nn.Parameter(torch.tensor(math.log(math.exp(cfg.get("mam_dt", 0.1)) - 1.0)))

	def reward_from_state(self, z, c, d, e):
		"""
		Reward model: C z^2 + D z + E. C is diagonal for tractability.
		"""
		C_diag = -self.reward_c_head(c)
		D_vec = self.reward_d_head(d)
		E = self.reward_e_head(e)
		return ((C_diag * z.square()) + (D_vec * z)).sum(dim=-1, keepdim=True) / self.latent_dim + E

	def terminal_value_from_state(self, state):
		"""
		Terminal value model: predicted mean reward after the rollout horizon.
		"""
		z = state["z"]
		c = state["c"]
		d = state["d"]
		e = state["e"]
		C_diag = -self.terminal_c_head(c)
		D_vec = self.terminal_d_head(d)
		E = self.terminal_e_head(e)
		return ((C_diag * z.square()) + (D_vec * z)).sum(dim=-1, keepdim=True) / self.latent_dim + E

	def _ode_terms(self, a, b, c, d, e, context):
		"""
		Shared MamODE factor generator.

		step(), forward(), and controller_matrices() all call this method, so the
		controller sees the same A/B/C/D factors that the rollout model uses.
		"""
		ode_input_a = torch.cat([a, context], dim=-1)
		ode_input_b = torch.cat([b, context], dim=-1)
		ode_input_c = torch.cat([c, context], dim=-1)
		ode_input_d = torch.cat([d, context], dim=-1)
		ode_input_e = torch.cat([e, context], dim=-1)
		da          = self.a_ode(ode_input_a)
		db          = self.b_ode(ode_input_b)
		dc          = self.c_ode(ode_input_c)
		dd          = self.d_ode(ode_input_d)
		de          = self.e_ode(ode_input_e)
		diag        = self.diag_head(a)
		B_mat       = self.control_head(b).view(-1, self.latent_dim, self.action_dim)
		return da, db, dc, dd, de, diag, B_mat

	def init_history(self, z_hist, action_hist, task_emb=None):
		"""
		Build MamODE rollout state from a history window.

		Args:
			z_hist: (O, B, latent_dim), encoded normalized observations.
			action_hist: (O-1, B, action_dim), actions before the current state.
			task_emb: (B, task_dim) or None.
		"""
		O, B, _        = z_hist.shape
		if self.task_dim:
			task_seq   = task_emb.unsqueeze(0).expand(O, -1, -1)
		else:
			task_seq   = z_hist.new_zeros(O, B, 0)
		if action_hist.shape[0] == O - 1:
			pad        = action_hist[-1:].clone() if action_hist.shape[0] > 0 else z_hist.new_zeros(1, B, self.action_dim)
			action_seq = torch.cat([action_hist, pad], dim=0)
		else:
			action_seq = action_hist[-O:]
		hist           = torch.cat([z_hist, task_seq, action_seq], dim=-1).permute(1, 2, 0)
		hist           = F.silu(self.history_conv(hist))[..., -O:].mean(dim=-1)
		context        = self.history_context(hist)
		a, b, c, d, e  = self.history_ab(hist).chunk(5, dim=-1)
		return {
			"z": z_hist[-1],
			"a": a,
			"b": b,
			"c": c,
			"d": d,
			"e": e,
			"context": context,
			"task_emb": task_emb,
		}

	def step(self, state, action):
		z = state["z"]
		a = state["a"]
		b = state["b"]
		c = state["c"]
		d = state["d"]
		e = state["e"]
		context  = state["context"]
		task_emb = state["task_emb"]
		dt       = F.softplus(self.log_dt)
		da, db, dc, dd, de, diag, B_mat = self._ode_terms(a, b, c, d, e, context)
		dz       = diag * z + torch.einsum("bij,bj->bi", B_mat, action)
		a        = a + dt * da
		b        = b + dt * db
		c        = c + dt * dc
		d        = d + dt * dd
		e        = e + dt * de
		z        = z + dt * dz

		if self.task_dim:
			x = self.obs_head(torch.cat([z, task_emb], dim=-1))
		else:
			x = self.obs_head(z)
		reward = self.reward_from_state(z, c, d, e)
		return x, reward, {
			"z": z,
			"a": a,
			"b": b,
			"c": c,
			"d": d,
			"e": e,
			"context": context,
			"task_emb": task_emb,
		}

	def controller_matrices(self, state):
		"""
		Return the one-step matrices used by a Control.py-style MPC.

		This mirrors the role of init_for_controller/A_save/B_save in the
		standalone MamODE code: the matrices are generated inside the model,
		with the same ODE update order as step(). Since a,c,d evolve from the
		history context and not from the planned action in this implementation,
		the controller can roll these factors forward before solving the QP.
		"""
		z = state["z"]
		a = state["a"]
		b = state["b"]
		c = state["c"]
		d = state["d"]
		e = state["e"]
		context  = state["context"]
		task_emb = state["task_emb"]
		dt       = F.softplus(self.log_dt)
		da, db, dc, dd, de, diag, B_step = self._ode_terms(a, b, c, d, e, context)
		A_diag   = 1.0 + dt * diag
		B        = dt * B_step
		a        = a + dt * da
		b        = b + dt * db
		c        = c + dt * dc
		d        = d + dt * dd
		e        = e + dt * de

		C_diag = -self.reward_c_head(c) / self.latent_dim
		D_vec  = self.reward_d_head(d) / self.latent_dim
		E      = self.reward_e_head(e)
		return A_diag, B, C_diag, D_vec, E, {
			"z": z,
			"a": a,
			"b": b,
			"c": c,
			"d": d,
			"e": e,
			"context": context,
			"task_emb": task_emb,
		}

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
		leading_shape            = x.shape[:-1]
		x                        = x.reshape(-1, x.shape[-1])
		z, context_input, action = self._split(x)
		context                  = self.context(context_input)
		a, b, c, d, e            = self.ab_init(context_input).chunk(5, dim=-1)
		dt                       = F.softplus(self.log_dt)
		da, db, dc, dd, de, diag, B_mat = self._ode_terms(a, b, c, d, e, context)
		dz                       = diag * z + torch.einsum("bij,bj->bi", B_mat, action)
		a                        = a + dt * da
		b                        = b + dt * db
		c                        = c + dt * dc
		d                        = d + dt * dd
		e                        = e + dt * de
		z                        = z + dt * dz

		if self.task_dim:
			x = self.obs_head(torch.cat([z, context_input[:, self.latent_dim:]], dim=-1))
		else:
			x = self.obs_head(z)
		return x.reshape(*leading_shape, self.obs_dim)
