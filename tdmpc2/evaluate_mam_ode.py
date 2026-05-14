import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')
import time

import hydra
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.world_model import WorldModel
from envs import make_env


torch.backends.cudnn.benchmark = True

try:
	from tqdm import tqdm
except ImportError:
	tqdm = None


class MamODEMPCAgent(torch.nn.Module):
	"""CVXPY MPC policy using the learned MamODE linear/reward factors."""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		try:
			import cvxpy as cp
		except ImportError as exc:
			raise ImportError(
				'MamODE CVXPY-MPC requires cvxpy. Install it in the environment used to run evaluation.'
			) from exc
		self.cp = cp
		self.model = WorldModel(cfg).to(self.device).eval()
		assert self.model.is_mam_ode, 'MamODEMPCAgent requires world_model=mam_ode.'
		self._prev_actions = np.zeros((self.cfg.horizon, self.cfg.action_dim), dtype=np.float64)
		self._obs_hist = None
		self._action_hist = None
		self._last_action = None
		self.last_solve_time = 0.0
		self.last_solver_status = 'not_started'

	def load(self, fp):
		state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict['model'] if 'model' in state_dict else state_dict
		self.model.load_state_dict(state_dict)

	def _task_tensor(self, task, n=1):
		if task is None:
			return None
		return torch.full((n,), int(task), dtype=torch.long, device=self.device)

	def _reset_history(self, obs):
		O = self.cfg.get('model_history', 1)
		obs = obs.to(self.device, non_blocking=True)
		self._obs_hist = [obs.clone() for _ in range(O)]
		zero_action = torch.zeros(self.cfg.action_dim, device=self.device)
		self._action_hist = [zero_action.clone() for _ in range(max(O-1, 0))]
		self._last_action = None
		self._prev_actions.fill(0)

	def _advance_history(self, obs):
		obs = obs.to(self.device, non_blocking=True)
		if self._last_action is not None and len(self._action_hist) > 0:
			self._action_hist.pop(0)
			self._action_hist.append(self._last_action.clone())
		if self._obs_hist is not None:
			self._obs_hist.pop(0)
			self._obs_hist.append(obs.clone())

	def _history_tensors(self, num_samples, task):
		obs_hist = torch.stack(self._obs_hist, dim=0).unsqueeze(1).repeat(1, num_samples, 1)
		if len(self._action_hist) > 0:
			action_hist = torch.stack(self._action_hist, dim=0).unsqueeze(1).repeat(1, num_samples, 1)
		else:
			action_hist = obs_hist.new_zeros(0, num_samples, self.cfg.action_dim)
		task_tensor = self._task_tensor(task, num_samples)
		return obs_hist, action_hist, task_tensor

	@torch.no_grad()
	def _local_matrices(self, dyn_state):
		"""Extract a one-step linear dynamics and quadratic reward approximation."""
		dynamics = self.model._dynamics
		a = dyn_state["a"]
		c = dyn_state["c"]
		d = dyn_state["d"]
		dt = F.softplus(dynamics.log_dt).item()
		diag = dynamics.diag_head(a)[0]
		control = dynamics.control_head(a).view(-1, dynamics.latent_dim, dynamics.action_dim)[0]
		reward_c = dynamics.reward_c_head(c)[0] / dynamics.latent_dim
		reward_d = dynamics.reward_d_head(d)[0, 0]
		A_diag = (1.0 + dt * diag).detach().cpu().double().numpy()
		B = (dt * control).detach().cpu().double().numpy()
		C_diag = reward_c.detach().cpu().double().numpy()
		D = float(reward_d.detach().cpu())
		return A_diag, B, C_diag, D

	@torch.no_grad()
	def _linearize_horizon(self, task):
		H                                  = self.cfg.horizon
		obs_hist, action_hist, task_tensor = self._history_tensors(1, task)
		dyn_state                          = self.model.init_dynamics_history(obs_hist, action_hist, task_tensor)
		z0                                 = dyn_state["z"][0].detach().cpu().double().numpy()
		A_list, B_list, C_list, D_list     = [], [], [], []

		for k in range(H):
			A_diag, B, C_diag, D = self._local_matrices(dyn_state)
			A_list.append(A_diag)
			B_list.append(B)
			C_list.append(C_diag)
			D_list.append(D)
			nominal_action = torch.as_tensor(
				self._prev_actions[k], dtype=torch.float32, device=self.device
			).unsqueeze(0)
			if self.cfg.multitask:
				nominal_action = nominal_action * self.model._action_masks[task_tensor]
			_, _, dyn_state = self.model.step_obs_norm(dyn_state, nominal_action, task_tensor)
		return z0, A_list, B_list, C_list, D_list, task_tensor

	def _action_mask(self, task_tensor):
		if not self.cfg.multitask:
			return np.ones(self.cfg.action_dim, dtype=np.float64)
		return self.model._action_masks[task_tensor][0].detach().cpu().double().numpy()

	def _solve_qp(self, z0, A_list, B_list, C_list, D_list, action_mask):
		H = self.cfg.horizon
		cp = self.cp
		z_dim = self.cfg.latent_dim
		a_dim = self.cfg.action_dim
		u = cp.Variable((a_dim, H))
		z = cp.Variable((z_dim, H + 1))
		constraints = [z[:, 0] == z0, u <= 1.0, u >= -1.0]
		invalid_actions = np.where(action_mask < 0.5)[0]
		for idx in invalid_actions:
			constraints.append(u[idx, :] == 0.0)

		action_penalty = float(self.cfg.get('mam_mpc_action_penalty', 1e-3))
		delta_penalty = float(self.cfg.get('mam_mpc_delta_penalty', 1e-2))
		terminal_weight = float(self.cfg.get('mam_mpc_terminal_weight', 1.0))
		objective = 0

		for k in range(H):
			constraints.append(
				z[:, k + 1] == cp.multiply(A_list[k], z[:, k]) + B_list[k] @ u[:, k]
			)
			# Max reward z^T C z + D. CVXPY needs a convex minimization, so use
			# the concave part exactly and drop the non-convex positive curvature.
			q = np.maximum(-C_list[k], 0.0) + 1e-8
			weight = terminal_weight if k == H - 1 else 1.0
			objective += weight * cp.sum(cp.multiply(q, cp.square(z[:, k + 1]))) - D_list[k]
			objective += action_penalty * cp.sum_squares(u[:, k])
			if k > 0:
				objective += delta_penalty * cp.sum_squares(u[:, k] - u[:, k - 1])

		problem = cp.Problem(cp.Minimize(objective), constraints)
		start_time = time.time()
		for solver in ('OSQP', 'CLARABEL', 'SCS'):
			try:
				problem.solve(solver=solver, warm_start=True, verbose=False)
			except Exception:
				continue
			self.last_solve_time = time.time() - start_time
			self.last_solver_status = f'{solver}:{problem.status}'
			if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and u.value is not None:
				return np.asarray(u.value, dtype=np.float64).T
		self.last_solve_time = time.time() - start_time
		self.last_solver_status = 'failed'
		return self._prev_actions.copy()

	@torch.no_grad()
	def plan(self, task=None, eval_mode=True):
		z0, A_list, B_list, C_list, D_list, task_tensor = self._linearize_horizon(task)
		action_mask = self._action_mask(task_tensor)
		planned_actions = self._solve_qp(z0, A_list, B_list, C_list, D_list, action_mask)
		planned_actions = np.clip(planned_actions, -1.0, 1.0) * action_mask[None, :]
		action = planned_actions[0].copy()
		self._prev_actions[:-1] = planned_actions[1:]
		self._prev_actions[-1] = planned_actions[-1]
		return torch.as_tensor(action, dtype=torch.float32, device=self.device)

	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=True, task=None):
		if t0 or self._obs_hist is None:
			self._reset_history(obs)
		else:
			self._advance_history(obs)
		action = self.plan(task=task, eval_mode=eval_mode)
		self._last_action = action.detach()
		return action.cpu()


@hydra.main(config_name='config', config_path='.')
def evaluate(cfg: dict):
	"""Evaluate a MamODE model-only checkpoint with MPC, mirroring TD-MPC2 evaluate.py."""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored(f'Task: {cfg.task}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored(f'MPC horizon: {cfg.horizon}', 'blue', attrs=['bold']))
	print(colored(f'History: {cfg.get("model_history", 1)}', 'blue', attrs=['bold']))

	env   = make_env(cfg)
	agent = MamODEMPCAgent(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)

	if cfg.multitask:
		print(colored(f'Evaluating MamODE-CVXPY-MPC on {len(cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
	else:
		print(colored(f'Evaluating MamODE-CVXPY-MPC on {cfg.task}:', 'yellow', attrs=['bold']))
	if cfg.save_video:
		video_dir = os.path.join(cfg.work_dir, 'videos')
		os.makedirs(video_dir, exist_ok=True)

	scores = []
	tasks = cfg.tasks if cfg.multitask else [cfg.task]
	for task_idx, task in enumerate(tasks):
		if not cfg.multitask:
			task_idx = None
		print(colored(f'[{task_idx + 1 if task_idx is not None else 1}/{len(tasks)}] {task}', 'cyan'))
		ep_rewards, ep_successes = [], []
		for i in range(cfg.eval_episodes):
			obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
			if cfg.save_video:
				frames = [env.render()]
			episode_length = cfg.episode_lengths[task_idx] if cfg.multitask else cfg.episode_length
			progress = tqdm(
				total=episode_length,
				desc=f'{task} ep {i + 1}/{cfg.eval_episodes}',
				leave=False,
				dynamic_ncols=True,
			) if tqdm is not None else None
			while not done:
				action = agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
				obs, reward, done, info = env.step(action)
				ep_reward += reward
				t += 1
				if progress is not None:
					progress.update(1)
					progress.set_postfix(
						reward=f'{ep_reward:.1f}',
						mpc=f'{agent.last_solve_time:.2f}s',
						solver=agent.last_solver_status,
					)
				elif t % 25 == 0:
					print(
						f'  {task} ep {i + 1}/{cfg.eval_episodes} '
						f'step {t}/{episode_length} '
						f'R={ep_reward:.1f} '
						f'MPC={agent.last_solve_time:.2f}s '
						f'{agent.last_solver_status}'
					)
				if cfg.save_video:
					frames.append(env.render())
			if progress is not None:
				progress.close()
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			print(colored(
				f'  episode {i + 1}/{cfg.eval_episodes}: '
				f'R={ep_reward:.1f} S={info["success"]:.2f} '
				f'steps={t} last_mpc={agent.last_solve_time:.2f}s '
				f'{agent.last_solver_status}',
				'yellow'
			))
			if cfg.save_video:
				imageio.mimsave(os.path.join(video_dir, f'{task}-{i}.mp4'), frames, fps=15)
		ep_rewards = np.mean(ep_rewards)
		ep_successes = np.mean(ep_successes)
		if cfg.multitask:
			scores.append(ep_successes*100 if task.startswith('mw-') else ep_rewards/10)
		print(colored(f'  {task:<22}'
			f'\tR: {ep_rewards:.01f}  '
			f'\tS: {ep_successes:.02f}', 'yellow'))
	if cfg.multitask:
		print(colored(f'Normalized score: {np.mean(scores):.02f}', 'yellow', attrs=['bold']))


if __name__ == '__main__':
	evaluate()
