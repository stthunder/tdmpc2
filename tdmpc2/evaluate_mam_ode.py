import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
import warnings
warnings.filterwarnings('ignore')
import time

import hydra
import imageio
import numpy as np
import torch
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


def _fmt_optional(value):
	if value is None:
		return 'nan'
	return f'{float(value):.6f}'


def _parse_time_points(value):
	if value in (None, False, 'none', 'None', ''):
		return None
	if isinstance(value, (list, tuple)):
		points = [int(v) for v in value]
	else:
		text = str(value)
		for ch in ',;:/':
			text = text.replace(ch, '_')
		points = [int(v) for v in text.split('_') if v]
	assert points and all(p > 0 for p in points), 'mam_mpc_time_points must contain positive integers.'
	assert all(b > a for a, b in zip(points, points[1:])), 'mam_mpc_time_points must be strictly increasing.'
	return points


def print_eval_step(task, episode, step, action, reward, agent, progress=None):
	action_str = np.array2string(
		action.detach().cpu().numpy(),
		precision=4,
		suppress_small=True,
		separator=', ',
	)
	model_rewards = agent.last_model_plan_rewards
	qp_rewards = agent.last_plan_rewards
	model_first = model_rewards[0] if model_rewards is not None and len(model_rewards) else None
	model_mean = np.mean(model_rewards) if model_rewards is not None and len(model_rewards) else None
	qp_first = qp_rewards[0] if qp_rewards is not None and len(qp_rewards) else None
	qp_mean = np.mean(qp_rewards) if qp_rewards is not None and len(qp_rewards) else None
	message = (
		f'  {task} ep {episode} step {step} '
		f'env_reward={float(reward):.6f} '
		f'model_reward={_fmt_optional(model_first)} '
		f'model_reward_mean={_fmt_optional(model_mean)} '
		f'qp_reward={_fmt_optional(qp_first)} '
		f'qp_reward_mean={_fmt_optional(qp_mean)} '
		f'action={action_str}'
	)
	if progress is not None:
		progress.write(message)
	else:
		print(message)


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
		self.mpc_time_points = _parse_time_points(self.cfg.get('mam_mpc_time_points', None))
		if self.mpc_time_points is None:
			self.mpc_interval_steps = [1] * self.cfg.horizon
		else:
			prev = 0
			self.mpc_interval_steps = []
			for point in self.mpc_time_points:
				self.mpc_interval_steps.append(point - prev)
				prev = point
		self.plan_horizon = len(self.mpc_interval_steps)
		self._prev_actions = np.zeros((self.plan_horizon, self.cfg.action_dim), dtype=np.float64)
		self._obs_hist = None
		self._action_hist = None
		self._last_action = None
		self.last_solve_time = 0.0
		self.last_solver_status = 'not_started'
		self.last_plan_rewards = None
		self.last_model_plan_rewards = None
		self._build_qp_problem()

	def load(self, fp):
		state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict['model'] if 'model' in state_dict else state_dict
		state_dict.setdefault("_reward_mean", self.model._reward_mean)
		state_dict.setdefault("_reward_std", self.model._reward_std)
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
		self.last_model_plan_rewards = None

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
		"""Extract model-generated controller matrices."""
		A_diag, B, C_diag, D, E, next_state = self.model._dynamics.controller_matrices(dyn_state)
		A_diag                           = A_diag[0].detach().cpu().double().numpy()
		B                                = B[0].detach().cpu().double().numpy()
		C_diag                           = C_diag[0].detach().cpu().double().numpy()
		D                                = D[0].detach().cpu().double().numpy()
		E                                = float(E[0].detach().cpu())
		return A_diag, B, C_diag, D, E, next_state

	@torch.no_grad()
	def _macro_matrices(self, dyn_state, interval_steps):
		macro_A = None
		macro_B = None
		C_diag = D = E = None
		for _ in range(int(interval_steps)):
			A_diag, B, C_diag, D, E, dyn_state = self._local_matrices(dyn_state)
			if macro_A is None:
				macro_A = A_diag
				macro_B = B
			else:
				macro_B = A_diag[:, None] * macro_B + B
				macro_A = A_diag * macro_A
		return macro_A, macro_B, C_diag, D, E, dyn_state

	@torch.no_grad()
	def _linearize_horizon(self, task):
		obs_hist, action_hist, task_tensor = self._history_tensors(1, task)
		dyn_state                          = self.model.init_dynamics_history(obs_hist, action_hist, task_tensor)
		z0                                 = dyn_state["z"][0].detach().cpu().double().numpy()
		A_list, B_list, C_list, D_list, E_list = [], [], [], [], []

		for interval_steps in self.mpc_interval_steps:
			A_diag, B, C_diag, D, E, dyn_state = self._macro_matrices(dyn_state, interval_steps)
			A_list.append(A_diag)
			B_list.append(B)
			C_list.append(C_diag)
			D_list.append(D)
			E_list.append(E)
		return z0, A_list, B_list, C_list, D_list, E_list, task_tensor

	def _action_mask(self, task_tensor):
		if not self.cfg.multitask:
			return np.ones(self.cfg.action_dim, dtype=np.float64)
		return self.model._action_masks[task_tensor][0].detach().cpu().double().numpy()

	def _build_qp_problem(self):
		H                    = self.plan_horizon
		cp                   = self.cp
		z_dim                = self.cfg.latent_dim
		a_dim                = self.cfg.action_dim
		self.qp_u            = cp.Variable((a_dim, H))
		self.qp_z            = cp.Variable((z_dim, H + 1))
		self.qp_z0           = cp.Parameter(z_dim)
		self.qp_action_bound = cp.Parameter((a_dim, H), nonneg=True)
		self.qp_A            = [cp.Parameter(z_dim) for _ in range(H)]
		self.qp_B            = [cp.Parameter((z_dim, a_dim)) for _ in range(H)]
		self.qp_q            = [cp.Parameter(z_dim, nonneg=True) for _ in range(H)]
		self.qp_D            = [cp.Parameter(z_dim) for _ in range(H)]
		constraints = [
			self.qp_z[:, 0] == self.qp_z0,
			self.qp_u <= self.qp_action_bound,
			self.qp_u >= -self.qp_action_bound,
		]

		action_penalty  = float(self.cfg.get('mam_mpc_action_penalty', 1e-3))
		delta_penalty   = float(self.cfg.get('mam_mpc_delta_penalty', 1e-2))
		reward_weight   = float(self.cfg.get('mam_mpc_reward_weight', 5.0))
		terminal_weight = float(self.cfg.get('mam_mpc_terminal_weight', 1.0))
		objective = 0

		for k in range(H):
			constraints.append(
				self.qp_z[:, k + 1] == cp.multiply(self.qp_A[k], self.qp_z[:, k]) + self.qp_B[k] @ self.qp_u[:, k]
			)
			# Max reward C z^2 + D z. CVXPY needs a convex minimization, so use
			# the concave part exactly and drop the non-convex positive curvature.
			weight = terminal_weight if k == H - 1 else 1.0
			objective += reward_weight * weight * (
				cp.sum(cp.multiply(self.qp_q[k], cp.square(self.qp_z[:, k + 1])))
				- self.qp_D[k] @ self.qp_z[:, k + 1]
			)
			objective += action_penalty * cp.sum_squares(self.qp_u[:, k])
			if k > 0:
				objective += delta_penalty * cp.sum_squares(self.qp_u[:, k] - self.qp_u[:, k - 1])

		self.qp_problem = cp.Problem(cp.Minimize(objective), constraints)

	def _solve_qp(self, z0, A_list, B_list, C_list, D_list, E_list, action_mask):
		H = self.plan_horizon
		cp = self.cp
		self.last_plan_rewards = None
		self.qp_z0.value = z0
		self.qp_action_bound.value = np.repeat(action_mask[:, None], H, axis=1)
		for k in range(H):
			self.qp_A[k].value = A_list[k]
			self.qp_B[k].value = B_list[k]
			self.qp_q[k].value = np.maximum(-C_list[k], 0.0) + 1e-8
			self.qp_D[k].value = D_list[k]

		# Diagnostic: print reward matrix norms once to verify reward model is informative
		if not hasattr(self, '_diag_printed'):
			self._diag_printed = True
			q_mean = np.mean([np.mean(np.abs(self.qp_q[k].value)) for k in range(H)])
			d_mean = np.mean([np.mean(np.abs(self.qp_D[k].value)) for k in range(H)])
			b_mean = np.mean([np.mean(np.abs(self.qp_B[k].value)) for k in range(H)])
			a_mean = np.mean([np.mean(np.abs(self.qp_A[k].value - 1.0)) for k in range(H)])
			print(f'[MPC diag] |C|={q_mean:.4e}  |D|={d_mean:.4e}  |B|={b_mean:.4e}  |A-I|={a_mean:.4e}')

		start_time = time.time()
		for solver in ('OSQP', 'CLARABEL', 'SCS'):
			try:
				self.qp_problem.solve(solver=solver, warm_start=True, verbose=False)
			except Exception:
				continue
			self.last_solve_time = time.time() - start_time
			self.last_solver_status = f'{solver}:{self.qp_problem.status}'
			if self.qp_problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and self.qp_u.value is not None:
				z_plan = np.asarray(self.qp_z.value, dtype=np.float64).T
				self.last_plan_rewards = np.asarray([
					np.sum(C_list[k] * np.square(z_plan[k + 1])) + np.dot(D_list[k], z_plan[k + 1]) + E_list[k]
					for k in range(H)
				], dtype=np.float64)
				return np.asarray(self.qp_u.value, dtype=np.float64).T
		self.last_solve_time = time.time() - start_time
		self.last_solver_status = 'failed'
		return self._prev_actions.copy()

	@torch.no_grad()
	def _model_rollout_rewards(self, planned_actions, task):
		obs_hist, action_hist, task_tensor = self._history_tensors(1, task)
		dyn_state = self.model.init_dynamics_history(obs_hist, action_hist, task_tensor)
		actions = torch.as_tensor(planned_actions, dtype=torch.float32, device=self.device)
		rewards = []
		for action, interval_steps in zip(actions, self.mpc_interval_steps):
			for _ in range(int(interval_steps)):
				_, reward, dyn_state = self.model._dynamics.step(dyn_state, action.unsqueeze(0))
			rewards.append(float(reward.squeeze().detach().cpu()))
		return np.asarray(rewards, dtype=np.float64)

	@torch.no_grad()
	def plan(self, task=None, eval_mode=True):
		z0, A_list, B_list, C_list, D_list, E_list, task_tensor = self._linearize_horizon(task)
		action_mask                                     = self._action_mask(task_tensor)
		planned_actions = self._solve_qp(z0, A_list, B_list, C_list, D_list, E_list, action_mask)
		planned_actions = np.clip(planned_actions, -1.0, 1.0) * action_mask[None, :]
		self.last_model_plan_rewards = self._model_rollout_rewards(planned_actions, task)
		if self.cfg.get('mam_mpc_print_plan', False) and self.last_plan_rewards is not None:
			reward_str = np.array2string(
				self.last_plan_rewards,
				precision=4,
				suppress_small=True,
				separator=', ',
			)
			print(f'  planned_reward={reward_str}')
			model_reward_str = np.array2string(
				self.last_model_plan_rewards,
				precision=4,
				suppress_small=True,
				separator=', ',
			)
			print(f'  model_rollout_reward={model_reward_str}')
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
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'red', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored(f'MPC horizon: {cfg.horizon}', 'blue', attrs=['bold']))
	if cfg.get('mam_mpc_time_points', None) not in (None, False, 'none', 'None', ''):
		print(colored(f'MPC time points: {cfg.get("mam_mpc_time_points")}', 'blue', attrs=['bold']))
	print(colored(f'History: {cfg.get("model_history", 1)}', 'blue', attrs=['bold']))
	print(colored(f'MPC reward weight: {cfg.get("mam_mpc_reward_weight", 5.0)}', 'red', attrs=['bold']))

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
	if cfg.multitask:
		task_items = list(enumerate(cfg.tasks))
		eval_task = cfg.get('eval_task', None)
		if eval_task not in {None, True, 'all'}:
			task_items = [(idx, task) for idx, task in task_items if task == eval_task]
			assert task_items, f'eval_task={eval_task} is not in task set {cfg.task}.'
	else:
		task_items = [(None, cfg.task)]
	for task_num, (task_idx, task) in enumerate(task_items, start=1):
		print(colored(f'[{task_num}/{len(task_items)}] {task}', 'cyan'))
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
				print_eval_step(task, i + 1, t, action, reward, agent, progress)
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
