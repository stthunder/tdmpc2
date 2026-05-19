from copy import deepcopy
import warnings

import gymnasium as gym
import numpy as np
import torch

from common import TASK_SET
from envs.wrappers.multitask import MultitaskWrapper
from envs.wrappers.tensor import TensorWrapper

def missing_dependencies(task):
	raise ValueError(f'Missing dependencies for task {task}; install dependencies to use this environment.')

try:
	from envs.dmcontrol import make_env as make_dm_control_env
except:
	make_dm_control_env = missing_dependencies
try:
	from envs.maniskill import make_env as make_maniskill_env
except:
	make_maniskill_env = missing_dependencies
try:
	from envs.metaworld import make_env as make_metaworld_env
except:
	make_metaworld_env = missing_dependencies
try:
	from envs.myosuite import make_env as make_myosuite_env
except:
	make_myosuite_env = missing_dependencies
try:
	from envs.mujoco import make_env as make_mujoco_env
except:
	make_mujoco_env = missing_dependencies


warnings.filterwarnings('ignore', category=DeprecationWarning)


class SingleTaskPaddingWrapper(gym.Wrapper):
	"""Pad a single-task environment to match a source multi-task action/obs shape."""

	def __init__(self, env, obs_dim, action_dim):
		super().__init__(env)
		self._obs_dim = obs_dim
		self._action_dim = action_dim
		self._env_action_dim = env.action_space.shape[0]
		self.observation_space = gym.spaces.Box(
			low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
		)
		self.action_space = gym.spaces.Box(
			low=-1, high=1, shape=(action_dim,), dtype=np.float32
		)

	def _pad_obs(self, obs):
		if obs.shape[-1] == self._obs_dim:
			return obs
		return torch.cat((obs, torch.zeros(self._obs_dim-obs.shape[-1], dtype=obs.dtype, device=obs.device)))

	def reset(self, task_idx=None):
		return self._pad_obs(self.env.reset())

	def step(self, action):
		obs, reward, done, info = self.env.step(action[:self._env_action_dim])
		return self._pad_obs(obs), reward, done, info


def make_single_env(cfg):
	"""
	Make one unpadded environment.
	"""
	env = None
	for fn in [make_dm_control_env, make_maniskill_env, make_metaworld_env, make_myosuite_env, make_mujoco_env]:
		try:
			env = fn(cfg)
		except ValueError:
			pass
	if env is None:
		raise ValueError(f'Failed to make environment "{cfg.task}": please verify that dependencies are installed and that the task exists.')
	return TensorWrapper(env)


def get_source_task_dims(cfg, source_task):
	"""
	Get the padded dimensions used by a multi-task source dataset.
	"""
	obs_dims, action_dims = [], []
	for task in TASK_SET[source_task]:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		_cfg.pad_to_source_task = False
		env = make_single_env(_cfg)
		obs_dims.append(env.observation_space.shape[0])
		action_dims.append(env.action_space.shape[0])
	return max(obs_dims), max(action_dims)


def make_multitask_env(cfg):
	"""
	Make a multi-task environment for TD-MPC2 experiments.
	"""
	print('Creating multi-task environment with tasks:', cfg.tasks)
	envs = []
	for task in cfg.tasks:
		_cfg = deepcopy(cfg)
		_cfg.task = task
		_cfg.multitask = False
		env = make_single_env(_cfg)
		if env is None:
			raise ValueError('Unknown task:', task)
		envs.append(env)
	env = MultitaskWrapper(cfg, envs)
	cfg.obs_shapes = env._obs_dims
	cfg.action_dims = env._action_dims
	cfg.episode_lengths = env._episode_lengths
	return env
	

def make_env(cfg):
	"""
	Make an environment for TD-MPC2 experiments.
	"""
	gym.logger.set_level(40)
	if cfg.multitask:
		env = make_multitask_env(cfg)

	else:
		env = make_single_env(cfg)
		source_task = cfg.get('source_task', None)
		if cfg.get('pad_to_source_task', False) and source_task in TASK_SET:
			obs_dim, action_dim = get_source_task_dims(cfg, source_task)
			env = SingleTaskPaddingWrapper(env, obs_dim, action_dim)
	try: # Dict
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except: # Box
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5*cfg.episode_length)
	return env
