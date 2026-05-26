import os
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common import TASK_SET
from common.buffer import Buffer
from trainer.base import Trainer


class ModelTrainer(Trainer):
	"""Offline trainer for dynamics modeling only; no policy evaluation."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
		self._history = []

	def _load_dataset(self):
		fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
		fps = sorted(glob(str(fp)))
		assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {fp}')
		source_task = self.cfg.get('source_task', self.cfg.task)
		if self.cfg.multitask or source_task is None or source_task is True or source_task not in TASK_SET:
			source_task = self.cfg.task
		if len(fps) < (20 if source_task == 'mt80' else 4):
			print(f'WARNING: expected 20 files for mt80 task set, 4 files for mt30 task set, found {len(fps)} files.')
		filter_task_id = None
		if not self.cfg.multitask:
			assert source_task in TASK_SET, \
				'Single-task model-only training from offline data requires source_task=mt30 or source_task=mt80.'
			assert self.cfg.task in TASK_SET[source_task], \
				f'Task {self.cfg.task} is not in source task set {source_task}.'
			filter_task_id = TASK_SET[source_task].index(self.cfg.task)
			print(f'Filtering {source_task} dataset for task {self.cfg.task} (task id {filter_task_id}).')

		_cfg = deepcopy(self.cfg)
		_cfg.episode_length = 101 if source_task == 'mt80' else 501
		_cfg.buffer_size = 550_450_000 if source_task == 'mt80' else 345_690_000
		_cfg.steps = _cfg.buffer_size
		cfg_terminal_horizon = self.cfg.get("mam_terminal_horizon", None)
		terminal_horizon = self.cfg.horizon if cfg_terminal_horizon in (None, False) else int(cfg_terminal_horizon)
		model_horizon = max(self.cfg.horizon, terminal_horizon)
		_cfg.horizon = self.cfg.get('model_history', 1) + model_horizon - 1
		assert _cfg.horizon + 1 <= _cfg.episode_length, \
			f'Model history + horizon requires sequence length {_cfg.horizon + 1}, ' \
			f'but {source_task} episodes have length {_cfg.episode_length}.'
		self.buffer = Buffer(_cfg)
		self.test_buffer = Buffer(_cfg)
		test_ratio = self.cfg.get('test_ratio', 0.1)
		generator = torch.Generator().manual_seed(self.cfg.seed)
		obs_dim = self.cfg.obs_shape[self.cfg.obs][0]
		if self.cfg.multitask:
			stat_shape = (len(self.cfg.tasks), obs_dim)
		else:
			stat_shape = (obs_dim,)
		obs_sum = torch.zeros(stat_shape, dtype=torch.float64)
		obs_sumsq = torch.zeros(stat_shape, dtype=torch.float64)
		obs_count = torch.zeros(stat_shape, dtype=torch.float64)
		reward_stat_shape = (len(self.cfg.tasks), 1) if self.cfg.multitask else (1,)
		reward_sum = torch.zeros(reward_stat_shape, dtype=torch.float64)
		reward_sumsq = torch.zeros(reward_stat_shape, dtype=torch.float64)
		reward_count = torch.zeros(reward_stat_shape, dtype=torch.float64)

		def pad_to_model_dims(td):
			obs_target_dim = self.cfg.obs_shape[self.cfg.obs][0]
			action_target_dim = self.cfg.action_dim
			obs_dim = td['obs'].shape[-1]
			action_dim = td['action'].shape[-1]
			assert obs_dim <= obs_target_dim, \
				f'Dataset obs dim {obs_dim} exceeds model obs dim {obs_target_dim}.'
			assert action_dim <= action_target_dim, \
				f'Dataset action dim {action_dim} exceeds model action dim {action_target_dim}.'
			if obs_dim < obs_target_dim:
				td['obs'] = F.pad(td['obs'], (0, obs_target_dim - obs_dim))
			if action_dim < action_target_dim:
				td['action'] = F.pad(td['action'], (0, action_target_dim - action_dim))
			return td

		def update_stats(train_td):
			obs = train_td['obs'].double()
			reward = train_td['reward'].double()

			def update_reward_stats(task_reward, task_idx=None):
				finite = torch.isfinite(task_reward)
				if not finite.any():
					return
				values = task_reward[finite]
				if task_idx is None:
					reward_sum[:] += values.sum()
					reward_sumsq[:] += values.square().sum()
					reward_count[:] += values.numel()
				else:
					reward_sum[task_idx] += values.sum()
					reward_sumsq[task_idx] += values.square().sum()
					reward_count[task_idx] += values.numel()

			if self.cfg.multitask:
				task_ids = train_td['task'][:, 0].long()
				for task_idx in torch.unique(task_ids).tolist():
					task_obs = obs[task_ids == task_idx]
					task_reward = reward[task_ids == task_idx]
					obs_size = self.cfg.obs_shapes[task_idx]
					obs_sum[task_idx, :obs_size] += task_obs[:, :, :obs_size].sum(dim=(0, 1))
					obs_sumsq[task_idx, :obs_size] += task_obs[:, :, :obs_size].square().sum(dim=(0, 1))
					obs_count[task_idx, :obs_size] += task_obs.shape[0] * task_obs.shape[1]
					update_reward_stats(task_reward, task_idx)
			else:
				obs_sum[:] += obs.sum(dim=(0, 1))
				obs_sumsq[:] += obs.square().sum(dim=(0, 1))
				obs_count[:] += obs.shape[0] * obs.shape[1]
				update_reward_stats(reward)

		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp, weights_only=False)
			assert td.shape[1] == _cfg.episode_length, \
				f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
				f'please double-check your config.'
			if filter_task_id is not None:
				task_mask = td['task'][:, 0] == filter_task_id
				if not task_mask.any():
					continue
				td = td[task_mask]
			td = pad_to_model_dims(td)
			num_eps = td.shape[0]
			num_test = max(1, int(num_eps * test_ratio)) if test_ratio > 0 else 0
			perm = torch.randperm(num_eps, generator=generator)
			test_idx = perm[:num_test]
			train_idx = perm[num_test:]
			if len(train_idx) > 0:
				train_td = td[train_idx]
				update_stats(train_td)
				self.buffer.load(train_td)
			if len(test_idx) > 0:
				self.test_buffer.load(td[test_idx])
		print(f'Train episodes: {self.buffer.num_eps:,}')
		print(f'Test episodes: {self.test_buffer.num_eps:,}')
		valid = obs_count > 0
		mean = torch.zeros_like(obs_sum)
		std = torch.ones_like(obs_sum)
		mean[valid] = obs_sum[valid] / obs_count[valid]
		var = torch.zeros_like(obs_sum)
		var[valid] = obs_sumsq[valid] / obs_count[valid] - mean[valid].square()
		std[valid] = var[valid].clamp_min(1e-12).sqrt()
		self.agent.model.set_obs_stats(mean.float(), std.float())
		reward_valid = reward_count > 0
		reward_mean = torch.zeros_like(reward_sum)
		reward_std = torch.ones_like(reward_sum)
		reward_mean[reward_valid] = reward_sum[reward_valid] / reward_count[reward_valid]
		reward_var = torch.zeros_like(reward_sum)
		reward_var[reward_valid] = reward_sumsq[reward_valid] / reward_count[reward_valid] - reward_mean[reward_valid].square()
		reward_std_min = float(self.cfg.get("reward_std_min", 0.1))
		reward_std[reward_valid] = reward_var[reward_valid].clamp_min(reward_std_min ** 2).sqrt()
		reward_mean = torch.nan_to_num(reward_mean, nan=0.0, posinf=0.0, neginf=0.0)
		reward_std = torch.nan_to_num(reward_std, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(reward_std_min)
		self.agent.model.set_reward_stats(reward_mean.float(), reward_std.float())
		print('Observation and reward normalization statistics computed from train split.')
		print(
			f'Reward stats: mean [{reward_mean.min():.4f}, {reward_mean.max():.4f}], '
			f'std [{reward_std.min():.4f}, {reward_std.max():.4f}], '
			f'std_min={reward_std_min:.4f}, finite_count={int(reward_count.sum().item())}'
		)

	def _write_csv(self):
		if not self.cfg.save_csv:
			return
		keys = [
			"epoch",
			"train_total_loss",
			"train_x_loss",
			"train_reward_loss",
			"train_terminal_value_loss",
			"train_one_step_x_loss",
			"train_final_step_x_loss",
			"train_one_step_reward_loss",
			"train_final_step_reward_loss",
			"test_total_loss",
			"test_x_loss",
			"test_reward_loss",
			"test_terminal_value_loss",
			"test_one_step_x_loss",
			"test_final_step_x_loss",
			"grad_norm",
			"elapsed_time",
		]
		pd.DataFrame(self._history, columns=keys).to_csv(
			Path(self.cfg.work_dir) / "model_loss.csv", index=None
		)

	def _plot_predictions(self, epoch):
		try:
			import matplotlib
			matplotlib.use('Agg')
			import matplotlib.pyplot as plt
		except ModuleNotFoundError:
			print('matplotlib is not installed; skipping prediction plot.')
			return

		pred, target, reward_pred, reward_target, action, task = self.agent.predict(self.test_buffer, include_reward=True)
		pred = pred.detach().cpu()
		target = target.detach().cpu()
		reward_pred = reward_pred.detach().cpu()
		reward_target = reward_target.detach().cpu()
		action = action.detach().cpu()
		task = task.detach().cpu() if task is not None else None

		sample_idx = 0
		pred_sample = pred[:, sample_idx]
		target_sample = target[:, sample_idx]
		reward_pred_sample = reward_pred[:, sample_idx].reshape(-1)
		reward_target_sample = reward_target[:, sample_idx].reshape(-1)
		action_sample = action[:, sample_idx]
		reward_mse = F.mse_loss(reward_pred, reward_target).item()
		pred_norm = self.agent.model.normalize_obs(pred.to(self.agent.device), task.to(self.agent.device) if task is not None else None).cpu()
		target_norm = self.agent.model.normalize_obs(target.to(self.agent.device), task.to(self.agent.device) if task is not None else None).cpu()
		per_dim_mse = ((pred_norm[:, sample_idx] - target_norm[:, sample_idx]) ** 2).mean(dim=0)
		if task is not None and hasattr(self.agent.model, "_obs_masks"):
			task_idx = int(task[sample_idx])
			valid_dims = torch.nonzero(self.agent.model._obs_masks[task_idx].detach().cpu() > 0, as_tuple=False).flatten()
		else:
			task_idx = None
			valid_dims = torch.nonzero(target_sample.abs().sum(dim=0) > 0, as_tuple=False).flatten()
			if len(valid_dims) == 0:
				valid_dims = torch.arange(target_sample.shape[-1])

		num_plot_dims = min(self.cfg.get('plot_dims', 8), len(valid_dims))
		worst_dims = per_dim_mse[valid_dims].topk(num_plot_dims).indices
		dims = valid_dims[worst_dims].tolist()
		horizon = pred_sample.shape[0]
		t = np.arange(1, horizon + 1)
		task_names = list(self.cfg.tasks)

		fig, axes = plt.subplots(4, 1, figsize=(12, 14), constrained_layout=True)
		for dim in dims:
			axes[0].plot(t, target_sample[:, dim], linewidth=2, label=f'x{dim} target')
			axes[0].plot(t, pred_sample[:, dim], '--', linewidth=2, label=f'x{dim} pred')
		if task is not None and task_idx is not None:
			task_label = task_names[task_idx] if task_idx < len(task_names) else str(task_idx)
		else:
			task_label = self.cfg.task
		axes[0].set_title(f'MamODE test rollout prediction at epoch {epoch} ({task_label})')
		axes[0].set_xlabel('prediction step')
		axes[0].set_ylabel('state value')
		axes[0].grid(True, alpha=0.3)
		axes[0].legend(ncol=2, fontsize=8)

		axes[1].plot(t, reward_target_sample.numpy(), linewidth=2, label='reward target')
		axes[1].plot(t, reward_pred_sample.numpy(), '--', linewidth=2, label='reward pred')
		axes[1].set_title(f'Reward rollout prediction, sampled batch MSE {reward_mse:.6f}')
		axes[1].set_xlabel('prediction step')
		axes[1].set_ylabel('reward')
		axes[1].grid(True, alpha=0.3)
		axes[1].legend(fontsize=8)

		if task is not None and hasattr(self.agent.model, "_action_masks"):
			action_dim = int(self.agent.model._action_masks[int(task[sample_idx])].sum().item())
		else:
			action_dim = action_sample.shape[-1]
		for dim in range(action_dim):
			axes[2].plot(t, action_sample[:, dim].numpy(), linewidth=1.5, label=f'a{dim}')
		axes[2].set_title('Actions used for the rollout')
		axes[2].set_xlabel('prediction step')
		axes[2].set_ylabel('action')
		axes[2].set_ylim(-1.05, 1.05)
		axes[2].grid(True, alpha=0.3)
		axes[2].legend(ncol=3, fontsize=8)

		axes[3].bar(valid_dims.numpy(), per_dim_mse[valid_dims].numpy())
		axes[3].set_title('Per-state normalized prediction MSE for sampled task')
		axes[3].set_xlabel('state dimension')
		axes[3].set_ylabel('MSE')
		axes[3].grid(True, axis='y', alpha=0.3)

		out_dir = Path(self.cfg.work_dir) / "model_predictions"
		out_dir.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_dir / f'prediction_epoch_{epoch:04d}.png', dpi=150)
		plt.close(fig)

	def _plot_task_error_distribution(self, epoch):
		try:
			import matplotlib
			matplotlib.use('Agg')
			import matplotlib.pyplot as plt
		except ModuleNotFoundError:
			print('matplotlib is not installed; skipping task error plot.')
			return

		task_names = list(self.cfg.tasks)
		task_errors = {i: [] for i in range(len(task_names))}
		num_batches = self.cfg.get('task_plot_batches', 8)
		for _ in range(num_batches):
			pred, target, task = self.agent.predict(self.test_buffer)
			pred = pred.detach().cpu()
			target = target.detach().cpu()
			task = task.detach().cpu() if task is not None else None
			pred_norm = self.agent.model.normalize_obs(pred.to(self.agent.device), task.to(self.agent.device) if task is not None else None).cpu()
			target_norm = self.agent.model.normalize_obs(target.to(self.agent.device), task.to(self.agent.device) if task is not None else None).cpu()
			err = (pred_norm - target_norm) ** 2

			task_indices = task.tolist() if task is not None else [0] * pred.shape[1]
			for b, task_idx in enumerate(task_indices):
				if not self.cfg.multitask:
					task_idx = 0
				if hasattr(self.agent.model, "_obs_masks"):
					mask = self.agent.model._obs_masks[task_idx].detach().cpu()
					valid = mask.sum().clamp_min(1)
					sample_err = (err[:, b] * mask).sum() / (err.shape[0] * valid)
				else:
					sample_err = err[:, b].mean()
				task_errors[task_idx].append(float(sample_err))

		data = [task_errors[i] if task_errors[i] else [np.nan] for i in range(len(task_names))]
		means = np.array([np.nanmean(x) for x in data])

		fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
		axes[0].boxplot(data, showfliers=False)
		axes[0].set_title(f'Per-task test rollout MSE distribution at epoch {epoch}')
		axes[0].set_ylabel('MSE')
		axes[0].set_xticks(np.arange(1, len(task_names) + 1))
		axes[0].set_xticklabels(task_names, rotation=70, ha='right', fontsize=8)
		axes[0].grid(True, axis='y', alpha=0.3)

		order = np.argsort(means)[::-1]
		axes[1].bar(np.arange(len(task_names)), means[order])
		axes[1].set_title('Mean rollout MSE by task, sorted high to low')
		axes[1].set_ylabel('mean MSE')
		axes[1].set_xticks(np.arange(len(task_names)))
		axes[1].set_xticklabels([task_names[i] for i in order], rotation=70, ha='right', fontsize=8)
		axes[1].grid(True, axis='y', alpha=0.3)

		out_dir = Path(self.cfg.work_dir) / "model_predictions"
		out_dir.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_dir / f'task_error_epoch_{epoch:04d}.png', dpi=150)
		plt.close(fig)

		pd.DataFrame({
			"task": task_names,
			"mean_mse": means,
			"num_samples": [len(task_errors[i]) for i in range(len(task_names))],
		}).to_csv(out_dir / f'task_error_epoch_{epoch:04d}.csv', index=None)

	def train(self):
		source_task = self.cfg.get('source_task', self.cfg.task)
		if self.cfg.multitask or source_task is None or source_task is True or source_task not in TASK_SET:
			source_task = self.cfg.task
		assert (self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}) or source_task in {'mt30', 'mt80'}, \
			'Model-only training supports mt30/mt80 or a single task filtered from source_task=mt30/mt80.'
		self._load_dataset()

		model_epochs = self.cfg.get('model_epochs', 100)
		updates_per_epoch = self.cfg.get('updates_per_epoch', 1000)
		eval_batches = self.cfg.get('eval_batches', 20)
		plot_every_epochs = self.cfg.get('plot_every_epochs', 10)
		save_every_epochs = self.cfg.get('save_every_epochs', 10)
		print(f'Training dynamics model for {model_epochs} epochs x {updates_per_epoch} updates/epoch...')
		for epoch in range(1, model_epochs + 1):
			train_acc = {
				"total_loss": 0.,
				"x_loss": 0.,
				"reward_loss": 0.,
				"terminal_value_loss": 0.,
				"one_step_x_loss": 0.,
				"final_step_x_loss": 0.,
				"one_step_reward_loss": 0.,
				"final_step_reward_loss": 0.,
				"grad_norm": 0.,
			}
			for _ in range(updates_per_epoch):
				metrics = self.agent.update(self.buffer)
				for k in train_acc.keys():
					train_acc[k] += float(metrics[k])
			for k in train_acc.keys():
				train_acc[k] /= updates_per_epoch

			test_metrics = self.agent.evaluate(self.test_buffer, num_batches=eval_batches)
			elapsed_time = time() - self._start_time
			test_total_loss = (
				self.cfg.get("state_model_coef", 0.0) * test_metrics["x_loss"] +
				self.cfg.get("reward_model_coef", 1.0) * test_metrics["reward_loss"] +
				self.cfg.get("mam_terminal_value_coef", 0.0) * test_metrics["terminal_value_loss"]
			)
			row = [
				epoch,
				train_acc["total_loss"],
				train_acc["x_loss"],
				train_acc["reward_loss"],
				train_acc["terminal_value_loss"],
				train_acc["one_step_x_loss"],
				train_acc["final_step_x_loss"],
				train_acc["one_step_reward_loss"],
				train_acc["final_step_reward_loss"],
				float(test_total_loss),
				float(test_metrics["x_loss"]),
				float(test_metrics["reward_loss"]),
				float(test_metrics["terminal_value_loss"]),
				float(test_metrics["one_step_x_loss"]),
				float(test_metrics["final_step_x_loss"]),
				train_acc["grad_norm"],
				elapsed_time,
			]
			self._history.append(row)
			print(
				f'  epoch {epoch:04d}   '
				f'train total/x/r/t: {row[1]:.6f} / {row[2]:.6f} / {row[3]:.6f} / {row[4]:.6f}   '
				f'test total/x/r/t: {row[9]:.6f} / {row[10]:.6f} / {row[11]:.6f} / {row[12]:.6f}   '
				f'grad: {row[15]:.3f}   T: {elapsed_time:.0f}s'
			)
			self._write_csv()

			if plot_every_epochs > 0 and (epoch % plot_every_epochs == 0 or epoch == model_epochs):
				self._plot_predictions(epoch)
				self._plot_task_error_distribution(epoch)
			if self.cfg.save_agent and save_every_epochs > 0 and (epoch % save_every_epochs == 0 or epoch == model_epochs):
				self.logger.save_agent(self.agent, identifier=f'epoch_{epoch}')

		self.logger.finish(self.agent)
