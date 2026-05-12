import os
from copy import deepcopy
from glob import glob
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
		if len(fps) < (20 if self.cfg.task == 'mt80' else 4):
			print(f'WARNING: expected 20 files for mt80 task set, 4 files for mt30 task set, found {len(fps)} files.')

		_cfg = deepcopy(self.cfg)
		_cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
		_cfg.buffer_size = 550_450_000 if self.cfg.task == 'mt80' else 345_690_000
		_cfg.steps = _cfg.buffer_size
		self.buffer = Buffer(_cfg)
		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp, weights_only=False)
			assert td.shape[1] == _cfg.episode_length, \
				f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
				f'please double-check your config.'
			self.buffer.load(td)

	def _write_csv(self):
		if not self.cfg.save_csv:
			return
		keys = ["iteration", "x_loss", "one_step_x_loss", "final_step_x_loss", "grad_norm", "elapsed_time"]
		pd.DataFrame(self._history, columns=keys).to_csv(
			Path(self.cfg.work_dir) / "model_loss.csv", index=None
		)

	def _plot_predictions(self, iteration):
		try:
			import matplotlib
			matplotlib.use('Agg')
			import matplotlib.pyplot as plt
		except ModuleNotFoundError:
			print('matplotlib is not installed; skipping prediction plot.')
			return

		pred, target, task = self.agent.predict(self.buffer)
		pred = pred.detach().cpu()
		target = target.detach().cpu()
		task = task.detach().cpu() if task is not None else None

		sample_idx = 0
		pred_sample = pred[:, sample_idx]
		target_sample = target[:, sample_idx]
		per_dim_mse = ((pred - target) ** 2).mean(dim=(0, 1))
		valid_dims = torch.nonzero(target_sample.abs().sum(dim=0) > 0, as_tuple=False).flatten()
		if len(valid_dims) == 0:
			valid_dims = torch.arange(target_sample.shape[-1])

		num_plot_dims = min(self.cfg.get('plot_dims', 8), len(valid_dims))
		worst_dims = per_dim_mse[valid_dims].topk(num_plot_dims).indices
		dims = valid_dims[worst_dims].tolist()
		horizon = pred_sample.shape[0]
		t = np.arange(1, horizon + 1)

		fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
		for dim in dims:
			axes[0].plot(t, target_sample[:, dim], linewidth=2, label=f'x{dim} target')
			axes[0].plot(t, pred_sample[:, dim], '--', linewidth=2, label=f'x{dim} pred')
		task_label = f'task {int(task[sample_idx])}' if task is not None else 'single task'
		axes[0].set_title(f'MamODE rollout prediction at iteration {iteration} ({task_label})')
		axes[0].set_xlabel('prediction step')
		axes[0].set_ylabel('state value')
		axes[0].grid(True, alpha=0.3)
		axes[0].legend(ncol=2, fontsize=8)

		axes[1].bar(np.arange(len(per_dim_mse)), per_dim_mse.numpy())
		axes[1].set_title('Per-state prediction MSE over sampled batch')
		axes[1].set_xlabel('state dimension')
		axes[1].set_ylabel('MSE')
		axes[1].grid(True, axis='y', alpha=0.3)

		out_dir = Path(self.cfg.work_dir) / "model_predictions"
		out_dir.mkdir(parents=True, exist_ok=True)
		fig.savefig(out_dir / f'prediction_{iteration:08d}.png', dpi=150)
		plt.close(fig)

	def _plot_task_error_distribution(self, iteration):
		try:
			import matplotlib
			matplotlib.use('Agg')
			import matplotlib.pyplot as plt
		except ModuleNotFoundError:
			print('matplotlib is not installed; skipping task error plot.')
			return

		task_errors = {i: [] for i in range(len(self.cfg.tasks))}
		num_batches = self.cfg.get('task_plot_batches', 8)
		for _ in range(num_batches):
			pred, target, task = self.agent.predict(self.buffer)
			pred = pred.detach().cpu()
			target = target.detach().cpu()
			task = task.detach().cpu()
			err = (pred - target) ** 2

			for b, task_idx in enumerate(task.tolist()):
				if hasattr(self.agent.model, "_obs_masks"):
					mask = self.agent.model._obs_masks[task_idx].detach().cpu()
					valid = mask.sum().clamp_min(1)
					sample_err = (err[:, b] * mask).sum() / (err.shape[0] * valid)
				else:
					sample_err = err[:, b].mean()
				task_errors[task_idx].append(float(sample_err))

		task_names = list(self.cfg.tasks)
		data = [task_errors[i] if task_errors[i] else [np.nan] for i in range(len(task_names))]
		means = np.array([np.nanmean(x) for x in data])

		fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
		axes[0].boxplot(data, showfliers=False)
		axes[0].set_title(f'Per-task rollout MSE distribution at iteration {iteration}')
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
		fig.savefig(out_dir / f'task_error_{iteration:08d}.png', dpi=150)
		plt.close(fig)

		pd.DataFrame({
			"task": task_names,
			"mean_mse": means,
			"num_samples": [len(task_errors[i]) for i in range(len(task_names))],
		}).to_csv(out_dir / f'task_error_{iteration:08d}.csv', index=None)

	def train(self):
		assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
			'Model-only training currently supports multitask mt30/mt80 datasets.'
		self._load_dataset()

		log_freq = self.cfg.get('log_freq', 1000)
		plot_freq = self.cfg.get('plot_freq', 5000)
		print(f'Training dynamics model for {self.cfg.steps} iterations...')
		for i in range(self.cfg.steps):
			metrics = self.agent.update(self.buffer)
			if i % log_freq == 0 or i == self.cfg.steps - 1:
				elapsed_time = time() - self._start_time
				row = [
					i,
					float(metrics["x_loss"]),
					float(metrics["one_step_x_loss"]),
					float(metrics["final_step_x_loss"]),
					float(metrics["grad_norm"]),
					elapsed_time,
				]
				self._history.append(row)
				print(
					f'  model          I: {i:,}   '
					f'x_loss: {row[1]:.6f}   '
					f'one_step: {row[2]:.6f}   '
					f'final_step: {row[3]:.6f}   '
					f'T: {elapsed_time:.0f}s'
				)
				self._write_csv()
				if plot_freq > 0 and (i % plot_freq == 0 or i == self.cfg.steps - 1):
					self._plot_predictions(i)
					self._plot_task_error_distribution(i)
				if self.cfg.save_agent and i > 0 and i % self.cfg.save_freq == 0:
					self.logger.save_agent(self.agent, identifier=f'{i}')

		self.logger.finish(self.agent)
