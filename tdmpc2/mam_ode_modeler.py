import torch
import torch.nn.functional as F
from tensordict import TensorDict

from common.world_model import WorldModel


class MamODEModeler(torch.nn.Module):
	"""
	Offline dynamics modeler for MamODE.

	This is intentionally not an RL agent: it trains only a dynamics model with
	multi-step state prediction loss over padded observations.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		assert self.model.is_mam_ode, 'MamODEModeler requires world_model=mam_ode.'
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []},
		], lr=self.cfg.lr, capturable=True)
		self.model.eval()

	def save(self, fp):
		torch.save({"model": self.model.state_dict()}, fp)

	def load(self, fp):
		state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		state_dict.setdefault("_reward_mean", self.model._reward_mean)
		state_dict.setdefault("_reward_std", self.model._reward_std)
		self.model.load_state_dict(state_dict, strict=False)

	def _rollout(self, obs, action, reward, task):
		O = self.cfg.get("model_history", 1)
		H = self.cfg.horizon
		cfg_terminal_horizon = self.cfg.get("mam_terminal_horizon", None)
		terminal_horizon = H if cfg_terminal_horizon in (None, False) else max(int(cfg_terminal_horizon), H)
		obs_hist = obs[:O]
		action_hist = action[:max(O-1, 0)]
		future_actions = action[O-1:O-1+H]
		reward_target = reward[O-1:O-1+H]
		target = obs[O:O+H]

		dyn_state = self.model.init_dynamics_history(obs_hist, action_hist, task)
		x_preds_norm = []
		x_preds = []
		reward_preds = []
		step_losses = []
		reward_step_losses = []
		for t, _action in enumerate(future_actions.unbind(0)):
			x_pred_norm, reward_pred, dyn_state = self.model.step_obs_norm(dyn_state, _action, task)
			x_preds_norm.append(x_pred_norm)
			reward_preds.append(reward_pred)
			step_losses.append(self.model.obs_loss(x_pred_norm, target[t], task))
			reward_step_losses.append(self.model.reward_loss(reward_pred, reward_target[t], task))
			x_preds.append(self.model.denormalize_obs(x_pred_norm, task))
		terminal_value_pred = self.model._dynamics.terminal_value_from_state(dyn_state)
		if terminal_horizon > H:
			terminal_value_target = reward[O-1+H:O-1+terminal_horizon].mean(dim=0)
		else:
			terminal_value_target = reward.new_zeros(reward_target.shape[1], reward_target.shape[2])
		terminal_value_loss = F.mse_loss(terminal_value_pred, terminal_value_target)
		return (
			torch.stack(x_preds_norm),
			torch.stack(x_preds),
			target,
			torch.stack(reward_preds),
			reward_target,
			future_actions,
			step_losses,
			reward_step_losses,
			terminal_value_pred,
			terminal_value_target,
			terminal_value_loss,
		)

	@torch.no_grad()
	def predict(self, buffer, include_reward=False):
		obs, action, reward, _, task = buffer.sample()
		self.model.eval()

		_, x_preds, target, reward_preds, reward_target, future_actions, _, _, _, _, _ = self._rollout(obs, action, reward, task)
		x_preds = self.model.mask_obs(x_preds, task)
		target = self.model.mask_obs(target, task)
		if include_reward:
			return x_preds, target, reward_preds, reward_target, future_actions, task
		return x_preds, target, task

	@torch.no_grad()
	def evaluate(self, buffer, num_batches=10):
		self.model.eval()
		x_loss, reward_loss, terminal_loss, one_step_loss, final_step_loss = 0, 0, 0, 0, 0
		for _ in range(num_batches):
			obs, action, reward, _, task = buffer.sample()
			x_preds_norm, _, target, reward_preds, reward_target, _, step_losses, _, _, _, terminal_value_loss = self._rollout(obs, action, reward, task)
			x_loss += self.model.obs_loss(x_preds_norm, target, task)
			reward_loss += self.model.reward_loss(reward_preds, reward_target, task)
			terminal_loss += terminal_value_loss
			one_step_loss += step_losses[0]
			final_step_loss += step_losses[-1]

		num_batches = max(num_batches, 1)
		return TensorDict({
			"x_loss": x_loss / num_batches,
			"reward_loss": reward_loss / num_batches,
			"terminal_value_loss": terminal_loss / num_batches,
			"one_step_x_loss": one_step_loss / num_batches,
			"final_step_x_loss": final_step_loss / num_batches,
		}).detach().mean()

	def update(self, buffer):
		obs, action, reward, _, task = buffer.sample()
		self.model.train()

		x_preds_norm, _, target, reward_preds, reward_target, _, step_losses, reward_step_losses, _, _, terminal_value_loss = self._rollout(obs, action, reward, task)
		x_loss = self.model.obs_loss(x_preds_norm, target, task)
		reward_loss = self.model.reward_loss(reward_preds, reward_target, task)
		total_loss = (
			self.cfg.get("state_model_coef", 0.0) * x_loss +
			self.cfg.get("reward_model_coef", 1.0) * reward_loss +
			self.cfg.get("mam_terminal_value_coef", 0.0) * terminal_value_loss
		)
		one_step_loss = step_losses[0]
		final_step_loss = step_losses[-1]
		one_step_reward_loss = reward_step_losses[0]
		final_step_reward_loss = reward_step_losses[-1]

		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		self.model.eval()
		return TensorDict({
			"x_loss": x_loss,
			"reward_loss": reward_loss,
			"terminal_value_loss": terminal_value_loss,
			"total_loss": total_loss,
			"one_step_x_loss": one_step_loss,
			"final_step_x_loss": final_step_loss,
			"one_step_reward_loss": one_step_reward_loss,
			"final_step_reward_loss": final_step_reward_loss,
			"grad_norm": grad_norm,
		}).detach().mean()
