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
		self.model.load_state_dict(state_dict)

	@torch.no_grad()
	def predict(self, buffer):
		obs, action, _, _, task = buffer.sample()
		self.model.eval()

		z = self.model.encode(obs[0], task)
		x_preds = []
		for _action in action.unbind(0):
			x_pred = self.model.next_obs(z, _action, task)
			x_preds.append(x_pred)
			z = self.model.encode(x_pred, task)

		x_preds = torch.stack(x_preds)
		target = self.model.mask_obs(obs[1:], task)
		x_preds = self.model.mask_obs(x_preds, task)
		return x_preds, target, task

	def update(self, buffer):
		obs, action, _, _, task = buffer.sample()
		self.model.train()

		z = self.model.encode(obs[0], task)
		x_preds = []
		step_losses = []
		for t, _action in enumerate(action.unbind(0)):
			x_pred = self.model.next_obs(z, _action, task)
			x_preds.append(x_pred)
			step_losses.append(self.model.obs_loss(x_pred, obs[t+1], task))
			z = self.model.encode(x_pred, task)

		x_preds = torch.stack(x_preds)
		x_loss = self.model.obs_loss(x_preds, obs[1:], task)
		one_step_loss = step_losses[0]
		final_step_loss = step_losses[-1]

		x_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		self.model.eval()
		return TensorDict({
			"x_loss": x_loss,
			"one_step_x_loss": one_step_loss,
			"final_step_x_loss": final_step_loss,
			"grad_norm": grad_norm,
		}).detach().mean()
