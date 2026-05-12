"""mamba_joint_ode.py
------------------------------------------------------------
End‑to‑end learnable **linear time‑varying** (LTV) model in which
*the diagonal of A(t), the flattened B(t), and the latent state z(t)*
are integrated **jointly** with a single call to `torchdiffeq.odeint`.

Author : ChatGPT (o3)
Date   : 2025‑06‑12
License: MIT
------------------------------------------------------------
Usage
-----
```python
from mamba_joint_ode import MambaJoint

model = MambaJoint(cfg).to(device)
loss, y_pred = model(x, u, p)  # shapes see docstring
```
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchdiffeq import odeint  # pip install torchdiffeq

import torch
import numpy as np
import math
import copy

import torch.nn as nn
import torch.optim as optim
import torch.distributions as torchd
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
from einops import rearrange, repeat, einsum
import torch.nn.functional as F
import torch.nn.init as init

from torchdiffeq import odeint,odeint_adjoint



# from torch.utils.tensorboard import SummaryWriter
# import datetime, math, torch

# writer = SummaryWriter(log_dir=f"runs/{datetime.datetime.now():%Y-%m-%d_%H-%M}")

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


SCALE_DIAG_MIN_MAX = (-20, 2)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.normal_(m.bias, mean=0,std=0.1)

class Koopman_Desko(object):
    """
    """
    def __init__(
        self,
        args,
        **kwargs
    ):
        self.shift = None
        self.scale = None
        self.shift_u = None
        self.scale_u = None

        self.loss = 0

        self.loss_store = 0
        self.loss_store_t = 0

        # if args['control']:
        #     self.device = torch.device('cpu')
        # else:
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        args['device']      = self.device
        self.time_training  = args['times_training'] 
        self.env            = args['env']
        self.step_          = args['pred_horizon']
        self.eval           = args['eval']

        self.net = Mamba(args)
        self.net.apply(weights_init)

        self.net_para = {}
        self.loss_buff = 100000

        self.stable_x = []
        self.stable_u = []



        self.optimizer1 = self.get_optimizer(self.net,args['lr1'])

        # self.optimizer1 = optim.Adam([{'params': self.net.parameters(),'lr':args['lr1'],'weight_decay':args['weight_decay']}])
        self.optimizer1_sch = torch.optim.lr_scheduler.StepLR(self.optimizer1, step_size=args['optimize_step'], gamma=args['gamma']) 

        self.MODEL_SAVE     = args['save_model_path']
        self.OPTI1          = args['save_opti_path']

        self.loss_t = 0
        self.epoch  = 0

        # self.USE_DIM = [0,2,3,4,5,10,11,12]

        for m in self.net.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0        # 统一放大

        self.net.to(self.device)

        # writer.add_graph(self.net, input_to_model = (torch.rand(1, args['old_horizon']+args['pred_horizon'], args['state_dim']),torch.rand(1, args['old_horizon']+args['pred_horizon']-1, args['act_dim'])))
    
    def get_optimizer(self, model, lr=1e-3):
        # 1. Collect parameters
        a_ode_params  = list(model.a_ode.parameters())
        x_proj_params = list(model.x_proj.parameters())
        p_proj_params = list(model.p_proj.parameters())
        conv_params   = list(model.conv.parameters())

        # 其他参数 = 除去 a_ode / b_ode / conv 的其余部分
        exclude = set(a_ode_params + conv_params + x_proj_params + p_proj_params)
        other_params = [p for p in model.parameters() if p not in exclude]

        # 2. 定义参数组
        param_groups = [
            {"params": a_ode_params,   "weight_decay": 1e-3,   "lr": 0.001,  "name": "A_ODE"},
            {"params": x_proj_params,  "weight_decay": 1e-3,   "lr": 0.001,  "name": "x_proj"},
            {"params": p_proj_params,  "weight_decay": 1e-3,   "lr": 0.001,  "name": "p_proj"},
            {"params": conv_params,    "weight_decay": 1e-3,   "lr": 0.001,  "name": "conv_params"},
            {"params": other_params,   "weight_decay": 1e-3,   "lr": 0.001,  "name": "Other"},
        ]

        # 3. 构建优化器
        optimizer = torch.optim.AdamW(param_groups)
        return optimizer

    def learn(self, e, x_train,x_val,x_test,args):
        self.epoch     = e
        self.net.epoch = e
        #-----------------------------------------------validation------------------------------------------------------#
        count = 0
        loss_actual_use = 0
        self.train_data = DataLoader(dataset = x_val, batch_size = args['batch_size'], shuffle = False, drop_last = True)
        self.net.eval()
        with torch.no_grad():
            for x_,u_ in self.train_data:
                _, loss_actual,_   = self.pred_forward(x_,u_,args)
                count             += 1
                loss_actual_use   += loss_actual
            self.loss_store = loss_actual_use/count/self.step_ 

        #-----------------------------------------------test------------------------------------------------------------#
        loss_actual_use = 0
        loss_actual_c_t = 0
        count = 0
        self.net.eval()
        loss_save       = []
        self.train_data = DataLoader(dataset = x_test, batch_size = args['batch_size'], shuffle = True, drop_last = True)
        with torch.no_grad():
            for x_,u_ in self.train_data:
                _,loss_actual_orgi_test,loss_actual  = self.pred_forward(x_,u_,args)
                count                               += 1
                loss_actual_use                     += loss_actual_orgi_test
                loss_actual_c_t                     += loss_actual
                loss_save.append(loss_actual_use/self.step_ )
                # print(loss_actual.item())

        self.loss_store_t = loss_actual_use/count/self.step_ 
        loss_actual_c_t   = loss_actual_c_t/count/self.step_ 
        print("Test  loss min {} loss max {}".format(np.min(loss_save),np.max(loss_save)))

        #-----------------------------------------------train-----------------------------------------------------------#
        # TODO: NOT SHUFFLE
        self.net.train()
        self.train_data = DataLoader(dataset = x_train, batch_size = args['batch_size'], shuffle = True, drop_last = True)
        count           = 0
        loss_buff       = 0
        loss_save       = []
        loss_save_v     = 0
        for x_,u_ in self.train_data:
            loss_, loss_actual_,v  = self.pred_forward(x_,u_,args)
            loss_buff             += loss_actual_/self.step_ 
            count                 += 1
            self.optimizer1.zero_grad()
            loss_.backward()
            # if count == 7:
            #     for name, param in self.net.named_parameters():
            #         if param.grad is not None:
            #             print(f"{name:30} | grad norm: {param.grad.norm().item():.4e}")
            # if e<100:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
            self.optimizer1.step()
            loss_save.append(loss_actual_/self.step_ )
            loss_save_v          +=(v/self.step_ )
        
        print("Train loss min {} loss max {}".format(np.min(loss_save),np.max(loss_save)))

        self.optimizer1_sch.step()
        loss_buff = loss_buff/count

        if loss_buff<self.loss_buff:
            self.net_para  = copy.deepcopy(self.net.state_dict())
            self.loss_buff = loss_buff

        print(
            "epoch {:03d}: "
            "loss_train {:.4f}  loss_val {:.4f}  test {:.4f}  "
            "convex_train {:.4f}  convex_test {:.4f}  "
            "min_loss {:.4f}  lr {}".format(
                e,
                float(loss_buff),                  # tensor -> python float
                float(self.loss_store),        # 取最新 val
                float(self.loss_store_t),      # 最新 test
                float(loss_save_v / count),
                float(loss_actual_c_t),
                float(self.loss_buff),
                self.optimizer1_sch.get_last_lr()   # list → scalar
        ))
    def test_(self,test,args):
        self.test_data = DataLoader(dataset = test, batch_size = 10, shuffle = True)
        with torch.no_grad():
            for x_,u_ in self.test_data:
                self.pred_forward_test(x_,u_,args)

    def pred_forward(self,x,u,args):
        pred_idx = args['pred_tracking']
        act_dim  = args['act_dim']

        x_pred = x[:, :, pred_idx]          # CPU or GPU slice, stride 连续
        u_act  = u[:, :, :act_dim]
        u_dist = u[:, :, act_dim:]

        sigma_p = torch.std(u_dist, dim=(0,1), keepdim=True)      # (1,1,P)

        # 2. 设定噪声系数
        alpha = 0.0      # 抖动强度 = 5% * σ_p

        # 3. 生成与 u_dist 同形状的噪声（GPU 上）
        noise        = torch.randn_like(u_dist) * (alpha * sigma_p)
        u_dist_noisy = u_dist + noise
        u_dist_noisy = u_dist_noisy[:,:,:]

        # 只在必要时搬到 GPU，且一次搬完
        if x_pred.device != self.device:
            x_pred  = x_pred.to(self.device, non_blocking=True, copy=False).contiguous()
            u_act   = u_act. to(self.device, non_blocking=True, copy=False).contiguous()
            u_dist  = u_dist_noisy.to(self.device, non_blocking=True, copy=False).contiguous()

        loss, loss_actual, loss_actual_c ,_, _ = self.net(x_pred, u_act, u_dist)
        return loss, loss_actual, loss_actual_c


    def pred_forward_test(self,x,u,test,args,e=0):
        x_pred_list = []
        x_real_list = []
        x_time_list = []

        self.shift_x = np.loadtxt(args['shift_x'],dtype=np.float32)
        self.scale_x = np.loadtxt(args['scale_x'],dtype=np.float32)

        self.shift_x = self.shift_x[args['pred_tracking']]
        self.scale_x = self.scale_x[args['pred_tracking']]

        print("done")
        self.loss_t = 0
        count = 0
        self.net.eval()
        if test:
            for i in range(args['old_horizon']*2,x.shape[1]-args['pred_horizon'],args['pred_horizon']-1):
                x_pred = x[:,i-args['old_horizon']:i+args['pred_horizon']]
                u_pred = u[:,i-args['old_horizon']:i+args['pred_horizon']-1]
                x_pred_list_buff,x_real_list_buff, everydim = \
                    self.pred_forward_test_buff(x_pred,u_pred,args)
                x_pred_list.append(x_pred_list_buff*self.scale_x+self.shift_x)
                x_real_list.append(x_real_list_buff*self.scale_x+self.shift_x)
                x_time_list.append(np.arange(i,i+args['pred_horizon']))
                count+=1
            plt.close()
            plt.bar(range(len(everydim)), everydim)
            plt.xlabel("Output dimension")
            plt.ylabel("Average MSE Loss")
            plt.title("Loss per output dimension")
            plt.savefig('data/mam_ODE_ODE_J/'+str(self.env)+'/'+str(self.time_training)+'/predictions_' + str(e) + '_every_dim.png')
            plt.close()
            print(self.loss_t/count/self.step_ )
            f, axs = plt.subplots(args['draw_state_num'], sharex=True, figsize=(15, 15))
            time_all = np.arange(x.shape[1])
            # to show the performance of modeling 
            for i in range(args['draw_state_num']):

                draw_ = args['draw_state'][i]
                draw_c= args['draw_state_c'][i]
                x_    = x[:, :,args['pred_tracking']]*self.scale_x+self.shift_x

                axs[i].plot(time_all, x_[:, :,draw_c].T, 'k')
                for j in range(len(x_time_list)):
                    # axs[i].plot(x_time_list[j], x_pred_list[j][0,:,draw_c], 'r')
                    axs[i].plot(x_time_list[j], x_pred_list[j][:,0,draw_c], 'r')
            
            plt.xlabel('Time Step')
            plt.savefig('data/mam_ODE_ODE_J/'+str(self.env)+'/'+str(self.time_training)+'/predictions_' + str(e) + '.png')
            plt.close()
            print("plot")

            if e == args['num_epochs']-1 or self.eval:
                # plt.savefig('data/predictions_' + str(e) + '.pdf')
                np.save('Prediction/mam_ODE_ODE_J/'+str(self.env)+'/'+str(self.time_training)+'/x_pred.npy',np.array(x_pred_list))
                np.save('Prediction/mam_ODE_ODE_J/'+str(self.env)+'/'+str(self.time_training)+'/x_time.npy',np.array(x_time_list))
                np.save('Prediction/mam_ODE_ODE_J/'+str(self.env)+'/'+str(self.time_training)+'/x_all_time.npy',np.array(time_all))
                np.save('Prediction/mam_ODE_ODE_J/'+str(self.env)+'/'+str(self.time_training)+'/x_.npy',np.array(x[:,:,args['pred_tracking']])*self.scale_x+self.shift_x)
                print('store!!!!')

            return x_pred_list,x_real_list
        ##----------------------------------------------##
        else:
            return self.pred_forward_test_buff(x,u,args)
    
    def pred_forward_test_buff(self,x,u,args):
        
        pred_horizon = args['old_horizon']
        # USE_DIM = [0,2,3,4,5,10,11,12]
        P_use        = u[:,:,args['act_dim']:]
        P_use        = P_use[:,:,:]
        loss,_,_,result, everydim  = self.net(x[:,:,args['pred_tracking']].to(self.device),u[:,:,:args['act_dim']].to(self.device),P_use.to(self.device),True)
        self.loss_t += loss
        # print(loss)

        return result,x[:,pred_horizon:,args['pred_tracking']],everydim



    def parameter_store(self,args):
        """
        TODO: store data from mamba block
        """
        torch.save(self.net_para,self.MODEL_SAVE)
        torch.save(self.optimizer1.state_dict(),self.OPTI1)

        print("store!!!")

    def parameter_restore(self,args):
        """
        restore data from mamba block
        """
        # if args['control']:
        #     self.device = torch.device('cpu')

        self.net = Mamba(args)
        self.net.load_state_dict(torch.load(self.MODEL_SAVE,map_location=self.device))    
        self.net.to(self.device)

        self.optimizer1 = optim.AdamW([{'params': self.net.parameters(),'lr':args['lr1'],'weight_decay':args['weight_decay']}])


        # self.net.eval()
        print("restore!")

# ---------------------------------------------------------------------------
#  Small utility blocks
# ---------------------------------------------------------------------------

class Householder(nn.Module):
    """Batch Householder reflection producing orthogonal matrices.

    Input  : x ∈ ℝ^{B×N}
    Output : H ∈ ℝ^{B×N×N} (orthogonal) such that Hx = ‑ sign(x₁)‖x‖ e₁.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.N = latent_dim
        I = torch.eye(self.N)
        self.register_buffer("_I", I.unsqueeze(0))  # (1,N,N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,N) → (B,N,N)
        B, N = x.shape
        assert N == self.N, f"Householder expects N={self.N}, got {N}"

        e1 = torch.zeros_like(x)
        e1[:, 0] = 1.0
        norm_x = x.norm(dim=1, keepdim=True)             # (B,1)
        sign   = torch.where(x[:, :1] >= 0, 1.0, -1.0)    # (B,1)
        v      = x + sign * norm_x * e1                  # (B,N)
        v      = v / v.norm(dim=1, keepdim=True).clamp_min(1e-7)

        vvT = v.unsqueeze(2) @ v.unsqueeze(1)            # (B,N,1)@(B,1,N)→(B,N,N)
        H   = self._I.expand(B, -1, -1) - 2.0 * vvT      # (B,N,N)
        return H


class AODEFunc(nn.Module):
    """d a / dt = f_a(t, a)  (a ∈ ℝ^N)"""
    def __init__(self, N: int, hidden_dim: int = 64, dim_B: int = 64, dim_C: int = 64, prune_hidden: int = 32, dim_Z: int = 64,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prune_hidden+prune_hidden, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, prune_hidden), nn.Tanh(),
        )
        self.lat2out    = nn.Sequential(nn.Linear(prune_hidden, 8), nn.SiLU(),  nn.Linear(8, N))
        self.lat2out_h  = nn.Sequential(nn.Linear(prune_hidden, 8), nn.SiLU(),  nn.Linear(8, N))
        self.lat2out_b  = nn.Sequential(nn.Linear(prune_hidden, 4), nn.SiLU(),  nn.Linear(4, dim_B))
        self.lat2out_z  = nn.Sequential(nn.Linear(N, N),            nn.SiLU(),  nn.Linear(N, dim_Z))
        self.lat2out_c  = nn.Sequential(nn.Linear(prune_hidden, 8), nn.SiLU(),  nn.Linear(8, dim_C))

    def forward(self, t: torch.Tensor, a: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, context], -1)
        return self.net(x)

class BODEFunc(nn.Module):
    """d a / dt = f_a(t, a)  (a ∈ ℝ^N)"""
    def __init__(self, N: int, hidden_dim: int = 64, dim_B: int = 64, dim_C: int = 64, prune_hidden: int = 32, dim_Z: int = 64,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(prune_hidden+prune_hidden, hidden_dim), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, prune_hidden), nn.Tanh(),
        )
        self.lat2out_Q  = nn.Sequential(nn.Linear(prune_hidden, 8), nn.SiLU(),  nn.Linear(8, N))

    def forward(self, t: torch.Tensor, a: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = torch.cat([a, context], -1)
        return self.net(x)
class _JointODEFunc(nn.Module):
    """Augmented dynamics for **a, b, z** used by `MambaJoint`.

    Parameters
    ----------
    model : parent MambaJoint (provides sub‑modules & hyper‑params)
    up_traj : (B,L+1,U+P) zero‑order‑held control/disturbance sequence (batch major)
    t_knots : (L+1,) time stamps in [0,1] aligned with up_traj
    """
    def __init__(self, model: 'Mamba', u_traj: torch.Tensor, p_traj: torch.Tensor, t_knots: torch.Tensor, context: torch.Tensor, control_: bool):
        super().__init__()
        self.model    = model
        self.u_traj   = u_traj            # (B,L+1,U+P)
        self.p_traj   = p_traj.unsqueeze(-1)            # (B,L+1,U+P)
        self.t_knots  = t_knots            # (L+1,)
        self.context  = context

        self.RK_count    = 1
        self.A_save      = []
        # self.A_save_buff = []
        self.B_save      = []
        self.eigen_v     = []
        # self.B_save_buff = []
        self.control     = control_
        self.count       = 1
        self.loss        = 0

    # ----- helper: zero‑order hold u/p(t) over [0,1] -----
    def _up(self, t: torch.Tensor) -> torch.Tensor:  # scalar t → (B,U+P)
        # bucketize wants 1‑D tensor; broadcast batch index later
        idx = torch.bucketize(t, self.t_knots,right=True) - 1   # ∈ [0,L]
        idx = idx.clamp_min(0).clamp_max(self.u_traj.shape[1] - 1)
        # print(t)
        # print(idx)
        # print(self.u_traj[:, idx])
        return self.u_traj[:, idx]                   # (B,U+P)

    # ----------------------------------------------------
    def forward(self, t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:  # (B,*)
        M      = self.model
        Bsz    = s.shape[0]
        N      = M.prune_hidden
        a      = s[:, :N]                    # (B,N)9
        b      = s[:, N:2*N] 
        z      = s[:, 2*N:]                  # (B,N)
        da     = M.a_ode(t, a, self.context)
        db     = M.b_ode(t, b, self.context)
        d      = M.a_ode.lat2out(a).unsqueeze(-1)          # (B,N,1)
        b_flat = M.a_ode.lat2out_b(a)        # (B, N(U+P))
        B_mat  = b_flat.view(Bsz, M.N, M.U+M.P)
        up_t   = self._up(t).unsqueeze(-1)  
        u_all  = torch.cat([up_t,self.p_traj], dim=1)                    # (B,U+P,1)
        dz     = (d * z.unsqueeze(-1) + B_mat @ u_all).squeeze(-1)#+z/10  # (B,N)
        
        
        
        if self.control:
            self.A1    = d.squeeze()
            self.B1    = B_mat.squeeze()
            self.A_save.append(torch.diag(self.A1*M.span_A).detach().cpu().numpy()+np.eye(M.N))
            self.B_save.append((self.B1*M.span_A).detach().cpu().numpy())
            self.eigen_v.append(d.detach().cpu().numpy().squeeze())
            self.count    += 1
        
        return torch.cat([da, db, dz], dim=-1)

class HistConv(nn.Module):
    def __init__(self, D, kernel=3):
        super().__init__()
        # 普通 conv：dilation=1 ⇒ padding = 1
        self.conv1 = nn.Conv1d(D, D, kernel_size=kernel,
                               padding=kernel//2,      # =1
                               groups=D)
        # dilated conv：dilation=2 ⇒ padding = 2
        self.conv2 = nn.Conv1d(D, D, kernel_size=kernel,
                               padding=((kernel-1)*2)//2,  # =2
                               dilation=2,
                               groups=D)
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x):          # x: (B,D,L)
        y = F.silu(self.conv1(x))
        y = F.silu(self.conv2(y))
        return x + 0.5 * y         # 半残差，抑制梯度爆炸
# ---------------------------------------------------------------------------
#  Main module with joint ODE
# ---------------------------------------------------------------------------
class Mamba(nn.Module):
    """Mamba‑inspired LTV learner – *joint ODE* version.

    Notes
    -----
    * Input shapes
        x : (B,T,S)   – measured states (S ≤ N)
        u : (B,T,U)   – control
        p : (B,T,P)   – disturbances
    * Requires T ≥ O+L.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        # ------------- hyper‑params -------------
        self.L      = cfg["pred_horizon"]
        self.O      = cfg["old_horizon"]
        self.N      = cfg["latent_dim"]
        self.S      = cfg["state_use"]
        self.S_r    = cfg["state_use"]-1
        self.U      = cfg["act_dim"]
        self.P      = cfg["disturbance"]
        self.d_conv = cfg.get("d_conv", 3)
        hdim        = 64
        hdim_p      = self.P
        hdim_o      = 64
        self.ctrl   = cfg["control"]

        self.loss_54 = 0
        self.loss_19 = 0
        self.loss_r  = 0


        self.log_sigma1 = nn.Parameter(torch.zeros(()))
        self.log_sigma2 = nn.Parameter(torch.zeros(()))
        # ------------- learnable projections -------------
        # self.C       = nn.Parameter(0.01 * torch.randn(self.S, self.N))
        dim_C             = self.S*self.N
        # self.x_proj_o     = nn.Sequential(nn.Linear(self.S, hdim_o), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hdim_o, hdim_o), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hdim_o, self.N))
        self.x_proj       = nn.Sequential(nn.Linear(self.S, hdim), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hdim, hdim), nn.SiLU(), nn.Dropout(0.1), nn.Linear(hdim, hdim), nn.SiLU(), \
                                          nn.Dropout(0.1), nn.Linear(hdim, hdim), nn.SiLU(), \
                                          nn.Dropout(0.1), nn.Linear(hdim, self.N))
        self.x_proj_      = nn.Sequential(nn.Linear(self.N, hdim), nn.SiLU(), nn.Dropout(0.0), nn.Linear(hdim, self.N))
        self.p_proj       = nn.Sequential(nn.Linear(self.P, hdim_p), nn.SiLU(), nn.Dropout(0.0), nn.Linear(hdim_p, self.P)) ###TANH
        self.x_proj_u     = nn.Sequential(nn.Linear(self.N, hdim), nn.SiLU(), nn.Dropout(0.0), nn.Linear(hdim, self.N))
        self.prune_hidden = 8

        # ------------- conv over history -------------
        # self.conv = nn.Conv1d(self.N + self.U + self.P, self.N + self.U + self.P,
        #                        kernel_size=self.d_conv, padding=self.d_conv - 1,
        #                        groups=self.N + self.U + self.P)

        self.conv = nn.Conv1d(
            in_channels = self.N + self.U + self.P,
            out_channels = self.N + self.U + self.P,
            kernel_size = self.d_conv,      # 3
            padding     = (self.d_conv - 1) * 2,   # dilation 会撑大感受野，要调 padding
            dilation    = 2,
            groups      = self.N + self.U + self.P
        )

        self.conv_zero = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)


        # D = self.N + self.U + self.P
        # self.hist_conv = HistConv(D, kernel=3)
        

        nn.init.kaiming_uniform_(self.conv.weight, mode="fan_in", nonlinearity="relu")

        # ------------- ODE nets & helpers -------------
        self.span_A      = nn.Parameter(torch.tensor(0.1))
        # self.span_A      = torch.tensor(0.1)
        self.compress1    = nn.Sequential(nn.Linear((self.N + self.U + self.P), self.prune_hidden), nn.Dropout(0.1))
        self.compress2    = nn.Sequential(nn.Linear((self.N + self.U + self.P), self.prune_hidden), nn.Dropout(0.1))
        dim_B            = self.N * (self.U + self.P)
        dim_C            = self.S * self.N
        dim_Z            = self.S
        self.a_ode       = AODEFunc(self.N, hdim, dim_B, dim_C, self.prune_hidden,dim_Z)
        self.b_ode       = BODEFunc(self.N, hdim, dim_B, dim_C, self.prune_hidden,dim_Z)
        self.householder = Householder(self.N)
        self.delta_raw   = nn.Parameter(torch.zeros(self.L))
    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _make_times(self) -> torch.Tensor:
        """Return monotonic t₁…t_L ∈ (0,1] (does *not* include 0)."""
        delta = F.softplus(self.delta_raw) + 1e-4
        tspan = torch.cumsum(delta, 0)
        return tspan / tspan[-1]
    
    def init_for_controller(self, x: torch.Tensor, u: torch.Tensor, p: torch.Tensor, Draw, is_controller = False):
        B, T, _ = x.shape
        # assert T >= self.O + self.L, "Trajectory too short for given horizons"
        device = x.device

        # 1) history embedding → initial a0 & b0
        x_old      = self.x_proj(x[:, : self.O - 1])           # (B,O‑1,N)
        u_old      = u[:, : self.O - 1]
        p_old      = self.p_proj(p[:, : self.O - 1])           # (B,O‑1,P)
        hist       = torch.cat([x_old, u_old, p_old], dim=-1)  # (B,O‑1,D)
        hist       = rearrange(hist, "B L D -> B D L")        # (B,D,O‑1)
        # hist     = self.hist_conv(hist)[:, :, -self.O:]
        hist       = F.silu(self.conv(hist))[:, :, -self.O:]   # (B,D,O)
        hist       = rearrange(hist, "B D L -> B L D")
        hist_f     = self.conv_zero(hist)
        # hist_f     = hist.reshape(B, -1)
        vector_F   = self.compress1(hist_f).squeeze(1)
        a0         = vector_F
        b0         = self.compress2(hist_f).squeeze(1)
        z0         = self.x_proj(x[:, self.O - 1])                  # (B,N)
        self.H     = self.householder(self.x_proj_(z0))
        u_pred     = u[:, self.O - 1 : self.O + self.L]         # (B,L+1,U) – includes u₀
        p_pred     = self.p_proj(p[:, self.O - 1])
        span_A     = torch.clamp(self.span_A, min=1e-8,max=7.0)
        tspan      = torch.arange(1, self.L+1).float().to(x.device)*span_A
        tspan      = torch.cat([torch.zeros(1, device=device), tspan], dim=0)
        s0         = torch.cat([a0, b0, z0], dim=-1)        # (B, N+NU+N)
        joint_func = _JointODEFunc(self, u_pred, p_pred, tspan, vector_F, self.ctrl)
        sol        = odeint(joint_func, s0, tspan, method="euler", rtol=1e-3, atol=1e-4)  # (L+1,B,D)

        return z0, sol, joint_func

    # ---------------------------------------------------------------------
    # forward
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, u: torch.Tensor, p: torch.Tensor, Draw=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roll‑out prediction for horizon L and compute MSE loss.

        Returns
        -------
        loss  : scalar tensor
        y_hat : (B,L,S) predictions over horizon
        """
        # 5) integrate joint ODE
        if self.ctrl:
            z0, sol, ODE    = self.init_for_controller(x,u,p,Draw)
            if self.eval:
                self.A_save     = ODE.A_save
                self.B_save     = ODE.B_save
                self.eigen_v    = ODE.eigen_v
        else:
            z0, sol, _    = self.init_for_controller(x,u,p,Draw)

        z_traj        = sol[:, :, -self.N:]                      # (L,B,N)
        c_traj        = sol[1:, :, :self.prune_hidden]
        C             = self.a_ode.lat2out_c(c_traj).permute(1,0,2).view(c_traj.shape[1], c_traj.shape[0], self.S,self.N)
        # Q             = self.b_ode.lat2out_Q(c_traj).permute(1,0,2).view(c_traj.shape[1], c_traj.shape[0], self.N)
        # Q             = F.relu(Q)

        if self.ctrl:
            # pass
            ys            = []
            loss_per_dim  = []
            loss_act      = torch.tensor(0.0, device=z0.device)
            loss_act_u    = torch.tensor(0.0, device=z0.device)
            loss_act_c    = torch.tensor(0.0, device=z0.device)

            for k in range(self.L):
                y_hat_c   = torch.einsum("bij,bj->bi", C[:,k,:], z_traj[k+1,:,:])
                Added_z   = self.a_ode.lat2out_z(z_traj[k])
                # jjj       = Q[:,k,:]*z_traj[k+1,:,:]
                # Convex    = torch.einsum("bij,bjk->bik", jjj.unsqueeze(1), z_traj[k+1,:,:].unsqueeze(2))
                # y_hat_c[:,-1] += Convex[:,0,0]
                y_hat     = y_hat_c+Added_z
                # Added_z   = self.a_ode.lat2out_z(z_traj[k])
                if Draw:
                    loss_per_sample = F.mse_loss(y_hat, x[:, self.O+k], reduction='none')
                    loss_per_dim.append(loss_per_sample.mean(dim=0))
                    loss_act   += loss_per_sample.mean()
                    self.loss_54    += loss_per_sample.mean(dim=0)[54]
                    self.loss_19    += loss_per_sample.mean(dim=0)[19]
                    self.loss_r     += loss_per_sample.mean(dim=0)[130]
                else:
                    loss_act_u +=  F.mse_loss(y_hat,   x[:, self.O+k]) 
                    loss_act   +=  F.mse_loss(y_hat,   x[:, self.O+k]) + F.mse_loss(Added_z, torch.zeros_like(Added_z))*0.05
                    loss_act_c +=  F.mse_loss(y_hat_c, x[:, self.O+k])

                # reward   += Pred_use[0,-1]
                # reward_i += Pred_i[0,-1]
                z0, sol, ODE    = self.init_for_controller(x,u,p,Draw)
                ys.append(y_hat.detach().cpu().numpy())
            
            print("54 {} 19 {} reward {}".format(self.loss_54, self.loss_19, self.loss_r))
            # print("reward {} reward_actual {}".format(reward, reward_i))

            if Draw:
                loss_per_dim      = torch.stack(loss_per_dim, dim=0)  # shape: (L, S)
                loss_per_dim      = loss_per_dim.mean(dim=0).detach().cpu().numpy()
        else:
            ys            = []
            loss_per_dim  = []
            loss_act      = torch.tensor(0.0, device=z0.device)
            loss_act_u    = torch.tensor(0.0, device=z0.device)
            loss_act_c    = torch.tensor(0.0, device=z0.device)
            for k in range(self.L):
                y_hat_c   = torch.einsum("bij,bj->bi", C[:,k,:], z_traj[k+1,:,:])
                Added_z   = self.a_ode.lat2out_z(z_traj[k])
                # jjj       = Q[:,k,:]*z_traj[k+1,:,:]
                # Convex    = torch.einsum("bij,bjk->bik", jjj.unsqueeze(1), z_traj[k+1,:,:].unsqueeze(2))
                # y_hat_c[:,-1] += Convex[:,0,0]
                y_hat     = y_hat_c+Added_z
                if Draw:
                    loss_per_sample = F.mse_loss(y_hat, x[:, self.O+k], reduction='none')
                    loss_per_dim.append(loss_per_sample.mean(dim=0))
                    loss_act   += loss_per_sample.mean()
                else:
                    loss_act_u +=  F.mse_loss(y_hat,   x[:, self.O+k]) 
                    loss_act   +=  F.mse_loss(y_hat,   x[:, self.O+k]) + F.mse_loss(Added_z, torch.zeros_like(Added_z))*0.05
                    loss_act_c +=  F.mse_loss(y_hat_c, x[:, self.O+k])
                ys.append(y_hat.detach().cpu().numpy())

            if Draw:
                loss_per_dim      = torch.stack(loss_per_dim, dim=0)  # shape: (L, S)
                loss_per_dim      = loss_per_dim.mean(dim=0).detach().cpu().numpy()
        return loss_act, loss_act_u.item(), loss_act_c.item() ,np.array(ys), loss_per_dim


def _cfg_get(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _state_dim_from_cfg(cfg):
    obs_shape = _cfg_get(cfg, "obs_shape")
    obs_key = _cfg_get(cfg, "obs", "state")
    if isinstance(obs_shape, dict) and obs_key in obs_shape:
        return obs_shape[obs_key][0]
    return _cfg_get(cfg, "state_use", _cfg_get(cfg, "state_dim"))


class MamODEWorldModel(nn.Module):
    """MamODE dynamics exposed with a TD-MPC style world-model interface.

    This module keeps the multi-task padding convention: observations and
    actions are represented at the maximum dimension, while task-specific
    action dimensions are masked before dynamics/reward prediction.

    Core interface:
        encode(obs, task) -> z
        next(z, action, task, disturbance=None) -> z_next
        decode(z, task) -> padded state prediction
        reward(z, action, task, disturbance=None) -> reward prediction
        rollout(obs, actions, task, disturbances=None) -> state predictions
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.obs_dim = int(_state_dim_from_cfg(cfg))
        self.action_dim = int(_cfg_get(cfg, "action_dim", _cfg_get(cfg, "act_dim")))
        self.disturbance_dim = int(_cfg_get(cfg, "disturbance", 0))
        self.latent_dim = int(_cfg_get(cfg, "latent_dim"))
        self.task_dim = int(_cfg_get(cfg, "task_dim", 0))
        self.multitask = bool(_cfg_get(cfg, "multitask", False))
        self.hidden_dim = int(_cfg_get(cfg, "mam_hidden_dim", 128))
        self.prune_hidden = int(_cfg_get(cfg, "mam_prune_hidden", 8))
        self.ode_substeps = int(_cfg_get(cfg, "mam_ode_substeps", 1))

        if self.multitask:
            tasks = _cfg_get(cfg, "tasks")
            self._task_emb = nn.Embedding(len(tasks), self.task_dim, max_norm=1)
            self.register_buffer("_action_masks", torch.zeros(len(tasks), self.action_dim))
            self.register_buffer("_obs_masks", torch.zeros(len(tasks), self.obs_dim))
            action_dims = _cfg_get(cfg, "action_dims")
            obs_dims = _cfg_get(cfg, "obs_shapes", _cfg_get(cfg, "obs_dims", None))
            for i in range(len(tasks)):
                self._action_masks[i, :action_dims[i]] = 1.
                obs_dim = obs_dims[i] if obs_dims is not None else self.obs_dim
                self._obs_masks[i, :obs_dim] = 1.
        else:
            self.task_dim = 0

        enc_in = self.obs_dim + self.task_dim
        dyn_in = self.latent_dim + self.task_dim
        control_dim = self.action_dim + self.disturbance_dim

        self._encoder = nn.Sequential(
            nn.Linear(enc_in, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim),
        )
        self._decoder = nn.Sequential(
            nn.Linear(dyn_in, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.obs_dim),
        )
        self._context = nn.Sequential(
            nn.Linear(dyn_in, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.prune_hidden), nn.Tanh(),
        )
        self._ab_init = nn.Sequential(
            nn.Linear(dyn_in, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, 2 * self.prune_hidden),
        )
        self.a_ode = AODEFunc(
            self.latent_dim,
            hidden_dim=self.hidden_dim,
            dim_B=self.latent_dim * control_dim,
            prune_hidden=self.prune_hidden,
            dim_Z=self.obs_dim,
        )
        self.b_ode = BODEFunc(
            self.latent_dim,
            hidden_dim=self.hidden_dim,
            dim_B=self.latent_dim * control_dim,
            prune_hidden=self.prune_hidden,
            dim_Z=self.obs_dim,
        )
        self._reward = nn.Sequential(
            nn.Linear(dyn_in + self.action_dim + self.disturbance_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self._termination = nn.Sequential(
            nn.Linear(dyn_in, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.log_dt = nn.Parameter(torch.zeros(()))
        self.apply(weights_init)

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def task_emb(self, x, task):
        if not self.multitask:
            return x
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def _mask_action(self, action, task):
        if self.multitask:
            return action * self._action_masks[task]
        return action

    def _mask_obs(self, obs, task):
        if self.multitask:
            return obs * self._obs_masks[task]
        return obs

    def _disturbance(self, z, disturbance):
        if self.disturbance_dim == 0:
            return z.new_zeros(*z.shape[:-1], 0)
        if disturbance is None:
            return z.new_zeros(*z.shape[:-1], self.disturbance_dim)
        return disturbance

    def encode(self, obs, task=None):
        if self.multitask:
            obs = self._mask_obs(obs, task)
        return self._encoder(self.task_emb(obs, task))

    def decode(self, z, task=None):
        obs = self._decoder(self.task_emb(z, task))
        return self._mask_obs(obs, task)

    def next(self, z, action, task=None, disturbance=None):
        action = self._mask_action(action, task)
        disturbance = self._disturbance(z, disturbance)
        control = torch.cat([action, disturbance], dim=-1)
        z_task = self.task_emb(z, task)
        context = self._context(z_task)
        a, b = self._ab_init(z_task).chunk(2, dim=-1)
        dt = F.softplus(self.log_dt) / max(self.ode_substeps, 1)

        for _ in range(max(self.ode_substeps, 1)):
            da = self.a_ode(dt, a, context)
            db = self.b_ode(dt, b, context)
            diag = self.a_ode.lat2out(a)
            b_flat = self.a_ode.lat2out_b(a)
            b_mat = b_flat.view(z.shape[0], self.latent_dim, control.shape[-1])
            dz = diag * z + torch.einsum("bij,bj->bi", b_mat, control)
            a = a + dt * da
            b = b + dt * db
            z = z + dt * dz
        return z

    def reward(self, z, action, task=None, disturbance=None):
        action = self._mask_action(action, task)
        disturbance = self._disturbance(z, disturbance)
        x = torch.cat([self.task_emb(z, task), action, disturbance], dim=-1)
        return self._reward(x)

    def termination(self, z, task=None, unnormalized=False):
        logits = self._termination(self.task_emb(z, task))
        return logits if unnormalized else torch.sigmoid(logits)

    def rollout(self, obs, actions, task=None, disturbances=None):
        z = self.encode(obs, task)
        preds = []
        for t, action in enumerate(actions.unbind(0)):
            disturbance = None
            if disturbances is not None:
                disturbance = disturbances[t]
            z = self.next(z, action, task, disturbance)
            preds.append(self.decode(z, task))
        return torch.stack(preds)

    def forward(self, obs, actions, task=None, disturbances=None, target_obs=None):
        preds = self.rollout(obs, actions, task, disturbances)
        if target_obs is None:
            return preds
        if self.multitask:
            target_obs = self._mask_obs(target_obs, task)
        return F.mse_loss(preds, target_obs), preds


# ---------------------------------------------------------------------------
#  End of file
# ---------------------------------------------------------------------------
