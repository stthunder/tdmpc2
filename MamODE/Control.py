import numpy as np
from cvxpy import *
import mosek
from scipy.linalg import solve_discrete_are
from scipy.linalg import solve_discrete_lyapunov
from scipy.linalg import block_diag
import time
import torch
from einops import rearrange, repeat, einsum

import casadi as ca
class Upper_MPC_mamba_ODE_evolution(object):
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['state_dim']
        self.args = args
        self.model = model
        self._build_matrices(args)
        self.init = False
        self.next = False

        self.R = R
        self.Q = Q
        self.P = P

        self.last_use    = None
        self.data_save_x = []
        self.data_save_u = []
        self.data_save_p = []

        self.pred_tracking = args['pred_tracking']

        if args['disturbance']>0:
            self.data = np.loadtxt('envs/Inf_dry_constQ.txt') #TODO: CHECK
            self.data = np.concatenate([self.data[:,[14]],self.data[:,1:14]],axis=1) 
            self.data = (self.data-self.model.shift_u[self.a_dim:])/self.model.scale_u[self.a_dim:]

    def _get_set_point_u(self):
        # self.ref.value = self.reference
        pass



    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                if args['disturbance']>0:
                    self.len_u        = np.size(self.shift_u)-args['disturbance']
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                else:
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None


    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim = args['latent_dim']
        self.u_dim = args['act_dim']
        self.L     = args['pred_horizon']
        self.p_dim = args['disturbance']
        self.x_use = args['state_use']     

        self.A_holder = Parameter((self.x_dim*self.L, self.x_dim))# diagnoal matrix
        self.B_holder = Parameter((self.x_dim*self.L, self.u_dim+self.p_dim))  
        self.C_holder = Parameter((self.x_use*self.L, self.x_dim)) 
        # self.Q_holder = Parameter((self.L, self.x_dim), nonneg=True)

    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        
        # self.shift = self.shift[self.pred_tracking]
        # self.scale = self.scale[self.pred_tracking]

        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)

        self.create_prob(self.args)
        self._get_set_point_u()

    def create_prob(self, args):
        self.u       = Variable((self.a_dim, self.control_horizon))
        self.x       = Variable((self.x_dim, self.control_horizon+1))
        self.x_a     = Variable((self.x_use, self.control_horizon))
        self.x_non_c = Parameter((self.x_use, self.control_horizon))

        if self.p_dim>0:
            self.p = Parameter((self.p_dim,))

        self.x_init = Parameter(self.x_dim)           
        objective   = 0.
        constraints =  [self.x[:, 0] == self.x_init]

        for k in range(self.control_horizon):
            if self.p_dim>0:
                mix = hstack([self.u[:,k],self.p])
            else:
                mix = self.u[:,k]
            constraints      += [self.x[:, k + 1] == self.A_holder[k*self.x_dim:(k+1)*self.x_dim,:] @ self.x[:, k]  + self.B_holder[k*self.x_dim:(k+1)*self.x_dim,:] @ mix]
            constraints      += [self.x_a[:,k]    == self.C_holder[k*self.x_use:(k+1)*self.x_use,:] @ self.x[:,k+1] + self.x_non_c[:,k]]
            # objective        += self.x_a[-1, k]+sum(multiply(self.Q_holder[k,:], square(self.x[:, k + 1])))
            objective        += quad_form(self.x_a[args['state_tracking_c'],k]-self.reference, self.Q)

            if k <= self.control_horizon-1 and k>0:
                objective += quad_form(self.u[:,k]-self.u[:,k-1],self.R)
            #     objective += quad_form(self.x_a[args['state_tracking_c'],k]-self.x_a[args['state_tracking_c'],k-1],self.R)

            if args['apply_action_constraints']:
                constraints += [self.a_bound_low <= self.u[:, k], self.u[:, k] <= self.a_bound_high]
        
        k = k+1
        self.prob = Problem(Minimize(objective), constraints)
        
    def vector_matrix(vector):
        return vector.reshape([vector.shape[0],-1])
    

    def choose_action(self, x_0, reference, *args):
        """
        reference is the time instant
        """
        self.model.net.eval()
        x_0k = (x_0-self.shift)/self.scale
        x_0  = x_0k[self.pred_tracking]

        self.data_save_x.append(x_0)
        self.data_save_p.append(self.data[reference]) 
        self.data_save_u.append(self.data_save_u[-1])

        add_x        = torch.from_numpy(np.array(self.data_save_x).squeeze()).to(torch.float32)
        add_u        = torch.from_numpy(np.array(self.data_save_u).squeeze()).to(torch.float32)
        add_p        = torch.from_numpy(np.array(self.data_save_p).squeeze()).to(torch.float32)

        phi_0, sol, H= self.model.net.init_for_controller(add_x.unsqueeze(0).to(self.model.device), 
                                                          add_u.unsqueeze(0).to(self.model.device),add_p.unsqueeze(0).to(self.model.device), False)
        h_in         = sol[1:,0,:self.model.net.prune_hidden]
        C            = self.model.net.a_ode.lat2out_c(h_in).view(self.L, self.x_use, self.x_dim).detach().cpu().numpy()

        self.A_holder.value = np.array(H.A_save).reshape(self.L*self.x_dim, self.x_dim)
        self.B_holder.value = np.array(H.B_save).reshape(self.L*self.x_dim, self.u_dim+self.p_dim)
        self.C_holder.value = C.reshape(self.L*self.x_use, self.x_dim)


        self.last_use = np.zeros([self.control_horizon, self.x_use])
        self.x_non_c.value  = self.last_use.T #TODO: FIRST SLOW OR OTHER METHOD
        self.x_init.value   = phi_0[0,:].detach().cpu().numpy()
        if self.p_dim > 0:
            self.p.value   = self.model.net.p_proj(torch.from_numpy(self.data[reference]).to(torch.float32).to(self.model.device)).detach().cpu().numpy()

        # self.x_non_c.value  = self.last_use.T #TODO: FIRST SLOW OR OTHER METHOD
        self.prob.solve(solver=MOSEK, 
                        warm_start=True, 
                        verbose=False,
                        mosek_params={
                            mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                            mosek.iparam.intpnt_max_iterations: 400,
                            mosek.dparam.intpnt_tol_rel_gap: 1e-4
                        })
        u    = self.u[:, 0].value
        last = torch.from_numpy(self.x[:,1:].value).to(torch.float32).to(self.model.device)

        self.last_use = self.model.net.a_ode.lat2out_z(last.T).detach().cpu().numpy()
        self.x_non_c.value  = self.last_use.T #TODO: FIRST SLOW OR OTHER METHOD
        self.prob.solve(solver=MOSEK, 
                        warm_start=True, 
                        verbose=False,
                        mosek_params={
                            mosek.iparam.intpnt_solve_form: mosek.solveform.dual,
                            mosek.iparam.intpnt_max_iterations: 400,
                            mosek.dparam.intpnt_tol_rel_gap: 1e-4
                        })
        u    = self.u[:, 0].value



        # self.last_use = self.model.net.a_ode.lat2out_z(last.T).detach().cpu().numpy()

        # x_historical = torch.from_numpy(np.array(self.data_save_x).squeeze()).to(torch.float32).unsqueeze(0)
        # u_historical = torch.from_numpy(np.concat([np.array(self.data_save_u)[:-1,:].squeeze(), self.u.value.T])).to(torch.float32).unsqueeze(0)
        # p_historical = torch.from_numpy(np.array(self.data_save_p).squeeze()).to(torch.float32).unsqueeze(0)
        
        # np.save('A.npy',np.array(H.A_save))
        # np.save('B.npy',np.array(H.B_save))
        # np.save('C.npy',self.C_holder.value)

        # np.save('C_.npy',C)
        # np.save('ok.npy',self.x.value)
        # np.save('add_z.npy',self.x_non_c.value)
        # np.save('ou.npy',self.u.value)
        # np.save('op.npy',self.p.value)
        # np.save('okk.npy',self.x_a.value)
        # self.model.net(x_historical.to(self.model.device),u_historical.to(self.model.device),p_historical.to(self.model.device))

        # np.save('A.npy',np.array(H.A_save))



        if self.p_dim > 0:
            self.data_save_p.pop(0)
    
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            if self.p_dim>0:
                su = u
                self.u_save = u
                u = u * self.scale_u[:self.len_u] + self.shift_u[:self.len_u]
            else:
                su = u
                self.u_save = u
                u = u * self.scale_u + self.shift_u

        else:
            print("Error: Cannot solve mpc..")
            su = u
            u = u * self.scale_u + self.shift_u
        # print(u)
        #--------------------store the old data------------------------#

        self.data_save_u[-1] = su

        self.data_save_x.pop(0)
        self.data_save_u.pop(0)

        print(len(self.data_save_x))
        print(len(self.data_save_u))
        print(len(self.data_save_p))

        return u
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success
    
    def reset(self):
        pass

class Upper_MPC_mamba_ODE(object):
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['state_dim']
        self.args = args
        self.model = model
        self._build_matrices(args)
        self.init = False
        self.next = False

        self.R = R
        self.Q = Q
        self.P = P


        self.data_save_x = []
        self.data_save_u = []
        self.data_save_p = []

        self.pred_tracking = args['pred_tracking']

        if args['disturbance']>0:
            self.data = np.loadtxt('envs/Inf_dry_constQ.txt')
            self.data = np.concatenate([self.data[:,[14]],self.data[:,1:14]],axis=1) 
            self.data = (self.data-self.model.shift_u[self.a_dim:])/self.model.scale_u[self.a_dim:]

    def _get_set_point_u(self):
        # self.ref.value = self.reference
        pass



    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                if args['disturbance']>0:
                    self.len_u        = np.size(self.shift_u)-args['disturbance']
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                else:
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None


    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim = args['latent_dim']
        self.u_dim = args['act_dim']
        self.L     = args['pred_horizon']
        self.p_dim = args['disturbance']
        self.x_use = args['state_use']

        # self.Q = np.diag([0.5, 0.5, 0.5, 0.5 ,0.5, 0.5, 0.5, 0.5, 0.5])
        # self.Q = np.diag([0.75, 0.5, 0.5, 0.75 ,0.5, 0.75, 0.75, 0.5, 2.5])
        # self.R = np.diag([0.1,0.1,0.1])
        # self.Q = np.diag([0.5, 0.002, 1.7, 1.1 ,0.005, 1.7, 1.1, 0.05, 1.5])

        # self.A_holder = Parameter((self.L*self.x_dim, self.x_dim))              
        self.A_holder = Parameter((self.x_dim*self.L, self.x_dim))# diagnoal matrix
        self.B_holder = Parameter((self.x_dim*self.L, self.u_dim+self.p_dim))  
        self.C_holder = Parameter((self.x_dim, self.x_use)) 

        # try to adjust Q P R via delta
        # self.delta    = Parameter((self.L,self.x_dim)) 

        # self.u_s_holder = Parameter((self.a_dim))



    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)

        self.create_prob(self.args)
        self._get_set_point_u()

    
    
    def create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.x_dim, self.control_horizon+1))
        if self.p_dim>0:
            self.p = Parameter((self.p_dim, self.control_horizon))

        self.x_init = Parameter(self.x_dim)           
        objective = 0.
        constraints =  [self.x[:, 0] == self.x_init]

        for k in range(self.control_horizon-1):
            # k_u = k if k <= self.control_horizon-1 else self.control_horizon-1
            if self.p_dim>0:
                mix = hstack([self.u[:,k],self.p[:,k]])
            else:
                mix = self.u[:,k]
            # constraints += [self.x[:, k + 1] == multiply(self.A_holder[k,:],self.x[:, k]) + self.B_holder[k*self.x_dim:(k+1)*self.x_dim,:] @ mix]
            constraints += [self.x[:, k + 1] == self.A_holder[k*self.x_dim:(k+1)*self.x_dim,:] @ self.x[:, k] + self.B_holder[k*self.x_dim:(k+1)*self.x_dim,:] @ mix]
            j_ = self.C_holder.T @ self.x[:,k]
            objective += quad_form(j_[args['state_tracking_c']]-self.reference, self.Q)

            if k <= self.control_horizon-1 and k>0:
                objective += quad_form(self.u[:,k]-self.u[:,k-1],self.R)

            if args['apply_action_constraints']:
                constraints += [self.a_bound_low <= self.u[:, k], self.u[:, k] <= self.a_bound_high]
        
        k = k+1
        j_ = self.C_holder.T @ self.x[:,k]
        objective += quad_form(j_[args['state_tracking_c']]-self.reference, self.P)
        self.prob = Problem(Minimize(objective), constraints)
        
    def vector_matrix(vector):
        return vector.reshape([vector.shape[0],-1])
    

    def choose_action(self, x_0, reference, *args):
        """
        reference is the time instant
        """

        x_0 = (x_0-self.shift)/self.scale
        x_0 = x_0[self.pred_tracking]

        if not self.init:
            self.init = True
            self.start_x = x_0
            
            for i in range(self.L):
                self.data_save_x.append(self.start_x)
                if self.p_dim>0:
                    len_u = np.size(self.shift_u)-self.p_dim
                    self.data_save_u.append((np.zeros([self.u_dim])-self.shift_u[:len_u])/self.scale_u[:len_u])
                    self.data_save_p.append(self.data[reference])
                else:
                    self.data_save_u.append((np.zeros([self.u_dim])-self.shift_u)/self.scale_u)


        # phi_0 = self.model.net.project_x(torch.from_numpy(x_0).to(torch.float32)) # project_x 
        
        add_x        = torch.from_numpy(np.array(self.data_save_x).squeeze()).to(torch.float32)
        add_u        = torch.from_numpy(np.array(self.data_save_u).squeeze()).to(torch.float32)
        add_p        = torch.from_numpy(np.array(self.data_save_p).squeeze()).to(torch.float32)
        p_           = self.data[reference:reference+self.pred_horizon,:]
        p_           = torch.from_numpy(p_).to(torch.float32).to(self.model.device)
        p_           = self.model.net.p_proj(p_).detach().cpu().numpy().T
        self.p.value = p_[:,:self.control_horizon]

    
        phi_0, sol, H= self.model.net.init_for_controller(add_x.unsqueeze(0).to(self.model.device), add_u.unsqueeze(0).to(self.model.device), add_p.unsqueeze(0).to(self.model.device),False)
        h_step       = self.model.net.span_A.item()
        
        A_in         = sol[:20,0,:self.x_dim].detach().cpu().numpy()
        B_in         = sol[:20,0,self.x_dim:-self.x_dim].detach().cpu().numpy()
        H            = H[0,:,:].detach().cpu().numpy()
        C            = self.model.net.C.detach().cpu().numpy()
        A            = (A_in[:, :, None] * H) @ H.T*h_step
        B            = B_in*h_step

        self.A_holder.value = A.squeeze().reshape(-1,self.x_dim)
        self.B_holder.value = B.squeeze().reshape(-1,self.u_dim+self.p_dim)
        self.C_holder.value = C.squeeze().reshape(-1,self.x_use)
        
        self.x_init.value = phi_0[0,:].detach().cpu().numpy()

        self.prob.solve(solver=MOSEK, warm_start=True, mosek_params={mosek.iparam.intpnt_solve_form:mosek.solveform.dual})
        u = self.u[:, 0].value
        print(self.prob.objective.value)
    
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            if self.p_dim>0:
                su = u
                self.u_save = u
                u = u * self.scale_u[:self.len_u] + self.shift_u[:self.len_u]
            else:
                su = u
                self.u_save = u
                u = u * self.scale_u + self.shift_u

        else:
            print("Error: Cannot solve mpc..")
            su = u
            u = u * self.scale_u + self.shift_u

        #--------------------store the old data------------------------#
        self.data_save_x.append(x_0)
        self.data_save_u.append(su)



        # delte some old data
        self.data_save_x.pop(0)
        self.data_save_u.pop(0)

        if self.p_dim > 0:
            self.data_save_p.append(self.data[reference,:])
            self.data_save_p.pop(0)

        return u
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success
    
    def reset(self):
        pass
# 不如直接重写 不然很麻烦
class Upper_MPC_mamba(object):
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['state_dim']
        self.args = args
        self.model = model
        self._build_matrices(args)
        self.init = False
        self.next = False

        self.R = R
        self.Q = Q
        self.P = P


        self.data_save_x = []
        self.data_save_u = []
        self.data_save_p = []

        self.pred_tracking = args['pred_tracking']

        if args['disturbance']>0:
            self.data = np.loadtxt('envs/Inf_dry_constQ.txt')
            self.data = np.concatenate([self.data[:,[14]],self.data[:,1:14]],axis=1) 
            self.data = (self.data-self.model.shift_u[self.a_dim:])/self.model.scale_u[self.a_dim:]

    def _get_set_point_u(self):
        # self.ref.value = self.reference
        pass



    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                if args['disturbance']>0:
                    self.len_u        = np.size(self.shift_u)-args['disturbance']
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                else:
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None


    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim = args['latent_dim']
        self.u_dim = args['act_dim']
        self.L     = args['pred_horizon']
        self.p_dim = args['disturbance']
        self.x_use = args['state_use']

        # self.Q = np.diag([0.5, 0.5, 0.5, 0.5 ,0.5, 0.5, 0.5, 0.5, 0.5])
        # self.Q = np.diag([0.75, 0.5, 0.5, 0.75 ,0.5, 0.75, 0.75, 0.5, 2.5])
        # self.R = np.diag([0.1,0.1,0.1])
        # self.Q = np.diag([0.5, 0.002, 1.7, 1.1 ,0.005, 1.7, 1.1, 0.05, 1.5])

        # self.A_holder = Parameter((self.L*self.x_dim, self.x_dim))              
        self.A_holder = Parameter((self.L, self.x_dim))              # diagnoal matrix
        self.B_holder = Parameter((self.x_dim*self.L, self.u_dim+self.p_dim))  
        self.C_holder = Parameter((self.x_dim*self.L, self.x_use)) 

        # try to adjust Q P R via delta
        # self.delta    = Parameter((self.L,self.x_dim)) 

        # self.u_s_holder = Parameter((self.a_dim))



    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)

        self.create_prob(self.args)
        self._get_set_point_u()

    
    
    def create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.x_dim, self.control_horizon+1))
        if self.p_dim>0:
            self.p = Parameter((self.p_dim, self.control_horizon))

        self.x_init = Parameter(self.x_dim)           
        objective = 0.
        constraints =  [self.x[:, 0] == self.x_init]

        for k in range(self.control_horizon-1):
            # k_u = k if k <= self.control_horizon-1 else self.control_horizon-1
            if self.p_dim>0:
                mix = hstack([self.u[:,k],self.p[:,k]])
            else:
                mix = self.u[:,k]
            constraints += [self.x[:, k + 1] == multiply(self.A_holder[k,:],self.x[:, k]) + self.B_holder[k*self.x_dim:(k+1)*self.x_dim,:] @ mix]
            j_ = self.C_holder[k*self.x_dim:(k+1)*self.x_dim,:].T @ self.x[:,k]
            objective += quad_form(j_[args['state_tracking_c']]-self.reference, self.Q)

            if k <= self.control_horizon-1 and k>0:
                objective += quad_form(self.u[:,k]-self.u[:,k-1],self.R)

            if args['apply_action_constraints']:
                constraints += [self.a_bound_low <= self.u[:, k], self.u[:, k] <= self.a_bound_high]
        
        k = k+1
        j_ = self.C_holder[k*self.x_dim:(k+1)*self.x_dim,:].T @ self.x[:,k]
        objective += quad_form(j_[args['state_tracking_c']]-self.reference, self.P)
        self.prob = Problem(Minimize(objective), constraints)
        
    def vector_matrix(vector):
        return vector.reshape([vector.shape[0],-1])
    

    def choose_action(self, x_0, reference, *args):
        """
        reference is the time instant
        """

        x_0 = (x_0-self.shift)/self.scale
        x_0 = x_0[self.pred_tracking]

        # if not self.init:
        #     self.init = True
        #     self.start_x = x_0
            
        #     for i in range(self.L):
        #         self.data_save_x.append(self.start_x)
        #         if self.p_dim>0:
        #             len_u = np.size(self.shift_u)-self.p_dim
        #             self.data_save_u.append((np.zeros([self.u_dim])-self.shift_u[:len_u])/self.scale_u[:len_u])
        #             self.data_save_p.append(self.data[reference])
        #         else:
        #             self.data_save_u.append((np.zeros([self.u_dim])-self.shift_u)/self.scale_u)


        phi_0 = self.model.net.project_x(torch.from_numpy(x_0).to(torch.float32)) # project_x 
        
        add_x = torch.from_numpy(np.array(self.data_save_x).squeeze()).to(torch.float32)
        add_u = torch.from_numpy(np.array(self.data_save_u).squeeze()).to(torch.float32)


        if self.p_dim>0:
            add_p = torch.from_numpy(np.array(self.data_save_p).squeeze()).to(torch.float32)
            p_ = self.data[reference:reference+self.pred_horizon,:]
            p_ = torch.from_numpy(p_).to(torch.float32)
            p_ = self.model.net.project_p(p_).detach().numpy().T
            self.p.value = p_[:,:self.control_horizon]
            delta, B, C = self.model.net.generate_matrices(add_x.unsqueeze(0), add_u.unsqueeze(0), add_p.unsqueeze(0))
        else:
            delta, B, C = self.model.net.generate_matrices(add_x.unsqueeze(0), add_u.unsqueeze(0), phi_0.unsqueeze(0))

        # if reference<self.L: #not enough to generate A B C
        #     add_x = np.repeat(self.start_x[self.model.pred_index].reshape(-1,1),self.L-reference,axis = 1).T
        #     add_u = np.repeat(np.zeros([self.u_dim, 1]),self.L-reference,axis = 1).T
        #     add_p = repeat(self.start_x[145:],'n -> d n', d=self.L-reference)
        #     if reference != 0:
        #         add_x = np.concatenate([add_x,np.array(self.data_save_x)],axis=0)
        #         add_u = np.concatenate([add_u,np.array(self.data_save_u)],axis=0)
        #         add_p = np.concatenate([add_p,np.array(self.data_save_p)],axis=0)
        # else:
        #     add_x = np.array(self.data_save_x)
        #     add_u = np.array(self.data_save_u)
        #     add_p = np.array(self.data_save_p)


        # add_u = torch.from_numpy(np.array(self.data_save_u).squeeze()).to(torch.float32)


        
        A,B,C = self.model.net.discretion_matrices(self.model.net.A, B, C, delta)

        self.A_holder.value = A.squeeze().reshape(-1,self.x_dim).detach().numpy()
        self.B_holder.value = B.squeeze().reshape(-1,self.u_dim+self.p_dim).detach().numpy()
        self.C_holder.value = C.squeeze().reshape(-1,self.x_use).detach().numpy()
        
        self.x_init.value = phi_0.detach().numpy()

        self.prob.solve(solver=MOSEK, warm_start=True, mosek_params={mosek.iparam.intpnt_solve_form:mosek.solveform.dual})
        u = self.u[:, 0].value
        print(self.prob.objective.value)
        # print(self.prob.objective.value)
        # print("+++++++++++this is u+++++++++++++")
        # print(u)
        # print("+++++++++++++++++++++++++++++++++")


        #-------------------for test-----------------#
        # j_u = self.u.value.T
        # j_u = j_u.reshape(1,self.L,2)
        # j_p = self.p.value.T
        # j_p = j_p.reshape(1,self.L,14)

        # j_u = torch.from_numpy(j_u).to(torch.float32)
        # j_p = torch.from_numpy(j_p).to(torch.float32)
        # z0 = phi_0.unsqueeze(0)
        # out = self.model.net.selective_scan(z0,j_u,j_p,A,B,C).squeeze().detach().numpy()
        # out_ = self.x.value.T
        # out_ = out_[1:,:]
        # out_ = torch.from_numpy(out_).to(torch.float32)
        # C_ = C[0,:,:,:]
        # out_ = einsum(C_,out_,'A B C, A B -> A C').detach().numpy()
        # print(np.max(np.abs(out-out_)))

  
        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            if self.p_dim>0:
                su = u
                self.u_save = u
                u = u * self.scale_u[:self.len_u] + self.shift_u[:self.len_u]
            else:
                su = u
                self.u_save = u
                u = u * self.scale_u + self.shift_u
            # print("-------------this is u-------------")
            # print(u)
            # print("-----------------------------------")
            

        else:
            print("Error: Cannot solve mpc..")
            su = u
            u = u * self.scale_u + self.shift_u

        #--------------------store the old data------------------------#
        self.data_save_x.append(x_0)
        self.data_save_u.append(su)



        # delte some old data
        self.data_save_x.pop(0)
        self.data_save_u.pop(0)

        if self.p_dim > 0:
            self.data_save_p.append(self.data[reference,:])
            self.data_save_p.pop(0)

        return u
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success
    
    def reset(self):
        pass



    

class Upper_MPC_DKO(object):
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['state_dim']
        self.args = args
        self.model = model
        self._build_matrices(args)
        self.init = False
        self.next = False

        self.Q = Q
        self.R = R
        self.P = P

        self.pred_tracking = args['pred_tracking']

        if args['disturbance']>0:
            self.data = np.loadtxt('envs/Inf_dry_constQ.txt')
            self.data = np.concatenate([self.data[:,[14]],self.data[:,1:14]],axis=1) 
            self.data = (self.data-self.model.shift_u[self.a_dim:])/self.model.scale_u[self.a_dim:]



    def _get_set_point_u(self):
        # self.ref.value = self.reference
        pass

    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                if args['disturbance']>0:
                    self.len_u        = np.size(self.shift_u)-args['disturbance']
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                else:
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None


    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim = args['latent_dim']
        self.u_dim = args['act_dim']
        self.L     = args['pred_horizon']
        self.p_dim = args['disturbance']
        self.x_use = args['state_dim']


        # self.Q = np.diag([0.5, 0.5, 0.5, 0.5 ,0.5, 0.5, 0.5, 0.5, 0.5])
        # self.Q = np.diag([1.5, 0.05, 0.5, 0.5 ,0.05, 0.5, 0.5, 0.5, 0.5])
        # self.Q = np.diag([0.1, 0.02, 3.5, 0.1 ,0.05, 3.5, 0.1, 0.05, 2.5])

        # self.R = np.diag([0.1,0.1,0.1])




        # self.A_holder = Parameter((self.L*self.x_dim, self.x_dim))              
        # self.A_holder = Parameter((self.x_dim, self.x_dim))              # diagnoal matrix
        # self.B_holder = Parameter((self.x_dim, self.u_dim+self.p_dim))  
        # self.C_holder = Parameter((self.x_use, self.x_dim)) 


        # self.u_s_holder = Parameter((self.a_dim))



    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)

        self.A_holder = self.model.net.A.detach().numpy()
        self.B_holder = self.model.net.B.detach().numpy()
        self.C_holder = self.model.net.C.detach().numpy()

        self.create_prob(self.args)
        self._get_set_point_u()

    
    
    def create_prob(self, args):
        self.u = Variable((self.a_dim, self.control_horizon))
        self.x = Variable((self.x_dim, self.control_horizon+1))
        
        if self.p_dim>0:
            self.p = Parameter((self.p_dim, self.control_horizon))

        self.x_init = Parameter(self.x_dim)        
        objective = 0.
        constraints =  [self.x[:, 0] == self.x_init]


        for k in range(self.control_horizon-1):
            if self.p_dim>0:
                mix = hstack([self.u[:,k],self.p[:,k]])
            else:
                mix = self.u[:,k]
            constraints += [self.x[:, k + 1] == self.A_holder @ self.x[:, k] + self.B_holder @ mix]
            j_ = self.C_holder @ self.x[:,k]
            objective += quad_form(j_[args['state_tracking_c']]-self.reference, self.Q)

            if k <= self.control_horizon-1 and k>0:
                objective += quad_form(self.u[:,k]-self.u[:,k-1],self.R)

            if args['apply_action_constraints']:
                constraints += [self.a_bound_low <= self.u[:, k], self.u[:, k] <= self.a_bound_high]
        
        j_ = self.C_holder @ self.x[:,k+1]
        objective += quad_form(j_[args['state_tracking_c']]-self.reference, self.P)
        self.prob = Problem(Minimize(objective), constraints)
        
    def vector_matrix(vector):
        return vector.reshape([vector.shape[0],-1])
    

    def choose_action(self, x_0, reference, *args):
        """
        reference is the time instant
        """

        x_0 = (x_0-self.shift)/self.scale
        x_0 = x_0[self.pred_tracking]

        phi_0 = self.model.net.project_x(torch.from_numpy(x_0).to(torch.float32)) # project_x 

        if self.p_dim>0:
            p_ = self.data[reference:reference+self.pred_horizon,:]
            p_ = torch.from_numpy(p_).to(torch.float32)
            p_ = self.model.net.project_p(p_).detach().numpy().T
            self.p.value = p_[:,:self.control_horizon]

        self.x_init.value = phi_0.detach().numpy()

        self.prob.solve(solver=MOSEK, warm_start=True, mosek_params={mosek.iparam.intpnt_solve_form:mosek.solveform.dual})
        u = self.u[:, 0].value
        print(self.prob.objective.value)

        if self.prob.status == OPTIMAL or self.prob.status == OPTIMAL_INACCURATE:
            if self.p_dim>0:
                u = u * self.scale_u[:self.len_u] + self.shift_u[:self.len_u]
            else:
                u = u * self.scale_u + self.shift_u
        else:
            print("Error: Cannot solve mpc..")
            u = u * self.scale_u + self.shift_u

        return u
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success
    
    def reset(self):
        pass

class Upper_MPC_MLP(object):
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['pred_tracking'].shape[0]
        self.args = args
        self.model = model

        self.model.net.eval()

        self._build_matrices(args)
        self.init = False
        self.next = False

        self.Q = Q
        self.R = R
        self.P = P

        self.state_dict = self.model.net.state_dict()

        self.w = []

        for param_tensor in self.state_dict:
            self.w.append(self.state_dict[param_tensor].numpy())
        
        self.pred_tracking = args['pred_tracking']

        if args['disturbance']>0:
            self.data = np.loadtxt('envs/Inf_dry_constQ.txt')
            self.data = np.concatenate([self.data[:,[14]],self.data[:,1:14]],axis=1) 
            self.data = (self.data-self.model.shift_u[4:])/self.model.scale_u[4:]


    def _get_set_point_u(self):
        # self.ref.value = self.reference
        pass

    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                if args['disturbance']>0:
                    self.len_u        = np.size(self.shift_u)-args['disturbance']
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                else:
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None


    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim = args['latent_dim']
        self.u_dim = args['act_dim']
        self.L     = args['pred_horizon']
        self.p_dim = args['disturbance']
        self.x_use = args['state_dim']


        self.u_dim   = args['act_dim'] + self.p_dim
        self.state_c = args['state_tracking_c']
        self.state_t = args['pred_tracking']

        self.c_length= self.state_c.shape[0]
        self.t_length= self.state_t.shape[0]
    
    def Reward_state(self,Z):
        Err = Z-self.reference
        return ca.mtimes([Err.T,self.Q, Err])
    
    def Reward_state_T(self,Z):
        Err = Z-self.reference
        return ca.mtimes([Err.T,self.P, Err])
    
    def Reward_input(self,U_0,U_1):
        Err = U_0-U_1
        return  ca.mtimes([Err.T,self.R,Err])



    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)
        self.create_prob(self.args)
        self._get_set_point_u()


    def system_dynamics(self,x, u, p=None):

        if p is not None and self.p_dim > 0:
            m = ca.vertcat(x,u,p)
        else:
            m = ca.vertcat(x,u)

        j = 0
        while(j<len(self.w)):
            m = ca.mtimes(self.w[j],m)
            j+=1
            m = m + self.w[j]
            j+=1
            if j<=4:
                # numerically stable softplus: max(m,0) + log(1+exp(-|m|))
                m = ca.fmax(m, 0) + ca.log(1 + ca.exp(-ca.fabs(m)))
        return m


    def create_prob(self, args):


        x =  ca.MX.sym('x',self.s_dim,self.control_horizon+1)
        u =  ca.MX.sym('u',self.a_dim,self.control_horizon)
        self.p =  ca.MX.sym('p',self.p_dim,self.control_horizon)


        self.x0    = ca.MX.sym('x0', self.s_dim)
        self.J   = 0

        constraints = []
        constraints.append(x[:, 0] - self.x0)
        if self.p_dim==0:
            for k in range(self.control_horizon-1):

                constraints.append(x[:,k+1]-self.system_dynamics(x[:,k],u[:,k]))
                self.J += (self.Reward_state(x[self.state_c,k])+self.Reward_input(u[:,k],u[:,k+1]))

            k = self.control_horizon - 1
            constraints.append(x[:,k+1]-self.system_dynamics(x[:,k],u[:,k]))
            self.J += self.Reward_state(x[self.state_c,k])

            self.J += self.Reward_state_T(x[self.state_c, self.control_horizon])


            nlp = {'x': ca.vertcat(ca.reshape(u, -1, 1), ca.reshape(x,-1, 1)),  #first U and then X
                'f': self.J,
                'g': ca.vertcat(*constraints),
                'p': ca.vertcat(self.x0)}

        else:
            for k in range(self.control_horizon-1):

                constraints.append(x[:,k+1]-self.system_dynamics(x[:,k],u[:,k],self.p[:,k]))
                self.J += (self.Reward_state(x[self.state_c,k])+self.Reward_input(u[:,k],u[:,k+1]))

            k = self.control_horizon - 1
            constraints.append(x[:,k+1]-self.system_dynamics(x[:,k],u[:,k],self.p[:,k]))
            self.J += self.Reward_state(x[self.state_c,k])

            self.J += self.Reward_state_T(x[self.state_c, self.control_horizon])


            nlp = {'x': ca.vertcat(ca.reshape(u, -1, 1), ca.reshape(x,-1, 1)),  #first U and then X
                'f': self.J,
                'g': ca.vertcat(*constraints),
                'p': ca.vertcat(self.x0, ca.reshape(self.p, -1, 1))}



        opts = {'ipopt': {'print_level': 3, 'max_cpu_time': 100, 'max_iter': 500}}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbx = np.concatenate([np.tile(self.a_bound_low,self.control_horizon),  np.full(self.s_dim*(self.control_horizon+1), -np.inf)])  # No constraints on x
        self.ubx = np.concatenate([np.tile(self.a_bound_high,self.control_horizon), np.full(self.s_dim*(self.control_horizon+1), np.inf)])

        self.lbg = 0
        self.ubg = 0

        
    def vector_matrix(vector):
        return vector.reshape([vector.shape[0],-1])
    

    def choose_action(self, x_0, reference, *args):
        """
        reference is the time instant
        """
        # phi_0 = (x_0-self.shift)/self.scale
        x_0 = (x_0-self.shift)/self.scale
        phi_0 = x_0[self.pred_tracking]

        if not self.init:
            self.init = True
            self.x0_guess = np.concatenate([np.zeros((self.a_dim*self.control_horizon, 1)), np.tile(phi_0, (self.control_horizon+1, 1)).T.reshape(-1, 1)])
        
        if self.p_dim>0:
            p_ = self.data[reference:reference+self.pred_horizon,:].T
            # p_ = torch.from_numpy(p_).to(torch.float32)
            # p_ = self.model.net.project_p(p_).detach().numpy().T
        
        
        sol = self.solver(x0=self.x0_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=ca.vertcat(phi_0, p_.reshape(-1,1)))

        U_opt = np.array(ca.vertsplit(sol['x'])[:self.a_dim*(self.control_horizon)]).flatten()
        X_opt = np.array(ca.vertsplit(sol['x'])[self.a_dim*(self.control_horizon):]).flatten()
        Pred_U = U_opt.reshape([self.a_dim,-1])
        Pred_X = X_opt.reshape([self.s_dim,-1])

        self.x0_guess = np.array(ca.vertsplit(sol['x'])).flatten()
        u_out  = Pred_U[:,0] * self.scale_u[:self.len_u] + self.shift_u[:self.len_u]

        print(u_out)
        
        return u_out
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success
    
    def reset(self):
        pass



class Diff_MPC_DKO_D(object):
    """
    Write with casadi

    1. Considering disturbance
    2. Considering the augmented system
    3. Considering the economic index

    """
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['state_dim']
        self.args = args
        self.model = model
        self.init = False
        self.next = False
        # self.stable_u = np.array([2.900422837603922367e+09,1.000326699772864342e+09,2.900133719693773746e+09])

        self.u_last_init = np.array([0,0]) # TODO: steady state
        self.init        = False

        self.Q = Q
        self.R = R        
        self.learning_rate = 0.001   

        self.pred_tracking = args['pred_tracking']
        self.model.net.eval()
        self._build_matrices(args)

        self.state_dict = self.model.net.state_dict()

        self.w = []

        for param_tensor in self.state_dict:
            self.w.append(self.state_dict[param_tensor].numpy())


        if args['disturbance']>0:
            self.data = np.loadtxt('envs/Inf_dry_2006.txt')
            self.data = np.concatenate([self.data[:,[14]],self.data[:,1:14]],axis=1) 
            self.data = (self.data-self.model.shift_u[2:])/self.model.scale_u[2:]
    
    def _build_matrices(self, args):
        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim   = args['latent_dim']
        self.L       = args['pred_horizon']
        self.p_dim   = args['disturbance']
        self.x_use   = args['state_dim']
        self.u_dim   = args['act_dim'] + self.p_dim
        self.state_c = args['state_tracking_c']
        self.state_t = args['pred_tracking']

        self.c_length= self.state_c.shape[0]
        self.t_length= self.state_t.shape[0]


    def system_dynamics(self,x, u, p):
        m = ca.vertcat(x,u,p)
        j = 0
        while(j<len(self.w)):
            m = ca.mtimes(self.w[j],m)
            j+=1
            m = m + self.w[j]
            j+=1
            if j<=4:
                # numerically stable softplus: max(m,0) + log(1+exp(-|m|))
                m = ca.fmax(m, 0) + ca.log(1 + ca.exp(-ca.fabs(m)))
        return m
    
    
    def Reward_state(self,x,u):
        Z = ca.vertcat(x,u)
        Err = ca.mtimes([self.C_holder, Z])-self.xs
        return ca.mtimes([Err.T,self.Q, Err])
    
    def Reward_input(self,U):
        # return U.T@self.R@U
        return  ca.mtimes([U.T,self.R,U])
    
    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                if args['disturbance']>0:
                    self.len_u        = np.size(self.shift_u)-args['disturbance']
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                else:
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None
    
    def _build_controller(self):
        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference   = self.args['reference']
        self.reference_  = self.args['reference_']
        self.reference_d = self.reference

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)

        # self.A_holder = self.model.net.A.detach().numpy()
        # self.B_holder = self.model.net.B.detach().numpy()
        # self.C_holder = self.model.net.C.detach().numpy()

        self.A_holder = np.zeros([self.x_dim+self.a_dim,self.x_dim+self.a_dim])
        self.B_holder = np.zeros([self.a_dim+self.x_dim, self.a_dim])
        self.P_holder = np.zeros([self.x_dim+self.a_dim, self.p_dim])
        self.C_holder = np.zeros([self.t_length,self.x_dim+self.a_dim])

        self.index_   = np.arange(self.c_length)

        B             = self.model.net.B.detach().numpy()
        B1            = B[:,:self.a_dim]
        B2            = B[:,self.a_dim:]

        self.A_holder[:self.x_dim,:self.x_dim]             = self.model.net.A.detach().numpy()
        self.A_holder[-self.a_dim:,-self.a_dim:]           = np.eye(self.a_dim,self.a_dim)
        self.A_holder[:self.x_dim,self.x_dim:]             = B1
        self.B_holder[:self.x_dim,:]                       = B1
        self.B_holder[self.x_dim:,:]                       = np.eye(self.a_dim,self.a_dim)
        self.P_holder[:self.x_dim,:]                       = B2
        self.C_holder[:,:self.x_dim]                       = self.model.net.C.detach().numpy()


        self.zeros_                                        = np.zeros([self.c_length,self.t_length])
        self.zeros_[self.index_,self.state_c]=1
        self.C_holder                                      = self.zeros_@self.C_holder
        
        self.J    = 0

        # Variables
        x       = ca.MX.sym('x', self.x_dim,self.control_horizon+1)
        u       = ca.MX.sym('u', self.a_dim,self.control_horizon+1)
        delta_u = ca.MX.sym('delta_u', self.a_dim,self.control_horizon)

        # Parameters
        self.x0     = ca.MX.sym('x0', self.x_dim)
        self.xs     = ca.MX.sym('xs', self.c_length,1)
        self.u_last = ca.MX.sym('u_last',self.a_dim)
        self.p_last = ca.MX.sym('p_',self.p_dim,self.control_horizon)


        constraints = []
        constraints.append(x[:, 0] - self.x0)
        constraints.append(u[:, 0] - self.u_last)

        for k in range(self.control_horizon):
            x_next,u_next =self.Koopman_model(x[:,k],u[:,k],delta_u[:,k],self.p_last[:,k])
            constraints.append(x[:,k+1]-x_next)
            constraints.append(u[:,k+1]-u_next)
            self.J += (self.Reward_state(x[:,k],u[:,k])+self.Reward_input(delta_u[:,k]))

        self.J += self.Reward_state(x[:,self.control_horizon],u[:,self.control_horizon])

        nlp = {'x': ca.vertcat(reshape(u, -1, 1), reshape(x,-1, 1), reshape(delta_u,-1, 1)),  #U X delta_U
               'f': self.J*0.5, 
               'g': ca.vertcat(*constraints),
               'p': ca.vertcat(self.x0, reshape(self.xs, -1, 1), reshape(self.u_last, -1, 1), reshape(self.p_last, -1, 1))} # x0,xs,u_last,p
        
        opts = {'ipopt': {'print_level': 0}}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # self.lbx = np.concatenate([np.tile(self.a_bound_low,self.control_horizon),  np.full(self.x_dim*(self.control_horizon+1), -np.inf)])  # No constraints on x
        # self.ubx = np.concatenate([np.tile(self.a_bound_high,self.control_horizon), np.full(self.x_dim*(self.control_horizon+1), np.inf)]) 

        self.lbx = np.concatenate([np.tile(self.a_bound_low,self.control_horizon+1),  np.full(self.x_dim*(self.control_horizon+1), -np.inf), np.full(self.a_dim*(self.control_horizon), -np.inf)])  # No constraints on x
        self.ubx = np.concatenate([np.tile(self.a_bound_high,self.control_horizon+1), np.full(self.x_dim*(self.control_horizon+1), np.inf) , np.full(self.a_dim*(self.control_horizon), np.inf)]) 

        self.lbg = 0
        self.ubg = 0


        # """
        # With Casadi build MPC problem
        # """
        # self.opti = Opti()
        
        # self.J    = 0
        # self.X0   = self.opti.parameter(self.x_dim,1) # which is z0
        # self.Q_   = self.opti.parameter(self.s_dim,self.s_dim) # which is z0
        # # self.Q
        # self.X    = self.opti.variable(self.x_dim,self.control_horizon)
        # self.U    = self.opti.variable(self.u_dim,self.control_horizon) #TODO: Pay attention U 

        # self.opti.subject_to(self.X[:,0]==self.Koopman_model(self.X0,self.U[:,0]))

        # self.J    += self.Reward_state(self.X[:,0])
        # self.J    += self.Reward_input(self.U[:,0])

        # self.opti.subject_to(self.U[:,0]<=self.a_bound_high)
        # self.opti.subject_to(self.a_bound_low<=self.U[:,0])

        # for k in range(self.control_horizon-1):

        #     self.opti.subject_to(self.X[:,k+1]==self.Koopman_model(self.X[:,k],self.U[:,k+1]))

        #     self.J += self.Reward_state(self.X[:,k+1])
        #     self.J += self.Reward_input(self.U[:,k+1])

        #     self.opti.subject_to(self.U[:,k+1]<=self.a_bound_high)
        #     self.opti.subject_to(self.a_bound_low<=self.U[:,k+1])

        # # opti.solve()
        #     # opts = {'print_level': 0}
        #     self.opti.solver('ipopt')
        # # self.opti.solver.options['print_level'] = 0

    
    def choose_action(self,x_0, reference, *args):

        x_0 = (x_0-self.shift)/self.scale
        x_0 = x_0[self.state_t]
        phi_0 = self.model.net.project_x(torch.from_numpy(x_0).to(torch.float32)).detach().numpy() # project_x 

        if not self.init:
            x0_guess = np.concatenate([np.zeros((self.a_dim*(self.control_horizon+1), 1)), np.tile(phi_0, (self.control_horizon+1, 1)).T.reshape(-1, 1), np.zeros((self.a_dim*self.control_horizon, 1))])
        else:
            x0_guess = self.x_store

        if self.p_dim>0:
            p_ = self.data[reference:reference+self.pred_horizon,:]
            p_ = torch.from_numpy(p_).to(torch.float32)
            p_ = self.model.net.project_p(p_).detach().numpy().T
            # self.p.value = p_[:,:self.control_horizon]
            # self.x_init.value = phi_0.detach().numpy()
        sol   = self.solver(x0=x0_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=ca.vertcat(phi_0, self.reference_d,self.u_last_init.reshape(-1, 1), p_.reshape(-1,1)))
        U_opt = np.array(ca.vertsplit(sol['x'])[:self.a_dim*(self.control_horizon+1)]).flatten()
        X_opt = np.array(ca.vertsplit(sol['x'])[:(self.a_dim+self.x_dim)*(self.control_horizon+1)]).flatten()
        # print("jjj")
        Pred_U       = U_opt.reshape([-1,self.a_dim])
        self.x_store = np.array(ca.vertsplit(sol['x']))
        Pred_X       = X_opt.reshape([-1,self.x_dim+self.a_dim])

        if self.p_dim>0:
            u_out  = Pred_U[1,:] * self.scale_u[:self.len_u] + self.shift_u[:self.len_u]

        else:
            u_out  = Pred_U[1,:] * self.scale_u + self.shift_u

        print(u_out)


        # return sol.value(self.U)[:,0]
        return u_out
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        self._build_controller()
        return success

    def reset(self):
        pass


class Upper_MPC_MLP(object):
    def __init__(self, model, args, Q, R, P):
        self.control_horizon = args['control_horizon']
        self.pred_horizon = args['pred_horizon']
        self.a_dim = args['act_dim']
        self.s_dim = args['pred_tracking'].shape[0]
        self.args = args
        self.model = model

        self.model.net.eval()

        self._build_matrices(args)
        self.init = False
        self.next = False

        self.Q = Q
        self.R = R
        self.P = P

        self.state_dict = self.model.net.state_dict()

        self.w = []

        for param_tensor in self.state_dict:
            self.w.append(self.state_dict[param_tensor].numpy())
        
        self.pred_tracking = args['pred_tracking']

        if args['disturbance']>0:
            self.data = np.loadtxt('envs/Inf_dry_constQ.txt')
            self.data = np.concatenate([self.data[:,[14]],self.data[:,1:14]],axis=1) 
            self.data = (self.data-self.model.shift_u[4:])/self.model.scale_u[4:]


    def _get_set_point_u(self):
        # self.ref.value = self.reference
        pass

    def _shift_and_scale_bounds(self, args):
        if np.sum(self.scale) > 0. and np.sum(self.scale_u) > 0.:
            if args['apply_state_constraints']:
                self.s_bound_high = (args['s_bound_high'] - self.shift) / self.scale
                self.s_bound_low = (args['s_bound_lowh'] - self.shift) / self.scale
            else:
                self.s_bound_low = None
                self.s_bound_high = None
            if args['apply_action_constraints']:
                if args['disturbance']>0:
                    self.len_u        = np.size(self.shift_u)-args['disturbance']
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u[:self.len_u]) / self.scale_u[:self.len_u]
                else:
                    self.a_bound_high = (args['a_bound_high'] - self.shift_u) / self.scale_u
                    self.a_bound_low  = (args['a_bound_low'] - self.shift_u) / self.scale_u
                print("wwwwwwww------------------------------------------------------------------------")
            else:
                self.a_bound_low = None
                self.a_bound_high = None


    def _build_matrices(self, args):

        # 通过cvxpy构建线性问题 -> 包含参数设计
        self.x_dim = args['latent_dim']
        self.u_dim = args['act_dim']
        self.L     = args['pred_horizon']
        self.p_dim = args['disturbance']
        self.x_use = args['state_dim']


        self.u_dim   = args['act_dim'] + self.p_dim
        self.state_c = args['state_tracking_c']
        self.state_t = args['pred_tracking']

        self.c_length= self.state_c.shape[0]
        self.t_length= self.state_t.shape[0]
    
    def Reward_state(self,Z):
        Err = Z-self.reference
        return ca.mtimes([Err.T,self.Q, Err])
    
    def Reward_state_T(self,Z):
        Err = Z-self.reference
        return ca.mtimes([Err.T,self.P, Err])
    
    def Reward_input(self,U_0,U_1):
        Err = U_0-U_1
        return  ca.mtimes([Err.T,self.R,Err])



    def _build_controller(self):

        [self.shift, self.scale, self.shift_u, self.scale_u] = self.model.shift_
        self.reference = self.args['reference']
        self.reference_ = self.args['reference_']

        self.reference_use = self.reference.reshape(self.reference.shape[0],1)
        self._shift_and_scale_bounds(self.args)
        self.create_prob(self.args)
        self._get_set_point_u()


    def system_dynamics(self,x, u, p=None):

        if p is not None and self.p_dim > 0:
            m = ca.vertcat(x,u,p)
        else:
            m = ca.vertcat(x,u)

        j = 0
        while(j<len(self.w)):
            m = ca.mtimes(self.w[j],m)
            j+=1
            m = m + self.w[j]
            j+=1
            if j<=4:
                # exact ReLU — matches training activation (nn.ReLU)
                m = ca.fmax(0, m)
        return m


    def create_prob(self, args):


        x =  ca.MX.sym('x',self.s_dim,self.control_horizon+1)
        u =  ca.MX.sym('u',self.a_dim,self.control_horizon)
        self.p =  ca.MX.sym('p',self.p_dim,self.control_horizon)


        self.x0    = ca.MX.sym('x0', self.s_dim)
        self.J   = 0

        constraints = []
        constraints.append(x[:, 0] - self.x0)
        if self.p_dim==0:
            for k in range(self.control_horizon-1):

                constraints.append(x[:,k+1]-self.system_dynamics(x[:,k],u[:,k]))
                self.J += self.Reward_state(x[self.state_c,k])#+self.Reward_input(u[:,k],u[:,k+1]))

            k = self.control_horizon - 1
            constraints.append(x[:,k+1]-self.system_dynamics(x[:,k],u[:,k]))
            self.J += self.Reward_state(x[self.state_c,k])

            self.J += self.Reward_state_T(x[self.state_c, self.control_horizon])


            nlp = {'x': ca.vertcat(ca.reshape(u, -1, 1), ca.reshape(x,-1, 1)),  #first U and then X
                'f': self.J,
                'g': ca.vertcat(*constraints),
                'p': ca.vertcat(self.x0)}

        else:
            for k in range(self.control_horizon-1):

                constraints.append(x[:,k+1]-self.system_dynamics(x[:,k],u[:,k],self.p[:,k]))
                self.J += (self.Reward_state(x[self.state_c,k])+self.Reward_input(u[:,k],u[:,k+1]))

            k = self.control_horizon - 1
            constraints.append(x[:,k+1]-self.system_dynamics(x[:,k],u[:,k],self.p[:,k]))
            self.J += self.Reward_state(x[self.state_c,k])

            self.J += self.Reward_state_T(x[self.state_c, self.control_horizon])


            nlp = {'x': ca.vertcat(ca.reshape(u, -1, 1), ca.reshape(x,-1, 1)),  #first U and then X
                'f': self.J,
                'g': ca.vertcat(*constraints),
                'p': ca.vertcat(self.x0, ca.reshape(self.p, -1, 1))}



        opts = {'ipopt': {'print_level': 3, 'max_cpu_time': 100, 'max_iter': 500}}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        self.lbx = np.concatenate([np.tile(self.a_bound_low,self.control_horizon),  np.full(self.s_dim*(self.control_horizon+1), -np.inf)])  # No constraints on x
        self.ubx = np.concatenate([np.tile(self.a_bound_high,self.control_horizon), np.full(self.s_dim*(self.control_horizon+1), np.inf)])

        self.lbg = 0
        self.ubg = 0

        
    def vector_matrix(vector):
        return vector.reshape([vector.shape[0],-1])
    

    def choose_action(self, x_0, reference, *args):
        """
        reference is the time instant
        """
        x_0 = (x_0-self.shift)/self.scale
        phi_0 = x_0[self.pred_tracking]

        if not self.init:
            self.init = True
            self.x0_guess = np.concatenate([
                np.zeros(self.a_dim * self.control_horizon),
                np.tile(phi_0, self.control_horizon + 1)
            ])

        if self.p_dim>0:
            # shape (control_horizon, 14): row-major reshape → [step0 14dists, step1 14dists, ...]
            # matches CasADi column-major ca.reshape(self.p, -1, 1) for self.p of shape (14, control_horizon)
            p_ = self.data[reference:reference+self.control_horizon,:]

        sol = self.solver(x0=self.x0_guess, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=ca.vertcat(phi_0, p_.reshape(-1,1)))

        stats = self.solver.stats()
        if not stats['success']:
            print(f"[MLP-MPC] WARNING: solver status={stats['return_status']}, iter={stats['iter_count']}, obj={float(sol['f']):.4f}")

        U_opt = np.array(sol['x'][:self.a_dim*self.control_horizon]).flatten()
        X_opt = np.array(sol['x'][self.a_dim*self.control_horizon:]).flatten()
        Pred_U = U_opt.reshape([self.a_dim, -1])
        Pred_X = X_opt.reshape([self.s_dim, -1])

        # Shifted warm start: move solution one step forward for next call
        U_shifted = np.hstack([Pred_U[:, 1:], Pred_U[:, -1:]])
        X_shifted = np.hstack([Pred_X[:, 1:], Pred_X[:, -1:]])
        self.x0_guess = np.concatenate([U_shifted.flatten(), X_shifted.flatten()])

        u_out = Pred_U[:,0] * self.scale_u[:self.len_u] + self.shift_u[:self.len_u]

        print(f"[MLP-MPC] u={u_out}, obj={float(sol['f']):.4f}, iter={stats['iter_count']}")

        return u_out
    
    def restore(self):
        success = self.model.parameter_restore(self.args)
        # re-extract weights after restore
        self.state_dict = self.model.net.state_dict()
        self.w = []
        for param_tensor in self.state_dict:
            self.w.append(self.state_dict[param_tensor].numpy())
        self._build_controller()
        self.verify_model()
        return success

    def reset(self):
        pass

    def verify_model(self, n_tests=5):
        """
        Verify that the CasADi system_dynamics matches the PyTorch MLP.
        Uses ReLU (exact) for this test to isolate weight-loading errors from
        the softplus approximation used during MPC.
        """
        import torch

        # Build a CasADi function with exact ReLU for comparison
        x_sym = ca.MX.sym('xv', self.s_dim)
        u_sym = ca.MX.sym('uv', self.a_dim)
        p_sym = ca.MX.sym('pv', self.p_dim)

        m = ca.vertcat(x_sym, u_sym, p_sym)
        j = 0
        while j < len(self.w):
            m = ca.mtimes(self.w[j], m);  j += 1
            m = m + self.w[j];            j += 1
            if j <= 4:
                m = ca.fmax(0, m)   # exact ReLU
        f_relu = ca.Function('f_relu', [x_sym, u_sym, p_sym], [m])

        print("\n========== Upper_MPC_MLP: model verification ==========")
        max_errs = []
        for i in range(n_tests):
            np.random.seed(i)
            x_t = np.random.randn(self.s_dim).astype(np.float32)
            u_t = np.random.randn(self.a_dim).astype(np.float32)
            p_t = np.random.randn(self.p_dim).astype(np.float32)

            # --- CasADi (exact ReLU): use .full() to get proper numpy float64 array ---
            ca_out = f_relu(x_t, u_t, p_t).full().flatten()

            # --- PyTorch ---
            inp = torch.tensor(np.concatenate([x_t, u_t, p_t])).unsqueeze(0)
            with torch.no_grad():
                pt_out = self.model.net.layers(inp).numpy().flatten()

            err = np.abs(ca_out - pt_out)
            max_err = float(err.max())
            max_errs.append(max_err)
            print(f"  test {i}: max_err={max_err:.3e}  "
                  f"ca[:3]={ca_out[:3].round(4)}  pt[:3]={pt_out[:3].round(4)}")

        overall = float(np.max(max_errs))
        if overall < 1e-4:
            print(f"[PASS] weights loaded correctly, max error = {overall:.3e}")
        else:
            print(f"[FAIL] large mismatch! max error = {overall:.3e}  "
                  f"-- check weight extraction or input ordering")
        print("=======================================================\n")
        return overall




if __name__ == '__main__':
    pass