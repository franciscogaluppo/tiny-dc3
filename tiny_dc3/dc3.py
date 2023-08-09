from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
from functools import reduce
from .newton import newton

import numpy as np

class dc3:
    """
        minimize_x f(x)
        s.t.    g(x) <= 0
                h(x) == 0
    """
    def __init__(
        self, length, eq_resid, ineq_resid, newton_tol=1e-5, newton_iters=50,
        corr_iters=5, corr_lr=0.001, corr_momentum=0.9, corr_clip_grad_norm=None,
        int_mask=None, int_step=0.1, int_iters=5
    ):
        self.n = length
        self.m = length - sum(reduce(lambda x,y: x*y, f.eval_shape()) for f in eq_resid)
        self.completion = completion(self.n, self.m, eq_resid, newton_tol, newton_iters)
        self.ineq_resid = ineq_resid
        self.t = corr_iters
        self.gamma = corr_lr
        self.momentum = corr_momentum
        self.clip_grad = corr_clip_grad_norm
        self.int_mask = int_mask
        self.int_step = int_step
        self.int_iters = int_iters

    def forward(self, z):
        y, _, _ = self.completion.forward(z)
        old_step = 0
        for _ in range(self.t):
            z = Tensor(y[:,:self.m].numpy())
            opt = optim.SGD([z],lr=1)
            y_new, jac, jac_inv = self.completion.forward(z)
            opt.zero_grad()
            Tensor.cat(*[f(y_new)[:,None] for f in self.ineq_resid], dim=1).relu().pow(2).sum().backward()
            delta_z = z.grad
            dphi_dz = -jac_inv @ jac[:,:,:self.m]
            delta_phi = Tensor(dphi_dz) @ delta_z[:,:,None]
            step = Tensor.cat(delta_z, delta_phi[:,:,0], dim=1)

            if self.clip_grad is not None:
                norm = step.pow(2).sum(axis=1).sqrt().maximum(self.clip_grad)[:,None] / self.clip_grad
                step = step / norm

            step = self.gamma*step + self.momentum*old_step
            y = y - step
            old_step = step

            # Integer constraint: own loop or this loop?
            if self.int_mask is not None:
                y = y - self.int_step*self.int_mask*(2*np.pi*y).sin()/(2*np.pi)

        return y
    
    
class completion:
    def __init__(self, n, m, eq_resid, tol, max_iters):
        self.n = n
        self.m = m
        self.eq_resid = eq_resid
        self.tol = tol
        self.max_iters = max_iters

    def forward(self, z):
        y_tilda, self.jac, self.jac_inv = newton(z, self.eq_resid, self.tol, self.max_iters)
        return y_tilda, self.jac, self.jac_inv
    
    def backward(self, grad_output):
        m = self.m
        dl_dz = grad_output[:,:m]
        dl_dphi = grad_output[:,m:]
        dphi_dz = -self.jac_inv @ self.jac[:,:,:m]
        return dl_dz + dl_dphi @ Tensor(dphi_dz)