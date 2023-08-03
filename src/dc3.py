from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
from .compilation import compile
from .newton import newton

class dc3:
    """
        minimize_x f(x)
        s.t.    g(x) <= 0
                h(x) == 0
    """
    def __init__(self, constraints, objective, tol=1e-5, newton_iters=50, correction_iters=5, gamma=0.001, momentum=0.9):
        eq_resid, ineq_resid = compile(constraints, objective)
        self.n = len(eq_resid) + len(ineq_resid)
        self.completion = completion(self.n, eq_resid, tol, newton_iters)
        self.ineq_resid = ineq_resid
        self.tol = tol
        self.t = correction_iters
        self.gamma = gamma
        self.momentum = momentum

    def forward(self, z):
        y, _, _ = self.completion.forward(z)
        old_step = 0
        for _ in range(self.t):
            m = len(self.ineq_resid)
            z = Tensor(y[:m].numpy(), requires_grad=True).cpu()
            opt = optim.SGD([z],lr=1)
            y_new, jac, jac_inv = self.completion.forward(z)
            opt.zero_grad()
            relu = Tensor.cat(*[f(y_new) for f in self.ineq_resid], dim=0).relu()

            if relu.numpy().mean() < self.tol:
                break
            
            relu.mul(relu).sum().backward()
            delta_z = z.grad
            dphi_dz = -jac_inv @ jac[:,:m]
            delta_phi = Tensor(dphi_dz).cpu() @ delta_z[:,None]
            step = Tensor.cat(delta_z, delta_phi[:,0])
            step = self.gamma*step + self.momentum*old_step
            y = y - step
            old_step = step
        return y
    
    
class completion:
    def __init__(self, n, eq_resid, tol, max_iters):
        self.n = n
        self.eq_resid = eq_resid
        self.tol = tol
        self.max_iters = max_iters

    def forward(self, z):
        y_tilda, self.jac, self.jac_inv = newton(z, self.eq_resid, self.tol, self.max_iters)
        return y_tilda, self.jac, self.jac_inv
    
    def backward(self, grad_output):
        m = self.n - len(self.eq_resid)
        dl_dz = grad_output[:m]
        dl_dphi = grad_output[m:]
        dphi_dz = -self.jac_inv @ self.jac[:,:m]   
        return dl_dz + dl_dphi @ Tensor(dphi_dz).cpu()