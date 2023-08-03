from tinygrad.tensor import Tensor
from .newton import newton

class dc3:
    """
        minimize_x f(x)
        s.t.    g(x) <= 0
                h(x) == 0
    """
    def __init__(self, constraints, objective, correction_iters=5):
        eq_resid, ineq_resid = compile()
        self.n = len(eq_resid) + len(ineq_resid)
        self.completion = completion(self.n, eq_resid)
        self.correction = correction(self.n, ineq_resid)
        self.t = correction_iters

    def forward(self, z):
        y = self.completion.forward(z)
        for _ in range(self.t):
            y = self.correction.forward(y)
        return y
    
    
class completion:
    def __init__(self, n, eq_resid):
        self.n = n
        self.eq_resid = eq_resid

    def forward(self, z):
        y_tilda, self.jac, self.jac_inv = newton(z, self.eq_resid)
        return y_tilda
    
    def backward(self, grad_output):
        m = self.n - len(self.eq_resid)
        dl_dz = grad_output[:m]
        dl_dphi = grad_output[m:]
        dphi_dz = -self.jac_inv @ self.jac[:,:m]    
        return dl_dz + dl_dphi @ Tensor(dphi_dz).cpu()
    

class correction:
    def __init__(self, n, ineq_resid):
        self.n = n
        self.ineq_resid = ineq_resid

    def forward(self, y):
        pass

    def backward(self, grad_output):
        pass