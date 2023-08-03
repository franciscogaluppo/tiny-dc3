import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim

def jacobian(y, eq_resid):
    y = Tensor(y.numpy(), requires_grad=True).cpu()
    jac = np.zeros((len(eq_resid), *y.shape), np.float32)
    opt = optim.SGD([y],lr=1)
    for i, f in enumerate(eq_resid):
        opt.zero_grad()
        f(y).backward()
        jac[i] = y.grad.numpy()
    return jac, np.linalg.inv(jac[:,-len(eq_resid):])

def newton(z, eq_resid, tol=1e-5, max_iters=50):
    x = Tensor.zeros(len(eq_resid), requires_grad=True).cpu()
    for _ in range(max_iters):
        y = Tensor.cat(z, x, dim=0)
        h = Tensor.cat(*[f(y) for f in eq_resid], dim=0)
        jac, jac_inv = jacobian(y, eq_resid)
        delta = Tensor(jac_inv).cpu() @ h[:,None]
        x = x - delta[:,0]

        if delta.pow(2).sum().sqrt().numpy()[0] < tol:
            break

    return y, jac, jac_inv