import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
from functools import reduce
from .tiny_utils import keep_backward

def jacobian(y, eq_resid, const_len):
    y = Tensor(y.numpy())
    jac = np.zeros((y.shape[0], sum(const_len), y.shape[1]), np.float32)
    opt = optim.SGD([y],lr=1)
    
    for i, f in enumerate(eq_resid):
        h = f(y)
        if len(h.shape) == 1:
            h = h[:,None]
        for j in range(const_len[i]):
            keep_backward(h[:,j].sum())
            jac[:,(0 if i==0 else const_len[i-1])+j,:] = y.grad.numpy()
    
    return jac

def newton(z, eq_resid, const_len, tol=1e-5, max_iters=50):
    n_m = sum(const_len)
    x = Tensor.rand(z.shape[0], n_m)
    for _ in range(max_iters):
        y = Tensor.cat(z, x, dim=1)
        h = Tensor.cat(*[f(y)[:,None] for f in eq_resid], dim=1).reshape(z.shape[0], -1)
        jac = jacobian(y, eq_resid, const_len)
        delta = Tensor(np.linalg.solve(jac[:,:,-n_m:], h.numpy()))
        x = x - delta

        if delta.pow(2).sum(axis=1).sqrt().mean().numpy() < tol:
            break

    return Tensor.cat(z, x, dim=1), jac, np.linalg.inv(jac[:,:,-n_m:])