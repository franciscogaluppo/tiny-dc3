import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim
from functools import reduce

def jacobian(y, eq_resid):
    y = Tensor(y.numpy())
    const_len = [reduce(lambda x,y: x*y, f.eval_shape()) for f in eq_resid]
    jac = np.zeros((y.shape[0], sum(const_len), y.shape[1]), np.float32)
    
    for i, f in enumerate(eq_resid):
        for j in range(const_len[i]):
            opt = optim.SGD([y],lr=1)
            h = f(y) # EXPENSIVE
            opt.zero_grad()
            h[:,j].sum().backward()
            jac[:,(0 if i==0 else const_len[i-1])+j,:] = y.grad.numpy()
    
    return jac

def newton(z, eq_resid, tol=1e-5, max_iters=50):
    n_m = sum(reduce(lambda x,y: x*y, f.eval_shape()) for f in eq_resid)
    x = Tensor.rand(z.shape[0], n_m)
    for _ in range(max_iters):
        y = Tensor.cat(z, x, dim=1)
        h = Tensor.cat(*[f(y)[:,None] for f in eq_resid], dim=1).reshape(z.shape[0], -1)
        jac = jacobian(y, eq_resid)
        delta = Tensor(np.linalg.solve(jac[:,:,-n_m:], h.numpy()))
        x = x - delta

        if delta.pow(2).sum(axis=1).sqrt().mean().numpy() < tol:
            break

    return Tensor.cat(z, x, dim=1), jac, np.linalg.inv(jac[:,:,-n_m:])