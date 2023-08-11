from tinygrad.tensor import Tensor
from tinygrad.lazy import Device
import tinygrad.nn.optim as optim

from tiny_dc3 import Variable, dc3, soft_loss

Device.DEFAULT = 'CPU'

samples = 500
assets = 10

w = Variable(assets)[:,None]
A = Tensor.rand(samples, assets, assets)
sig = A.transpose(1, 2) @ A
r = Tensor.rand(samples, assets) * 2
data = Tensor.cat(r, sig.reshape(samples, -1), dim=1)

eq = [w.sum()-1]
ineq = [1-(w[...,0]*r).sum(), w-1, -w]
obj = w.transpose() @ sig @ w / 2

class TinyNet:
    def __init__(self, ins, outs):
        self.l1 = Tensor.glorot_uniform(ins, 128)
        self.l2 = Tensor.glorot_uniform(128, outs)
        self.l3 = Tensor.glorot_uniform(128, 1)

    def forward(self, x):
        x = x.dot(self.l1).gelu()
        return x.dot(self.l2).softmax() * x.dot(self.l3).sigmoid()
    
model = dc3(assets, eq, ineq, newton_iters=50, corr_iters=10, corr_lr=1e-3, corr_momentum=0.9, corr_clip_grad_norm=100)
net = TinyNet(assets*assets+assets, assets-1)
opt = optim.AdamW([net.l1, net.l2, net.l3], lr=1e-3)
lossf = soft_loss(obj, eq, ineq, 1, 1000, pow=2)
losses = list()

for i in range(10):
    x = net.forward(data)
    y = model.forward(x)
    opt.zero_grad()
    loss = lossf(y)
    loss.backward()
    opt.step()