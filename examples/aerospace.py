from tinygrad.tensor import Tensor
from tinygrad.lazy import Device
import tinygrad.nn.optim as optim

from tiny_dc3 import Variable, dc3, soft_loss

Device.DEFAULT = 'CPU'

samples = 1000
length = 3

var = Variable(length)
x = var[0,None]
y = var[1,None]
s = var[2,None]

per_max = 2.5
area_max = 1
x_min = 0.1
y_min = 0.1

a = Tensor.rand(samples,1)*0.02 + 0.04 # height of rectangle
b = Tensor.rand(samples,1)*0.1 + 0.6   # width of rectangle
data = (Tensor.cat(a, b, dim=1) - [[.05,.65]]) / Tensor([[.02,.1]])

eq = [x**2 + y**2 - s**2]
ineq = [x*a/y + y*a/x - s + b, x+y+s-per_max, x*y/2-area_max, x_min-x, y_min-y, max(x_min,y_min)-s]
obj = y/x

class TinyNet:
    def __init__(self, ins, outs):
        self.l1 = Tensor.glorot_uniform(ins, 128)
        self.l2 = Tensor.glorot_uniform(128, outs)

    def forward(self, x):
        return x.dot(self.l1).gelu().dot(self.l2).sigmoid() * per_max / 2
    
model = dc3(length, eq, ineq, newton_iters=50, corr_iters=5, corr_lr=1e-3, corr_momentum=0.9, corr_clip_grad_norm=100)
net = TinyNet(2, 2)
opt = optim.AdamW([net.l1, net.l2], lr=1e-3)
lossf = soft_loss(obj, eq, ineq, 5, 5)

for i in range(10):
    x = net.forward(data)
    y = model.forward(x)
    opt.zero_grad()
    loss = lossf(y)
    loss.backward()
    opt.step()