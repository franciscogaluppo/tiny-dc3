from tinygrad.tensor import Tensor
import numpy as np

class Variable:
    def __init__(self, length):
        self.length = length
        def func(x):
            if len(x.shape) == 1:
                x = x[None,...]
            x = x.reshape(x.shape[0],-1)
            if x.shape[1] == self.length:
                return x
            else:
                raise Exception("Shape Mismatch")
        self.transform = func

    def __str__(self):
        return f"Variable with size {self.length}"

    def __call__(self, x):
        return self.transform(x)

    def __add__(self, b):
        c = Variable(self.length)
        if isinstance(b, Variable):
            c.transform = lambda x: self.transform(x) + b.transform(x)
        else:
            c.transform = lambda x: self.transform(x) + b
        return c

    def __radd__(self, b):
        return self.__add__(b)

    def __sub__(self, b):
        c = Variable(self.length)
        if isinstance(b, Variable):
            c.transform = lambda x: self.transform(x) - b.transform(x)
        else:
            c.transform = lambda x: self.transform(x) - b
        return c

    def __rsub__(self, b):
        c = Variable(self.length)
        if isinstance(b, Variable):
            c.transform = lambda x: b.transform(x) - self.transform(x)
        else:
            c.transform = lambda x: b - self.transform(x)
        return c
    
    def __neg__(self):
        c = Variable(self.length)
        c.transform = lambda x: -self(x)
        return c

    def __mul__(self, b):
        c = Variable(self.length)
        if isinstance(b, Variable):
            c.transform = lambda x: self(x) * b(x)
        else:
            c.transform = lambda x: self(x) * b
        return c

    def __rmul__(self, b):
        return self.__mul__(b)

    def __truediv__(self, b):
        c = Variable(self.length)
        if isinstance(b, Variable):
            c.transform = lambda x: self(x) / b(x)
        else:
            c.transform = lambda x: self(x) / b
        return c

    def __rtruediv__(self, b):
        c = Variable(self.length)
        if isinstance(b, Variable):
            c.transform = lambda x: b(x) / self(x)
        else:
            c.transform = lambda x: b / self(x)
        return c

    def __pow__(self, b):
        c = Variable(self.length)
        c.transform = lambda x: self(x) ** b
        return c

    def __matmul__(self, b):
        c = Variable(self.length)
        if isinstance(b, Variable):
            c.transform = lambda x: self(x) @ b(x)
        else:
            c.transform = lambda x: self(x) @ b
        return c

    def __rmatmul__(self, b):
        c = Variable(self.length)
        if isinstance(b, Variable):
            c.transform = lambda x: b(x) @ self(x)
        else:
            c.transform = lambda x: b @ self(x)
        return c

    def reshape(self, shape):
        c = Variable(self.length)
        c.transform = lambda x: self(x).reshape(-1, *shape)
        return c
    
    def min(self, axis=None):
        c = Variable(self.length)
        if axis is None:
            def transform(x):
                axis = tuple(y for y in range(1, len(x.shape)))
                return self(x).min(axis=axis)
            c.transform = transform
        elif isinstance(axis, tuple):
            axis = tuple(x+1 for x in axis)
            c.transform = lambda x: self(x).min(axis=axis)
        else:
            c.transform = lambda x: self(x).min(axis=axis+1)
        return c
    
    def max(self, axis=None):
        c = Variable(self.length)
        if axis is None:
            def transform(x):
                axis = tuple(y for y in range(1, len(x.shape)))
                return self(x).max(axis=axis)
            c.transform = transform
        elif isinstance(axis, tuple):
            axis = tuple(x+1 for x in axis)
            c.transform = lambda x: self(x).max(axis=axis)
        else:
            c.transform = lambda x: self(x).max(axis=axis+1)
        return c
    
    def sum(self, axis=None):
        c = Variable(self.length)
        if axis is None:
            def transform(x):
                axis = tuple(y for y in range(1, len(x.shape)+1))
                return self(x).sum(axis=axis)
            c.transform = transform
        elif isinstance(axis, tuple):
            axis = tuple(x+1 for x in axis)
            c.transform = lambda x: self(x).sum(axis=axis)
        else:
            c.transform = lambda x: self(x).sum(axis=axis+1)
        return c
    
    def mean(self, axis=None):
        c = Variable(self.length)
        if axis is None:
            def transform(x):
                axis = tuple(y for y in range(1, len(x.shape)))
                return self(x).mean(axis=axis)
            c.transform = transform
        elif isinstance(axis, tuple):
            axis = tuple(x+1 for x in axis)
            c.transform = lambda x: self(x).mean(axis=axis)
        else:
            c.transform = lambda x: self(x).mean(axis=axis+1)
        return c
    
    def abs(self):
        c = Variable(self.length)
        c.transform = lambda x: self(x).abs()
        return c
    
    def sqrt(self):
        c = Variable(self.length)
        c.transform = lambda x: self(x).sqrt()
        return c
    
    def log(self):
        c = Variable(self.length)
        c.transform = lambda x: self(x).log()
        return c
    
    def transpose(self, ax1=1, ax2=0):
        c = Variable(self.length)
        c.transform = lambda x: self(x).transpose(ax1+1, ax2+1)
        return c

    def __getitem__(self, item):
        c = Variable(self.length)
        if isinstance(item, tuple):
            c.transform = lambda x: self(x)[:, *item]
        elif isinstance(item, np.ndarray):
            def transform(x):
                eval = self(x)
                return Tensor.cat(*[eval[:,int(x)][:,None] for x in item])
            c.transform = transform
        else:
            c.transform = lambda x: self(x)[:, item]
        return c

    def eval_shape(self):
        return self.transform(Tensor.zeros(self.length)).shape