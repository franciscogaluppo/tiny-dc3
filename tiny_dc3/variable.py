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
    
    def min(self, axis=0): # BAD BEHAVIOR
        c = Variable(self.length)
        c.transform = lambda x: self(x).min(axis=axis+1)
        return c
    
    def max(self, axis=0): # BAD BEHAVIOR
        c = Variable(self.length)
        c.transform = lambda x: self(x).max(axis=axis+1)
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