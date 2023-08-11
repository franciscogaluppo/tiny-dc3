from tinygrad.tensor import Tensor

def keep_backward(self):
    assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

    # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
    # this is "implicit gradient creation"
    self.grad = Tensor(1, device=self.device, requires_grad=False)

    # Clean grads
    for t0 in reversed(self.deepwalk()):
      if not t0.requires_grad:
        del t0._ctx # TODO: does it help to delete this here ever?
        continue
      assert (t0.grad is not None)
      grads = t0._ctx.backward(t0.grad.lazydata)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = Tensor.zeros(t.shape)

    # Compute grads
    for t0 in reversed(self.deepwalk()):
      if not t0.requires_grad:
        del t0._ctx # TODO: does it help to delete this here ever?
        continue
      assert (t0.grad is not None)
      grads = t0._ctx.backward(t0.grad.lazydata)
      grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
        for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
      for t, g in zip(t0._ctx.parents, grads):
        if g is not None and t.requires_grad:
          assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
          t.grad = g if t.grad is None else (t.grad + g)