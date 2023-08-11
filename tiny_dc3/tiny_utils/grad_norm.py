def clip_grad_norm(opt, value):
    norm = sum(p.grad.pow(2).sum() for p in opt.params).sqrt()
    if norm.numpy() > value:
        for p in opt.params:
            p.grad = p.grad / norm * value