import torch
from torch.optim import Optimizer

class ProxYogi(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3, prox=None):
        if lr <= 0:
            raise ValueError(f"Invalid lr: {lr}")
        beta1, beta2 = betas
        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1: {beta1}")
        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2: {beta2}")
        if prox is None or not callable(prox):
            raise ValueError("A callable `prox(param, lr)` must be provided")

        defaults = dict(lr=lr, betas=betas, eps=eps, prox=prox)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['m'] = torch.zeros_like(p.data)
                state['v'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2), eps, prox = (
                group['lr'], group['betas'], group['eps'], group['prox']
            )

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                m = state['m']
                v = state['v']

                # Yogi moment updates
                m.mul_(beta1).add_(grad, alpha=(1 - beta1))

                grad2 = grad.mul(grad)
                diff = v - grad2
                v.addcmul_(- (1 - beta2), grad2, diff.sign())

                # Gradient step
                update = m.div(v.sqrt().add(eps)).mul(lr)
                u = p.data - update

                # Proximal step: p <- prox(u; lr)
                p.data.copy_(prox(u, lr))

        return loss
