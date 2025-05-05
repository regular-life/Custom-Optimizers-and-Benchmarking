import torch
from torch.optim import Optimizer

class ProximalHB(Optimizer):
    def __init__(self, params, lr, momentum=0.0, lambd=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if not 0.0 <= lambd:
            raise ValueError(f"Invalid λ: {lambd}")

        defaults = dict(lr=lr, momentum=momentum, lambd=lambd)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def _soft_threshold(self, u, alpha, lambd):
        return u.sign().mul_((u.abs() - alpha * lambd).clamp_min_(0.0))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr     = group['lr']
            beta   = group['momentum']
            lambd  = group['lambd']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                buf  = self.state[p]['momentum_buffer']

                # heavy‐ball momentum
                buf.mul_(beta).add_(grad)

                # gradient step + prox
                u = p.data - lr * buf
                p.data.copy_( self._soft_threshold(u, lr, lambd) )

        return loss
