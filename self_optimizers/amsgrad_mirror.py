import torch
from torch.optim import Optimizer

class AMSGradMirror(Optimizer):
    def __init__(self, params, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid lr: {lr}")
        b1, b2 = betas
        if not 0.0 <= b1 < 1.0:
            raise ValueError(f"Invalid beta1: {b1}")
        if not 0.0 <= b2 < 1.0:
            raise ValueError(f"Invalid beta2: {b2}")
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, (beta1, beta2), eps = group['lr'], group['betas'], group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                max_exp_avg_sq = state['max_exp_avg_sq']

                # biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # Maintain the maxm of all 2nd moment running avg till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)

                state['step'] += 1

                # Compute the mirror‐descent (AMSGrad) update
                denom = max_exp_avg_sq.sqrt().add_(eps)
                step_size = lr

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
