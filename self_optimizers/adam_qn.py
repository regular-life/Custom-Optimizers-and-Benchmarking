import torch
from torch.optim.optimizer import Optimizer

class AdamQN(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, gamma=0.1, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, gamma=gamma, eps=eps)
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['m'] = torch.zeros_like(p.data)
                state['v'] = torch.zeros_like(p.data)
                state['h'] = torch.ones_like(p.data)
                state['prev_grad'] = torch.zeros_like(p.data)
                state['prev_param'] = p.data.clone()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['beta1'], group['beta2']
            gamma, eps = group['gamma'], group['eps']
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data
                m, v = state['m'], state['v']
                h = state['h']
                prev_grad, prev_param = state['prev_grad'], state['prev_param']
                step = state['step'] + 1
                if step > 1:
                    y = grad - prev_grad
                    s = p.data - prev_param
                    h_update = s / (y + eps)
                    h.mul_(1 - gamma).add_(gamma * h_update)
                precond_grad = grad / (h + eps)
                # Adam update
                m.mul_(beta1).add_(precond_grad, alpha=1 - beta1)
                v.mul_(beta2).add_(precond_grad.pow(2), alpha=1 - beta2)
                # Bias correction
                m_hat = m / (1 - beta1**step)
                v_hat = v / (1 - beta2**step)
                # Parameter update
                p.data.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)
                # Save state for next step
                state['prev_grad'].copy_(grad)
                state['prev_param'].copy_(p.data)
                state['step'] = step