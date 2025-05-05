import torch
from torch.optim import Optimizer

class SGDPolyakMomentum(Optimizer):
    def __init__(self, params, lr, momentum=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

        self._step = 0

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum_buffer'] = torch.zeros_like(p.data)
                state['avg_param'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1
        t = self._step

        for group in self.param_groups:
            lr = group['lr']
            beta = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                m = state['momentum_buffer']
                avg_p = state['avg_param']

                # update momentum
                m.mul_(beta).add_(grad, alpha=(1.0 - beta))

                # update parameter
                p.data.add_(m, alpha=-lr)

                # update Polyak average
                avg_p.mul_((t-1)/t).add_(p.data, alpha=1.0/t)

        return loss

    def averaged_parameters(self):
        return [self.state[p]['avg_param'] for group in self.param_groups for p in group['params']]
