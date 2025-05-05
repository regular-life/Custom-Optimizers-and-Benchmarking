import torch
from torch.optim.optimizer import Optimizer

class RpropWithPolyakAveraging(Optimizer):
    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50), momentum=0.0):
        defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes, momentum=momentum)
        super(RpropWithPolyakAveraging, self).__init__(params, defaults)
        self.averaged_params = [p.clone().detach() for group in self.param_groups for p in group['params']]
        self.step_count = 0

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.step_count += 1
        idx = 0

        for group in self.param_groups:
            eta_minus, eta_plus = group['etas']
            step_min, step_max = group['step_sizes']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if 'step_size' not in state:
                    state['step_size'] = torch.full_like(p.data, group['lr'])
                    state['prev_grad'] = torch.zeros_like(p.data)
                    state['velocity'] = torch.zeros_like(p.data)

                step_size = state['step_size']
                prev_grad = state['prev_grad']
                grad_sign = grad.sign()
                prev_sign = prev_grad.sign()
                sign_prod = grad_sign * prev_sign

                pos_mask = sign_prod > 0
                neg_mask = sign_prod < 0

                step_size[pos_mask] = torch.clamp(step_size[pos_mask] * eta_plus, max=step_max)
                step_size[neg_mask] = torch.clamp(step_size[neg_mask] * eta_minus, min=step_min)
                grad[neg_mask] = 0

                v = state['velocity']
                v.mul_(momentum).add_(grad)
                p.data.addcmul_(v.sign(), step_size, value=-1)

                state['prev_grad'] = grad.clone()

                self.averaged_params[idx] = self.averaged_params[idx] * ((self.step_count - 1) / self.step_count) + p.data / self.step_count
                idx += 1

        return loss

    def get_averaged_parameters(self):
        return self.averaged_params