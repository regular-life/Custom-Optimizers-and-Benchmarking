import torch


def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2


def matyas(x):
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]


def mccormick(x):
    return torch.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5*x[0] + 2.5*x[1] + 1


def power_sum(x):
    b = torch.tensor([8, 18], dtype=x.dtype, device=x.device)
    return (x[0] + x[1] - b[0])**2 + (x[0]**2 + x[1]**2 - b[1])**2


def zakharov(x):
    sum1 = torch.sum(x**2)
    sum2 = torch.sum(0.5 * torch.arange(1, len(x)+1, device=x.device) * x)
    return sum1 + sum2**2 + sum2**4
