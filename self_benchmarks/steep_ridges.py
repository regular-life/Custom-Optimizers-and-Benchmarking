import torch
import math


def de_jong_n5(x):
    a = torch.tensor([-32, -16, 0, 16, 32], dtype=x.dtype, device=x.device)
    a1, a2 = torch.meshgrid(a, a, indexing='ij')
    a1_flat = a1.reshape(-1)
    a2_flat = a2.reshape(-1)
    i_vals = torch.arange(1, 26, dtype=x.dtype, device=x.device)
    sum_terms = torch.sum(
        1.0 / (i_vals + (x[0] - a1_flat)**6 + (x[1] - a2_flat)**6))
    return 1.0 / (0.002 + sum_terms)


def easom(x):
    return -torch.cos(x[0]) * torch.cos(x[1]) * torch.exp(-(x[0] - math.pi)**2 - (x[1] - math.pi)**2)


def michalewicz(x):
    m = 10
    term1 = torch.sin(x[0]) * torch.sin(1 * x[0]**2 / math.pi) ** (2*m)
    term2 = torch.sin(x[1]) * torch.sin(2 * x[1]**2 / math.pi) ** (2*m)
    return -(term1 + term2)
