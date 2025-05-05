import torch
import math


def bohachevsky(x):
    term1 = x[0]**2 + 2 * x[1]**2
    term2 = -0.3 * torch.cos(3 * math.pi * x[0])
    term3 = -0.4 * torch.cos(4 * math.pi * x[1])
    return term1 + term2 + term3 + 0.7


def perm_0(x):
    beta = 0
    d = 2
    result = 0.0
    for i in range(1, d+1):
        sum_term = 0.0
        for j in range(1, d+1):
            j_float = j
            j_tensor = torch.tensor(j_float, dtype=x.dtype, device=x.device)
            term = (j_float**i + beta) * ((x[j-1]/j_tensor)**i - 1)
            sum_term += term
        result += sum_term ** 2
    return result


def rotated_hyper_ellipsoid(x):
    sum1 = x[0]**2
    sum2 = (x[0] + x[1])**2
    return sum1 + sum2


def sphere(x):
    return torch.sum(x**2)


def sum_different_powers(x):
    exponents = 2 * \
        (torch.arange(x.size(0), device=x.device, dtype=x.dtype) + 2)
    return torch.sum(x**exponents)


def sum_squares(x):
    indices = torch.arange(1, x.size(0)+1, device=x.device, dtype=x.dtype)
    return torch.sum(indices * x**2)


def trid(x):
    sum1 = torch.sum((x - 1)**2)
    sum2 = torch.sum(x[1:] * x[:-1]) if x.size(0) >= 2 else 0.0
    return sum1 - sum2
