import torch
import math


def ackley(x):
    term1 = -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x[0]**2 + x[1]**2))
                            )
    term2 = -torch.exp(0.5 * (torch.cos(2 * math.pi *
                       x[0]) + torch.cos(2 * math.pi * x[1])))
    return term1 + term2 + math.e + 20


def bukin_n6(x):
    term1 = 100 * torch.sqrt(torch.abs(x[1] - 0.01 * x[0]**2))
    term2 = 0.01 * torch.abs(x[0] + 10)
    return term1 + term2


def cross_in_tray(x):
    exponent = torch.abs(100 - (torch.sqrt(x[0]**2 + x[1]**2) / math.pi))
    term_inner = torch.sin(x[0]) * torch.sin(x[1]) * torch.exp(exponent)
    term_abs = torch.abs(term_inner)
    return -0.0001 * (term_abs + 1) ** 0.1


def drop_wave(x):
    r = torch.sqrt(x[0]**2 + x[1]**2)
    numerator = 1 + torch.cos(12 * r)
    denominator = 0.5 * r**2 + 2
    return - numerator / denominator


def eggholder(x):
    term1 = -(x[1] + 47) * torch.sin(torch.sqrt(torch.abs(x[0]/2 + x[1] + 47)))
    term2 = -x[0] * torch.sin(torch.sqrt(torch.abs(x[0] - (x[1] + 47))))
    return term1 + term2


def gramacy_lee(x):
    return torch.sin(10 * math.pi * x) / (2 * x) + (x - 1)**4


def griewank(x):
    term1 = (x[0]**2 + x[1]**2) / 4000
    term2 = torch.cos(x[0]) * torch.cos(x[1] / math.sqrt(2))
    return term1 - term2 + 1


def holder_table(x):
    term = torch.sin(x[0]) * torch.cos(x[1]) * \
        torch.exp(torch.abs(1 - (torch.sqrt(x[0]**2 + x[1]**2) / math.pi)))
    return -torch.abs(term)


def langermann(x):
    A = torch.tensor([[3, 5], [5, 2], [2, 1], [1, 4],
                     [7, 9]], dtype=torch.float32)
    c = torch.tensor([1, 2, 5, 2, 3], dtype=torch.float32)
    result = 0.0
    for i in range(5):
        diff = x - A[i]
        squared = torch.sum(diff ** 2)
        exponent = -squared / math.pi
        cosine = torch.cos(math.pi * squared)
        term = c[i] * torch.exp(exponent) * cosine
        result += term
    return result


def levy_n13(x):
    term1 = torch.sin(3 * math.pi * x[0]) ** 2
    term2 = (x[0] - 1)**2 * (1 + torch.sin(3 * math.pi * x[1]) ** 2)
    term3 = (x[1] - 1)**2 * (1 + torch.sin(2 * math.pi * x[0]) ** 2)
    return term1 + term2 + term3


def levy(x):
    w = 1 + (x - 1) / 4
    term1 = torch.sin(math.pi * w[0]) ** 2
    term2 = (w[0] - 1)**2 * (1 + 10 * torch.sin(math.pi * w[0] + 1)**2)
    term3 = (w[1] - 1)**2 * (1 + torch.sin(2 * math.pi * w[1])**2)
    return term1 + term2 + term3


def rastrigin(x):
    return 10 * len(x) + torch.sum(x**2 - 10 * torch.cos(2 * math.pi * x))


def schaffer_n2(x):
    num = torch.sin(x[0]**2 - x[1]**2)**2 - 0.5
    den = (1 + 0.001 * (x[0]**2 + x[1]**2)) ** 2
    return 0.5 + num / den


def schaffer_n4(x):
    num = torch.cos(torch.sin(torch.abs(x[0]**2 - x[1]**2))) ** 2 - 0.5
    den = (1 + 0.001 * (x[0]**2 + x[1]**2)) ** 2
    return 0.5 + num / den


def schwefel(x):
    return -torch.sum(x * torch.sin(torch.sqrt(torch.abs(x))))


def shubert(x):
    sum_x = torch.sum(torch.tensor(
        [i * torch.cos((i+1)*x[0] + i) for i in range(1, 6)]))
    sum_y = torch.sum(torch.tensor(
        [i * torch.cos((i+1)*x[1] + i) for i in range(1, 6)]))
    return sum_x * sum_y
