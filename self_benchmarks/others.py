import torch
import math


def beale(x):
    term1 = (1.5 - x[0] + x[0]*x[1])**2
    term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
    term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return term1 + term2 + term3


def branin(x):
    a = 1
    b = 5.1/(4*math.pi**2)
    c = 5/math.pi
    r = 6
    s = 10
    t = 1/(8*math.pi)
    return a*(x[1] - b*x[0]**2 + c*x[0] - r)**2 + s*(1 - t)*torch.cos(x[0]) + s


def colville(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2 + 90*(x[3]-x[2]**2)**2 + \
        (1-x[2])**2 + 10.1*((x[1]-1)**2 + (x[3]-1)**2) + 19.8*(x[1]-1)*(x[3]-1)


def forrester(x):
    return (6*x - 2)**2 * torch.sin(12*x - 4)


def goldstein_price(x):
    term1 = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3 *
                                        x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
    term2 = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12 *
                                         x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
    return term1 * term2


def hartmann3d(x):
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.tensor([[3.0, 10, 30],
                      [0.1, 10, 35],
                      [3.0, 10, 30],
                      [0.1, 10, 35]])
    P = 1e-4 * torch.tensor([[3689, 1170, 2673],
                             [4699, 4387, 7470],
                             [1091, 8732, 5547],
                             [381, 5743, 8828]])
    return -torch.sum(alpha * torch.exp(-torch.sum(A * (x.unsqueeze(0) - P)**2, dim=1)))


def hartmann4d(x):
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.tensor([[10, 3, 17, 3.5],
                      [0.05, 10, 17, 0.1],
                      [3, 3.5, 1.7, 10],
                      [17, 8, 0.05, 10]])
    P = 1e-4 * torch.tensor([[1312, 1696, 5569, 124],
                             [2329, 4135, 8307, 3736],
                             [2348, 1451, 3522, 2883],
                             [4047, 8828, 8732, 5743]])
    return -torch.sum(alpha * torch.exp(-torch.sum(A * (x.unsqueeze(0) - P))**2))


def hartmann6d(x):
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
    A = torch.tensor([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
    P = 1e-4 * torch.tensor([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])
    return -torch.sum(alpha * torch.exp(-torch.sum(A * (x.unsqueeze(0) - P)**2, dim=1)))


def perm(x, d=4, beta=10):
    result = 0.0
    for i in range(1, d+1):
        sum_term = 0.0
        for j in range(1, d+1):
            sum_term += (j**i + beta) * ((x[j-1]/j)**i - 1)
        result += sum_term**2
    return result


def powell(x):
    return (x[0] + 10*x[1])**2 + 5*(x[2] - x[3])**2 + \
           (x[1] - 2*x[2])**4 + 10*(x[0] - x[3])**4


def shekel(x, m=10):
    beta = torch.tensor([[4, 4, 4, 4],
                        [1, 1, 1, 1],
                        [8, 8, 8, 8],
                        [6, 6, 6, 6],
                        [3, 7, 3, 7],
                        [2, 9, 2, 9],
                        [5, 5, 3, 3],
                        [8, 1, 8, 1],
                        [6, 2, 6, 2],
                        [7, 3.6, 7, 3.6]])
    c = torch.tensor([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    return -torch.sum(1 / (torch.sum((x - beta[:m])**2, dim=1) + c[:m]))


def styblinski_tang(x):
    return 0.5 * torch.sum(x**4 - 16*x**2 + 5*x)
