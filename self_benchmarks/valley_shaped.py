def three_hump_camel(x):
    return 2*x[0]**2 - 1.05*x[0]**4 + x[0]**6/6 + x[0]*x[1] + x[1]**2


def six_hump_camel(x):
    return (4 - 2.1*x[0]**2 + x[0]**4/3) * x[0]**2 + x[0]*x[1] + (-4 + 4*x[1]**2)*x[1]**2


def dixon_price(x):
    return (x[0] - 1)**2 + 2*(2*x[1]**2 - x[0])**2


def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2
