A = 1
B = 100


def rosenbrock(xs):
    x, y = xs
    return (A - x) ** 2 + B * (y - x * x) ** 2
