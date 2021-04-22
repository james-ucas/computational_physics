import numpy as np

A = 1
B = 100


def rosenbrock(xs):
    x, y = xs
    return (A - x) ** 2 + B * (y - x * x) ** 2


def rosenbrock_derivative(xs):
    x, y = xs
    return np.array([-2 * (A - x) - 4 * x * B * (y - x * x), 2 * B * (y - x * x)])


def rosenbrock_second_derivative(xs):
    x, y = xs
    return np.array([[2 - 4 * B * y + 12 * B * x * x, -4 * x * B], [-4 * x * B, 2 * B]])
