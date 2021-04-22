import numpy as np


def quartic_double_well(xs):
    x, y = xs
    return x ** 4 - x * x + y * y


def quartic_double_well_derivative(xs):
    x, y = xs
    return np.array([4 * x ** 3 - 2 * x, 2 * y])


def quartic_double_well_second_derivative(xs):
    x, y = xs
    return np.array([[12 * x * x - 2, 0], [0, 2]])
