import numpy as np
from itertools import combinations_with_replacement


def first_forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h


def first_backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h


def first_central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / 2 / h


def second_central_difference(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h / h


def five_point_formula(f, x, h):
    return (f(x - 2 * h) - 8 * f(x - h) + 8 * f(x + h) - f(x + 2 * h)) / 12 / h


def first_central_multi(f, x, h):
    """
    Compute the first central differences approximation
    to the derivative of f at the point x using step size h.

    :param callable f: function f(x)
    :param ndarray x: point at which to evaluate the first derivative
    :param float h: step size
    :return array: approximate value of the first derivative
    """
    x_flat = x.reshape(-1)
    x_copy = x.copy()
    df = np.zeros_like(x)
    df_flat = df.reshape(-1)
    for i in range(x.size):
        x_flat[i] += h
        df_flat[i] += f(x)
        x_flat[i] -= 2 * h
        df_flat[i] -= f(x)
        x[:] = x_copy[:]
    df /= 2 * h
    return df


def second_central_multi(f, x, h):
    """
    Compute the first second differences approximation
    to the derivative of f at the point x using step size h.

    :param callable f: function f(x)
    :param ndarray x: point at which to evaluate the second derivative
    :param float h: step size
    :return array: approximate value of the second derivative
    """
    x_flat = x.reshape(-1)
    x_copy = x.copy()
    d2f = np.zeros((*x.shape, *x.shape))
    d2f_flat = d2f.reshape((x.size, x.size))
    for i, j in combinations_with_replacement(range(x.size), 2):
        x_flat[i] += h
        x_flat[j] += h
        d2f_flat[i, j] += f(x)  # f++
        x_flat[j] -= 2 * h
        d2f_flat[i, j] -= f(x)  # f+-
        x_flat[i] -= 2 * h
        x_flat[j] += 2 * h
        d2f_flat[i, j] -= f(x)  # f-+
        x_flat[j] -= 2 * h
        d2f_flat[i, j] += f(x)  # f--
        x[:] = x_copy[:]
        if i != j:
            d2f_flat[j, i] = d2f_flat[i, j]
    d2f /= 4 * h * h
    return d2f
