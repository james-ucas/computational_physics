import numpy as np
from numpy.linalg import norm

from .line_search import line_search


def bfgs_update_hessian(binv, y, s):
    """
    Update the approximate inverse Hessian matrix in the BFGS scheme
    using the Sherman-Morrison formula

    :param numpy.ndarray binv: the approximate inverse Hessian at step i
    :param numpy.ndarray  y: the difference between gradient at step i+1 and step i
    :param numpy.ndarray s: the BFGS coordinate update at step i
    :return numpy.ndarray : the approximate inverse Hessian at step i+1

    """
    y, yT = y[:, np.newaxis], y[np.newaxis, :]
    s, sT = s[:, np.newaxis], s[np.newaxis, :]
    if (sT @ y) < 0:
        raise ValueError(f'{sT}, {y}')
    return (binv +
            (sT @ y + yT @ binv @ y) * (s @ sT) / (sT @ y) ** 2 -
            (binv @ y @ sT + s @ yT @ binv) / (sT @ y))


def take_step(f, x0, df0, binv):
    p = -binv @ df0
    alpha = line_search(f, x0, p, df0)
    return x0 + alpha * p


def bfgs(f, x0, df, tol):
    """
    Using the Broyden-Fletcher-Goldfarb-Shanno method,
    find a minimum of a function f with first derivative g
    in the region of the point x0.

    :param callable f: the objective function
    :param numpy.ndarray x0: the initial coordinates
    :param callable df: the first derivative of the objective function
    :param float tol: the convergence criterion
    :return numpy.ndarray: x, the coordinates of a local minimum

    """

    x0 = x0.copy()
    df0 = df(x0)
    binv = np.eye(x0.size)

    while norm((x1 := take_step(f, x0, df0, binv)) - x0) > tol and norm(df1 := df(x1)) > tol:
        binv = bfgs_update_hessian(binv, df1 - df0, x1 - x0)
        x0, df0 = x1, df1
    return x1
