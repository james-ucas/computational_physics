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
    y, y_t = y[:, np.newaxis], y[np.newaxis, :]
    s, s_t = s[:, np.newaxis], s[np.newaxis, :]
    if (s_t @ y) == 0:
        raise ValueError(f's_t @ y = {s_t} @ {y} = {s_t @ y}')
    elif (s_t @ y) < 0:
        y *= -1
    return (binv +
            (s_t @ y + y_t @ binv @ y) * (s @ s_t) / (s_t @ y) ** 2 -
            (binv @ y @ s_t + s @ y_t @ binv) / (s_t @ y))


def take_step(f, x0, df0, binv):
    p = (-binv @ df0.reshape(-1)).reshape(x0.shape)
    alpha = line_search(f, x0, p, df0)
    return x0 + alpha * p


def bfgs(f, x0, df, xtol=1e-5, gtol=1e-5):
    """
    Using the Broyden-Fletcher-Goldfarb-Shanno method,
    find a minimum of a function f with first derivative g
    in the region of the point x0.

    :param callable f: the objective function
    :param numpy.ndarray x0: the initial coordinates
    :param callable df: the first derivative of the objective function
    :param float xtol: the convergence criterion for the coordinates
    :param float gtol: the convergence criterion for the gradient
    :return numpy.ndarray: x, the coordinates of a local minimum

    """

    x0 = x0.copy()
    df0 = df(x0)
    b_inv = np.eye(x0.size)

    while any((norm((x1 := take_step(f, x0, df0, b_inv)) - x0) > xtol, norm(df1 := df(x1)) > gtol),):
        b_inv = bfgs_update_hessian(b_inv, (df1 - df0).reshape(-1), (x1 - x0).reshape(-1))
        x0, df0 = x1, df1
    return x1
