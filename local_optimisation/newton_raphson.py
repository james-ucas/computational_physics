from math import nan

import numpy as np

from matrices import matrix_inverse
from roots.newton_raphson import newton_raphson_step


def newton_raphson_step_ensure_minumum(df, x0, d2f):
    return x0 - df(x0) / abs(d2f(x0))


def newton_raphson(f, x0, df, d2f, tol, ensure_minimum=False):
    """
    Using the Newton--Raphson method,
    find a stationary point of the function f to accuracy tol with starting point x0.

    If ensure_minimum is true, the algorithm will try to find a minimum;
    in certain circumstances, the algorithm may enter into a cycle and fail to converge,
    upon which nan is returned.

    :param callable f: the function whose stationary points are to be found
    :param float x0: the starting point
    :param callable df: the first derivative of f
    :param callable d2f: the second derivative of f
    :param float tol: tightness of convergence
    :param bool ensure_minimum: only step toward a minimum
    :return float: x1, the coordinate of the stationary point if found, otherwise nan
    """
    values = {x0}

    step_function = newton_raphson_step_ensure_minumum if ensure_minimum else newton_raphson_step

    while abs((x1 := step_function(df, x0, d2f)) - x0) > tol:
        if x1 in values:
            return nan
        values.add(x1)
        x0 = x1
    return x1


def newton_raphson_step_multi(df, x0, d2f):
    df0 = df(x0)
    d2f0i = matrix_inverse(d2f(x0))
    return x0 - df0 @ d2f0i


def newton_raphson_multi(f, x0, df, d2f, tol):
    step_function = newton_raphson_step_multi
    while np.linalg.norm((x1 := step_function(df, x0, d2f)) - x0) > tol:
        x0 = x1
    return x1
