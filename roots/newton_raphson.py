from math import nan


def newton_raphson_step(f, x0, df):
    return x0 - f(x0) / df(x0)


def newton_raphson(f, x0, df, tol):
    """
    Using the secant method,
    find a zero of the function f to accuracy tol with starting point x0.

    If there is a stationary point between x0 and the zero,
    the algorithm may enter into a cycle;
    if this happens, nan is returned.

    (x0, x1) need not bracket a zero;
    as such, the algorithm may not converge.

    :param callable f: the function whose zeros are to be found
    :param float x0: the starting point
    :param callable df: the first derivative of f
    :param float tol: tightness of convergence
    :return float: x1, the coordinate of the zero if found, otherwise nan
    """
    values = {x0}
    while abs((x1 := newton_raphson_step(f, x0, df)) - x0) > tol:
        if x1 in values:
            return nan
        values.add(x1)
        x0 = x1
    return x1
