def secant(f, x0, x1, tol):
    """
    Using the secant method,
    find a zero of the function f to accuracy tol with starting points (x0,x1).

    (x0, x1) need not bracket a zero;
    as such, the algorithm may not converge.

    :param callable f: the function whose zeros are to be found
    :param float x0: the first starting point
    :param float x1: the second starting point
    :param float tol: tightness of convergence
    :return float: x2, the coordinate of the zero if found, otherwise nan
    """
    f0, f1 = f(x0), f(x1)

    def secant_step():
        return x1 - f1 * (x1 - x0) / (f1 - f0)

    while abs((x2 := secant_step()) - x1) > tol:
        x0, x1 = x1, x2
        f0, f1 = f1, f(x1)

    return x2
