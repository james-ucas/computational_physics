def iteration(f, x0, tol):
    """
    Solve x = f(x) for x to within tolerance tol starting at x0.

    :param callable f: function s.t. x = f(x) at the solution
    :param callable x0: starting point
    :param float tol: tightness of convergence
    :return float: the approximate solution
    """
    while abs((x1 := f(x0)) - x0) > tol:
        x0 = x1
    return x1
