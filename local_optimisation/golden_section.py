PHI = (1 + 5 ** 0.5) / 2


def divide_interval(a, b):
    x = (b - a) / PHI
    return b - x, a + x


def golden_section(f, a, b, tol):
    """
    Using the golden section method,
    find an infinimum of the function f in the open interval (a,b),
    converging to a tolerance less than tol.

    Note this method is not guaranteed to find a local minimum
    unless the interval contains the minimum and is locally convex at the minimum.

    :param callable f: the function whose infima are to be found
    :param float a: the left-hand side of the interval
    :param float b: the right-hand side of the interval
    :param float tol: tightness of convergence
    :return float: c, the coordinate of the stationary point if found, otherwise nan
    """
    c, d = divide_interval(a, b)
    fa, fb, fc, fd = map(f, (a, b, c, d))

    while abs(c - d) > tol:
        if fc < fd:
            a, c, d, b = (a, d - (d - a) / PHI, c, d)
            fa, fc, fd, fb = fa, f(c), fc, fd
        else:
            a, c, d, b = (c, d, c - (c - b) / PHI, b)
            fa, fc, fd, fb = fc, fd, f(d), fb
    return (c + d) / 2
