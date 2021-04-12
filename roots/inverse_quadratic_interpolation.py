from math import inf, nan


def inverse_quadratic_interpolation(f, a, b, tol):
    """
    Using the inverse quadratic interpolation,
    find a zero of the function f to accuracy tol with starting points (a,b).

    (a, b) need not bracket a zero.

    Each iteration of the algorithm requires three points, (a, b, c);
    on the first iteration, the third point, c, is chosen as the arithmetic mean of (a,b).

    Fails whenever any two of f(a), f(b), f(c) are equal (no inverse).

    :param callable f: the function whose zeros are to be found
    :param float a: the first starting point
    :param float b: the second starting point
    :param float tol: tightness of convergence
    :return float: d, the coordinate of the zero if found, otherwise nan
    """
    dx = inf
    c = (a + b) / 2
    d = nan
    fa, fc, fb = f(a), f(c), f(b)
    while abs(dx) > tol:
        if not fa == fb or fb == fc or fc == fa:
            return nan
        d = (a * fc * fb / (fa - fc) / (fa - fb) +
             c * fa * fb / (fc - fa) / (fc - fb) +
             b * fa * fc / (fb - fa) / (fb - fc))
        a, c, b = c, b, d
        fa, fc, fb = fc, fb, f(d)
        dx = b - a
    return d
