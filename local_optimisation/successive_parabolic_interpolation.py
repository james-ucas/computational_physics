from math import nan


def successive_parabolic_interpolation(f, a, b, tol):
    """
    Using the successive parabolic interpolation method,
    find a stationary point of a function f by iteratively finding
    the extremum of the parabola passing through (a,f(a)), (b,f(b)), (c,f(c))
    where c = (a+b)/2 .

    If f(a)==f(b)==f(c), then a stationary point cannot be found.

    :param callable f: the function whose stationary points are to be found
    :param float a: a starting point
    :param float b: another starting point
    :param float tol: tightness of convergence
    :return float: c, the coordinate of the stationary point if found, otherwise nan
    """
    c = (a + b) / 2
    fa, fb, fc = map(f, (a, b, c))

    if (a == b or b == c or c == a) or (fa == fb and fb == fc):
        return nan

    def get_minimum():
        ga = a * (fc - fb)
        gb = b * (fa - fc)
        gc = c * (fb - fa)
        return (a * ga + b * gb + c * gc) / (ga + gb + gc) / 2

    while abs(c - b) > tol:
        a, b, c = b, c, get_minimum()
        fa, fb, fc = fb, fc, f(c)
    return c
