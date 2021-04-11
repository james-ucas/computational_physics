from math import nan


def bisection(f, a, b, tol):
    """
    Using the bisection method,
    find a zero of the function f to accuracy tol on the bracket (a,b).

    If (a,b) is not a bracket, return nan.

    :param callable f: function
    :param float a: left-hand-side of the bracket
    :param float b: left-hand-side of the bracket
    :param float tol: tightness of convergence
    :return float: c, the coordinate of the zero if found, otherwise nan
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return nan
    while abs(a - b) > tol:
        c = (a + b) / 2
        fc = f(c)
        if fc * fb < 0:
            a, fa = (c, fc)
        else:
            b, fb = (c, fc)
    return (a + b) / 2
