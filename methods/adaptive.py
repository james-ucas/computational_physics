from calculus.difference import first_central_difference, second_central_difference


def adaptive_first(f, x, h, tol):
    """
    Converge the central differences approximation to the first derivative of f at x
    to within an error tol by iteratively halving the step size h.

    :param callable f: function f(x, h) whose first derivative is to found evaluated
    :param float x: point at which the derivative is evaluated
    :param float h: initial step size
    :param float tol: tolerance

    :raises ValueError: if floating-point underflow occurs (when h/x < eps)

    :return float: derivative
    """
    h, df0, df1, = h / 2, first_central_difference(f, x, h), first_central_difference(f, x, h / 2)
    while h * h * abs(df1 - df0) > tol:
        h, df0, df1 = h / 2, df1, first_central_difference(f, x, h / 2)
    if df1 == df0:
        raise ValueError("floating point underflow occurred; consider increasing the value of tol")
    return (4 * df1 - df0) / 3


def adaptive_second(f, x, h, tol):
    """
    Converge the central differences approximation to the second derivative of f at x
    to within an error tol by iteratively halving the step size h.

    :param callable f: function f(x, h) whose second derivative is to found evaluated
    :param float x: point at which the derivative is evaluated
    :param float h: initial step size
    :param float tol: tolerance

    :raises ValueError: if floating-point underflow occurs (when h/x < eps)

    :return float: derivative
    """
    h, df0, df1, = h / 2, second_central_difference(f, x, h), second_central_difference(f, x, h / 2)
    while h * h * abs(df1 - df0) > tol:
        h, df0, df1 = h / 2, df1, second_central_difference(f, x, h / 2)
    if df1 == df0:
        raise ValueError("floating point underflow occurred; consider increasing the value of tol")
    return (4 * df1 - df0) / 3
