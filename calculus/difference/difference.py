def first_forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h


def first_backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h


def first_central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / 2 / h


def iterative_difference(f, x, h, tol, difference_function=first_central_difference):
    """

    Estimate the derivative of f at x using the specified difference function
    by iteratively reducing the initial step size h until
    | Delta(h) - Delta(h/2) | < tol .

    If the ratio h/x becomes so small that x+h==x, raise a ValueError reporting
    floating point underflow.

    :param f: function whose derivative is required
    :type f: callable, call signature f(x)
    :param x: point at which the function derivative is required
    :type x: float
    :param h: initial step size
    :type h: float
    :param tol: desired precision
    :type tol: float
    :param difference_function: function to calculate the derivative
    :type difference_function: callable, call signature g(f, x, h)
    :return: derivative to required precision, if found
    :rtype: float
    """
    f0 = difference_function(f, x, h)
    f1 = 1e10

    while abs(f0 - f1) > tol:
        h /= 2
        if x+h == x:
            raise ValueError('desired tolerance cannot be reached due to floating point underflow'
                             f' (current step size: {h})')
        f0, f1 = f1, difference_function(f, x, h)
    return f1
