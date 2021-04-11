import numpy as np


def rectangle(f, a, b, n=1):
    """

    Integrate the function f on the interval [a,b] using the rectangle method
    with n sub-intervals of 2 points, i.e., n+1 points including [a,b]

    :param f: integrand, callable
    :param a: lower limit, float
    :param b: upper limit, float
    :param n: number of sub-intervals between the limits
    :return: the integral, float
    """

    xs = np.linspace(a, b, n + 1)
    h = xs[1] - xs[0]
    fxs = f(xs[:-1])
    return (h * fxs).sum()


def trapezium(f, a, b, n=1):
    """

    Integrate the function f on the interval [a,b] using the trapezium method
    with n sub-intervals of 2 points, i.e., n+2 points including [a,b]

    :param f: integrand, callable
    :param a: lower limit, float
    :param b: upper limit, float
    :param n: number of sub-intervals between the limits
    :return: the integral, float
    """

    xs = np.linspace(a, b, n + 1)
    h = xs[1] - xs[0]
    fxs = f(xs)
    return (h / 2 * (fxs[1:] + fxs[:-1])).sum()


def simpson(f, a, b, n=1):
    """

    Integrate the function f on the interval [a,b] using Simpson's method
    with sub-intervals of 3 points, i.e., 2*n+1 points including [a,b]


    :param f: integrand, callable
    :param a: lower limit, float
    :param b: upper limit, float
    :param n: number of sub-intervals between the limits
    :return: the integral, float
    """
    xs = np.linspace(a, b, 2 * n + 1)
    h = xs[1] - xs[0]
    fxs = f(xs)
    return (h / 3 * (fxs[0:-2:2] + 4 * fxs[1:-1:2] + fxs[2::2])).sum()
