import numpy as np
import sympy as xp


def lagrange_basis_polynomial_factory(xs, ys):
    """
    Creates a function that returns the value at a point x of the unique order (n-1) polynomial
    that passes through the (n) points zip(xs, ys).

    :param xs: x values
    :type xs: one-dimensional iterable
    :param ys: y values
    :type ys: one-dimensional iterable (same length as xs)
    :return: function that evaluates the polynomial
    :rtype: callable
    """

    xs = np.array([*xs]).reshape(-1, 1)
    ys = np.array([*ys]).reshape(-1, 1)
    order = xs.size
    xss = np.vstack(order * [xs]).flatten()
    mask = [False if i == j else True for i in range(order) for j in range(order)]
    xss = xss[mask].reshape(order - 1, order, 1)
    diff = (xs - xss).prod(0)

    def lagrange(x):
        shape = x.shape
        x = x.reshape(1, 1, x.size)
        values = (ys * ((x - xss).prod(0) / diff)).sum(0)
        return values.reshape(*shape)

    return lagrange


def lagrange_polynomials(x, xs):
    """
    return a list of Lagrange basis polynomials with dependent variable x

    :param Symbol x: dependent variable
    :param [Symbol] xs: list of points where the value of the polynomials are known
    :return [Symbol]: list of polynomials
    """
    return [np.product([(x - xj) / (xi - xj) for xj in xs if xi != xj]) for xi in xs]


def lagrange_integral(xs, fs, a, b):
    """
    Integrate the Lagrange polynomial defined by points xs and function values fs on the interval [a,b].

    :param [float] xs: points defining the polynomial
    :param [float] fs: function values defining the polynomial
    :param float a: lower limit of integration
    :param float b: upper limit of integration
    :return float: definite integral
    """
    x = xp.symbols('x')
    polys = lagrange_polynomials(x, xs)
    polynomial = sum(f * poly for (f, poly) in zip(fs, polys))
    return xp.integrate(polynomial, (x, a, b))


def get_newton_cotes_weights(n, h):
    """
    Get the Newton-Cotes weights using n points spaced h apart

    :param int n: number of points
    :param float or Symbol h: spacing
    :return [float] or [Symbol]: list of weights
    """
    x, xi = xp.symbols('x xi')
    xs = [xi + i * h for i in range(n)]

    polys = lagrange_polynomials(x, xs)
    integrals = [xp.integrate(poly, (x, xs[0], xs[-1])) for poly in polys]
    integrals = [xp.simplify(ipoly) for ipoly in integrals]
    return integrals
