from numpy.linalg import norm

from .line_search import line_search


def gradient_descent(f, x0, df, tol):
    """
    Using the steepest descent method,
    find a minimum of a function f with first derivative g
    in the region of the point x0.

    :param callable f: the objective function
    :param numpy.ndarray x0: the initial coordinates
    :param callable df: the first derivative of the objective function
    :param float tol: the convergence criterion
    :return numpy.ndarray: x, the coordinates of a local minimum

    """

    def gradient_descent_step():
        df0 = df(x0)
        p = -df0 / norm(df0)
        alpha = line_search(f, x0, p, df0)
        return x0 + alpha * p

    while norm((x1 := gradient_descent_step()) - x0) > tol:
        x0 = x1

    return x1
