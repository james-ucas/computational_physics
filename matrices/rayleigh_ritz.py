import numpy as np
from numpy.linalg import norm

from local_optimisation.bfgs import bfgs


def rayleigh_ritz(d2f, x0, tol=1e-3):
    d2f0 = d2f(x0)
    evec0 = np.ones(d2f0.shape[0])

    def f(evec):
        xs = evec[:, np.newaxis]
        return ((xs.T @ d2f0 @ xs) / (xs.T @ xs))[0, 0]

    def df(evec):
        xs = evec[:, np.newaxis]
        left = (xs.T @ d2f0 @ xs)
        leftp = d2f0 @ xs + d2f0.T @ xs
        right = (xs.T @ xs)
        rightp = 2 * xs
        return ((leftp * right - left * rightp) / (right * right)).flatten()

    xmin = bfgs(f, evec0, df, tol)
    return f(xmin), xmin / norm(xmin)
