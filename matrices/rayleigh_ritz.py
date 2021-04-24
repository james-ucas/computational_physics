import numpy as np
from numpy.linalg import norm

from local_optimisation.bfgs import bfgs


def rayleigh_ritz(aa, tol=1e-3):
    evec0 = np.ones(aa.shape[0])

    def f(evec):
        v = evec[:, np.newaxis]
        return ((v.T @ aa @ v) / (v.T @ v))[0, 0]

    def df(evec):
        v, v_t = evec[:, np.newaxis], evec[np.newaxis, :]
        df0 = 2 * ((v.T @ v)*(aa @ v) - v @ (v.T @ aa @ v)) / (v.T @ v) ** 2
        return df0.flatten()

    xmin = bfgs(f, evec0, df, tol)
    return f(xmin), xmin / norm(xmin)
