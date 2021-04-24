import numpy as np
from numpy.linalg import norm

from matrices import rayleigh_ritz
from .bfgs import bfgs_update_hessian
from .line_search import line_search


def get_hessian_approximator(df, h=1e-4):
    def central_differences(x):
        n = x.size
        hessian = np.zeros((n, n), dtype=float)
        for i in range(n):
            x[i] += h
            hessian[i, :] += df(x)
            x[i] -= 2 * h
            hessian[i, :] -= df(x)
            x[i] += h
        return hessian / 2 / h

    return central_differences


def project_out(aa, v):
    return aa - (aa @ v) * v


def hybrid_eigenvector_following(f, x0, df, d2f=None, tolg=1e-5, tolev=1e-6):
    x0 = x0.copy()
    binv = np.eye(x0.size)
    dfx0 = df(x0)
    modg = norm(dfx0)

    if d2f is None:
        d2f = get_hessian_approximator(df)

    def negativef(x_):
        return -f(x_)

    while modg > tolg:
        d2f0 = d2f(x0)
        eval_, evec = rayleigh_ritz(d2f0, tolev)
        gx_ = -(dfx0 @ evec) * evec

        if norm(gx_) > tolg:
            alpha = line_search(negativef, x0, evec, gx_, max_alpha=1.0)
            x0 = x0 + alpha * evec

        p = -binv @ dfx0
        dfx0 = project_out(dfx0, evec)
        p = project_out(p, evec)
        p /= norm(p)

        if norm(dfx0) > tolg:
            alpha = line_search(f, x0, p, dfx0, max_alpha=1.0)
            x0 = x0 + alpha * p
            s = alpha * p
            gx_new = df(x0)
            binv = bfgs_update_hessian(binv, gx_new - dfx0, s)
            dfx0 = gx_new
        modg = norm(dfx0)

    return x0
