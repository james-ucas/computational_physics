def reorder(x0s, f0s):
    order = f0s.argsort()
    x0s[:] = x0s[:, order]
    f0s[:] = f0s[order]


def nelder_mead(f, x0s, tol, alpha=1, gamma=2, rho=1 / 2, sigma=1 / 2):
    """

    An implementation of the Nelder-Mead method for
    finding a minimum of an n-dimensional function f
    by evolving the simplex x0s until the standard deviation
    of the function values of the simplex is less than tol.

    :param callable f: function to minimise
    :param ndarray(n,n+1) x0s: sample n+1 points in n-dimensional space (a simplex)
    :param float tol: termination parameter
    :param float alpha: control parameter for reflection operation, alpha > 0
    :param float gamma: control parameter for expansion operation, gamma > 1
    :param float rho: control parameter for contraction operation, 0 < rho <= 1/2
    :param float sigma: control parameter for shrinking operation, 0 < sigma < 1
    :return ndarray(n,): point with lowest known function value
    """
    f0s = f(x0s)
    while x0s.std(1).mean() > tol:
        centroid = x0s[:, :-1].mean(axis=1)
        reflected = centroid + alpha * (centroid - x0s[:, -1])
        fr = f(reflected)
        if f0s[0] <= fr < f0s[-2]:
            # reflect
            x_new, f_new = reflected, fr
        elif fr < f0s[0]:
            expanded = centroid + gamma * (reflected - centroid)
            fe = f(expanded)
            x_new, f_new = (expanded, fe) if fe < fr else (reflected, fr)
        else:
            contracted = centroid + rho * (x0s[:, -1] - centroid)
            fc = f(contracted)
            if fc < f0s[-1]:
                x_new, f_new = contracted, fc
            else:
                x0s[:, 1:] = x0s[:, 0] + sigma * (x0s[:, 1:] - x0s[:, 0])
                f0s[:] = f(x0s)
                x_new, f_new = x0s[:, -1], f0s[-1]
        x0s[:, -1], f0s[-1] = x_new, f_new
        reorder(x0s, f0s)
    return x0s[:, 0]
