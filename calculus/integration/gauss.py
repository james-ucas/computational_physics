import numpy as np

LEGENDRE_RULES = {
    1: ((0,), (2,)),
    2: ((-1 / 3 ** 0.5, 1 / 3 ** 0.5), (1, 1)),
    3: ((-(3 / 5) ** 0.5, 0, (3 / 5) ** 0.5), (5 / 9, 8 / 9, 5 / 9))
}


def gauss_legendre_factory(nodes):
    """
    Make a function that computes the Gauss-Legendre quadrature on the required number of nodes.

    :param int nodes: number of nodes (1--3)

    :return callable: function to compute the quadrature
    """
    try:
        nodes, weights = np.array(LEGENDRE_RULES[nodes]).reshape(2, 1, -1)
    except KeyError:
        raise NotImplementedError(f'Gauss-Legendre quadrature is not implemented on ({nodes}) nodes')

    def gauss_legendre(f, a, b, h):
        """
        Compute the Gauss-Legendre quadrature of function f between a and b with a step size h.

        :param callable f: function to integrate, call signature f(x)
        :param float a: lower limit of integration
        :param float b: upper limit of integration
        :param float or int h: step size if float, number of intervals if int

        :raises ValueError: if h does not divide the interval exactly
        :raises ValueError: if h is not float or int

        :return float: integral

        """
        if isinstance(h, int):
            ts = np.linspace(a, b, h + 1).reshape(-1, 1)
        elif isinstance(h, float):
            m = abs(int(round((b - a) / h, 0)))
            ts = np.linspace(a, b, m + 1).reshape(-1, 1)
            h_ = abs(ts[1, 0] - ts[0, 0])
            if abs(h - h_) / h > 2 ** -50:
                raise ValueError(f'can\'t divide the interval [{a},{b}] with step size {h}')
        else:
            raise TypeError(f'h must be float or integer, not {type(h)}')
        width = (ts[1:] - ts[:-1]) / 2
        mean = (ts[1:] + ts[:-1]) / 2
        new_nodes = width * nodes + mean
        return (width * weights * f(new_nodes)).sum()

    return gauss_legendre


gauss_legendre_1 = gauss_legendre_factory(1)
gauss_legendre_2 = gauss_legendre_factory(2)
gauss_legendre_3 = gauss_legendre_factory(3)
