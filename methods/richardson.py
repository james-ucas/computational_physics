def richardson(f, h, k, m):
    """
    Applies Richardson extrapolation to
        f = f(h) + O(h^(m))
    yielding
        f = g(h) + O(h^(m+2k))

    Assumes terms in odd, or even, powers of h are zero.

    :param callable f: function to extrapolate
    :param float h: argument of the expansion
    :param int k: number of iterations of the extrapolation
    :param int m:
    :return float: approximation to f(x) with error O(h^(m+2k))
    """

    def gamma(_k, l):
        if l == 0:
            return f(h / 2 ** _k)
        else:
            return (2 ** m * 4 ** (l - 1) * gamma(_k, l - 1) - gamma(_k - 1, l - 1)) / (2 ** m * 4 ** (l - 1) - 1)

    return gamma(k, k)
