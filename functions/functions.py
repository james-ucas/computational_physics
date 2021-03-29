from numpy import exp, arange, array, where


def polynomial_generator(coefficients):
    coefficients = array(coefficients)[:, None]
    ks = arange(coefficients.size)[:, None]
    ks_inv = where(ks == 0, 1, 1 / ks)

    def poly(x):
        return (coefficients * x ** ks).sum(0)

    def poly_derivative(x):
        return (coefficients * ks * x ** (ks - 1)).sum(0)

    def poly_indefinite_integral(x):
        return (coefficients * ks_inv * x ** (ks + 1)).sum(0)

    def poly_integral(a, b):
        return poly_indefinite_integral(b) - poly_indefinite_integral(a)

    return poly, poly_derivative, poly_integral


def xexpx(x):
    return x * exp(x)


def xexpx_derivative(x):
    return (x + 1) * exp(x)


def xexpx_integral(x):
    return (x - 1) * exp(x)
