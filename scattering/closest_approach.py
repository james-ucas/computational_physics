from roots import newton_raphson


def closest_approach_factory(b, e, f, df, fargs):
    def closest_approach_function(r):
        return 1 - (b / r) ** 2 - f(r, *fargs) / e

    def closest_approach_derivative(r):
        return 2 * b ** 2 / r ** 3 - df(r, *fargs) / e

    return closest_approach_function, closest_approach_derivative


def compute_closest_approach(b, e, f, df, fargs, x0=1, tol=1e-8):
    f, df = closest_approach_factory(b, e, f, df, fargs)
    return newton_raphson(f, x0, df, tol)
