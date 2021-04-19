from calculus.integration import gauss_legendre_3
from .closest_approach import compute_closest_approach


def integrand_factory(b, e, f, fargs):
    def left_integrand(r):
        return 1 / r / r / (1 - b * b / r / r) ** 0.5

    def right_integrand(r):
        return 1 / r / r / (1 - b * b / r / r - f(r, *fargs) / e) ** 0.5

    return left_integrand, right_integrand


def scattering_angle_factory(e, f, df, fargs):
    def compute_scattering_angle(b):
        rm = compute_closest_approach(b, e, f, df, fargs)
        left_integrand, right_integrand = integrand_factory(b, e, f, fargs)
        left_integral = gauss_legendre_3(left_integrand, b, 100 * b, 10000)
        right_integral = gauss_legendre_3(right_integrand, rm, 100 * rm, 10000)
        theta = 2 * b * (left_integral - right_integral)
        return theta

    return compute_scattering_angle
