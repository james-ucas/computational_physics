import numpy as np

from calculus.integration import gauss_legendre_factory, simpson

from icp3_4 import relative_error


def gaussian_integrand(x):
    return np.exp(-x * x)


def gaussian_integrand_change_of_variables(t):
    """
    Substitutes x = 1/t ; dx = -1/t^2 dt
    """
    return -1 / t / t * np.exp(-1 / t / t)


def main():
    exact = np.sqrt(np.pi)
    intervals = 20

    left = simpson(gaussian_integrand, 0, 1, intervals)
    right = simpson(gaussian_integrand_change_of_variables, 1, 1e-5, intervals)
    approx = 2 * (left + right)
    error = relative_error(exact, approx)
    print(f'simpson error: {error}')

    nodes = 3
    gl = gauss_legendre_factory(nodes)
    left = gl(gaussian_integrand, 0, 1, intervals)
    right = gl(gaussian_integrand_change_of_variables, 1, 0, intervals)
    approx = 2 * (left + right)
    error = relative_error(exact, approx)
    print(f'gauss({nodes}) error: {error}')


if __name__ == '__main__':
    main()
