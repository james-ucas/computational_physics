from typing import NamedTuple as NamedTuple

Polynomial = NamedTuple('Polynomial', function=callable, derivative=callable, rearranged=callable)


def quintic_function(x):
    return x ** 5 - x - 1


def quintic_derivative(x):
    return 5 * x ** 4 - 1


def quintic_rearranged(x):
    return (1 + x) ** (1 / 5)


def quartic_function(x):
    return 16 * x ** 4 - 8 * x + 3


def quartic_derivative(x):
    return 64 * x ** 3 - 8


def quartic_rearranged(x):
    return (8 * x - 3) ** (1 / 4) / 2


def cubic_function(x):
    return x ** 3 - 2 * x ** 2 - 11 * x + 12


def cubic_derivative(x):
    return 3 * x ** 2 - 4 * x - 11


def cubic_rearranged(x):
    return (2 * x ** 2 + 11 * x - 12) ** (1 / 3)


quintic = Polynomial(quintic_function, quintic_derivative, quintic_rearranged)
quartic = Polynomial(quartic_function, quartic_derivative, quartic_rearranged)
cubic = Polynomial(cubic_function, cubic_derivative, cubic_rearranged)
