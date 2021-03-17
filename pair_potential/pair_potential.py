from itertools import combinations
from math import sqrt

import numpy as np


def slow_pair_potential(x, potential=None, args=()):
    """

    Calculates the potential energy of configuration of particles.

    :param x: positions of the particles
    :type x: list of lists
    :param potential: the pairwise potential function.
                      must be of the form f(x, *args).
    :type potential: callable
    :param args: arguments to pass to the function

    :return: energy of the configuration
    :rtype: float
    """

    if potential is None:
        return 0.0

    energy = 0.0

    for x1, x2 in combinations(x, 2):
        r_squared = 0.0
        for c1, c2 in zip(x1, x2):
            r_squared += (c1 - c2) * (c1 - c2)
        r = sqrt(r_squared)
        energy += potential(r, *args)

    return energy


def faster_pair_potential(x, potential=None, args=()):
    """

    Calculates the potential energy of configuration of particles.

    :param x: positions of the particles
    :type x: list of lists
    :param potential: the pairwise potential function.
                      must be of the form f(x, *args).
    :type potential: callable
    :param args: arguments to pass to the function

    :return: energy of the configuration
    :rtype: float
    """

    if potential is None:
        return 0.0

    energy = 0.0

    for x1, x2 in combinations(x, 2):
        r = np.linalg.norm(x1 - x2)
        energy += potential(r, *args)

    return energy


def fast_pair_potential(x, potential=None, args=()):
    """

    Calculates the potential energy of configuration of n particles in d dimensions.

    :param x: positions of the particles
    :type x: numpy ndarray, shape n x d
    :param potential: the pairwise potential function.
                      must be of the form f(x, *args).
    :type potential: callable
    :param args: arguments to pass to the function

    :return: energy of the configuration
    :rtype: float
    """

    if potential is None:
        return 0.0

    n, _ = x.shape
    left, right = np.triu_indices(n, 1)
    r = np.linalg.norm(x[left] - x[right], axis=1)
    energy = np.sum(potential(r, *args))

    return energy
