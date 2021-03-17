from itertools import combinations
from math import sqrt


def pair_potential(x, potential=None, args=()):
    """

    Calculates the potential energy of configuration of particles.

    >>> from lj_potential import lennard_jones_potential as lj

    >>> pair_potential(x=[[0.0,0.0,0.0]], potential=lj)
    0.0

    >>> pair_potential(x=[[0.0,0.0,0.0],[0.0,0.0,1.0]], potential=None)
    0.0

    >>> pair_potential(x=[[0.0,0.0,0.0],[0.0,0.0,1.0]], potential=lj)
    0.0

    >>> pair_potential(x=[[0.0,0.0],[0.0,1.0],[0.0,2.0]],
    ... potential=lj,
    ... args=(1.0, 1.0)) # 2D configuration
    -0.0615234375

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

    for x1, x2 in combinations(x, 2):  # for statement
        r_squared = 0.0
        for c1, c2 in zip(x1, x2):  # for statement
            r_squared += (c1 - c2) * (c1 - c2)
        r = sqrt(r_squared)  # sqrt imported from math module
        energy += potential(r, *args)

    return energy
