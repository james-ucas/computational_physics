import numpy as np


def lennard_jones_potential(r, sigma=1.0, epsilon=1.0):
    """

    Calculates the Lennard-Jones potential for particles with diameter sigma
    at a separation r with a well-depth epsilon.

    >>> lennard_jones_potential(1.0, 1.0, 1.0)
    0.0

    >>> lennard_jones_potential(2**(1/6), 1.0, 1.0)
    -1.0

    >>> lennard_jones_potential(0.0, 1.0, 1.0)
    Traceback (most recent call last):
    ZeroDivisionError: float division by zero

    >>> lennard_jones_potential(-1.0, 1.0, 1.0)
    Traceback (most recent call last):
    ValueError: distance between particles is negative

    >>> lennard_jones_potential(1.0, -1.0, 1.0)
    Traceback (most recent call last):
    ValueError: particle diameter is not strictly positive

    :param r: the distance between two particles
    :type r: float
    :param sigma: the diameter of a particle
    :type sigma: float
    :param epsilon: the well depth of the potential
    :type epsilon: float

    :return: the Lennard-Jones energy of the particle pair
    :rtype: float
    """

    if r < 0.0:
        raise ValueError("distance between particles is negative")
    elif sigma <= 0.0:
        raise ValueError("particle diameter is not strictly positive")

    r6 = (sigma / r) ** 6

    return 4 * epsilon * r6 * (r6 - 1)


def vectorised_lennard_jones_potential(r, sigma=1.0, epsilon=1.0):
    """

    Calculates the Lennard-Jones potential for particles with diameter sigma
    at a separation r with a well-depth epsilon.

    >>> vectorised_lennard_jones_potential(1.0, 1.0, 1.0)
    0.0

    >>> vectorised_lennard_jones_potential(2**(1/6), 1.0, 1.0)
    -1.0

    >>> vectorised_lennard_jones_potential(0.0, 1.0, 1.0)
    Traceback (most recent call last):
    ZeroDivisionError: float division by zero

    >>> vectorised_lennard_jones_potential(-1.0, 1.0, 1.0)
    Traceback (most recent call last):
    ValueError: distance between particles is negative

    >>> vectorised_lennard_jones_potential(1.0, -1.0, 1.0)
    Traceback (most recent call last):
    ValueError: particle diameter is not strictly positive

    >>> import numpy as np
    >>> r = np.array([1.0,2.0,3.0])
    >>> vectorised_lennard_jones_potential(r)  #doctest: +ELLIPSIS
    array([ 0.        , -0.061523..., -0.005479...])

    :param r: the distance between two particles
    :type r: float
    :param sigma: the diameter of a particle
    :type sigma: float
    :param epsilon: the well depth of the potential
    :type epsilon: float

    :return: the Lennard-Jones energy of the particle pair
    :rtype: float
    """

    if np.any(r < 0.0):
        raise ValueError("distance between particles is negative")
    elif np.any(sigma <= 0.0):
        raise ValueError("particle diameter is not strictly positive")

    r6 = (sigma / r) ** 6

    return 4 * epsilon * r6 * (r6 - 1)
