# the perimeters of hexagons circumscribed and inscribed about a circle with radius 0.5
HEXAGON_INNER = 3  # don't forget to set the perimeters
HEXAGON_OUTER = 12 ** 0.5  #


def archimedes_formula(precision):
    """
    Uses the polygon method of Archimedes to approximate pi, starting from the perimeters of regular hexagons.
    https://en.wikipedia.org/wiki/Approximations_of_%CF%80

    >>> archimedes_formula(1e-8) #doctest: +ELLIPSIS
    (49152, 3.14159265...)

    :param precision: the precision to which pi should be approximated
    :type precision: float
    :return: number of sides of the polygon and upper bound on pi approximated to precision
    :rtype: int, float
    """
    p_inner = HEXAGON_INNER
    p_outer = HEXAGON_OUTER
    number_sides = 6
    while p_outer - p_inner > precision:
        number_sides *= 2
        p_outer_2 = 2 * p_inner * p_outer / (p_inner + p_outer)
        p_inner, p_outer = (p_inner * p_outer_2) ** 0.5, p_outer_2
    return number_sides, p_outer
