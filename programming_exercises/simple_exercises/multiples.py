def sum_of_multiples_below(n):
    """
    calculate the sum of multiples of 3 and 5 below n

    >>> sum_of_multiples_below(10)
    23

    :param n: upper limit
    :return: sum of multiples
    """

    total = 0
    while n > 2:
        n -= 1
        if (n % 3) == 0 or (n % 5) == 0:
            total += n
    return total
