def sum_of_digits(n):
    """
    find the sum of the digits of integer n

    >>> sum_of_digits(32768)
    26

    :param n: integer value
    :return: sum of digits
    """

    total = 0
    while n:
        total += n % 10
        n //= 10
    return total
