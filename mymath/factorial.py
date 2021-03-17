def factorial(n):  # function definition statement
    """

    evaluates n! = n * (n - 1) * ... * 2 * 1
    0! evaluates to 1

    >>> factorial(0)
    1

    >>> factorial(10)
    3628800

    >>> factorial(-1)
    Traceback (most recent call last):
    ValueError: n! is undefined for n less than zero

    >>> factorial(3.141)
    Traceback (most recent call last):
    TypeError: n is not an integer

    :param n: element of the factorial sequence to be evaluated
    :type n: int

    :return: n!
    :rtype: int
    """

    if not isinstance(n, int):
        raise TypeError("n is not an integer")  # raise statement
    elif n < 0:  # if statement
        raise ValueError("n! is undefined for n less than zero")  # raise statement

    n_factorial = 1  # assignment statement

    while n > 1:  # while statement
        n_factorial = n_factorial * n  # assignment statement
        n = n - 1  # assignment statement

    return n_factorial  # return statement
