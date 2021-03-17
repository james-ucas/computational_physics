def fibonacci(n):
    """

    calculates the n^th value of the Fibonacci sequence, F_n

    >>> fibonacci(0)
    0

    >>> fibonacci(1)
    1

    >>> fibonacci(100)
    354224848179261915075

    >>> fibonacci(-1)
    Traceback (most recent call last):
    ValueError: n must be >= 0, not -1

    >>> fibonacci('1')
    Traceback (most recent call last):
    TypeError: n must have type integer, not str

    :param n: element of the Fibonacci sequence to be calculated
    :type n: int

    :return: F_n
    :rtype: int
    """

    if not isinstance(n, int):
        raise TypeError(f'n must have type integer, not {type(n).__name__}')
    elif n < 0:
        raise ValueError(f'n must be >= 0, not {n}')

    fib0, fib1 = 0, 1
    while n > 0:
        fib0, fib1 = fib1, fib0 + fib1
        n -= 1
    return fib0
