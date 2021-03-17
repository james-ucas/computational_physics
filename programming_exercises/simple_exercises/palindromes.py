def is_palindrome(n):
    s = str(n)
    return s == s[::-1]


def largest_palindromic_n_digit_product(n):
    """
    find the largest palindromic product of two n-digit numbers

    >>> largest_palindromic_n_digit_product(2)
    9009

    :param n: number of digits
    :return: largest palindromic product if exists, otherwise, None
    """

    max_val = 10 ** n - 1
    min_val = 10 ** (n - 1)
    largest_palindrome = (min_val + 1) ** 2

    x = max_val
    while (product := x * x) > largest_palindrome:
        if is_palindrome(product):
            largest_palindrome = product
            break
        y = x - 1
        while (product := x * y) > largest_palindrome:  # assignment expression
            if is_palindrome(product):
                largest_palindrome = product
                break
            y -= 1
        x -= 1
    return largest_palindrome
