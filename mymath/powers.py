def square(x):
    return x * x


def cube(x):
    return x*x*x


def power(x, n=1):
    y = x
    while n > 1:
        y *= x
        n -= 1
    return y


if __name__ == '__main__':
    print(power(5, n=4))
