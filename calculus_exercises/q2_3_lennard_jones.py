import numpy as np

from calculus.difference import first_central_multi, second_central_multi


def simple_function(x):
    return (x * x).sum() + (x[:, None] * x[None, :]).sum()


def simple_first_derivative(x):
    return 2 * x.sum() + 2 * x


def simple_second_derivative(x):
    return np.full((x.size, x.size), 2.0) + np.eye(x.size) * 2.0


def main():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    df_exact = simple_first_derivative(x)
    d2f_exact = simple_second_derivative(x)
    df_approx = first_central_multi(simple_function, x, h=1e-6)
    d2f_approx = second_central_multi(simple_function, x, h=1e-4)
    df_error = np.linalg.norm(df_approx - df_exact)
    d2f_error = np.linalg.norm(d2f_approx - d2f_exact)
    print(df_error, d2f_error)


if __name__ == '__main__':
    main()
