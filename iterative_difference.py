from calculus.difference import iterative_difference
from functions import xexpx, xexpx_derivative


def main():
    f = xexpx
    x = 2.0
    h = 0.1
    df = xexpx_derivative(x)
    tol = 1e-11

    dfapprox = iterative_difference(f, x, h, tol)
    error = abs(dfapprox-df)
    print(f'true value: {df:.15f}\n'
          f'approx. value: {dfapprox:.15f}\n'
          f'tolerance: {tol}\n'
          f'absolute error: {error:.0e}\n')


if __name__ == '__main__':
    main()
