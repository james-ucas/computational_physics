import numpy as np

from calculus.difference import first_forward_difference, first_central_difference, first_backward_difference, \
    five_point_formula
from functions import xexpx, xexpx_derivative


def main():
    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    fig, ax = plt.subplots(1, 1, figsize=plt.figaspect(1 / 2), tight_layout=True)

    x = 2.0
    h = np.logspace(0, -53, 100, base=2)
    h_error = abs(1.0 - ((x + h) - x) / h)
    func = xexpx
    func_der = xexpx_derivative

    df = func_der(x)
    approximations = (first_forward_difference, first_backward_difference, first_central_difference, five_point_formula)
    labels = ('first forward difference', 'first backward difference', 'first central difference', 'five point formula')

    for approximation, label in zip(approximations, labels):
        dx_approx = approximation(f=func, x=x, h=h)
        df_error = abs(df - dx_approx) / abs(df)
        ax.plot(h, df_error, label=label)
    ax.plot(h, h_error, label='error in h')
    ax.plot(h, 2 ** -53 * x / h, label=r'$2^{-53}\times x/h$')

    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('h')
    ax.set_ylabel('relative error')
    plt.show()


if __name__ == '__main__':
    main()
