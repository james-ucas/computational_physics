import numpy as np

from methods.adaptive import adaptive_first, adaptive_second
from plotting import plt


def func(x):
    return np.log(x) * np.exp(-x)


def dfunc(x):
    return np.exp(-x) * (1 / x - np.log(x))


def d2func(x):
    return np.exp(-x) * (-1 / x / x - 2 / x + np.log(x))


def main():
    x = 2
    f = func
    df_exact = dfunc(x)
    d2f_exact = d2func(x)

    h = 1
    tols = np.logspace(0, -57, 58, base=2)
    df_approxs = [adaptive_first(f, x, h, tol) for tol in tols]
    df_errors = abs(df_approxs - df_exact)
    d2f_approxs = [adaptive_second(f, x, h, tol) for tol in tols]
    d2f_errors = abs(d2f_approxs - d2f_exact)

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.plot(tols, df_errors, label='first central')
    ax.plot(tols, d2f_errors, label='second central')
    ax.set_xscale('log')
    ax.set_xlabel('tolerance')
    ax.set_yscale('log')
    ax.set_ylabel('actual error')
    ax.legend()
    plt.savefig('./figures/icp5.pdf')


if __name__ == '__main__':
    main()
