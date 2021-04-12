from plotting import plt
import numpy as np

from calculus.difference import first_central_difference, second_central_difference
from methods import richardson


def delta1_factory(f, x):
    def delta1(h):
        return first_central_difference(f, x, h)

    return delta1


def delta2_factory(f, x):
    def delta2(h):
        return second_central_difference(f, x, h)

    return delta2


def absolute_error(exact, approx):
    return abs(exact - approx)


def relative_error(exact, approx):
    return abs((exact - approx) / exact)


def make_richardson_plot(ax, approx_func, hs, exact, m):
    for k in range(3):
        approxs = [richardson(approx_func, h, k=k, m=m) for h in hs]
        errors = [relative_error(exact, approx) for approx in approxs]
        ax.plot(hs, errors, '-', label=f'$k={k}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$h$')
    ax.set_ylabel('relative error')
    ax.legend()


def main():
    f = np.sin
    x = 1.0
    hs = np.logspace(1, -26, 281, base=2)
    fig, axes = plt.subplots(2, 1, sharex='all', sharey='all', tight_layout=True)

    make_richardson_plot(axes[0], approx_func=delta1_factory(f, x), hs=hs, exact=np.cos(x), m=2)
    make_richardson_plot(axes[1], approx_func=delta2_factory(f, x), hs=hs, exact=-np.sin(x), m=2)
    for ax in axes:
        ax.plot(hs, 2 ** -53 * x / hs, 'r--', label=r'$2^{-53}\times x/h$')
        ax.plot(hs, 2 ** -53 * x / hs / hs, 'r--', label=r'$2^{-53}\times x/h$')
    plt.savefig('./figures/icp3_4.pdf')


if __name__ == '__main__':
    main()
