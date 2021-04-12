from plotting import plt
import numpy as np

from calculus.integration import gauss_legendre_factory
from icp3_4 import make_richardson_plot


def integrator_factory(f, a, b, n):
    integrator = gauss_legendre_factory(n)

    def evaluate_integral(h):
        return integrator(f, a, b, h)

    return evaluate_integral


def main():
    f = np.exp
    a, b = 0.0, 2.0
    hs = np.logspace(1, -5, 7, base=2)
    fig, ax = plt.subplots(1, 1, sharex='all', sharey='all')

    approx_func = integrator_factory(f, a, b, 2)
    make_richardson_plot(ax, approx_func=approx_func, hs=hs, exact=np.exp(2.0) - 1, m=4)
    plt.savefig('./figures/q8.pdf')


if __name__ == '__main__':
    main()
