import numpy as np

from calculus.integration import trapezium, simpson
from functions import polynomial_generator
from interpolation.lagrange_basis import lagrange_basis_polynomial_factory
from plotting import plt


def standard_plot(*args):
    return plt.subplots(*args, figsize=plt.figaspect(1 / 2), tight_layout=True)


def tangent_generator(x0, f, df):
    a = df(x0)
    b = f(x0) - a * x0

    def tangent(x):
        return a * x + b

    return tangent


def chord_generator(x0, x1, f):
    a = (f(x0) - f(x1)) / (x0 - x1)
    b = f(x0) - a * x0

    def chord(x):
        return a * x + b

    return chord


def derivative():
    fig, ax = standard_plot()
    x = np.linspace(0.4, 0.8, 1000)
    coefficients = [0, +1, -1, +1, -1]
    f, df, sf = polynomial_generator(coefficients)
    x0 = 0.6
    h = 0.1
    left_tangent = tangent_generator(x0 - h, f, df)
    right_tangent = tangent_generator(x0 + h, f, df)
    true_tangent = tangent_generator(x0, f, df)
    chord = chord_generator(x0 - h, x0 + h, f)
    ax.plot(x, f(x), linewidth=4)

    ax.plot(x, left_tangent(x), 'tab:red', linestyle='--', label='forward')
    ax.plot(x, right_tangent(x), 'tab:orange', linestyle='--', label='backward')
    ax.plot(x, chord(x), 'tab:green', linestyle='--', label='central')
    ax.plot(x, true_tangent(x), 'tab:purple', linestyle='--', label='exact')

    ax.plot([x0 + h, x0 + h], [0, f(x0 + h)], 'k--')
    ax.plot([x0, x0], [0, f(x0)], 'k--')
    ax.plot([x0 - h, x0 - h], [0, f(x0 + h)], 'k--')
    ax.set_ylim(0.25, 0.4)
    ax.set_xticks([0.5, 0.6, 0.7])
    ax.set_xticklabels(['$x-h$', '$x$', '$x+h$'])
    ax.set_yticks([])
    ax.legend()

    return ax


def integral():
    x0s = np.array([-2, -1, 0, 1, 2])
    f0s = np.array([4, 2, 4, 2, 3], dtype=int)

    poly = lagrange_basis_polynomial_factory(x0s, f0s)
    xs = np.linspace(-2, 2, 100)
    fig, (ax_trap, ax_simp) = plt.subplots(2, 1, sharex='all', sharey='all', tight_layout=True,
                                           figsize=plt.figaspect(1 / 2))
    ax_trap.plot(xs, poly(xs), color='tab:blue', linewidth=3)
    ax_simp.plot(xs, poly(xs), color='tab:blue', linewidth=3)
    x0s = np.linspace(-2, 2, 9)

    integral_trapezium = trapezium(poly, -2, 2, 8)
    integral_simpson = simpson(poly, -2, 2, 4)

    ax_trap.plot(x0s, poly(x0s), 'o', color='tab:red')
    ax_simp.plot(x0s, poly(x0s), 'o', color='tab:green')

    for x1s in np.vstack([x0s[0:-2:2], x0s[1:-1:2], x0s[2::2]]).T:
        newpoly = lagrange_basis_polynomial_factory(x1s, poly(x1s))
        ax_trap.plot(x1s, newpoly(x1s), color='tab:red')
        ax_trap.fill_between(x1s[:2], newpoly(x1s[:2]), 0, color='tab:red', alpha=0.3)
        ax_trap.fill_between(x1s[1:], newpoly(x1s[1:]), 0, color='tab:red', alpha=0.3)
        x1s = np.linspace(x1s[0], x1s[-1], 100)
        ax_simp.plot(x1s, newpoly(x1s), color='tab:green')
        ax_simp.fill_between(x1s, newpoly(x1s), 0, color='tab:green', alpha=0.3)

    ax_trap.text(0.4, 1.5, rf'$I_{{\text{{trapezium}}}}={integral_trapezium:.3f}$')
    ax_simp.text(0.4, 1.5, rf'$I_{{\text{{simpson}}}}={integral_simpson:.3f}$')
    for ax in (ax_trap, ax_simp):
        ax.set_xlim([-2, 2])
        ax.set_ylim([0, 4.4])
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([0, 4])
    plt.show()
