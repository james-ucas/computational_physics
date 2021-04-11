import matplotlib.pyplot as plt
import numpy as np

from functions.functions import count_calls_wrapper, quintic, nasty_quartic
from roots import iteration, bisection, inverse_quadratic_interpolation, newton_raphson

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


@count_calls_wrapper
def quintic_rearranged(x):
    return (x + 1) ** (1 / 5)


@count_calls_wrapper
def quintic_derivative(x):
    return 5 * x ** 4 - 1


@count_calls_wrapper
def quartic_rearranged(x):
    return ((8 * x - 3) / 16) ** (1 / 4)


@count_calls_wrapper
def quartic_derivative(x):
    return 64 * x ** 3 - 8


def test_methods(f, df, phi, x0, x1, tol):
    x = iteration(phi, x0, tol)
    print(f'{"iteration method":20s}| value, calls: {x}; {phi.calls}')
    x = bisection(f, x0, x1, tol)
    print(f'{"bisection":20s}| value, calls: {x}; {f.calls}')
    f.reset()
    x = inverse_quadratic_interpolation(f, x0, x1, tol)
    print(f'{"IQI":20s}| value, calls: {x}; {f.calls}')
    f.reset()
    x = newton_raphson(f, x0, df, tol)
    print(f'{"newton--raphson":20s}| value, calls: {x}; {f.calls}+{df.calls}')
    f.reset()


def main2():
    print("quintic")
    test_methods(quintic, quintic_derivative, quintic_rearranged, 1, 1.32, 1e-8)
    print("nasty quartic")
    test_methods(nasty_quartic, quartic_derivative, quartic_rearranged, 0, 2, 1e-8)


def bisection_figure(axes):
    from itertools import cycle
    colors = cycle(['tab:red', 'tab:green', 'tab:green', 'tab:red'])
    f = quintic
    f.reset()
    x0, x1 = 0.5, 1.5
    for ax in axes:
        color = next(colors)
        ax.plot([x0, x0], [-2, 2], '--', color=color)
        ax.plot([x0], [f(x0)], 'o', color=color)
        color = next(colors)
        ax.plot([x1, x1], [-2, 2], '--', color=color)
        ax.plot([x1], [f(x1)], 'o', color=color)
        x2 = (x0 + x1) / 2
        if f(x0) * f(x2) < 0:
            x0, x1 = x0, x2
        else:
            x0, x1 = x1, x2
    return ax


def inverse_quadratic_interpolation_figure(axes):
    from interpolation.lagrange_basis import lagrange_basis_polynomial_factory
    f = quintic
    f.reset()
    ys = np.linspace(-40, 40, 1000)
    x0, x1 = 1, 2
    inverse_quadratic_interpolation(f, x0, x1, tol=1e-8)
    data = [d for d in f.args if isinstance(d, list)]
    for (xs, fs), ax in zip(data, axes):
        ax.set_xlim([-3, 3])
        ax.set_ylim([-40, 40])

        print(xs, f(np.array(xs)))
        poly = lagrange_basis_polynomial_factory(fs, xs)
        ax.plot(poly(ys), ys, color='tab:red', linewidth=0.3)
        ax.plot(xs, f(np.array(xs)), 'o', color='tab:red')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])


def newton_raphson_figure(axes):
    f = quintic
    df = quintic_derivative
    f.reset()
    df.reset()
    x0 = 1.0
    xs = np.linspace(-20, 20, 1000)
    newton_raphson(f, x0, df, tol=1e-8)
    for (x0, x1), ax in zip(zip(f.args[0:], f.args[1:]), axes):
        df1 = df(x0)
        tangent = lambda y: (-x1 + y) * df1
        ax.plot([x0], [f(x0)], 'o', color='tab:red')
        ax.plot(xs, tangent(xs), '--', color='tab:red')
        ax.plot([x1], tangent(x1), 'o', color='tab:green')
        ax.plot([x1, x1], [-30, 30], '--', color='tab:green')
        ax.plot()


def iteration_figure(axes):
    f = quintic
    fr = quintic_rearranged
    x0 = 1.0
    iteration(fr, x0, tol=1e-8)
    for x0, ax in zip(fr.args, axes):
        ax.plot([x0], [f(x0)], 'ro')
    return axes


def quintic_ax(ax):
    xs = np.linspace(-20, 20, 1000)
    ax.plot(xs, quintic(xs))
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.plot(1, 0, '>k', transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, '^k', transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xticks([-1, 1])
    ax.set_yticks([-1, 1])
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])


def get_fig_ax():
    fig, axes = plt.subplots(1, 4,
                             figsize=plt.figaspect(1 / 5),
                             tight_layout=True)
    try:
        axes = axes.flatten()
    except AttributeError:
        axes = [axes]
    for ax in axes:
        quintic_ax(ax)
    return fig, axes


def main():
    fig, axes = get_fig_ax()
    iteration_figure(axes)
    plt.savefig('/home/farrelljd/Dropbox/compphys2021/iteration.pdf')
    # fig, axes = get_fig_ax()
    # bisection_figure(axes)
    # plt.savefig('/home/farrelljd/Dropbox/compphys2021/bisection.pdf')
    # fig, axes = get_fig_ax()
    # newton_raphson_figure(axes)
    # plt.savefig('/home/farrelljd/Dropbox/compphys2021/newtonraphson.pdf')
    # fig, axes = get_fig_ax()
    # inverse_quadratic_interpolation_figure(axes)
    # plt.savefig('/home/farrelljd/Dropbox/compphys2021/invquaint.pdf')


if __name__ == '__main__':
    main2()
