import matplotlib.pyplot as plt
import numpy as np
from functions import polynomial_generator
from matplotlib.patches import Polygon

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12


def standard_plot(*args):
    return plt.subplots(*args, figsize=plt.figaspect(1 /2), tight_layout=True)


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
    chord = chord_generator(x0-h, x0+h, f)
    ax.plot(x, f(x), linewidth=4)

    ax.plot(x, left_tangent(x), 'tab:red', linestyle='--', label='forward')
    ax.plot(x, right_tangent(x), 'tab:orange', linestyle='--', label='backward')
    ax.plot(x, chord(x), 'tab:green', linestyle='--', label='central')
    ax.plot(x, true_tangent(x), 'tab:purple', linestyle='--', label='exact')

    ax.plot([x0+h, x0+h], [0, f(x0+h)], 'k--')
    ax.plot([x0, x0], [0, f(x0)], 'k--')
    ax.plot([x0-h, x0-h], [0, f(x0+h)], 'k--')
    ax.set_ylim(0.25,0.4)
    ax.set_xticks([0.5,0.6,0.7])
    ax.set_xticklabels(['$x-h$','$x$','$x+h$'])
    ax.set_yticks([])
    ax.legend()

    return ax


def integral():
    fig, ax = standard_plot(1,1)
    a, b = -1,2
    x = np.linspace(a,b, 1000)
    coefficients = [2,-2,0,1]
    f, df, sf = polynomial_generator(coefficients)
    ax.plot(x, f(x), linewidth=2)
    h = 0.5
    for corner in np.arange(a,b, h):
        vertices = [(corner, 0), (corner, f(corner)), (corner+h, f(corner+h)),(corner+h,0)]
        polygon_edges = Polygon(vertices, fill=False, linestyle='-', linewidth=1, edgecolor='k')
        polygon = Polygon(vertices, facecolor='b', alpha=0.25)
        ax.add_patch(polygon_edges)
        ax.add_patch(polygon)
    ax.set_ylim([0, None])
    ax.set_xlim([a, b])


def main():
    derivative()
    plt.show()
    integral()
    plt.show()


if __name__ == '__main__':
    main()
