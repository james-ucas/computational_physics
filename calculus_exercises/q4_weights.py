import sympy as xp

from interpolation.lagrange_basis import get_newton_cotes_weights


def main():
    h = xp.symbols('h')
    simpson_weights = get_newton_cotes_weights(n=3, h=h)
    boole_weights = get_newton_cotes_weights(n=5, h=h)
    print(simpson_weights, boole_weights)


if __name__ == '__main__':
    main()
