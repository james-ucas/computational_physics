import numpy as np

from plotting import plt
from scattering import compute_differential_cross_section
from scattering.rutherford import compute_differential_cross_section_coulomb, compute_angle_coulomb


def yukawa(r, kappa, alpha):
    return kappa / r * np.exp(-alpha * r)


def yukawa_derivative(r, kappa, alpha):
    return -yukawa(r, kappa, alpha) * (1 / r + alpha)


def lennard_jones(r, sigma, epsilon):
    r6 = sigma * sigma / r / r
    r6 *= r6 * r6
    return 4 * epsilon * r6 * (r6 - 1)


def lennard_jones_derivative(r, sigma, epsilon):
    r6 = sigma * sigma / r / r
    r6 *= r6 * r6
    return 24 * epsilon * r6 * (1 - 2 * r6) / r


def main():
    fig, axes = plt.subplots(1, 2, figsize=plt.figaspect(1/2))
    kappa, e = 1.0, 1.0
    bs = np.logspace(-1, 1, 100)

    thetas = np.linspace(0.0, np.pi, 1000)
    theta_coulomb = compute_angle_coulomb(bs, e, kappa)
    sigma_coulomb = compute_differential_cross_section_coulomb(theta_coulomb, e, kappa)
    axes[0].plot(theta_coulomb, bs, 'k-.', label='Coulomb')
    axes[1].plot(theta_coulomb, np.log(sigma_coulomb), 'k-.', label='Coulomb')

    f, df = yukawa, yukawa_derivative
    for alpha in [0.1,1]:
        fargs = (kappa, alpha,)
        thetas, sigmas = compute_differential_cross_section(bs, e, f, df, fargs)
        axes[0].plot(thetas, bs, '.-', label=fr'$\alpha={alpha}$')
        axes[1].plot(thetas, np.log(sigmas), '.-', label=fr'$\alpha={alpha}$')

    f, df = lennard_jones, lennard_jones_derivative
    for epsilon in [0.1,1]:
        fargs = (1.0, epsilon,)
        thetas, sigmas = compute_differential_cross_section(bs, e, f, df, fargs)
        thetas = thetas-np.rint(thetas/2/np.pi)*2*np.pi
        axes[0].plot(thetas, bs, '.-', label=fr'$\epsilon={epsilon}$')
        axes[1].plot(thetas, np.log(sigmas), '.-', label=fr'$\epsilon={epsilon}$')

    axes[1].legend()
    axes[0].set_ylabel(r'$\text{impact parameter}$')
    axes[0].set_xlabel(r'$\text{polar scattering angle, }\theta$')
    axes[1].set_xlabel(r'$\text{polar scattering angle, }\theta$')
    axes[1].set_ylabel(r'$\text{log differential cross section, }\log{d\sigma/d\Omega}$')
    plt.show()


if __name__ == '__main__':
    main()
