import numpy as np


def coulomb(r, kappa=1):
    return kappa / r


def coulomb_derivative(r, kappa=1):
    return -kappa / r / r


def compute_closest_approach_coulomb(b, e, kappa=1):
    return (kappa + ((2 * e * b) ** 2 + kappa) ** 0.5) / 2 / e


def compute_angle_coulomb(b, e, kappa=1):
    return 2 * np.arctan(kappa / 2 / e / b)


def compute_differential_cross_section_coulomb(t, e, kappa=1):
    return (kappa / 4 / e) ** 2 / np.sin(t / 2) ** 4
