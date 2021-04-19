import numpy as np

from calculus.difference import first_central_difference
from scattering.scattering_angle import scattering_angle_factory


def compute_differential_cross_section(bs, e, f, df, fargs):
    scattering_angle = scattering_angle_factory(e, f, df, fargs)
    thetas = np.array([scattering_angle(b) for b in bs])
    dtdbs = np.array([first_central_difference(scattering_angle, b, h=b * 1e-6) for b in bs])
    sigmas = bs / abs(np.sin(thetas) * dtdbs)
    return thetas, sigmas
