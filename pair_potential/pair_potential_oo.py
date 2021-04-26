import abc

import numpy as np


class PairPotential(abc.ABC):
    """
    Base class for pair potentials.
    """

    def __init__(self, particles, dimensions):
        self.particles = particles
        self.dimensions = dimensions
        self.shape = (particles, dimensions,)
        self.mask = np.zeros((particles, particles, dimensions), dtype=np.bool)
        self.mask[np.diag_indices_from(self.mask[:, :, 0])] = True

    @abc.abstractmethod
    def pair_potential(self, r, *args, **kwargs):
        """compute the potential energy for one pair"""

    @abc.abstractmethod
    def pair_gradient(self, r, *args, **kwargs):
        """compute the gradient/distance for one pair"""

    def get_distances(self, x):
        rij = x[:, None, :] - x[None, :, :]
        dij = np.einsum('ija,ija->ij', rij, rij) ** 0.5
        return np.ma.array(rij, mask=self.mask), np.ma.array(dij, mask=self.mask[:, :, 0])

    def get_energy(self, x, **kwargs):
        x = x.reshape(self.shape)
        _, dij = self.get_distances(x)
        return self.pair_potential(dij, **kwargs).sum() / 2

    def get_gradient(self, x, **kwargs):
        shape = x.shape
        x = x.reshape(self.shape)
        rij, dij = self.get_distances(x)
        gia = np.einsum('ij,ija->ia', self.pair_gradient(dij, **kwargs), rij)
        return gia.reshape(shape)


class LJPotential(PairPotential):
    """
    Lennard-Jones pair potential
    """

    def pair_potential(self, r, sigma=1.0, epsilon=1.0):
        r6 = (sigma / r) ** 6
        return 4 * epsilon * r6 * (r6 - 1)

    def pair_gradient(self, r, sigma=1.0, epsilon=1.0):
        r6 = (sigma / r) ** 6
        return 24 * epsilon * r6 * (1 - 2 * r6) / r / r


def main():
    n, d = 13, 3
    x = np.loadtxt('tests/_lj13.txt').flatten()
    pot = LJPotential(n, d)
    print(pot.get_gradient(x, sigma=1.0, epsilon=1.0))
    pass


if __name__ == '__main__':
    main()
