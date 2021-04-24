import os
import unittest

import numpy as np

from pair_potential import LJPotential
from local_optimisation import bfgs


class TestPairPotential(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.pos = np.loadtxt(f'{current_dir}/_lj13.txt')
        self.energy = -44.32680141

    def test_ljpotential_energy(self):
        n, d = self.pos.shape
        potential = LJPotential(n, d)
        self.assertAlmostEqual(potential.get_energy(self.pos), self.energy)

    def test_ljpotential_gradient(self):
        n, d = self.pos.shape
        potential = LJPotential(n, d)
        f, df = potential.get_energy, potential.get_gradient
        pos = bfgs(f, self.pos.flatten(), df, xtol=1e-8, gtol=1e-8)
        np.testing.assert_array_almost_equal(df(pos), 0.0)
