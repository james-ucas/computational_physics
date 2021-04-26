import os
import unittest

import numpy as np

from pair_potential import lj_energy_c, lj_gradient_c


class TestLJPotentialC(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.pos = np.loadtxt(f'{current_dir}/_lj13.txt').flatten()
        self.energy = -44.32680141

    def test_lj_energy_c(self):
        energy = lj_energy_c(self.pos, sigma=1.0, epsilon=1.0)
        self.assertAlmostEqual(energy, self.energy)

    def test_lj_gradient_c(self):
        gradient = lj_gradient_c(self.pos, sigma=1.0, epsilon=1.0)
        np.testing.assert_array_almost_equal(gradient, 0.0, decimal=3)
