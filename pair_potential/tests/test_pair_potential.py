import os
import unittest

import numpy as np

from pair_potential.lj_potential import lennard_jones_potential, vectorised_lennard_jones_potential
from pair_potential.pair_potential import slow_pair_potential, faster_pair_potential, fast_pair_potential


class TestPairPotential(unittest.TestCase):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        self.pos = np.loadtxt(f'{current_dir}/_lj13.txt')
        self.energy = -44.326801

    def test_slow_pair_potential(self):
        v = slow_pair_potential(self.pos, potential=lennard_jones_potential, args=(1.0, 1.0))
        self.assertAlmostEqual(v, self.energy, places=6)

    def test_faster_pair_potential(self):
        v = faster_pair_potential(self.pos, potential=lennard_jones_potential, args=(1.0, 1.0))
        self.assertAlmostEqual(v, self.energy, places=6)

    def test_fast_pair_potential(self):
        v = fast_pair_potential(self.pos, potential=vectorised_lennard_jones_potential, args=(1.0, 1.0))
        self.assertAlmostEqual(v, self.energy, places=6)
