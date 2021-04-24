import unittest
from math import sqrt

import numpy as np

from matrices import rayleigh_ritz


class TestRayleighRitz(unittest.TestCase):
    def setUp(self):
        self.aa = np.array([0, 1, 1, 1], dtype=float).reshape((2, 2,))
        self.eval = (1 - sqrt(5)) / 2
        self.evec = [(1 + sqrt(5)) / sqrt(10 + 2 * sqrt(5)), -2 / sqrt(10 + 2 * sqrt(5))]

    def test_rayleigh_ritz_asymmetric(self):
        eval0, evec0 = rayleigh_ritz(self.aa, tol=1e-12)
        self.assertAlmostEqual(eval0, self.eval)
        np.testing.assert_allclose(evec0, self.evec)
