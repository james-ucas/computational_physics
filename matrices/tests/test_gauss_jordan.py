import unittest

import numpy as np

from matrices import determinant, solve_system, matrix_inverse


class TestGaussJordan(unittest.TestCase):
    def setUp(self):
        self.aa = np.array([2, 1, -1, -3, -1, 2, -2, 1, 2], dtype=float).reshape((3, 3,))
        self.b = np.array([[8, -11, -3]], dtype=float).T
        self.x = np.array([[2, 3, -1]], dtype=float)
        self.aa_inv = np.array([4, 3, -1, -2, -2, 1, 5, 4, -1], dtype=float).reshape((3, 3,))
        self.aa_det = -1.0

    def test_solve_system(self):
        np.testing.assert_allclose(solve_system(self.aa, self.b), self.x)

    def test_determinant(self):
        self.assertAlmostEqual(determinant(self.aa), self.aa_det)

    def test_inverse(self):
        np.testing.assert_allclose(matrix_inverse(self.aa), self.aa_inv)
