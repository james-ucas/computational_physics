import unittest

import numpy as np

from functions import quartic_double_well, quartic_double_well_derivative, quartic_double_well_second_derivative
from local_optimisation import newton_raphson_multi, gradient_descent, hybrid_eigenvector_following


class TestNewtonRaphsonMulti(unittest.TestCase):
    def setUp(self):
        self.x0 = np.array([-0.1, 0.1])
        self.x1 = np.array([-0.5, 0.5])
        self.x_sad = np.array([0.0, 0.0])
        self.x_min = np.array([-1 / 2 ** 0.5, 0.0])
        self.functions = [quartic_double_well, quartic_double_well_derivative, quartic_double_well_second_derivative]

    def test_hef_quartic_double_well(self):
        f, df, d2f = self.functions
        x_sty = hybrid_eigenvector_following(f, self.x0, df, d2f, tolg=1e-7, tolev=1e-7)
        np.testing.assert_allclose(x_sty + 1, self.x_sad + 1)
        x_sty = hybrid_eigenvector_following(f, self.x1, df, d2f, tolg=1e-7, tolev=1e-7)
        np.testing.assert_allclose(x_sty + 1, self.x_sad + 1)

    def test_newton_quartic_double_well(self):
        f, df, d2f = self.functions
        x_sty = newton_raphson_multi(f, self.x0, df, d2f, tol=1e-8)
        np.testing.assert_allclose(x_sty + 1, self.x_sad + 1)
        x_sty = newton_raphson_multi(f, self.x1, df, d2f, tol=1e-8)
        np.testing.assert_allclose(x_sty + 1, self.x_min + 1)

    def test_gradient_descent_quartic_double_well(self):
        f, df, d2f = self.functions
        x_sty = gradient_descent(f, self.x0, df, tol=1e-8)
        np.testing.assert_allclose(x_sty + 1, self.x_min + 1)
        x_sty = gradient_descent(f, self.x1, df, tol=1e-8)
        np.testing.assert_allclose(x_sty + 1, self.x_min + 1)
