import unittest

import numpy as np
from scipy.optimize import rosen, rosen_der, rosen_hess

from local_optimisation import newton_raphson_multi, gradient_descent, bfgs


class TestLocalOptimisersRosenbrock(unittest.TestCase):
    def setUp(self):
        self.x0 = np.array([-1.0, 2.0])
        self.xi = np.array([1.0, 1.0])
        self.tol = 1e-10
        self.functions = [rosen, rosen_der, rosen_hess]

    def test_newton_raphson_rosenbrock(self):
        f, df, d2f = self.functions
        x = newton_raphson_multi(f, self.x0, df, d2f, tol=self.tol)
        np.testing.assert_allclose(x, self.xi)

    def test_gradient_descent_rosenbrock(self):
        f, df, d2f = self.functions
        x = gradient_descent(f, self.x0, df, tol=self.tol)
        np.testing.assert_allclose(x, self.xi)

    def test_bfgs_rosenbrock(self):
        f, df, d2f = self.functions
        x = bfgs(f, self.x0, df, xtol=self.tol, gtol=self.tol)
        np.testing.assert_allclose(x, self.xi)
