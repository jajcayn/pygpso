"""
Test for the full optimisation run.
"""

import unittest

import numpy as np

from gpso.optimisation import GPSOptimiser
from gpso.param_space import ParameterSpace


class TestGPSOptimiser(unittest.TestCase):

    X_BOUNDS = [-3, 5]
    Y_BOUNDS = [-3, 3]
    BEST_COORDS = np.array([0.23525377, 0.68518519])
    BEST_SCORE = 8.10560594

    @staticmethod
    def _obj_func(point):
        """
        Objective function to optimise for. See paper for reference.
        """
        x, y = point
        ct = np.cos(np.pi / 4)
        st = np.sin(np.pi / 4)
        xn = ct * x + st * y
        yn = ct * y - st * x
        x = xn
        y = yn
        return (
            3 * (1 - x) ** 2.0 * np.exp(-(x ** 2) - (y + 1) ** 2)
            - 10 * (x / 5.0 - x ** 3 - y ** 5) * np.exp(-(x ** 2) - y ** 2)
            - 1 / 3 * np.exp(-((x + 1) ** 2) - y ** 2)
        )

    # WARNING: runs approx. 13 seconds
    def test_optimise(self):
        space = ParameterSpace(
            parameter_names=["x", "y"],
            parameter_bounds=[self.X_BOUNDS, self.Y_BOUNDS],
        )
        opt = GPSOptimiser(
            parameter_space=space,
            exploration_method="tree",
            exploration_depth=3,
            budget=50,
            stopping_condition="evaluations",
            update_cycle=1,
            gp_lik_sigma=1.0e-3,
            n_workers=1,
        )
        best_point = opt.run(self._obj_func)
        np.testing.assert_almost_equal(
            self.BEST_COORDS, best_point.normed_coord
        )
        self.assertEqual(
            np.around(best_point.score_mu, decimals=8), self.BEST_SCORE
        )


if __name__ == "__main__":
    unittest.main()
