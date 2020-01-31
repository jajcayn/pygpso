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
    # very slightly different scores and coordinates
    BEST_COORDS_v1 = np.array([0.23525377, 0.68518519])
    BEST_SCORE_v1 = 8.10560594
    BEST_COORDS_v2 = np.array([0.2283951, 0.68518519])
    BEST_SCORE_v2 = 8.07790009

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
    def test_optimise_v1(self):
        """
        With tree method, for 50 evaluations, 1 worker and default init sample.
        """
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
            self.BEST_COORDS_v1, best_point.normed_coord
        )
        self.assertEqual(
            np.around(best_point.score_mu, decimals=8), self.BEST_SCORE_v1
        )

    def test_optimise_v2(self):
        """
        With sample method, for 13 iterations, 2 workers and custom init sample.
        """
        np.random.seed(42)
        space = ParameterSpace(
            parameter_names=["x", "y"],
            parameter_bounds=[self.X_BOUNDS, self.Y_BOUNDS],
        )
        opt = GPSOptimiser(
            parameter_space=space,
            exploration_method="sample",
            exploration_depth=3,
            budget=14,
            stopping_condition="iterations",
            update_cycle=1,
            gp_lik_sigma=1.0e-3,
            n_workers=2,
        )
        best_point = opt.run(
            self._obj_func,
            init_samples=np.array(
                [[-1.0, 0.0], [1.0, 0.0], [-1.5, 1], [1.5, 1]]
            ),
        )
        np.testing.assert_almost_equal(
            self.BEST_COORDS_v2, best_point.normed_coord
        )
        self.assertEqual(
            np.around(best_point.score_mu, decimals=8), self.BEST_SCORE_v2
        )


if __name__ == "__main__":
    unittest.main()
