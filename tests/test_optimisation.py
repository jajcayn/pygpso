"""
Test for the full optimisation run.
"""

import os
import unittest
from shutil import rmtree

import numpy as np
import pytest
from gpso import GPSOptimiser, ParameterSpace
from gpso.gp_surrogate import GPPoint

TEMP_FOLDER = "temp_optimisation_test"


class TestGPSOptimiser(unittest.TestCase):

    X_BOUNDS = [-3, 5]
    Y_BOUNDS = [-3, 3]
    # very slightly different scores and coordinates
    BEST_COORDS_v1 = np.array([0.23525377, 0.68518519])
    BEST_SCORE_v1 = 8.10560594
    BEST_SCORE_v2 = 6.5

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

    @staticmethod
    def _obj_func_w_return(point):
        """
        Objective function to optimise for. Also returns time series of the
        model.
        """
        x, y = point
        ct = np.cos(np.pi / 4)
        st = np.sin(np.pi / 4)
        xn = ct * x + st * y
        yn = ct * y - st * x
        x = xn
        y = yn
        score = (
            3 * (1 - x) ** 2.0 * np.exp(-(x ** 2) - (y + 1) ** 2)
            - 10 * (x / 5.0 - x ** 3 - y ** 5) * np.exp(-(x ** 2) - y ** 2)
            - 1 / 3 * np.exp(-((x + 1) ** 2) - y ** 2)
        )
        np.random.seed(42)
        dummy_ts = np.random.rand(100, 2)
        return dummy_ts, score

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

    def test_optimise_resume(self):
        """
        Basic optimisation with resume.
        """
        space = ParameterSpace(
            parameter_names=["x", "y"],
            parameter_bounds=[self.X_BOUNDS, self.Y_BOUNDS],
        )
        opt = GPSOptimiser(
            parameter_space=space,
            exploration_method="tree",
            exploration_depth=3,
            budget=25,
            stopping_condition="evaluations",
            update_cycle=1,
            gp_lik_sigma=1.0e-3,
            n_workers=1,
        )
        # optimise for 25 evaluations
        _ = opt.run(self._obj_func)
        # resume for additional 25 evaluations
        best_point = opt.resume_run(additional_budget=25)
        np.testing.assert_almost_equal(
            self.BEST_COORDS_v1, best_point.normed_coord
        )
        self.assertEqual(
            np.around(best_point.score_mu, decimals=8), self.BEST_SCORE_v1
        )

    def test_optimiser_save_resume(self):
        """
        Basic optimisation with saving and loading optimiser in between.
        """
        os.makedirs(TEMP_FOLDER)
        space = ParameterSpace(
            parameter_names=["x", "y"],
            parameter_bounds=[self.X_BOUNDS, self.Y_BOUNDS],
        )
        opt = GPSOptimiser(
            parameter_space=space,
            exploration_method="tree",
            exploration_depth=3,
            budget=25,
            stopping_condition="evaluations",
            update_cycle=1,
            gp_lik_sigma=1.0e-3,
            n_workers=1,
        )
        # optimise for 25 evaluations
        _ = opt.run(self._obj_func)
        # save
        opt.save_state(TEMP_FOLDER)
        # resume from saved for another 25 evaluations
        best_point, opt_loaded = GPSOptimiser.resume_from_saved(
            TEMP_FOLDER, additional_budget=25, objective_function=self._obj_func
        )
        self.assertTrue(isinstance(opt_loaded, GPSOptimiser))
        np.testing.assert_almost_equal(
            self.BEST_COORDS_v1, best_point.normed_coord
        )
        self.assertEqual(
            np.around(best_point.score_mu, decimals=8), self.BEST_SCORE_v1
        )
        rmtree(TEMP_FOLDER)

    def test_optimise_v2(self):
        """
        With sample method, for 13 iterations, 2 workers and custom init sample.
        """
        space = ParameterSpace(
            parameter_names=["x", "y"],
            parameter_bounds=[self.X_BOUNDS, self.Y_BOUNDS],
        )
        opt = GPSOptimiser(
            parameter_space=space,
            exploration_method="sample",
            exploration_depth=3,
            budget=12,
            stopping_condition="iterations",
            update_cycle=1,
            gp_lik_sigma=1.0e-3,
            n_workers=4,
        )
        best_point = opt.run(
            self._obj_func,
            init_samples=np.array(
                [[-1.0, 0.0], [1.0, 0.0], [-1.5, 1], [1.5, 1]]
            ),
            eval_repeats=4,
            seed=42,
        )
        # non-deterministic test, test only whether it runs
        self.assertTrue(isinstance(best_point, GPPoint))
        self.assertGreaterEqual(best_point.score_mu, self.BEST_SCORE_v2)

    def test_optimise_v3(self):
        """
        With tree method, for 50 evaluations, 1 worker and default init sample
        and saver.
        """
        HDF_FILENAME = "test.h5"
        _ = pytest.importorskip("pandas")
        _ = pytest.importorskip("tables")
        from gpso.saving_helper import TableSaver

        space = ParameterSpace(
            parameter_names=["x", "y"],
            parameter_bounds=[self.X_BOUNDS, self.Y_BOUNDS],
        )
        # init saver
        saver = TableSaver(HDF_FILENAME)
        opt = GPSOptimiser(
            parameter_space=space,
            exploration_method="tree",
            exploration_depth=3,
            budget=50,
            stopping_condition="evaluations",
            update_cycle=1,
            gp_lik_sigma=1.0e-3,
            n_workers=1,
            saver=saver,
        )
        best_point = opt.run(self._obj_func_w_return, eval_repeats=4)
        # properly close
        saver.close()

        np.testing.assert_almost_equal(
            self.BEST_COORDS_v1, best_point.normed_coord
        )
        self.assertEqual(
            np.around(best_point.score_mu, decimals=8), self.BEST_SCORE_v1
        )
        # just test whether file was created, the internals are tested elsewhere
        self.assertTrue(os.path.exists(HDF_FILENAME))
        os.remove(HDF_FILENAME)


if __name__ == "__main__":
    unittest.main()
