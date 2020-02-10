"""
Set of tests for plotting helpers for GPSO.
"""

import os
import unittest
from shutil import rmtree

import numpy as np
import pytest
from gpso.optimisation import GPSOptimiser
from gpso.param_space import ParameterSpace
from gpso.plotting import (
    plot_conditional_surrogate_distributions,
    plot_parameter_marginal_distributions,
)


class TestPlotting(unittest.TestCase):
    """
    Just save the plots and check whether the file exists. Do not actually
    check the figure as that would be problematic.
    """

    TEMP_FOLDER = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "temp"
    )

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

    @classmethod
    def setUpClass(cls):
        """
        Optimise example problem as test starts. We need to plot something..
        """
        X_BOUNDS = [-3, 5]
        Y_BOUNDS = [-3, 3]
        space = ParameterSpace(
            parameter_names=["x", "y"], parameter_bounds=[X_BOUNDS, Y_BOUNDS],
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
        opt.run(cls._obj_func)
        # save optimised problem
        cls.opt_done = opt
        # make directory
        os.makedirs(cls.TEMP_FOLDER)

    @classmethod
    def tearDownClass(cls):
        """
        Remove temp folder - we do not need it.
        """
        rmtree(cls.TEMP_FOLDER)

    def test_plot_ternary_tree(self):
        _ = pytest.importorskip("igraph")

        from gpso.plotting import plot_ternary_tree

        FILENAME = os.path.join(self.TEMP_FOLDER, "tree.png")
        plot_ternary_tree(
            self.opt_done.param_space, fname=FILENAME,
        )
        self.assertTrue(os.path.exists(FILENAME))

    def test_plot_parameter_marginal_distributions(self):
        for plot_type in ["kde", "hist"]:
            FILENAME = os.path.join(
                self.TEMP_FOLDER, f"param_marginals_{plot_type}.png"
            )
            plot_parameter_marginal_distributions(
                self.opt_done,
                percentile=0.1,
                plot_type=plot_type,
                fname=FILENAME,
            )
            self.assertTrue(os.path.exists(FILENAME))

        with pytest.raises(ValueError):
            plot_parameter_marginal_distributions(
                self.opt_done, percentile=0.1, plot_type="abcd", fname=None,
            )

    def test_plot_conditional_surrogate_distributions(self):
        FILENAME = os.path.join(self.TEMP_FOLDER, "cond_surr.png")
        plot_conditional_surrogate_distributions(
            self.opt_done, fname=FILENAME,
        )
        self.assertTrue(os.path.exists(FILENAME))


if __name__ == "__main__":
    unittest.main()
