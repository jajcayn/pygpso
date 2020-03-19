"""
Tests for callbacks.
"""

import os
import unittest
from shutil import rmtree

import numpy as np
import pytest
from gpso import GPSOptimiser, ParameterSpace
from gpso.callbacks import (
    GPFlowCheckpoints,
    PostIterationPlotting,
    PostUpdateLogging,
    PreFinaliseSave,
)
from gpso.optimisation import CallbackTypes, GPSOCallback
from gpso.utils import JSON_EXT, LOG_EXT, PKL_EXT, set_logger

TEMP_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")


class DummyCallback(GPSOCallback):
    callback_type = CallbackTypes.post_initialise


class TestBaseCallback(unittest.TestCase):
    def test_init(self):
        # first test that assertion is raised when callback_type is None
        with pytest.raises(AssertionError):
            callback = GPSOCallback()
        # now test actual callback with properly defined callback_type
        callback = DummyCallback()
        self.assertTrue(isinstance(callback, GPSOCallback))
        self.assertEqual(callback.callback_type, CallbackTypes.post_initialise)
        self.assertTrue(hasattr(callback, "run"))
        self.assertTrue(callable(callback.run))


class TestCallbacks(unittest.TestCase):
    callbacks = [
        PostIterationPlotting(
            filename_pattern=os.path.join(
                TEMP_FOLDER, "post_iteration_callback"
            ),
            marginal_percentile=0.1,
            from_iteration=12,
        ),
        PostUpdateLogging(),
        PreFinaliseSave(path=TEMP_FOLDER),
        GPFlowCheckpoints(path=os.path.join(TEMP_FOLDER, "checkpoints")),
    ]
    LOG_FILENAME = f"log{LOG_EXT}"
    PARAM_SPACE_FILENAME = f"parameter_space{PKL_EXT}"
    LIST_OF_POINTS_FILRNAME = f"points{JSON_EXT}"
    GPR_MODEL_FILENAME = f"GPRmodel{PKL_EXT}"
    GPR_INFO_FILENAME = f"GPRinfo{JSON_EXT}"

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
        # make directory
        os.makedirs(TEMP_FOLDER)
        os.makedirs(os.path.join(TEMP_FOLDER, "checkpoints"))
        X_BOUNDS = [-3, 5]
        Y_BOUNDS = [-3, 3]
        set_logger(log_filename=os.path.join(TEMP_FOLDER, cls.LOG_FILENAME))

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
            callbacks=cls.callbacks,
        )
        opt.run(cls._obj_func)
        # save optimised problem
        cls.opt_done = opt

    @classmethod
    def tearDownClass(cls):
        """
        Remove temp folder - we do not need it.
        """
        rmtree(TEMP_FOLDER)

    def _check_file_exists_and_remove(self, filename):
        file_ = os.path.join(TEMP_FOLDER, filename)
        self.assertTrue(os.path.exists(file_))
        os.remove(file_)

    def test_callback(self):
        # first test logging callback
        log = os.path.join(TEMP_FOLDER, self.LOG_FILENAME)
        self.assertTrue(os.path.exists(log))
        # remove log file
        os.remove(log)

        # now test pre-finalise saving
        self._check_file_exists_and_remove(self.PARAM_SPACE_FILENAME)
        self._check_file_exists_and_remove(self.LIST_OF_POINTS_FILRNAME)
        self._check_file_exists_and_remove(self.GPR_MODEL_FILENAME)
        self._check_file_exists_and_remove(self.GPR_INFO_FILENAME)

        # test checkpoints
        self.assertTrue(
            os.path.exists(os.path.join(TEMP_FOLDER, "checkpoints"))
        )
        num_checkpoints = len(
            os.listdir(os.path.join(TEMP_FOLDER, "checkpoints"))
        )
        # assert number of files - should 2x number of checkpoints + 1
        self.assertEqual(
            num_checkpoints, 2 * self.opt_done.callbacks[3].max_to_keep + 1
        )
        rmtree(os.path.join(TEMP_FOLDER, "checkpoints"))

        # now test plots after each iteration
        n_iterations = self.opt_done.callbacks[0].iterations_counter
        from_iter = self.opt_done.callbacks[0].from_iteration
        extension = self.opt_done.callbacks[0].plot_ext
        all_files = os.listdir(TEMP_FOLDER)
        # assert number of plots
        self.assertEqual(len(all_files), (n_iterations - from_iter) * 3)
        # assert all are png
        self.assertTrue(all(fname.endswith(extension) for fname in all_files))


if __name__ == "__main__":
    unittest.main()
