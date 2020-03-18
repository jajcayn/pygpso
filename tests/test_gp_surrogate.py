"""
Set of tests for GP surrogate and related code.
"""

import os
import unittest
from copy import deepcopy
from shutil import rmtree

import gpflow
import numpy as np
from gpso.gp_surrogate import (
    DUPLICATE_TOLERANCE,
    GPListOfPoints,
    GPPoint,
    GPSurrogate,
    PointLabels,
)
from gpso.utils import JSON_EXT
from scipy.special import erfcinv


class TestGPListOfPoints(unittest.TestCase):

    TEMP_FILENAME = "test"

    def _make_random_gp_points(self, n_points=1, ndim=2):
        points = []
        for _ in range(n_points):
            points.append(
                GPPoint(
                    normed_coord=np.random.rand(ndim),
                    score_mu=np.random.rand(),
                    score_sigma=np.random.rand(),
                    score_ucb=np.random.rand(),
                    label=PointLabels.evaluated,
                )
            )
        return points

    def test_init(self):
        # test empty init
        l1 = GPListOfPoints()
        self.assertTrue(isinstance(l1, GPListOfPoints))
        self.assertEqual(len(l1), 0)
        # test init with points
        p = self._make_random_gp_points(n_points=1, ndim=2)
        l2 = GPListOfPoints(p)
        self.assertTrue(isinstance(l2, GPListOfPoints))
        self.assertEqual(len(l2), 1)
        self.assertEqual(p[0], l2[0])

    def test_append(self):
        N_POINTS = 3
        NEW_POINTS = 2
        gp_list = GPListOfPoints(self._make_random_gp_points(n_points=N_POINTS))
        new_points = self._make_random_gp_points(n_points=NEW_POINTS)
        for point in new_points:
            gp_list.append(point)
        self.assertTrue(isinstance(gp_list, GPListOfPoints))
        self.assertEqual(len(gp_list), N_POINTS + NEW_POINTS)
        for idx, point in enumerate(new_points):
            self.assertEqual(point, gp_list[N_POINTS + idx])

    def test_append_duplicate(self):
        N_POINTS = 3
        NEW_SCORE = 12.0
        POINT_IDX = 1
        gp_list = GPListOfPoints(self._make_random_gp_points(n_points=N_POINTS))
        # set point which we will duplicate to gp-based
        gp_list[POINT_IDX] = gp_list[POINT_IDX]._replace(
            label=PointLabels.gp_based
        )
        duplicate_point = deepcopy(gp_list[POINT_IDX])
        # assign new score
        duplicate_point = duplicate_point._replace(score_mu=NEW_SCORE)
        # change coordinates a bit, but within tolerance
        duplicate_point = duplicate_point._replace(
            normed_coord=gp_list[POINT_IDX].normed_coord
            + np.array([DUPLICATE_TOLERANCE / 2.0, -DUPLICATE_TOLERANCE / 2.0])
        )

        gp_list.append(duplicate_point)
        self.assertEqual(len(gp_list), N_POINTS)
        self.assertEqual(duplicate_point, gp_list[POINT_IDX])
        self.assertEqual(gp_list[POINT_IDX].score_mu, NEW_SCORE)

    def test_save_load(self):
        N_POINTS = 3
        gp_list = GPListOfPoints(self._make_random_gp_points(n_points=N_POINTS))
        gp_list.save(self.TEMP_FILENAME)
        loaded = GPListOfPoints.from_file(self.TEMP_FILENAME)
        self.assertTrue(isinstance(loaded, GPListOfPoints))
        self.assertListEqual(gp_list, loaded)
        os.remove(self.TEMP_FILENAME + JSON_EXT)

    def test_find_by_coords(self):
        N_POINTS = 10
        POINT_IDX = 7
        gp_list = GPListOfPoints(self._make_random_gp_points(n_points=N_POINTS))
        coords_to_seek = gp_list[POINT_IDX].normed_coord
        found = gp_list.find_by_coords(coords_to_seek)
        self.assertTrue(isinstance(found, GPPoint))
        self.assertEqual(found, gp_list[POINT_IDX])
        not_found = gp_list.find_by_coords(
            np.random.rand(*coords_to_seek.shape)
        )
        self.assertEqual(not_found, None)


class TestGPSurrogate(unittest.TestCase):

    TEMP_FOLDER = "test_gpsurr"

    def _create_gpsurrogate(self, init_points=0, seed=None):
        if init_points > 0:
            points = []
            for i in range(init_points):
                if seed is not None:
                    np.random.seed(seed + i)
                points.append(
                    GPPoint(
                        normed_coord=np.random.rand(2),
                        score_mu=np.random.rand(),
                        score_sigma=np.random.rand(),
                        score_ucb=np.random.rand(),
                        label=PointLabels(
                            np.random.choice([1, 2], p=[0.8, 0.2])
                        ),
                    )
                )
        else:
            points = None
        self.gp_surr = GPSurrogate(
            gp_kernel=gpflow.kernels.Matern52(),
            gp_meanf=gpflow.mean_functions.Constant(),
            gauss_likelihood_sigma=1e-3,
            varsigma=erfcinv(0.01),
            points=points,
        )

    def test_init(self):
        # init empty
        self._create_gpsurrogate(init_points=0)
        self.assertTrue(isinstance(self.gp_surr, GPSurrogate))
        self.assertTrue(isinstance(self.gp_surr.points, GPListOfPoints))
        self.assertEqual(len(self.gp_surr.points), 0)
        # init with points
        NUM_INIT_POINTS = 10
        self._create_gpsurrogate(init_points=NUM_INIT_POINTS)
        self.assertTrue(isinstance(self.gp_surr, GPSurrogate))
        self.assertTrue(isinstance(self.gp_surr.points, GPListOfPoints))
        self.assertEqual(len(self.gp_surr.points), NUM_INIT_POINTS)

    def test_properties(self):
        NUM_INIT_POINTS = 10
        NUM_EVALUATED = 6
        self._create_gpsurrogate(init_points=NUM_INIT_POINTS, seed=42)
        # test number of evaluated and GP-based points
        self.assertEqual(self.gp_surr.num_evaluated, NUM_EVALUATED)
        self.assertEqual(
            self.gp_surr.num_gp_based, NUM_INIT_POINTS - NUM_EVALUATED
        )

        # test highest score
        HIGHEST_SCORE = 0.9263351350591756
        HIGHEST_COORDS = np.array([0.30096446, 0.24706183])
        highest = self.gp_surr.highest_score
        self.assertTrue(isinstance(highest, GPPoint))
        self.assertEqual(highest.label, PointLabels.evaluated)
        self.assertEqual(highest.score_mu, HIGHEST_SCORE)
        np.testing.assert_almost_equal(highest.normed_coord, HIGHEST_COORDS)

    def test_training_data(self):
        NUM_INIT_POINTS = 10
        NUM_EVALUATED = 6
        self._create_gpsurrogate(init_points=NUM_INIT_POINTS, seed=42)
        x, y = self.gp_surr.current_training_data
        self.assertTupleEqual(x.shape, (NUM_EVALUATED, 2))
        self.assertTupleEqual(y.shape, (NUM_EVALUATED,))

        x = self.gp_surr.gp_based_coords
        self.assertTupleEqual(x.shape, (NUM_INIT_POINTS - NUM_EVALUATED, 2))

    def test_gp_train(self):
        NUM_INIT_POINTS = 10
        self._create_gpsurrogate(init_points=NUM_INIT_POINTS, seed=42)
        x_train, y_train = self.gp_surr.current_training_data
        self.gp_surr._gp_train(x=x_train, y=y_train[:, np.newaxis])
        EXP_MEAN = 0.61633117
        EXP_VAR = 0.06010023
        mean, var = self.gp_surr.gpr_model.predict_y(np.array([[0.5, 0.5]]))
        self.assertEqual(float(np.around(mean, decimals=8)), EXP_MEAN)
        self.assertEqual(float(np.around(var, decimals=8)), EXP_VAR)

    def test_gp_train_and_predict(self):
        NUM_INIT_POINTS = 10
        PREDICT_AT = np.array([[0.5, 0.5]])
        EXP_MEAN = 0.61633117
        EXP_VAR = 0.06010023
        self._create_gpsurrogate(init_points=NUM_INIT_POINTS, seed=42)
        x_train, y_train = self.gp_surr.current_training_data
        self.gp_surr._gp_train(x=x_train, y=y_train[:, np.newaxis])
        self.gp_surr.gp_predict(PREDICT_AT)
        # test if point was appended
        self.assertEqual(len(self.gp_surr.points), NUM_INIT_POINTS + 1)
        appended_point = self.gp_surr.points[-1]
        # test point attributes
        np.testing.assert_equal(
            appended_point.normed_coord, PREDICT_AT.squeeze()
        )
        self.assertEqual(
            float(np.around(appended_point.score_mu, decimals=8)), EXP_MEAN
        )
        self.assertEqual(
            float(np.around(appended_point.score_sigma, decimals=8)), EXP_VAR
        )

    def test_gp_eval_best_ucb(self):
        NUM_INIT_POINTS = 10
        PREDICT_AT = np.array([[0.5, 0.5], [0.5, 0.3]])
        EXP_MEAN = 0.61633117
        EXP_VAR = 0.06010023
        self._create_gpsurrogate(init_points=NUM_INIT_POINTS, seed=42)
        EXP_UCB = np.around(
            EXP_MEAN + self.gp_surr.gp_varsigma * EXP_VAR, decimals=8
        )
        x_train, y_train = self.gp_surr.current_training_data
        self.gp_surr._gp_train(x=x_train, y=y_train[:, np.newaxis])
        best_score = self.gp_surr.gp_eval_best_ucb(PREDICT_AT)
        self.assertEqual(float(np.around(best_score[0], decimals=8)), EXP_MEAN)
        self.assertEqual(float(np.around(best_score[1], decimals=8)), EXP_VAR)
        self.assertEqual(float(np.around(best_score[2], decimals=8)), EXP_UCB)
        # assert no point was added
        self.assertEqual(len(self.gp_surr.points), NUM_INIT_POINTS)

    def test_save_load(self):
        NUM_INIT_POINTS = 10
        # create class and train the GPR
        self._create_gpsurrogate(init_points=NUM_INIT_POINTS, seed=42)
        x_train, y_train = self.gp_surr.current_training_data
        self.gp_surr._gp_train(x=x_train, y=y_train[:, np.newaxis])
        # save class (model and list of points)
        self.gp_surr.save(self.TEMP_FOLDER)

        # load class
        loaded = GPSurrogate.from_saved(self.TEMP_FOLDER)
        self.assertTrue(isinstance(loaded, GPSurrogate))
        # assert GPR parameters
        self.assertDictEqual(
            gpflow.utilities.parameter_dict(self.gp_surr.gpr_model),
            gpflow.utilities.parameter_dict(loaded.gpr_model),
        )
        # do test prediction with GPR model
        test_points = [
            point
            for point in self.gp_surr.points
            if point.label == PointLabels.evaluated
        ]
        x = np.array([point.normed_coord for point in test_points])
        mean_orig, var_orig = self.gp_surr.gpr_model.predict_y(x)
        mean_loaded, var_loaded = loaded.gpr_model.predict_y(x)
        np.testing.assert_equal(mean_orig.numpy(), mean_loaded.numpy())
        np.testing.assert_equal(var_orig.numpy(), var_loaded.numpy())
        # assert varsigma, likelihood and points
        self.assertEqual(self.gp_surr.gp_varsigma, loaded.gp_varsigma)
        self.assertEqual(self.gp_surr.gp_lik_sigma, loaded.gp_lik_sigma)
        self.assertListEqual(self.gp_surr.points, loaded.points)

        # remove test dir
        rmtree(self.TEMP_FOLDER)


if __name__ == "__main__":
    unittest.main()
