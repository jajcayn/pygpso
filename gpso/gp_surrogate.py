"""
Surrogate of the objective function using GPR.
"""
import json
import logging
from collections import namedtuple
from enum import Enum, unique

import numpy as np

import gpflow

from .utils import JSON_EXT, load_json

GP_TRAIN_MAX_ITER = 100
DUPLICATE_TOLERANCE = 1.0e-12
GPFLOW_EXT = ".gpflow"


class GPPoint(
    namedtuple(
        "GPPoint",
        ["normed_coord", "score_mu", "score_sigma", "score_ucb", "label"],
    )
):
    """
    Tuple for storing GPR training data - coordinates as per X and scores as
    per Y, with overloaded equality to accommodate for numpy array comparison.
    """

    def __eq__(self, other):
        return all([np.all(a == b for a, b in zip(self, other))])


class GPListOfPoints(list):
    """
    Subclass of list that overwrites append method.
    """

    @classmethod
    def from_file(cls, filename):
        """
        Load list of GP points from json file.

        :param filename: filename for the json file
        :type filename: str
        """
        if not filename.endswith(JSON_EXT):
            filename += JSON_EXT
        loaded = load_json(filename)
        point_list = []
        for point in loaded:
            # restore original types
            point["normed_coord"] = np.array(point["normed_coord"])
            point["label"] = PointLabels[point["label"]]
            point_list.append(GPPoint(**point))
        return cls(point_list)

    def __init__(self, *args, **kwargs):
        if args:
            assert all(isinstance(it, GPPoint) for it in args[0])
        super().__init__(*args, **kwargs)

    def append(self, object):
        assert isinstance(object, GPPoint)
        append_flag = True
        # check for duplicates in the list
        for idx in range(len(self)):
            if (
                np.linalg.norm(self[idx].normed_coord - object.normed_coord)
                < DUPLICATE_TOLERANCE
            ):
                # if it is a duplicate we will not append it
                append_flag = False
                # if once evaluated, do not overwrite: evaluated will be the
                # same, GP-based does not make sense
                if self[idx].label == PointLabels.evaluated:
                    continue
                # update current point
                logging.warning(
                    "Duplicate found! Overwriting with newer point..."
                )
                self[idx] = object
        if append_flag:
            super().append(object)

    def find_by_coords(self, coords):
        """
        Find and return point by its coordinates.

        :param coords: coords to find in the list of points
        :type coords: np.ndarray
        :return: point if found, or None if not
        :rtype: `GPPoint`|None
        """
        for point in self:
            if (
                np.linalg.norm(point.normed_coord - coords)
                < DUPLICATE_TOLERANCE
            ):
                return point

    def save(self, filename):
        """
        Save list of GPPoints as a json.

        :param filename: filename for the json file
        :type filename: str
        """
        if not filename.endswith(JSON_EXT):
            filename += JSON_EXT
        serialised = [point._asdict() for point in self]
        for point in serialised:
            # need to make all objects json-serialisable
            point["normed_coord"] = point["normed_coord"].tolist()
            point["label"] = point["label"].name
        with open(filename, "w") as file_handler:
            file_handler.write(json.dumps(serialised))


@unique
class PointLabels(Enum):
    """
    Helper for leaf labels - empty, evaluated using LSBM or GPR-based.
    """

    not_assigned = 0  # when no score is yet known
    evaluated = 1  # objective function is evaluated at the centre of the leaf
    gp_based = 2  # score was obtained using UCB via GP


class GPSurrogate:
    """
    Class handles GP surrogate of the objective function surface.
    """

    @classmethod
    def from_saved(cls, folder):
        raise NotImplementedError("Not supported right now")

    def __init__(
        self,
        gp_kernel,
        gp_meanf,
        gauss_likelihood_sigma,
        varsigma,
        points=None,
        gpr_model=None,
    ):
        # init GP model
        self.gpr_model = gpr_model
        self.gp_varsigma = varsigma
        assert isinstance(gp_kernel, gpflow.kernels.Kernel)
        self.gp_kernel = gp_kernel
        assert isinstance(gp_meanf, (gpflow.mean_functions.MeanFunction, None))
        self.gp_meanf = gp_meanf
        self.gp_lik_sigma = gauss_likelihood_sigma

        if points is None:
            points = []
        self.points = GPListOfPoints(points)

    @property
    def num_evaluated(self):
        """
        Return number of evaluated points.

        :rtype: int
        """
        return sum(
            map(lambda point: point.label == PointLabels.evaluated, self.points)
        )

    @property
    def num_gp_based(self):
        """
        Return number of GP-based points.

        :rtype: int
        """
        return sum(
            map(lambda point: point.label == PointLabels.gp_based, self.points)
        )

    @property
    def highest_score(self):
        """
        Return point with highest score.

        :rtype: `GPPoint`
        """
        if self.num_evaluated > 0:
            return sorted(
                [
                    point
                    for point in self.points
                    if point.label == PointLabels.evaluated
                ],
                key=lambda point: point.score_mu,
                reverse=True,
            )[0]

    @property
    def highest_ucb(self):
        """
        Return point with highest UCB.

        :rtype: `GPPoint`
        """
        if self.num_gp_based > 0:
            return sorted(
                [
                    point
                    for point in self.points
                    if point.label == PointLabels.gp_based
                ],
                key=lambda point: point.score_ucb,
                reverse=True,
            )[0]

    @property
    def current_training_data(self):
        """
        Return current training data (i.e. points marked as evaluated)
        """
        eval_points = [
            point
            for point in self.points
            if point.label == PointLabels.evaluated
        ]
        x = np.array([point.normed_coord for point in eval_points])
        y = np.array([point.score_mu for point in eval_points])
        return x, y

    @property
    def gp_based_coords(self):
        """
        Return coordinates of GP-based points.
        """
        gp_points = [
            point
            for point in self.points
            if point.label == PointLabels.gp_based
        ]
        x = np.array([point.normed_coord for point in gp_points])
        return x

    def _gp_train(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert x.ndim == 2 and y.ndim == 2
        # init model
        self.gpr_model = gpflow.models.GPR(
            data=(x, y),
            kernel=self.gp_kernel,
            mean_function=self.gp_meanf,
            noise_variance=self.gp_lik_sigma,
        )
        optimizer = gpflow.optimizers.Scipy()

        def objective_closure():
            return -self.gpr_model.log_marginal_likelihood()

        opt_logs = optimizer.minimize(
            objective_closure,
            self.gpr_model.trainable_variables,
            options=dict(maxiter=GP_TRAIN_MAX_ITER),
        )
        if not opt_logs.success:
            logging.error(
                "Something went wrong, optimisation was not successful; "
                f"log: {opt_logs}"
            )

    def append(self, coords, scores):
        """
        Append evaluated points using the objective function. These are the
        training points for GPR.

        :param coords: normalised coordinates
        :type coords: np.ndarray
        :param scores: scores from the objective function
        :type scores: np.ndarray
        """
        assert coords.ndim == 2
        assert scores.ndim == 1
        # assert number of appending points is the same
        assert coords.shape[0] == scores.shape[0]
        for idx in range(coords.shape[0]):
            self.points.append(
                GPPoint(
                    normed_coord=coords[idx, :],
                    score_mu=scores[idx],
                    score_sigma=0.0,
                    score_ucb=0.0,
                    label=PointLabels.evaluated,
                )
            )

    def gp_predict(self, normed_coords):
        """
        Predict points at `normed_coords` with trained GPR and append to list
        of points.

        :param normed_coords: normalised coordinates at which to predict
        :type normed_coords: np.ndarray
        """
        assert isinstance(self.gpr_model, gpflow.models.GPModel)
        # predict and include the noise variance
        mean, var = self.gpr_model.predict_y(normed_coords)
        # append to points
        for idx in range(normed_coords.shape[0]):
            self.points.append(
                GPPoint(
                    normed_coord=normed_coords[idx, :],
                    score_mu=float(mean[idx, 0]),
                    score_sigma=float(var[idx, 0]),
                    score_ucb=float(
                        mean[idx, 0] + self.gp_varsigma * var[idx, 0]
                    ),
                    label=PointLabels.gp_based,
                )
            )

    def gp_eval_best_ucb(self, normed_coords):
        """
        Predict points at `normed_coords` with trained GPR and select one best
        point with UCB and return mean, var, and UCB for that point.

        :param normed_coords: normalised coordinates at which to predict
        :type normed_coords: np.ndarray
        :return: mean, var and UCB for the best score
        :rtype: (float, float, float)
        """
        assert isinstance(self.gpr_model, gpflow.models.GPModel)
        # predict and include the noise variance
        mean, var = self.gpr_model.predict_y(normed_coords)
        ucb = mean + self.gp_varsigma * var
        best_ucb = np.argmax(ucb)
        return float(mean[best_ucb]), float(var[best_ucb]), float(ucb[best_ucb])

    def gp_update(self):
        """
        Retrain GP with current evaluated samples.
        """
        # retrain the GPR
        x_train, y_train = self.current_training_data
        self._gp_train(x=x_train, y=y_train[:, np.newaxis])
        # reevaluate GP-based samples
        if self.num_gp_based > 0:
            self.gp_predict(self.gp_based_coords)

    def save(self, folder):
        """
        Save GPFlow model and list of points.
        """
        # make_dirs(folder)
        # # save list of points
        # self.points.save(filename=os.path.join(folder, f"points{JSON_EXT}"))
        # # save GPR model
        # checkpoint = tf.train.Checkpoint(model=self.gpr_model)
        # checkpoint.save(file_prefix=os.path.join(folder, f"ckpt{GPFLOW_EXT}"))
        raise NotImplementedError("GPFlow cannot save models right now")
