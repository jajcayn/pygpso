"""
Surrogate of the objective function using GPR.
"""
import json
import logging
import os
import pickle
from collections import namedtuple

import gpflow
import numpy as np
import tensorflow as tf
from scipy.special import erfcinv

from .param_space import NORM_PARAMS_BOUNDS
from .utils import JSON_EXT, PKL_EXT, PointLabels, load_json, make_dirs

GP_TRAIN_MAX_ITER = 100
VGP_TRAIN_ITERATIONS = 10
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


class GPSurrogate:
    """
    Class handles GP surrogate of the objective function surface.
    """

    POINTS_FILE = f"points{JSON_EXT}"
    GPR_FILE = f"GPRmodel{PKL_EXT}"
    GPR_INFO = f"GPRinfo{JSON_EXT}"

    @classmethod
    def from_saved(cls, folder):
        raise NotImplementedError

    def __init__(
        self,
        gp_kernel,
        gp_meanf=None,
        optimiser=gpflow.optimizers.Scipy(),
        varsigma=erfcinv(0.01),
        points=None,
        gpflow_model=None,
    ):
        """
        :param gp_kernel: kernel for GP to be used
        :type gp_kernel: `gpflow.kernels.Kernel`
        :param gp_meanf: mean function for GP to be used
        :type gp_meanf: `gpflow.mean_functions.MeanFunction`|None
        :param optimiser: optimiser to be used for training GP model
        :type optimiser: any supported class, must implement callable
        :param varsigma: expected probability that UCB < f; it controls how
            "optimistic" we are during the exploration step; at a point x
            evaluated using GP, the UCB will be: mu(x) + varsigma*sigma(x);
            varsigma = 1/erfc(p/100) which corresponds to the upper bound of a
                `p` confidence interval for Gaussian likelihood kernel
        :type varsigma: float
        :param points: list of GPPoints
        :type points: list|None|`GPListOfPoints`
        :param gpflow_model: initialised `GPFlow` model, usually None, only used
            when loading from saved
        :type gpflow_model: `gpflow.models.GPModel`|None
        """
        # init GP model
        self.gpflow_model = gpflow_model
        self.gp_varsigma = varsigma
        assert isinstance(gp_kernel, gpflow.kernels.Kernel)
        self.gp_kernel = gp_kernel
        assert isinstance(gp_meanf, (gpflow.mean_functions.MeanFunction, None))
        self.gp_meanf = gp_meanf
        assert hasattr(optimiser, "minimize")
        self.optimiser = optimiser

        self.points = GPListOfPoints(points or list())

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
        # need to redefine for each type of model
        raise NotImplementedError

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
        assert isinstance(self.gpflow_model, gpflow.models.GPModel)
        # predict and include the noise variance
        mean, var = self.gpflow_model.predict_y(normed_coords)
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
        assert isinstance(self.gpflow_model, gpflow.models.GPModel)
        # predict and include the noise variance
        mean, var = self.gpflow_model.predict_y(normed_coords)
        ucb = mean + self.gp_varsigma * var
        best_ucb = np.argmax(ucb)
        return float(mean[best_ucb]), float(var[best_ucb]), float(ucb[best_ucb])

    def gp_update(self):
        """
        Retrain GP with current evaluated samples.
        """
        # retrain the GPR
        x_train, y_train = self.current_training_data
        logging.debug(
            f"Retraining GPR with x data: {x_train}; y data: {y_train}"
        )
        self._gp_train(x=x_train, y=y_train[:, np.newaxis])
        # reevaluate GP-based samples
        if self.num_gp_based > 0:
            self.gp_predict(self.gp_based_coords)

    def _serialise_optimiser(self):
        """
        Serialise optimiser.
        """
        optimiser = self.optimiser.__class__.__name__
        if optimiser == "Scipy":
            return tuple([optimiser])
        elif optimiser == "Adam":
            return (
                optimiser,
                float(self.optimiser.get_config()["learning_rate"]),
            )
        else:
            raise ValueError(f"{optimiser} not currently supported.")

    @staticmethod
    def _deserialse_optimiser(from_json):
        """
        Deserialise optimiser.

        :param from_json: tuple from json saved info
        :type from_json: tuple
        """
        optimiser = from_json[0]
        if optimiser == "Scipy":
            return gpflow.optimizers.Scipy()
        elif optimiser == "Adam":
            return tf.optimizers.Adam(from_json[1])

    def save(self, folder):
        """
        Save GPFlow model and list of points. This is intermediate, "hacky"
        method until GPFlow 2.0 solves the saving problem. Currently, the GPR
        model is saved to pickle but for its recreation one needs to initialise
        model (with the same tree of parameters) and hyperparameters are loaded
        from the file. The kernel and mean-function names are saved to JSON, so
        it won't work with complex kernels and mean-functions.

        :param folder: path to which save the model
        :type folder: str
        """
        raise NotImplementedError


class GPRSurrogate(GPSurrogate):
    """
    Surrogate exploiting vanilla GP Regression model.
    """

    def __init__(
        self,
        gp_kernel,
        gp_meanf=None,
        optimiser=gpflow.optimizers.Scipy(),
        varsigma=erfcinv(0.01),
        gauss_likelihood_sigma=1.0e-3,
        points=None,
        gpflow_model=None,
    ):
        """
        :param gp_lik_sigma: initial std of Gaussian likelihood function (in
            normalised units)
        :type gp_lik_sigma: float
        """
        super().__init__(
            gp_kernel=gp_kernel,
            gp_meanf=gp_meanf,
            optimiser=optimiser,
            varsigma=varsigma,
            points=points,
            gpflow_model=gpflow_model,
        )
        self.gp_lik_sigma = gauss_likelihood_sigma

    @classmethod
    def default(cls):
        """
        Return GPR model with sensible defaults.
        """
        return cls(
            gp_kernel=gpflow.kernels.Matern52(
                lengthscales=np.sum(NORM_PARAMS_BOUNDS) * 0.25, variance=1.0,
            ),
            gp_meanf=gpflow.mean_functions.Constant(0.0),
            optimiser=gpflow.optimizers.Scipy(),
            varsigma=erfcinv(0.01),
            gauss_likelihood_sigma=1.0e-3,
            points=None,
            gpflow_model=None,
        )

    @classmethod
    def from_saved(cls, folder):
        # load points
        points = GPListOfPoints.from_file(os.path.join(folder, cls.POINTS_FILE))
        # get current data as saved
        eval_points = [
            point for point in points if point.label == PointLabels.evaluated
        ]
        x = np.array([point.normed_coord for point in eval_points])
        y = np.array([point.score_mu for point in eval_points])[:, np.newaxis]

        # load GPR info
        gpr_info = load_json(os.path.join(folder, cls.GPR_INFO))
        # recreate kernel
        assert hasattr(gpflow.kernels, gpr_info["gpr_kernel"])
        gp_kernel = getattr(gpflow.kernels, gpr_info["gpr_kernel"])(
            lengthscales=np.ones(gpr_info["gpr_kernel_shape"])
        )
        # recreate mean function
        assert hasattr(gpflow.mean_functions, gpr_info["gpr_meanf"])
        gp_meanf = getattr(gpflow.mean_functions, gpr_info["gpr_meanf"])(
            np.zeros(gpr_info["gpr_meanf_shape"])
        )
        # recreate optimiser
        optimiser = cls._deserialse_optimiser(gpr_info["optimiser"])

        # create placeholder model
        gpflow_model = gpflow.models.GPR(
            data=(x, y),
            kernel=gp_kernel,
            mean_function=gp_meanf,
            noise_variance=gpr_info["gp_likelihood"],
        )
        # load GPR parameters
        with open(os.path.join(folder, cls.GPR_FILE), "rb") as handle:
            gpr_params = pickle.load(handle)
        # assign hyperparameters
        gpflow.utilities.multiple_assign(gpflow_model, gpr_params)
        return cls(
            gp_kernel=gp_kernel,
            gp_meanf=gp_meanf,
            optimiser=optimiser,
            gauss_likelihood_sigma=gpr_info["gp_likelihood"],
            varsigma=gpr_info["gp_varsigma"],
            points=points,
            gpflow_model=gpflow_model,
        )

    def _gp_train(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert x.ndim == 2 and y.ndim == 2

        if self.gpflow_model is None:
            # if None, init model
            self.gpflow_model = gpflow.models.GPR(
                data=(x, y),
                kernel=self.gp_kernel,
                mean_function=self.gp_meanf,
                noise_variance=self.gp_lik_sigma,
            )
        else:
            # just assign new data
            self.gpflow_model.data = (x, y)

        self.optimiser.minimize(
            self.gpflow_model.training_loss,
            self.gpflow_model.trainable_variables,
        )

    def save(self, folder):
        """
        :param folder: path to which save the model
        :type folder: str
        """
        make_dirs(folder)
        # save list of points
        self.points.save(filename=os.path.join(folder, self.POINTS_FILE))
        # hack to untie weak references
        _ = gpflow.utilities.freeze(self.gpflow_model)
        # save GPR model to pickle - now doable
        with open(os.path.join(folder, self.GPR_FILE), "wb") as handle:
            pickle.dump(
                gpflow.utilities.parameter_dict(self.gpflow_model), handle
            )
        # save other info to json
        save_info = {
            "gpr_kernel": self.gpflow_model.kernel.__class__.__name__,
            "gpr_kernel_shape": self.gpflow_model.kernel.lengthscales.shape.as_list(),
            "gpr_meanf": self.gpflow_model.mean_function.__class__.__name__,
            "gpr_meanf_shape": self.gpflow_model.mean_function.parameters[
                0
            ].shape.as_list(),
            "gp_varsigma": self.gp_varsigma,
            "gp_likelihood": self.gp_lik_sigma,
            "optimiser": self._serialise_optimiser(),
        }
        with open(os.path.join(folder, self.GPR_INFO), "w") as handle:
            handle.write(json.dumps(save_info))


class VGPSurrogate(GPSurrogate):
    """
    Surrogate exploiting Variational GP model with arbitrary continuous
    likelihood. VGP uses one optimiser (usually Adam or Scipy) to optimise
    hyperparameters and natural gradients to optimise variational parameters.
    """

    def __init__(
        self,
        gp_kernel,
        gp_meanf=None,
        likelihood=gpflow.likelihoods.Gaussian(variance=1.0e-3),
        optimiser=tf.optimizers.Adam(0.01),
        varsigma=erfcinv(0.01),
        points=None,
        gpflow_model=None,
        natgrad_learning_rate=1.0,
        train_iterations=VGP_TRAIN_ITERATIONS,
    ):
        """
        :param likelihood: likelihood for VGP model
        :type likelihood: `gpflow.likelihoods.base.ScalarLikelihood`
        :param natgrad_learning_rate: step length (gamma) for Natural gradient
        :type natgrad_learning_rate: float
        :param train_iterations: number of iterations for VGP Adam vs NatGrad
            training loop
        :type train_iterations: int
        """
        super().__init__(
            gp_kernel=gp_kernel,
            gp_meanf=gp_meanf,
            optimiser=optimiser,
            varsigma=varsigma,
            points=points,
            gpflow_model=gpflow_model,
        )
        assert isinstance(likelihood, gpflow.likelihoods.base.ScalarLikelihood)
        self.likelihood = likelihood
        self.natgrad_optimiser = gpflow.optimizers.NaturalGradient(
            natgrad_learning_rate
        )
        self.train_iters = train_iterations

    @classmethod
    def from_saved(cls, folder):
        # load points
        points = GPListOfPoints.from_file(os.path.join(folder, cls.POINTS_FILE))
        # get current data as saved
        eval_points = [
            point for point in points if point.label == PointLabels.evaluated
        ]
        x = np.array([point.normed_coord for point in eval_points])
        y = np.array([point.score_mu for point in eval_points])[:, np.newaxis]

        # load GPR info
        vgp_info = load_json(os.path.join(folder, cls.GPR_INFO))
        # recreate kernel
        assert hasattr(gpflow.kernels, vgp_info["vgp_kernel"])
        gp_kernel = getattr(gpflow.kernels, vgp_info["vgp_kernel"])(
            lengthscales=np.ones(vgp_info["vgp_kernel_shape"])
        )
        # recreate mean function
        assert hasattr(gpflow.mean_functions, vgp_info["vgp_meanf"])
        gp_meanf = getattr(gpflow.mean_functions, vgp_info["vgp_meanf"])(
            np.zeros(vgp_info["vgp_meanf_shape"])
        )
        # recreate likelihood
        assert hasattr(gpflow.likelihoods, vgp_info["vgp_likelihood"])
        gp_likelihood = getattr(
            gpflow.likelihoods, vgp_info["vgp_likelihood"]
        )()
        # recreate optimiser
        optimiser = cls._deserialse_optimiser(vgp_info["optimiser"])

        # create placeholder model
        gpflow_model = gpflow.models.VGP(
            data=(x, y),
            kernel=gp_kernel,
            mean_function=gp_meanf,
            likelihood=gp_likelihood,
            num_latent_gps=1,
        )
        # load GPR parameters
        with open(os.path.join(folder, cls.GPR_FILE), "rb") as handle:
            gpr_params = pickle.load(handle)
        # assign hyperparameters
        gpflow.utilities.multiple_assign(gpflow_model, gpr_params)
        return cls(
            gp_kernel=gp_kernel,
            gp_meanf=gp_meanf,
            likelihood=gp_likelihood,
            optimiser=optimiser,
            varsigma=vgp_info["gp_varsigma"],
            points=points,
            gpflow_model=gpflow_model,
            natgrad_learning_rate=vgp_info["vgp_natgrad_lr"],
            train_iterations=vgp_info["vgp_iters"],
        )

    def _gp_train(self, x, y):
        assert x.shape[0] == y.shape[0]
        assert x.ndim == 2 and y.ndim == 2

        if self.gpflow_model is None:
            # if None, init model
            self.gpflow_model = gpflow.models.VGP(
                data=(x, y),
                kernel=self.gp_kernel,
                mean_function=self.gp_meanf,
                likelihood=self.likelihood,
                num_latent_gps=1,
            )
        else:
            # just assign new data
            self.gpflow_model.data = (x, y)

        # training loop
        gpflow.set_trainable(self.gpflow_model.q_mu, False)
        gpflow.set_trainable(self.gpflow_model.q_sqrt, False)

        for i in range(self.train_iters):
            self.natgrad_optimiser.minimize(
                self.gpflow_model.training_loss,
                [(self.gpflow_model.q_mu, self.gpflow_model.q_sqrt)],
            )
            self.optimiser.minimize(
                self.gpflow_model.training_loss,
                self.gpflow_model.trainable_variables,
            )
            logging.debug(
                f"VGP iteration {i+1}. ELBO: {self.gpflow_model.elbo():.04f}"
            )

    def save(self, folder):
        """
        :param folder: path to which save the model
        :type folder: str
        """
        make_dirs(folder)
        # save list of points
        self.points.save(filename=os.path.join(folder, self.POINTS_FILE))
        # hack to untie weak references
        _ = gpflow.utilities.freeze(self.gpflow_model)
        # save GPR model to pickle - now doable
        with open(os.path.join(folder, self.GPR_FILE), "wb") as handle:
            pickle.dump(
                gpflow.utilities.parameter_dict(self.gpflow_model), handle
            )
        # save other info to json
        save_info = {
            "vgp_kernel": self.gpflow_model.kernel.__class__.__name__,
            "vgp_kernel_shape": self.gpflow_model.kernel.lengthscales.shape.as_list(),
            "vgp_meanf": self.gpflow_model.mean_function.__class__.__name__,
            "vgp_meanf_shape": self.gpflow_model.mean_function.parameters[
                0
            ].shape.as_list(),
            "vgp_likelihood": self.gpflow_model.likelihood.__class__.__name__,
            "gp_varsigma": self.gp_varsigma,
            "optimiser": self._serialise_optimiser(),
            "vgp_iters": self.train_iters,
            "vgp_natgrad_lr": self.natgrad_optimiser.gamma,
        }
        with open(os.path.join(folder, self.GPR_INFO), "w") as handle:
            handle.write(json.dumps(save_info))
