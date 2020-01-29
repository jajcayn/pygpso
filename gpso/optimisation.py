"""
Optimisation using Bayesian GP regression leveraging GPFlow.
"""
import logging

import numpy as np

import gpflow
from anytree import PreOrderIter
from pathos.multiprocessing import Pool
from scipy.special import erfcinv

from .gp_surrogate import GPPoint, GPSurrogate, PointLabels
from .param_space import NORM_PARAMS_BOUNDS, ParameterSpace


class GPSOptimiser:
    """
    Main class for GP surrogate optimisation.
    """

    def __init__(
        self,
        parameter_space,
        exploration_method="tree",
        exploration_depth=5,
        budget=100,
        stopping_condition="evaluations",
        update_cycle=1,
        varsigma=erfcinv(0.01),
        gp_lik_sigma=1.0e-3,
        n_workers=1,
        **kwargs,
    ):
        """
        :param parameter_space: parameter space to explore
        :type parameter_space: `gpso.param_space.ParameterSpace`
        :param exploration_method: method used for exploration of children
            intervals: `tree` or `sample`
        :type exploration_method: str
        :param exploration_depth: depth of the exploration step, for partition
            tree is is tree depth up to which the GPSO will be evaluated, for
            random sampling it will be multiplied by ndim^2 and is number of
            samples within the domain
        :type exploration_depth: int
        :param budget: budget for the iterations, either number of evaluations,
            or iterations, or depth of the tree
        :type budget: int
        :param stopping_condition: to what condition the budget refers to, use
            of the "evaluations", "iterations", "depth"
        :type stopping_condition: str
        :param update_cycle: how often update GPR hyperparameters
        :type update_cycle: int
        :param varsigma: expected probability that UCB < f; it controls how
            "optimistic" we are during the exploration step; at a point x
            evaluated using GP, the UCB will be: mu(x) + varsigma*sigma(x);
            varsigma = 1/erfc(p/100) which corresponds to the upper bound of a
                `p` confidence interval for Gaussian likelihood kernel
        :type varsigma: float
        :param gp_lik_sigma: initial std of Gaussian likelihood function (in
            normalised units)
        :type gp_lik_sigma: float
        :param n_workers: number of workers to use where applicable
        :type n_workers: int
        :**kwargs: keyword arguments
            - `gp_kernel`: covariance kernel for GPR, Matern52 by default
            - `gp_meanf`: mean function for GPR, constant by default
            - `gp_mu`: initial value for constant mean function in GPR
            - `gp_length`: initial value for lengthscales in the kernel
                function in GPR
            - `gp_var`: initial value for variance in the kernel function in
                GPR
        """
        assert isinstance(parameter_space, ParameterSpace)
        self.param_space = parameter_space

        self.method = exploration_method
        if self.method == "tree":
            self.max_depth = exploration_depth
        elif self.method == "sample":
            self.max_depth = exploration_depth * self.param_space.ndim ** 2
        else:
            raise ValueError(f"Unknown exploration method: {self.method}")

        self.budget = budget
        assert stopping_condition in ["evaluations", "iterations", "depth"]
        self.stop_cond = stopping_condition
        self.update_cycle = update_cycle
        self.n_eval_counter = 0
        self.n_workers = n_workers

        # GPR model hyperparameters
        self.gp_surr = GPSurrogate(
            gp_kernel=kwargs.pop(
                "gp_kernel",
                gpflow.kernels.Matern52(
                    lengthscale=kwargs.pop(
                        "gp_length", np.sum(NORM_PARAMS_BOUNDS) * 0.25
                    ),
                    variance=kwargs.pop("gp_var", 1.0),
                ),
            ),
            gp_meanf=kwargs.pop(
                "gp_meanf",
                gpflow.mean_functions.Constant(c=kwargs.pop("gp_mu", 0.0)),
            ),
            gauss_likelihood_sigma=gp_lik_sigma,
            varsigma=varsigma,
        )

    def _initialise(self, init_samples):
        """
        :param init_samples: initial samples; either array of !original!
            coordinates for which the objective function should be evaluated or
            None - in that case the two vertices per dimension, equally spaced
            from the centre (diamond-shape)
        :type init_samples: np.ndarray|None
        """
        # default initialisation with two vertices per dimension
        if init_samples is None:
            logging.info(
                "Sampling 2 vertices per dimension within L1 ball of "
                "0.25 of the domain size radius in normalised coordinates "
                f"using {self.n_workers} workers..."
            )
            # 2 vertices per dim at L1 ball with 1/4 of the domain size radius
            # in normed coords
            normed_coords = np.vstack(
                [
                    np.mean(NORM_PARAMS_BOUNDS)
                    - np.sum(NORM_PARAMS_BOUNDS)
                    * 0.25
                    * np.eye(self.param_space.ndim),
                    np.mean(NORM_PARAMS_BOUNDS)
                    + np.sum(NORM_PARAMS_BOUNDS)
                    * 0.25
                    * np.eye(self.param_space.ndim),
                ]
            )
            orig_coords = self.param_space.denormalise_coords(normed_coords)

        # user defined initial points to evaluate
        elif isinstance(init_samples, np.ndarray):
            assert init_samples.ndim == 2
            assert init_samples.shape[1] == self.param_space.ndim
            if init_samples.shape[0] <= 2:
                logging.warning(
                    f"Only {init_samples.shape[0]} points selected for "
                    "sampling, you might want to add more..."
                )
            elif init_samples.shape[0] > 10:
                logging.warning(
                    "Too many initial points obtained, you will run out of "
                    "budget of objective function evaluations!"
                )
            logging.info(
                f"Got {init_samples.shape[0]} points for initial sampling. "
                "Note that these are interpreted in the original parameter "
                "space coordinates!"
            )
            orig_coords = init_samples.copy()

        # evaluate at the center of the domain
        coords_center = np.array(
            [[np.mean(NORM_PARAMS_BOUNDS)] * self.param_space.ndim]
        )
        orig_coords_center = self.param_space.denormalise_coords(coords_center)
        all_coords = np.vstack([orig_coords, orig_coords_center])
        all_scores = self.evaluate_objective_function(all_coords)
        # assign score to root leaf
        self.param_space.score = float(all_scores[-1])
        self.param_space.label = PointLabels.evaluated

        # append to GP surrogate as training points with normalised coordinates
        self.gp_surr.append(
            self.param_space.normalise_coords(all_coords), all_scores
        )
        debug_msg = "".join(
            [
                f"\n\t{coord}: {score}"
                for coord, score in zip(all_coords, all_scores)
            ]
        )
        logging.debug(
            f"Initialised with {all_coords.shape[0]} points:{debug_msg}"
        )

    def _gp_update(self, update_idx):
        """
        Update (retrain) GP if needed, based on the update cycle.

        :param update_idx: number of evaluated points in the GP surrogate during
            last update
        :type update_idx: int
        :return: current number of evaluated points
        :rtype: int
        """
        if (self.gp_surr.num_evaluated - update_idx) >= self.update_cycle:
            logging.info(
                "Update step: retraining GP model and updating scores..."
            )
            # update GP - retrain and update scores in points
            self.gp_surr.gp_update()
            # update tree - update UCB score for GP-based leaves
            for leaf in PreOrderIter(self.param_space):
                leaf_point = self.gp_surr.points.find_by_coords(
                    np.array(leaf.get_center_as_list(normed=True))
                )
                assert leaf_point is not None
                if leaf_point.label == PointLabels.gp_based:
                    leaf.score = leaf_point.score_ucb
        return self.gp_surr.num_evaluated

    def _tree_explore(self, levels_to_explore):
        """
        Exploration step: split leaf into children and sample GP in the highest
        scored leaf in each level.

        :param levels_to_explore: which levels to explore
        :type levels_to_explore: list[bool]
        """
        logging.info(
            "Exploration step: sampling children in the ternary tree..."
        )

        assert len(levels_to_explore) == self.param_space.max_depth + 1

        def child_sample(child):
            if self.method == "tree":
                return child.grow(depth=self.max_depth)
            elif self.method == "sample":
                return child.sample_uniformly(n_points=self.max_depth)

        for level in range(self.param_space.max_depth + 1):
            if levels_to_explore[level]:
                logging.debug(f"Exploring {level} level...")
                # explore leaf with best UCB
                parent = self.param_space.get_best_score_leaf(depth=level)
                # split using ternary partition function
                children = parent.ternary_split()
                for child in children:
                    child_point = self.gp_surr.points.find_by_coords(
                        np.array(child.get_center_as_list(normed=True))
                    )
                    # if point does not exists
                    if child_point is None:
                        best_score = self.gp_surr.gp_eval_best_ucb(
                            child_sample(child)
                        )
                        # assign leaf score to best UCB from sampled
                        child.score = best_score[2]
                        child.label = PointLabels.gp_based
                        # append points to GP
                        self.gp_surr.points.append(
                            GPPoint(
                                normed_coord=np.array(
                                    child.get_center_as_list(normed=True)
                                ),
                                score_mu=best_score[0],
                                score_sigma=best_score[1],
                                score_ucb=best_score[2],
                                label=PointLabels.gp_based,
                            )
                        )
                    else:
                        # otherwise just copy parent's score
                        child.score = parent.score
                        child.label = parent.label
                    logging.debug(f"{child.name} best score: {child.score}")
                # set parent as already sampled
                parent.sampled = True

    def _tree_select(self):
        """
        Select step: find leaf with highest score per level and evaluate
        objective function there.

        :return: levels to explore in the next step
        :rtype: list[bool]
        """
        logging.info("Selecting step: evaluating best leaves...")
        max_score = -np.inf
        # default to not exploring any level
        levels_to_explore = [False] * (self.param_space.max_depth + 1)

        for level in range(self.param_space.max_depth + 1):
            logging.debug(f"Selecting within {level} level...")
            # find max leaf which was not sampled yet
            max_leaf = self.param_space.get_best_score_leaf(
                depth=level, only_not_sampled=True
            )
            if max_leaf and max_leaf.score > max_score:
                # set this level for exploration in the next step
                levels_to_explore[level] = True
                # set new max score
                max_score = float(max_leaf.score)
                # get point from GP
                leaf_point = self.gp_surr.points.find_by_coords(
                    np.array(max_leaf.get_center_as_list(normed=True))
                )
                # if the max leaf is GP-based, evaluate it
                if leaf_point.label == PointLabels.gp_based:
                    new_score = float(
                        self.evaluate_objective_function(
                            self.param_space.denormalise_coords(
                                leaf_point.normed_coord[np.newaxis, :]
                            )
                        )
                    )
                    # update point
                    new_point = GPPoint(
                        normed_coord=leaf_point.normed_coord,
                        score_mu=new_score,
                        score_sigma=0.0,
                        score_ucb=0.0,
                        label=PointLabels.evaluated,
                    )
                    self.gp_surr.points.append(new_point)
                    # update leaf
                    max_leaf.score = new_score
                    max_leaf.label = PointLabels.evaluated
                    logging.debug(
                        f"Leaf {max_leaf.name} updated to new evaluated score:"
                        f" {max_leaf.score}"
                    )

        logging.debug(
            f"Level to explore in the next iteration: {levels_to_explore}"
        )
        return levels_to_explore

    def evaluate_objective_function(self, orig_coords):
        """
        Evaluate objective function at given coordinates in the original
        parameter space.

        :param orig_coords: coordinates in the original parameter space to
            evaluate as [n points x ndim]
        :type orig_coords: np.ndarray
        :return: associated scores from the objective function
        :rtype: np.ndarray
        """
        assert orig_coords.ndim == 2
        assert orig_coords.shape[1] == self.param_space.ndim

        if self.n_workers > 1 and orig_coords.shape[0] > 1:
            pool = Pool(self.n_workers)
            map_func = pool.map
        else:
            pool = None
            map_func = map

        scores = list(map_func(self._obj_func_call, orig_coords))

        if pool is not None:
            pool.close()
            pool.join()

        return np.array(scores)

    def _obj_func_call(self, params):
        """
        Objective function call.

        :param params: parameters for the objective function - coordinates of
            the domain
        :type params: list|dict
        :return: score from the objective function
        :rtype: float
        """
        score = self.obj_func(params)
        self.n_eval_counter += 1
        return float(score)

    def _stopping_condition(self, iteration):
        """
        Evaluate stopping condition.

        :param iteration: current number of iterations
        :type iteration: int
        :return: whether we should or not
        :rtype: bool
        """
        if self.stop_cond == "evaluations":
            return self.n_eval_counter < self.budget
        elif self.stop_cond == "iterations":
            return iteration < self.budget
        elif self.stop_cond == "depth":
            return self.param_space.max_depth <= self.budget

    def run(self, objective_function, init_samples=None):
        """
        Run the optimisation.

        :param objective_function: objective function to evaluate the score,
            must be callable and take unnormalised parameters as an argument
            and output scalar score
        :type objective_function: callable
        :param init_samples: initial samples; either array of coordinates for
            which the objective function should be evaluated or tuple
            (coords, scores) for user-defined scores (e.g. already evaluated
            points), or None - in that case the two vertices per dimension,
            equally spaced from the centre (diamond-shape)
        :type init_samples: np.ndarray|tuple(np.ndarray,np.ndarray)|None
        :return: point with highest score of objective function
        :rtype: `gpso.gp_surrogate.GPPoint`
        """
        assert callable(objective_function)
        self.obj_func = objective_function
        logging.info(
            f"Starting {self.param_space.ndim}-dimensional optimisation with "
            f"budget of {self.budget} objective function evaluations..."
        )
        self._initialise(init_samples)
        update_idx = self._gp_update(0)
        # keep notes on whether we should explore specific level
        explore_levels = [True]

        # iterate while condition is True - so first set it to True and
        # evaluate each iteration
        cond = True
        iterations = 0
        while cond:
            # explore
            self._tree_explore(levels_to_explore=explore_levels)
            # select
            explore_levels = self._tree_select()
            # update
            update_idx = self._gp_update(update_idx)

            iterations += 1
            logging.info(
                f"After {iterations}th iteration: \n\t number of obj. func. "
                f"evaluations: {self.n_eval_counter} \n\t highest score: "
                f"{self.gp_surr.highest_score.score_mu} \n\t highest UCB: "
                f"{self.gp_surr.highest_ucb.score_ucb}"
            )
            logging.debug(
                f"\n\t Total number of points: {len(self.gp_surr.points)} \n\t"
                f" evaluated points: {self.gp_surr.num_evaluated} \n\t"
                f" GP-based estimates: {self.gp_surr.num_gp_based} \n\t"
                f" depth of the tree: {self.param_space.max_depth}"
            )
            # reevaluate condition based on the condition itself and its budget
            cond = self._stopping_condition(iterations)

        logging.info(
            "Done. Highest evaluated score: "
            f"{self.gp_surr.highest_score.score_mu}"
        )

        return self.gp_surr.highest_score
