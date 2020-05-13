"""
Optimisation using Bayesian GP regression leveraging GPFlow.
"""
import json
import logging
import os
from enum import Enum, auto, unique
from functools import partial

import numpy as np
from anytree import PreOrderIter
from pathos.pools import ProcessPool

from .gp_surrogate import GPPoint, GPRSurrogate, GPSurrogate, PointLabels
from .param_space import NORM_PARAMS_BOUNDS, ParameterSpace
from .utils import JSON_EXT, PKL_EXT, load_json, make_dirs


@unique
class CallbackTypes(Enum):
    """
    Define callback types.
    """

    post_initialise = auto()
    pre_iteration = auto()
    post_iteration = auto()
    post_update = auto()
    pre_finalise = auto()


class GPSOCallback:
    # do not forget to define type of the callback
    callback_type = None

    def __init__(self):
        # all arguments for callback needs to be defined here

        # when subclassing, it is recommended to call super().__init__() for
        # sanity check
        assert (
            self.callback_type in CallbackTypes
        ), "Callback type must be one of `CallbackTypes`"

    def run(self, optimiser):
        # run only takes one argument - the GPSOptimiser itself

        # when subclassing, it is recommended to call super().run() for sanity
        # check
        assert isinstance(optimiser, GPSOptimiser)
        logging.info(f"Running {self.__class__.__name__} callback...")


class GPSOptimiser:
    """
    Main class for GP surrogate optimisation.
    """

    SAVE_ATTRS = [
        "iterations",
        "budget",
        "eval_repeats",
        "last_explored_levels",
        "last_update_idx",
        "method",
        "max_depth",
        "stop_cond",
        "update_cycle",
        "n_eval_counter",
        "n_workers",
        "expl_seed",
    ]
    PARAM_SPACE_FILE = f"parameter_space{PKL_EXT}"
    OPT_ATTRS_FILE = f"opt_attributes{JSON_EXT}"

    @classmethod
    def resume_from_saved(
        cls,
        folder,
        additional_budget,
        objective_function,
        gp_surrogate=GPRSurrogate,
        eval_repeats_function=np.mean,
        callbacks=None,
        saver=None,
    ):
        """
        Resume optimisation from saved optimiser. Callbacks and saver are not
        saved so need to be passed again.

        :param folder: path from which to load the optimiser's state
        :type folder: str
        :param additional_budget: budget for cycles to resume for
        :type additional_budget: int
        :param objective_function: objective function to evaluate the score,
            must be callable and take unnormalised parameters as an argument
            and output scalar score
        :type objective_function: callable
        :param gp_surrogate: GP surrogate class used in original implementation,
            only class, not initialised!
        :type gp_surrogate: `gpso.gp_surrogate.GPSurrogate` not initialised
        :param eval_repeats_function: function for aggregating multiple
            evaluations (see `eval_repeats`), has to take axis as an argument,
            good choices are mean, median or max; their nan- version can be
            used as well, when the stochastic evaluation might expectedly fail
        :type eval_repeats_function: callable
        :param callbacks: list of (initialised) user-defined callbacks
        :type callbacks: list[GPSOCallback|None]
        :param saver: saver object, which is able to save intermediate results
            (e.g. timeseries from objective function), if passed, it has to
            implement `save_runs` method - ideal candidate is `TableSaver`
            which is part of `gpso` and saves results to HDF file, if saver is
            passed, the objective function has to return (result, score)
        :type saver: object|None
        :return: point with highest score of objective function and loaded
            optimiser
        :rtype: (`gpso.gp_surrogate.GPPoint`, `gpso.optimisation.GPSOptimiser`)
        """
        # load parameter space
        param_space = ParameterSpace.from_file(
            os.path.join(folder, cls.PARAM_SPACE_FILE)
        )
        # load surrogate
        gp_surr = gp_surrogate.from_saved(folder)
        # load attributes
        opt_attrs = load_json(os.path.join(folder, cls.OPT_ATTRS_FILE))

        # init optimiser
        optimiser = cls(
            parameter_space=param_space, callbacks=callbacks, saver=saver
        )
        # assign attribute values from saved file
        for attr, value in opt_attrs.items():
            setattr(optimiser, attr, value)
        # assign GPSurrogate from saved file
        optimiser.gp_surr = gp_surr
        assert callable(objective_function)
        optimiser.obj_func = objective_function
        assert callable(eval_repeats_function)
        optimiser.eval_repeats_function = partial(eval_repeats_function, axis=0)

        return (
            optimiser.resume_run(additional_budget=additional_budget),
            optimiser,
        )

    def __init__(
        self,
        parameter_space,
        gp_surrogate=None,
        exploration_method="tree",
        exploration_depth=5,
        budget=100,
        stopping_condition="evaluations",
        update_cycle=1,
        n_workers=1,
        callbacks=None,
        saver=None,
    ):
        """
        :param parameter_space: parameter space to explore
        :type parameter_space: `gpso.param_space.ParameterSpace`
        :param gp_surrogate: Gaussian Processes surrogate object, if None, will
            use default (sensible) settings
        :type gp_surrogate: `gpso.gp_surrogate.GPSurrogate`|None
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
        :param n_workers: number of workers to use where applicable
        :type n_workers: int
        :param callbacks: list of (initialised) user-defined callbacks
        :type callbacks: list[GPSOCallback|None]
        :param saver: saver object, which is able to save intermediate results
            (e.g. timeseries from objective function), if passed, it has to
            implement `save_runs` method - ideal candidate is `TableSaver`
            which is part of `gpso` and saves results to HDF file, if saver is
            passed, the objective function has to return (result, score)
        :type saver: object|None
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
        self.iterations = 0
        self.n_workers = n_workers

        if callbacks is None:
            callbacks = []
        assert all(isinstance(callback, GPSOCallback) for callback in callbacks)
        self.callbacks = callbacks

        # surrogate object
        self.gp_surr = gp_surrogate or GPRSurrogate.default()
        assert isinstance(self.gp_surr, GPSurrogate)

        # saver
        self.saver = saver
        if saver is not None:
            # saver has to implement `save_runs` function
            save_func = getattr(self.saver, "save_runs", None)
            assert callable(save_func)

    def _run_callbacks(self, callback_type):
        """
        Run callbacks of given type.
        """
        assert callback_type in CallbackTypes
        for callback in self.callbacks:
            if callback.callback_type == callback_type:
                callback.run(self)

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
                f"using {self.n_workers} worker(s)..."
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
            elif init_samples.shape[0] > (2 * self.param_space.ndim):
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

            self._run_callbacks(callback_type=CallbackTypes.post_update)
        return self.gp_surr.num_evaluated

    def _tree_explore(self, levels_to_explore, **kwargs):
        """
        Exploration step: split leaf into children and sample GP in the highest
        scored leaf in each level.

        :param levels_to_explore: which levels to explore
        :type levels_to_explore: list[bool]
        :kwargs:
            - "seed": seed for uniform sampler when sampling strategy is used
        """
        logging.info(
            "Exploration step: sampling children in the ternary tree..."
        )

        assert len(levels_to_explore) == self.param_space.max_depth + 1

        def child_sample(child):
            if self.method == "tree":
                return child.grow(depth=self.max_depth)
            elif self.method == "sample":
                return child.sample_uniformly(
                    n_points=self.max_depth, seed=kwargs.pop("seed", None)
                )

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

        if (
            self.n_workers > 1
            and (self.eval_repeats * orig_coords.shape[0]) > 1
        ):
            pool = ProcessPool(self.n_workers)
            map_func = pool.map
        else:
            pool = None
            map_func = map

        # run for number of desired repeats
        repeated_coords = np.vstack(self.eval_repeats * [orig_coords])
        scores = list(map_func(self.obj_func, repeated_coords))

        if pool is not None:
            pool.close()
            pool.join()
            pool.clear()

        # update evaluation counter (not by repeats!)
        self.n_eval_counter += orig_coords.shape[0]

        # save if needed
        if self.saver is not None:
            results = [score[0] for score in scores]
            scores = [score[1] for score in scores]
            # save each coordinate as single run
            for coord_idx, coords in enumerate(orig_coords):
                run_results = results[coord_idx :: orig_coords.shape[0]]
                run_scores = scores[coord_idx :: orig_coords.shape[0]]
                assert len(run_results) == len(run_scores) == self.eval_repeats
                self.saver.save_runs(
                    run_results,
                    run_scores,
                    {
                        param_name: coord
                        for param_name, coord in zip(
                            self.param_space.parameter_names, coords
                        )
                    },
                )
        # return aggregation over repeats
        return self.eval_repeats_function(
            np.array(scores).astype(np.float).reshape((self.eval_repeats, -1))
        )

    def _stopping_condition(self):
        """
        Evaluate stopping condition.

        :return: whether we should or not
        :rtype: bool
        """
        if self.stop_cond == "evaluations":
            return self.n_eval_counter < self.budget
        elif self.stop_cond == "iterations":
            return self.iterations < self.budget
        elif self.stop_cond == "depth":
            return self.param_space.max_depth <= self.budget

    def run(
        self,
        objective_function,
        init_samples=None,
        eval_repeats=1,
        eval_repeats_function=np.mean,
        **kwargs,
    ):
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
        :param eval_repeats: number of repetitions for objective evaluation
            function when it is stochastic and some statistics on the score is
            necessary; repeats are not counted towards the budget of objective
            evaluations; multiprocessing is used when self.n_workers > 1
        :type eval_repeats: int
        :param eval_repeats_function: function for aggregating multiple
            evaluations (see `eval_repeats`), has to take axis as an argument,
            good choices are mean, median or max; their nan- version can be
            used as well, when the stochastic evaluation might expectedly fail
        :type eval_repeats_function: callable
        :kwargs:
            - "seed": seed for uniform sampler when sampling strategy is used
        :return: point with highest score of objective function
        :rtype: `gpso.gp_surrogate.GPPoint`
        """
        assert callable(objective_function)
        self.obj_func = objective_function
        self.eval_repeats = eval_repeats
        assert callable(eval_repeats_function)
        self.eval_repeats_function = partial(eval_repeats_function, axis=0)
        self.expl_seed = kwargs.pop("seed", None)
        logging.info(
            f"Starting {self.param_space.ndim}-dimensional optimisation with "
            f"budget of {self.budget} objective function evaluations..."
        )
        self._initialise(init_samples)
        self._run_callbacks(callback_type=CallbackTypes.post_initialise)

        update_idx = self._gp_update(0)
        # keep notes on whether we should explore specific level
        explore_levels = [True]

        # iterate while condition is True - so first set it to True and
        # evaluate each iteration
        cond = True
        while cond:
            self._run_callbacks(callback_type=CallbackTypes.pre_iteration)

            # explore
            self._tree_explore(
                levels_to_explore=explore_levels, seed=self.expl_seed
            )
            # select
            explore_levels = self._tree_select()
            # update
            update_idx = self._gp_update(update_idx)

            self.iterations += 1
            logging.info(
                f"After {self.iterations}th iteration: \n\t number of obj. "
                f"func. evaluations: {self.n_eval_counter} \n\t highest score:"
                f" {self.gp_surr.highest_score.score_mu} \n\t highest UCB: "
                f"{self.gp_surr.highest_ucb.score_ucb}"
            )
            logging.debug(
                f"\n\t Total number of points: {len(self.gp_surr.points)} \n\t"
                f" evaluated points: {self.gp_surr.num_evaluated} \n\t"
                f" GP-based estimates: {self.gp_surr.num_gp_based} \n\t"
                f" depth of the tree: {self.param_space.max_depth}"
            )
            self._run_callbacks(callback_type=CallbackTypes.post_iteration)

            # reevaluate condition based on the condition itself and its budget
            cond = self._stopping_condition()

        logging.info(
            "Done. Highest evaluated score: "
            f"{self.gp_surr.highest_score.score_mu}"
        )
        self._run_callbacks(callback_type=CallbackTypes.pre_finalise)
        self.last_explored_levels = explore_levels
        self.last_update_idx = update_idx

        return self.gp_surr.highest_score

    def resume_run(self, additional_budget):
        """
        Resume optimisation for additional budget cycles.

        :param additional_budget: budget for cycles to resume for
        :type additional_budget: int
        :return: point with highest score of objective function
        :rtype: `gpso.gp_surrogate.GPPoint`
        """
        assert callable(self.obj_func)
        assert callable(self.eval_repeats_function)
        assert self.iterations > 0
        self.budget += additional_budget
        logging.info(
            "Resuming optimisation for with additional budget of "
            f"{additional_budget}"
        )

        # iterate while condition is True - so first set it to True and
        # evaluate each iteration
        explore_levels = self.last_explored_levels
        update_idx = self.last_update_idx
        cond = True
        while cond:
            self._run_callbacks(callback_type=CallbackTypes.pre_iteration)

            # explore
            self._tree_explore(
                levels_to_explore=explore_levels, seed=self.expl_seed
            )
            # select
            explore_levels = self._tree_select()
            # update
            update_idx = self._gp_update(update_idx)

            self.iterations += 1
            logging.info(
                f"After {self.iterations}th iteration: \n\t number of obj. "
                f"func. evaluations: {self.n_eval_counter} \n\t highest score:"
                f" {self.gp_surr.highest_score.score_mu} \n\t highest UCB: "
                f"{self.gp_surr.highest_ucb.score_ucb}"
            )
            logging.debug(
                f"\n\t Total number of points: {len(self.gp_surr.points)} \n\t"
                f" evaluated points: {self.gp_surr.num_evaluated} \n\t"
                f" GP-based estimates: {self.gp_surr.num_gp_based} \n\t"
                f" depth of the tree: {self.param_space.max_depth}"
            )
            self._run_callbacks(callback_type=CallbackTypes.post_iteration)

            # reevaluate condition based on the condition itself and its budget
            cond = self._stopping_condition()

        logging.info(
            "Done. Highest evaluated score: "
            f"{self.gp_surr.highest_score.score_mu}"
        )
        self._run_callbacks(callback_type=CallbackTypes.pre_finalise)
        self.last_explored_levels = explore_levels
        self.last_update_idx = update_idx

        return self.gp_surr.highest_score

    def save_state(self, folder):
        """
        Save current state of the optimisation so that it can be resumed.

        :param folder: path to which save the optimiser's state
        :type folder: str
        """
        make_dirs(folder)
        logging.warning("When saving, all callbacks and saver will be lost!")
        # save parameter space
        self.param_space.save(os.path.join(folder, self.PARAM_SPACE_FILE))
        # save surrogate
        self.gp_surr.save(folder)
        # save optimiser attributes
        opt_attrs = {}
        for attr in self.SAVE_ATTRS:
            opt_attrs[attr] = getattr(self, attr)
        with open(
            os.path.join(folder, self.OPT_ATTRS_FILE), "w"
        ) as file_handler:
            file_handler.write(json.dumps(opt_attrs))
        logging.info(f"Saved optimiser to {folder}")
