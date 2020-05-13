"""
Set of useful callbacks.
"""
import logging
import os

import tensorflow as tf
from gpflow.utilities import tabulate_module_summary

from .optimisation import CallbackTypes, GPSOCallback
from .plotting import (
    plot_conditional_surrogate_distributions,
    plot_parameter_marginal_distributions,
    plot_ternary_tree,
)
from .utils import PKL_EXT


class PostIterationPlotting(GPSOCallback):
    """
    Callback for plotting conditional surrogate distributions, marginal
    distributions of parameters and ternary tree after each iteration.
    """

    callback_type = CallbackTypes.post_iteration

    def __init__(
        self,
        filename_pattern,
        plot_ext=".png",
        gp_mean_limits=[-10, 10],
        gp_var_limits=[0, 5],
        marginal_plot_type="kde",
        marginal_percentile=0.9,
        from_iteration=1,
    ):
        """
        :param filename_pattern: pattern for the filename, should include full
            path, but no extension
        :type filename_pattern: str
        :param plot_ext: extension for the plots
        :type plot_ext: str
        :param gp_mean_limits: limits for GPR predicted mean
        :type gp_mean_limits: list[float]
        :param gp_var_limits: limits for GPR predicted var
        :type gp_var_limits: list[float]
        :param marginal_plot_type: type of plot for marginal parameters: "kde"
            or "bin"
        :type marginal_plot_type: str
        :param marginal_percentile: percentile of highest scores to consider
            when plotting marginal distribution
        :type marginal_percentile: float
        :param from_iteration: from which iteration we should plot
        :type from_iteration: int
        """
        super().__init__()
        self.filename_pattern = filename_pattern
        self.plot_ext = plot_ext
        self.gp_mean_limits = gp_mean_limits
        self.gp_var_limits = gp_var_limits
        self.marginal_plot_type = marginal_plot_type
        self.marginal_percentile = marginal_percentile
        self.from_iteration = from_iteration

    def run(self, optimiser):
        super().run(optimiser)
        if optimiser.iterations >= self.from_iteration:
            filename_pat_w_it = (
                self.filename_pattern + f"_iter{optimiser.iterations}"
            )
            plot_conditional_surrogate_distributions(
                optimiser,
                mean_limits=self.gp_mean_limits,
                var_limits=self.gp_var_limits,
                fname=filename_pat_w_it + f"_surrogate_dist{self.plot_ext}",
            )
            plot_parameter_marginal_distributions(
                optimiser,
                plot_type=self.marginal_plot_type,
                percentile=self.marginal_percentile,
                fname=filename_pat_w_it + f"_param_marginals{self.plot_ext}",
            )
            plot_ternary_tree(
                optimiser.param_space,
                cmap_limits=self.gp_mean_limits,
                fname=filename_pat_w_it + f"_tree{self.plot_ext}",
            )


class PostUpdateLogging(GPSOCallback):
    """
    Callback for logging GPR summary after its update.
    """

    callback_type = CallbackTypes.post_update

    def run(self, optimiser):
        super().run(optimiser)
        logging.info(
            "GPR summary:\n"
            + tabulate_module_summary(optimiser.gp_surr.gpflow_model)
        )


class PreFinaliseSave(GPSOCallback):
    """
    Callback for saving before finalisation.
    """

    callback_type = CallbackTypes.pre_finalise

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self, optimiser):
        super().run(optimiser)
        optimiser.param_space.save(
            os.path.join(self.path, f"parameter_space{PKL_EXT}")
        )
        optimiser.gp_surr.save(self.path)


class GPFlowCheckpoints(GPSOCallback):
    """
    Callback for creating checkpoints while training GP.
    """

    callback_type = CallbackTypes.post_update

    def __init__(self, path, max_to_keep=10):
        super().__init__()
        self.path = path
        self.first_update = True
        self.max_to_keep = max_to_keep
        self.n_evals = tf.Variable(0)

    def run(self, optimiser):
        super().run(optimiser)
        self.n_evals.assign(optimiser.n_eval_counter)
        if self.first_update:
            ckpt = tf.train.Checkpoint(
                model=optimiser.gp_surr.gpflow_model, evluations=self.n_evals,
            )
            self.manager = tf.train.CheckpointManager(
                ckpt, self.path, max_to_keep=self.max_to_keep
            )
            saved_to = self.manager.save()
            self.first_update = False
        else:
            saved_to = self.manager.save()
        logging.debug(f"Checkppoint saved to {saved_to}")
