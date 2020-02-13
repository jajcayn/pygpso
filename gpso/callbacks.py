"""
Set of useful callbacks.
"""
import logging

from gpflow.utilities import tabulate_module_summary

from .optimisation import CallbackTypes, GPSOCallback
from .plotting import (
    plot_conditional_surrogate_distributions,
    plot_parameter_marginal_distributions,
    plot_ternary_tree,
)


class PostIterationPlotting(GPSOCallback):
    """
    Callback for plotting conditional surrogate distributions, marginal
    distributions of parameters and ternary tree after each iteration.
    """

    callback_type = CallbackTypes.post_iteration

    def __init__(
        self,
        filename_pattern,
        gp_mean_limits=[-10, 10],
        gp_var_limits=[0, 5],
        marginal_plot_type="kde",
        marginal_percentile=0.9,
    ):
        super().__init__()
        self.filename_pattern = filename_pattern
        self.gp_mean_limits = gp_mean_limits
        self.gp_var_limits = gp_var_limits
        self.marginal_plot_type = marginal_plot_type
        self.marginal_percentile = marginal_percentile
        self.iterations_counter = 0

    def run(self, optimiser):
        super().run(optimiser)
        filename_pat_w_it = (
            self.filename_pattern + f"_iter{self.iterations_counter}"
        )
        plot_conditional_surrogate_distributions(
            optimiser,
            mean_limits=self.gp_mean_limits,
            var_limits=self.gp_var_limits,
            fname=filename_pat_w_it + "_surrogate_dist.png",
        )
        plot_parameter_marginal_distributions(
            optimiser,
            plot_type=self.marginal_plot_type,
            percentile=self.marginal_percentile,
            fname=filename_pat_w_it + "_param_marginals.png",
        )
        plot_ternary_tree(
            optimiser.param_space,
            cmap_limits=self.gp_mean_limits,
            fname=filename_pat_w_it + "_tree.png",
        )
        self.iterations_counter += 1


class PostUpdateLogging(GPSOCallback):
    """
    Callback for logging GPR summary after its update.
    """

    callback_type = CallbackTypes.post_update

    def run(self, optimiser):
        super().run(optimiser)
        logging.info(
            "GPR summary:\n"
            + tabulate_module_summary(optimiser.gp_surr.gpr_model)
        )
