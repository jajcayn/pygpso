"""
Plotting functions related to GPSO.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from anytree import PreOrderIter
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler

from .gp_surrogate import PointLabels
from .optimisation import GPSOptimiser
from .param_space import ParameterSpace

# for reasonable quality plots...
DPI = 300

# default number of bins for histogram like plots, and squared as default number
# for linspaces when creating data for surrogate plots
N_BINS = 10

# CONSTANT FOR TERNARY TREE PLOTTING
RADIUS_EVAL = 8e-3
RADIUS_GP = 5e-3

# number of decimal numbers for x- and y-ticklabels
TICK_DECIMALS = 3


def plot_ternary_tree(
    param_space,
    cmap="Spectral",
    cmap_limits=[None, None],
    center_root_node=False,
    fname=None,
    **kwargs,
):
    """
    Plot ternary partition tree associated with optimisation problem. Each node
    corresponds to a different subinterval of the search space, colours
    correspond to associated scores. Bigger nodes with black edges indicate
    that the objective function was evaluated at their center, while smaller
    nodes without border were assessed using GP surrogate only. Tree is drawn
    using igraph's implementation of Reingold-Tilford algorithm for tree layout.

    :param param_space: parameter space of already optimised problem
    :type param_space: `gpso.param_space.ParameterSpace`
    :param cmap: colormap for scores
    :type cmap: str
    :param cmap_limits: limits for the colormapping of the scores, if None will
        be inferred from the data
    :param center_root_node: whether to center root node within the figure -
        graph might become less readable
    :type center_root_node: bool
    :type cmap_limits: List[float|None]
    :param fname: filename for the plot, if None, will show
    :type fname: str|None
    :*kwargs: keyword arguments for `plt.figure()`
    """
    from igraph import Graph

    assert isinstance(param_space, ParameterSpace)
    leaves_preorder = list(PreOrderIter(param_space))
    # assign indexes in pre-order order to nodes
    for i, leaf in enumerate(leaves_preorder):
        leaf.index = i
    # add all edges as (parent, child)
    edges = [
        (node.parent.index, node.index)
        for node in leaves_preorder
        if node.parent is not None
    ]
    graph = Graph(n=len(leaves_preorder), edges=edges, directed=False)
    # create Reingold-Tilford layout
    rt_layout = graph.layout_reingold_tilford(
        root=[0] if center_root_node else None
    )
    # scale layout so it fits to matplotlib's coordinates
    scaled_rt_layout = np.array(rt_layout.coords).copy()
    # x between 0. - 0.92 (space for colorbar)
    scaler = MinMaxScaler(feature_range=(0, 0.92))
    scaled_rt_layout[:, 0] = scaler.fit_transform(
        scaled_rt_layout[:, 0].reshape(-1, 1)
    ).squeeze()
    # y between 0. - 1.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_rt_layout[:, 1] = scaler.fit_transform(
        scaled_rt_layout[:, 1].reshape(-1, 1)
    ).squeeze()

    fig = plt.figure(**kwargs)
    colors = plt.get_cmap(cmap)
    min_score = cmap_limits[0] or np.min(
        [leaf.score for leaf in leaves_preorder]
    )
    max_score = cmap_limits[1] or np.max(
        [leaf.score for leaf in leaves_preorder]
    )

    for leaf, leaf_coords in zip(leaves_preorder, scaled_rt_layout):
        if leaf.label == PointLabels.evaluated:
            edgecolor = "k"
            radius = RADIUS_EVAL
        elif leaf.label == PointLabels.gp_based:
            edgecolor = "none"
            radius = RADIUS_GP
        circle_color = (leaf.score - min_score) / (max_score - min_score)
        circle = plt.Circle(
            leaf_coords,
            radius=radius,
            edgecolor=edgecolor,
            facecolor=colors(circle_color),
        )
        plt.gca().add_artist(circle)

    for edge in edges:
        parent, child = edge
        plt.plot(
            [scaled_rt_layout[parent, 0], scaled_rt_layout[child, 0]],
            [scaled_rt_layout[parent, 1], scaled_rt_layout[child, 1]],
            color="k",
            linewidth=0.7,
        )

    plt.axis("off")
    plt.gca().invert_yaxis()

    # colorbars
    cbar_ax = fig.add_axes([0.97, 0.3, 0.03, 0.4])
    norm_mean = mpl.colors.Normalize(vmin=min_score, vmax=max_score)
    cb_mean = mpl.colorbar.ColorbarBase(
        cbar_ax, cmap=plt.get_cmap(cmap), norm=norm_mean, orientation="vertical"
    )
    cb_mean.set_label("Score / evaluated or UCB")

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", dpi=DPI)
    else:
        plt.show()
    plt.close()


def plot_parameter_marginal_distributions(
    gpso_optimiser, percentile=0.9, plot_type="kde", fname=None, **kwargs
):
    """
    Plot marginal distributions of parameter values corresponding to selected
    percentile of all evaluated samples.

    :param gpso_optimiser: already optimised problem
    :type gpso_optimiser: `gpso.optimisation.GPSOptimiser`
    :param percentile: which percentile to plot
    :type percentile: float
    :param plot_type: "kde" for Gaussian kernel density estimate or "hist" for
        basic histogram
    :type plot_type: str
    :param fname: filename for the plot, if None, will show
    :type fname: str|None
    :kwargs:
        "bins" - number of bins if plot_type == "hist"
        "bw_method" - estimator bandwidth if plot_type == "kde", see
            `gaussian_kde` for more info
    """
    assert isinstance(gpso_optimiser, GPSOptimiser)
    # get GP points that were evaluated sorted by the score
    points = sorted(
        [
            point
            for point in gpso_optimiser.gp_surr.points
            if point.label == PointLabels.evaluated
        ],
        key=lambda point: point.score_mu,
        reverse=True,
    )
    # take first based on percentile
    points = points[: int(len(points) * (1 - percentile))]
    norm_coords = np.vstack([point.normed_coord for point in points])
    orig_coords = gpso_optimiser.param_space.denormalise_coords(norm_coords)
    if plot_type == "hist":

        def plot_func(x):
            plt.hist(
                x,
                bins=kwargs.pop("bins", N_BINS),
                weights=np.ones_like(x) / float(len(x)),
            )

    elif plot_type == "kde":

        def plot_func(x):
            kde_est = gaussian_kde(x, bw_method=kwargs.pop("bw_method", None))
            x_points = np.linspace(x.min(), x.max(), N_BINS ** 2)
            plt.plot(x_points, kde_est.pdf(x_points))

    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    plt.figure()
    plt.subplots_adjust(hspace=0.3)
    for param_idx in range(gpso_optimiser.param_space.ndim):
        plt.subplot(gpso_optimiser.param_space.ndim, 1, param_idx + 1)
        plot_func(orig_coords[:, param_idx])
        plt.xlim(
            [
                gpso_optimiser.param_space.scaler.data_min_[param_idx],
                gpso_optimiser.param_space.scaler.data_max_[param_idx],
            ]
        )
        plt.title(gpso_optimiser.param_space.parameter_names[param_idx])

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", dpi=DPI)
    else:
        plt.show()
    plt.close()


def _set_share_axes(axs, target=None, sharex=False, sharey=False):
    """
    Manual override for sharing x- or y-axis within arbitrary subplots.
    Thanks to: https://stackoverflow.com/a/51684195

    :param axs: array of axes which should share x- or y-axis
    :type axs: np.ndarray
    :param target: target for the axes sharing - first axs by default
    :type target: `mpl.axes._subplots.AxesSubplot`
    :param sharex: whether to share x-axis
    :type sharex: bool
    :param sharey: whether to share y-axis
    :type sharey: bool
    """
    if target is None:
        target = axs.flat[0]
    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)
    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1, :].flat:
            ax.xaxis.set_tick_params(
                which="both", labelbottom=False, labeltop=False
            )
            ax.xaxis.offsetText.set_visible(False)
    # Turn off y tick labels and offset text for all but the left most column
    if sharey and axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.yaxis.set_tick_params(
                which="both", labelleft=False, labelright=False
            )
            ax.yaxis.offsetText.set_visible(False)


def plot_conditional_surrogate_distributions(
    gpso_optimiser,
    granularity=N_BINS ** 2,
    mean_limits=[-10, 10],
    var_limits=[0, 5],
    mean_cmap="Spectral",
    var_cmap="plasma",
    fname=None,
):
    """
    Plot conditional surrogate distributions into a grid of N x N, where N is
    the number of parameters that were optimised. Lower triangle contain
    surrogate similarity (i.e. predicted mean) computed on orthogonal slices of
    the search space, going through the best sample for each pair of dimensions.
    Upper triangle shows associated surrogate uncertainty around the best sample
    (serves as a indicator of convergence). Lastly, the diagonal shows weighted
    mean and std of evaluated scores of evaluated scores.

    :param gpso_optimiser: already optimised problem
    :type gpso_optimiser: `gpso.optimisation.GPSOptimiser`
    :param granularity: how many points for surrogate estimation
    :type granularity: int
    :param mean_limits: limits for GP surrogate mean
    :type mean_limits: list|tuple
    :param var_limits: limits for GP surrogate var
    :type var_limits: list|tuple
    :param mean_cmap: colormap for mean plots
    :type mean_cmap: str
    :param var_cmap: colormap for var plots
    :type var_cmap: str
    :param fname: filename for the plot, if None, will show
    :type fname: str|None
    """
    assert isinstance(gpso_optimiser, GPSOptimiser)
    # setup plot
    fig, axes = plt.subplots(
        nrows=gpso_optimiser.param_space.ndim,
        ncols=gpso_optimiser.param_space.ndim,
    )
    # make space at the bottom for colorbars
    plt.subplots_adjust(bottom=0.1)
    # set all subplots in column (except diagonal) to share x-axis
    for column in range(axes.shape[0]):
        _set_share_axes(
            np.array(
                [ax for row, ax in enumerate(axes[:, column]) if row != column]
            ),
            sharex=True,
        )
    # set all subplots in row (except diagonal) to share y-axis
    for row in range(axes.shape[1]):
        _set_share_axes(
            np.array(
                [ax for column, ax in enumerate(axes[row, :]) if row != column]
            ),
            sharey=True,
        )

    # create coordinates at which we would evaluate the surrogate - just copy
    # the best point coordinates granularity squared times
    best_point_coords = np.vstack(
        [gpso_optimiser.gp_surr.highest_score.normed_coord] * granularity ** 2
    )
    # now create coordinates for two dimensions which are changing so we can
    # plot conditional surrogate
    x_normed = np.linspace(0, 1, granularity)
    x_normed, y_normed = np.meshgrid(x_normed, x_normed)
    x_normed_flat, y_normed_flat = x_normed.flatten(), y_normed.flatten()

    # prepare points for histograms on diagonal
    all_eval_points = [
        point
        for point in gpso_optimiser.gp_surr.points
        if point.label == PointLabels.evaluated
    ]

    # iterate over all two member combinations of parameters
    for param1_idx in range(gpso_optimiser.param_space.ndim):
        # per parameter axes and ticks
        axes[0, param1_idx].set_title(
            gpso_optimiser.param_space.parameter_names[param1_idx]
        )
        axes[param1_idx, 0].set_ylabel(
            gpso_optimiser.param_space.parameter_names[param1_idx]
        )
        for param2_idx in range(param1_idx, gpso_optimiser.param_space.ndim):
            # if different parameters - estimate the surrogate
            if param1_idx != param2_idx:
                predict_at = best_point_coords.copy()
                predict_at[:, param1_idx] = x_normed_flat
                predict_at[:, param2_idx] = y_normed_flat
                # coords of best point
                ii = best_point_coords[0, param1_idx]
                jj = best_point_coords[0, param2_idx]
                mean, var = gpso_optimiser.gp_surr.gpflow_model.predict_y(
                    predict_at
                )
                # mean to the lower triangle
                axes[param2_idx, param1_idx].imshow(
                    mean.numpy().reshape(x_normed.shape),
                    vmax=mean_limits[1],
                    vmin=mean_limits[0],
                    cmap=plt.get_cmap(mean_cmap),
                    origin="lower",
                )
                axes[param2_idx, param1_idx].scatter(
                    int(np.around(ii * granularity)),
                    int(np.around(jj * granularity)),
                    marker="x",
                    color="black",
                )
                # var to the upper triangle
                axes[param1_idx, param2_idx].imshow(
                    var.numpy().reshape(x_normed.shape).T,
                    vmax=var_limits[1],
                    vmin=var_limits[0],
                    cmap=plt.get_cmap(var_cmap),
                    origin="lower",
                )
                axes[param1_idx, param2_idx].scatter(
                    int(np.around(jj * granularity)),
                    int(np.around(ii * granularity)),
                    marker="x",
                    color="black",
                )

            elif param1_idx == param2_idx:
                # create bins
                bins = np.linspace(0, 1, N_BINS, endpoint=False)
                scores = np.zeros((bins.shape[0], 2))
                bin_diff = bins[1] - bins[0]
                for i, left_bin in enumerate(bins):
                    # gather scores within the bin
                    bin_scores = [
                        point.score_mu
                        for point in all_eval_points
                        if (
                            point.normed_coord[param1_idx] >= left_bin
                            and point.normed_coord[param1_idx]
                            <= left_bin + bin_diff
                        )
                    ]
                    scores[i, 0] = np.mean(bin_scores)
                    scores[i, 1] = np.std(bin_scores)

                axes[param1_idx, param2_idx].bar(
                    x=bins + (bin_diff * 0.1 / 2),  # offset as 5% of bin diff
                    height=scores[:, 0],
                    width=bin_diff * 0.9,  # width as 90% of bin diff
                    align="edge",
                    facecolor="#AAAAAA",
                    edgecolor="#222222",
                    yerr=scores[:, 1],
                    label="evaluated score",
                )
                axes[param1_idx, param2_idx].set_xlim([0, 1])
                axes[param1_idx, param2_idx].legend()
                # force square plot
                axes[param1_idx, param2_idx].set_aspect(
                    1.0 / axes[param1_idx, param2_idx].get_data_ratio()
                )

    # turn off ticklabels on inner plots
    [
        plt.setp(axes[row, col].get_xticklabels(), visible=False)
        for col in range(axes.shape[1])
        for row in range(0, axes.shape[0] - 1)
    ]
    [
        plt.setp(axes[row, col].get_yticklabels(), visible=False)
        for col in range(1, axes.shape[1] - 1)
        for row in range(0, axes.shape[0])
    ]
    # set last column yticks to be in the right hand side
    for ax in axes[:, -1]:
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

    # set correct x-ticks per original coordinates
    for column in range(axes.shape[1]):
        if column == axes.shape[1] - 1:
            axes[-1, column].set_xticks(np.linspace(0, 1, 5))
        else:
            axes[-1, column].set_xticks(np.linspace(0, granularity, 5))
        axes[-1, column].set_xticklabels(
            np.around(
                np.linspace(
                    gpso_optimiser.param_space.scaler.data_min_[column],
                    gpso_optimiser.param_space.scaler.data_max_[column],
                    5,
                ),
                decimals=TICK_DECIMALS,
            )
        )
    # set correct y-ticks per original coordinates
    for row in range(axes.shape[0]):
        for column in [0, axes.shape[1] - 1]:
            if column == row:
                continue
            axes[row, column].set_yticks(np.linspace(0, granularity, 5))
            axes[row, column].set_yticklabels(
                np.around(
                    np.linspace(
                        gpso_optimiser.param_space.scaler.data_min_[row],
                        gpso_optimiser.param_space.scaler.data_max_[row],
                        5,
                    ),
                    decimals=TICK_DECIMALS,
                )
            )

    # colorbars
    cbar_spacing = 1.0 / 15.0
    mean_cbar_ax = fig.add_axes([cbar_spacing, 0.0, 6 * cbar_spacing, 0.04])
    norm_mean = mpl.colors.Normalize(vmin=mean_limits[0], vmax=mean_limits[1])
    cb_mean = mpl.colorbar.ColorbarBase(
        mean_cbar_ax,
        cmap=plt.get_cmap(mean_cmap),
        norm=norm_mean,
        orientation="horizontal",
    )
    cb_mean.set_label("GP surrogate mean")

    var_cbar_ax = fig.add_axes([8 * cbar_spacing, 0.0, 6 * cbar_spacing, 0.04])
    norm_var = mpl.colors.Normalize(vmin=var_limits[0], vmax=var_limits[1])
    cb_var = mpl.colorbar.ColorbarBase(
        var_cbar_ax,
        cmap=plt.get_cmap(var_cmap),
        norm=norm_var,
        orientation="horizontal",
    )
    cb_var.set_label("GP surrogate uncertainty")

    if fname is not None:
        plt.savefig(fname, bbox_inches="tight", dpi=DPI)
    else:
        plt.show()
    plt.close()
