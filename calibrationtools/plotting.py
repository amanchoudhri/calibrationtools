"""Convenient plotting utilities."""

import logging
import math

from typing import List, Tuple

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def subplots(
    n_plots,
    scale_factor=4,
    sharex=True,
    sharey=True,
    **kwargs
) -> Tuple[Figure, List[Axes]]:
    """
    Create nicely sized and laid-out subplots for a desired number of plots.
    """
    if n_plots == 1:
        fig, ax = plt.subplots(1, 1, figsize=(scale_factor, scale_factor))
        return fig, [ax]  # wrap ax in a list so it's iterable downstream

    # essentially we want to make the subplots as square as possible
    # number of rows is the largest factor of n_plots less than sqrt(n_plots)
    options = range(1, int(math.sqrt(n_plots) + 1))
    n_rows = max(filter(lambda n: n_plots % n == 0, options))
    n_cols = int(n_plots / n_rows)
    # now generate the Figure and Axes pyplot objects
    # cosmetic scale factor to make larger plot
    figsize = (n_cols * scale_factor, n_rows * scale_factor)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=figsize,
        sharex=sharex, sharey=sharey,
        **kwargs
        )
    if n_rows == 1:
        flattened_axes = axs
    else:
        flattened_axes = [ax for ax_row in axs for ax in ax_row]
    return fig, flattened_axes


def plot_calibration_curve(
    true_props,
    ax=None
) -> Axes:
    """
    Plot the given calibration curve, returning the x and y values
    along with the axis on which the curve was plotted.
    """
    if not ax:
        _, ax = plt.subplots()
    bin_edges = np.arange(1, len(true_props) + 1) / len(true_props)
    ax.scatter(bin_edges, true_props)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='dashed')
    ax.set_xlabel('Probability assigned to region around mode')
    ax.set_ylabel('True proportion of samples in region')
    return ax


def plot_err_curve(abs_err, signed_err, xlabels=None) -> Tuple[Figure, Tuple[Axes, Axes]]:
    if len(abs_err) != len(signed_err):
        raise ValueError('`abs_err` and `signed_err` must have the same length!')
    if len(abs_err) == 1:
        logger.info('Error array of only one point encountered. Skipping...')
        return None, (None, None)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))

    if xlabels is None:
        xlabels = np.arange(len(abs_err))
    ax0.scatter(xlabels, abs_err)
    ax0.set_ylabel('absolute calibration error')
    ax1.scatter(xlabels, signed_err)
    ax1.set_ylabel('signed calibration error')
    return fig, (ax0, ax1)
