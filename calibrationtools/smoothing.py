import functools
import logging

from typing import Tuple, Union

import numpy as np
import scipy.stats
import scipy.ndimage

from calibrationtools.util import check_valid_pmfs, make_xy_grids

logger = logging.getLogger(__name__)

class UnexpectedShapeError(Exception):
    """Raise when an input numpy array has a different shape than expected."""

def dynamic_spherical_gaussian(
    model_output: np.ndarray,
    fracs_of_est_variance: np.ndarray,
    arena_dims: Union[Tuple[float, float], np.ndarray],
    desired_resolution: float,
    **kwargs
    ):
    """
    Place a spherical Gaussian at the mean of the given point estimates,
    with the variance set as the expectation of the distance between
    each point estimate and the mean estimate.
    """
    # check if fracs of est variance are negative
    if (fracs_of_est_variance < 0).any():
        raise ValueError(
            'Values in array `fracs_of_est_variance` must be nonnegative!'
            )
    # check that the input is a collection of x,y coordinates
    if model_output.shape[1] != 2 or model_output.ndim != 2:
        raise UnexpectedShapeError(
            f'Expected `model_output` to be an array of (x, y) coordinates ' \
            f'and have shape (n_estimates, 2). Recieved shape: {model_output.shape}.'
            )

    mean = model_output.mean(axis=0)
    logger.debug(f'mean estimate: {mean}, with shape {mean.shape}')
    mean_distance = np.linalg.norm(model_output - mean[None, :]).mean()
    logger.debug(
        f'mean distance between point estimates and centroid: {mean_distance}'
        )

    # create grid of points at which to evaluate the pdf
    xgrid, ygrid = make_xy_grids(
        arena_dims,
        resolution=desired_resolution,
        return_center_pts=True
        )
    coord_grid = np.dstack((xgrid, ygrid))

    # now assemble an array of probability mass functions by smoothing
    # the location estimates with each std value
    # since grids track the edgepoints and pmfs tracks the bins,
    # pmfs should have one less value in each coordinate direction
    pmfs = np.zeros((len(fracs_of_est_variance), *xgrid.shape))

    for i, frac in enumerate(fracs_of_est_variance):
        distr = scipy.stats.multivariate_normal(
            mean=mean,
            cov=(frac * mean_distance)
        ).pdf(coord_grid)
        distr /= distr.sum()
        pmfs[i] = distr
    
    return pmfs


def gaussian_mixture(
    model_output: np.ndarray,
    std_values: np.ndarray,
    arena_dims: Union[Tuple[float, float], np.ndarray],
    desired_resolution: float,
    **kwargs
    ):
    """
    Create a Gaussian mixture pmf where one spherical Gaussian of a certain variance
    is placed at each location estimate provided.
    """
    for kwarg, value in kwargs.items():
        logger.info(f'Ignoring unexpected keyword argument {kwarg} with value {value}.')
    # check that the input is a collection of x,y coordinates
    if model_output.shape[1] != 2 or model_output.ndim != 2:
        raise UnexpectedShapeError(
            f'Expected `model_output` to be an array of (x, y) coordinates ' \
            f'and have shape (n_estimates, 2). Recieved shape: {model_output.shape}.'
            )
    
    # create grid of points at which to evaluate the pdfs we define
    xgrid, ygrid = make_xy_grids(
        arena_dims,
        resolution=desired_resolution,
        return_center_pts=True
        )
    coord_grid = np.dstack((xgrid, ygrid))

    # now assemble an array of probability mass functions by smoothing
    # the location estimates with each std value
    pmfs = np.zeros((len(std_values), *xgrid.shape))

    for i, std in enumerate(std_values):
        for loc_estimate in model_output:
            # place a spherical gaussian with variance sigma
            # at the individual location estimate
            distr = scipy.stats.multivariate_normal(
                mean=loc_estimate,
                cov= (std ** 2)
            )
            # and add it to the corresponding entry in pmfs
            pmfs[i] += distr.pdf(coord_grid)

    # renormalize
    # get total sum over grid, adding axes so broadcasting works
    sum_per_sigma_val = pmfs.sum(axis=(1, 2)).reshape((-1, 1, 1))
    pmfs /= sum_per_sigma_val
    return pmfs

def gaussian_blur(
    model_output: np.ndarray,
    std_values: np.ndarray,
    renormalize=True,
    **kwargs
    ) -> np.ndarray:
    """
    Convolve a Gaussian kernel over the input, applying a softmax transformation
    before doing the blurring.
    """
    for kwarg, value in kwargs.items():
        logger.info(f'Ignoring unexpected keyword argument {kwarg} with value {value}.')
    # check input shape. expected shape: (n_ests, n_x_pts, n_y_pts)
    if model_output.ndim != 3:
        raise UnexpectedShapeError(
            f'Expected `model_output` to be a collection of grids, with three dimensions. ' \
            f'Instead, recieved the following shape: {model_output.shape}.'
            )
    
    # if the provided standard deviation values don't include 0 (no smoothing),
    # add it to the list. this is crude but it allows us to use one `std_values`
    # array for both smoothing methods.
    # if 0 not in std_values:
    #     std_values = np.insert(std_values, 0, 0)

    softmaxed = softmax(model_output, renormalize=renormalize)

    blurred = np.zeros((len(std_values), *softmaxed.shape))

    for i, std in enumerate(std_values):
        blurred[i] = scipy.ndimage.gaussian_filter(
            softmaxed,
            sigma=[0, std, std]  # blur only within grids, not between them
        )

    # and finally average the grids for each sample
    return blurred.mean(axis=1)

def no_smoothing(model_output, **kwargs):
    """
    Apply no smoothing to the output. Here for consistency and to ensure
    that model_output can be interpreted as pmfs.
    """
    # if the output only has two dimensions, add one at the start
    # to represent the number of estimates per sample. this is just
    # for convenience.
    if model_output.ndim == 2:
        model_output = model_output[None]
    err_prefix = (
        f'Expected output to be a valid probability '
        f'mass function since no smoothing method was provided to '
        f'the CalibrationAccumulator constructor.'
        )
    check_valid_pmfs(model_output, prefix_str=err_prefix)
    return model_output

def softmax(model_output, renormalize=True, **kwargs):
    """
    Apply a softmax to the model output, optionally renormalizing it.
    """
    if model_output.ndim != 3:
        raise UnexpectedShapeError(
            f'Expected `model_output` to be a collection of grids, with three dimensions. ' \
            f'Instead, recieved the following shape: {model_output.shape}.'
            )
    
    # renormalize the grids so each entry is in the range [-1, 1],
    # since MUSE RSRP values seem to be able to reach magnitudes like 1e13,
    # which the exp function inside softmax just blows up to infinity.
    if renormalize:
        model_output /= model_output.max(axis=(1, 2))[:, None, None]

    # softmax the grids
    axes_to_sum_over = range(2, model_output.ndim)  # sum over every axis but the first two
    sum_per_grid = np.exp(model_output).sum(axis=tuple(axes_to_sum_over))    
    softmaxed = np.exp(model_output) / sum_per_grid[:, :, None]
    return softmaxed


SMOOTHING_FUNCTIONS = {
    'gaussian_mixture': gaussian_mixture,
    'gaussian_blur': gaussian_blur,
    'no_smoothing': no_smoothing,
}

# names of the necessary arguments for the smoothing functions
# used by CalibrationAccumulator
# to check whether the correct params have been passed for
# a desired smoothing function
NECCESARY_KWARGS = {
    'gaussian_mixture': ['std_values', 'desired_resolution', 'arena_dims'],
    'gaussian_blur': ['std_values'],
    'no_smoothing': [],
    'softmax': [],
}

N_CURVES_PER_FN = {
    'gaussian_mixture': 'std_values',
    'gaussian_blur': 'std_values',
    'no_smoothing': 1,
    'softmax': 1,
}
