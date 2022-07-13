import logging

from typing import Tuple, Union

import numpy as np
import scipy.stats
import scipy.ndimage

from calibrationtools.util import make_xy_grids

logger = logging.getLogger(__name__)

class UnexpectedShapeError(Exception):
    """Raise when an input numpy array has a different shape than expected."""

def gaussian_mixture(
    model_output: np.ndarray,
    std_values: np.ndarray,
    arena_dims: Union[Tuple[float, float], np.ndarray],
    desired_resolution = 0.005,
    **kwargs
    ):
    for kwarg, value in kwargs.items():
        logger.info(f'Ignoring unexpected keyword argument {kwarg} with value {value}.')
    # check that the input is a collection of x,y coordinates
    if model_output.shape[1] != 2 or model_output.ndim != 2:
        raise UnexpectedShapeError(
            f'Expected `model_output` to be an array of (x, y) coordinates ' \
            f'and have shape (n_estimates, 2). Recieved shape: {model_output.shape}.'
            )
    
    # create grid of points at which to evaluate the pdfs we define
    xgrid, ygrid = make_xy_grids(arena_dims, resolution=desired_resolution)
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

    # renormalize the grids so each entry is in the range [-1, 1],
    # since MUSE RSRP values seem to be able to reach magnitudes like 1e13,
    # which the exp function inside softmax just blows up to infinity.
    if renormalize:
        model_output /= model_output.max(axis=(1, 2))[:, None, None]

    # softmax the grids
    axes_to_sum_over = range(2, model_output.ndim)  # sum over every axis but the first two
    sum_per_grid = np.exp(model_output).sum(axis=tuple(axes_to_sum_over))    
    softmaxed = np.exp(model_output) / sum_per_grid[:, :, None]

    blurred = np.zeros((len(std_values), *softmaxed.shape))

    for i, std in enumerate(std_values):
        blurred[i] = scipy.ndimage.gaussian_filter(
            softmaxed,
            sigma=[0, std, std]  # blur only within grids, not between them
        )

    # and finally average the grids for each sample
    return blurred.mean(axis=1)

SMOOTHING_FUNCTIONS = {
    'gaussian_mixture': gaussian_mixture,
    'gaussian_blur': gaussian_blur,
}
