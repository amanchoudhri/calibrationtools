import functools
import logging

from typing import Optional, Tuple, Union

import numpy as np
import scipy.stats
import scipy.ndimage

from calibrationtools.util import check_valid_pmfs, make_xy_grids

logger = logging.getLogger(__name__)


class UnexpectedShapeError(Exception):
    """Raise when an input numpy array has a different shape than expected."""


SMOOTHING_FUNCTIONS = {}

# names of the necessary arguments for the smoothing functions
# used by CalibrationAccumulator
# to check whether the correct params have been passed for
# a desired smoothing function
NECCESARY_KWARGS = {
    'gaussian_mixture': ['std_values', 'desired_resolution', 'arena_dims'],
    'gaussian_blur': ['std_values'],
    'dynamic_spherical_gaussian': ['fracs_of_est_variance'],
}

N_CURVES_PER_FN = {}


def _check_shape(model_output, mode=None):
    """
    Verify that the provided model output array has the correct
    shape and number of dimensions.
    """
    if not mode:
        return
    elif mode == 'locations':
        # input should be a collection of x,y coordinates
        if model_output.shape[1] != 2 or model_output.ndim != 2:
            raise UnexpectedShapeError(
                f'Expected `model_output` to be an array of (x,y) coordinates '
                f'and have shape (n_estimates, 2). Recieved shape: '
                f'{model_output.shape}.'
                )
    elif mode == 'grids':
        # expected shape: (n_ests, n_x_pts, n_y_pts)
        if model_output.ndim != 3:
            raise UnexpectedShapeError(
                f'Expected `model_output` to be a collection of grids, with '
                f'three dimensions. Instead, recieved the following shape: '
                f'{model_output.shape}.'
                )
    else:
        raise ValueError('mode must be one of \'locations\', \'grids\'!')


def _smoothing_fn(
    n_curves_per_sample: Union[str, int],
    input_type: Optional[str] = None
):
    """
    Decorator to register the number of calibration curves per sample
    the given smoothing function returns. This is used to correctly
    allocate space in the CalibrationAccumulator class.

    Optionally, also verify that the provided model output array has the
    correct number of dimensions.

    In addition, register a smoothing function `f` in the dict
    `SMOOTHING_FUNCTIONS` under the key `f.__name__` and set the
    default value in `NECESSARY_KWARGS` for the function
    as the empty list.
    """
    def wrapper(f):
        smoothing_fn = f.__name__
        # register the function into the necessary dictionaries
        SMOOTHING_FUNCTIONS[smoothing_fn] = f
        NECCESARY_KWARGS.setdefault(smoothing_fn, [])
        N_CURVES_PER_FN[smoothing_fn] = n_curves_per_sample

        @functools.wraps(f)
        def decorator(*args, **kwargs):
            # check the correct model output ndims
            model_output = args[0]
            _check_shape(model_output, input_type)
            output = f(*args, **kwargs)
            check_valid_pmfs(output)
            return output
        return decorator
    return wrapper


@_smoothing_fn('fracs_of_est_variance', 'locations')
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

    mean = model_output.mean(axis=0)
    logger.debug(f'mean estimate: {mean}, with shape {mean.shape}')
    mean_distance = np.linalg.norm(model_output - mean[None, :]).mean()
    logger.debug(
        f'mean distance between point estimates and centroid: {mean_distance}'
        )

    # lower bound the distance away from zero by
    # some arbitrary value
    if mean_distance == 0:
        mean_distance = 1e-3

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


@_smoothing_fn('std_values', 'locations')
def gaussian_mixture(
    model_output: np.ndarray,
    std_values: np.ndarray,
    arena_dims: Union[Tuple[float, float], np.ndarray],
    desired_resolution: float,
    **kwargs
):
    """
    Create a Gaussian mixture pmf where one spherical Gaussian of a certain
    variance is placed at each location estimate provided.
    """
    for kwarg, value in kwargs.items():
        logger.info(
            f'Ignoring unexpected keyword argument {kwarg} with value {value}.'
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
                cov=(std ** 2)
            )
            # and add it to the corresponding entry in pmfs
            pmfs[i] += distr.pdf(coord_grid)

    # renormalize
    # get total sum over grid, adding axes so broadcasting works
    sum_per_sigma_val = pmfs.sum(axis=(1, 2)).reshape((-1, 1, 1))
    pmfs /= sum_per_sigma_val
    return pmfs


@_smoothing_fn('std_values', 'grids')
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
        logger.info(
            f'Ignoring unexpected keyword argument {kwarg} with value {value}.'
            )

    softmaxed = softmax(model_output, renormalize=renormalize)

    blurred = np.zeros((len(std_values), *softmaxed.shape))

    for i, std in enumerate(std_values):
        blurred[i] = scipy.ndimage.gaussian_filter(
            softmaxed,
            sigma=[0, std, std]  # blur only within grids, not between them
        )

    # and finally average the grids for each sample
    return blurred.mean(axis=1)


@_smoothing_fn(1)
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
        'Expected output to be a valid probability '
        'mass function since no smoothing method was provided to '
        'the CalibrationAccumulator constructor.'
        )
    check_valid_pmfs(model_output, prefix_str=err_prefix)
    return model_output


@_smoothing_fn(1, 'grids')
def softmax(model_output, renormalize=True, **kwargs):
    """
    Apply a softmax to the model output, optionally renormalizing it.
    """
    # renormalize the grids so each entry is in the range [-1, 1],
    # since MUSE RSRP values seem to be able to reach magnitudes like 1e13,
    # which the exp function inside softmax just blows up to infinity.
    axes_to_sum_over = (1, 2)
    if renormalize:
        model_output /= model_output.max(axis=axes_to_sum_over)[:, None, None]

    # softmax the grids
    sum_per_grid = np.exp(model_output).sum(axis=axes_to_sum_over)
    softmaxed = np.exp(model_output) / sum_per_grid[:, None, None]

    # average the grids to return one pmf
    # and add an extra dimension to match expected shape
    averaged = softmaxed.mean(axis=0)[None]
    return averaged
