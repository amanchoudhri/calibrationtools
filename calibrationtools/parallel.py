import logging

from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from calibrationtools.calculate import min_mass_containing_location
from calibrationtools.smoothing import SMOOTHING_FUNCTIONS
from calibrationtools.util import check_valid_pmfs, make_xy_grids

logger = logging.getLogger(__name__)

def _calibration_step(
    pmfs,
    true_location,
    arena_dims,
    std_values,
    n_calibration_bins: int = 10,
    grid_resolution: float = 0.5,
    ) -> np.ndarray:
    """
    Actual calibration step calculation function. See `calibration_step`
    for more details.
    """
    # get our x and ygrids
    xgrid, ygrid = make_xy_grids(arena_dims, grid_resolution)

    # repeat location so we can use the vectorized min_mass_containing_location fn
    true_loc_repeated = true_location.repeat(len(std_values), axis=0)

    # perform the calibration calculation step
    m_vals = min_mass_containing_location(
        pmfs,
        true_loc_repeated,
        xgrid,
        ygrid
    )

    # transform to the bin in [0, 1] to which each value corresponds,
    # essentially iteratively building a histogram with each step
    bins = np.linspace(0, 1, n_calibration_bins)
    # subtract one to track the left hand side of the bin
    bin_idxs = np.digitize(m_vals, bins) - 1

    return bin_idxs

def calibration_from_steps(cal_step_bulk: np.array):
    """
    Calculate calibration curves and error from the collected
    results of `calibration_step`.
    
    Args:
        cal_step_bulk: Results from `calibration_step`, should have
            shape (n_sigma_values, n_bins).
    """
    # calculate the calibration curve by taking the cumsum
    # and dividing by the total sum (adding extra axis so broadcasting works)
    calibration_curves = cal_step_bulk.cumsum(axis=1) / cal_step_bulk.sum(axis=1)[:, None]
    # next, calculate the errors
    # get probabilities the model assigned to each region
    # note: these are also the bucket edges in the histogram
    n_bins = calibration_curves.shape[1]
    assigned_probabilities = np.arange(1, n_bins + 1) / n_bins
    # get the sum of residuals between the assigned probabilities
    # and the true observed proportions for each value of sigma
    residuals = calibration_curves - assigned_probabilities
    abs_err = np.abs(residuals).sum(axis=1)
    signed_err = residuals.sum(axis=1)
    return calibration_curves, abs_err, signed_err


def calibration_step(
    model_output: np.ndarray,
    true_location: np.ndarray,
    arena_dims: Union[Tuple[float, float], np.ndarray],
    smoothing_method: Optional[Union[str, Callable]] = None,
    sigma_values: Optional[np.ndarray] = None,
    n_calibration_bins: int = 10,
):
    """
    Perform one step of the calibration process on `model_output`. Optionally,
    smooth `model_output` before performing the calibration step.

    Args:
        model_output: Array of location estimates from a model,
            for one audio sample. Expected shape: (n_predictions, 2).
        true_location: Array contianing the true location of the audio sample, in
            centimeters. Expected shape: (1, 2).
        arena_dims: Config parameter storing the dimensions of the arena, in
            centimeters.
        smoothing_method: Optional parameter indicating whether the model output
            should be smoothed before calculating the calibration step. Can either
            be a user-defined smoothing function, or a string referring to a
            function predefined in `smoothing.py`.
        sigma_values: Array of variances used to smooth the predictions.
        n_calibration_bins: integer representing how fine the calibration
            curve should be. Default: 10.
        grid_resolution: Desired resolution to use when creating the discrete
            probability distributions representing the model output. Default: 0.5.

    Returns:
        An array `arr` of shape (len(sigma_values),), where each entry arr[i]
        represents the calibration mass output for the given sample when predictions
        were smoothed with variance sigma_values[i].

    Essentially, we place a spherical Gaussian with variance $sigma^2$ at
    each location estimate in `model_output`, then sum and average over all location 
    estimates for the given sample to create a new probability mass function.
    
    Then, we calculate $m$, the probability assigned to the smallest region in the xy 
    plane containing the true location, where these regions are defined by
    progressively taking the location bins to which the model assigns the highest probability mass.

    We repeat this process for each value of sigma in `sigma_vals`, and return
    the resulting array.

    Apply the provided smoothing methods to `model_output` before calculating
    one calibration step using `calibration_step`.
    """
    allowed_method_strs = list(SMOOTHING_FUNCTIONS.keys())
    # if no smoothing method is provided, make sure that the model output
    # is an array of valid pmfs.
    if smoothing_method is None:
        err_prefix = 'Expected `model_output` to be a valid probability ' \
            'mass function since param `smoothing_method` was not provided ' \
            'to `calibration_step`. '
        check_valid_pmfs(model_output, prefix_str=err_prefix)
    # if smoothing method is a string, it should refer to
    # one of the predefined smoothing functions in
    # smoothing.py
    elif type(smoothing_method) == str:
        try:
            smoothing_fn = SMOOTHING_FUNCTIONS[smoothing_method]
        except KeyError as e:
            err_msg = f'Smoothing method {smoothing_method} unrecognized. ' \
                f'Allowed options: {allowed_method_strs}'
            logger.error(err_msg)
            raise KeyError(err_msg) from e
    # allow the user to pass in their own smoothing_method as a callable
    elif callable(smoothing_method):
        smoothing_fn = smoothing_method
    # if it's provided but is neither a string nor a callable, throw an error
    else:
        raise ValueError(
            f'Expected smoothing_method to be either None, a callable, ' \
            f'or a string in {allowed_method_strs}. Instead, recieved object ' \
            f'{smoothing_method} of type {type(smoothing_method)}.'
            )

    pmfs = model_output

    if smoothing_method:
        pmfs = smoothing_fn(
            model_output=pmfs,
            std_values=sigma_values,
            arena_dims=arena_dims
        )

    bin_idxs = _calibration_step(
        pmfs,
        true_location,
        arena_dims,
        n_calibration_bins=n_calibration_bins,
    )

    return bin_idxs
