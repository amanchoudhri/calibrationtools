import functools
import logging
import multiprocessing
import pathlib

from collections import defaultdict
from typing import Any, Callable, Mapping, Optional, Tuple, Union

import h5py
import numpy as np

from matplotlib import pyplot as plt

from calibrationtools.calculate import digitize, min_mass_containing_location
from calibrationtools.smoothing import (
    SMOOTHING_FUNCTIONS,
    NECCESARY_KWARGS,
    N_CURVES_PER_FN
    )
from calibrationtools.util import check_valid_pmfs, make_xy_grids
from calibrationtools.plotting import subplots, plot_calibration_curve, plot_err_curve

logger = logging.getLogger(__name__)


SmoothingFunction = Union[str, Callable]
SmoothingSpec = Mapping[SmoothingFunction, Mapping[str, Any]]


class CalibrationAccumulator:
    """
    Helper class to simplify the calculation of calibration,
    with support for multiple output types and applying various
    smoothing methods to each output type.
    """

    def __init__(
        self,
        arena_dims: Union[Tuple[float, float], np.ndarray],
        smoothing_specs_for_outputs: Mapping[str, SmoothingSpec],
        use_multiprocessing: bool = True,
        n_calibration_bins: Optional[int] = 10,
    ):
        """
        Initialize a CalibrationAccumulator object, which is a helper class
        to simplify the calculation of calibration in an online manner (iterating
        through or asynchronously going through a dataset).

        Note: the parameter `smoothing_specs_for_outputs` has a very specific
        schema, described as follows.

        The top level keywords should be names corresponding to various outputs
        from the model. For example, MUSE outputs point estimates (`r_ests`) as well
        as RSRP grids (`rsrp_grids`).

        For each model output type, the value should be another dictionary, which
        I've typed as a `SmoothingSpec`. Essentially, the keys of this smoothing spec
        should be strings referring to predefined smoothing functions or optionally a
        custom function, which should be applied to the given input. To apply no
        smoothing, simply pass an empty dictionary.

        Lastly, the values associated with each `SmoothingFunction` should be one more
        dictionary of the necessary kwargs for the smoothing function. This is mostly
        relevant for predefined smoothing functions.

        All in all, an example of a valid usage could be the following:
        ```
        ARENA_DIMS = (0.6, 0.4)  # dimensions of the arena, in meters
        STD_VALUES = np.linspace(0.1, 10, 25)
        FRACTIONS_OF_EST_VARIANCE = np.linspace(0.01, 3, 50)
        PMF_RESOLUTION = 0.01  # desired resolution for the pmf, in meters
        smoothing_specs = {
            'r_ests': {
                'gaussian_mixture': {
                    'std_values': STD_VALUES,
                    'desired_resolution': PMF_RESOLUTION
                },
                'dynamic_spherical_gaussian': {
                    'fracs_of_est_variance': FRACTIONS_OF_EST_VARIANCE,
                    'desired_resolution': PMF_RESOLUTION
                }
            },
            'rsrp_grids': {
                'softmax': {},
                'gaussian_blur': {'std_values': STD_VALUES}
            }
        }
        ```
        """
        if use_multiprocessing:
            self.pool = multiprocessing.Pool()
            self.logger = multiprocessing.log_to_stderr()
        else:
            self.logger = logging.getLogger('CalibrationAccumulator')
        self.output_names = list(smoothing_specs_for_outputs.keys())
        # todo: validate smoothing methods ?
        # todo: build smoothing functions here ?
        self.smoothing_for_outputs = smoothing_specs_for_outputs
        self.n_calibration_bins = n_calibration_bins

        self.arena_dims = arena_dims
        self.logger.debug(f'arena_dims recieved: {self.arena_dims}')
        self.use_mp = use_multiprocessing

        # initialize the internal mass counts tracking arrays
        self.mass_counts = defaultdict(dict)

        for output_type, smoothing_spec in smoothing_specs_for_outputs.items():
            # if the smoothing_spec dict is empty, assume the user
            # meant to apply no smoothing.
            if not smoothing_spec:
                smoothing_spec['no_smoothing'] = {}

            for smoothing_method, params in smoothing_spec.items():
                # preload arena_dims into params so the user
                # doesn't have to pass it twice
                params['arena_dims'] = self.arena_dims
                # TODO: validate smoothing_method
                if type(smoothing_method) == str:
                    # check that the smoothing method is valid
                    if smoothing_method not in SMOOTHING_FUNCTIONS:
                        raise KeyError(
                            f'Invalid smoothing string \'{smoothing_method}\''
                            f'recieved! Valid options are: '
                            f'{SMOOTHING_FUNCTIONS.keys()}.'
                        )
                    # if the user selected a predefined smoothing method but
                    # didn't pass the necessary kwargs for the function
                    # (like std_values), throw an error
                    all_kwargs_present = all(
                        kwarg in params for kwarg in NECCESARY_KWARGS[smoothing_method]
                        )
                    if not all_kwargs_present:
                        raise ValueError((
                            'Invalid smoothing params passed for method '
                            f'{smoothing_method}! Expected param dictionary '
                            f'to contain keys: {NECCESARY_KWARGS[smoothing_method]}. '
                            f'Instead found keys: {params.keys()}'
                            ))
                    # if all necessary args were provided, allocate the correct
                    # amount of space for the smoothing method
                    n_curves = N_CURVES_PER_FN[smoothing_method]
                    # if the entry in N_CURVES_PER_FN is a string, it means
                    # look to the length of that keyword argument in the param
                    # dictionary
                    if type(n_curves) == str:
                        n_curves = len(params[n_curves])
                    # otherwise, it's an integer representing how many curves
                    # per sample are returned and we can continue.
                    mass_counts_arr = np.zeros((n_curves, n_calibration_bins))

                elif callable(smoothing_method):
                    self.logger.debug(
                        f'smoothing method {smoothing_method} recieved '
                        f'with params {params}'
                        )
                    # if the user passes in a custom function, have them
                    # pass in how many calibration curves their function
                    # will create per sample
                    if 'n_curves_per_sample' not in params:
                        raise ValueError(
                            f'Parameter dictionary for custom smoothing method '
                            f'must containg key `n_curves_per_sample`. '
                            f'For smoothing method {smoothing_method}, '
                            f'instead encountered param dict: {params}'
                            )
                    mass_counts_arr = np.zeros(
                        (params['n_curves_per_sample'], n_calibration_bins)
                        )

                self.mass_counts[output_type][smoothing_method] = mass_counts_arr
                logging.debug(
                    f'Created mass counts array for output: `{output_type}` '
                    f'and smoothing `{smoothing_method}`'
                    )

        self.logger.info('Successfully initialized CalibrationAccumulator.')
        self.logger.debug(f'Smoothing specs for CalibrationAccumulator: {self.smoothing_for_outputs}')

    def calculate_step(
        self,
        model_outputs: Mapping[str, np.ndarray],
        true_location: np.ndarray,
        pmf_save_path: Optional[str] = None,
    ):
        """
        Perform one step of the calibration process on `model_output`.
        Optionally, smooth `model_output` with the smoothing methods
        specified to the constructor before calculating.

        Args:
            model_output: Dictionary mapping the string names of the outputs
                defined from constructor argument smoothing_specs_for_outputs
                to numpy arrays representing the corresponding model output.
            true_location: Array contianing the true location of the audio sample, in
                centimeters. Expected shape: (1, 2).

        Essentially, this function takes in the output from a model,
        optionally smoothing it to make it a valid pmf. Then, it
        calculates $m$, the probability assigned to the smallest region
        in the xy plane containing the true location, where these regions
        are defined by progressively taking the location bins to which the
        model assigns the highest probability mass.

        We repeat this process as many times as is defined by the parameters in
        self.smoothing_for_outputs.
        """
        for output_name in self.output_names:
            model_output = model_outputs[output_name]
            smoothing_specs = self.smoothing_for_outputs[output_name]

            for smoothing_method, params in smoothing_specs.items():
                smoothing_fn = smoothing_method
                # if the smoothing method is a string,
                # get the actual function object and prefill all the
                # arguments so the only argument is the model output
                if type(smoothing_method) == str:
                    smoothing_fn = SMOOTHING_FUNCTIONS[smoothing_method]
                    smoothing_fn = functools.partial(smoothing_fn, **params)

                def _update_mass_counts(bin_idxs):
                    """
                    Given the calibration step result from one smoothing method
                    applied to one output type, update the corresponding internal
                    mass counts tracker.
                    """
                    counts_to_update = self.mass_counts[output_name][smoothing_method]
                    for sigma_idx, bin_idx in enumerate(bin_idxs):
                        counts_to_update[sigma_idx][bin_idx] += 1

                smoothed_output = smoothing_fn(model_output)

                # check to make sure that the output is a valid pmf
                check_valid_pmfs(
                    smoothed_output,
                    f'Expected the output of smoothing method '
                    f'{smoothing_method} to be a valid pmf.'
                    )

                # if a pmf save path was provided, save the outputs
                # to that directory
                if pmf_save_path is not None:
                    outdir = pathlib.Path(pmf_save_path) / output_name
                    outdir.mkdir(exist_ok=True, parents=True)
                    outfile = outdir / str(smoothing_method)
                    self.logger.info(f'Saving smoothed pmfs to path {outfile}')
                    # rescale loc since we dont have x/y grid information
                    n_y_pts, n_x_pts = smoothed_output[0].shape
                    rescaled_loc = (true_location / self.arena_dims) * (n_x_pts, n_y_pts)
                    rescaled_loc = rescaled_loc.squeeze() # reduce to a (2,) vector for pyplot
                    self.logger.debug(f'rescaled location: {rescaled_loc}')
                    if len(smoothed_output) > 1:
                        fig, axs = subplots(len(smoothed_output))
                        for i, (ax, pmf) in enumerate(zip(axs, smoothed_output)):
                            ax.contourf(pmf)
                            if smoothing_method in SMOOTHING_FUNCTIONS:
                                params = self.smoothing_for_outputs[output_name][
                                    smoothing_method]
                                kw = N_CURVES_PER_FN[smoothing_method]
                                ax.set_title(f'varying param: {params[kw][i]}')
                                ax.plot(*rescaled_loc, 'ro', label='true location')
                        fig.tight_layout()
                    else:
                        _, ax = plt.subplots()
                        ax.contourf(smoothed_output[0])
                        ax.plot(*rescaled_loc, 'ro', label='true location')
                    plt.savefig(outfile)

                if self.use_mp:
                    self.pool.apply_async(
                        _calibration_step,
                        (
                            smoothed_output,
                            true_location,
                            self.arena_dims
                        ),
                        {
                            'n_calibration_bins': self.n_calibration_bins
                        },
                        callback=_update_mass_counts
                    )
                else:
                    bin_idxs = _calibration_step(
                        smoothed_output,
                        true_location,
                        self.arena_dims,
                        n_calibration_bins=self.n_calibration_bins,
                    )
                    _update_mass_counts(bin_idxs)

    def calculate_curves_and_error(self, h5_file: Optional[h5py.File] = None):
        """
        Calculate calibration curves and error from the collected
        results of all the calibration steps.
        """
        # if we use multiprocessing, wait for the workers to finish
        if self.use_mp:
            self.pool.close()
            self.logger.info('Joining pool processes. Waiting for workers to finish...')
            self.pool.join()

        # initialize a result dictionary
        self.results = {
            output_name: defaultdict(dict) for output_name in self.output_names
            }
        self.logger.info('Calculating results.')
        for output_name, mass_counts_by_smoothing in self.mass_counts.items():
            for smoothing_method, mass_counts_arr in mass_counts_by_smoothing.items():
                curves, abs_err, signed_err = calibration_from_steps(
                    mass_counts_arr
                    )
                result_subdict = self.results[output_name][smoothing_method]
                result_subdict['curves'] = curves
                result_subdict['abs_err'] = abs_err
                result_subdict['signed_err'] = signed_err

        self.logger.info('Successfully calculated results.')

        if h5_file:
            self.logger.info(f'Writing results to h5 file {h5_file}.')
            cal_grp = h5_file.create_group('calibration')
            for output_name, res_by_smoothing in self.results.items():
                output_grp = cal_grp.create_group(output_name)
                for smoothing_method, results in res_by_smoothing.items():
                    g = output_grp.create_group(smoothing_method)
                    # save the parameters to the attrs of g
                    params = self.smoothing_for_outputs[output_name][smoothing_method]
                    for kwarg, value in params.items():
                        g.attrs[kwarg] = value
                    # and write the actual result data as well
                    for r_type, result in results.items():
                        g.create_dataset(r_type, data=result)
            self.logger.info(f'Successfully wrote results to file {h5_file}')
        return self.results

    def plot_results(self, img_directory: str):
        """
        Plot the results for all combinations in the given
        directory with the filestructure:
        ```
        img_directory
            |- output_method
                |- smoothing_method
                    |- 'curves.png'
                    |- 'errs.png'
                |- smoothing_method
                    |- ...
            |- ...
        ```
        """
        # make sure the user calculated the results first
        if not hasattr(self, 'results'):
            raise Exception(
                '`plot_results` called before results were calculated! '
                'Please call `calculate_curves_and_error` first.'
            )
        basedir = pathlib.Path(img_directory)
        # then iterate through them and save the files after plotting
        for output, results_by_smoothing in self.results.items():
            for smoothing_method, r in results_by_smoothing.items():
                params = self.smoothing_for_outputs[output][smoothing_method]
                # plot the calibration curves

                # if a predefined smoothing method was
                # passed, create more descriptive
                # titles by accessing the params varied over
                varied_params = None
                if type(smoothing_method) == str:
                    varied_param_name = N_CURVES_PER_FN[smoothing_method]
                    if type(varied_param_name) == str:
                        varied_params = params[varied_param_name]

                curves = r['curves']
                fig, axs = subplots(len(curves))
                for i, (ax, curve) in enumerate(zip(axs, curves)):
                    plot_calibration_curve(curve, ax=ax)
                    if varied_params is not None:
                        ax.set_title(f'{varied_param_name}: {varied_params[i]}')
                fig.tight_layout()

                # save the figure
                outdir = basedir / output / smoothing_method
                outdir.mkdir(exist_ok=True, parents=True)

                plt.savefig(outdir / 'curves.png')
                plt.close()

                # plot the errors
                fig, axs = plot_err_curve(
                    r['abs_err'],
                    r['signed_err'],
                    xlabels=varied_params
                    )
                # if only one error point is recieved, plot_err_curve
                # returns None for fig.
                if fig:
                    fig.tight_layout()
                    plt.savefig(outdir / 'errs.png')
                plt.close()


def _calibration_step(
    pmfs,
    true_location,
    arena_dims,
    n_calibration_bins: int = 10
) -> np.ndarray:
    """
    Actual calibration step calculation function. See
    `CalibrationAccumulator.calibration_step` for more details.
    """
    # get our x and ygrids to match the shape of the pmfs
    # since the grids track the edge points, we should have
    # one more point in each coordinate direction.
    grid_shape = np.array(pmfs.shape[1:]) + 1
    xgrid, ygrid = make_xy_grids(arena_dims, shape=grid_shape)

    # reshape location to (1, 2) if necessary
    # we do this so the repeat function works out correctly
    if true_location.shape != (1, 2):
        true_location = true_location.reshape((1, 2))

    # repeat location so we can use the vectorized min_mass_containing_location fn
    true_loc_repeated = true_location.repeat(len(pmfs), axis=0)

    # perform the calibration calculation step
    m_vals = min_mass_containing_location(
        pmfs,
        true_loc_repeated,
        xgrid,
        ygrid
    )

    # transform to the bin in [0, 1] to which each value corresponds,
    # essentially iteratively building a histogram with each step
    bins = np.arange(n_calibration_bins + 1) / n_calibration_bins
    bin_idxs = digitize(m_vals, bins)

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
    progressively taking the location bins to which the model assigns the highest
    probability mass.

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
            f'Expected smoothing_method to be either None, a callable, '
            f'or a string in {allowed_method_strs}. Instead, recieved object '
            f'{smoothing_method} of type {type(smoothing_method)}.'
            )

    pmfs = model_output

    if smoothing_method:
        pmfs = smoothing_fn(
            model_output=model_output,
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
