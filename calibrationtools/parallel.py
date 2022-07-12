from util import min_mass_containing_location

def calibration_step(
    model_output,
    true_location,
    arena_dims,
    std_values,
    n_calibration_bins: int = 10,
    grid_resolution: float = 0.5,
    ) -> np.ndarray:
    """
    Perform one step of the calibration process on `model_output`,
    vectorized over multiple different smoothing parameters `sigma_values`.

    Args:
        model_output: Array of location estimates (in centimeters) from a model,
            for one audio sample. Expected shape: (n_predictions, 2).
        true_location: Array contianing the true location of the audio sample, in
            centimeters. Expected shape: (1, 2).
        arena_dims: Config parameter storing the dimensions of the arena, in
            centimeters.
        sigma_values: Array of variances, in cm, used to smooth the predictions.
        n_calibration_bins: integer representing how fine the calibration curve should
            be. Default: 10.
        grid_resolution: Desired resolution (in centimeters) to use when creating
            the discrete probability distributions representing the model output.
            Default: 0.5.

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
    """
    # check to make sure that we have a collection of point estimates
    # rather than a full probability map
    if model_output.ndim != 2 or model_output.shape[1] != 2:
        raise TypeError(
            'Expected `model_output` to have shape (n_estimates, 2)' \
            f'but encountered: {model_output.shape}. Maybe the model is' \
            'outputting a probability distribution instead of a collection' \
            'of point estimates?'
       )

    # setup grids on which to smooth the precitions and create the pmfs
    get_coords = lambda dim_cm: np.linspace(0, dim_cm, int(dim_cm / grid_resolution))

    xs, ys = map(get_coords, arena_dims)
    xgrid, ygrid = np.meshgrid(xs, ys)
    coord_grid = np.dstack((xgrid, ygrid))

    # now assemble an array of probability mass functions by smoothing
    # the location estimates with each value of sigma
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

