from typing import Tuple

import numpy as np

from calibrationtools.calculate import (
    min_mass_containing_location,
    min_mass_containing_location_mp
    )

def calibration_curve(
    model_output: np.ndarray,
    true_coords: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    disable_multiprocessing = False,
    n_bins=10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an array of probability maps and true locations, calculate and return
    the values plotted on a calibration curve.
    
    Returns:
    A tuple (bin_edges, observed_props).
    
    bin_edges: An array of shape (n_bins,) containing the edges of each calibration
        bin. These represent the probabilities the model assigned.
    observed_props: An array of shape (n_bins,) containing the true observed proportions
        of times the true location fell into the given interval.
    """
    # if the number of samples is less than around 200,
    # use the vectorized version
    # if not, use the multiprocessing version
    if len(model_output) < 200 or disable_multiprocessing:
        s = min_mass_containing_location(model_output, true_coords, xgrid, ygrid)
    else:
        s = min_mass_containing_location_mp(model_output, true_coords, xgrid, ygrid)

    counts, bin_edges = np.histogram(
        s,
        bins=n_bins,
        range=(0, 1)
    )

    observed_props = counts.cumsum() / counts.sum()

    return bin_edges[1:], observed_props

