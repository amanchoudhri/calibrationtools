import multiprocessing as mp

from typing import Tuple

import numpy as np

from matplotlib import pyplot as plt


def assign_to_bin_2d(locations, xgrid, ygrid):
    """
    Return an array of indices of the 2d bins to which each input in
    `locations` corresponds.
    
    The indices correspond to the "flattened" version of the grid. In essence,
    for a point in bin (i, j), the output is i * n_y_pts + j, where n_y_pts
    is the number of gridpoints in the y direction.
    """
    # locations: (NUM_SAMPLES, 2)
    # xgrid: (n_y_pts, n_x_pts)
    # xgrid: (n_y_pts, n_x_pts)
    x_coords = locations[:, 0]
    y_coords = locations[:, 1]
    # 1d array of numbers representing x coord of each bin
    x_bins = xgrid[0]
    # same for y coord
    y_bins = ygrid[:, 0]
    # assign each coord to a bin in one dimension
    # note: subtract one to ignore leftmost bin (0)
    x_idxs = np.digitize(x_coords, x_bins) - 1
    y_idxs = np.digitize(y_coords, y_bins) - 1
    # NOTE: we expect model output to have shape (NUM_SAMPLES, n_x_pts, n_y_pts)
    # so when we flatten, the entry at coordinate (i, j) gets mapped to
    # (n_y_pts * i) + j
    n_y_pts = len(y_bins)
    return (n_y_pts * x_idxs) + y_idxs


def min_mass_containing_location(
    maps: np.ndarray,
    locations: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray
    ):
    # maps: (NUM_SAMPLES, n_x_pts, n_y_pts)
    # locations: (NUM_SAMPLES, 2)
    # coord_bins: (n_y_pts, n_x_pts, 2)  ( output of meshgrid then dstack ) 
    # reshape maps to (NUM_SAMPLES, N_BINS)
    num_samples = maps.shape[0]
    flattened_maps = maps.reshape((num_samples, -1))
    idx_matrix = flattened_maps.argsort(axis=1)[:, ::-1]
    # bin number for each location
    loc_idxs = assign_to_bin_2d(locations, xgrid, ygrid)
    # bin number for first interval containing location
    bin_idxs = (idx_matrix == loc_idxs[:, np.newaxis]).argmax(axis=1)
    # distribution with values at indices above bin_idxs zeroed out
    # x_idx = [
    # [0, 1, 2, 3, ...],
    # [0, 1, 2, 3, ...]
    # ]
    num_bins = xgrid.shape[0] * xgrid.shape[1]
    x_idx = np.arange(num_bins)[np.newaxis, :].repeat(num_samples, axis=0)
    condition = x_idx > bin_idxs[:, np.newaxis]
    sorted_maps = np.take_along_axis(flattened_maps, idx_matrix, axis=1)
    s = np.where(condition, 0, sorted_maps).sum(axis=1)
    return s


def min_mass_containing_location_single(pmf, loc, xgrid, ygrid):
    """
    Find the min mass containing the true location for a single
    pmf. Used to map across args in `min_mass_containing_location_mp`.
    """
    # flatten the pmf
    flattened = pmf.flatten()
    # argsort in descending order
    argsorted = flattened.argsort()[::-1]
    # assign the true location to a coordinate bin
    # reshape loc to a (1, 2) array so the vectorized function
    # assign_to_bin_2d still works
    loc_idx = assign_to_bin_2d(loc[np.newaxis, :], xgrid, ygrid)
    # bin number for first interval containing location
    bin_idx = (argsorted == loc_idx).argmax()
    # distribution with values at indices above bin_idxs zeroed out
    # x_idx = [
    # [0, 1, 2, 3, ...],
    # [0, 1, 2, 3, ...]
    # ]
    num_bins = xgrid.shape[0] * xgrid.shape[1]
    sorted_maps = flattened[argsorted]
    s = np.where(np.arange(num_bins) > bin_idx, 0, sorted_maps).sum()
    return s


def min_mass_containing_location_mp(
    maps: np.ndarray,
    locations: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray
):
    """
    Given an array of pmfs and locations, find the minimum mass containing
    the true location for each pmf, using multiprocessing across the samples.
    """
    def arg_iter():
        for pmf, loc in zip(maps, locations):
            yield (pmf, loc, xgrid, ygrid)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        arg_iterator = arg_iter()
        masses = pool.starmap(min_mass_containing_location_single, arg_iterator)
    
    return np.array(masses)


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

