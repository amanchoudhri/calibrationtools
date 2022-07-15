import logging
import multiprocessing as mp

from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

def digitize(locations, bin_edges):
    """
    Wrapper for np.digitize where an error is raised if a value far
    outside the given range is encountered. NOTE: We assume that the
    bins are given in increasing order.
    """
    # check that the bins are in increasing order
    diffs = np.diff(bin_edges)
    if (diffs <= 0).any():
        raise ValueError('Expected array `bins` to be in increasing order.')
    # get max distance between bins
    max_dx = diffs.max()
    # define a new bin array where the highest bin
    # is bins[-1] + tol, with our tolerance as
    # 0.01 * max_dx. 
    # say values less than this are in the highest bin.
    # this is to catch floating point errors that push values
    # greater than bin_edges[-1], while still
    # letting us catch extreme values that are way too high.
    tol = 0.01 * max_dx
    extended_bins = np.append(bin_edges, bin_edges[-1] + tol)
    # digitize locations using these new bins, removing the
    # leftmost bin edge to avoid off-by-one errors.
    edges_to_use = extended_bins[1:]
    bin_idxs = np.digitize(locations, edges_to_use)
    # if any value was greater than bins[-1] + max_dx,
    # raise a value error.
    if (bin_idxs == len(edges_to_use)).any():
        positions = bin_idxs == len(edges_to_use)
        values = locations[positions]
        err_display = [
            f'idx: {p} | value: {v}' for (p, v) in zip(positions, values)
            ]
        raise ValueError(
            f'Encountered value far greater than the largest bin edge! '
            f'Largest bin edge: {bin_edges[-1]}; Invalid values and their '
            f'positions: {err_display}'
            )
    # if not, say that the values were sufficiently close to the bin edges
    # and clip them to match the number of bins
    num_bins = len(bin_edges) - 1
    highest_bin_idx = num_bins - 1
    return bin_idxs.clip(0, highest_bin_idx)

def assign_to_bin_2d(locations, xgrid, ygrid):
    """
    Return an array of indices of the 2d bins to which each input in
    `locations` corresponds.
    
    The indices correspond to the "flattened" version of the grid. In essence,
    for a point in bin (i, j), the output is i + (n_x_bins * j), where n_x_bins
    is the number of grid bins in the x direction--essentially the number
    of gridpoints in that direction - 1.
    """
    # locations: (NUM_SAMPLES, 2)
    # xgrid: (n_y_pts, n_x_pts)
    # xgrid: (n_y_pts, n_x_pts)
    x_coords = locations[:, 0]
    y_coords = locations[:, 1]
    # 1d array of numbers representing x coord of each bin
    x_bin_edges = xgrid[0]
    # same for y coord
    y_bin_edges = ygrid[:, 0]
    # assign each coord to a bin in one dimension
    x_idxs = digitize(x_coords, x_bin_edges)
    y_idxs = digitize(y_coords, y_bin_edges)
    # NOTE: we expect model output to have shape (NUM_SAMPLES, n_y_bins, n_x_bins)
    # where n_y_bins = len(y_bin_edges) - 1, and similar for n_x_bins.
    # so when we flatten, the entry at coordinate (i, j) gets mapped to
    # (n_x_bins * j) + i
    n_x_bins = len(x_bin_edges) - 1
    return (n_x_bins * y_idxs) + x_idxs

def _check_grid_shape(pmf_shape, xgrid, ygrid):
    """
    Check that xgrid and ygrid have the correct shapes for the shape
    of the pmf.
    """
    n_y_bins, n_x_bins = pmf_shape
    expected_grid_shape = (n_y_bins + 1, n_x_bins + 1)
    if xgrid.shape != expected_grid_shape or ygrid.shape != expected_grid_shape:
        raise ValueError(
            f'Expected `xgrid` and `ygrid` to have shape {expected_grid_shape}, '
            f'since each pmf has shape {pmf_shape}. '
            f'Instead, encountered `xgrid` shape {xgrid.shape} and '
            f'`ygrid` shape {ygrid.shape}'
            )

def min_mass_containing_location(
    maps: np.ndarray,
    locations: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray
    ):
    # maps: (NUM_SAMPLES, n_y_bins, n_x_bins)
    # locations: (NUM_SAMPLES, 2)
    # coord_bins: (n_y_bins + 1, n_x_bins + 1, 2)  ( output of meshgrid then dstack ) 
    n_y_bins, n_x_bins = maps.shape[1:]
    # first verify that xgrid and ygrid have correct shapes
    _check_grid_shape(maps.shape[1:], xgrid, ygrid)
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
    num_bins = n_y_bins * n_x_bins
    x_idx = np.arange(num_bins)[np.newaxis, :].repeat(num_samples, axis=0)
    condition = x_idx > bin_idxs[:, np.newaxis]
    sorted_maps = np.take_along_axis(flattened_maps, idx_matrix, axis=1)
    s = np.where(condition, 0, sorted_maps).sum(axis=1)
    # clip values to remove any floating point errors
    # where the sum is greater than 1
    return s.clip(0, 1)


def min_mass_containing_location_single(pmf, loc, xgrid, ygrid):
    """
    Find the min mass containing the true location for a single
    pmf. Used to map across args in `min_mass_containing_location_mp`.
    """
    # check to make sure xgrid and ygrid have the correct shape
    _check_grid_shape(pmf.shape, xgrid, ygrid)
    # flatten the pmf
    flattened = pmf.flatten()
    # argsort in descending order
    argsorted = flattened.argsort()[::-1]
    # assign the true location to a coordinate bin
    # reshape loc to a (1, 2) array so the vectorized function
    # assign_to_bin_2d still works
    loc = loc.reshape(1, 2)
    loc_idx = assign_to_bin_2d(loc, xgrid, ygrid)
    # bin number for first interval containing location
    bin_idx = (argsorted == loc_idx).argmax()
    # distribution with values at indices above bin_idxs zeroed out
    # x_idx = [
    # [0, 1, 2, 3, ...],
    # [0, 1, 2, 3, ...]
    # ]
    num_bins = pmf.size
    sorted_maps = flattened[argsorted]
    s = np.where(np.arange(num_bins) > bin_idx, 0, sorted_maps).sum()
    # clip values to remove any floating point errors
    # where the sum is greater than 1
    return s.clip(0, 1)


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


