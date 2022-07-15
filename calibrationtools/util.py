import logging

from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

def make_xy_grids(
    arena_dims: Union[Tuple[float, float], np.ndarray],
    resolution: Optional[float] = None,
    shape: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a two-tuple or np array storing the x and y dimensions and a desired
    grid resolution, return a tuple of arrays stooring the x and y coordinates
    at every point in the arena spaced apart by `desired_resolution`. 
    """
    if resolution is None and shape is None:
        raise ValueError('One of `resolution`, `shape` is required!')

    if not resolution:
        pts_per_dim = np.array(shape)
    else:
        pts_per_dim = (np.array(arena_dims) / resolution).astype(int)

    get_coord_arrays = lambda dim_pts: np.linspace(0, dim_pts[0], dim_pts[1])
    xs, ys = map(get_coord_arrays, zip(arena_dims, pts_per_dim))
    xgrid, ygrid = np.meshgrid(xs, ys)
    return (xgrid, ygrid)

def check_valid_pmfs(pmfs: np.ndarray, prefix_str=None):
    """
    Check that the grids in the provided array are valid pmfs.

    Expect the axes of `pmfs` to be ordered as follows:
        dimension 0: the number of estimates
        dimension 1: the number of gridpoints in the x direction
        dimension 2: the number of gridpoints in the y direction
    Will log a warning if this expected convention appears to be violated.
    """
    if not prefix_str:
        prefix_str = 'Expected pmfs to be an array of valid probability mass functions.'
    # make sure that the pmfs have the correct shape. i.e., the arr should have
    # three dimensions of the form (n_estimates, n_x_pts, n_y_pts)
    if pmfs.ndim != 3:
        raise ValueError(prefix_str + f' Instead, found shape: {pmfs.shape}.')
    # add log msg if the number of estimates (pmfs.shape[0]) is greater than
    # the number of gridpoints in either direction (max(pmfs.shape[1:]))
    if pmfs.shape[0] > max(pmfs.shape[1:]):
        logger.warning(
            'Dimension 0 of the provided pmfs appears to be less than ' \
            'the number of gridpoints in either the x or y direction. ' \
            'Perhaps the provided array should be reshaped?'
            )
    # also, check that each grid is a valid pmf
    # first, make sure each sums to approximately 1.
    distrs_sum_to_one = np.isclose(pmfs.sum(axis=(1, 2)), 1)
    if not distrs_sum_to_one.all():
        distrs_not_sum_to_one = (~distrs_sum_to_one).nonzero()
        raise ValueError(
            prefix_str + f'However, the following distributions do not sum ' \
            f'to 1: {distrs_not_sum_to_one}'
        )
    elems_positive = pmfs >= 0 
    if not elems_positive.all():
        negative_idxs = np.argwhere(~elems_positive)
        negative_idxs = [tuple(idxs) for idxs in negative_idxs]
        raise ValueError(
            prefix_str + f'However, elements at the following indices were ' \
            f'negative: {negative_idxs}'
        )