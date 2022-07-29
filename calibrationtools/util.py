import logging

from typing import Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def make_xy_grids(
    arena_dims: Union[Tuple[float, float], np.ndarray],
    resolution: Optional[float] = None,
    shape: Optional[Tuple[float, float]] = None,
    return_center_pts: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a two-tuple or np array storing the x and y dimensions and a desired
    grid resolution, return a tuple of arrays stooring the x and y coordinates
    at every point in the arena spaced apart by `desired_resolution`.

    Optionally, calculate those gridpoints and instead return the CENTER of each
    bin based on the flag `return_center_pts`.

    Note that this function expects shape in the format (n_y_pts, n_x_pts).

    Examples:
    ```
    >>> make_xy_grids((4, 3), shape=(2, 3))
    (
        array([
            [0., 2., 4.],
            [0., 2., 4.]
        ]),
        array([
            [0., 0., 0.],
            [3., 3., 3.]
        ])
    )
    >>> make_xy_grids((4, 3), shape=(1, 2), return_center_pts=True)
    (
        array([
            [0.66666667, 2.        , 3.33333333],
            [0.66666667, 2.        , 3.33333333]
        ]),
        array([
            [0.75, 0.75, 0.75],
            [2.25, 2.25, 2.25]
        ])
    )
    ```

    """
    if resolution is None and shape is None:
        raise ValueError('One of `resolution`, `shape` is required!')

    if not resolution:
        # np.meshgrid returns a shape of (n_y_pts, n_x_pts)
        # but expects (xs, ys) as arguments.
        # reverse the shape so we match this convention.
        pts_per_dim = np.array(shape)[::-1]
    else:
        pts_per_dim = (np.array(arena_dims) / resolution).astype(int)

    def _coord_array(dim_pts):
        """
        Get an array of coordinates along one axis.

        Expects `dim_pts` to be a tuple (dim, n_pts), where `dim`
        is the length of the grid along the current axis, and `n_pts`
        is the desired number of points to be placed along the axis.
        """
        dimension, n_pts = dim_pts
        # if the user requested to return the CENTER of each bin,
        # create one extra gridpoint in each direction
        # then return the average of each successive bin
        if return_center_pts:
            edge_coords = np.linspace(0, dimension, n_pts + 1)
            # add half the successive differences to get avgs
            # between edge_pts[i] and edge_pts[i+1]
            coords = edge_coords[:-1] + (np.diff(edge_coords) / 2)
        else:
            coords = np.linspace(0, dimension, n_pts)
        return coords

    xs, ys = map(_coord_array, zip(arena_dims, pts_per_dim))

    xgrid, ygrid = np.meshgrid(xs, ys)

    return (xgrid, ygrid)


def check_valid_pmfs(pmfs: np.ndarray, prefix_str=None):
    """
    Check that the grids in the provided array are valid pmfs.

    Expect the axes of `pmfs` to be ordered as follows:
        dimension 0: the number of estimates
        dimension 1: the number of gridpoints in the y direction
        dimension 2: the number of gridpoints in the x direction
    Will log a warning if this expected convention appears to be violated.
    """
    if not prefix_str:
        prefix_str = 'Expected `pmfs` to be an array of valid probability ' \
            'mass functions.'
    # make sure that the pmfs have the correct shape. i.e., the arr should have
    # three dimensions of the form (n_estimates, n_x_pts, n_y_pts)
    if pmfs.ndim != 3:
        raise ValueError(
            prefix_str +
            f' Specifically, the pmfs array should be 3-dimensional, with '
            f'dim 0 as the number of estimates per sample, dim 1 as '
            f'the number of gridpoints in the y direction, and dim 2 '
            f'as the number of gridpoints in the x direction. '
            f'Instead, found shape: {pmfs.shape}.'
            )
    # add log msg if the number of estimates (pmfs.shape[0]) is greater than
    # the number of gridpoints in either direction (max(pmfs.shape[1:]))
    if pmfs.shape[0] > max(pmfs.shape[1:]):
        logger.warning(
            'Dimension 0 of the provided pmfs appears to be less than '
            'the number of gridpoints in either the x or y direction. '
            f'Shapes of the pmfs: {pmfs.shape}. '
            'Perhaps the array should be reshaped?'
            )
    # also, check that each grid is a valid pmf
    # first, make sure each sums to approximately 1.
    distrs_sum_to_one = np.isclose(pmfs.sum(axis=(1, 2)), 1)
    if not distrs_sum_to_one.all():
        distrs_not_sum_to_one = (~distrs_sum_to_one).nonzero()
        bad_sums = pmfs.sum(axis=(1, 2))[distrs_not_sum_to_one]
        raise ValueError(
            prefix_str + f' However, the distributions at the following indices '
            f'do not sum to 1: {distrs_not_sum_to_one}. Their sums: {bad_sums}.'
            f'Distributions: {pmfs[distrs_not_sum_to_one]}'
        )
    elems_positive = pmfs >= 0
    if not elems_positive.all():
        negative_idxs = np.argwhere(~elems_positive)
        negative_idxs = [tuple(idxs) for idxs in negative_idxs]
        raise ValueError(
            prefix_str + f' However, elements at the following indices were '
            f'negative: {negative_idxs}'
        )
