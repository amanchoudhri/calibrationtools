"""Constants used in testing, factored out for convenience."""

import numpy as np

MIN_DIM, MAX_DIM = 0, 16  # dimensions of the `arena`

DISTR_MEAN = np.array((MAX_DIM - MIN_DIM) / 2).repeat(2)

ARENA_DIMS = (MAX_DIM, MAX_DIM)

IDENTITY_COV = [[1, 0], [0, 1]]
SKEW_COV = [[2, -1], [-1, 2]]  # covariance matrix for skew normal distr

GRID_RESOLUTION = 0.1  # spacing of gridpoints on which pmfs are constructed

coords = np.linspace(MIN_DIM, MAX_DIM, int((MAX_DIM - MIN_DIM) / GRID_RESOLUTION))
xgrid, ygrid = np.meshgrid(coords, coords)