"""Utility functions to generate fake data and model output for testing."""
from typing import Dict, Mapping, Tuple

import numpy as np
import scipy.stats

from constants import (
    IDENTITY_COV, MAX_DIM, MIN_DIM, SKEW_COV, DISTR_MEAN, xgrid, ygrid
)


# group each distr into one convenient function 
def observed_data(n_samples, add_noise=False, noise_std=0.1):
    """
    Automatically generate datasets given the number of samples and
    the desired noise parameters.
    """
    # generate fake data of different distributions
    rng = np.random.default_rng(seed=2022)

    # rescale to be on the square [MIN_DIM, MAX_DIM]^2
    uniform_samples = lambda n_samples: (
        (rng.uniform(size=(n_samples, 2)) * (MAX_DIM - MIN_DIM))
        )

    std_normal_samples = lambda n_samples: rng.multivariate_normal(
        mean=DISTR_MEAN,
        cov=IDENTITY_COV,
        size=n_samples
        )

    skewed_normal_samples = lambda n_samples: rng.multivariate_normal(
        mean=DISTR_MEAN,
        cov=SKEW_COV,
        size=n_samples
        )

    def gen_data(fn):
        """
        Given the args from the parent function, generate samples for
        each distribution of interest (e.g. uniform, spherical Gaussian, etc).
        """
        samples = fn(n_samples)
        if add_noise:
            # transform to the real line using a logit function
            k = MAX_DIM - MIN_DIM
            transformed = samples + MAX_DIM  # make each entry >= 0
            transformed = np.log( transformed / (k - transformed) )
            # add noise
            transformed += rng.normal(scale=noise_std, size=(n_samples, 2))
            # transform back to constrained interval
            samples = k / (1 + np.exp(-transformed))
            # recenter at 0
            samples -= MAX_DIM
        # clip values outside allowed range
        samples = samples.clip(MIN_DIM, MAX_DIM)
        return samples

    return {
        'uniform': gen_data(uniform_samples),
        'std_normal': gen_data(std_normal_samples),
        'skew_normal': gen_data(skewed_normal_samples),
    }

def center_grids_from_edges(xgrid, ygrid):
    """
    Given xgrid and ygrid, return a new array storing the positions
    of the centers of each bin defined by the grids.
    """
    def _get_center_pts(edge_pts):
        """
        Get the coordinates for the center of each bin in one axis.
        """
        # add half the successive differences to get avgs
        # between edge_pts[i] and edge_pts[i+1]
        return edge_pts[:-1] + np.diff(edge_pts)/2

    center_xpts = _get_center_pts(xgrid[0])
    center_ypts = _get_center_pts(ygrid[:, 0])

    return np.meshgrid(center_xpts, center_ypts)

def create_model_output():
    # everything falls within square [MIN_X, MAX_X] x [MIN_Y, MAX_Y]

    center_xgrid, center_ygrid = center_grids_from_edges(xgrid, ygrid)
    uniform_model = np.ones(center_xgrid.shape) * (1 / center_xgrid.size)

    eval_pdf_at = np.dstack((center_xgrid, center_ygrid))
    std_normal_model = scipy.stats.multivariate_normal(
        mean=DISTR_MEAN, cov=IDENTITY_COV
        ).pdf(eval_pdf_at)

    skewed_normal_model = scipy.stats.multivariate_normal(
        mean=DISTR_MEAN, cov=SKEW_COV
        ).pdf(eval_pdf_at)

    # and renormalize
    std_normal_model /= std_normal_model.sum()
    skewed_normal_model /= skewed_normal_model.sum()

    return {
        'uniform': uniform_model,
        'std_normal': std_normal_model,
        'skew_normal': skewed_normal_model
    }

def spherical_normal_varying_std(std_values: np.ndarray):
    """
    Create a models dictionary of spherical normals with
    standard deviations from `std_values`.
    """
    center_xgrid, center_ygrid = center_grids_from_edges(xgrid, ygrid)

    models = {}

    for std in std_values:
        variance = std ** 2
        # create the normal distr with the specified std
        # and evaluate it at the center of every bin
        model = scipy.stats.multivariate_normal(
            mean=DISTR_MEAN, cov=variance
        ).pdf(np.dstack((center_xgrid, center_ygrid)))
        # renormalize so it sums to 1
        model /= model.sum()
        models[f'std_{std:0.3f}'] = model

    return models
