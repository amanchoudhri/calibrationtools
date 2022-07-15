"""Test calibration curve functions against known distributions."""

from collections import defaultdict, namedtuple
import os
import unittest

from pathlib import Path

import numpy as np
from calibrationtools.calculate import assign_to_bin_2d, digitize, min_mass_containing_location, min_mass_containing_location_mp, min_mass_containing_location_single

from calibrationtools.parallel import _calibration_step, calibration_from_steps

from constants import ARENA_DIMS
from util import observed_data, create_model_output


class TestDigitize(unittest.TestCase):
    """
    Tests for the digitize function in calculate.py.
    """
    def setUp(self):
        self.bin_edges = np.array([
            0, 0.1, 0.2, 0.3, 0.4, 0.5
        ])
        self.locations = np.array([
            0.05, 0.15, 0.25, 0.35, 0.45
        ])
        self.expected_output = np.array([
            0, 1, 2, 3, 4
        ])

    def test_digitize_success(self):
        """Make sure it's correct."""
        np.testing.assert_array_equal(
            digitize(self.locations, self.bin_edges),
            self.expected_output
        )

    def test_digitize_nonincreasing_bins(self):
        """Make sure it throws an error if bins nonincreasing."""
        # make the bin edges nonincreasing
        self.bin_edges[self.bin_edges == 0.3] = 0.15
        # and assert that digitize raises a value error
        with self.assertRaises(ValueError):
            digitize(self.locations, self.bin_edges)

    def test_digitize_values_too_high(self):
        """Make sure it throws an error if a value too high is found."""
        invalid_locs = np.append(self.locations, 0.55)
        with self.assertRaises(ValueError):
            digitize(invalid_locs, self.bin_edges)

    def test_digitize_tolerance(self):
        """Make sure it correctly handles floating point imprecision."""
        locs = np.append(self.locations, 0.50000000001)
        output = np.append(self.expected_output, 4)
        np.testing.assert_array_equal(
            digitize(locs, self.bin_edges),
            output
        )

class TestAssignToBin2D(unittest.TestCase):
    """
    Tests for calculate.assign_to_bin_2d.
    """
    def setUp(self):
        self.xs = np.array([0, 0.5, 1])
        self.ys = np.array([0, 0.3, 0.6, 0.9])

        self.xgrid, self.ygrid = np.meshgrid(self.xs, self.ys)

        # get one location for each bin, at the exact center
        self.locations = np.array([
            [0.25, 0.15],
            [0.75, 0.15],
            [0.25, 0.45],
            [0.75, 0.45],
            [0.25, 0.75],
            [0.75, 0.75]
        ])

        self.expected_output = np.array([
            0, 1, 2, 3, 4, 5
        ])

    def test_assign_correctly(self):
        np.testing.assert_array_equal(
            assign_to_bin_2d(self.locations, self.xgrid, self.ygrid),
            self.expected_output
        )
    
    def test_tolerance(self):
        """
        Test that this fn correctly incorporates the tolerance
        from calculate.digitize.
        """
        # add on a new point slightly past each edge in
        # each direction
        err = 0.0000001
        new_loc = np.array([self.xs[-1], self.ys[-1]]) + err
        locs = np.append(self.locations, new_loc[None, :], axis=0)
        # and add the corresponding location, which should
        # be the highest bin number
        output = np.append(self.expected_output, self.expected_output[-1])
        np.testing.assert_array_equal(
            assign_to_bin_2d(locs, self.xgrid, self.ygrid),
            output
        )

class TestMinMassContainingLocation(unittest.TestCase):
    """
    Tests for calculate.min_mass_containing_location and
    its associated functions.
    """
    def setUp(self):
        self.xs = np.array([0, 0.5, 1])
        self.ys = np.array([0, 0.3, 0.6, 0.9])

        self.xgrid, self.ygrid = np.meshgrid(self.xs, self.ys)

        # get one location for each bin, at the exact center
        self.locations = np.array([
            [0.25, 0.15],  # bin 0
            [0.75, 0.15],  # bin 1
            [0.25, 0.45],  # bin 2
            [0.75, 0.45],  # bin 3
            [0.25, 0.75],  # bin 4
            [0.75, 0.75]   # bin 5
        ])

        self.pmf = np.array([
            # -- bins --
            # 0     1
            [0.05, 0.3],
            # 2     3
            [0.06, 0.15],
            # 4     5
            [0.4, 0.04]
        ])

        self.pmfs = self.pmf[None, :].repeat(len(self.locations), axis=0)

        # argsorted bins:  4    1     3     2     0     5
        #                 0.4  0.3  0.15  0.06  0.05  0.04

        self.expected_output = np.array([
            0.96,  # bin 0: mass from 4, 1, 3, 2, 0
            0.7,   # bin 1: mass from 4, 1
            0.91,  # bin 2: mass from 4, 1, 3, 2
            0.85,  # bin 3: mass from 4, 1, 3
            0.4,   # bin 4: mass from 4
            1.0,   # bin 5: mass from 4, 1, 3, 2, 0, 5
        ])

    def test_invalid_grid_shape(self):
        """
        Test that the function raises a ValueError if the shape
        of the grid arrays don't match the shape of the pmfs passed.
        Specifically, there should be one more gridpoint than bin
        in each coordinate direction.
        """
        xs = np.append(self.xs, 1.5)
        ys = np.append(self.ys, 1.2)
        bad_xgrid, bad_ygrid = np.meshgrid(xs, ys)

        with self.assertRaises(ValueError):
            min_mass_containing_location(
                self.pmfs,
                self.locations,
                bad_xgrid,
                bad_ygrid
            )
    
    def test_vectorized_success(self):
        """
        Test that min_mass_containing_location returns what
        we would expect.
        """
        np.testing.assert_array_almost_equal(
            min_mass_containing_location(
                self.pmfs,
                self.locations,
                self.xgrid,
                self.ygrid
            ),
            self.expected_output
        )
    def test_single_success(self):
        """
        Test that min_mass_containing_location_single, a helper
        function used in the multiprocessing implementation,
        returns the correct results for each location.
        """
        for loc, pmf, output in zip(
            self.locations, self.pmfs, self.expected_output
        ):
            self.assertAlmostEqual(
                min_mass_containing_location_single(
                    pmf, loc, self.xgrid, self.ygrid
                ),
                output
            )

    def test_mp_success(self):
        """
        Test that the multiprocessing version of the function
        also returns the same results.
        """
        np.testing.assert_array_almost_equal(
            min_mass_containing_location_mp(
                self.pmfs,
                self.locations,
                self.xgrid,
                self.ygrid
            ),
            self.expected_output
        )
