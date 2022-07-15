"""Test calibration curve functions against known distributions."""

from collections import defaultdict, namedtuple
import os
import unittest

from pathlib import Path

import numpy as np
from calibrationtools.calculate import digitize

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

