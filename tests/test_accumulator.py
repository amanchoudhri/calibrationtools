"""Tests for the CalibrationAccumulator class."""

from collections import defaultdict
import logging
import pathlib
import unittest

import numpy as np

from calibrationtools.accumulator import CalibrationAccumulator
from calibrationtools.parallel import _calibration_step

from constants import ARENA_DIMS
from util import (
    observed_data, create_model_output
    )

logging.basicConfig(level=logging.DEBUG)


class TestInit(unittest.TestCase):
    """
    Test the __init__ method and its argument validation.
    """
    # =========== TEST FOR MISSING CONFIG ARGUMENTS ===========
    def test_missing_kwargs_predefined_smoothing_method(self):
        """
        Test that the init method throws a ValueError if the necessary
        kwargs weren't included in the smoothing specs for a predefined
        smoothing method.
        """
        smoothing_specs_no_kwargs = {
            'r_ests': {
                'gaussian_mixture': {}
            }
        }
        with self.assertRaises(ValueError):
            CalibrationAccumulator(
                ARENA_DIMS,
                smoothing_specs_no_kwargs
            )
        smoothing_specs_some_kwargs = {
            'r_ests': {
                'gaussian_mixture': {
                    'desired_resolution': 0.05
                }
            }
        }
        with self.assertRaises(ValueError):
            CalibrationAccumulator(
                ARENA_DIMS,
                smoothing_specs_some_kwargs
            )

    def test_missing_n_curves_per_sample_arg(self):
        """
        Test that the init method throws a ValueError if the user
        inputs a custom smoothing method but doesn't pass in
        the param 'n_curves_per_sample' in the corresponding
        smoothing spec.
        """
        def custom_smoothing_fn(x):
            return x

        invalid_smoothing_specs = {
            'rsrp_grids': {
                custom_smoothing_fn: {}
            }
        }
        with self.assertRaises(ValueError):
            CalibrationAccumulator(
                ARENA_DIMS,
                invalid_smoothing_specs
            )

    # =========================================================

    def test_invalid_smoothing_method(self):
        """
        Test that a KeyError is raised if an invalid smoothing
        method string is passed.
        """
        invalid_specs = {
            'output_name': {
                'fake_smoothing_function_1234': {}
            }
        }
        with self.assertRaises(KeyError):
            CalibrationAccumulator(
                ARENA_DIMS,
                invalid_specs
            )

    def test_mass_counts_arr_shape(self):
        """
        Test that the correct amount of space for the mass_counts
        arrays are allocated.
        """
        def custom_smoothing_fn(x):
            return x

        N_CURVES_FOR_CUSTOM_FN = 5

        GAUSSIAN_MIXTURE_STDS = np.linspace(0.1, 5, 10)

        smoothing_specs = {
            'r_ests': {
                'gaussian_mixture': {
                    'std_values': GAUSSIAN_MIXTURE_STDS,
                    'desired_resolution': 0.01
                }
            },
            'rsrp_grids': {
                custom_smoothing_fn: {
                    'n_curves_per_sample': N_CURVES_FOR_CUSTOM_FN
                }
            }
        }

        N_CALIBRATION_BINS = 10
        ca = CalibrationAccumulator(
            ARENA_DIMS,
            smoothing_specs,
            n_calibration_bins=N_CALIBRATION_BINS
            )

        GAUSSIAN_MIXTURE_SHAPE = (
            len(GAUSSIAN_MIXTURE_STDS),
            N_CALIBRATION_BINS
            )

        gm_shape_created = ca.mass_counts['r_ests']['gaussian_mixture'].shape

        self.assertEqual(GAUSSIAN_MIXTURE_SHAPE, gm_shape_created)

        CUSTOM_FN_SHAPE = (N_CURVES_FOR_CUSTOM_FN, N_CALIBRATION_BINS)

        custom_shape_created = ca.mass_counts['rsrp_grids'][
            custom_smoothing_fn].shape

        self.assertEqual(CUSTOM_FN_SHAPE, custom_shape_created)


class TestCalibrationMethods(unittest.TestCase):
    """
    Test that the results from the calibration methods
    agree with the functions defined at the module level.
    """
    @classmethod
    def setUpClass(cls):
        cls.N_SAMPLES = 100
        cls.data = observed_data(cls.N_SAMPLES)
        cls.models = create_model_output()

    def new_accumulator(self, mp=False) -> CalibrationAccumulator:
        smoothing_specs = {
            'uniform': {'no_smoothing': {}},
            'std_normal': {'no_smoothing': {}},
            'skew_normal': {'no_smoothing': {}}
        }
        return CalibrationAccumulator(
            ARENA_DIMS, smoothing_specs, use_multiprocessing=mp
        )

    def is_calculate_step_consistent(self, ca):
        """
        Test that the calculate_step method of CalibrationAccumulator
        is consistent with _calculate_step, which is tested thoroughly
        in test_calibration.py.
        """
        for _, samples in self.data.items():
            ca = self.new_accumulator()
            # initialize arrays of zeros to track mass counts
            cal_step_tracker = defaultdict(dict)
            for model_name, arrs_by_smoothing in ca.mass_counts.items():
                for smoothing_method, arr in arrs_by_smoothing.items():
                    cal_step_tracker[model_name][smoothing_method] = arr.copy()
            # iterate through the samples
            for s in samples:
                # run the accumulator
                ca.calculate_step(self.models, s)
                # call _calibration_step and update cal_step_tracker
                for model_name, model_output in self.models.items():
                    cal_step_output = _calibration_step(
                        model_output[None],
                        s,
                        ARENA_DIMS
                    )
                    mc_array = cal_step_tracker[model_name]['no_smoothing']
                    for curve_idx, bin_idx in enumerate(cal_step_output):
                        mc_array[curve_idx][bin_idx] += 1
                # compare the internal mass_counts arrays
                # with the results from _calibration_step
                for model_name in ca.mass_counts:
                    ca_mc_arr = ca.mass_counts[model_name]['no_smoothing']
                    cal_step_arr = cal_step_tracker[model_name]['no_smoothing']
                    np.testing.assert_array_almost_equal(
                        ca_mc_arr,
                        cal_step_arr
                    )

    def overall_results_sensible(self, ca):
        """
        Make sure the overall results make sense given known
        distributions.
        """
        results = {}
        for sample_name, samples in self.data.items():
            ca = self.new_accumulator()
            for s in samples:
                ca.calculate_step(
                    self.models, s
                )
            results[sample_name] = ca.calculate_curves_and_error()
            # save the results to a local directory for local debugging
            outdir = (
                pathlib.Path(__file__).parent /
                'accumulator_results' /
                f'sample_{sample_name}'
                )
            ca.plot_results(outdir)
        # make sure the correct model for each sample
        # has the lowest error
        for sample_name, results_for_sample in results.items():
            # correct model error
            correct_results = results_for_sample[sample_name]['no_smoothing']
            correct_model_err = correct_results['abs_err']
            for model_name, results_by_smoothing in results_for_sample.items():
                err = results_by_smoothing['no_smoothing']['abs_err']
                logging.debug(
                    f'sample: {sample_name}, model: {model_name}, err: {err}'
                )
                if model_name != sample_name:
                    self.assertLess(
                        correct_model_err,
                        err
                    )

        # make sure the normal distributions are overconfident
        # on the uniform data. this corresponds to a large negative
        # signed error, which should be approximately the same as
        # the negative absolute error.
        def get_errs(model):
            return results['uniform'][model]['no_smoothing']

        spherical_normal = get_errs('std_normal')
        skew_normal = get_errs('skew_normal')
        logging.debug(f'spherical_normal on uniform: {spherical_normal}')
        logging.debug(f'skew_normal on uniform: {skew_normal}')
        self.assertAlmostEqual(
            spherical_normal['abs_err'], -spherical_normal['signed_err']
        )
        self.assertAlmostEqual(
            skew_normal['abs_err'], -skew_normal['signed_err']
        )

    def test_no_mp(self):
        ca = self.new_accumulator()
        self.is_calculate_step_consistent(ca)
        self.overall_results_sensible(ca)

    def test_mp(self):
        """
        Apply the same tests, but with multiprocessing enabled.
        """
        ca = self.new_accumulator(mp=True)
        self.is_calculate_step_consistent(ca)
        self.overall_results_sensible(ca)
