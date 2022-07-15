"""Tests calibration curve functions."""

from collections import defaultdict, namedtuple
import unittest

import numpy as np

from calibrationtools.main import calibration_curve
from calibrationtools.parallel import (
    _calibration_step, calibration_from_steps
    )

from constants import ARENA_DIMS, xgrid, ygrid
from util import (
    observed_data, create_model_output, spherical_normal_varying_std
    )

# useful namedtuple to store our calibration curve results
ResultTuple = namedtuple(
    'ResultTuple',
    ['curves', 'abs_err', 'signed_err']
    )

def cal_curve_results(data, models):
    """
    Run the main.calibration_curve function on every combination
    of the given data and models and return the results.
    """
    results = defaultdict(dict)

    for samples_name, samples in data.items():
        for model_name, model in models.items():
            repeated_model = model[None].repeat(len(samples), axis=0)
            assigned_probs, true_props = calibration_curve(
                repeated_model,
                samples,
                xgrid,
                ygrid
            )
            residuals = true_props - assigned_probs
            abs_err = np.abs(residuals).sum()
            signed_err = residuals.sum()
            results[samples_name][model_name] = ResultTuple(
                true_props, abs_err, signed_err
                )

    return results

def parallel_cal_step_results(data, models):
    """
    Calculate calibration curves using `parallel._calibration_step` and
    `parallel.calibration_from_steps`.
    """
    results = defaultdict(dict)

    N_MODELS_PER_SAMPLE = 1
    N_CALIBRATION_BINS = 10
    
    # initialize our mass counts histogram arrays
    mass_counts = {
        distr_name: {
            model_name: np.zeros(
                (N_MODELS_PER_SAMPLE, N_CALIBRATION_BINS), dtype=int
                )
            for model_name in models
        }
        for distr_name in data
    }
    # apply calibration_step for each sample, for each model/data distribution
    # combination
    for distr_name, samples in data.items():
        for model_name, model in models.items():
            repeated_model = model[None, :].repeat(N_MODELS_PER_SAMPLE, axis=0)
            # prepare the mass counts subarray
            mass_counts_subarr = mass_counts[distr_name][model_name]
            # loop through the samples
            for sample in samples:
                bin_idxs = _calibration_step(
                    repeated_model,
                    sample,
                    ARENA_DIMS,
                    n_calibration_bins=N_CALIBRATION_BINS
                )
                # increment the correct bins for the corresponding histograms
                for model_idx, bin_idx in enumerate(bin_idxs):
                    mass_counts_subarr[model_idx][bin_idx] += 1

    # assemble the results using calibration_from_steps
    results = defaultdict(dict)
    for distr, mass_counts_by_model in mass_counts.items():
        for model_name, cal_step_bulk in mass_counts_by_model.items():
            curves, abs_err, signed_err = calibration_from_steps(cal_step_bulk)
            results[distr][model_name] = ResultTuple(curves, abs_err, signed_err)

    return results


class TestCalibrationCalculation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.N_SAMPLES = 300
        cls.n_calibration_bins = 10
        cls.data = observed_data(cls.N_SAMPLES)
        cls.models = create_model_output()

        cls.serial_results = cal_curve_results(cls.data, cls.models)
        cls.parallel_results = parallel_cal_step_results(cls.data, cls.models)
    
    def test_consistency(self):
        """
        Test to make sure the parallel/online function results are consistent
        with the straightforward implementation.
        """
        for sample_name in self.data:
            for model_name in self.models:
                regular_res = self.serial_results[sample_name][model_name]
                parallel_res = self.parallel_results[sample_name][model_name]

                # convert parallel abs err from np array of shape (1, 1)
                # to a regular float
                self.assertAlmostEqual(
                    regular_res.abs_err,
                    float(parallel_res.abs_err.squeeze())
                    )
                self.assertAlmostEqual(
                    regular_res.signed_err,
                    float(parallel_res.signed_err.squeeze())
                    )
                np.testing.assert_array_almost_equal(
                    regular_res.curves,
                    parallel_res.curves.squeeze()
                )
    
    def _affirm_correct_model_lowest_err(self, results):
        """
        Test to run on a result dictionary (output from cal_curve_results
        or parallel_cal_step_results), affirms that the lowest error
        on every sample comes from the correct model.
        """
        for sample_name, res_by_model in results.items():
            correct_model_err = res_by_model[sample_name].abs_err
            for model_name, r in res_by_model.items():
                if model_name != sample_name:
                    self.assertTrue(r.abs_err > correct_model_err)
    
    def _affirm_normal_on_uniform_overconfident(self, results):
        """
        Check to make sure that the normal models are overconfident
        on uniform samples. This corresponds to a large negative
        signed error.
        """
        uniform_sample_results = results['uniform']
        std_normal = uniform_sample_results['std_normal']
        skew_normal = uniform_sample_results['skew_normal']
        self.assertAlmostEqual(std_normal.abs_err, -std_normal.signed_err)
        self.assertAlmostEqual(skew_normal.abs_err, -skew_normal.signed_err)

    def _test_diff_distrs(self, results):
        """
        Run a suite of evaluations on the given result dictionary
        to make sure it conforms with our intuition on how
        the calibration results should behave.
        """
        self._affirm_correct_model_lowest_err(results)
        self._affirm_normal_on_uniform_overconfident(results)


    def test_cal_curve_diff_distributions(self):
        """
        Test to make sure the results from main.calibration_curve
        align with our intuition when applying various models to
        various observed data.
        """
        self._test_diff_distrs(self.serial_results)
    
    def test_parallel_cal_diff_distributions(self):
        """
        Test to make sure the results from the parallel calibration
        calculations align with our intuition when applying
        various models to various observed data.
        """
        self._test_diff_distrs(self.parallel_results)
    
    def _test_varying_std(self, results):
        """
        Test for a situation where we progressively increase
        the standard deviation of our spherical normal model.
        We expect to see the model start off very overconfident
        (large negative signed error), then move towards calibrated
        as the std approaches the true std, then finally become
        underconfident (positive signed error).
        """
        STD_NORMAL_KEY = 'std_normal'
        # get the standard deviations for each model from
        # their string name 'std_{std}'
        keys = results[STD_NORMAL_KEY].keys()
        stds = {float(k.split('_')[1]):k for k in keys}
        # the signed error should keep increasing as we increase
        # the variance
        signed_errs = np.zeros(len(keys))
        abs_errs = np.zeros((len(keys)))
        for i, std in enumerate(sorted(stds.keys())):
            model_name = stds[std]
            result_tuple = results[STD_NORMAL_KEY][model_name]
            signed_errs[i] = result_tuple.signed_err
            abs_errs[i] = result_tuple.abs_err
        # make sure the signed error increases as a function
        # of std, within a small tolerance for variability
        signed_err_diffs = np.diff(signed_errs)
        DESIRED_PROP = 0.9
        self.assertTrue((signed_err_diffs >= 0).mean() > DESIRED_PROP)
        # next, make sure that the graph of absolute errors
        # is U-shaped -- to do so, check that the mean of the second
        # differences is positive.
        print(abs_errs)
        second_abs_diffs = np.diff(abs_errs, n=2)
        print(second_abs_diffs)
        self.assertTrue(second_abs_diffs.mean() > 0)
    
    def test_varying_std_values(self):
        """
        Test the effect of varying the standard deviation
        of a spherical normal model on the calibration error.
        We expect the model to start off very overconfident,
        then gradually grow more and more underconfident as the
        std increases.
        """
        STD_LOW = 0.1
        STD_HI = 5
        NUM_STEPS = 50
        std_values = np.linspace(STD_LOW, STD_HI, NUM_STEPS)
        models = spherical_normal_varying_std(std_values)
        SMALLER_N_SAMPLES = 100
        smaller_n_data = observed_data(SMALLER_N_SAMPLES)
        serial_results = cal_curve_results(smaller_n_data, models)
        self._test_varying_std(serial_results)
        parallel_results = parallel_cal_step_results(smaller_n_data, models)
        self._test_varying_std(parallel_results)
