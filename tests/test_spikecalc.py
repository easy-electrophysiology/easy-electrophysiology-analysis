import sys
import numpy as np
import pandas as pd
import os
import pytest
import scipy.stats
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from ..easy_electrophysiology import easy_electrophysiology
MainWindow = easy_electrophysiology.MainWindow
from ephys_data_methods import current_calc, core_analysis_methods
from utils import utils
import utils_for_testing as test_utils
from generate_artificial_data import TestArtificialSkCntData, TestArtificialRiData
import peakutils
from artificial_configs import TestCfgs

class TestSpikecalc:
    """
    Load artificial data from test_generate_artificial_data that contains pre-generated traces with "spikes" or current
    injection. Test analysing within specific rec/bounds.
    """
    @pytest.fixture(autouse=True)
    def test_spkcnt(test):
        return TestArtificialSkCntData()

    @pytest.fixture(autouse=True)
    def test_ir(test):
        return TestArtificialRiData()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Setup Test Methods and Helper Functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------
    def generate_spike_info(self, test_spkcnt, rec_type, time_type, bounds_type):
        [rec_from, rec_to, bounds] = self.generate_analysis_parameters(test_spkcnt, rec_type, time_type, bounds_type)

        spike_info = current_calc.find_spikes_above_record_threshold(test_spkcnt, 25,
                                                                     rec_from, rec_to,
                                                                     bounds,
                                                                     ["start", "stop"])

        return spike_info, rec_from, rec_to, bounds

    def generate_analysis_parameters(self, test_data_object, rec_type, time_type, bounds_type):
        """
        Return parameters to test based on test settings
        """
        rec_from = rec_to = bounds = None
        if time_type == "normalised":
            test_data_object.time_array = test_data_object.norm_time_array
            test_data_object.min_max_time = test_data_object.norm_min_max_time
        elif time_type == "cumulative":
            test_data_object.time_array = test_data_object.cum_time_array
            test_data_object.min_max_time = test_data_object.cum_min_max_time

        if rec_type == "subset_recs":
            rec_from = np.random.randint(1, test_data_object.num_recs - 21, 1)[0]
            rec_to = np.random.randint(rec_from + 20, 75, 1)[0]  # TODO: store rec differences that dependent on eachother (i.e. generate_test_rheobase_data_from_spikeinfo())
        elif rec_type == "all_recs":
            rec_from = 0
            rec_to = 74

        if bounds_type == "all_samples":
            bounds = [[test_data_object.time_array[0][0] for rec in range(test_data_object.num_recs)],  # TODO: only test same bounds across all recs here
                      [test_data_object.time_array[0][-5] for rec in range(test_data_object.num_recs)]]
        elif bounds_type == "bound_samples":
            bounds = test_data_object.generate_bounds()

        return rec_from, rec_to, bounds

    def get_baseline_minus_inj(self, array, time_array, test_object, rec_from, rec_to):
        """
        Call the current_calc function which will return the average of the arrange between 0 and the current injection start time (baseline),
        and the current injection start time and stop time. The baseline will be subtracted from the baseline. Done per-record.
        """
        counted_recs, avg_over_period, baselines, steady_states = current_calc.calculate_baseline_minus_inj(array, time_array,
                                                                                                            [0, test_object.start_time,  # baseline, experimental; need to test sub-bounds
                                                                                                             test_object.start_time, test_object.stop_time],
                                                                                                            ["start", "stop", "start", "stop"],
                                                                                                            rec_from, rec_to,
                                                                                                            test_object.min_max_time
                                                                                                            )
        return counted_recs, avg_over_period, baselines

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Spike Count Tests - NOTE: Auto spike count not tested!
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("rec_type", ["all_recs", "subset_recs"])
    @pytest.mark.parametrize("time_type", ["normalised", "cumulative"])
    @pytest.mark.parametrize("bounds_type", ["all_samples", "bound_samples"])
    def test_count_spikes_norm_time(self, test_spkcnt, rec_type, time_type, bounds_type):
        """
        Takes the artificial dataset TestArtificialSkCntData(), runs through count_spikes and checks output against artificial data.
        Tests when all records are chosen, only a subset, normalised or cumulative time arrays are used, and when time bounds are set.
        """
        spike_info, rec_from, rec_to, bounds = self.generate_spike_info(test_spkcnt, rec_type, time_type, bounds_type)
        spike_count, counted_recs = current_calc.spkcount_and_recnums_from_spikeinfo(spike_info)

        # compare spike times, spike conunts and analysed records
        spike_times_equal, spike_count_equal = test_spkcnt.subtract_results_from_data(test_spkcnt, spike_info, spike_count, rec_from, rec_to, time_type, bounds)
        test_counted_recs = utils.np_empty_nan(test_spkcnt.num_recs)
        test_counted_recs[rec_from:rec_to+1] = np.arange(rec_from + 1,
                                                         rec_to + 2, 1)  # rec analysed is not zero indexed, and account for inclusive upper bound

        assert spike_times_equal, "Failed Spike Info"
        assert np.array_equal(test_counted_recs, counted_recs,
                              equal_nan=True), "Failed Spike Records"
        assert spike_count_equal, "Failed Spike Count"

    @pytest.mark.parametrize("step_size", np.linspace(1, 200, 1))
    @pytest.mark.parametrize("direction", [["increasing", [-1, 1]],
                                           ["decreasing", [1, -1]],
                                           ["repeat", [1/50, 1/50]]])
    def test_round_im_injection_to_user_stepsize(self, step_size, direction):
        current_injection_direction = direction[0]
        start_inj, stop_inj = direction[1][0:2]

        test_im_steps = np.linspace(start_inj * step_size * 50,
                                    stop_inj * step_size * 50,
                                    101)
        noise = np.random.uniform(-1, 1, (101, 1)).squeeze()
        noise = noise * (step_size - step_size * 0.95)  # add 95% noise
        test_im_steps_noise = test_im_steps + noise
        im_steps = current_calc.round_im_injection_to_user_stepsize(test_im_steps_noise, step_size, current_injection_direction)
        assert np.sum(im_steps - test_im_steps) == 0, "Error in test_round_im_injection_to_user_stepsize"

    @pytest.mark.parametrize("rec_type", ["all_recs", "subset_recs"])
    @pytest.mark.parametrize("time_type", ["normalised", "cumulative"])
    @pytest.mark.parametrize("bounds_type", ["all_samples", "bound_samples"])
    def test_get_first_spike_latency(self, test_spkcnt, rec_type, time_type, bounds_type):
        """
        Test first spike latency using the same option permutations as testing count spikes. If whole sample the first spike in the record is used,
        else the first spike after the start boundary.
        """
        spike_info, rec_from, rec_to, bounds = self.generate_spike_info(test_spkcnt, rec_type, time_type, bounds_type)

        if bounds_type == "all_samples":
            test_fs_latency = test_spkcnt.peak_times["normalised"][:, 0]

        elif bounds_type == "bound_samples":
            spike_times = test_utils.vals_within_bounds(test_spkcnt.peak_times["normalised"], bounds[0], bounds[1])
            test_fs_latency = utils.np_empty_nan((75, 1))
            for rec in range(test_spkcnt.num_recs):
                rec_spike_times = spike_times[rec, ~np.isnan(spike_times[rec])]
                test_fs_latency[rec] = rec_spike_times[0] if np.any(rec_spike_times) else 0

        fs_latency = current_calc.get_first_spike_latency(spike_info, test_spkcnt.min_max_time, 0)

        assert np.nansum(fs_latency[rec_from:rec_to] - test_fs_latency[rec_from:rec_to]) < 0.00000000001, "first spike latency error" # TODO: use np.isclose()

    @pytest.mark.parametrize("rec_type", ["all_recs", "subset_recs"])
    @pytest.mark.parametrize("time_type", ["normalised", "cumulative"])
    @pytest.mark.parametrize("bounds_type", ["all_samples", "bound_samples"])
    def test_calculate_mean_isi(self, test_spkcnt, rec_type, time_type, bounds_type):

        spike_info, rec_from, rec_to, bounds = self.generate_spike_info(test_spkcnt, rec_type, time_type, bounds_type)

        if time_type == "cumulative":
            bounds[0] = bounds[0] + test_spkcnt.cum_min_max_time[:, 0]
            bounds[1] = bounds[1] + test_spkcnt.cum_min_max_time[:, 0]

        mean_isi = current_calc.calculate_isi_measures(spike_info,
                                                       "mean_isi_ms")

        spiketimes_within_bounds = test_utils.vals_within_bounds(test_spkcnt.peak_times[time_type], bounds[0], bounds[1])
        test_mean_isi = np.nanmean(np.diff(spiketimes_within_bounds, axis=1), axis=1)
        test_mean_isi[np.isnan(test_mean_isi)] = 0

        assert utils.allclose(mean_isi[rec_from:rec_to], test_mean_isi[rec_from:rec_to], 1e-8), " Mean ISI"

    @pytest.mark.parametrize("rec_type", ["all_recs", "subset_recs"])
    @pytest.mark.parametrize("time_type", ["normalised", "cumulative"])
    @pytest.mark.parametrize("bounds_type", ["all_samples", "bound_samples"])
    @pytest.mark.parametrize("rec_or_exact", ["record", "exact"])
    def test_rheobase(self, test_spkcnt, rec_type, time_type, bounds_type, rec_or_exact):
        """
        """
        spike_info, rec_from, rec_to, bounds = self.generate_spike_info(test_spkcnt, rec_type, time_type, bounds_type)
        _, avg_over_period, baselines = self.get_baseline_minus_inj(test_spkcnt.im_array, test_spkcnt.time_array, test_spkcnt, rec_from, rec_to)

        spike_info, test_rheobase_rec, test_rheobase = test_spkcnt.generate_test_rheobase_data_from_spikeinfo(rec_from, rec_to, spike_info)

        rheobase_rec, rheobase = current_calc.calculate_rheobase(spike_info, test_spkcnt.im_array, test_spkcnt.im_array, rec_or_exact, baselines, rec_from, rec_to)

        if test_rheobase:
            if rec_or_exact == "record":  # current_calc output will be record number if rec_or_exact == record or pA rheobase is rec_or_exact == exact TODO: refactor
                test_rheobase = avg_over_period[test_rheobase_rec]
                rheobase = avg_over_period[rheobase_rec]

            elif rec_or_exact == "exact":
                first_spike_key = list(spike_info[test_rheobase_rec].keys())[0]
                first_spike_idx = spike_info[test_rheobase_rec][first_spike_key][1]
                test_rheobase = test_spkcnt.im_array[test_rheobase_rec][first_spike_idx] - baselines[test_rheobase_rec]

        assert test_rheobase == rheobase, " ".join(["Rheobase", time_type, bounds_type, rec_or_exact])

    @pytest.mark.parametrize("rec_type", ["all_recs", "subset_recs"])
    @pytest.mark.parametrize("time_type", ["normalised", "cumulative"])
    @pytest.mark.parametrize("bounds_type", ["all_samples", "bound_samples"])
    def test_rheobase_exact(self, test_spkcnt, rec_type, time_type, bounds_type):
        """
        Added this test as removed linear increasing im which made artificial data not effective for testing exact rheobase.
        Do here on the fly (TODO: not very neat)
        """
        spike_info, rec_from, rec_to, bounds = self.generate_spike_info(test_spkcnt, rec_type, time_type, bounds_type)
        im_injection = np.linspace(0, 1,  np.shape(test_spkcnt.im_array)[1])
        im_array = np.tile(im_injection, [100, 1]) * np.random.randint(1, 100, 100)[:, None]
        baselines = np.zeros((100, 1))
        __, rheobase = current_calc.calculate_rheobase(spike_info, None, im_array, "exact", baselines, rec_from, rec_to)
        test_rheobase_rec = [idx for idx, rec in enumerate(spike_info) if rec and rec != 0][0]  # find first non-empty and non-zero rec
        first_spike_key = list(spike_info[test_rheobase_rec].keys())[0]
        first_spike_idx = spike_info[test_rheobase_rec][first_spike_key][1]
        test_rheobase = im_array[test_rheobase_rec][first_spike_idx] - baselines[test_rheobase_rec]
        assert rheobase == test_rheobase


# ------spike_sample_idx------------------------------------------------------------------------------------------------------------------------------
# Input Resistance Tests
# ----------------------------------------------------------------------------------------------------------------------------------------------------
    @pytest.mark.parametrize("rec_type", ["all_recs", "subset_recs"])
    @pytest.mark.parametrize("time_type", ["normalised", "cumulative"])
    @pytest.mark.parametrize("array_to_test", ["im", "vm"])
    def test_calculate_baseline_minus_inj(self, test_ir, array_to_test, rec_type, time_type):

        if array_to_test == "im":
            array = test_ir.im_array
        else:
            array = test_ir.vm_array

        rec_from, rec_to, _ = self.generate_analysis_parameters(test_ir, rec_type, time_type, None)
        counted_recs, avg_over_period, _ = self.get_baseline_minus_inj(array, test_ir.time_array, test_ir, rec_from, rec_to)
        actual_inections = np.mean(test_ir.im_array[:, test_ir.start_idx:test_ir.stop_idx], axis=1) - test_ir.im_offset
        assert(avg_over_period == actual_inections, "test_calculate_baseline_minus_inj FAILED")
        test_counted_recs = utils.np_empty_nan(test_ir.num_recs)
        test_counted_recs[rec_from:rec_to+1] = np.arange(rec_from + 1,
                                                         rec_to + 2, 1)  # rec analysed is not zero indexed, and account for inclusive upper bound
        assert np.array_equal(counted_recs, test_counted_recs,
                              equal_nan=True), "Failed counted_recs"

    @pytest.mark.parametrize("rec_type", ["all_recs", "subset_recs"])
    @pytest.mark.parametrize("time_type", ["normalised", "cumulative"])
    def test_find_negative_peak(self, test_ir, rec_type, time_type):
        """
        Calculate sag/hump
        """
        rec_from, rec_to, _ = self.generate_analysis_parameters(test_ir, rec_type, time_type, None)
        baselines = np.tile(test_ir.resting_vm, test_ir.num_recs)
        steady_states = test_ir.current_injection_amplitude + test_ir.resting_vm
        __, sag_hump, __ = current_calc.find_negative_peak(test_ir.vm_array, test_ir.time_array,
                                                           test_ir.start_time, test_ir.stop_time, test_ir.min_max_time,
                                                           rec_from, rec_to,
                                                           test_ir.current_injection_amplitude, baselines, steady_states, "follow_im")
        assert np.array_equal(sag_hump[rec_from:rec_to],
                              test_ir.sag_hump_peaks[rec_from:rec_to, 0]), "Failed test_find_negative_spike"

    def test_calculate_sag_ratio(self):
        sags = np.array([1, 2, 3, 4, 5])
        peak_deflections = np.array([5, 4, 3, 2, 1])
        sag_ratio = current_calc.calculate_sag_ratio(sags, peak_deflections)
        assert np.array_equal(sag_ratio, sags / peak_deflections)

    def test_sanity_check_sag_ratio(self):
        """
        Quick sanity check on a single data to test the new method for calculating sag
        sag = min_peak - steady state voltage response
        sag_ratio = sag / peak deflection

        Here make 1 data array with a a sag peak followed by steady state voltage response
        and check all the key core analysis methods. Between this and the GUI test in
        input resistance this is fully covered.
        """
        class Data:
            def __init__(self):
                self.num_samples = 20000
                self.vm_array = np.atleast_2d(np.ones((self.num_samples)))
                self.time_array = np.atleast_2d(np.arange(self.num_samples))
                self.sag_start_idx = 1000
                self.sag_stop_idx = 5000
                self.steady_state_stop_idx = 15000

                self.offset = -10
                self.sag_peak = -100
                self.steady_state = -50

                self.vm_array *= self.offset
                sag_to_steady_state_slope = np.linspace(self.sag_peak, self.steady_state, (self.sag_stop_idx - self.sag_start_idx))
                self.vm_array[0][self.sag_start_idx:self.sag_stop_idx] = sag_to_steady_state_slope
                self.vm_array[0][self.sag_stop_idx+1:self.steady_state_stop_idx] = self.steady_state
                self.min_max_time = np.array([[0, self.num_samples]])
        data = Data()
        counted_recs, vm_avg, baseline, steady_state = current_calc.calculate_baseline_minus_inj(data.vm_array,
                                                                                                 data.time_array,
                                                                                                 [0,
                                                                                                 data.sag_start_idx - 1,
                                                                                                 data.sag_stop_idx + 1,
                                                                                                 data.steady_state_stop_idx],
                                                                                                 ["start", "stop", "start", "stop"],
                                                                                                 0,
                                                                                                 0,
                                                                                                 data.min_max_time)
        assert vm_avg == data.steady_state - data.offset
        assert baseline[0] == data.offset
        assert steady_state[0] == data.steady_state

        __, sag, peak_deflection = current_calc.find_negative_peak(data.vm_array,
                                                                   data.time_array,
                                                                   0,
                                                                   data.num_samples,
                                                                   data.min_max_time,
                                                                   0, 0,
                                                                   vm_avg,
                                                                   baseline,
                                                                   steady_state,
                                                                   "min")
        assert sag == data.sag_peak - data.steady_state
        assert peak_deflection == data.sag_peak - data.offset

        sag_ratio = current_calc.calculate_sag_ratio(sag,
                                                     peak_deflection)

        assert sag_ratio == sag / peak_deflection

    @pytest.mark.parametrize("cov", [0.25, 0.50, 0.75, 1])
    def test_calculate_input_resistance(self, cov):
        """
        create random normal bivariate distributions, assign as Im and Vm. manually calculate betas calculate_mean_isi
        and compare with output of current_calc.calculate_input_resistance (Im scaled to "pA")
        """
        data = np.random.multivariate_normal([0, 0], [[1, cov], [cov, 1]], (2, 100))[0]
        im_nA = np.reshape(data[:, 0], (100, 1))
        y = np.reshape(data[:, 1], (100, 1))
        X = im_nA  # np.concatenate((im_nA), np.ones((100, 1))), 1)
        slope_test, intercept_test, __, __, __ = scipy.stats.linregress(im_nA[:, 0], y[:, 0])

        # calculate OLS estimator and compare
        I = np.linalg.inv
        beta = I(X.T@X) @ X.T@y

        im_pA = im_nA * 1000
        slope, intercept = current_calc.calculate_input_resistance(im_pA.squeeze(),
                                                                   y.squeeze())
        assert utils.allclose(slope, slope_test, 1e-10)
        assert utils.allclose(intercept, intercept_test, 1e-10)

    def test_calculate_input_resistance_one_pair_case(self):
        V_in_mV = np.array([10])
        I_in_pa = np.array([100])
        R_in_mOHMs = V_in_mV / (I_in_pa / 1000)

        input_resistance, intercept = current_calc.calculate_input_resistance(I_in_pa, V_in_mV)
        assert input_resistance == R_in_mOHMs
        assert intercept is None

    @pytest.mark.parametrize("fs", [8192 * n for n in range(1, 10)])
    def test_get_fft(self, fs):
        """
        Create a array y with random frequencies added (hz_to_add), use fft function to retrieve frequencies and check them
        against the frequencies added.
        """
        y, hz_to_add, fs, __, __ = test_utils.generate_test_frequency_spectra(fs, 500*(fs/8192))
        fft_results = core_analysis_methods.get_fft(y, False, fs, True)
        peaks_idx = peakutils.indexes(fft_results["Y"], thres=0.05, min_dist=5)

        assert np.array_equal(fft_results["freqs"][peaks_idx],
                              hz_to_add), "Failed FFT Test"

    @pytest.mark.parametrize("samples", [8192 * n for n in range(1, 10)])
    @pytest.mark.parametrize("low_or_highpass", ["lowpass", "highpass"])
    @pytest.mark.parametrize("filter_", [["bessel", 0.08], ["butter", 0.2]])
    def test_filter_data(self, samples, filter_, low_or_highpass):
        """
        Similar idea to test_get_fft but filter under before retrieving frequencies.
        First generate array y with random frequencies added (hz_to_add). Then determine a cutoff frequency that is close to halfway between
        two inserted frequencies (that are generated with a minimum distance to account for filter rolloff). Filter y and
        check that the only remaining frequencies are above / below the cutoff frequency.
        """

        y, hz_to_add, samples, dist, n_freqs = test_utils.generate_test_frequency_spectra(samples,
                                                                                          dist=500*(samples/8192))
        cutoff = np.random.randint(dist * 0.45,
                                   dist * 0.55, 1)[0]
        cutoff_idx = np.random.randint(1, n_freqs, 1)[0]
        cutoff_hz = hz_to_add[cutoff_idx] - cutoff  # negative so dont get too close to nyquist
        y = core_analysis_methods.filter_data(y, samples, filter_[0], 8, cutoff_hz, low_or_highpass, 0)

        fft_results = core_analysis_methods.get_fft(y, False, samples, True)
        peaks_idx = peakutils.indexes(fft_results["Y"], thres=filter_[1], min_dist=5)

        if low_or_highpass == "lowpass":
            remaining_freqs = hz_to_add[0:cutoff_idx]
        elif low_or_highpass == "highpass":
            remaining_freqs = hz_to_add[cutoff_idx:]

        assert np.array_equal(fft_results["freqs"][peaks_idx], remaining_freqs), "Failed test_filter"
