import sys, os
import pytest
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtTest import QTest

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology.mainwindow.mainwindow import (
    MainWindow,
)
import numpy as np
import pandas as pd
from setup_test_suite import get_test_base_dir, GuiTestSetup
from ephys_data_methods import burst_analysis_methods
from utils import utils
import copy
import re
import keyboard

SPEED = "fast"


class TestBurstAnalysis:
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Fixtures / Data
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.fixture
    def test_data(test):
        """
        There are 5 datasets to test against, they were generated in MATLAB. Read the data into dictionaries, there is:

        hist: dictionary of bins and probabilities from logISI histograms
        max_isi: max_isis calculated from the histograms
        burst_results: results
        """
        base_dir = get_test_base_dir()

        spike_time = pd.read_csv(os.path.join(base_dir, "data_files/example_burst_trains.csv"))
        spike_binary = pd.read_csv(os.path.join(base_dir, "data_files/example_burst_trains_binary.csv"))

        isi_hist_data = pd.read_csv(os.path.join(base_dir, "data_files/example_burst_calcISILogHist_results.csv"))
        hist = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {}, "6": {}}
        hist["1"]["bins"] = isi_hist_data.iloc[:, 0]  # TODO
        hist["1"]["prob"] = isi_hist_data.iloc[:, 1]
        hist["2"]["bins"] = isi_hist_data.iloc[:, 2]
        hist["2"]["prob"] = isi_hist_data.iloc[:, 3]
        hist["3"]["bins"] = isi_hist_data.iloc[:, 4]
        hist["3"]["prob"] = isi_hist_data.iloc[:, 5]
        hist["4"]["bins"] = isi_hist_data.iloc[:, 6]
        hist["4"]["prob"] = isi_hist_data.iloc[:, 7]
        hist["5"]["bins"] = isi_hist_data.iloc[:, 8]
        hist["5"]["prob"] = isi_hist_data.iloc[:, 9]
        hist["6"]["bins"] = isi_hist_data.iloc[:, 10]
        hist["6"]["prob"] = isi_hist_data.iloc[:, 11]

        max_isi_data = pd.read_csv(
            os.path.join(base_dir, "data_files/example_burst_calcISImax.csv"),
            header=None,
        )
        max_isi = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {}, "6": {}}
        max_isi["1"]["max_isi"] = max_isi_data.iloc[0, 0]
        max_isi["1"]["peaks"] = max_isi_data.iloc[:, 1].dropna()
        max_isi["2"]["max_isi"] = max_isi_data.iloc[0, 2]
        max_isi["2"]["peaks"] = max_isi_data.iloc[:, 3].dropna()
        max_isi["3"]["max_isi"] = max_isi_data.iloc[0, 4]
        max_isi["3"]["peaks"] = max_isi_data.iloc[:, 5].dropna()
        max_isi["4"]["max_isi"] = max_isi_data.iloc[0, 6]
        max_isi["4"]["peaks"] = max_isi_data.iloc[:, 7].dropna()
        max_isi["5"]["max_isi"] = max_isi_data.iloc[0, 8]
        max_isi["5"]["peaks"] = max_isi_data.iloc[:, 9].dropna()
        max_isi["6"]["max_isi"] = max_isi_data.iloc[0, 10]
        max_isi["6"]["peaks"] = max_isi_data.iloc[:, 11].dropna()

        burst_results = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {}, "6": {}}
        for i in range(1, 7):
            i = str(i)
            burst_results[i] = pd.read_csv(os.path.join(base_dir, "data_files/burst_results_" + i + ".csv"))

        return {
            "spike_time": spike_time,
            "spike_binary": spike_binary,
            "hist": hist,
            "max_isi": max_isi,
            "burst_results": burst_results,
        }

    def get_individual_test_data(self, test_data, test_idx):
        """
        Index out the test data (1 of 6 possible) and return all params for this data in a dict.
        """
        test_num = str(test_idx + 1)
        spikes = test_data["spike_time"].iloc[:, test_idx].dropna().to_numpy()  # TODO: convenience function
        bins = test_data["hist"][test_num]["bins"].dropna().to_numpy()
        prob = test_data["hist"][test_num]["prob"].dropna().to_numpy()
        max_isi = test_data["max_isi"][test_num]["max_isi"]
        peaks = test_data["max_isi"][test_num]["peaks"].to_numpy()
        bursts = test_data["burst_results"][test_num]

        spikes_bin = test_data["spike_binary"].iloc[:, test_idx + 1].to_numpy()
        spikes_bin_time = test_data["spike_binary"].iloc[:, 0].to_numpy()

        return test_num, {
            "spikes": spikes,
            "bins": bins,
            "prob": prob,
            "max_isi": max_isi,
            "peaks": peaks,
            "bursts": bursts,
            "binary": {"data": spikes_bin, "time": spikes_bin_time},
        }

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Test logISI histogram and maxISI
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("test_idx", [0, 1, 2, 3, 4, 5])
    def test_log_isi(self, test_data, test_idx):
        """
        We don't test the bins because they are calculated differently, so just import them and force EE to analyze with those bins
        Test the probabilities are calculated correctly when the bins are held the same.
        """
        test_num, test = self.get_individual_test_data(test_data, test_idx)

        burst_cfgs = {
            "min_void_parameter": 0.7,
            "bins_per_decade": 10,
            "histogram_smoothing": False,
            "intraburst_peak_cutoff_ms": 100,
            "min_samples_between_peaks": 3,
            "bin_override": np.log10(test["bins"]),
            "bin_edge_method": "left_edge",
        }

        isi_ms = np.diff(test["spikes"]) * 1000

        bins, count_prob = burst_analysis_methods.calculate_logisi_histogram(isi_ms, burst_cfgs)

        assert utils.allclose(np.log10(test["bins"][:-1]), bins)  # test is left edge,
        assert utils.allclose(test["prob"][:-1], count_prob)

    def test_find_log_isi_peaks(self):
        """
        Test finding the intra-burst and other peaks from the logISI histogram.

        There is no direct equivilent findpeaks function between matlab and python, so we cannot test directly. Very annoying
        (also tried peakutils, still not the same in how it handles minimum distance). Instead just make up dataset
        and check its doing what is expected.
        """
        burst_cfgs = {
            "min_void_parameter": 0.7,
            "bins_per_decade": 10,
            "histogram_smoothing": False,
            "intraburst_peak_cutoff_ms": 100,
            "min_samples_between_peaks": 3,
        }

        test_bins = np.log10(
            [
                0,
                50,
                100,
                150,
                200,
                250,
                300,
                350,
                400,
                450,
                500,
                550,
                600,
                650,
                700,
                750,
            ]
        )
        test_prob = [
            0,
            0.25,
            0.15,
            0,
            0.15,
            0.26,
            0.12,
            0.12,
            0.27,
            0.125,
            0.05,
            0,
            0.25,
            0.1,
            0,
            0,
        ]  # peaks are 1, 5, 8, 12. 5 and 7 are 3 samples apart

        # but todo: they feed in the normal bins we use the log bins.... but log bins is better!
        (
            intra_burst_peak_idx,
            larger_isi_peak_indexes,
        ) = burst_analysis_methods.get_log_isi_hist_peaks(
            burst_cfgs, test_bins, test_prob
        )  # TODO: rename outputs!

        assert intra_burst_peak_idx == 1
        assert np.array_equal(larger_isi_peak_indexes, np.array([5, 8, 12]))

        burst_cfgs["min_samples_between_peaks"] = 4
        (
            intra_burst_peak_idx,
            larger_isi_peak_indexes,
        ) = burst_analysis_methods.get_log_isi_hist_peaks(
            burst_cfgs, test_bins, test_prob
        )  # TODO: rename outputs!

        assert intra_burst_peak_idx == 1
        assert np.array_equal(larger_isi_peak_indexes, np.array([8, 12]))

        burst_cfgs["intraburst_peak_cutoff_ms"] = 350
        (
            intra_burst_peak_idx,
            larger_isi_peak_indexes,
        ) = burst_analysis_methods.get_log_isi_hist_peaks(
            burst_cfgs, test_bins, test_prob
        )  # TODO: rename outputs!

        assert intra_burst_peak_idx == 5
        assert np.array_equal(larger_isi_peak_indexes, np.array([8, 12]))

    def test_void_parameters_raw(self):
        """
        Test the void parameter function by creating a fake histogram.
        later peaks (1, 2, 2) will be compared to the first peak (1).
        The minimums are 0.5 between the first peak and 2nd, 3rd peak and
        0 between 4th peak. The void parameters should thus be
        0.5, 1 - (0.5/sqrt(1*2)) and 1 - 0.
        """
        #          p            p            p         p
        prob = [0, 1, 0.5, 0.5, 1, 0.5, 0.5, 2, 0, 0, 2]

        intra_burst_peak_idx = 1
        larger_isi_peak_indexes = np.array([4, 7, 10])

        (
            minimum_indexes,
            void_parameters,
        ) = burst_analysis_methods.get_minimums_with_valid_void_parameter(
            prob, intra_burst_peak_idx, larger_isi_peak_indexes, min_void_parameter=0
        )

        assert minimum_indexes == [2, 2, 8]
        assert void_parameters == [
            0.5,
            1 - (0.5 / (np.sqrt(2))),
            1,
        ]  # void parameter = 1 - (min / sqrt(pk1 * pk2))

    @pytest.mark.parametrize("test_idx", [0, 1, 2, 3, 4, 5])
    def test_max_isi(self, test_data, test_idx):
        """
        Tst the maxISI is calculated correctly from the logISI histogram

        note different due to binning, they use right bin we use centre
        """
        test_num, test = self.get_individual_test_data(test_data, test_idx=test_idx)

        burst_cfgs = {
            "min_void_parameter": 0.7,  # TODO: convenience function!
            "bins_per_decade": 10,
            "histogram_smoothing": False,
            "intraburst_peak_cutoff_ms": 100,
            "min_samples_between_peaks": 3,
            "bin_override": np.log10(test["bins"]),
            "bin_edge_method": "left_edge",
        }  # key difference

        isi = np.diff(test["spikes"]) * 1000
        (
            max_long_isi,
            info_for_plotting,
        ) = burst_analysis_methods.calculate_log_threshold(isi, burst_cfgs)

        if np.isnan(test["max_isi"]):
            assert max_long_isi == "no_valid_minimum_error"
        else:
            assert utils.allclose(max_long_isi, test["max_isi"], 1e-06)

    def test_manually_log_isi_short(self):
        """
        Test that the algorithm behaves as expected when max long ISI <= max short ISI (should default to max long ISI).
        """
        burst_cfgs = {
            "detection_method": "log_isi",
            "min_spikes_per_burst": 2,
            "max_short_isi_ms": 100,
            "max_long_isi_ms": 100,
        }  # key difference

        peak_times_ms = (
            np.array([1, 2.01, 2.02, 2.03, 5, 6, 7.04, 7.05, 9, 10]) * 1000
        )  # spikes in burst separated by 10 ms
        test_start_idx = np.array([1, 6])
        test_stop_idx = np.array([3, 7])

        # both thresholds are 100 ms, so all detected
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert np.array_equal(start_idx, test_start_idx)
        assert np.array_equal(stop_idx, test_stop_idx)

        # long threshold is < short threshold, so long threshold is used. It is still > 10 ms so all detected
        burst_cfgs[
            "max_long_isi_ms"
        ] = 11  # if use 10 it works properly, but due to numerical issues not all isi are < 10

        start_idx, __ = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)
        assert np.array_equal(start_idx, test_start_idx)

        # now long threshold is less than burst isi, so no longer used
        burst_cfgs["max_long_isi_ms"] = 9

        start_idx, __ = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)
        assert start_idx is False

        # finally check min spikes per burst, when increased, ensures the 2-spike burst is not detected
        burst_cfgs["min_spikes_per_burst"] = 3
        burst_cfgs["max_long_isi_ms"] = 11
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)
        assert start_idx == 1
        assert stop_idx == 3

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # Test Max Interval
    # ------------------------------------------------------------------------------------------------------------------------------------------

    def test_max_interval(self, test_data):
        """
        log ISI and max interval should be identical when all vals are set to 100
        """
        burst_cfgs = {
            "detection_method": "log_isi",
            "min_spikes_per_burst": 2,
            "max_short_isi_ms": 100,
            "max_long_isi_ms": 100,  # key difference
            "interval_params_ms": {
                "max_interval": 100,
                "max_end_interval": 100,
                "min_burst_interval": 100,
                "min_burst_duration": 0.0001,
                "min_spikes_per_burst": 2,  # TODO: change name
            },
        }

        test_num, test = self.get_individual_test_data(test_data, test_idx=4)
        peak_times_ms = test["spikes"] * 1000

        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        burst_cfgs["detection_method"] = "interval"

        start_idx_1, stop_idx_1 = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        np.array_equal(start_idx, start_idx_1)

    def test_max_interval_manual(self):
        """
        Test max interval by creating a set of spike times to test all 5 criteria.
        """

        #        0  1  2    3    4    5    6    7  8  9    10   11   12   13   14 15 16
        peak_times_ms = (
            np.array([1, 2, 2.5, 2.6, 2.7, 2.8, 3.5, 4, 5, 5.2, 5.4, 5.6, 5.8, 6.2, 7, 8, 9]) * 1000
        )  # burst lengths (2.5 - 2.8), (5- 6.2)
        burst_cfgs = {
            "detection_method": "interval",
            "interval_params_ms": {
                "max_interval": 101,
                "max_end_interval": 101,
                "min_burst_interval": 100,
                "min_burst_duration": 0.0001,
                "min_spikes_per_burst": 2,
            },
        }

        # Check that only the first burst (sep by 100 ms) is detected)
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert start_idx == 2
        assert stop_idx == 5

        # Set to 99 ms and check nothing is detected
        burst_cfgs["interval_params_ms"]["max_interval"] = 99
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert start_idx is False
        assert stop_idx is False

        # set to 201 ms and check all bursts detected
        burst_cfgs["interval_params_ms"]["max_interval"] = 201
        burst_cfgs["interval_params_ms"]["max_end_interval"] = 201

        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert np.array_equal(start_idx, np.array([2, 8]))
        assert np.array_equal(stop_idx, np.array([5, 12]))

        # Extend max end interval to the last spike of the second burst
        burst_cfgs["interval_params_ms"]["max_end_interval"] = 401
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert np.array_equal(stop_idx, np.array([5, 13]))

        # Reset option and check that min_spikes_per_burst excludes the first burst (4 events)
        burst_cfgs["interval_params_ms"]["max_end_interval"] = 201
        burst_cfgs["interval_params_ms"]["min_spikes_per_burst"] = 5
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert start_idx == 8
        assert stop_idx == 12

        # reset num events and check burst duration set to 301 ms still excludes the first burst
        burst_cfgs["interval_params_ms"]["min_spikes_per_burst"] = 3
        burst_cfgs["interval_params_ms"]["min_burst_duration"] = 301
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert start_idx == 8
        assert stop_idx == 12

        # check all bursts are excluded if min burst duration larger than the largest burst duration
        burst_cfgs["interval_params_ms"]["min_burst_duration"] = 1200
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert start_idx is False

        # reset options, check min_burst_interval to the burst interval combines bothbursts
        burst_cfgs["interval_params_ms"]["min_burst_duration"] = 0.0001
        burst_cfgs["interval_params_ms"]["min_burst_interval"] = 2200
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert start_idx == 2
        assert stop_idx == 12

        # final reset to 1 ms under the burst interval, to check the bursts are separated again.
        burst_cfgs["interval_params_ms"]["min_burst_interval"] = 2199
        start_idx, stop_idx = burst_analysis_methods.detect_burst_start_stop_index(peak_times_ms, burst_cfgs)

        assert np.array_equal(start_idx, np.array([2, 8]))
        assert np.array_equal(stop_idx, np.array([5, 12]))

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Test Burst Results
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("test_idx", [1, 2, 3, 4, 5])
    def test_burst_results(self, test_data, test_idx):
        """
        Test that exactly the same bursts are detected between test data and EE implementation. Parameters are checked in next function.
        """
        test_num, test = self.get_individual_test_data(test_data, test_idx)

        peak_times_ms = test["spikes"] * 1000
        (
            burst_start_idx,
            burst_stop_idx,
        ) = burst_analysis_methods.log_isi_burst_detection_long(
            peak_times_ms,
            isi=np.diff(peak_times_ms),
            min_spikes_per_burst=3,
            max_short_isi=100,
            max_long_isi=test["max_isi"],
        )

        assert np.array_equal(test["spikes"][burst_start_idx], test["bursts"].iloc[:, 0])
        assert np.array_equal(test["spikes"][burst_stop_idx], test["bursts"].iloc[:, 1], 1e-10)

        # Quick check the main entry method gives the same results
        (
            burst_start_idx,
            burst_stop_idx,
        ) = burst_analysis_methods.detect_burst_start_stop_index(
            peak_times_ms,
            burst_cfgs=dict(
                detection_method="log_isi",
                min_spikes_per_burst=3,
                max_short_isi_ms=100,
                max_long_isi_ms=test["max_isi"],
            ),
        )

        assert np.array_equal(test["spikes"][burst_start_idx], test["bursts"].iloc[:, 0])
        assert np.array_equal(test["spikes"][burst_stop_idx], test["bursts"].iloc[:, 1], 1e-10)

    @pytest.mark.parametrize("test_idx", [1, 2, 3, 4, 5])
    def test_burst_parameters(self, test_data, test_idx):
        """
        Check all parameters are calculated correctly against test data

        Test Results:
            burstTrain:   matrix (size number_of_bursts x 6) containing the following data:
            1st col: time instant in which the burst begins (samples)  JZ EDIT: this is now time
            2nd col: time instant in which the burst ends (samples)
            3rd col: number of spikes in each burst
            4th col: duration (seconds)
            5th col: inter-burst interval (between the end of the burst
                and the begin of the following one) (seconds)
            6th col: burst period (time interval between the begin of the burst
                and the begin of the following one) (seconds)
        """
        test_num, test = self.get_individual_test_data(test_data, test_idx)

        peak_times_ms = test["spikes"] * 1000
        (
            burst_start_idx,
            burst_stop_idx,
        ) = burst_analysis_methods.log_isi_burst_detection_long(
            peak_times_ms,
            isi=np.diff(peak_times_ms),
            min_spikes_per_burst=3,
            max_short_isi=100,
            max_long_isi=test["max_isi"],
        )

        # Calculate burst parameters
        (
            burst_lengths_ms,
            num_spikes_in_burst,
            inter_burst_intervals,
            rec_fraction_of_spikes_in_burst,
            all_burst_idx,
            all_burst_peak_times,
            all_intra_burst_ISI,
        ) = burst_analysis_methods.calculate_burst_parameters(test["spikes"], burst_start_idx, burst_stop_idx)

        self.check_burst_parameters(
            peak_times_ms,
            test,
            burst_start_idx,
            burst_stop_idx,
            burst_lengths_ms,
            num_spikes_in_burst,
            inter_burst_intervals,
            rec_fraction_of_spikes_in_burst,
            all_burst_idx,
            all_burst_peak_times,
            all_intra_burst_ISI,
        )

    def test_load_gui(self, test_data):
        """
        Load the test data into the GUI, analyse the bursts and check results on the dialog.burst_results
        dict are all correct (for all 6 test datasets).
        """
        tgui = self.setup_skinetics_tgui()

        for test_idx in range(1, 6):
            test_num, test = self.get_individual_test_data(test_data, test_idx=test_idx)
            dialog = self.load_and_analyse_test_spikes(tgui, test)
            self.check_dialog_burst_results(dialog, test)

        tgui.mw.dialogs["burst_analysis_dialog"].accept()
        tgui.shutdown()

    def test_burst_parameters_manual(self):
        """
        Sanity-check dialog.burst_results() and the results table by making a manual dataset with known burst features.

        starts idx [2, 12, 36]
        stops idx [4,  26, 48]
        times = [0.2, 0.4], [1.2, 1.4, 1.7, 1.9, 2.2, 2.6], [3.6, 3.8, 4.2, 4.8]
                 0    1      2    3    4    5    6    7      8    9    10   11
        """
        # fmt: off
        tgui = self.setup_skinetics_tgui()

        test = {
            "binary": {
                "time": np.array(
                    [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
                     1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                     2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
                     3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,
                     4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9,
                     5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6,]
                ),
                "data": np.array(
                    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                     1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0, 1, 0,]
                ),
            },
            "max_isi": 601,
        }

        dialog = self.load_and_analyse_test_spikes(
            tgui,
            test,
            thr_search_region=100,
            extend_fahp_end=True,
            max_isi_short=601,
            min_num_spikes=2,
        )

        test_intra_burst_isi = [
            0.2 * 1000,
            np.mean([0.2, 0.3, 0.2, 0.3, 0.4]) * 1000,
            np.mean([0.2, 0.4, 0.6]) * 1000,
        ]
        test_all_peak_times = [0.2, 0.4, 1.2, 1.4, 1.7, 1.9, 2.2, 2.6, 3.6, 3.8, 4.2, 4.8, 5.9,]
        # fmt: off

        # test the dialog burst results against the known test data

        br = dialog.burst_results
        assert np.array_equal(br["start_idx"][0], [0, 2, 8])
        assert np.array_equal(br["stop_idx"][0], [1, 7, 11])
        assert utils.allclose(br["lengths_ms"][0], np.array([0.2, 1.4, 1.2]) * 1000)
        assert np.array_equal(br["num_spikes_per_burst"][0], [2, 6, 4])
        assert utils.allclose(br["inter_burst_intervals"][0], np.array([0.8, 1]) * 1000)
        assert br["rec_fraction_of_spikes_in_burst"][0] == 12 / 13
        assert br["total_fraction_of_spikes_in_burst"] == 12 / 13

        # all close
        assert utils.allclose(br["burst_peak_times"][0], [0.2, 0.4, 1.2, 1.4, 1.7, 1.9, 2.2, 2.6, 3.6, 3.8, 4.2, 4.8])
        assert utils.allclose(br["intra_burst_isi"][0], test_intra_burst_isi)

        assert utils.allclose(br["all_peak_times"][0], test_all_peak_times)
        assert utils.allclose(br["peak_time_burst_nums"][0], [1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3])
        assert br["total_num_peaks"] == 13

        assert np.array_equal(br["averages"]["intra_burst_isi"], np.mean(br["intra_burst_isi"][0]))
        assert np.array_equal(br["averages"]["inter_burst_intervals"], np.mean(br["inter_burst_intervals"][0]))
        assert np.array_equal(br["averages"]["num_spikes_per_burst"], np.mean(br["num_spikes_per_burst"][0]))
        assert np.array_equal(br["averages"]["length_ms"], np.mean(br["lengths_ms"][0]))
        assert br["averages"]["total_num_bursts"] == 3

        self.check_list_widget_against_burst_results(br, dialog)

        # check results are shown correctly on results table
        tgui.left_mouse_click(dialog.dia.show_results_table_button)

        table_data = tgui.get_entire_qtable(dialog.table_dialog.table, na_as_inf=True, start_row=1)

        assert np.array_equal(self.dropna(table_data[:, 0]), np.ones(13))  # record
        assert utils.allclose(self.dropna(table_data[:, 1]), test_all_peak_times)  # peak times

        # burst nums (np.inf == 'N/A')
        assert np.array_equal(self.dropna(table_data[:, 2]), [1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, np.inf])
        assert np.array_equal(self.dropna(table_data[:, 3]), [1, 2, 3])  # burst num
        assert np.array_equal(self.dropna(table_data[:, 4]), [2, 6, 4])  # num spikes in burst

        # burst length
        assert utils.allclose(self.dropna(table_data[:, 5]), np.array([0.2, 1.4, 1.2]) * 1000)

        # avg intra-burst interval
        assert utils.allclose(self.dropna(table_data[:, 6]), test_intra_burst_isi)

        assert self.dropna(table_data[:, 7]) == 3  # num bursts
        assert self.dropna(table_data[:, 8]) == 12 / 13  # fraction spikes in burst
        assert self.dropna(table_data[:, 9]) == np.mean([2, 6, 4])  # avg num spikes in burst
        assert utils.allclose(self.dropna(table_data[:, 10]), np.mean([0.2, 1.4, 1.2]) * 1000)  #  avg burst length
        assert utils.allclose(self.dropna(table_data[:, 11]), np.mean(test_intra_burst_isi))  # Avg. Intra-burst interval (ms)
        assert utils.allclose(self.dropna(table_data[:, 12]), np.mean([0.8, 1]) * 1000)  # Avg. Inter-burst Interval (ms)

        tgui.mw.dialogs["burst_analysis_dialog"].accept()
        tgui.shutdown()

    @pytest.mark.parametrize("norm_time", [True, False])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("skinetics_or_events", ["events", "skinetics"])
    def test_across_records(self, test_data, analyse_specific_recs, norm_time, skinetics_or_events):
        """
        Test burst analysis when restricting analysis to certain records. Very long test, could be broken up
        but each section is slightly different from the others, so just leave it as is. Is quite complex to
        break the test data up into records.

        NOTE: this function is horrible, lots of DRY and bad logic
        """
        tgui = self.setup_skinetics_tgui(skinetics_or_events=skinetics_or_events)

        # load a test dataset into EE, and reshape it into 6 records (50 s each). Test as one rec first.

        test_num, test = self.get_individual_test_data(test_data, test_idx=5)
        dialog = self.load_and_analyse_test_spikes(tgui, test, skinetics_or_events=skinetics_or_events)

        self.check_dialog_burst_results(dialog, test)

        tgui.mw.dialogs["burst_analysis_dialog"].accept()
        tgui.mw.loaded_file.reshape_records(6, norm_time=norm_time, cut_samples=2)
        tgui.mw.dialog_manager.reshape_recs_finished(1)

        dialog = self.analyse_test_spikes(
            tgui,
            test,
            analyse_specific_recs=analyse_specific_recs,
            skinetics_or_events=skinetics_or_events,
        )

        # Create indexing variables that index the test data into records 1 to 6 -------------------------------

        br = dialog.burst_results
        all_spike_rec_num = np.zeros(test["spikes"].size)
        all_spike_rec_num[np.where(test["spikes"] < 50)] = 1
        all_spike_rec_num[np.where(np.logical_and(test["spikes"] > 50, test["spikes"] < 100))] = 2
        all_spike_rec_num[np.where(np.logical_and(test["spikes"] > 100, test["spikes"] < 150))] = 3
        all_spike_rec_num[np.where(np.logical_and(test["spikes"] > 150, test["spikes"] < 200))] = 4
        all_spike_rec_num[np.where(np.logical_and(test["spikes"] > 200, test["spikes"] < 250))] = 5
        all_spike_rec_num[np.where(test["spikes"] > 250)] = 6

        burst_times = test["bursts"].iloc[:, 0].to_numpy()
        burst_rec_num = np.zeros(burst_times.size)
        burst_rec_num[np.where(burst_times < 50)] = 1
        burst_rec_num[np.where(np.logical_and(burst_times > 50, burst_times < 100))] = 2
        burst_rec_num[np.where(np.logical_and(burst_times > 100, burst_times < 150))] = 3
        burst_rec_num[np.where(np.logical_and(burst_times > 150, burst_times < 200))] = 4
        burst_rec_num[np.where(np.logical_and(burst_times > 200, burst_times < 250))] = 5
        burst_rec_num[np.where(burst_times > 250)] = 6

        # Cycle through each record, testing the dialog.burst_result variable contains
        # the correct data. Test data is indexed
        # by record from the indexing variable above.
        rec_from = 2 if analyse_specific_recs else 0  # TODO: hard coded in analyse_test_spikes
        rec_to = 5 if analyse_specific_recs else 6
        peak_idx_so_far = num_bursts_so_far = 0
        all_rec_burst_peak_times = []
        all_rec_intra_burst_ISI = []
        for rec_idx in range(6):  # 6
            rec_num = rec_idx + 1

            if analyse_specific_recs and rec_idx in [0, 1, 5]:
                peak_idx_so_far += test["spikes"][np.where(all_spike_rec_num == rec_num)].size  # TODO: dry
                # num_bursts_so_far += test["bursts"].iloc[burst_rec_num == rec_num, 0].size
                continue

            test_rec_start_idx = np.where(np.isin(test["spikes"], test["bursts"].iloc[burst_rec_num == rec_num, 0]))[0]
            assert utils.allclose(
                br["start_idx"][rec_idx], test_rec_start_idx - peak_idx_so_far
            )  # need to remove first rec now as indexing is by rec

            test_rec_stop_idx = np.where(np.isin(test["spikes"], test["bursts"].iloc[burst_rec_num == rec_num, 1]))[0]
            assert utils.allclose(br["stop_idx"][rec_idx], test_rec_stop_idx - peak_idx_so_far)

            assert utils.allclose(
                br["lengths_ms"][rec_idx],
                test["bursts"].iloc[burst_rec_num == rec_num, 3] * 1000,
            )
            assert np.array_equal(
                br["num_spikes_per_burst"][rec_idx],
                test["bursts"].iloc[burst_rec_num == rec_num, 2],
            )
            assert utils.allclose(
                br["inter_burst_intervals"][rec_idx],
                test["bursts"].iloc[burst_rec_num == rec_num, 4][:-1] * 1000,
            )  # there are no inter-burst interval between recs now!
            assert np.array_equal(
                br["rec_fraction_of_spikes_in_burst"][rec_idx],
                np.sum(test["bursts"].iloc[burst_rec_num == rec_num, 2]) / np.sum(all_spike_rec_num == rec_num),
            )

            bursts_in_rec_range = np.isin(burst_rec_num, range(rec_from + 1, rec_to + 1))  # 3, 4, 5
            spikes_in_rec_range = np.isin(all_spike_rec_num, range(rec_from + 1, rec_to + 1))

            assert np.array_equal(
                br["total_fraction_of_spikes_in_burst"],
                np.sum(test["bursts"].iloc[bursts_in_rec_range, 2]) / all_spike_rec_num[spikes_in_rec_range].size,
            )

            time_to_subtract = (
                (tgui.mw.loaded_file.data.time_array[rec_idx - 1][-1] + tgui.mw.loaded_file.data.ts) * rec_idx
                if norm_time
                else 0
            )
            rec_spike_times = test["spikes"][all_spike_rec_num == rec_num] - time_to_subtract

            (
                __,
                test_all_burst_peak_times,
                test_all_intra_burst_ISI,
            ) = self.make_test_burst_results_per_spike(
                rec_spike_times,
                peak_times_ms=rec_spike_times * 1000,
                burst_start_idx=test_rec_start_idx - peak_idx_so_far,  # dry
                burst_stop_idx=test_rec_stop_idx - peak_idx_so_far,
            )

            assert utils.allclose(
                br["burst_peak_times"][rec_idx], np.hstack(test_all_burst_peak_times)
            )  # TODO: test norm vs. cumu
            all_rec_burst_peak_times.append(test_all_burst_peak_times)
            all_rec_intra_burst_ISI.append(test_all_intra_burst_ISI)

            assert utils.allclose(br["intra_burst_isi"][rec_idx], test_all_intra_burst_ISI)

            test_peak_burst_nums = self.make_test_peak_burst_nums(
                num_spikes_in_burst=test["bursts"].iloc[burst_rec_num == rec_num, 2],
                start_idx=num_bursts_so_far,
            )
            assert np.array_equal(br["peak_time_burst_nums"][rec_idx], np.hstack(test_peak_burst_nums))

            assert utils.allclose(br["all_peak_times"][rec_idx], rec_spike_times)
            assert br["total_num_peaks"] == np.sum(spikes_in_rec_range)  # test["spikes"].size

            self.check_list_widget_against_burst_results(br, dialog)
            tgui.mw.update_displayed_rec(rec_idx)
            self.check_test_all_records_plot(
                tgui,
                test,
                rec_spike_times,
                test_all_burst_peak_times,
                num_bursts_so_far,
            )

            peak_idx_so_far += test["spikes"][np.where(all_spike_rec_num == rec_num)].size
            num_bursts_so_far += test["bursts"].iloc[burst_rec_num == rec_num, 0].size

        # Test that the average results are also correct

        assert utils.allclose(
            br["averages"]["intra_burst_isi"],
            np.mean(np.hstack(all_rec_intra_burst_ISI)),
        )
        assert br["averages"]["inter_burst_intervals"] == np.mean(np.hstack(br["inter_burst_intervals"]))
        assert br["averages"]["num_spikes_per_burst"] == np.mean(np.hstack(br["num_spikes_per_burst"]))
        assert br["averages"]["length_ms"] == np.mean(np.hstack(br["lengths_ms"]))
        assert br["averages"]["total_num_bursts"] == np.sum(bursts_in_rec_range)

        # Now test the table, unfortunately it is not simple to test this along with single-rec ---------------------------------------------------
        # analysis, so dont here with lots of DRY :)

        tgui.left_mouse_click(dialog.dia.show_results_table_button)

        table_data = tgui.get_entire_qtable(dialog.table_dialog.table, na_as_inf=True, start_row=1)

        assert np.array_equal(table_data[:, 0], all_spike_rec_num[spikes_in_rec_range])

        test_spike_in_burst = []  # create the test data for finding the burst that each individial spike is in
        test_spike_times = []
        for spike_rec_num, spike_time in zip(
            all_spike_rec_num[spikes_in_rec_range], test["spikes"][spikes_in_rec_range]
        ):
            spike_rec_idx = spike_rec_num - 1
            time_to_subtract = (
                (tgui.mw.loaded_file.data.time_array[int(spike_rec_idx) - 1][-1] + tgui.mw.loaded_file.data.ts)
                * spike_rec_idx
                if norm_time
                else 0
            )
            spike_time -= time_to_subtract
            test_spike_times.append(spike_time)

            for burst_idx, burst_spike_times in enumerate(
                [burst for sublist in all_rec_burst_peak_times for burst in sublist]
            ):  # expand recs TODO: double check when not tired
                if burst_rec_num[bursts_in_rec_range][burst_idx] == spike_rec_num:
                    if spike_time in burst_spike_times:
                        test_spike_in_burst.append(burst_idx + 1)
                        break
            else:
                test_spike_in_burst.append(np.inf)

        # test the table data
        assert utils.allclose(table_data[:, 1], test_spike_times)
        assert np.array_equal(table_data[:, 2], test_spike_in_burst)

        assert np.array_equal(
            self.dropna(table_data[:, 3]),
            np.arange(test["bursts"][bursts_in_rec_range].shape[0]) + 1,
        )
        assert np.array_equal(self.dropna(table_data[:, 4]), test["bursts"].iloc[bursts_in_rec_range, 2])
        assert utils.allclose(
            self.dropna(table_data[:, 5]),
            test["bursts"].iloc[bursts_in_rec_range, 3] * 1000,
        )
        assert utils.allclose(self.dropna(table_data[:, 6]), np.hstack(all_rec_intra_burst_ISI))

        assert (
            self.dropna(table_data[:, 7]) == test["bursts"][bursts_in_rec_range].shape[0]
        )  # TODO: so much dry, make functions?
        assert (
            self.dropna(table_data[:, 8])
            == np.sum(test["bursts"].iloc[bursts_in_rec_range, 2]) / all_spike_rec_num[spikes_in_rec_range].size
        )  # DRY
        assert self.dropna(table_data[:, 9]) == np.mean(np.hstack(br["num_spikes_per_burst"]))
        assert self.dropna(table_data[:, 10]) == np.mean(np.hstack(br["lengths_ms"]))
        assert utils.allclose(self.dropna(table_data[:, 11]), np.mean(np.hstack(all_rec_intra_burst_ISI)))
        assert self.dropna(table_data[:, 12]) == np.mean(np.hstack(br["inter_burst_intervals"]))

        # Test Save Table
        self.check_saved_table(tgui, table_data, skinetics_or_events, excel_or_csv="csv")

        tgui.mw.dialogs["burst_analysis_dialog"].accept()
        tgui.shutdown()

    def check_saved_table(self, tgui, table_data, skinetics_or_events, excel_or_csv):
        """"""
        dialog = tgui.mw.dialogs["burst_analysis_dialog"].table_dialog
        dialog.save_as_excel_or_csv = excel_or_csv

        filename = f"test_burst_analysis_{excel_or_csv}.{excel_or_csv}"
        path_ = "/".join([tgui.test_base_dir, filename])

        if os.path.isfile(path_):
            os.remove(path_)

        QtCore.QTimer.singleShot(1000, lambda: keyboard.write(filename))
        QtCore.QTimer.singleShot(1200, lambda: keyboard.press("enter"))
        dialog.save_table_as_excel_or_csv()

        # QTest.qWait(20000)
        QtCore.QThreadPool.globalInstance().waitForDone(20000)

        if excel_or_csv == "excel":
            data = pd.read_excel(path_)
        else:
            data = pd.read_csv(path_)

        title = "Spikes" if skinetics_or_events == "skinetics" else "Events"

        assert list(data.columns.values) == [
            "Record",
            f"{title[:-1]} Time (s)",
            "In Burst",
            "Burst Number",
            f"Num. {title} in Burst",
            "Burst Length (ms)",
            f"Mean Inter-{title[:-1].lower()} Interval (ms)",
            "Number of Bursts",
            f"Fraction {title} in Burst",
            f"Avg. Num. {title} in Burst",
            "Avg. Burst Length (ms)",
            f"Avg. Within-burst Inter-{title[:-1].lower()} Interval (ms)",
            "Avg. Inter-burst Interval (ms)",
        ]

        data = data.to_numpy()
        table_data[np.where(table_data == np.inf)] = np.nan
        assert utils.allclose(data, table_data)

    def check_test_all_records_plot(self, tgui, test, rec_spike_times, rec_burst_peak_times, num_bursts_so_far):
        """"""
        num_bursts_in_rec = len(rec_burst_peak_times)

        # first, check for each burst plot the appropriate spikes are in. This does look very similar to below
        # but this checks that each spike is in the correct plot dict too
        for burst_idx in range(num_bursts_so_far, num_bursts_so_far + num_bursts_in_rec):
            burst_num = str(burst_idx + 1)
            plot_spike_times = tgui.mw.loaded_file_plot.burst_plot_dict[burst_num].xData

            for spike_time in rec_burst_peak_times[burst_idx - num_bursts_so_far]:
                assert np.isclose(spike_time, plot_spike_times, 1e-10).any()  # check spike time is in plot

        all_in_burst = np.hstack(rec_burst_peak_times)
        all_in_plot = np.hstack([plot.xData for plot in tgui.mw.loaded_file_plot.burst_plot_dict.values()])
        for spike_time in rec_spike_times:
            if spike_time not in all_in_burst:
                assert not np.isclose(spike_time, all_in_plot, 1e-10).any()

    def test_na_inter_burst_intervals(self):
        """
        Load a file that has 1 burst per record and so no ISI. Check it does not crash and NA display correctly
        """
        tgui = GuiTestSetup("wcp")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo(norm=True)
        tgui.test_load_norm_time_file()
        tgui.raise_mw_and_give_focus()

        tgui.set_analysis_type("skinetics")
        tgui.left_mouse_click(tgui.mw.mw.skinetics_auto_count_spikes_button)

        tgui.mw.mw.actionBurst_Analysis.trigger()
        dialog = tgui.mw.dialogs["burst_analysis_dialog"]
        tgui.enter_number_into_spinbox(dialog.dia.short_max_isi_spinbox, 50)
        tgui.enter_number_into_spinbox(dialog.dia.max_long_isi_spinbox, 50)
        tgui.enter_number_into_spinbox(dialog.dia.log_isi_min_spikes_per_burst_spinbox, 3)
        tgui.left_mouse_click(dialog.dia.run_burst_detection_logisi_button)

        assert dialog.burst_results["averages"]["inter_burst_intervals"] == "N/A"
        assert dialog.dia.list_widget.item(8).text() == "Inter-burst interval (ms): N/A"

        # check results are shown correctly on results table
        tgui.left_mouse_click(dialog.dia.show_results_table_button)
        assert dialog.table_dialog.table.item(1, 12).text() == "N/A"

    # Testers -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def check_burst_parameters(
        self,
        peak_times_ms,
        test,
        burst_start_idx,
        burst_stop_idx,
        burst_lengths_ms,
        num_spikes_in_burst,
        inter_burst_intervals,
        rec_fraction_of_spikes_in_burst,
        all_burst_idx,
        all_burst_peak_times,
        all_intra_burst_ISI,
    ):
        """
        Check all burst paraameters against the test data
        """
        assert utils.allclose(burst_lengths_ms, test["bursts"].iloc[:, 3] * 1000)  # convert to ms
        assert np.array_equal(num_spikes_in_burst, test["bursts"].iloc[:, 2])
        assert utils.allclose(inter_burst_intervals, test["bursts"].iloc[:-1, 4] * 1000)  # test has 0 as last IBI
        assert rec_fraction_of_spikes_in_burst == sum(test["bursts"].iloc[:, 2]) / len(test["spikes"])

        # Test additional parameters by calculating using different methods
        (
            test_all_burst_idx,
            test_all_burst_peak_times,
            test_all_intra_burst_ISI,
        ) = self.make_test_burst_results_per_spike(test["spikes"], peak_times_ms, burst_start_idx, burst_stop_idx)

        assert np.hstack(test_all_burst_idx).size == sum(test["bursts"].iloc[:, 2])
        if all_burst_idx is not None:
            assert np.array_equal(np.hstack(test_all_burst_idx), np.hstack(all_burst_idx))
        assert utils.allclose(np.hstack(test_all_burst_peak_times), np.hstack(all_burst_peak_times), 1e-10)
        assert utils.allclose(np.hstack(test_all_intra_burst_ISI), np.hstack(all_intra_burst_ISI), 1e-10)

        return test_all_burst_idx, test_all_burst_peak_times, test_all_intra_burst_ISI

    def check_dialog_burst_results(self, dialog, test):
        """
        Check the dialog burst_results variable results are all correct
        """
        br = dialog.burst_results
        assert np.array_equal(
            br["start_idx"][0],
            np.where(np.isin(test["spikes"], test["bursts"].iloc[:, 0]))[0],
        )
        assert np.array_equal(
            br["stop_idx"][0],
            np.where(np.isin(test["spikes"], test["bursts"].iloc[:, 1]))[0],
        )

        assert np.array_equal(
            br["rec_fraction_of_spikes_in_burst"][0],
            np.sum(test["bursts"].iloc[:, 2]) / test["spikes"].size,
        )
        assert np.array_equal(
            br["total_fraction_of_spikes_in_burst"],
            np.sum(test["bursts"].iloc[:, 2]) / test["spikes"].size,
        )

        assert utils.allclose(br["all_peak_times"][0], test["spikes"])
        assert br["total_num_peaks"] == test["spikes"].size

        # br["burst_start_end_data_idx"] not tested, tested on the plot later

        test_peak_burst_nums = self.make_test_peak_burst_nums(num_spikes_in_burst=test["bursts"].iloc[:, 2])

        assert np.array_equal(np.hstack(test_peak_burst_nums), br["peak_time_burst_nums"][0])

        __, _, test_all_intra_burst_ISI = self.check_burst_parameters(
            test["spikes"] * 1000,
            test,
            br["start_idx"][0],
            br["stop_idx"][0],
            burst_lengths_ms=br["lengths_ms"][0],
            num_spikes_in_burst=br["num_spikes_per_burst"][0],
            inter_burst_intervals=br["inter_burst_intervals"][0],
            rec_fraction_of_spikes_in_burst=br["rec_fraction_of_spikes_in_burst"][0],
            all_burst_idx=None,
            all_burst_peak_times=br["burst_peak_times"][0],
            all_intra_burst_ISI=br["intra_burst_isi"][0],
        )
        # Averages

        assert utils.allclose(br["averages"]["intra_burst_isi"], np.mean(test_all_intra_burst_ISI), 1e-10)
        assert utils.allclose(
            br["averages"]["inter_burst_intervals"],
            np.mean(test["bursts"].iloc[:-1, 4]) * 1000,
            1e-10,
        )
        assert br["averages"]["num_spikes_per_burst"] == np.mean(test["bursts"].iloc[:, 2])
        assert utils.allclose(br["averages"]["length_ms"], np.mean(test["bursts"].iloc[:, 3]) * 1000)
        assert br["averages"]["total_num_bursts"] == np.shape(test["bursts"])[0]

        self.check_list_widget_against_burst_results(br, dialog)

    def check_list_widget_against_burst_results(self, br, dialog):
        """
        This assumes burst results are already tested and known to be correct
        """
        list_widget = dialog.dia.list_widget

        assert list_widget.item(0).text() == "Summary"
        assert self.get_num(list_widget.item(1).text(), find_float=False) == br["averages"]["total_num_bursts"]
        assert self.get_num(list_widget.item(2).text()) == round(br["total_fraction_of_spikes_in_burst"], 3)
        assert list_widget.item(3).text() == ""
        assert list_widget.item(4).text() == "Average Results"
        assert self.get_num(list_widget.item(5).text()) == round(br["averages"]["num_spikes_per_burst"], 3)
        assert self.get_num(list_widget.item(6).text()) == round(br["averages"]["length_ms"], 3)
        assert self.get_num(list_widget.item(7).text()) == round(br["averages"]["intra_burst_isi"], 3)
        assert self.get_num(list_widget.item(8).text()) == round(br["averages"]["inter_burst_intervals"], 3)

    def make_test_burst_results_per_spike(self, peak_times, peak_times_ms, burst_start_idx, burst_stop_idx):
        """ """
        test_all_burst_idx = []
        test_all_burst_peak_times = []
        test_all_intra_burst_ISI = []
        for i in range(burst_start_idx.size):
            idx = np.arange(burst_start_idx[i], burst_stop_idx[i] + 1)
            test_all_burst_idx.append(idx)
            test_all_burst_peak_times.append(peak_times[idx])
            test_all_intra_burst_ISI.append(np.mean(np.diff(peak_times_ms[idx])))

        return test_all_burst_idx, test_all_burst_peak_times, test_all_intra_burst_ISI

    # Gui Tests Helpers -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def dropna(self, data):
        return data[~np.isnan(data)]

    def get_num(self, text, find_float=True):
        re_str = "\d+\.\d+" if find_float else "\d+"
        return float(re.findall(re_str, text)[0])

    def setup_skinetics_tgui(self, norm_or_cumu="normalised", skinetics_or_events="skinetics"):
        """ """
        to_load = (
            ["artificial_skinetics", "skinetics"]
            if skinetics_or_events == "skinetics"
            else ["artificial_events_one_record", "events_one_record"]
        )
        tgui = GuiTestSetup(to_load[0])
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = SPEED
        tgui.setup_artificial_data(norm_or_cumu, analysis_type=to_load[1])
        tgui.raise_mw_and_give_focus()
        return tgui

    def load_and_upsample_test_spikes(self, tgui, test, skinetics_or_events):
        """ """
        if tgui.mw.dialogs["burst_analysis_dialog"]:
            tgui.mw.dialogs["burst_analysis_dialog"].accept()

        tgui.mw.loaded_file.raw_data.time_array = np.atleast_2d(test["binary"]["time"]).astype(np.float64)
        tgui.mw.loaded_file.raw_data.vm_array = np.atleast_2d(test["binary"]["data"] * 100).astype(np.float64)
        tgui.mw.loaded_file.raw_data.im_array = np.atleast_2d(test["binary"]["data"]).astype(np.float64)
        tgui.mw.loaded_file.set_data_params(tgui.mw.loaded_file.raw_data)
        tgui.mw.loaded_file.set_data_params(tgui.mw.loaded_file.data)
        tgui.mw.loaded_file.data = copy.deepcopy(tgui.mw.loaded_file.raw_data)
        tgui.mw.loaded_file.init_analysis_results_tables()
        tgui.mw.clear_and_reset_widgets_for_new_file()

        # interp_factor = 2 if skinetics_or_events == "skinetics" else 5  # need more interp for different analysis to detect close spikes
        tgui.mw.loaded_file.upsample_data(
            interp_method="linear", interp_factor=5
        )  # need to upsample data to allow spike / event detection

    def analyse_test_spikes(
        self,
        tgui,
        test,
        thr_search_region=5,
        extend_fahp_end=False,
        max_isi_short=False,
        min_num_spikes=False,
        analyse_specific_recs=False,
        skinetics_or_events="skinetics",
    ):
        """ """
        if skinetics_or_events == "skinetics":
            if extend_fahp_end:
                tgui.mw.mw.actionSpike_Kinetics_Options_2.trigger()
                tgui.enter_number_into_spinbox(
                    tgui.mw.dialogs["skinetics_options"].dia.fahp_stop, 205
                )  # for manual spikes

            if analyse_specific_recs:
                tgui.switch_groupbox(tgui.mw.mw.skinetics_recs_to_analyse_groupbox, on=True)
                tgui.enter_number_into_spinbox(tgui.mw.mw.skinetics_recs_to_spinbox, 5)  # not zero idx
                tgui.enter_number_into_spinbox(tgui.mw.mw.skinetics_recs_from_spinbox, 3)

            tgui.run_artificial_skinetics_analysis(
                spike_detection_method="manual",
                thr_search_region=thr_search_region,
                manual_threshold_override=50,
            )

        else:
            tgui.set_analysis_type("events_thresholding")

            if analyse_specific_recs:
                tgui.switch_groupbox(tgui.mw.mw.events_threshold_recs_to_analyse_groupbox, on=True)
                tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_recs_to_spinbox, 5)
                tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_recs_from_spinbox, 3)

            tgui.left_mouse_click(tgui.mw.mw.events_threshold_analyse_events_button)
            tgui.set_combobox(tgui.mw.mw.events_threshold_peak_direction_combobox, 0)
            tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_amplitude_threshold_spinbox, 0.5)
            tgui.enter_number_into_spinbox(
                tgui.mw.dialogs["events_threshold_analyse_events"].dia.threshold_lower_spinbox,
                0.1,
            )
            tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_local_maximum_period_spinbox, 1)
            tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)
            tgui.mw.dialogs["events_threshold_analyse_events"].close()

        tgui.mw.mw.actionBurst_Analysis.trigger()
        dialog = tgui.mw.dialogs["burst_analysis_dialog"]
        tgui.enter_number_into_spinbox(dialog.dia.max_long_isi_spinbox, test["max_isi"])

        if max_isi_short:
            tgui.enter_number_into_spinbox(dialog.dia.short_max_isi_spinbox, max_isi_short)

        min_spikes = min_num_spikes if min_num_spikes is not False else 3
        tgui.enter_number_into_spinbox(dialog.dia.log_isi_min_spikes_per_burst_spinbox, min_spikes)

        tgui.left_mouse_click(dialog.dia.run_burst_detection_logisi_button)
        QtWidgets.QApplication.processEvents()

        return dialog

    def load_and_analyse_test_spikes(
        self,
        tgui,
        test,
        thr_search_region=5,
        extend_fahp_end=False,
        max_isi_short=False,
        min_num_spikes=False,
        skinetics_or_events="skinetics",
    ):
        """ """
        self.load_and_upsample_test_spikes(tgui, test, skinetics_or_events)

        dialog = self.analyse_test_spikes(
            tgui,
            test,
            thr_search_region=thr_search_region,
            extend_fahp_end=extend_fahp_end,
            max_isi_short=max_isi_short,
            min_num_spikes=min_num_spikes,
            skinetics_or_events=skinetics_or_events,
        )
        return dialog

    def make_test_peak_burst_nums(self, num_spikes_in_burst, start_idx=0):
        """ """
        test_peak_burst_nums = []
        for burst_idx, num_peaks in enumerate(num_spikes_in_burst):
            test_peak_burst_nums.append(np.ones(num_peaks) * burst_idx + start_idx + 1)
        return test_peak_burst_nums
