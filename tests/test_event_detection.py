from PySide6 import QtWidgets, QtCore, QtGui
from PySide6 import QtTest
from PySide6.QtTest import QTest
import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology.mainwindow.mainwindow import (
    MainWindow,
)
from ephys_data_methods import (
    core_analysis_methods,
    event_analysis_master,
    voltage_calc,
)
import event_detection_method_test_against as test_correlation

os.environ["PYTEST_QT_API"] = "PySide6"
from setup_test_suite import GuiTestSetup, get_test_base_dir
import utils_for_testing as test_utils
from utils import utils
import peakutils


class TestEvents:

    @pytest.fixture(scope="function")
    def tgui(test):
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.load_a_filetype("voltage_clamp_1_record")
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Correlation Utils
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def setup_test_events_detection(self, tgui):
        """
        Convenience function to retrieve data required to run voltage_calc.clements_bekkers_sliding_window() outside of Easy Electrophysiology for testing.
        """
        data = tgui.mw.loaded_file.data.im_array
        ts = tgui.mw.loaded_file.data.ts
        run_settings = {
            "fs": 1 / ts,
            "deconv_options": {"filt_low_hz": 1, "filt_high_hz": 200},
            "direction": -1,
            "ts": ts,
            "window_len_s": 0.01,
            "rise_s": 0.0005,
            "decay_s": 0.005,
        }
        template_samples = int(run_settings["window_len_s"] / run_settings["ts"])

        x = core_analysis_methods.generate_time_array(
            0,
            run_settings["window_len_s"],
            int(run_settings["window_len_s"] / run_settings["ts"]),
            run_settings["ts"],
        )

        template = core_analysis_methods.biexp_event_function(
            x, (0, -1, run_settings["rise_s"], run_settings["decay_s"])
        )

        def progress_bar_callback():
            pass

        return data, template, template_samples, run_settings, progress_bar_callback

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Test Event Detection Functions
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_deconvolution(self, tgui):
        """
        Test the implementation of deconvolution method (see voltage_calc.get_filtered_template_data_deconvolution()) against
        alternative implementation (https://biosig.sourceforge.io), results loaded from text file after running on same data).
        """
        (
            data,
            template,
            template_samples,
            run_settings,
            progress_bar_callback,
        ) = self.setup_test_events_detection(tgui)

        deconv = event_analysis_master.deconvolution_template_detection(data[0], run_settings, progress_bar_callback)

        # in Easy Electrophysiology the normalisation is achieved via adjusting the amplitude of the template
        # to match the data range. Here test an alternative implementation, to rescale the data and template range to 0 to +/- 1
        template_ = template - np.max(template)
        template_ = template_ / abs(np.min(template_) - np.max(template_))
        data_ = data[0] - np.max(data[0])
        data_ = data_ / abs(np.min(data_) - np.max(data_))

        deconv_alt_norm = voltage_calc.get_filtered_template_data_deconvolution(
            data_,
            template_,
            run_settings["fs"],
            run_settings["deconv_options"]["filt_low_hz"],
            run_settings["deconv_options"]["filt_high_hz"],
        )

        assert utils.allclose(deconv, deconv_alt_norm, 1e-8)

        # Test against biosig implementation (which itself differs from stimfits https://groups.google.com/g/stimfit/c/mg7nSv319Vs/m/a-RQETWnAAAJ, likely minor
        # scaling from the filtering or data / template)
        biosig_deconvolution = np.loadtxt(
            os.path.realpath(get_test_base_dir() + "/data_files/biosig_deconvolution.csv"),
            delimiter=",",
        )

        assert utils.allclose(deconv, biosig_deconvolution, 1e-8)

    def test_deconvolution_bandpass_filter(self):
        """
        Quick check on the deconvolution bandpass filter. This is implicitly checked in test_deconvolution() as the output matches other implementations.
        However test here by creating a signal of summed sine waves with known frequencies. Apply the band-pass filter to filter out frequencies
        at the edge of the added frequencies. Check that the filtered frequencies no longer exist in the signal.

        This is simple for the sharp high-pass cutoff, only performed very roughly here for Gaussian cutoff (checking for gradual decrease in low frequencies).
        TODO: test low frequency cutoff is Gaussian
        """
        (
            y,
            added_hz,
            fs,
            dist,
            n_freqs,
        ) = (
            test_utils.generate_test_frequency_spectra()
        )  # large dist between freqs as a roll off on the gaussian filter is very gradual
        Y = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(y), 1 / fs)

        # Filter the signal, check no frequencies remain below the low cutoff and the high cutoff rolls off.
        min_hz = np.mean(added_hz[2] + 5)
        max_hz = np.mean(np.array([added_hz[-3]]) + 5)
        filt_Y = voltage_calc.fft_filter_gaussian_window(Y, min_hz, max_hz, len(y), fs) / fs

        peaks_idx_after = peakutils.indexes(filt_Y, thres=0.01, min_dist=5)
        peaks_idx_after = peaks_idx_after[np.where(freqs[peaks_idx_after] > 0)]
        freqs_after = freqs[peaks_idx_after]
        freq_amplitudes_after = np.real(filt_Y[peaks_idx_after])

        assert np.array_equal(freqs_after, added_hz[np.where(added_hz > min_hz)])

        percent_decrase = (freq_amplitudes_after / freq_amplitudes_after[0]) * 100
        assert (percent_decrase <= np.array([100, 85, 60])).all()

    def test_correlation_and_detection_criterion(self, tgui):
        """
        Test correlation and detection criterion routines against alternative implementations:

        Luke Campagnola: https://github.com/AllenInstitute/neuroanalysis
        Stimfit: https://github.com/neurodroid/stimfit
        """
        (
            data,
            template,
            template_samples,
            run_settings,
            progress_bar_callback,
        ) = self.setup_test_events_detection(tgui)

        # Test against the previous (<v2.3.0-beta) sliding window implementation. These won't match exactly as the fitting algorithm is different
        # but should correlate almost perfectly with the actual values matching pretty well.
        detection_crit, betas, r_ = voltage_calc.clements_bekkers_sliding_window(
            data[0], template, progress_bar_callback
        )

        test_r, test_betas = test_correlation.sliding_window(
            data,
            run_settings["window_len_s"],
            run_settings["rise_s"],
            run_settings["decay_s"],
            ts=run_settings["ts"],
            downsample={"on": False},
            min_chunk_factor=1,
        )

        assert np.corrcoef(test_r, r_)[0, 1] > 0.999
        assert np.where(test_r < 0)[0].size == 0

        assert tgui.calculate_percent_isclose(test_r, r_, 1e-2) > 0.60
        assert tgui.calculate_percent_isclose(test_betas[0], betas[0], 1e-2) > 0.90
        assert tgui.calculate_percent_isclose(abs(test_betas[1]), betas[1], 1e-0) > 0.5

        # Test Against Luke Campagnola's implementation
        (
            test_detection_crit,
            test_scale,
            test_offset,
        ) = test_correlation.clements_bekkers(data[0], template)

        assert utils.allclose(test_detection_crit, detection_crit[: -template_samples + 1], 1e-08)
        assert utils.allclose(test_offset, betas[0][: -template_samples + 1], 1e-08)
        assert utils.allclose(test_scale, betas[1][: -template_samples + 1], 1e-08)

        # Test against stimfit implementation. They use a numerically stable correlation calculation leading to the
        # minor differences in correlation below, but perfect correlation between methods
        stf_detection_crit = np.load(os.path.realpath(get_test_base_dir() + "/data_files/stf_detection_crit.npy"))
        stf_correlation = np.load(os.path.realpath(get_test_base_dir() + "/data_files/stf_correlation.npy"))

        assert tgui.calculate_percent_isclose(detection_crit[:-template_samples], stf_detection_crit, 1e-5) > 0.99
        assert tgui.calculate_percent_isclose(r_[:-template_samples], stf_correlation, 1e-1) > 0.99
        assert tgui.calculate_percent_isclose(r_[:-template_samples], stf_correlation, 1e-2) > 0.60
        assert utils.allclose(np.corrcoef(r_[:-template_samples], stf_correlation)[1, 1], 1, 1e-10)
