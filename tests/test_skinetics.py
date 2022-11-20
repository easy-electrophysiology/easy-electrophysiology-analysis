import numpy as np
import sys
import os
import pytest
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), 'easy_electrophysiology'))
from ..easy_electrophysiology import easy_electrophysiology
from ephys_data_methods import core_analysis_methods, current_calc
from utils import utils
from generate_artificial_data import TestSingleSineWave
import scipy

class TestsKinetics:

    @pytest.fixture(scope="function")
    def tsine(test):
        return TestSingleSineWave()

    @pytest.fixture(scope="function")
    def cfgs(self):
        class Cfgs:
            def __init__(self):

                self.skinetics = {
                    "thr_method": "first_deriv",  # "first_deriv" "third_deriv" "method_I" "method_II" "leading_inflection" "max_curvature"
                    "first_deriv_max_or_cutoff": "max",
                    "third_deriv_max_or_cutoff": "max",
                    "first_deriv_cutoff": 0,
                    "third_deriv_cutoff": 0,
                    "method_I_lower_bound": 1.00,
                    "method_II_lower_bound": 1.00,
                }
        return Cfgs()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def check_close(self, test_data, ee_data, ts):
        """
        Check the difference between expected and true data is less than one sampling point
        """
        return abs(test_data - ee_data) < ts

    def ahp_test_data(self, fs):
        class Data:
            def __init__(self, fs):
                self.fs = fs
        return Data(fs)

    def monoexp(self, x, l):
        return np.exp(x**l)

    def generate_exp(self, l):
        x = np.linspace(0, 1, 10000)
        y = self.monoexp(x, l)
        first_deriv = np.diff(y)
        second_deriv = np.diff(y, n=2, append=np.diff(y, n=2)[-1])
        third_deriv = np.diff(y, n=3, append=[np.diff(y, n=3)[-1], np.diff(y, n=3)[-1]])
        return x, y, first_deriv, second_deriv, third_deriv

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Unit Tests
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("freq", np.arange(0.5, 5, 0.5))
    @pytest.mark.parametrize("min_cutoff", np.arange(0, 45, 5))
    @pytest.mark.parametrize("max_cutoff", np.arange(100, 55, -5))
    @pytest.mark.parametrize("amplitude", np.arange(1, 5, 0.5))
    def test_skinetics_rise_time(self, tsine, freq, amplitude, min_cutoff, max_cutoff):
        """
        Test the rise time of the artificial sine
        """
        sin_x, sin_y, fs, ts, params, idxs = tsine.generate_sine(freq, amplitude)

        rise_min_time, rise_min_vm, rise_max_time, rise_max_vm, rise_time = core_analysis_methods.calc_rising_slope_time(sin_y[0:idxs["max"] + 1],
                                                                                                                         sin_x[0:idxs["max"] + 1],
                                                                                                                         min_=0,
                                                                                                                         max_=np.max(sin_y),
                                                                                                                         min_cutoff_perc=min_cutoff,
                                                                                                                         max_cutoff_perc=max_cutoff,
                                                                                                                         interp=False)
        expected_rise_time = (np.arcsin(max_cutoff/100) - np.arcsin(min_cutoff/100)) / freq
        assert self.check_close(expected_rise_time, rise_time, ts)

        # test amplitude, kind of pointless as not used in EE.
        rise = (params["max"] - params["offset"])
        assert self.check_close(rise * (max_cutoff / 100), rise_max_vm, 0.01)
        assert self.check_close(rise * (min_cutoff / 100), rise_min_vm, 0.01)

    @pytest.mark.parametrize("freq", np.arange(0.5, 5, 0.5))
    @pytest.mark.parametrize("amplitude", np.arange(1, 5, 0.5))
    def test_skinetics_half_width(self, tsine, freq, amplitude):
        """
        """
        sin_x, sin_y, fs, ts, params, idxs = tsine.generate_sine(freq, amplitude)

        rise_mid_time, rise_mid_vm, decay_mid_time, decay_mid_vm, \
            fwhm, __, __ = core_analysis_methods.calculate_fwhm(sin_x[0:idxs["max"] + 1],
                                                                sin_y[0:idxs["max"] + 1],
                                                                sin_x[idxs["max"]:idxs["min"]+1],
                                                                sin_y[idxs["max"]:idxs["min"]+1],
                                                                amplitude / 2,
                                                                interp=False)
        expected_fwhm = 2 * (np.arcsin(1) - np.arcsin(0.5)) / freq
        assert self.check_close(expected_fwhm, fwhm, ts)
        rise = decay = (params["max"] - params["offset"])
        assert self.check_close(rise * 0.5, rise_mid_vm, 0.01)
        assert self.check_close(decay * 0.5, decay_mid_vm, 0.01)

    @pytest.mark.parametrize("freq", np.arange(0.5, 5, 0.5))
    @pytest.mark.parametrize("max_cutoff", np.arange(0, 45, 5))
    @pytest.mark.parametrize("min_cutoff", np.arange(100, 55, -5))
    @pytest.mark.parametrize("amplitude", np.arange(1, 5, 0.5))
    @pytest.mark.parametrize("thr_type", ["thr", "fahp"])
    def test_skinetics_decay_time(self, tsine, freq, amplitude, min_cutoff, max_cutoff, thr_type):
        """
        This function is very confusing.
        First, min_cutoff is the cutoff when the function is minimum. Max cutoff is the cutoff when the function is maximum -
        this means min_cutoff is higher (e.g. 80% of the decay).

        Second, the decay can be measured from peak to Thr ot peak to decay. If measured peak to thr it is just the first half of the
        sine decay from 1 to 0. If it is to fAHP, it is from 1 to -1.

        The true values can be calculated using arcsin that gives you the timepoint at which the value occurs on the y axis.
        However, the true x value requires some manipulation of the arcsin value because sine is cyclical (e.g. arcsin(0.5) should
        give inf. values but only gives one, the first.

        This was a tricky test to write so if you are editing, make sure to get the sketch-pad out!
        """
        sin_x, sin_y, fs, ts, params, idxs = tsine.generate_sine(freq, amplitude)

        min_bound = 0 if thr_type == "thr" else -amplitude

        decay_min_time, decay_min_vm, decay_max_time, decay_max_vm, decay_time = core_analysis_methods.calc_falling_slope_time(sin_y[2000:6000+1],
                                                                                                                               sin_x[2000:6000+1],
                                                                                                                               max_=params["max"],
                                                                                                                               min_=min_bound,
                                                                                                                               max_cutoff_perc=max_cutoff,
                                                                                                                               min_cutoff_perc=min_cutoff,
                                                                                                                               interp=False)
        sin_max_cutoff = 1 - ((1 - 0) * max_cutoff/100) if min_bound == 0 else 1 - (2 * (max_cutoff / 100))
        expected_min_time = np.arcsin(1) + (np.arcsin(1) - np.arcsin(sin_max_cutoff))
        assert self.check_close(expected_min_time / freq, decay_min_time, ts)

        sin_min_cutoff = 1 - ((1 - 0) * min_cutoff/100) if min_bound == 0 else(2 * (min_cutoff / 100) - 1)
        expected_max_time = np.arcsin(1) + (np.arcsin(1) - np.arcsin(sin_min_cutoff)) if min_bound == 0 else np.arcsin(1) * 3 - (np.arcsin(1) - np.arcsin(sin_min_cutoff))
        assert self.check_close(expected_max_time / freq, decay_max_time, ts)

        assert self.check_close(expected_max_time / freq - expected_min_time / freq,
                                decay_time,
                                ts)

    @pytest.mark.parametrize("freq", np.arange(0.5, 5, 0.5))
    @pytest.mark.parametrize("amplitude", np.arange(1, 5, 0.5))
    def test_skinetics_fahp_mahp(self, tsine, freq, amplitude):
        """
        """
        sin_x, sin_y, fs, ts, params, idxs = tsine.generate_sine(freq, amplitude)
        data = self.ahp_test_data(fs * 1000)  # because ahp function assumes s and convert to ms

        mahp_amplitude = -amplitude * 1.5
        mahp_idx = params["num_samples"] - 5
        sin_y[mahp_idx] = mahp_amplitude

        search_period = sin_x[-10] - sin_x[idxs["max"]]
        fahp_time, fahp_vm, fahp_idx, fahp = core_analysis_methods.calculate_ahp(data,
                                                                                 0,
                                                                                 search_period,
                                                                                 sin_y,  sin_x,  idxs["max"], 0)
        expected_fahp_time = np.arcsin(1) * (3 / freq)

        assert self.check_close(expected_fahp_time, fahp_time, ts)
        assert fahp_vm == -amplitude
        assert fahp_vm - params["offset"] == -amplitude
        search_period = sin_x[-1] - sin_x[2000]
        mahp_time, mahp_vm, mahp_idx, mahp = core_analysis_methods.calculate_ahp(data,
                                                                                 0,
                                                                                 search_period,
                                                                                 sin_y,  sin_x,  idxs["max"], 0)
        expected_mahp_time = sin_x[mahp_idx]
        assert self.check_close(expected_mahp_time, mahp_time, ts)
        assert mahp_vm == mahp_amplitude
        assert mahp_vm - params["offset"] == mahp_amplitude

    @pytest.mark.parametrize("l", np.arange(1, 10, 0.5))
    def test_threshold(self, cfgs, l):

        x, y, first_deriv, second_deriv, third_deriv = self.generate_exp(l)

        # First Deriv
        cfgs.skinetics["thr_method"] = "first_deriv"
        cfgs.skinetics["first_deriv_max_or_cutoff"] = "max"
        test_max_idx = np.argmax(first_deriv)
        max_idx = core_analysis_methods.calculate_threshold(first_deriv, second_deriv,  third_deriv, 0, 10000, cfgs)
        assert test_max_idx == max_idx

        cfgs.skinetics["first_deriv_max_or_cutoff"] = "cutoff"
        cfgs.skinetics["first_deriv_cutoff"] = first_deriv[5000]
        test_cutoff_idx = np.where(first_deriv > first_deriv[5000])[0][0]
        cutoff_idx = core_analysis_methods.calculate_threshold(first_deriv, second_deriv,  third_deriv, 0, 10000, cfgs)
        assert test_cutoff_idx == cutoff_idx

        # third deriv
        cfgs.skinetics["thr_method"] = "third_deriv"
        cfgs.skinetics["third_deriv_max_or_cutoff"] = "max"
        test_max_idx = np.argmax(third_deriv)
        max_idx = core_analysis_methods.calculate_threshold(first_deriv, second_deriv,  third_deriv, 0, 10000, cfgs)
        assert test_max_idx == max_idx

        cfgs.skinetics["third_deriv_max_or_cutoff"] = "cutoff"
        cfgs.skinetics["third_deriv_cutoff"] = third_deriv[5000]
        test_cutoff_idx = np.where(third_deriv > third_deriv[5000])[0][0]
        cutoff_idx = core_analysis_methods.calculate_threshold(first_deriv, second_deriv,  third_deriv, 0, 10000, cfgs)
        assert test_cutoff_idx == cutoff_idx

        # method i
        cfgs.skinetics["thr_method"] = "method_I"
        method_i = second_deriv / first_deriv
        threshold = cfgs.skinetics["method_I_lower_bound"] = np.max(first_deriv) * 0.1
        method_i[np.where(first_deriv < threshold)] = np.nan
        test_method_i_idx = np.nanargmax(method_i)
        method_i_idx = core_analysis_methods.calculate_threshold(first_deriv, second_deriv,  third_deriv, 0, 10000, cfgs)
        assert test_method_i_idx == method_i_idx

        # method ii
        cfgs.skinetics["thr_method"] = "method_II"
        method_ii = ((third_deriv * first_deriv) - (second_deriv**2)) / (first_deriv**3)
        threshold = cfgs.skinetics["method_II_lower_bound"] = np.max(first_deriv) * 0.1
        method_ii[np.where(first_deriv < threshold)] = np.nan
        test_method_ii_idx = np.nanargmax(method_ii)
        method_ii_idx = core_analysis_methods.calculate_threshold(first_deriv, second_deriv,  third_deriv, 0, 10000, cfgs)
        assert test_method_ii_idx == method_ii_idx

        # inflection
        cfgs.skinetics["thr_method"] = "leading_inflection"
        test_inflection = np.argmin(first_deriv)
        inflection = core_analysis_methods.calculate_threshold(first_deriv, second_deriv,  third_deriv, 0, 10000, cfgs)
        assert test_inflection == inflection

        # max curve
        cfgs.skinetics["thr_method"] = "max_curvature"
        test_max_curve = np.argmax(second_deriv * (1 + (first_deriv**2))**(-3/2))
        max_curve = core_analysis_methods.calculate_threshold(first_deriv, second_deriv,  third_deriv, 0, 10000, cfgs)
        assert test_max_curve == max_curve

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Phase Plot Analysis Tests
# ----------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("interpolate", [True, False])
    @pytest.mark.parametrize("increase", [1, 2, 5, 10, 100])
    def test_calculate_phase_plot_linear(self, interpolate, increase):
        """
        When the increase is constant, the vm, vm_diff plot should be a straight line. Check wth interpoalte also (matches perfectly, ==, without interpolation).
        """
        data = np.arange(0, 1000, increase)
        ts = 1/1000  # i.e. 1 ms
        x, x_diff = current_calc.calculate_phase_plot(data, ts, interpolate=interpolate)
        slope, __, __, __, __ = scipy.stats.linregress(x, x_diff)

        if not interpolate:
            assert np.array_equal(x, data[:-1])
        assert utils.allclose(x_diff, increase, 1e-10)
        assert utils.allclose(slope, 0, 1e-10)

    @pytest.mark.parametrize("interpolate", [True, False])
    def test_calculate_phase_plot_exponential(self, interpolate):
        """
        For any expoential function on interger domain, the ratio increase is the base - 1.
        e.g. (a^(n+1) - a^n) / a^n = a - 1
        This is cool! and is used below
        """
        data = np.exp(np.arange(0, 100))
        ts = 1/1000
        x, x_diff = current_calc.calculate_phase_plot(data, ts, interpolate=interpolate)
        slope, __, __, __, __ = scipy.stats.linregress(x, x_diff)

        assert np.isclose(slope, np.e - 1, 1e-10)

        # test ts by scaling by a half
        ts = 1 / 500
        x, x_diff = current_calc.calculate_phase_plot(data, ts, interpolate=interpolate)
        slope, __, __, __, __ = scipy.stats.linregress(x, x_diff)

        assert np.isclose(slope, (np.e - 1) / 2, 1e-10)

    def test_phase_plot_params(self):
        """
        Manually generate an array with clear parameters to find. Check all parameter calculating functions.
        """
        #                               thr  vmax                min       last vmax (cut)
        diffs = np.array([0, 10, 5, 49, 50,  55, -5,  -25,  0,  10,  -100, 65, 1000])
        #         data       [0, 10, 15, 64, 114, 169, 164, 139, 139, 149,  49, 114]
        data = np.cumsum(diffs)
        
        x, x_diff = current_calc.calculate_phase_plot(data, ts=1/1000, interpolate=False)

        threshold = 49
        test_threshold_x, test_threshold_x_diff = current_calc.calculate_threshold(x, x_diff, threshold)

        assert test_threshold_x == 64
        assert test_threshold_x_diff == 50

        test_vmax_x, test_vmax_x_diff = current_calc.calculate_vmax(x, x_diff)  # the last vm is cut off because of diff(). Under normal circumstances, this will have no effect for AP analysis
        assert test_vmax_x == 169
        assert test_vmax_x_diff == -5

        test_vm_diff_min_x, test_vm_diff_min_x_diff = current_calc.calculate_vm_diff_min(x, x_diff)
        assert test_vm_diff_min_x == 149
        assert test_vm_diff_min_x_diff == -100

        test_vm_diff_max_x, test_vm_diff_max_x_diff = current_calc.calculate_vm_diff_max(x, x_diff)
        assert test_vm_diff_max_x == 114
        assert test_vm_diff_max_x_diff == 1000