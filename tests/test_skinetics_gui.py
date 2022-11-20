from PySide2 import QtWidgets, QtCore, QtGui
from PySide2 import QtTest
import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology.easy_electrophysiology.easy_electrophysiology import MainWindow
from ephys_data_methods import current_calc, core_analysis_methods
from utils import utils
from setup_test_suite import GuiTestSetup
import utils_for_testing as test_utils
import time
import pandas as pd
from types import SimpleNamespace
from slow_vs_fast_settings import get_settings
os.environ["PYTEST_QT_API"] = "pyside2"

SPEED = "fast"

class TestsKinetics:
    """
    Test all AP kinetics analysis through GUI.

    NOTE: for some reason pytest generates an absolutely enourmouse (~1GB) warnings for this test class. Not clear why, run with no-warnings flag.
    """
    @pytest.fixture(scope="function", params=["normalised", "cumulative"], ids=["normalised", "cumulative"])
    def tgui(test, request):
        tgui = GuiTestSetup("artificial_skinetics")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = SPEED
        tgui.setup_artificial_data(request.param, analysis_type="skinetics")
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def get_qtable_col_for_select_recs(self, tgui, header, rec, filenum=0):
        """
        """
        qtable = tgui.get_entire_qtable()
        header = tgui.mw.cfgs.get_table_col_headers("skinetics", header)[1]
        header_col = tgui.mw.mw.table_tab_tablewidget.findItems(header + "   ", QtGui.Qt.MatchExactly)[0].column()

        num_params = len(tgui.mw.cfgs.skinetics_params()) + 1
        cols_start = num_params * filenum
        qtable = qtable[:, cols_start:]

        if rec is None:
            qtable_results = qtable[:, header_col]
        else:
            rec_idx = np.where(qtable[:, 0] == rec + 1)
            qtable_results = qtable[rec_idx, header_col]

        return qtable_results

    @staticmethod
    def get_loaded_file(tgui, main_key, param_key, rec):
        return tgui.get_skinetics_param_from_skinetics_data(tgui.mw.loaded_file.skinetics_data, main_key, param_key, rec)

    @staticmethod
    def get_stored_tabledata(tgui, main_key, param_key, rec, filenum=0):
        return np.array([dict_[param_key] for dict_ in utils.flatten_dict(tgui.mw.stored_tabledata.skinetics_data[filenum], rec, main_key)])

    @staticmethod
    def get_spiketimes_within_bounds(tgui, rec_from, rec_to, bounds_vm):
        """
        """
        data = utils.np_empty_nan((tgui.adata.num_recs,
                                   tgui.adata.max_num_spikes))

        for rec in range(rec_from, rec_to + 1):

            within_rec = tgui.adata.get_within_bounds_spiketimes(rec, bounds_vm, tgui.time_type)

            new_array = utils.np_empty_nan(tgui.adata.max_num_spikes)
            new_array[0:len(within_rec)] = within_rec

            data[rec, :] = new_array

        return data

    @staticmethod
    def set_ahp(tgui, fahp_or_mahp, to, from_):

        if fahp_or_mahp == "fahp":
            tgui.enter_number_into_spinbox(tgui.mw.dialogs["skinetics_options"].dia.fahp_start,
                                           to)
            tgui.enter_number_into_spinbox(tgui.mw.dialogs["skinetics_options"].dia.fahp_stop,
                                           from_)
        elif fahp_or_mahp == "mahp":
            tgui.enter_number_into_spinbox(tgui.mw.dialogs["skinetics_options"].dia.mahp_start,
                                           to)
            tgui.enter_number_into_spinbox(tgui.mw.dialogs["skinetics_options"].dia.mahp_stop,
                                           from_)

    @staticmethod
    def data_within_bounds(tgui, data, rec, bounds_vm):

        if bounds_vm:
            within_bounds_bool = tgui.adata.get_within_bounds_spiketimes(rec, bounds_vm, tgui.time_type, return_as_bool=True, cut_nan=True)

            if not any(within_bounds_bool):  # some recs will have no spikes within the bounds interval
                data = False
            else:
                data = data[within_bounds_bool]

        return data

    @staticmethod
    def check_plot(tgui, data_to_test_against, rec, plot_key, xdata_or_ydata, sub_data_dict=False):
        """
        Test data against the data stored in skinetics result plot. See plotgraphs.skinetics_plot_dict for keys.

        sub_data_dict:
            The plot data can be organised as a single array of data in order (i.e. threshold). In this case sub_data_dict=False.

            We may want to subtract some threshold from the plot (i.e. fahp - threshold) to test it. In this case sub_data_dict["type"] = subtract
            and sub_data_dict["data"] = the value to subtract.

            Finally, the plot data may be ordered in two parts, the first half on the spike rise and the second half on the spike decay (i.e. half width).
            In this case we want to subtract the second half from the first half to calculate the parameter.
        """
        tgui.mw.update_displayed_rec(rec)
        peak_plot = tgui.mw.loaded_file_plot.skinetics_plot_dict[plot_key]

        if peak_plot:

            plot_data = peak_plot.xData if xdata_or_ydata == "xData" else peak_plot.yData

            if sub_data_dict:

                if sub_data_dict["type"] == "subtract":
                    plot_data = plot_data - sub_data_dict["data"]

                elif sub_data_dict["type"] == "half_rise_half_decay":
                    num_spikes = sub_data_dict["data"]
                    rise_data = plot_data[:num_spikes]
                    decay_data = plot_data[num_spikes:]

                    plot_data = decay_data - rise_data

            if not np.any(data_to_test_against):
                return plot_data
            else:
                assert np.array_equal(plot_data, data_to_test_against)

    @staticmethod
    def assert_within_a_ts(time_1, time_2, ts):
        assert (abs(time_1 - time_2) < ts).all()

    def get_rise_decay_and_fwhm_error(self, tgui, rec_from, rec_to, min_cutoff, max_cutoff):
        """
        Find the error between test decay / rise time / half width and the true, caused by signal
        discretion (interpolation will help reduce this).
        """
        test_decay_times = tgui.adata.get_times_paramteres("decay_time", min_cutoff, max_cutoff)
        test_rise_times = tgui.adata.get_times_paramteres("rise_time", max_cutoff, min_cutoff)
        test_half_widths = tgui.adata.get_times_paramteres("half_width")

        error = {k: np.array([]) for k in ["rise", "decay", "fwhm"]}

        for rec in range(rec_from, rec_to + 1):
            loaded_file_decay_times = self.get_loaded_file(tgui, "decay_time", "decay_time_ms", rec) / 1000
            error["decay"] = np.hstack([error["decay"], abs(test_decay_times[rec] - loaded_file_decay_times)])

            loaded_file_rise_times = self.get_loaded_file(tgui, "rise_time", "rise_time_ms", rec) / 1000
            error["rise"] = np.hstack([error["rise"], abs(test_rise_times[rec] - loaded_file_rise_times)])

            loaded_file_half_widths = self.get_loaded_file(tgui, "fwhm", "fwhm_ms", rec) / 1000
            error["fwhm"] = np.hstack([error["fwhm"], abs(test_half_widths[rec] - loaded_file_half_widths)])

        return error

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Utils - Testing Functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def check_spiketimes(self, tgui, rec_from, rec_to, bounds_vm, filenum):
        """
        Check the spike times in loaded_file, stored_tabledata and qTable on the plot all match the test data.
        """
        test_spike_times = tgui.set_out_of_rec_to_nan(rec_from, rec_to,
                                                      array_size=(tgui.adata.num_recs, tgui.adata.max_num_spikes),
                                                      data=tgui.adata.peak_times_)
        if bounds_vm:
            test_spike_times = self.get_spiketimes_within_bounds(tgui, rec_from, rec_to, bounds_vm)

        loaded_file_spike_times = test_utils.get_spike_times_from_spike_info(tgui.adata,
                                                                             tgui.mw.loaded_file.skinetics_data)
        tabledata_spike_times = test_utils.get_spike_times_from_spike_info(tgui.adata,
                                                                           tgui.mw.stored_tabledata.skinetics_data[filenum])

        tgui.check_spiketimes(tgui, loaded_file_spike_times,
                              test_spike_times, "model spiketimes")
        tgui.check_spiketimes(tgui, tabledata_spike_times, test_spike_times,
                              "tabledata_spiketimes")

        for rec in range(rec_from, rec_to + 1):

            qtable_vm = self.get_qtable_col_for_select_recs(tgui, "spike_time", rec, filenum)
            assert (qtable_vm == tgui.clean(test_spike_times[rec])).all()

            self.check_plot(tgui, tgui.clean(test_spike_times[rec]),
                            rec, "peak_plot", "xData")

    def check_skinetics_threshold(self, tgui, rec, analyse_specific_recs, bounds_vm, filenum):
        """
        Check the skinetics threshold s in loaded_file, stored_tabledata and qtTable on the plot all match the test data.
        """
        test_spike_sample_idx, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs,
                                                                                    tgui.adata.spike_sample_idx)

        test_thr_vm = tgui.adata.resting_vm
        test_thr_time = tgui.adata.time_array[rec][tgui.clean(test_spike_sample_idx[rec]).astype(int)]

        test_thr_time = self.data_within_bounds(tgui, test_thr_time,
                                                rec, bounds_vm)

        if test_thr_time is False:
            return

        loaded_file_time = self.get_loaded_file(tgui, "thr", "time", rec)
        loaded_file_vm = self.get_loaded_file(tgui, "thr", "vm", rec)

        tabledata_time = self.get_stored_tabledata(tgui, "thr", "time", rec, filenum)
        tabledata_vm = self.get_stored_tabledata(tgui, "thr", "vm", rec, filenum)

        qtable_vm = self.get_qtable_col_for_select_recs(tgui, "thr", rec, filenum)

        assert (test_thr_time == loaded_file_time).all()
        assert (test_thr_time == tabledata_time).all()

        assert (test_thr_vm == loaded_file_vm).all()
        assert (test_thr_vm == tabledata_vm).all()
        assert (test_thr_vm == qtable_vm).all()

        self.check_plot(tgui, np.tile(test_thr_vm, len(test_thr_time)),
                        rec, "thr_plot", "yData")
        self.check_plot(tgui, test_thr_time,
                        rec, "thr_plot", "xData")

    def check_skinetics_qtable_spike_order(self, tgui, rec, bounds_vm, filenum):
        """
        Check the spike order displayed in the table match the true order.
        """
        if bounds_vm:
            spikes_within_bounds = tgui.adata.get_within_bounds_spiketimes(rec, bounds_vm, tgui.time_type)
            num_spikes = len(spikes_within_bounds)
            test_spike_nums = np.arange(num_spikes) + 1
        else:
            test_spike_nums = np.arange(tgui.adata.spikes_per_rec[rec]) + 1

        qtable_spike_nums = self.get_qtable_col_for_select_recs(tgui, "spike_number", rec, filenum)
        assert (qtable_spike_nums == test_spike_nums).all()

        qtable_spike_times = self.get_qtable_col_for_select_recs(tgui, "spike_time", rec, filenum)
        assert (qtable_spike_times[0] == sorted(qtable_spike_times[0])).all()

    def check_skinetics_peak_and_amplitude(self, tgui, rec, bounds_vm, filenum):
        """
        Check the peaks and amplitudes in loaded_file, stored_tabledata and qtTable on the plot all match the test data.
        """
        test_amplitudes = tgui.adata.all_true_peaks[rec] - tgui.adata.resting_vm
        test_peaks = tgui.adata.all_true_peaks[rec]

        test_amplitudes = self.data_within_bounds(tgui, test_amplitudes, rec, bounds_vm)
        test_peaks = self.data_within_bounds(tgui, test_peaks, rec, bounds_vm)

        if test_amplitudes is False:
            return

        loaded_file_amplitudes = self.get_loaded_file(tgui, "amplitude", "vm", rec)
        loaded_file_peaks = self.get_loaded_file(tgui, "peak", "vm", rec)

        tabledata_amplitudes = self.get_stored_tabledata(tgui, "amplitude", "vm", rec, filenum)
        tabledata_peaks = self.get_stored_tabledata(tgui, "peak", "vm", rec, filenum)

        qtable_amplitudes = self.get_qtable_col_for_select_recs(tgui, "amplitude", rec, filenum)
        qtable_peaks = self.get_qtable_col_for_select_recs(tgui, "peak", rec, filenum)

        assert utils.allclose(test_amplitudes, loaded_file_amplitudes, 1e-10)
        assert utils.allclose(test_peaks, loaded_file_peaks, 1e-10)

        assert utils.allclose(test_amplitudes, tabledata_amplitudes, 1e-10)
        assert utils.allclose(test_peaks, tabledata_peaks, 1e-10)

        assert utils.allclose(test_amplitudes, qtable_amplitudes, 1e-10)
        assert utils.allclose(test_peaks, qtable_peaks, 1e-10)

        self.check_plot(tgui, test_peaks,
                        rec, "peak_plot", "yData")

    def check_skinetics_half_widths(self, tgui, rec, bounds_vm, filenum):
        """
        Check the half_widths in loaded_file, stored_tabledata and qtTable on the plot all match the test data.
        """
        test_half_widths = tgui.adata.get_times_paramteres("half_width", bounds_vm=bounds_vm, time_type=tgui.time_type)

        if bounds_vm:
            if not np.any(test_half_widths[rec]):
                return

        loaded_file_half_widths = self.get_loaded_file(tgui, "fwhm", "fwhm_ms", rec) / 1000
        self.assert_within_a_ts(test_half_widths[rec], loaded_file_half_widths, tgui.adata.ts)

        tabledata_half_widths = self.get_stored_tabledata(tgui, "fwhm", "fwhm_ms", rec, filenum) / 1000
        self.assert_within_a_ts(test_half_widths[rec], tabledata_half_widths, tgui.adata.ts)

        qtable_half_widths = self.get_qtable_col_for_select_recs(tgui, "fwhm", rec, filenum) / 1000
        self.assert_within_a_ts(test_half_widths[rec], qtable_half_widths, tgui.adata.ts)

        plot_hw_data = self.check_plot(tgui, False,
                                       rec, "fwhm_plot", "xData", sub_data_dict=dict(type="half_rise_half_decay",
                                                                                     data=len(test_half_widths[rec])))

        assert (abs(test_half_widths[rec] - plot_hw_data) < tgui.adata.ts).all()

    def check_skinetics_ahps(self, tgui, rec, bounds_vm, filenum):
        """
        Check the ahps in loaded_file, stored_tabledata and qtTable on the plot all match the test data. First test the
        actual fahp and mahp value (raw, without thr subtraction) are correct on the loaded_file and stored tabledata.

        Then check that the true fahp / mahp (minus threshold) is correct on the actual table. This should cover all bases.

        Note that the fahp ahp are further tested with variable start / stop search times in test_fahp_and_ahp_timings()
        """
        test_neg_peaks = tgui.adata.all_true_mins[rec]

        test_neg_peaks = self.data_within_bounds(tgui, test_neg_peaks, rec, bounds_vm)

        if test_neg_peaks is False:
            return

        loaded_file_fahp = self.get_loaded_file(tgui, "fahp", "vm", rec)
        loaded_file_mahp = self.get_loaded_file(tgui, "mahp", "vm", rec)

        tabledata_fahp = self.get_stored_tabledata(tgui, "fahp", "vm", rec, filenum)
        tabledata_mahp = self.get_stored_tabledata(tgui, "mahp", "vm", rec, filenum)

        assert utils.allclose(test_neg_peaks, loaded_file_fahp, 1e-10)
        assert utils.allclose(test_neg_peaks, loaded_file_mahp, 1e-10)

        assert utils.allclose(test_neg_peaks, tabledata_fahp, 1e-10)
        assert utils.allclose(test_neg_peaks, tabledata_mahp, 1e-10)

        test_actual_ahp = test_neg_peaks - tgui.adata.resting_vm

        tabledata_actual_fahp = self.get_stored_tabledata(tgui, "fahp", "value", rec, filenum)
        qtable_actual_fahp = self.get_qtable_col_for_select_recs(tgui, "fahp", rec, filenum)

        tabledata_actual_mahp = self.get_stored_tabledata(tgui, "mahp", "value", rec, filenum)
        qtable_actual_mahp = self.get_qtable_col_for_select_recs(tgui, "mahp", rec, filenum)

        assert utils.allclose(test_actual_ahp, tabledata_actual_fahp, 1e-10)
        assert utils.allclose(test_actual_ahp, qtable_actual_fahp, 1e-10)

        assert utils.allclose(test_actual_ahp, tabledata_actual_mahp, 1e-10)
        assert utils.allclose(test_actual_ahp, qtable_actual_mahp, 1e-10)

        self.check_plot(tgui, test_actual_ahp,
                        rec, "mahp_plot", "yData", sub_data_dict=dict(type="subtract",
                                                                      data=tgui.adata.resting_vm))
        self.check_plot(tgui, test_actual_ahp,
                        rec, "fahp_plot", "yData", sub_data_dict=dict(type="subtract",
                                                                      data=tgui.adata.resting_vm))

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [False, True])  
    @pytest.mark.parametrize("run_with_bounds", [True, False])
    @pytest.mark.parametrize("spike_detection_method", ["auto_record", "auto_spike", "manual"])
    def test_skinetics_main_results(self, tgui, spike_detection_method, analyse_specific_recs, run_with_bounds):
        """
        Run analysis with different settings (e.g. specific recs, within bounds) and check all analysis results are correct.

        TODO: optimisation, each test goes through every rec individually when the rec could be looped outside the functions
        """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        for filenum in range(3):

            bounds_vm = tgui.run_artificial_skinetics_analysis(run_with_bounds=run_with_bounds, spike_detection_method=spike_detection_method, override_mahp_fahp_defaults=True)

            self.check_spiketimes(tgui, rec_from, rec_to, bounds_vm, filenum)

            for rec in range(rec_from, rec_to + 1):
                self.check_skinetics_qtable_spike_order(tgui, rec, bounds_vm, filenum)

                self.check_skinetics_threshold(tgui, rec, analyse_specific_recs, bounds_vm, filenum)
                self.check_skinetics_peak_and_amplitude(tgui, rec, bounds_vm, filenum)
                self.check_skinetics_half_widths(tgui, rec, bounds_vm, filenum)
                self.check_skinetics_ahps(tgui, rec, bounds_vm, filenum)

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data("cumulative", analysis_type="skinetics")

        tgui.shutdown()

    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("min_max_cutoff", [[0, 100], [5, 95], [10, 90], [15, 85], [20, 80], [25, 75], [30, 70], [35, 65], [40, 60], [45, 55]])
    def test_skinetics_rise_times(self, tgui, analyse_specific_recs, min_max_cutoff):
        """
        Test rise time is correct when using a number of different ranges to calculate across. Due to discrete sampling, the results will
        not match test data exactly but should be within 1x ts. The main purpose of this test is to check that the order is correct,
        the underlying function is tested more thoroughly in test_skinetics.py

        No need to test further within bounds, tested enough above and slows things down a lot
        """
        min_cutoff, max_cutoff = min_max_cutoff

        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)
        tgui.run_artificial_skinetics_analysis(max_cutoff=max_cutoff, min_cutoff=min_cutoff)
        test_rise_times = tgui.adata.get_times_paramteres("rise_time", max_cutoff, min_cutoff)

        for rec in range(rec_from, rec_to + 1):

            loaded_file_rise_times = self.get_loaded_file(tgui, "rise_time", "rise_time_ms", rec) / 1000
            self.assert_within_a_ts(test_rise_times[rec], loaded_file_rise_times, tgui.adata.ts)

            tabledata_rise_times = self.get_stored_tabledata(tgui, "rise_time", "rise_time_ms", rec) / 1000
            self.assert_within_a_ts(test_rise_times[rec], tabledata_rise_times, tgui.adata.ts)

            qtable_rise_times = self.get_qtable_col_for_select_recs(tgui, "rise_time", rec) / 1000
            self.assert_within_a_ts(test_rise_times[rec], qtable_rise_times, tgui.adata.ts)

            plot_rise_data = self.check_plot(tgui, False,  rec, "rise_plot", "xData",
                                             sub_data_dict=dict(type="half_rise_half_decay", data=len(test_rise_times[rec])))
            self.assert_within_a_ts(test_rise_times[rec], plot_rise_data, tgui.adata.ts)

    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("min_max_cutoff", [[10, 90], [16, 85], [0, 100], [5, 95], [20, 80], [25, 75], [30, 70], [35, 65], [40, 60], [45, 55]])
    @pytest.mark.parametrize("min_bound", ["fAHP", "thr"])
    def test_skinetics_decay_times(self, tgui, analyse_specific_recs, min_bound, min_max_cutoff):
        """
        See test_skinetics_rise_times()
        """
        min_cutoff, max_cutoff = min_max_cutoff

        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)
        tgui.run_artificial_skinetics_analysis(max_cutoff=max_cutoff, min_cutoff=min_cutoff)

        test_decay_times = tgui.adata.get_times_paramteres("decay_time",
                                                           min_cutoff, max_cutoff)

        for rec in range(rec_from, rec_to + 1):

            loaded_file_decay_times = self.get_loaded_file(tgui, "decay_time", "decay_time_ms", rec) / 1000
            self.assert_within_a_ts(test_decay_times[rec], loaded_file_decay_times, tgui.adata.ts)

            tabledata_decay_times = self.get_stored_tabledata(tgui, "decay_time", "decay_time_ms", rec) / 1000
            self.assert_within_a_ts(test_decay_times[rec], tabledata_decay_times, tgui.adata.ts)

            qtable_decay_times = self.get_qtable_col_for_select_recs(tgui, "decay_time", rec) / 1000
            self.assert_within_a_ts(test_decay_times[rec], qtable_decay_times, tgui.adata.ts)

            plot_decay_data = self.check_plot(tgui, False,  rec, "decay_plot", "xData",
                                              sub_data_dict=dict(type="half_rise_half_decay", data=len(test_decay_times[rec])))
            self.assert_within_a_ts(test_decay_times[rec], plot_decay_data, tgui.adata.ts)


    def test_analyse_specific_recs(self, tgui, analyse_specific_recs=True):
        """
        The other tests dont actually check if a rec outside of the analysed specific recs
        is analysed. Do that explicitly here.
        """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        for filenum in range(3):

            tgui.run_artificial_skinetics_analysis()

            for rec in range(0, rec_from):
                assert tgui.mw.loaded_file.skinetics_data[rec] == {}
                assert tgui.mw.stored_tabledata.skinetics_data[filenum][rec] == {}
                assert self.get_qtable_col_for_select_recs(tgui, "spike_number", rec, filenum).size == 0

            for rec in range(rec_to + 1, tgui.adata.num_recs):
                assert tgui.mw.loaded_file.skinetics_data[rec] == {}
                assert tgui.mw.stored_tabledata.skinetics_data[filenum][rec] == {}
                assert self.get_qtable_col_for_select_recs(tgui, "spike_number", rec, filenum).size == 0

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data("cumulative", analysis_type="skinetics")

        tgui.shutdown()

    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    def test_interp(self, tgui, analyse_specific_recs):
        """
        Test that the error is reduced following interpolations.
        """
        max_cutoff, min_cutoff = [90, 10]

        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        tgui.run_artificial_skinetics_analysis(max_cutoff=max_cutoff, min_cutoff=min_cutoff, interp=True)
        error_interp_on = self.get_rise_decay_and_fwhm_error(tgui, rec_from, rec_to, min_cutoff, max_cutoff)

        tgui.run_artificial_skinetics_analysis(max_cutoff=max_cutoff, min_cutoff=min_cutoff, interp=False)
        error_interp_off = self.get_rise_decay_and_fwhm_error(tgui, rec_from, rec_to, min_cutoff, max_cutoff)

        assert np.mean(error_interp_on["rise"]) < np.mean(error_interp_off["rise"])

    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    def test_measure_decay_time_from_peak_to_threshold_rather_than_ahp(self, tgui, analyse_specific_recs):

        max_cutoff, min_cutoff = [90, 10]

        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        tgui.run_artificial_skinetics_analysis(max_cutoff=max_cutoff, min_cutoff=min_cutoff)

        test_decay_times = tgui.adata.get_times_paramteres("decay_time", min_cutoff, max_cutoff, decay_bound="thr")
        tgui.mw.cfgs.skinetics["decay_to_thr_not_fahp"] = True
        tgui.run_artificial_skinetics_analysis(max_cutoff=max_cutoff, min_cutoff=min_cutoff)

        for rec in range(rec_from, rec_to + 1):
            qtable_decay_times = self.get_qtable_col_for_select_recs(tgui, "decay_time", rec) / 1000

            assert (abs(test_decay_times[rec] - qtable_decay_times) < tgui.adata.ts).all()

            plot_decay_data = self.check_plot(tgui, False,  rec, "decay_plot", "xData",
                                              sub_data_dict=dict(type="half_rise_half_decay", data=len(test_decay_times[rec])))
            assert (abs(test_decay_times[rec] - plot_decay_data) < tgui.adata.ts).all()

    @pytest.mark.parametrize("n_samples_rise_decay", [[2, 2], [4, 2]])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    def test_max_slope(self, tgui, analyse_specific_recs, n_samples_rise_decay):
        """
        Test that the max slope results are stored and displayed correctly.

        We cannot use more than 2 samples to test the decay because it is a sine wave and there are duplicates.
        The underlying function
        is tested elsewhere.
        """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        n_samples_rise, n_samples_decay = n_samples_rise_decay

        tgui.set_analysis_type("skinetics")

        tgui.run_artificial_skinetics_analysis(max_slope={"n_samples_rise": n_samples_rise,
                                                          "n_samples_decay": n_samples_decay})

        test_max_rise, test_max_decay, rec_rise_y_datas, rec_decay_y_datas = tgui.adata.get_max_slope(n_samples_rise=n_samples_rise,
                                                                                                      n_samples_decay=n_samples_decay)
        for rec in range(rec_from, rec_to + 1):

            max_rise = self.get_qtable_col_for_select_recs(tgui, "max_rise", rec)
            max_decay = self.get_qtable_col_for_select_recs(tgui, "max_decay", rec)

            assert utils.allclose(test_max_rise[rec], max_rise, 1e-06)
            assert utils.allclose(test_max_decay[rec], max_decay, 1e-06)

            tgui.mw.update_displayed_rec(rec)
            plot_rise_y_data = tgui.mw.loaded_file_plot.skinetics_plot_dict["max_rise_plot"].yData
            assert utils.allclose(rec_rise_y_datas[rec], plot_rise_y_data, 1e-06)

            plot_decay_y_data = tgui.mw.loaded_file_plot.skinetics_plot_dict["max_decay_plot"].yData
            assert utils.allclose(rec_decay_y_datas[rec], plot_decay_y_data,  1e-06)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Manual Selection / Deletion
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_manual_selection_and_deletion(self, tgui):
        """
        Test manual selection and deletion of spikes. Test with analse specific recs = True as this setting should be ignored by manual selection.

        First manually select some random spikes. Check that the parameters are all correct. Then delete a subset of these spikes,
        and check the results table is properly updated.
        """
        tgui.set_analysis_type("skinetics")
        tgui.handle_analyse_specific_recs(analyse_specific_recs=True)

        skinetics_keys = ["spike_time", "peak", "amplitude", "thr", "rise_time", "decay_time", "fwhm", "fahp", "mahp"]

        for filenum in range(3):

            inserted_spike_data = {}
            for key in skinetics_keys:
                inserted_spike_data[key] = []

            # Select Spikes
            for spikes_to_select in get_settings(tgui.speed, tgui.analysis_type)["manually_sel"]:

                rec, rec_spike_idx = spikes_to_select
                tgui.mw.update_displayed_rec(rec)

                tgui.manually_select_spike(rec, rec_spike_idx)
                spike_params = self.check_all_skinetics_on_a_single_spike(tgui, rec, rec_spike_idx, skinetics_keys, filenum)

                for key in skinetics_keys:
                    inserted_spike_data[key].append(spike_params[key])

            # Delete a subset of selected spikes
            for spikes_deleted, spikes_to_delete in enumerate(get_settings(tgui.speed, tgui.analysis_type)["manually_del"]):

                rec, rec_spike_idx, list_spike_idx = spikes_to_delete
                tgui.mw.update_displayed_rec(rec)
                
                tgui.click_upperplot_spotitem(tgui.mw.loaded_file_plot.skinetics_plot_dict["peak_plot"],
                                              rec_spike_idx, doubleclick_to_delete=True)

                for key in skinetics_keys:
                    inserted_spike_data[key].pop(list_spike_idx - spikes_deleted)  # iterate from early idx to higher, so delete higher idx each time one is popped

                    table_data = self.get_qtable_col_for_select_recs(tgui, key, rec=None, filenum=filenum)
                    if key in ["decay_time", "rise_time", "fwhm"]:
                        assert np.array_equal(inserted_spike_data[key], table_data)

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data("cumulative", analysis_type="skinetics")

        tgui.shutdown()

    def check_all_skinetics_on_a_single_spike(self, tgui, rec, spike_num, skinetics_keys, filenum):
        """
        Check main parameters are correct on a single spike. A little bit redundant, ideally would be integrated with
        all above util functions.
        """
        params = {}
        for param in skinetics_keys:
            params[param] = self.get_qtable_col_for_select_recs(tgui, param, rec, filenum)[0][spike_num]

        assert params["spike_time"] == tgui.adata.peak_times_[rec][spike_num]
        assert params["peak"] == tgui.adata.all_true_peaks[rec][spike_num]
        assert params["amplitude"] == tgui.adata.all_true_peaks[rec][spike_num] - tgui.adata.resting_vm
        assert params["thr"] == tgui.adata.resting_vm

        test_decay_time = tgui.adata.get_times_paramteres("decay_time", min_cutoff=10, max_cutoff=90)[rec][spike_num]
        self.assert_within_a_ts(params["decay_time"] / 1000, test_decay_time, tgui.adata.ts)

        test_rise_time = tgui.adata.get_times_paramteres("rise_time", max_cutoff=10, min_cutoff=90)[rec][spike_num]
        self.assert_within_a_ts(params["rise_time"] / 1000, test_rise_time, tgui.adata.ts)

        test_half_width = tgui.adata.get_times_paramteres("half_width")[rec][spike_num]
        self.assert_within_a_ts(params["fwhm"] / 1000, test_half_width, tgui.adata.ts)

        return params

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# fAHP and mAHP more thorough test
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("fahp_start_len_ms", [[1, 4], [2, 23],  [0, 20]])
    @pytest.mark.parametrize("mahp_start_len_ms", [[20, 9], [5, 20],  [0, 30]])
    def test_fahp_and_ahp_timings(self, tgui, analyse_specific_recs, fahp_start_len_ms, mahp_start_len_ms):
        """
        Test fAHP and mAHP while setting random regions to search for the ahp within. To test, index out the same
        period of the test data and take the minimum, ensuring they match.
        """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        [fahp_start_ms, fahp_len_ms] = fahp_start_len_ms
        [mahp_start_ms, mahp_len_ms] = mahp_start_len_ms

        tgui.set_analysis_type("skinetics")

        tgui.left_mouse_click(tgui.mw.mw.skinetics_options_button)

        tgui.set_skinetics_ahp_spinboxes(tgui.mw.dialogs["skinetics_options"].dia,
                                         fahp_start=fahp_start_ms, fahp_stop=fahp_start_ms + fahp_len_ms,
                                         mahp_start=mahp_start_ms, mahp_stop=mahp_start_ms + mahp_len_ms)

        tgui.run_artificial_skinetics_analysis()

        for rec in range(rec_from, rec_to + 1):

            spike_peaks_idx = tgui.clean(tgui.adata.all_true_peaks_idx[rec])

            test_fahp = self.index_out_ahp_regions(tgui, spike_peaks_idx, rec, fahp_start_ms, fahp_len_ms, tgui.adata.fs)
            test_mahp = self.index_out_ahp_regions(tgui, spike_peaks_idx, rec, mahp_start_ms, mahp_len_ms, tgui.adata.fs)

            loaded_file_fahp = self.get_loaded_file(tgui, "fahp", "vm", rec)

            loaded_file_mahp = self.get_loaded_file(tgui, "mahp", "vm", rec)

            assert np.array_equal(test_fahp, loaded_file_fahp)
            assert np.array_equal(test_mahp, loaded_file_mahp)

    @staticmethod
    def index_out_ahp_regions(tgui, spike_peaks_idx, rec, ahp_start_ms, ahp_len_ms, fs):
        """
        We are given the ahp start time and length of search period in ms. First convert this to indices,
        then index out the relevant portions of the data into a spike x ahp search period matrix. Take the minimum
        for each row to get the true ahp for each spike.
        """
        samples_per_ms = fs / 1000

        ahp_start_samples = spike_peaks_idx + np.round(ahp_start_ms * samples_per_ms).astype(np.int32)
        ahp_region_len = np.round(ahp_len_ms * samples_per_ms).astype(np.int32)

        idx = np.tile(np.arange(ahp_region_len + 1), (len(ahp_start_samples), 1)) + np.atleast_2d(ahp_start_samples).T
        idx = idx.astype(np.int32)

        vms = tgui.adata.vm_array[rec][idx]

        return np.min(vms, axis=1)

# ----------------------------------------------------------------------------------------------------------------------------------------------
# Phase Plot Analysis Tests
# ----------------------------------------------------------------------------------------------------------------------------------------------

# Test Scrolling -------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    def test_phase_plot_spike_scrolling(self, tgui, analyse_specific_recs):
        """
        Scroll through all spikes and check the correct number is there and they are not skipped. Run only with analyse / not analyse specific recs.
        Bouunds etc. are tested below
        """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        for filenum in range(3):

            tgui.run_artificial_skinetics_analysis(run_with_bounds=False, spike_detection_method="manual", override_mahp_fahp_defaults=True, first_deriv_cutoff=10)

            tgui.left_mouse_click(tgui.mw.mw.phase_space_plots_button)
            dialog = tgui.mw.dialogs["phase_space_analysis"]

            # Repeatedly push left button to ensure it does not go further
            # than existing spike
            for press in range(50):
                tgui.left_mouse_click(dialog.dia.scroll_left_button)
                assert dialog.dia.record_spinbox.value() == rec_from + 1
                assert dialog.dia.action_potential_spinbox.value() == 1

            # start by cycling through each spike,
            # checking the record and AP box are correct
            for rec_idx in range(rec_from, rec_to + 1):
                num_spikes_in_rec = sum(~np.isnan(tgui.adata.peak_times_[rec_idx]))
                for spike_idx in range(num_spikes_in_rec):
                    self.press_right_or_left_button_and_check(tgui, dialog, rec_idx, spike_idx, "right")

            # test right to ensure does not go further than existing spike
            for press in range(50):
                tgui.left_mouse_click(dialog.dia.scroll_right_button)
                assert dialog.dia.record_spinbox.value() == rec_to + 1
                assert dialog.dia.action_potential_spinbox.value() == sum(~np.isnan(tgui.adata.peak_times_[rec_to]))

            # Now cycle back,
            for rec_idx in reversed(range(rec_from, rec_to + 1)):
                num_spikes_in_rec = sum(~np.isnan(tgui.adata.peak_times_[rec_idx]))
                for spike_idx in reversed(range(num_spikes_in_rec)):
                    self.press_right_or_left_button_and_check(tgui, dialog, rec_idx, spike_idx, "left")

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data("cumulative", analysis_type="skinetics")

        tgui.shutdown()

    def press_right_or_left_button_and_check(self, tgui, dialog, rec_idx, spike_idx, left_or_right):
        button = dialog.dia.scroll_left_button if left_or_right == "left" else dialog.dia.scroll_right_button
        assert dialog.dia.record_spinbox.value() == rec_idx + 1
        assert dialog.dia.action_potential_spinbox.value() == spike_idx + 1
        tgui.left_mouse_click(button)
        QtWidgets.QApplication.processEvents()

# Test All Analysis ---------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("run_with_bounds", [True, False])
    @pytest.mark.parametrize("interp", [True, False])
    def test_phase_plot_analysis(self, tgui, run_with_bounds, analyse_specific_recs, interp):
        """
        Run tests of many aspects of the phase-plot analysis. These are combined to reduce overhead with tear down and avoid duplicate code.

        First, run an skinetics analysis (either with / without bounds, restricted / not restricted). Check that the data displayed on the AP
        plot on the phase space dialog matches the true data for all AP. Check that the Vm / dVM of this signal matches what is shown
        on the phase space plot. Check all parameters are calculated correctly. Then the copied data (both from the plot and the copy data button).
        Finally check the save all data (every file).
        """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        for filenum in range(3):

            bounds_vm = tgui.run_artificial_skinetics_analysis(run_with_bounds=run_with_bounds, spike_detection_method="manual", override_mahp_fahp_defaults=True, first_deriv_cutoff=10)  # TODO: own function
            dialog, window_samples = self.setup_phase_space_dialog_and_set_windows_samples(tgui)

            if interp:
                tgui.switch_checkbox(dialog.dia.cubic_spline_interpolate_checkbox, on=True)

            if bounds_vm:
                test_spike_times = self.get_spiketimes_within_bounds(tgui, rec_from, rec_to, bounds_vm)
            else:
                test_spike_times = tgui.adata.peak_times_

            # start by cycling through each spike, checking the record and AP box are correct
            all_res = []
            for rec_idx in range(rec_from, rec_to + 1):
                for ap_idx, peak_time in enumerate(test_spike_times[rec_idx]):

                    if np.isnan(peak_time):
                        continue

                    peak_idx = self.peak_time_to_idx(tgui, rec_idx, peak_time)

                    res = self.check_and_return_all_spike_and_phase_plot_data(tgui, dialog, rec_idx, ap_idx, peak_idx, peak_time, window_samples, interp=interp)

                    test_headers = self.check_copied_data(tgui, dialog, rec_idx, ap_idx, res)

                    all_res.append([rec_idx, ap_idx, res])

                    QtWidgets.QApplication.processEvents()
                    tgui.left_mouse_click(dialog.dia.scroll_right_button)

            if not interp:
                self.check_save_data(tgui, dialog, all_res, test_headers, "excel")  # too big if interp

            self.check_save_data(tgui, dialog, all_res, test_headers, "csv")

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data("cumulative", analysis_type="skinetics")

        del dialog
        tgui.shutdown()

    def test_scrolling_left_sanity_check(self, tgui):
        """
        As a quick sanity check, do the same as above (without checking save data) but now scrolling
        using the left rather than right scroll.
        """
        dialog, window_samples = self.setup_for_phase_plot_tests(tgui)

        spikes_in_last_rec = sum(~np.isnan(tgui.adata.peak_times_[tgui.adata.num_recs - 1]))
        self.set_phase_rec_and_ap(tgui, dialog, tgui.adata.num_recs - 1, spikes_in_last_rec - 1)

        # start by cycling through each spike, checking the record and AP box are correct
        for rec_idx in reversed(range(tgui.adata.num_recs)):

            ap_idx = len(tgui.adata.peak_times_[rec_idx])
            for peak_time in reversed(tgui.adata.peak_times_[rec_idx]):  # cant reverse enuemrate object
                ap_idx -= 1
                if np.isnan(peak_time):
                    continue

                peak_idx = self.peak_time_to_idx(tgui, rec_idx, peak_time)

                res = self.check_and_return_all_spike_and_phase_plot_data(tgui, dialog, rec_idx, ap_idx, peak_idx, peak_time, window_samples, interp=False)

                self.check_copied_data(tgui, dialog, rec_idx, ap_idx, res)

                QtWidgets.QApplication.processEvents()
                tgui.left_mouse_click(dialog.dia.scroll_left_button)

# Testing functions -----------------------------------------------------------------------------------------------------------------------------------------

    def check_and_return_all_spike_and_phase_plot_data(self, tgui, dialog, rec_idx, ap_idx, peak_idx, peak_time, window_samples, interp, use_adata=True):
        """
        """
        file_data = tgui.adata if use_adata else tgui.mw.loaded_file.data
        spike_x = file_data.time_array[rec_idx][peak_idx - window_samples:peak_idx + window_samples + 1] - file_data.time_array[rec_idx][peak_idx - window_samples]
        spike_y = file_data.vm_array[rec_idx][peak_idx - window_samples:peak_idx + window_samples + 1]

        # Test AP data and plot
        assert utils.allclose(spike_x, dialog.time)
        assert utils.allclose(spike_y, dialog.data)

        assert np.array_equal(spike_x * 1000, dialog.ap_plot.xData)
        assert np.array_equal(spike_y, dialog.ap_plot.yData)

        test_vm, test_vm_diff, \
            test_threshold_vm, test_threshold_vdiff, test_vm_max_vm, test_vm_max_vmdiff, \
            test_dvm_min_vm, test_dvm_min_vmdiff, test_dvm_max_vm, test_dvm_max_vmdiff = self.check_phase_plot_vm_vm_diff_and_all_params(tgui, dialog, spike_y, rec_idx, ap_idx, interp, use_adata)

        res = SimpleNamespace(spike_time=peak_time, spike_x=spike_x, spike_y=spike_y, test_vm=test_vm, test_vm_diff=test_vm_diff, test_threshold_vm=test_threshold_vm,
                              test_threshold_vdiff=test_threshold_vdiff, test_vm_max_vm=test_vm_max_vm, test_vm_max_vmdiff=test_vm_max_vmdiff,
                              test_dvm_min_vm=test_dvm_min_vm, test_dvm_min_vmdiff=test_dvm_min_vmdiff, test_dvm_max_vm=test_dvm_max_vm, test_dvm_max_vmdiff=test_dvm_max_vmdiff)
        return res

    def check_save_data(self, tgui, dialog, all_res, test_headers, csv_or_excel):
        """
        Check the save data CSV or excel. When this button is pressed, all AP in the file are analysed
        and saved in csv / excel. Here use all_res ('all_results') which is the test data generated by
        looping through all APs. We compare this to the saved data.

        the dialog.save_data() function has a file overwrite option to avoid having to use keyboard to
        write into the file saving dialog.
        """
        file_path = tgui.test_base_dir + "/test_save_phase_analysis"

        if csv_or_excel == "csv":
            filename = (file_path + ".csv", 'CSV (*.csv)')
            load_func = pd.read_csv
        elif csv_or_excel == "excel":
            filename = (file_path + ".xlsx", 'Excel (*.xlsx)')
            load_func = pd.read_excel

        if os.path.isfile(filename[0]):
            os.remove(filename[0])

        dialog.save_data(filename=filename)

        tgui.wait_for_other_thread(30)
        test_saved_data = load_func(filename[0])

        assert np.shape(test_saved_data)[0] == len(all_res)
        assert test_saved_data.columns.to_list() == test_headers

        for row_idx, res in enumerate(all_res):  # loop through every AP in the file and check against test
            spike_info = test_saved_data.iloc[row_idx, :].to_numpy()
            test_spike_info = self.res_to_spike_info(rec_idx=res[0], ap_idx=res[1], res=res[2])
            assert utils.allclose(spike_info, test_spike_info)

    def check_copied_data(self, tgui, dialog, rec_idx, ap_idx, res):
        """
        Check data copied from both the copy data button (includes all data / params) and the phase plot
        (copies parameters only). Check the headers and data are correct.
        """
        first_headers = ["Record", "Spike", "Spike Time (s)"]
        last_headers = ["Threshold Vm (mV)"] + ["Threshold dVm (mV/ms)"] + ["Vm Max Vm (mV)"] + ["Vm Max dVm (mV/ms)"] + ["dVm Max Vm (mV)"] + ["dVm Max dVm (mV/ms)"] + ["dVm Min Vm (mV)"] + ["dVm Min dVm (mV/ms)"]

        # Test the main copy button
        tgui.left_mouse_click(dialog.dia.copy_data_button)
        copied_data = pd.read_clipboard(header=None)

        test_copied_data = self.res_to_spike_info(rec_idx, ap_idx, res)

        test_headers = first_headers + \
                       ["Spike Timepoint (s) " + str(idx) for idx in range(len(res.spike_x))] + ["Spike Vm (mV) " + str(idx) for idx in range(len(res.spike_y))] + \
                       ["Phase Vm (mV) " + str(idx) for idx in range(len(res.test_vm))] + ["Phase dVm (mV/ms) " + str(idx) for idx in range(len(res.test_vm_diff))] \
                       + last_headers

        assert copied_data.iloc[:, 0].to_list() == test_headers
        assert utils.allclose(copied_data.iloc[:, 1].to_numpy(), test_copied_data)

        # Also test the copy button on the phase plot
        self.check_copy_phase_plot_params(dialog, rec_idx, ap_idx, res, first_headers + last_headers)

        return test_headers  # used for saved data test

    def check_copy_phase_plot_params(self, dialog, rec_idx, ap_idx, res, plot_copy_headers):
        """"""
        dialog.copy_plot_data_action.trigger()
        copied_data = pd.read_clipboard(header=None)

        assert copied_data.iloc[:, 0].to_list() == plot_copy_headers
        assert utils.allclose(copied_data.iloc[:, 1], self.res_to_spike_info(rec_idx, ap_idx, res, parameters_only=True))

    def check_phase_plot_vm_vm_diff_and_all_params(self, tgui, dialog, spike_y, rec_idx, ap_idx, interp, use_adata=True):
        """
        Check that the data on the phase dialog plot matches vm / vm_diff data calculated here from the true data (spike_y)
        """
        test_vm, test_vm_diff = self.calculate_test_vm_and_vm_diff(spike_y, tgui.mw.loaded_file.data.ts)

        if interp:
            test_vm_diff = core_analysis_methods.interpolate_data(test_vm_diff, np.arange(len(test_vm)), "cubic", 100, 0)
            test_vm = core_analysis_methods.interpolate_data(test_vm, np.arange(len(test_vm)), "cubic", 100, 0)

        assert np.array_equal(test_vm, dialog.vm)
        assert np.array_equal(test_vm_diff, dialog.vm_diff)

        assert np.array_equal(test_vm, dialog.phase_plot.xData)
        assert np.array_equal(test_vm_diff, dialog.phase_plot.yData)

        test_threshold_vm, test_threshold_vdiff, test_vm_max_vm, test_vm_max_vmdiff, \
        test_dvm_min_vm, test_dvm_min_vmdiff, test_dvm_max_vm, test_dvm_max_vmdiff = self.get_and_test_phase_plot_parameters(tgui, rec_idx, ap_idx, dialog, test_vm, test_vm_diff, interp, use_adata)

        return test_vm, test_vm_diff, \
               test_threshold_vm, test_threshold_vdiff, test_vm_max_vm, test_vm_max_vmdiff, \
               test_dvm_min_vm, test_dvm_min_vmdiff, test_dvm_max_vm, test_dvm_max_vmdiff


    def get_and_test_phase_plot_parameters(self, tgui, rec_idx, ap_idx, dialog, test_vm, test_vm_diff, interp, use_adata=True):
        """
        Test all phase-plot parameters against those shown on the plot. Return them so they can be compared in the copy / save tests.
        """
        # threshold
        idx = np.min(np.nonzero(test_vm_diff > dialog.threshold_cutoff)[0])
        test_threshold_vm = test_vm[idx]
        test_threshold_vdiff = test_vm_diff[idx]

        assert test_threshold_vm == dialog.scatter_threshold.xData
        assert test_threshold_vdiff == dialog.scatter_threshold.yData

        if not interp and use_adata:  # interp will not match real data as not interp
            assert test_threshold_vm == tgui.adata.vm_array[rec_idx][int(tgui.adata.spike_sample_idx[rec_idx][ap_idx])]

        # Vm max
        idx = np.argmax(test_vm)
        test_vm_max_vm = test_vm[idx]
        test_vm_max_vmdiff = test_vm_diff[idx]
        assert dialog.scatter_vmax.xData == test_vm_max_vm
        assert dialog.scatter_vmax.yData == test_vm_max_vmdiff

        # dVm min
        idx = np.argmin(test_vm_diff)
        test_dvm_min_vm = test_vm[idx]
        test_dvm_min_vmdiff = test_vm_diff[idx]
        assert dialog.scatter_vdiff_min.xData == test_dvm_min_vm
        assert dialog.scatter_vdiff_min.yData == test_dvm_min_vmdiff

        # dVm max
        idx = np.argmax(test_vm_diff)
        test_dvm_max_vm = test_vm[idx]
        test_dvm_max_vmdiff = test_vm_diff[idx]
        assert dialog.scatter_vdiff_max.xData == test_dvm_max_vm
        assert dialog.scatter_vdiff_max.yData == test_dvm_max_vmdiff

        return test_threshold_vm, test_threshold_vdiff, test_vm_max_vm, test_vm_max_vmdiff, \
               test_dvm_min_vm, test_dvm_min_vmdiff, test_dvm_max_vm, test_dvm_max_vmdiff

# Test Phase Utils ------------------------------------------------------------------------------------------------------------------------------

    def peak_time_to_idx(self, tgui, rec_idx, peak_time):
        """
        Convert the time of a peak to its index in the data
        """
        peak_time = peak_time - tgui.mw.loaded_file.data.time_array[rec_idx][0]
        peak_idx = core_analysis_methods.quick_get_time_in_samples(tgui.mw.loaded_file.data.ts, peak_time).astype('int')
        return peak_idx

    def res_to_spike_info(self, rec_idx, ap_idx, res, parameters_only=False):
        """
        Convert the res results output of check_and_return_all_spike_and_phase_plot_data() to an array
        so it can be copied with copied data.
        """
        if parameters_only:
            return np.hstack([rec_idx + 1, ap_idx + 1, res.spike_time,
                              res.test_threshold_vm, res.test_threshold_vdiff, res.test_vm_max_vm, res.test_vm_max_vmdiff,
                              res.test_dvm_max_vm, res.test_dvm_max_vmdiff, res.test_dvm_min_vm, res.test_dvm_min_vmdiff])
        else:
            return np.hstack([rec_idx + 1, ap_idx + 1, res.spike_time,
                              res.spike_x, res.spike_y, res.test_vm, res.test_vm_diff, res.test_threshold_vm, res.test_threshold_vdiff,
                              res.test_vm_max_vm, res.test_vm_max_vmdiff, res.test_dvm_max_vm, res.test_dvm_max_vmdiff, res.test_dvm_min_vm, res.test_dvm_min_vmdiff])

    def calculate_test_vm_and_vm_diff(self, spike_y, ts):
        """"""
        test_vm = spike_y[:-1]
        test_vm_diff = np.diff(spike_y) / (ts * 1000)
        return test_vm, test_vm_diff

    def set_and_get_phase_analysis_window_samples(self, tgui, dialog, window_ms):
        """"""
        tgui.enter_number_into_spinbox(dialog.dia.window_size_spinbox_left,
                                       window_ms)
        tgui.enter_number_into_spinbox(dialog.dia.window_size_spinbox_right,
                                       window_ms)
        window_samples = round(window_ms / (tgui.mw.loaded_file.data.ts * 1000))

        return window_samples

    def setup_for_phase_plot_tests(self, tgui, run_with_bounds=False, spike_detection_method="auto", override_mahp_fahp_defaults=True, first_deriv_cutoff=10, thr_search_region=2, window_ms=2):
        """"""
        tgui.run_artificial_skinetics_analysis(run_with_bounds=run_with_bounds, spike_detection_method=spike_detection_method, override_mahp_fahp_defaults=override_mahp_fahp_defaults,
                                               first_deriv_cutoff=first_deriv_cutoff, thr_search_region=thr_search_region)  # TODO: own function
        dialog, window_samples = self.setup_phase_space_dialog_and_set_windows_samples(tgui, window_ms)

        return dialog, window_samples

    def setup_phase_space_dialog_and_set_windows_samples(self, tgui, windows_ms=1.5):
        """"""
        tgui.left_mouse_click(tgui.mw.mw.phase_space_plots_button)
        dialog = tgui.mw.dialogs["phase_space_analysis"]
        window_samples = self.set_and_get_phase_analysis_window_samples(tgui, dialog, window_ms=windows_ms)
        return dialog, window_samples

# Test Graph Viewing Options ------------------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("window_ms", [1.5, 1])
    def test_analysis_random_window_sizes_and_spikes(self, tgui, window_ms):
        """
        Do a less thorough test (not testing save data here) testing randomly selected APs (i.e. in a random order)
        just to check in case of any issues that were not observed just scrolling every AP. Also, test with a few different
        window sizes just in case (these must always be small enough to only include 1 AP).
        """
        if tgui.time_type == "cumulative":
            return

        for filenum in range(3):

            dialog, window_samples = self.setup_for_phase_plot_tests(tgui, window_ms=window_ms)

            for __ in range(20):
                rec_idx, ap_idx, peak_time = self.get_random_rec_and_ap_idx_and_set_on_phase_dialog(tgui, dialog)

                peak_idx = self.peak_time_to_idx(tgui, rec_idx, peak_time)

                res = self.check_and_return_all_spike_and_phase_plot_data(tgui, dialog, rec_idx, ap_idx, peak_idx, peak_time, window_samples, interp=False)

                self.check_copied_data(tgui, dialog, rec_idx, ap_idx, res)

                QtWidgets.QApplication.processEvents()

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data("cumulative", analysis_type="skinetics")
        tgui.shutdown()

    def test_phase_plots_line_thickness(self, tgui):
        """
        Test the line thickness on the plot changes when the spinbox is changed.
        """
        dialog, __ = self.setup_for_phase_plot_tests(tgui)

        assert dialog.ap_plot.opts["pen"].widthF() == 1
        assert dialog.phase_plot.opts["pen"].widthF() == 1

        tgui.enter_number_into_spinbox(dialog.dia.pen_width_spinbox, 0.5)
        assert dialog.ap_plot.opts["pen"].widthF() == 0.5
        assert dialog.phase_plot.opts["pen"].widthF() == 0.5

        tgui.enter_number_into_spinbox(dialog.dia.pen_width_spinbox, 1.2)
        assert dialog.ap_plot.opts["pen"].widthF() == 1.2
        assert dialog.phase_plot.opts["pen"].widthF() == 1.2

    def test_phase_plots_grid(self, tgui):
        """
        Check the plot gridlines change when the button is click / unclick.
        """
        dialog, __ = self.setup_for_phase_plot_tests(tgui)

        self.check_grid_on_or_off(dialog, on=True)

        tgui.switch_checkbox(dialog.dia.show_grid_checkbox, on=False)

        self.check_grid_on_or_off(dialog, on=False)

        tgui.switch_checkbox(dialog.dia.show_grid_checkbox, on=True)

        self.check_grid_on_or_off(dialog, on=True)

    def check_grid_on_or_off(self, dialog, on):
        """"""
        val = 80 if on else 0

        ap_plot = dialog.action_potential_plot_class.plot
        phase_plot = dialog.phase_plot_class.plot

        assert ap_plot.getAxis("left").grid == val
        assert ap_plot.getAxis("bottom").grid == val
        assert phase_plot.getAxis("left").grid == val
        assert phase_plot.getAxis("bottom").grid == val

    def get_random_rec_and_ap_idx_and_set_on_phase_dialog(self, tgui, dialog):
        """
        Get a randomly selected AP and change the phase plot rec / ap spinboxes
        to select it
        """
        rec_idx = np.random.randint(low=0, high=tgui.adata.num_recs)
        num_spikes_in_rec = len(tgui.mw.loaded_file.skinetics_data[rec_idx])
        ap_idx = np.random.randint(low=0, high=num_spikes_in_rec)

        rec_peak_times = list(tgui.mw.loaded_file.skinetics_data[rec_idx].keys())
        peak_time = float(rec_peak_times[ap_idx])

        self.set_phase_rec_and_ap(tgui, dialog, rec_idx, ap_idx)

        return rec_idx, ap_idx, peak_time


# Threshold --------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("threshold_cutoff", [10, 20, 30])
    def test_threshold_matches_raw_data(self, tgui, threshold_cutoff):
        """
        Another check on the threshold calculated in the AP phase plot. Scroll though all AP and check the
        threshold as calculated from the phase plot matches the threshold calculated using the standard
        method (if first deriv threshold is used, these should match exactly).
        """
        tgui.load_a_filetype("current_clamp")

        dialog, window_samples = self.setup_for_phase_plot_tests(tgui, first_deriv_cutoff=threshold_cutoff)

        tgui.enter_number_into_spinbox(dialog.dia.threshold_cutoff_spinbox, threshold_cutoff)

        for rec_idx in range(tgui.mw.loaded_file.data.num_recs):

            if tgui.mw.loaded_file.skinetics_data[rec_idx] in [0, []]:
                continue

            for ap_idx in range(len(tgui.mw.loaded_file.skinetics_data[rec_idx].keys())):

                rec_data = tgui.mw.loaded_file.skinetics_data[rec_idx]

                # test threshold matches analysis
                analysis_thr = rec_data[list(rec_data.keys())[ap_idx]]["thr"]["vm"]
                assert analysis_thr == dialog.scatter_threshold.xData

                # test Vm max (i.e. peak) matches analysis
                analysis_peak = rec_data[list(rec_data.keys())[ap_idx]]["peak"]["vm"]
                assert analysis_peak == dialog.scatter_vmax.xData

                QtWidgets.QApplication.processEvents()
                tgui.left_mouse_click(dialog.dia.scroll_right_button)

# Test Phase Plots with Manual Selection / Deletion ----------------------------------------------------------------------------------------------- TODO: also test re-analysis

    def test_manual_deleting_spikes(self, tgui):
        """
        Check the phase plot dialog propery updates when a AP is manually deleted. Delete spikes and check GUI
        is properly updated.
        """
        tgui.load_a_filetype("current_clamp")
        dialog, window_samples = self.setup_for_phase_plot_tests(tgui)

        rec_idx = 4
        ap_idx = 3

        tgui.mw.update_displayed_rec(rec_idx)
        self.set_phase_rec_and_ap(tgui, dialog, rec_idx, ap_idx)

        # get all spike times
        rec_4_times = np.sort(list(tgui.mw.loaded_file.skinetics_data[rec_idx].keys())).astype(np.float64)

        # check 4
        peak_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx, ap_idx, window_samples)
        assert peak_time == rec_4_times[ap_idx]

        # del 4
        tgui.click_upperplot_spotitem(tgui.mw.loaded_file_plot.skinetics_plot_dict["peak_plot"], ap_idx, doubleclick_to_delete=True)

        # check now selected spike is next spike along
        peak_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx, ap_idx, window_samples)
        assert peak_time == rec_4_times[ap_idx + 1]

        # scroll left 1 and check spike is 1 before deleted spike
        tgui.left_mouse_click(dialog.dia.scroll_left_button)
        peak_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx, ap_idx - 1, window_samples)
        assert peak_time == rec_4_times[ap_idx - 1]

        # delete to end and check last spike is shown
        for __ in range(10):  # delete all spikes until only 1 left
            tgui.click_upperplot_spotitem(tgui.mw.loaded_file_plot.skinetics_plot_dict["peak_plot"], 0, doubleclick_to_delete=True)

        peak_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx, 0, window_samples)
        assert peak_time == rec_4_times[-1]

        # scroll to next re, check and delete the first spike
        tgui.left_mouse_click(dialog.dia.scroll_right_button)
        assert dialog.dia.record_spinbox.value() == rec_idx + 2
        assert dialog.dia.action_potential_spinbox.value() == 1

        rec_5_times = np.sort(list(tgui.mw.loaded_file.skinetics_data[rec_idx + 1].keys())).astype(np.float64)
        peak_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx + 1, 0, window_samples)
        assert peak_time == rec_5_times[0]

        tgui.mw.update_displayed_rec(rec_idx + 1)
        tgui.click_upperplot_spotitem(tgui.mw.loaded_file_plot.skinetics_plot_dict["peak_plot"], 0, doubleclick_to_delete=True)
        peak_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx + 1, 0, window_samples)
        assert peak_time == rec_5_times[1]

        # now go to the record before, check and delete the last spike
        tgui.left_mouse_click(dialog.dia.scroll_left_button)
        tgui.left_mouse_click(dialog.dia.scroll_left_button)

        # check the last spike
        rec_3_times = np.sort(list(tgui.mw.loaded_file.skinetics_data[rec_idx - 1].keys())).astype(np.float64)
        assert dialog.dia.record_spinbox.value() == rec_idx  # rec_idx = rec_num - 1 (i.e. the num of the rec before)
        assert dialog.dia.action_potential_spinbox.value() == len(rec_3_times)

        peak_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx -1, len(rec_3_times) -1, window_samples)
        assert peak_time == rec_3_times[-1]

        # delete the last spike
        tgui.mw.update_displayed_rec(rec_idx - 1)
        tgui.click_upperplot_spotitem(tgui.mw.loaded_file_plot.skinetics_plot_dict["peak_plot"], len(rec_3_times) - 1, doubleclick_to_delete=True)
        peak_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx - 1, 0, window_samples) # auto goes to first spike
        assert peak_time == rec_3_times[0]

        # how go back, check and delete the final spike
        tgui.mw.update_displayed_rec(rec_idx)
        self.set_phase_rec_and_ap(tgui, dialog, rec_idx, 0)

        peak_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx, 0, window_samples)  # auto goes to first spike
        assert peak_time == rec_4_times[-1]
        tgui.click_upperplot_spotitem(tgui.mw.loaded_file_plot.skinetics_plot_dict["peak_plot"], 0, doubleclick_to_delete=True)

        assert dialog.dia.record_spinbox.value() == rec_idx  # rec_idx = rec_num - 1 (i.e. the num of the rec before)
        assert dialog.dia.action_potential_spinbox.value() == 1
        self.quick_check_displayed_spike(tgui, dialog, rec_idx - 1, 0, window_samples)  # auto goes to first spike

    def test_manual_deleting_all_spikes(self, tgui):
        """
        Delete evry single spike and check the GUI shows warning and closes
        """
        tgui.load_a_filetype("current_clamp")

        dialog, window_samples = self.setup_for_phase_plot_tests(tgui)

        for rec, rec_data in enumerate(tgui.mw.loaded_file.skinetics_data):
            tgui.mw.update_displayed_rec(rec)

            if rec_data in [0, {}]:
                continue
            else:
                num_spikes = len(rec_data)

            for ap_idx in range(num_spikes):
                if rec == len(tgui.mw.loaded_file.skinetics_data) - 1 and ap_idx == num_spikes - 1: # stop before the very last spike
                    break
                tgui.click_upperplot_spotitem(tgui.mw.loaded_file_plot.skinetics_plot_dict["peak_plot"], 0, doubleclick_to_delete=True)
                if ap_idx != num_spikes - 1:
                    self.quick_check_displayed_spike(tgui, dialog, rec, 0, window_samples)  # cant test when all spieks in rec deleted

        QtCore.QTimer.singleShot(2500, lambda: self.check_last_spike_messagebox(tgui))
        tgui.click_upperplot_spotitem(tgui.mw.loaded_file_plot.skinetics_plot_dict["peak_plot"], 0, doubleclick_to_delete=True)

        assert tgui.mw.dialogs["phase_space_analysis"] is None

    def test_manually_selecting_spikes(self, tgui):
        """
        Manually select a spike and check the GUI updates properly
        """
        tgui.load_a_filetype("current_clamp")

        # restrict to certain recs so that an AP outside of analysed range can be selected
        tgui.analysis_type = "None"
        tgui.rec_from_value = 4
        tgui.rec_to_value = 5
        tgui.handle_analyse_specific_recs(analyse_specific_recs=True)

        dialog, window_samples = self.setup_for_phase_plot_tests(tgui)

        # select a spike, check and delete it.
        rec_idx = 5
        ap_idx = 0
        tgui.enter_number_into_spinbox(dialog.dia.record_spinbox, rec_idx + 1)
        tgui.mw.update_displayed_rec(rec_idx)

        rec_4_times = np.sort(list(tgui.mw.loaded_file.skinetics_data[rec_idx].keys())).astype(np.float64)
        del_spike_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx, ap_idx, window_samples)  # cant test when all spieks in rec deleted
        del_spike_peak = tgui.mw.loaded_file.skinetics_data[rec_idx][str(del_spike_time)]["peak"]["vm"]

        tgui.click_upperplot_spotitem(tgui.mw.loaded_file_plot.skinetics_plot_dict["peak_plot"], 0, doubleclick_to_delete=True)

        # check the spike has updated properly, then select the deleted spike.
        spike_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx, ap_idx, window_samples)
        assert spike_time == rec_4_times[1]

        tgui.expand_xaxis_around_peak(tgui, spike_time)
        tgui.left_mouse_click(tgui.mw.mw.skinetics_click_mode_button)
        tgui.select_spike_action(del_spike_time, del_spike_peak)

        # check the re-selected spike
        spike_time = self.quick_check_displayed_spike(tgui, dialog, rec_idx, ap_idx, window_samples)
        assert spike_time == rec_4_times[0]
        assert spike_time == del_spike_time

        # select a spike on an un-analysed rec and check it is shown.
        empty_rec = 6
        tgui.mw.update_displayed_rec(empty_rec)
        tgui.select_spike_action(180.5049, 56.427001953125)  # hard coded from known file

        self.set_phase_rec_and_ap(tgui, dialog, 6, 0)
        self.quick_check_displayed_spike(tgui, dialog, 6, 0, window_samples)

    def test_whole_file_window(self, tgui):
        """
        Set the window so large it shows the entire file. Check the data matches.
        """
        dialog, __ = self.setup_for_phase_plot_tests(tgui)

        self.set_and_get_phase_analysis_window_samples(tgui, dialog, window_ms=10000)

        assert np.array_equal(tgui.mw.loaded_file.data.time_array[0], dialog.time)
        assert np.array_equal(tgui.mw.loaded_file.data.vm_array[0], dialog.data)
        assert np.array_equal(tgui.mw.loaded_file.data.time_array[0] * 1000, dialog.ap_plot.xData)
        assert np.array_equal(tgui.mw.loaded_file.data.vm_array[0], dialog.ap_plot.yData)

        test_vm, test_vm_diff = self.calculate_test_vm_and_vm_diff(dialog.data, tgui.mw.loaded_file.data.ts)

        assert np.array_equal(test_vm, dialog.vm)
        assert np.array_equal(test_vm_diff, dialog.vm_diff)
        assert np.array_equal(test_vm, dialog.phase_plot.xData)
        assert np.array_equal(test_vm_diff, dialog.phase_plot.yData)

# Manually select spike phase plot test utils -----------------------------------------------------------------------------------------------------------------

    def check_last_spike_messagebox(self, tgui):
        """ errors from these functions do not propagate! If assert is FAlse need to break in
        BEFORE assert and change it so correct. Otherwise hangs without closing messagebox"""
        assert tgui.mw.messagebox.text() == "<p align='center'>There are no analysed action potentials. Phase-plot window will now close.</p>"
        tgui.mw.messagebox.close()

    def quick_check_displayed_spike(self, tgui, dialog, rec_idx, ap_idx, window_samples):
        peak_time = float(list(tgui.mw.loaded_file.skinetics_data[rec_idx].keys())[ap_idx])
        peak_idx = self.peak_time_to_idx(tgui, rec_idx, peak_time)
        self.check_and_return_all_spike_and_phase_plot_data(tgui, dialog, rec_idx, ap_idx, peak_idx, peak_time, window_samples, interp=False, use_adata=False)
        return peak_time

    def set_phase_rec_and_ap(self, tgui, dialog, rec_idx, ap_idx):
        tgui.enter_number_into_spinbox(dialog.dia.record_spinbox, rec_idx + 1)
        tgui.enter_number_into_spinbox(dialog.dia.action_potential_spinbox, ap_idx + 1)
