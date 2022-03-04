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
from ephys_data_methods import current_calc
from utils import utils
from setup_test_suite import GuiTestSetup
import utils_for_testing as test_utils
os.environ["PYTEST_QT_API"] = "pyside2"

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

    def check_skinetics_threshold(self, tgui, analyse_specific_recs, bounds_vm, filenum):
        """
        Check the skinetics threshold s in loaded_file, stored_tabledata and qtTable on the plot all match the test data.
        """
        test_spike_sample_idx, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs,
                                                                                    tgui.adata.spike_sample_idx)

        for rec in range(rec_from, rec_to + 1):

            test_thr_vm = tgui.adata.resting_vm
            test_thr_time = tgui.adata.time_array[rec][tgui.clean(test_spike_sample_idx[rec]).astype(int)]

            test_thr_time = self.data_within_bounds(tgui, test_thr_time,
                                                    rec, bounds_vm)

            if test_thr_time is False:
                continue

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

    def check_skinetics_qtable_spike_order(self, tgui, rec_from, rec_to, bounds_vm, filenum):
        """
        Check the spike order displayed in the table match the true order.
        """
        for rec in range(rec_from, rec_to + 1):

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

    def check_skinetics_peak_and_amplitude(self, tgui, rec_from, rec_to, bounds_vm, filenum):
        """
        Check the peaks and amplitudes in loaded_file, stored_tabledata and qtTable on the plot all match the test data.
        """
        for rec in range(rec_from, rec_to + 1):

            test_amplitudes = tgui.adata.all_true_peaks[rec] - tgui.adata.resting_vm
            test_peaks = tgui.adata.all_true_peaks[rec]

            test_amplitudes = self.data_within_bounds(tgui, test_amplitudes, rec, bounds_vm)
            test_peaks = self.data_within_bounds(tgui, test_peaks, rec, bounds_vm)

            if test_amplitudes is False:
                continue

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

    def check_skinetics_half_widths(self, tgui, rec_from, rec_to, bounds_vm, filenum):
        """
        Check the half_widths in loaded_file, stored_tabledata and qtTable on the plot all match the test data.
        """
        test_half_widths = tgui.adata.get_times_paramteres("half_width", bounds_vm=bounds_vm, time_type=tgui.time_type)

        for rec in range(rec_from, rec_to + 1):

            if bounds_vm:
                if not np.any(test_half_widths[rec]):
                    continue

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

    def check_skinetics_ahps(self, tgui, rec_from, rec_to, bounds_vm, filenum):
        """
        Check the ahps in loaded_file, stored_tabledata and qtTable on the plot all match the test data. First test the
        actual fahp and mahp value (raw, without thr subtraction) are correct on the loaded_file and stored tabledata.

        Then check that the true fahp / mahp (minus threshold) is correct on the actual table. This should cover all bases.

        Note that the fahp ahp are further tested with variable start / stop search times in test_fahp_and_ahp_timings()
        """
        for rec in range(rec_from, rec_to + 1):

            test_neg_peaks = tgui.adata.all_true_mins[rec]

            test_neg_peaks = self.data_within_bounds(tgui, test_neg_peaks, rec, bounds_vm)

            if test_neg_peaks is False:
                continue

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
    @pytest.mark.parametrize("run_with_bounds", [False, True])
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
            self.check_skinetics_qtable_spike_order(tgui, rec_from, rec_to, bounds_vm, filenum)
            self.check_skinetics_threshold(tgui, analyse_specific_recs, bounds_vm, filenum)
            self.check_skinetics_peak_and_amplitude(tgui, rec_from, rec_to, bounds_vm, filenum)
            self.check_skinetics_half_widths(tgui, rec_from, rec_to, bounds_vm, filenum)
            self.check_skinetics_ahps(tgui, rec_from, rec_to, bounds_vm, filenum)

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
# Test Manual Selection / Deletetion
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
            for spikes_to_select in [[0, 0], [0, 1], [0, 2], [3, 0],
                                     [6, 0], [6, 1], [8, 0], [9, 0]]:

                rec, rec_spike_idx = spikes_to_select
                tgui.mw.update_displayed_rec(rec)

                tgui.manually_select_spike(rec, rec_spike_idx)
                spike_params = self.check_all_skinetics_on_a_single_spike(tgui, rec, rec_spike_idx, skinetics_keys, filenum)

                for key in skinetics_keys:
                    inserted_spike_data[key].append(spike_params[key])

            # Delete a subset of selected spikes
            for spikes_deleted, spikes_to_delete in enumerate([[0, 1, 1], [3, 0, 3], [6, 1, 5]]):

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
