from PySide2 import QtWidgets, QtCore, QtGui
from PySide2 import QtTest
from PySide2.QtTest import QTest
import pytest
import sys
import os
import numpy as np
import scipy
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology.easy_electrophysiology.easy_electrophysiology import MainWindow
from ephys_data_methods import core_analysis_methods, event_analysis_master, voltage_calc
from ephys_data_methods_private import curve_fitting_master
import test_curve_fitting
from utils import utils
from setup_test_suite import GuiTestSetup, get_test_base_dir
from slow_vs_fast_settings import get_settings
import copy
from sys import platform
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
os.environ["PYTEST_QT_API"] = "pyside2"

SPEED = "fast"

DATA_TYPES = ["one_record", "multi_record", "multi_record_norm"]

def get_time_type(tgui):
    norm_or_cumu_time = "normalised" if tgui.analysis_type == "events_multi_record_norm" else "cumulative"
    return norm_or_cumu_time

# EVENTS / EVENT KINETICS ----------------------------------------------------------------------------------------------------------------------------

class TestEvents:

    @pytest.fixture(scope="function",  params=DATA_TYPES, ids=DATA_TYPES)
    def tgui(test, request):
        tgui = GuiTestSetup("artificial_events_" + request.param)
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.analysis_type = "events_" + request.param
        tgui.speed = SPEED
        tgui.setup_artificial_data(get_time_type(tgui), analysis_type="events_" + request.param)
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

# Helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def check_event_numbers(self, tgui, filenum):

        tgui.switch_mw_tab(1)
        qtable_num_events = tgui.get_data_from_qtable("event_num", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert qtable_num_events[0] == 1, "first table num events is not 1"

        increase = np.mean(np.diff(qtable_num_events)) if qtable_num_events.size > 1 else 1

        assert increase == 1, "table num events is not increasing by 1 for filenum " + str(filenum)
        assert qtable_num_events.size == tgui.adata.num_events(), "table num events does not equal true num events for filenum " + str(filenum)


    def check_event_times(self, tgui, filenum, rec_from=None, rec_to=None):

        if rec_from is not None and rec_to is not None:
            test_event_times = tgui.adata.peak_times[tgui.time_type][rec_from:rec_to + 1, :][~np.isnan(tgui.adata.peak_times[tgui.time_type][rec_from:rec_to + 1, :])]
        else:
            test_event_times = tgui.adata.peak_times[tgui.time_type][~np.isnan(tgui.adata.peak_times[tgui.time_type])]

        model_event_times = self.unpack_event_info_keys(tgui.mw.loaded_file.event_info)
        tabledata_event_times = self.unpack_event_info_keys(tgui.mw.stored_tabledata.event_info[filenum])
        qtable_event_times = tgui.get_data_from_qtable("event_time", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert np.array_equal(test_event_times, model_event_times), "loaded_file incorrect event times for filenum " + str(filenum)
        assert np.array_equal(test_event_times, tabledata_event_times), "tabledata incorrect event times for filenum " + str(filenum)
        assert np.array_equal(test_event_times, qtable_event_times), "qtable incorrect event times for filenum " + str(filenum)

    def check_baseline(self, tgui, filenum):
        test_baseline = np.repeat(tgui.adata.resting_vm, tgui.adata.num_events())

        model_baseline = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "baseline", "im")
        tabledata_baseline = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[filenum], "baseline", "im")
        qtable_baseline = tgui.get_data_from_qtable("baseline", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert np.array_equal(test_baseline, model_baseline), "loaded_file incorrect baseline for filenum " + str(filenum)
        assert np.array_equal(test_baseline, tabledata_baseline), "tabledata incorrect baseline for filenum " + str(filenum)
        assert np.array_equal(test_baseline, qtable_baseline), "qtable incorrect event baseline for filenum " + str(filenum)


    def check_amplitudes(self, tgui, filenum):

        test_amplitudes = tgui.adata.b1_offsets * tgui.adata.b1

        model_amplitudes = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "amplitude", "im")
        tabledata_amplitudes = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[filenum], "amplitude", "im")
        qtable_amplitudes = tgui.get_data_from_qtable("amplitude", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert np.array_equal(test_amplitudes, model_amplitudes), "loaded_file incorrect amplitudes for filenum " + str(filenum)
        assert np.array_equal(test_amplitudes, tabledata_amplitudes), "tabledata incorrect amplitudes for filenum " + str(filenum)
        assert np.array_equal(test_amplitudes, qtable_amplitudes), "qtable incorrect event amplitudes for filenum " + str(filenum)

    def check_rise_times(self, tgui, filenum):
        """
        test 10-90 rise times (EE default). The samples between 10th and 90th percent
        of rise is (90% sample - 80% sample - 1) * sample_spacing.
        """
        test_rise_times = ((tgui.adata.rise_samples * 0.79) * tgui.adata.ts) * 1000
        test_rise_times = np.repeat(test_rise_times, tgui.adata.num_events())

        model_rise_times = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "rise", "rise_time_ms")
        tabledata_rise_times = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[filenum], "rise", "rise_time_ms")
        qtable_rise_times = tgui.get_data_from_qtable("rise", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert utils.allclose(test_rise_times, model_rise_times, 1e-10), "loaded_file incorrect rise times for filenum " + str(filenum)
        assert utils.allclose(test_rise_times, tabledata_rise_times, 1e-10), "tabledata incorrect rise times for filenum " + str(filenum)
        assert utils.allclose(test_rise_times, qtable_rise_times, 1e-10), "qtable incorrect rise times for filenum " + str(filenum)

    def check_peaks(self, tgui, filenum):
        test_peaks = tgui.adata.resting_im + (tgui.adata.b1_offsets * tgui.adata.b1)

        model_peaks = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "peak", "im")
        tabledata_peaks = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[filenum], "peak", "im")
        qtable_peaks = tgui.get_data_from_qtable("peak", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert np.array_equal(test_peaks, model_peaks), "loaded_file incorrect peaks for filenum " + str(filenum)
        assert np.array_equal(test_peaks, tabledata_peaks), "tabledata incorrect peaks for filenum " + str(filenum)
        assert np.array_equal(test_peaks, qtable_peaks), "qtable incorrect rise peaks for filenum " + str(filenum)

    def check_decay_tau(self, tgui, filenum):

        test_tau_ms = self.get_test_tau_ms(tgui)

        model_decay_tau = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "monoexp_fit", "tau_ms")
        tabledata_decay_tau = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[filenum], "monoexp_fit", "tau_ms")
        qtable_decay_tau = tgui.get_data_from_qtable("monoexp_fit_tau", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert utils.allclose(test_tau_ms, model_decay_tau, 1e-10), "loaded_file incorrect decay tau for filenum " + str(filenum)
        assert utils.allclose(test_tau_ms, tabledata_decay_tau, 1e-10), "tabledata incorrect decay tau for filenum " + str(filenum)
        assert utils.allclose(test_tau_ms, qtable_decay_tau, 1e-10), "qtable incorrect decay tau for filenum " + str(filenum)


    def check_decay_percent(self, tgui, filenum):
        test_decay_perc = tgui.adata.get_all_decay_times()

        model_decay_perc = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "decay_perc", "decay_time_ms")
        tabledata_decay_perc = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[filenum], "decay_perc", "decay_time_ms")
        qtable_decay_perc = tgui.get_data_from_qtable("decay_perc", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert utils.allclose(test_decay_perc, model_decay_perc, 1e-10), "loaded_file incorrect decay percfor filenum " + str(filenum)
        assert utils.allclose(test_decay_perc, tabledata_decay_perc, 1e-10), "tabledata incorrect decay perc for filenum " + str(filenum)
        assert utils.allclose(test_decay_perc, qtable_decay_perc, 1e-10), "qtable incorrect decay perc for filenum " + str(filenum)

    def check_half_width(self, tgui, filenum):
        test_half_width = tgui.adata.get_all_half_widths()

        model_half_width = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "half_width", "fwhm_ms")
        tabledata_half_width = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[filenum], "half_width", "fwhm_ms")
        qtable_half_width = tgui.get_data_from_qtable("half_width", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert utils.allclose(test_half_width, model_half_width, 1e-10), "loaded_file incorrect half width for filenum " + str(filenum)
        assert utils.allclose(test_half_width, tabledata_half_width, 1e-10), "tabledata incorrect half width for filenum " + str(filenum)
        assert utils.allclose(test_half_width, qtable_half_width, 1e-10), "qtable incorrect half width for filenum " + str(filenum)

    def check_auc(self, tgui, filenum):
        """
        auc time is ested in test event endpoint
        """
        test_auc = tgui.adata.area_under_curves * 1000

        model_auc = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "area_under_curve", "im")
        tabledata_auc = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[filenum], "area_under_curve", "im")
        qtable_auc = tgui.get_data_from_qtable("area_under_curve", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert utils.allclose(test_auc, model_auc, 1e-10), "loaded_file incorrect area under curve for filenum " + str(filenum)
        assert utils.allclose(test_auc, tabledata_auc, 1e-10), "tabledata incorrect area under curve for filenum " + str(filenum)
        assert utils.allclose(test_auc, qtable_auc, 1e-10), "qtable incorrect area under curve for filenum " + str(filenum)

    def check_auc_event_periods(self, tgui, filenum=0):
        """
        """
        event_periods = self.get_auc_event_periods(tgui)

        model_event_periods = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "event_period", "time_ms")
        tabledata_event_periods = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[filenum], "event_period", "time_ms")
        qtable_event_periods = tgui.get_data_from_qtable("event_period", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert utils.allclose(event_periods, model_event_periods, 1e-10)
        assert utils.allclose(event_periods, tabledata_event_periods, 1e-10)
        assert utils.allclose(event_periods, qtable_event_periods, 1e-10)

    def unpack_event_info_keys(self, event_info):
        all_times = []
        for rec_event_info in event_info:
            for key in rec_event_info.keys():
                all_times.append(float(key))
        return all_times

    def unpack_event_info_values(self, event_info, key1, key2):
        all_params = []
        for rec_event_info in event_info:
            for ev_info in rec_event_info.values():
                all_params.append(ev_info[key1][key2])
        return all_params

    def get_all_frequency_data_from_table(self, tgui, num_isi):
        all_isi = utils.np_empty_nan(num_isi)
        for i in range(2, num_isi + 2):
            all_isi[i - 2] = float(tgui.mw.mw.table_tab_tablewidget.item(i, 1).data(0))

        return all_isi

    def get_all_test_isi_method_i(self, tgui):
        test_isi = []
        for rec in range(tgui.adata.num_recs):
            rec_peak_times = tgui.adata.peak_times[tgui.time_type][rec]
            num_rec_events = np.count_nonzero(~np.isnan(rec_peak_times))

            for ev_time_idx in range(1, num_rec_events):
                isi = rec_peak_times[ev_time_idx] - rec_peak_times[ev_time_idx - 1]
                test_isi.append(isi)

        test_isi = np.sort(np.array(test_isi))

        return test_isi

    def get_all_test_isi_method_ii(self, tgui):
        all_spikes = tgui.adata.peak_times[tgui.time_type][~np.isnan(tgui.adata.peak_times[tgui.time_type])]
        all_isi = np.diff(all_spikes)
        test_all_isi = np.sort(np.delete(all_isi, np.cumsum(tgui.adata.spikes_per_rec[:-1]) - 1))

        return test_all_isi

    def reshape_from_multi_rec_to_single_rec_or_back(self, tgui, direction, num_recs=False):
        """
        Direction: "multi_to_single" or "single_to_multi"
        remember if using "single_to_multi" to set num_recs
        """
        if tgui.mw.dialogs["template_analyse_events"]:  # critiical or will access error
            tgui.mw.dialogs["template_analyse_events"].close()
        if tgui.mw.dialogs["events_threshold_analyse_events"]:
            tgui.mw.dialogs["events_threshold_analyse_events"].close()

        if direction == "multi_to_single":
            tgui.mw.mw.actionReshape_Records.trigger()
            tgui.left_mouse_click(
                tgui.mw.dialogs["reshape"].dia.reshape_data_button_box.button(QtWidgets.QDialogButtonBox.Apply))

        elif direction == "single_to_multi":

            tgui.mw.mw.actionReshape_Records.trigger()
            tgui.left_mouse_click(tgui.mw.dialogs["reshape"].dia.reshape_to_multiple_records_radiobutton)
            tgui.enter_number_into_spinbox(tgui.mw.dialogs["reshape"].dia.reshape_data_dialog_spinbox, str(num_recs), setValue=True)
            tgui.left_mouse_click(tgui.mw.dialogs["reshape"].dia.reshape_data_button_box.button(QtWidgets.QDialogButtonBox.Apply))

    def grab_events_monoexp_fit_data_from_table(self, tgui):
        results = utils.np_empty_nan((tgui.mw.mw.table_tab_tablewidget.rowCount() - 2, 14))
        for row in range(2, tgui.mw.mw.table_tab_tablewidget.rowCount()):
            for col in [0, 1, *range(3, 14)]:
                results[row - 2, col] = float(tgui.mw.mw.table_tab_tablewidget.item(row, col).data(0))

        return results

    def calculate_rms_from_baseline(self, tgui, dialog, y_hat):
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)  # run the analysis to update the rms organically
        y = tgui.adata.im_array
        rms = np.sqrt(np.sum((y_hat - y)**2, axis=1) / tgui.adata.num_samples)

        return rms

    def get_test_tau_ms(self, tgui):
        test_tau_samples = tgui.adata.tau * tgui.adata.tau_offsets
        test_tau_ms = test_tau_samples * tgui.adata.ts * 1000
        return test_tau_ms

    def analysis_type(self, template_or_threshold):
        analysis_type = "events_template_matching" if template_or_threshold else "events_threshold"
        return analysis_type

    def get_events_analysis_dialog(self, tgui, template_or_threshold):
        analysis = "template_analyse_events" if template_or_threshold == "template" else "events_threshold_analyse_events"  # sort this out too
        dialog = tgui.mw.dialogs[analysis]
        return dialog

    def data_first_event_time(self, tgui):
        """
        """
        try:
            first_event_time = list(tgui.mw.loaded_file.event_info[0].keys())[0]
        except IndexError:
            return "None"

        return float(first_event_time)

    def data_first_event_param(self, tgui, param_key_1, param_key_2):
        """
        """
        event_time = str(self.data_first_event_time(tgui))
        if event_time == "None":
            return event_time

        try:
            first_spike = tgui.mw.loaded_file.event_info[0][event_time][param_key_1][param_key_2]
        except IndexError:
            first_spike = tgui.mw.loaded_file.event_info[1][event_time][param_key_1][param_key_2]

        return first_spike

    def get_peak_decay_times(self, tgui):
        """
        """
        peak_times = tgui.mw.loaded_file.make_list_from_event_info_all_recs("peak", "time")
        decay_times = tgui.mw.loaded_file.make_list_from_event_info_all_recs("decay_point", "time")
        peak_to_decay_times = (np.array(decay_times) - np.array(peak_times)) * 1000

        return peak_times, decay_times, peak_to_decay_times

    def get_auc_event_periods(self, tgui):
        """
        """
        peak_times, decay_times, peak_to_decay_times = self.get_peak_decay_times(tgui)

        bl_times = tgui.mw.loaded_file.make_list_from_event_info_all_recs("baseline", "time")
        bl_to_peak_times = (np.array(peak_times) - np.array(bl_times)) * 1000

        event_periods = bl_to_peak_times + peak_to_decay_times

        return event_periods

    def add_noise_to_loaded_file_traces(self, tgui, noise_divisor=25):
        tgui.mw.loaded_file.data.im_array = tgui.adata.im_array + np.random.randn(tgui.adata.num_samples) / noise_divisor
        tgui.mw.reset_after_data_manipulation()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test monoexponential event one and two channel
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    @pytest.mark.parametrize("vary_amplitude_and_tau", [True, False])
    @pytest.mark.parametrize("delete_events", [True])  # False
    @pytest.mark.parametrize("negative_events", [False, True])
    def test_events_overview_and_average(self, tgui, template_or_threshold, vary_amplitude_and_tau, delete_events, negative_events):
        """
        """
        for filenum in range(2):

            if not negative_events:
                tgui.setup_artificial_data(norm_or_cumu_time=get_time_type(tgui), analysis_type=tgui.analysis_type, negative_events=False)

            if vary_amplitude_and_tau:
                tgui.update_events_to_varying_amplitude_and_tau()

            tgui.run_artificial_events_analysis(tgui, template_or_threshold, negative_events=negative_events)

            if delete_events:
                tgui.delete_event_from_gui_and_artificial_data(tgui, template_or_threshold)

            self.check_event_numbers(tgui, filenum)
            self.check_event_times(tgui, filenum)
            self.check_baseline(tgui, filenum)
            self.check_peaks(tgui, filenum)
            self.check_amplitudes(tgui, filenum)
            self.check_rise_times(tgui, filenum)
            self.check_decay_percent(tgui, filenum)
            self.check_half_width(tgui, filenum)
            self.check_decay_tau(tgui, filenum)
            self.check_auc(tgui, filenum)
            self.check_auc_event_periods(tgui, filenum)

            # Test Event Averaging -------------------------------------------------------------------------------------------------------------------

            for idx in range(3):
                if template_or_threshold == "template":  # TODO: in average events too
                    tgui.left_mouse_click(tgui.mw.dialogs["template_analyse_events"].dia.average_all_events_button)
                    tgui.set_combobox(tgui.mw.dialogs["template_analyse_events"].average_all_events_dialog.dia.alignment_method_combobox, idx=idx)
                    dialog = tgui.mw.dialogs["template_analyse_events"].average_all_events_dialog
                else:
                    tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.average_all_events_button)
                    tgui.set_combobox(tgui.mw.dialogs["events_threshold_analyse_events"].average_all_events_dialog.dia.alignment_method_combobox, idx=idx)  # NEW
                    dialog = tgui.mw.dialogs["events_threshold_analyse_events"].average_all_events_dialog

                dialog.dia.window_size_spinbox.setValue(tgui.adata.event_width_ms)  # important, as if too large will discard events too close to the edge
                average_event = dialog.average_event_y
                test_average_event = tgui.adata.get_average_of_all_events()

                test_peak_idx = np.argmin(test_average_event)  if negative_events else np.argmax(test_average_event)
                peak_idx = np.argmin(average_event) if negative_events else np.argmax(average_event)

                aligned_test_average_event = test_average_event[test_peak_idx-70:test_peak_idx+70]
                aligned_average_event = average_event[peak_idx-70:peak_idx+70]

                assert utils.allclose(aligned_average_event, aligned_test_average_event, 1e-10), "average event filenum : {0} index: {1}".format(str(filenum), str(idx))

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data(get_time_type(tgui), analysis_type=tgui.analysis_type, negative_events=negative_events)

        del dialog
        tgui.shutdown()

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    def test_event_analyse_specific_recs(self, tgui, template_or_threshold):
        """
        Test analyse specific recs for events. This is a bit more difficult in events analysis. The easiest
        way to do it is just to remove all event times and parameters that are outside of the analysed recs
        and use then test against these with the standard functions.
        """
        if "one_record" in tgui.analysis_type:
            return

        tgui.update_events_to_varying_amplitude_and_tau()

        tgui.set_analysis_type(self.analysis_type(template_or_threshold))
        tgui.mw.mw.events_threshold_recs_to_analyse_groupbox.setChecked(True)
        tgui.handle_analyse_specific_recs(True)

        # make recs that are not within recs  NaN
        cut_bad_recs_peak_times = utils.np_empty_nan((tgui.adata.num_recs, tgui.adata.peak_times[tgui.time_type].shape[1]))
        cut_bad_recs_peak_times[tgui.rec_from():tgui.rec_to() + 1, :] = tgui.adata.peak_times[tgui.time_type][tgui.rec_from():tgui.rec_to() + 1, :]
        tgui.adata.peak_times[tgui.time_type] = cut_bad_recs_peak_times

        # delete event params from recs that we will not analyse
        evs_to_del = []
        for rec in range(tgui.adata.num_recs):
            if tgui.rec_from() <= rec <= tgui.rec_to():
                indicator = False
            else:
                indicator = True
            evs_to_del.append([indicator for ev in range(tgui.adata.spikes_per_rec[rec])])

        to_del_list = utils.flatten_list(evs_to_del)

        for ev_idx in reversed(range(len(to_del_list))):
            if to_del_list[ev_idx]:
                tgui.adata.b1_offsets = np.delete(tgui.adata.b1_offsets, ev_idx)
                tgui.adata.tau_offsets = np.delete(tgui.adata.tau_offsets, ev_idx)
                tgui.adata.area_under_curves = np.delete(tgui.adata.area_under_curves, ev_idx)

        # run analysis and check all output matches exactly
        tgui.run_artificial_events_analysis(tgui, template_or_threshold)

        filenum = 0
        self.check_event_numbers(tgui, filenum)
        self.check_event_times(tgui, filenum)
        self.check_baseline(tgui, filenum)
        self.check_peaks(tgui, filenum)
        self.check_amplitudes(tgui, filenum)
        self.check_rise_times(tgui, filenum)
        self.check_decay_percent(tgui, filenum)
        self.check_half_width(tgui, filenum)
        self.check_decay_tau(tgui, filenum)
        self.check_auc(tgui, filenum)

    def test_event_isi_in_record_mode(self, tgui):
        """
        Test files with time that have spacing in time between recs (e.g. rec 1 0-1, rec2 2-3)
         and that isi between events on different records are ignored.

        In the second half of the method, switch to frequency data to get the clculated ISI. The number of isi should be the number of
        rows minus the first 2 headers, and test this against the hypothetical (number of events minus the number of recs
        as ISI are not analysed between recs. Finally test the ISI on the table match ISI generated from the test data.
        """
        if tgui.test_filetype == "artificial_events_one_record":
            return

        tgui.update_events_time_to_irregularly_spaced()
        tgui.run_artificial_events_analysis(tgui, "template")
        tgui.switch_mw_tab(1)

        self.check_event_numbers(tgui, filenum=0)
        self.check_event_times(tgui, 0)

        tgui.left_mouse_click(tgui.mw.mw.table_event_cum_prob_frequency_checkbox)
        num_isi = self.get_num_isi_from_table(tgui)

        assert num_isi == tgui.adata.num_events() - tgui.adata.num_recs, "isis are not excluded across records"

        all_isi = self.get_all_frequency_data_from_table(tgui, num_isi)
        test_isi = self.get_all_test_isi_method_i(tgui)

        assert np.array_equal(all_isi, test_isi), "isi does not match test_isi"

    def get_num_isi_from_table(self, tgui):
        return tgui.mw.mw.table_tab_tablewidget.rowCount() - 2 - 4  # delete the first 2 rows (headers) and last 4 (summary stats)

    def test_conversion_from_multi_rec_to_single_and_back(self, tgui):
        """
        """
        if tgui.test_filetype in ["artificial_events_one_record",
                                  "artificial_events_multi_record_norm"]:
            return

        tgui.run_artificial_events_analysis(tgui, "template")
        tgui.switch_mw_tab(1)
        tgui.left_mouse_click(tgui.mw.mw.table_event_cum_prob_frequency_checkbox)

        num_isi = self.get_num_isi_from_table(tgui)
        assert num_isi == tgui.adata.num_events() - tgui.adata.num_recs, "isis are not excluded across records"

        rec_isi = self.get_all_frequency_data_from_table(tgui, num_isi)
        test_rec_isi = self.get_all_test_isi_method_ii(tgui)
        assert np.array_equal(rec_isi, test_rec_isi)

        self.reshape_from_multi_rec_to_single_rec_or_back(tgui, "multi_to_single")

        tgui.run_artificial_events_analysis(tgui, "template")
        tgui.switch_mw_tab(1)
        tgui.switch_checkbox(tgui.mw.mw.table_event_cum_prob_frequency_checkbox, on=True)

        num_isi = self.get_num_isi_from_table(tgui)
        assert num_isi == tgui.adata.num_events() - 1

        all_isi = self.get_all_frequency_data_from_table(tgui, num_isi)
        all_test_isi = np.sort(np.diff(tgui.adata.peak_times[tgui.time_type][~np.isnan(tgui.adata.peak_times[tgui.time_type])]))
        assert utils.allclose(all_test_isi, all_isi, 1e-10)

        self.reshape_from_multi_rec_to_single_rec_or_back(tgui, "single_to_multi", num_recs=tgui.adata.num_recs)
        tgui.run_artificial_events_analysis(tgui, "template")

        num_isi = self.get_num_isi_from_table(tgui)
        assert num_isi == tgui.adata.num_events() - tgui.adata.num_recs, "isis are not excluded across records"
        tgui.run_artificial_events_analysis(tgui, "template")

        rec_isi = self.get_all_frequency_data_from_table(tgui, num_isi)
        test_rec_isi = self.get_all_test_isi_method_ii(tgui)
        assert utils.allclose(rec_isi,  test_rec_isi, 1e-10)

    @pytest.mark.parametrize("vary_amplitude_and_tau", [False, True])
    def test_all_results_are_the_same_after_reshape_from_multi_to_single_rec_and_back(self, tgui, vary_amplitude_and_tau):
        """"""
        if tgui.test_filetype in ["artificial_events_one_record",
                                  "artificial_events_multi_record_norm"]:
            return

        if vary_amplitude_and_tau:
            tgui.update_events_to_varying_amplitude_and_tau()

        tgui.run_artificial_events_analysis(tgui, "template")
        rec_results = self.grab_events_monoexp_fit_data_from_table(tgui)

        self.reshape_from_multi_rec_to_single_rec_or_back(tgui, "multi_to_single")

        tgui.run_artificial_events_analysis(tgui, "template")
        new_results = self.grab_events_monoexp_fit_data_from_table(tgui)

        assert utils.allclose(new_results, rec_results, 1e-7)

        self.reshape_from_multi_rec_to_single_rec_or_back(tgui, "single_to_multi", num_recs=tgui.adata.num_recs)
        tgui.run_artificial_events_analysis(tgui, "template")

        reshaped_back_results = self.grab_events_monoexp_fit_data_from_table(tgui)
        assert utils.allclose(reshaped_back_results, rec_results, 1e-7)

    @pytest.mark.parametrize("vary_amplitude_and_tau", [True, False])
    def test_biexponential(self, tgui, vary_amplitude_and_tau):
        """
        This and test_events_template_numbers are extremely fiddly with sample number. For this function, biexp curve needs to be sampled
        enough so that the tau is calculated correctly and needs more samples that usual.
        """
        tgui.setup_artificial_data(norm_or_cumu_time="cumulative", analysis_type="events_multi_record_biexp_7500")

        tgui.mw.mw.actionEvents_Analyis_Options.trigger()
        tgui.mw.dialogs["events_analysis_options"].dia.event_fit_method_combobox.setCurrentIndex(1)
        tgui.mw.dialogs["events_analysis_options"].close()

        if vary_amplitude_and_tau:
            tgui.update_events_to_varying_amplitude_and_tau()
            tgui.set_widgets_for_artificial_event(tgui, run=True)
        else:
            tgui.run_artificial_events_analysis(tgui, "threshold", biexp=True)

        model_biexp_rise = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "biexp_fit", "rise_ms")
        model_biexp_decay = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "biexp_fit", "decay_ms")
        model_biexp_b1 = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "biexp_fit", "b1")

        tabledata_biexp_rise = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[0], "biexp_fit", "rise_ms")
        tabledata_biexp_decay = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[0], "biexp_fit", "decay_ms")
        tabledata_biexp_b1 = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[0], "biexp_fit", "b1")

        qtable_biexp_rise = tgui.get_data_from_qtable("biexp_fit_rise", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")
        qtable_biexp_decay = tgui.get_data_from_qtable("biexp_fit_decay", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")
        qtable_biexp_b1 = tgui.get_data_from_qtable("biexp_fit_b1", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        test_b1 = tgui.adata.b1 * tgui.adata.b1_offsets
        assert utils.allclose(model_biexp_b1, test_b1, 1e-05)
        assert utils.allclose(tabledata_biexp_b1, test_b1, 1e-05)
        assert utils.allclose(qtable_biexp_b1, test_b1, 1e-05)

        test_rise = (tgui.adata.rise * np.squeeze(tgui.adata.rise_offsets)) * tgui.adata.ts * 1000
        assert utils.allclose(model_biexp_rise, test_rise, 1e-05)
        assert utils.allclose(tabledata_biexp_rise, test_rise, 1e-05)
        assert utils.allclose(qtable_biexp_rise, test_rise, 1e-05)

        test_decay = (tgui.adata.decay * np.squeeze(tgui.adata.decay_offsets)) * tgui.adata.ts * 1000
        assert utils.allclose(model_biexp_decay, test_decay, 1e-05)
        assert utils.allclose(tabledata_biexp_decay, test_decay, 1e-05)
        assert utils.allclose(qtable_biexp_decay, test_decay, 1e-05)

    def test_unit_r2(self):
        """
        """
        y = np.random.randn(1000)
        y_hat = np.random.randn(1000)

        assert core_analysis_methods.calc_r2(y, y) == 1
        assert core_analysis_methods.calc_r2(y, np.mean(y)) == 0
        assert core_analysis_methods.calc_r2(y, y_hat) == r2_score(y, y_hat)

    def test_no_events(self, tgui):
        """
        Test zero events is handled correctly and that events are correctly cleared if no events are analysed.
        """
        tgui.switch_to_threshold_and_open_analsis_dialog(tgui)
        tgui.set_widgets_for_artificial_event_data(tgui, "threshold", biexp=False)

        # Test with no analysed events
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_threshold_analyse_events"].dia.threshold_lower_spinbox, -1000)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)
        QtWidgets.QApplication.processEvents()

        assert not any(tgui.mw.loaded_file.event_info)
        assert core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info) == 0
        qtable_num_events = tgui.get_data_from_qtable("event_num", row_from=0, row_to=tgui.mw.mw.table_tab_tablewidget.rowCount(), analysis_type="events")
        assert not np.any(qtable_num_events)

        # Test with analysed events
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_threshold_analyse_events"].dia.threshold_lower_spinbox, -68)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        assert any(tgui.mw.loaded_file.event_info)
        assert core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info) == tgui.adata.num_events()
        qtable_num_events = tgui.get_data_from_qtable("event_num", row_from=0, row_to=tgui.mw.mw.table_tab_tablewidget.rowCount(), analysis_type="events")
        assert len(qtable_num_events) == tgui.adata.num_events()

        # Test against with no analysd events
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_threshold_analyse_events"].dia.threshold_lower_spinbox, -1000)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)
        QtWidgets.QApplication.processEvents()

        assert not any(tgui.mw.loaded_file.event_info)
        assert core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info) == 0
        qtable_num_events = tgui.get_data_from_qtable("event_num", row_from=0, row_to=tgui.mw.mw.table_tab_tablewidget.rowCount(), analysis_type="events")
        assert not np.any(qtable_num_events)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_events_template_numbers(self, tgui):
        """
        """
        tgui.setup_artificial_data(norm_or_cumu_time="cumulative", analysis_type="events_multi_record_biexp")
        tgui.update_events_to_varying_amplitude_and_tau()

        template_1_coefs, template_2_coefs, template_3_coefs = tgui.set_widgets_for_artificial_event(tgui)

        template_nums = tgui.get_data_from_qtable("template_num", row_from=0, row_to=tgui.adata.num_events(),
                                                  analysis_type="events")
        decay_coefs = np.squeeze(tgui.adata.decay_offsets)
        decay_coefs[np.where(decay_coefs == template_1_coefs)] = 1
        decay_coefs[np.where(decay_coefs == template_2_coefs)] = 2
        decay_coefs[np.where(decay_coefs == template_3_coefs)] = 3

        assert np.array_equal(decay_coefs, template_nums)

        self.check_multiple_peak_plots(tgui)

    def check_multiple_peak_plots(self, tgui):
        for rec in range(tgui.adata.num_recs):
            tgui.mw.update_displayed_rec(rec)
            rec_template_nums, __ = self.get_all_rec_data(tgui, rec, "peak", "template_num", "im")
            all_peaks_time, all_peaks_im = self.get_all_rec_data(tgui, rec, "peak", "time", "im")

            rec_template_nums = rec_template_nums.astype(int)

            peak_1_idx = np.where(rec_template_nums == 1)
            peak_2_idx = np.where(rec_template_nums == 2)
            peak_3_idx = np.where(rec_template_nums == 3)

            main_plot = tgui.mw.loaded_file_plot
            for plot, peak_n_idx in zip([main_plot.peak_plot,              main_plot.peak_plot_2,            main_plot.peak_plot_3],
                                        [peak_1_idx, peak_2_idx, peak_3_idx]):

                assert np.array_equal(plot.xData, all_peaks_time[peak_n_idx])
                assert np.array_equal(plot.yData, all_peaks_im[peak_n_idx])

    def test_template_manual_select(self, tgui):
        """

        """
        tgui.run_artificial_events_analysis(tgui, "template")
        tgui.left_mouse_click(tgui.mw.mw.events_template_manually_select_button)

        ev_1_time, ev_2_time, ev_3_time = tgui.mw.loaded_file.make_list_from_event_info("peak", "time")[0:3]
        ev_1_im, ev_2_im, ev_3_im = tgui.mw.loaded_file.make_list_from_event_info("peak", "im")[0:3]

        for ev_time in [ev_1_time, ev_2_time, ev_3_time]:
            tgui.mw.loaded_file_plot.handle_events_delete_event("events_analyse",
                                                                [ev_time])

        for idx, (ev_time, ev_im) in enumerate(zip([ev_1_time, ev_2_time, ev_3_time],
                                                   [ev_1_im, ev_2_im, ev_3_im])):
            tgui.left_mouse_click(tgui.mw.mw.events_template_generate_button)
            tgui.set_combobox(tgui.mw.dialogs["events_template_generate"].dia.choose_template_combobox, idx)

            tgui.left_mouse_click(tgui.mw.mw.events_template_analyse_all_button)

            pos = QtCore.QRectF(0, 0, 0, 0)
            pos.setCoords(ev_time - 0.001, ev_im - 1, ev_time + 0.001, ev_im + 1)
            tgui.mw.loaded_file.handle_select_event_click(pos, "template_analyse")

        tgui.switch_mw_tab(1)  # necessary to update table
        first_3_templates = tgui.get_data_from_qtable("template_num", row_from=0, row_to=tgui.adata.num_events(),
                                                      analysis_type="events")[0:3]
        assert np.array_equal(first_3_templates, np.array([1, 2, 3]))

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Lower Threshold
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("template_or_threshold", ["threshold"]) # , "template"])
    @pytest.mark.parametrize("baseline_method", ["auto", "linear", "curve"])
    def test_rms_lower_threshold(self, tgui, template_or_threshold, baseline_method):
        """
        Set threshold lower to RMS and check against all baseline methods.

        This test does not work on macos due to some insane Qt bug in the test environment. First, the function does not
        wait for the loading cruve etc. dialogs to finish when set_combobox to RMS. If this is fixed by changing
        refine_diaclass to currentIndexChanged() to activated() and using .setCurrentIndex() here, the function
        completes sucessfully but then errors due to a signal issue 'AttributeError: 'NoneType' object has no attribute 'handle_manual_thr_spinbox_edit'
        Have never seen such a strange bug - the test completes sucessfully then errors due to signal at some point in teardown. Nothing like this
        happens on windows - just skip for macOS. if it is working on windows it should work on macos, and other tests / manual checks will
        pick up any other issues.
        """
        if platform == "darwin":
            return

        tgui.run_artificial_events_analysis(tgui, template_or_threshold)
        num_events = core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info)
        full_event_info = copy.deepcopy(tgui.mw.loaded_file.event_info)

        peak_val = tgui.adata.peak_im
        dialog_key = "template_analyse_events" if template_or_threshold == "template" else "events_threshold_analyse_events"  # TODO: own function
        dialog = tgui.mw.dialogs[dialog_key]

        # Set to n times RMS. Check the default can find all events.
        tgui.set_combobox(dialog.dia.threshold_lower_combobox, 3)

        tgui.left_mouse_click(dialog.dia.fit_all_events_button)
        new_num_events = core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info)
        assert new_num_events == num_events

        # Calculate and check the rms depending on the baseline method (std for auto, otherwise data - baseline)
        if baseline_method == "auto":
            rms = np.std(tgui.adata.im_array, axis=1)

        elif baseline_method == "linear":

            tgui.set_combobox(dialog.dia.baseline_combobox, 1)

            tgui.enter_number_into_spinbox(dialog.dia.baseline_spinbox, tgui.adata.resting_im)

            y_hat = tgui.mw.cfgs.events["baseline_axline_settings"]["linear_baseline_value"]
            rms = self.calculate_rms_from_baseline(tgui, dialog, y_hat)

            assert np.array_equal(rms, dialog.rms_threshold_lower["rms"])

        elif baseline_method == "curve":

            tgui.set_combobox(dialog.dia.baseline_combobox, 2)
            y_hat = dialog.curved_baseline_w_displacement
            rms = self.calculate_rms_from_baseline(tgui, dialog, y_hat)

            assert np.array_equal(rms, dialog.rms_threshold_lower["rms"])

        # Find the rms per record and take the record
        # with the max RMS and calculate the multiplier to get this threshold over the peak value. Set this multiplier and analyse,
        # and check that the record with the highest RMS events are not analysed. As some recs RMS are very close, find all the recs
        # where the n times rms is above peak val and check these are selectively removed.
        max_rec_rms = np.argmax(rms)
        max_rms = rms[max_rec_rms]
        rms_multiplier = abs(abs(peak_val) - abs(tgui.adata.resting_im)) / max_rms
        rms_multiplier = np.ceil(rms_multiplier * 100) / 100  # ceil to 2 decimal places
        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, rms_multiplier)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        if baseline_method == "linear":
            baselines = np.tile(tgui.adata.resting_im, tgui.adata.num_recs)
        else:
            baselines = np.mean(tgui.adata.im_array, axis=1)
        rms_thresholds = baselines - dialog.rms_threshold_lower["n_times_rms"]
        recs_with_thr_below_peak = np.where(rms_thresholds < peak_val)[0]

        assert max_rec_rms in list(recs_with_thr_below_peak)

        if baseline_method == "curve":
            # If curve, find the sample-wise cutoff as the curve baseline is changing across all datapoints
            new_num_events = 0
            for rec in range(tgui.adata.num_recs):
                all_rec_peak_idx = tgui.mw.loaded_file.make_list_from_event_info("peak", "idx", event_info=full_event_info, rec=rec)
                peak_ims = tgui.adata.im_array[rec][all_rec_peak_idx]
                cutoff_ims = dialog.curved_baseline_w_displacement[rec][all_rec_peak_idx] - rms[rec] * rms_multiplier
                new_num_events += np.sum(peak_ims < cutoff_ims)
            assert new_num_events == core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info)

        else:
            new_num_events = core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info)
            assert new_num_events == num_events - np.sum(tgui.adata.spikes_per_rec[recs_with_thr_below_peak])

        # check that none are detected if the threshold high enough
        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, rms_multiplier * 100)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)
        assert any(tgui.mw.loaded_file.event_info) is False

        del dialog
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Baseline
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    @pytest.mark.parametrize("combobox_idx_to_test", [2])  # 1,
    def test_baselines_not_auto(self, tgui, template_or_threshold, combobox_idx_to_test):
        """
        Test baselines, but not auto. Auto would be impossible to test here and is unit tested.

        if linear, set to half-point and check above and below the half -point
        if curve, set the offset as half the amplitude (same as setting linear to half-point) and check the idx that
        the curve is at as the baseline.
        """
        tgui.run_artificial_events_analysis(tgui, template_or_threshold)

        event_midpoint = tgui.adata.resting_im + tgui.adata.b1 / 2
        dialog_key = "template_analyse_events" if template_or_threshold == "template" else "events_threshold_analyse_events"  # TODO: own function
        dialog = tgui.mw.dialogs[dialog_key]

        if platform == "darwin":
            dialog.dia.baseline_combobox.setCurrentIndex(combobox_idx_to_test)  # see tgui.set_combobox() same macOS / Qt bug
        else:
            tgui.set_combobox(dialog.dia.baseline_combobox, combobox_idx_to_test)

        baseline_cutoff = tgui.adata.b1 / 2 if combobox_idx_to_test == 2 else event_midpoint

        tgui.enter_number_into_spinbox(dialog.dia.baseline_spinbox, baseline_cutoff)
        tgui.run_artificial_events_analysis(tgui, template_or_threshold)

        for rec in range(tgui.adata.num_recs):

            baseline_idx = tgui.mw.loaded_file.make_list_from_event_info("baseline", "idx", event_info=tgui.mw.loaded_file.event_info, rec=rec)
            baseline_idx = list(baseline_idx)

            test_point = event_midpoint if combobox_idx_to_test == 1 else dialog.curved_baseline_w_displacement[rec][baseline_idx]

            QtWidgets.QApplication.processEvents()
            assert (tgui.adata.im_array[rec][baseline_idx] > test_point).all()
            baseline_idx = np.array(baseline_idx) + 1
            assert (tgui.adata.im_array[rec][baseline_idx] < test_point).all()

    def test_threshold_lower_plot_shown(self, tgui):
        """
        Test all threshold lower plots (linear, curved, RMS under all baselines except drawn) and
        baseline plots.
        """
        dialog = tgui.switch_to_threshold_and_open_analsis_dialog(tgui)

        # Linear -------------------------------------------------------------------------------------------------------

        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, tgui.adata.peak_im)  # use peak which will always be on plot

        for rec in range(tgui.adata.num_recs):
            tgui.mw.update_displayed_rec(rec)

            assert dialog.man_thr_axline.axline.y() == tgui.adata.peak_im

        # Curved -------------------------------------------------------------------------------------------------------

        tgui.set_combobox(dialog.dia.threshold_lower_combobox, 1)
        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, -4)

        for rec in range(tgui.adata.num_recs):
            tgui.mw.update_displayed_rec(rec)

            curved_thr_poly_coefs = np.polyfit(tgui.adata.time_array[rec],
                                               tgui.adata.im_array[rec],
                                               tgui.mw.cfgs.events["dynamic_curve_polynomial_order"])
            test_curved_thr = np.polyval(curved_thr_poly_coefs, tgui.adata.time_array[rec])
            assert np.array_equal(test_curved_thr - 4, tgui.mw.loaded_file_plot.curved_threshold_lower_plot.yData)

        # RMS - Auto ---------------------------------------------------------------------------------------------------

        tgui.set_combobox(dialog.dia.threshold_lower_combobox, 3)
        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, 5)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        for rec in range(tgui.adata.num_recs):
            tgui.mw.update_displayed_rec(rec)

            test_rms = np.std(tgui.adata.im_array[rec])
            rms = dialog.rms_threshold_lower["rms"][rec]
            assert rms == test_rms

            test_rms = np.tile(np.mean(tgui.adata.im_array[rec]), tgui.adata.num_samples) - (rms * 5)
            assert np.array_equal(test_rms, tgui.mw.loaded_file_plot.rms_threshold_lower_plot.yData)

        # RMS - Linear -------------------------------------------------------------------------------------------------
        # tgui.set_combobox(dialog.dia.baseline_combobox, 1) (Qt test bug)
        dialog.dia.baseline_combobox.setCurrentIndex(1)
        tgui.enter_number_into_spinbox(dialog.dia.baseline_spinbox, -62)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)
        baseline = np.tile(-62, tgui.adata.num_samples)

        for rec in range(tgui.adata.num_recs):
            tgui.mw.update_displayed_rec(rec)

            test_rms = mean_squared_error(tgui.adata.im_array[rec], baseline, squared=False)
            rms = dialog.rms_threshold_lower["rms"][rec]

            assert test_rms == rms
            assert np.array_equal(baseline - (rms * 5), tgui.mw.loaded_file_plot.rms_threshold_lower_plot.yData)
            assert baseline[0] == tgui.mw.cfgs.events["baseline_axline_settings"]["linear_baseline_value"]
            assert baseline[0] == dialog.baseline_axline.axline.y()

        # RMS - Curved -------------------------------------------------------------------------------------------------

        tgui.set_combobox(dialog.dia.baseline_combobox, 2)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        for rec in range(tgui.adata.num_recs):
            tgui.mw.update_displayed_rec(rec)

            curved_thr_poly_coefs = np.polyfit(tgui.adata.time_array[rec],
                                               tgui.adata.im_array[rec],
                                               tgui.mw.cfgs.events["dynamic_curve_polynomial_order"])
            baseline = np.polyval(curved_thr_poly_coefs, tgui.adata.time_array[rec])

            test_rms = mean_squared_error(tgui.adata.im_array[rec], baseline, squared=False)
            rms = dialog.rms_threshold_lower["rms"][rec]

            assert test_rms == rms
            assert np.array_equal(baseline - (test_rms * 5), tgui.mw.loaded_file_plot.rms_threshold_lower_plot.yData)
            assert np.array_equal(baseline, tgui.mw.loaded_file_plot.curved_baseline_plot.yData)

        del dialog

    def test_threshold_lower_show_and_hidden_correctly(self, tgui):
        """
        Just test threshold as both tested above and it all uses the same code anyway...
        """
        dialog = tgui.switch_to_threshold_and_open_analsis_dialog(tgui)
        upperplot = tgui.mw.loaded_file_plot.upperplot

        # linear on / off
        assert dialog.man_thr_axline.axline in tgui.mw.loaded_file_plot.upperplot.items

        tgui.switch_checkbox(dialog.dia.hide_threshold_lower_from_plot_checkbox, on=True)
        assert dialog.man_thr_axline.axline not in tgui.mw.loaded_file_plot.upperplot.items

        tgui.switch_checkbox(dialog.dia.hide_threshold_lower_from_plot_checkbox, on=False)
        assert dialog.man_thr_axline.axline in tgui.mw.loaded_file_plot.upperplot.items

        tgui.set_combobox(dialog.dia.threshold_lower_combobox, idx=1)

        # curved on / off
        assert dialog.man_thr_axline.axline not in upperplot.items
        assert tgui.mw.loaded_file_plot.curved_threshold_lower_plot in upperplot.items

        tgui.switch_checkbox(dialog.dia.hide_threshold_lower_from_plot_checkbox, on=True)
        assert dialog.man_thr_axline.axline not in tgui.mw.loaded_file_plot.upperplot.items

        tgui.switch_checkbox(dialog.dia.hide_threshold_lower_from_plot_checkbox, on=False)
        assert tgui.mw.loaded_file_plot.curved_threshold_lower_plot in upperplot.items

        tgui.set_combobox(dialog.dia.threshold_lower_combobox, idx=3)

        # rms on / off
        assert tgui.mw.loaded_file_plot.rms_threshold_lower_plot.xData is None
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        assert tgui.mw.loaded_file_plot.rms_threshold_lower_plot.xData is not None
        assert tgui.mw.loaded_file_plot.rms_threshold_lower_plot in upperplot.items
        tgui.switch_checkbox(dialog.dia.hide_threshold_lower_from_plot_checkbox, on=True)

        assert tgui.mw.loaded_file_plot.rms_threshold_lower_plot not in upperplot.items

        tgui.switch_checkbox(dialog.dia.hide_threshold_lower_from_plot_checkbox, on=False)
        assert tgui.mw.loaded_file_plot.rms_threshold_lower_plot in upperplot.items

        # back to linear
        tgui.set_combobox(dialog.dia.threshold_lower_combobox, idx=0)

        assert tgui.mw.loaded_file_plot.rms_threshold_lower_plot not in upperplot.items
        assert dialog.man_thr_axline.axline in tgui.mw.loaded_file_plot.upperplot.items

        del dialog

    def test_baseline_show_and_hidden_correctly(self, tgui):
        """
        """
        dialog = tgui.switch_to_threshold_and_open_analsis_dialog(tgui)
        upperplot = tgui.mw.loaded_file_plot.upperplot

        # linear on / off
        tgui.set_combobox(dialog.dia.baseline_combobox, idx=1)

        assert dialog.baseline_axline.axline in upperplot.items

        tgui.switch_checkbox(dialog.dia.hide_baseline_from_plot_checkbox, on=True)
        assert dialog.baseline_axline.axline not in upperplot.items

        tgui.switch_checkbox(dialog.dia.hide_baseline_from_plot_checkbox, on=False)
        assert dialog.baseline_axline.axline in upperplot.items

        # curved on / off
        tgui.set_combobox(dialog.dia.baseline_combobox, idx=2)

        assert tgui.mw.loaded_file_plot.curved_baseline_plot in upperplot.items
        assert tgui.mw.loaded_file_plot.curved_baseline_plot.xData is not None

        tgui.switch_checkbox(dialog.dia.hide_baseline_from_plot_checkbox, on=True)
        assert tgui.mw.loaded_file_plot.curved_baseline_plot not in upperplot.items

        tgui.switch_checkbox(dialog.dia.hide_baseline_from_plot_checkbox, on=False)
        assert tgui.mw.loaded_file_plot.curved_baseline_plot in upperplot.items

        tgui.set_combobox(dialog.dia.baseline_combobox, idx=1)
        assert dialog.baseline_axline.axline in upperplot.items
        assert tgui.mw.loaded_file_plot.curved_baseline_plot not in upperplot.items

        del dialog

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Average Events
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_simple_unit_test_average_events(self):
        # here the left edge is 0 and the right edge is num samples ( 5)
        im_array = np.array([0, 0, 5, 4, 3, 2, 1,  0, 0, 50, 40, 30, 20, 10, 0, 0, 500, 400, 300, 200, 100, 0, 0, 0, 0])
        im_array = np.vstack([im_array, im_array * 1000])

        event_info = [{"5": {"peak": {"idx": 2}, "baseline": {"idx": 3}, "half_width": {"rise_mid_idx": 3}},
                       "50": {"peak": {"idx": 9}, "baseline": {"idx": 11}, "half_width": {"rise_mid_idx": 12}},
                       "500": {"peak": {"idx": 16}, "baseline": {"idx": 19}, "half_width": {"rise_mid_idx": 20}}},

                      {"5000": {"peak": {"idx": 2}, "baseline": {"idx": 3}, "half_width": {"rise_mid_idx": 3}},
                       "50000": {"peak": {"idx": 9}, "baseline": {"idx": 11}, "half_width": {"rise_mid_idx": 12}},
                       "500000": {"peak": {"idx": 16}, "baseline": {"idx": 19}, "half_width": {"rise_mid_idx": 20}}

                       }]
        average_event = event_analysis_master.make_average_event(im_array, 5, "peak", event_info)
        assert np.array_equal(average_event, np.array([np.mean([5, 50, 500, 5000, 50000, 500000]), np.mean([4, 40, 400, 4000, 40000, 400000]), np.mean([3, 30, 300, 3000, 30000, 300000]),
                                                       np.mean([2, 20, 200, 2000, 20000, 200000]), np.mean([1, 10, 100, 1000, 10000, 100000])])), "peak"

        average_event = event_analysis_master.make_average_event(im_array, 5, "baseline", event_info)
        assert np.array_equal(average_event, np.array([np.mean([4, 30, 200, 4000, 30000, 200000]), np.mean([3, 20, 100, 3000, 20000, 100000]),
                                                       np.mean([2, 10, 0, 2000, 10000, 0]), np.mean([1000, 0, 0, 1, 0, 0]), np.mean([0, 0, 0, 0, 0, 0])])), "baseline"

        average_event = event_analysis_master.make_average_event(im_array, 5, "rise_half_width", event_info)
        assert np.array_equal(average_event, np.array([np.mean([4, 20, 4000, 20000]), np.mean([3, 10, 3000, 10000]),
                                                       np.mean([0, 2, 0, 2000]), np.mean([0, 1, 0, 1000]), np.mean([500, 0, 500000, 0])])), "rise_half_width"  # the last event is lost because it goes off the end

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    @pytest.mark.parametrize("peak_method", ["baseline", "rise_half_width", "baseline"])
    def test_unit_test_average_events(self, tgui, template_or_threshold, peak_method):
        """
        Tried biexp, was a real pain to sort out the alignment so skip - the existing way is sufficient.
        """
        tgui.update_events_to_varying_amplitude_and_tau()

        tgui.run_artificial_events_analysis(tgui, template_or_threshold)

        test_average_event = tgui.adata.get_average_of_all_events()
        average_event = event_analysis_master.make_average_event(tgui.adata.im_array, tgui.adata.event_samples, peak_method, tgui.mw.loaded_file.event_info)

        average_event = np.delete(average_event, np.where(np.isclose(average_event, tgui.adata.resting_im, 1e-10)))
        test_average_event = np.delete(test_average_event, np.where(np.isclose(test_average_event, tgui.adata.resting_im, 1e-10)))

        assert utils.allclose(average_event, test_average_event, 1e-8)

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    def test_average_events_gui(self, tgui, template_or_threshold):
        """

        """
        tgui.update_events_to_varying_amplitude_and_tau()

        test_average_event = tgui.adata.get_average_of_all_events()
        num_samples_rise = np.argmin(test_average_event)
        num_samples_decay = len(test_average_event) - np.argmin(test_average_event)
        decay_time = (num_samples_decay + 1) * tgui.adata.ts
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_decay_search_period_spinbox,
                                       str(decay_time * get_settings(tgui.speed,
                                                                     "curve_fitting")["decay_search_period"]))

        tgui.mw.cfgs.events["decay_or_biexp_fit_method"] = "do_not_fit"
        tgui.mw.cfgs.events["templates"]["1"]["window_len_s"] = decay_time
        tgui.mw.cfgs.events["templates"]["1"]["window_len_samples"] = num_samples_decay

        tgui.mw.cfgs.events["interp_200khz"] = True
        tgui.run_artificial_events_analysis(tgui, template_or_threshold)

        for idx in range(3):

            if template_or_threshold == "template":
                tgui.left_mouse_click(tgui.mw.dialogs["template_analyse_events"].dia.average_all_events_button)
                tgui.set_combobox(tgui.mw.dialogs["template_analyse_events"].average_all_events_dialog.dia.alignment_method_combobox, idx=idx)
                dialog = tgui.mw.dialogs["template_analyse_events"].average_all_events_dialog
            else:
                tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.average_all_events_button)
                tgui.set_combobox(tgui.mw.dialogs["events_threshold_analyse_events"].average_all_events_dialog.dia.alignment_method_combobox, idx=idx)
                dialog = tgui.mw.dialogs["events_threshold_analyse_events"].average_all_events_dialog

            tgui.enter_number_into_spinbox(dialog.dia.window_size_spinbox, tgui.adata.event_width_ms)

            average_event = dialog.average_event_y

            av_peak = np.argmin(average_event)
            average_event_cut = average_event[av_peak - num_samples_rise:av_peak + num_samples_decay]

            assert utils.allclose(test_average_event, average_event_cut, 1e-10)  # TODO: extremely rarely this can assert as win len is set slightly too long (test not EE issue)

    def test_normalise_average_event(self, tgui):

        tgui.run_artificial_events_analysis(tgui, "template")
        tgui.left_mouse_click(tgui.mw.dialogs["template_analyse_events"].dia.average_all_events_button)

        avg_event_dialog = tgui.mw.dialogs["template_analyse_events"].average_all_events_dialog

        raw_y = copy.deepcopy(avg_event_dialog.average_event_y)
        norm_raw_gui_avg_event = (raw_y - np.mean(raw_y)) / (np.max(raw_y) - np.min(raw_y))

        tgui.switch_checkbox(avg_event_dialog.dia.normalize_event_checkbox, on=True)

        assert utils.allclose(avg_event_dialog.average_event_y, norm_raw_gui_avg_event)
        assert np.abs(np.max(avg_event_dialog.average_event_y) - np.min(avg_event_dialog.average_event_y)) == 1

        tgui.set_combobox(avg_event_dialog.dia.alignment_method_combobox, idx=1)
        assert self.event_amplitude(avg_event_dialog.average_event_y) == 1

        tgui.set_combobox(avg_event_dialog.dia.alignment_method_combobox, idx=2)
        assert self.event_amplitude(avg_event_dialog.average_event_y) == 1

        tgui.switch_groupbox(avg_event_dialog.dia.filter_data_groupbox, on=True)
        tgui.enter_number_into_spinbox(avg_event_dialog.dia.filter_data_spinbox, 10000)
        assert self.event_amplitude(avg_event_dialog.average_event_y) == 1

        del avg_event_dialog

    def event_amplitude(self, data):
        return np.abs(np.max(data) - np.min(data))

    def test_sanity_check_average_event_widget_refine(self, tgui):

        tgui.set_analysis_type("events_template_matching")
        tgui.left_mouse_click(tgui.mw.mw.events_template_refine_button)

        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_baseline_search_period_spinbox, 25)
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_template_refine"].dia.threshold_lower_spinbox, -62)
        tgui.left_mouse_click(tgui.mw.dialogs["events_template_refine"].dia.fit_all_events_button)

        for idx, config in enumerate(["rise_half_width", "peak", "baseline"]):
            tgui.set_combobox(tgui.mw.dialogs["events_template_refine"].dia.alignment_method_combobox, idx)
            assert config == tgui.mw.dialogs["events_template_refine"].average_alignment_method

    def test_sanity_check_average_event_widget_average(self, tgui):
        tgui.run_artificial_events_analysis(tgui, "template")
        tgui.left_mouse_click(tgui.mw.dialogs["template_analyse_events"].dia.average_all_events_button)

        for idx, config in enumerate(["rise_half_width", "peak", "baseline"]):
            tgui.set_combobox(tgui.mw.dialogs["template_analyse_events"].average_all_events_dialog.dia.alignment_method_combobox, idx)
            assert config == tgui.mw.dialogs["template_analyse_events"].average_all_events_dialog.alignment_method


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Save Event Data
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# Unit test serialization ----------------------------------------------------------------------------------------------

    def test_unit_test_handle_event_info_serialization(self, tgui):
        """
        Test all entries in the event info are properly prepared for serialization (not numpy) before saving in json format.
        Note this only tests the settings on by default (e.g. not biexp) but this is good enough as long as the
        event_info dict is 3 levels deep, which is also tested (in self.break_down_event_info)
        """
        tgui.run_artificial_events_analysis(tgui, "template")
        event_info = tgui.mw.loaded_file.event_info
        linear_event_info = self.break_down_event_info_or_compare_two(event_info)
        self.check_json_normalised_type(linear_event_info, "raw")

        event_info_saved = tgui.mw.loaded_file.handle_event_info_serialization("save", event_info)
        linear_event_info_saved = self.break_down_event_info_or_compare_two(event_info_saved)
        self.check_json_normalised_type(linear_event_info_saved, "after_save")

        event_info_loaded = tgui.mw.loaded_file.handle_event_info_serialization("load", event_info)
        linear_event_info_loaded = self.break_down_event_info_or_compare_two(event_info_loaded)
        self.check_json_normalised_type(linear_event_info_loaded, "after_load")

    def check_json_normalised_type(self, linear_event_info, mode):
        """
        Input is output of break_down_event_info
        """
        for cell in linear_event_info:
            if cell is not None and type(cell) != str and np.all(~np.isnan(cell)) and np.all(cell != [np.nan]):

                if mode == "raw":
                    assert type(cell) in [np.int32, np.float64, np.int64, np.ndarray, int]

                elif mode == "after_save":
                    assert type(cell) in [float, int, list]

                elif mode == "after_load":
                    assert type(cell) in [np.float64, np.int64, np.ndarray]

    def break_down_event_info_or_compare_two(self, event_info, second_event_info=None, skip_dict_keys=None):

        if not skip_dict_keys:
            skip_dict_keys = []

        linear_event_info = []
        for rec in range(len(event_info)):
            for key1 in event_info[rec].keys():
                for key2 in event_info[rec][key1].keys():
                    assert type(event_info[rec][key1][key2]) == dict

                    if key2 not in skip_dict_keys:
                        for key3 in event_info[rec][key1][key2].keys():

                            lowest_level = event_info[rec][key1][key2][key3]
                            assert type(lowest_level) != dict  # event_info dict must be 3 levels max (time, dict layer 1, dict layer 2)
                            linear_event_info.append(lowest_level)

                            if second_event_info:
                                if type(lowest_level) == np.ndarray:
                                    np.array_equal(event_info[rec][key1][key2][key3], second_event_info[rec][key1][key2][key3])
                                elif np.all(lowest_level == [np.nan]):
                                    pass
                                else:
                                    assert lowest_level == second_event_info[rec][key1][key2][key3]

        return linear_event_info

# Unit test serialization ----------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    def test_save_events_structure_is_correct(self, tgui, template_or_threshold):
        """
        """
        tgui.run_artificial_events_analysis(tgui, template_or_threshold)
        full_filepath = tgui.save_events_analysis(tgui)

        loaded_file_dict = self.load_saved_event_info(tgui, full_filepath)

        self.break_down_event_info_or_compare_two(tgui.mw.loaded_file.event_info,
                                                  loaded_file_dict["event_info"])
        assert loaded_file_dict["filename"] == tgui.mw.loaded_file.fileinfo["filename"]
        assert loaded_file_dict["num_samples"] == tgui.mw.loaded_file.data.num_samples
        assert loaded_file_dict["num_recs"] == tgui.mw.loaded_file.data.num_recs

    @pytest.mark.parametrize("version", ["v2.3.3", "v2.4.0", "v2.5.0"])
    def test_old_save_versions_load_correctly(self, tgui, version):
        """
        Make sure save events files in previous versions are loaded
        into this version.
        Note must update this for all new versions!
        """
        test_filepath = get_test_base_dir() + "/saved_events_tests/2022_01_11_0000 extract filt.abf"

        tgui.mw.load_file(test_filepath)

        saved_events_path = get_test_base_dir()  + f"/saved_events_tests/{version}_2022_01_11_0000 extract filt.json"

        tgui.mw.loaded_file.load_event_analysis(saved_events_path)

        loaded_file_dict = self.load_saved_event_info(tgui, saved_events_path)

        skip_dict_keys = self.get_skip_dict_keys(version)

        self.break_down_event_info_or_compare_two(tgui.mw.loaded_file.event_info,
                                                  loaded_file_dict["event_info"],
                                                  skip_dict_keys=skip_dict_keys)

    def get_skip_dict_keys(self, version):
        """
        v2.3.3 was before area_under_curve, event_period was
        added
        """
        if version == "v2.3.3":
            return ["area_under_curve", "event_period"]

    def load_saved_event_info(self, tgui, filepath):
        """"""
        with open(filepath, "rb") as f:
            load_defaults = f.read()
        loaded_file_dict = tgui.mw.cfgs.decrypt_data(load_defaults)
        return loaded_file_dict


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Individual Events
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_unit_test_get_event_rec_and_idx_from_total_event_idx__AND__total_event_idx_from_rec_and_idx(self, tgui):
        """
        """
        tgui.set_analysis_type("events_template_matching")
        tgui.left_mouse_click(tgui.mw.mw.events_template_refine_button)

        event_info = [{"1": None, "2": None, "3": None},
                      {"4": None, "5": None, "6": None},
                      {"7": None, "8": None, "9": None}]
        for idx_to_find, (test_idx, test_rec) in enumerate(zip([0, 1, 2, 0, 1, 2, 0, 1, 2],
                                                               [0, 0, 0, 1, 1, 1, 2, 2, 2])):
            rec, idx = tgui.mw.dialogs["events_template_refine"].get_event_rec_and_idx_from_total_event_idx(event_info, idx_to_find)
            assert test_idx == idx
            assert test_rec == rec

            idx = tgui.mw.dialogs["events_template_refine"].total_event_idx_from_rec_and_idx(event_info, test_rec, test_idx)
            assert idx_to_find == idx

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Template GUI
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("to_test", ["b0", "width"])
    def test_generate_template_dialog(self, tgui, to_test):
        """
        The spinbox lower input is very low for b1, rise ande decay spinbox so just ignored. This is
        mainly to test width which will error if the minimum is not set properly.
        """
        tgui.set_analysis_type("events_template_matching")
        tgui.left_mouse_click(tgui.mw.mw.events_template_generate_button)

        dialog = tgui.mw.dialogs["events_template_generate"]

        tgui.mw.cfgs.events["template_data_first_selection"]["x"] = tgui.mw.loaded_file.data.time_array[0][0:50]
        tgui.mw.cfgs.events["template_data_first_selection"]["y"] = tgui.mw.loaded_file.data.im_array[0][0:50]
        dialog.handle_user_selected_raw_data_for_template()

        spinbox = dialog.dia.b0_spinbox if to_test == "b0" else dialog.dia.width_spinbox
        slider = dialog.dia.b0_slider if to_test == "b0" else dialog.dia.width_slider

        tgui.enter_number_into_spinbox(spinbox, spinbox.minimum())
        assert spinbox.value() == spinbox.minimum()

        if to_test != "width":  # width cannot be neg
            tgui.enter_number_into_spinbox(spinbox, -9999)
            assert spinbox.value() == -999

        tgui.enter_number_into_spinbox(spinbox, spinbox.maximum())
        assert spinbox.value() == spinbox.maximum()
        assert slider.value() == slider.maximum()

        tgui.enter_number_into_spinbox(spinbox, 9999)
        assert spinbox.value() == 999

        del dialog

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    def test_omit_times(self, tgui, template_or_threshold):
        """
        """
        tgui.set_analysis_type(self.analysis_type(template_or_threshold))

        if template_or_threshold == "template":
            tgui.left_mouse_click(tgui.mw.mw.events_template_analyse_all_button)
        else:
            tgui.left_mouse_click(tgui.mw.mw.events_threshold_analyse_events_button)

        tgui.set_widgets_for_artificial_event_data(tgui, template_or_threshold, biexp=False)

        start_stop_times = get_settings(tgui.speed,
                                        tgui.analysis_type)["start_stop_times"]

        dialog = self.get_events_analysis_dialog(tgui, template_or_threshold)
        tgui.enter_numbers_into_omit_times_table(dialog, start_stop_times)

        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        time_ = tgui.adata.peak_times[tgui.time_type]  # passed by reference, then tested on in check_event_times()

        for times in start_stop_times:
            start_time, stop_time = times
            idx_to_null = np.logical_and(start_time < time_, time_ < stop_time)
            time_[idx_to_null] = np.nan

        self.check_event_times(tgui, 0)

        del dialog

    def test_omit_times_specific(self, tgui):
        """
        """
        first_event_time = tgui.adata.peak_times[tgui.time_type][0][0]

        tgui.run_artificial_events_analysis(tgui, "threshold")

        assert self.data_first_event_time(tgui) == first_event_time

        dialog = tgui.mw.dialogs["events_threshold_analyse_events"]
        tgui.enter_numbers_into_omit_times_table(dialog, [[first_event_time - 0.001,
                                                           first_event_time + 1]])

        tgui.run_artificial_events_analysis(tgui, "threshold")

        assert self.data_first_event_time(tgui) != first_event_time

        tgui.enter_numbers_into_omit_times_table(dialog, [[first_event_time + 0.001,
                                                           first_event_time + 1]])

        tgui.run_artificial_events_analysis(tgui, "threshold")

        assert self.data_first_event_time(tgui) == first_event_time

        del dialog

    def test_decay_endpoint_method_and_auc_time(self, tgui):
        """
        Check that auc times are correct event when not all the same (i.e. using decay endpoint method per event not always the same length)

        This also tests decay_search_period(ms) is working correctly (see test_decay_endpoint_method_and_auc_time())
        """
        self.add_noise_to_loaded_file_traces(tgui)
        tgui.run_artificial_events_analysis(tgui, "template")

        peak_times, decay_times, peak_to_decay_times = self.get_peak_decay_times(tgui)

        # check that decay period == decay search period
        idx = (np.abs(tgui.adata.time_array[0] - tgui.mw.cfgs.events["decay_search_period_s"])).argmin()
        true_decay_search_period_ms = tgui.adata.time_array[0][idx] * 1000

        assert utils.allclose(peak_to_decay_times, true_decay_search_period_ms, 1e-10)
        assert utils.allclose(true_decay_search_period_ms, tgui.get_time_of_monoexp_decay(tgui), 1e-10)
        self.check_auc_event_periods(tgui)

        # Change option and check that all event decay endpoint is first baseline cross
        options_dialog = tgui.open_analysis_options_dialog(tgui)
        tgui.set_combobox(options_dialog.dia.decay_endpoint_search_method_combobox, 1)

        tgui.run_artificial_events_analysis(tgui, "template")
        self.check_auc_event_periods(tgui)

        search_num_samples = core_analysis_methods.quick_get_time_in_samples(tgui.adata.ts, tgui.mw.cfgs.events["decay_search_period_s"])
        for rec in range(tgui.adata.num_recs):

            decay_point = tgui.mw.loaded_file.make_list_from_event_info("decay_point", "idx", rec=rec)
            peak_idx = tgui.mw.loaded_file.make_list_from_event_info("peak", "idx", rec=rec)
            bl_im = tgui.mw.loaded_file.make_list_from_event_info("baseline", "im", rec=rec)

            for i in range(len(bl_im)):
                test_decay_point = voltage_calc.decay_point_first_crossover_method(tgui.mw.loaded_file.data.time_array[rec],
                                                                                   tgui.mw.loaded_file.data.im_array[rec],
                                                                                   peak_idx[i],
                                                                                   search_num_samples,
                                                                                   dict(next_event_baseline_idx=None, direction=-1),
                                                                                   bl_im[i])[0]
                assert test_decay_point == decay_point[i]

        del options_dialog

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# test all threshold widgets
# ------------------------------------------------------------------------------------------------------------------------------------------
# decay_search_period (ms) tested in test_decay_endpoint_method_and_auc_time()  / tgui.get_time_of_monoexp_decay()

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    def test_amplitude_threshold(self, tgui, template_or_threshold):
        """
        """
        tgui.update_events_to_varying_amplitude_and_tau()
        tgui.run_artificial_events_analysis(tgui, template_or_threshold)

        first_event_amplitude = self.data_first_event_param(tgui, "amplitude", "im")

        threshold = abs(first_event_amplitude) + 0.01

        spinbox = tgui.mw.mw.events_threshold_amplitude_threshold_spinbox  if template_or_threshold == "threshold" else tgui.mw.mw.events_template_amplitude_threshold_spinbox
        tgui.enter_number_into_spinbox(spinbox, threshold)

        dialog = self.get_events_analysis_dialog(tgui, template_or_threshold)

        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        new_first_event_amplitude = self.data_first_event_param(tgui, "amplitude", "im")
        assert first_event_amplitude != new_first_event_amplitude
        all_amplitudes = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("amplitude", "im"))
        assert (abs(all_amplitudes) >= threshold).all()

        tgui.enter_number_into_spinbox(spinbox, 1)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)
        assert first_event_amplitude == self.data_first_event_param(tgui, "amplitude", "im")

        del dialog

    @pytest.mark.parametrize("monoexp_or_biexp", ["monoexp", "biexp"])
    def test_r2_cutoff(self, tgui, monoexp_or_biexp):
        """
        """
        param_key = monoexp_or_biexp + "_fit"
        tgui.update_events_to_varying_amplitude_and_tau()

        self.add_noise_to_loaded_file_traces(tgui, noise_divisor=2)
        options_dialog = tgui.open_analysis_options_dialog(tgui)

        if monoexp_or_biexp == "monoexp":
            idx = 0
            checkbox = options_dialog.dia.monoexp_fit_exclude_r2_checkbox
            spinbox = options_dialog.dia.monoexp_fit_exclude_r2_spinbox

        elif monoexp_or_biexp == "biexp":
            idx = 1
            checkbox = options_dialog.dia.biexp_fit_exclude_r2_checkbox
            spinbox = options_dialog.dia.biexp_fit_exclude_r2_spinbox

        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, idx)

        biexp = True if monoexp_or_biexp == "biexp" else False
        tgui.run_artificial_events_analysis(tgui, "threshold", biexp)

        first_r2 = self.data_first_event_param(tgui, param_key, "r2")
        threshold = np.ceil(first_r2 * 100) / 100

        tgui.switch_checkbox(checkbox, on=True)
        tgui.enter_number_into_spinbox(spinbox, threshold)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        assert threshold != self.data_first_event_param(tgui, param_key, "r2")
        all_r2 = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs(param_key, "r2"))
        assert (abs(all_r2) >= threshold).all()

        del options_dialog

    def get_list_of_fit_times(self, tgui, monoexp_or_biexp):
        """
        """
        times = []
        for rec in range(tgui.adata.num_recs):

            if monoexp_or_biexp == "monoexp":
                rec_time = [fit_time["monoexp_fit"]["fit_time"][0] for fit_time in tgui.mw.loaded_file.event_info[rec].values()]
            else:
                rec_time = [fit_time["biexp_fit"]["fit_time"][0] for fit_time in tgui.mw.loaded_file.event_info[rec].values()]

            times.append(rec_time)

        unpacked_times = np.array([item for sublist in times for item in sublist])

        return unpacked_times

    def test_fitting_bounds_specific_monoexp(self, tgui):
        """
        """
        tgui.update_events_to_varying_amplitude_and_tau()

        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=False)

        options_dialog = tgui.open_analysis_options_dialog(tgui)

        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 0)

        checkbox = options_dialog.dia.monoexp_exclude_outside_of_bounds_checkbox
        spinbox_low = options_dialog.dia.monoexp_min_tau_spinbox
        spinbox_high = options_dialog.dia.monoexp_max_tau_spinbox

        first_event_tau = self.data_first_event_param(tgui, "monoexp_fit", "tau_ms")

        tgui.switch_checkbox(checkbox, on=True)
        min_thr = first_event_tau + 0.001
        max_thr = first_event_tau - 0.001

        tgui.enter_number_into_spinbox(spinbox_low, min_thr)
        tgui.enter_number_into_spinbox(spinbox_high, 100000)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        all_tau = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("monoexp_fit", "tau_ms"))
        assert first_event_tau != self.data_first_event_param(tgui, "monoexp_fit", "tau_ms")
        assert (all_tau >= min_thr).all()

        tgui.enter_number_into_spinbox(spinbox_low, 0.001)
        tgui.enter_number_into_spinbox(spinbox_high, max_thr)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        all_tau = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("monoexp_fit", "tau_ms"))
        assert first_event_tau != self.data_first_event_param(tgui, "monoexp_fit", "tau_ms")
        assert (all_tau <= max_thr).all()

        tgui.switch_checkbox(checkbox, on=False)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)
        assert first_event_tau == self.data_first_event_param(tgui, "monoexp_fit", "tau_ms")

        del options_dialog

    def test_fitting_bounds_specific_biexp(self, tgui):
        """
        """
        tgui.update_events_to_varying_amplitude_and_tau()

        # setup options dialog, run analysis and get list of rise / decay times
        options_dialog = tgui.open_analysis_options_dialog(tgui)

        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 1)
        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=True)

        checkbox = options_dialog.dia.biexp_exclude_outside_of_bounds_checkbox

        first_event_rise = self.data_first_event_param(tgui, "biexp_fit", "rise_ms")
        first_event_decay = self.data_first_event_param(tgui, "biexp_fit", "decay_ms")

        tgui.switch_checkbox(checkbox, on=True)
        min_rise_thr = round(first_event_rise + 0.001, 3)
        max_rise_thr = round(first_event_rise - 0.001, 3)
        min_decay_thr = round(first_event_decay + 0.001, 3)
        max_decay_thr = round(first_event_decay - 0.001, 3)

        # Set min rise
        self.set_biexp_bounds_spinboxes(tgui, min_rise_thr, max_rise_thr=1000000, min_decay_thr=0, max_decay_thr=1000000)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        assert first_event_rise != self.data_first_event_param(tgui, "biexp_fit", "rise_ms")
        all_rise = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("biexp_fit", "rise_ms"))
        assert (all_rise >= min_rise_thr).all()

        # set max rise
        self.set_biexp_bounds_spinboxes(tgui, min_rise_thr=0, max_rise_thr=max_rise_thr, min_decay_thr=0, max_decay_thr=1000000)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        assert first_event_rise != self.data_first_event_param(tgui, "biexp_fit", "rise_ms")
        all_rise = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("biexp_fit", "rise_ms"))
        assert (all_rise <= max_rise_thr).all()

        # set min decay
        self.set_biexp_bounds_spinboxes(tgui, min_rise_thr=0, max_rise_thr=1000000, min_decay_thr=min_decay_thr, max_decay_thr=1000000)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        assert first_event_decay != self.data_first_event_param(tgui, "biexp_fit", "decay_ms")

        all_decay = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("biexp_fit", "decay_ms"))
        assert (all_decay >= min_decay_thr).all()

        # set max decay
        self.set_biexp_bounds_spinboxes(tgui, min_rise_thr=0, max_rise_thr=1000000, min_decay_thr=0, max_decay_thr=max_decay_thr)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        assert first_event_decay != self.data_first_event_param(tgui, "biexp_fit", "decay_ms")
        all_decay = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("biexp_fit", "decay_ms"))
        assert (all_decay <= max_decay_thr).all()

        # re-check origional
        tgui.switch_checkbox(checkbox, on=False)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)
        assert first_event_decay == self.data_first_event_param(tgui, "biexp_fit", "decay_ms")

        del options_dialog

    def set_biexp_bounds_spinboxes(self, tgui, min_rise_thr, max_rise_thr, min_decay_thr, max_decay_thr):
        """
        """
        options_dialog = tgui.mw.dialogs["events_analysis_options"]

        tgui.enter_number_into_spinbox(options_dialog.dia.biexp_min_rise_spinbox, min_rise_thr)
        tgui.enter_number_into_spinbox(options_dialog.dia.biexp_max_rise_spinbox, max_rise_thr)
        tgui.enter_number_into_spinbox(options_dialog.dia.biexp_min_decay_spinbox, min_decay_thr)
        tgui.enter_number_into_spinbox(options_dialog.dia.biexp_max_decay_spinbox, max_decay_thr)

    def test_decay_percent_point_is_calc_on_fit_vs_data(self, tgui):
        """
        """
        tgui.update_events_to_varying_amplitude_and_tau()
        self.add_noise_to_loaded_file_traces(tgui, noise_divisor=2)

        # setup options dialog, run analysis and get list of rise / decay times
        options_dialog = tgui.open_analysis_options_dialog(tgui)

        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 0)
        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=False)

        # Check decay HW, and decay % are calculated on the monoexponential decay
        monoexp_fit_ims = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("monoexp_fit", "fit_im"))
        monoexp_decay_perc_ims = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("decay_point", "im"))
        assert np.isin(monoexp_decay_perc_ims, monoexp_fit_ims).all()
        assert not np.isin(monoexp_decay_perc_ims, tgui.mw.loaded_file.data.im_array).all()

        monoexp_hw_decays = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("half_width", "decay_mid_im"))
        assert np.isin(monoexp_hw_decays, monoexp_fit_ims).all()

        # Check rise and decay HW and decay % are calculated on the biexponential decay
        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 1)

        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=True)

        biexp_fit_ims = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("biexp_fit", "fit_im"))
        biexp_decay_perc_ims = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("decay_point", "im"))

        assert np.isin(biexp_decay_perc_ims, biexp_fit_ims).all()
        assert not np.isin(monoexp_decay_perc_ims, biexp_decay_perc_ims).all()
        assert not np.isin(biexp_decay_perc_ims, tgui.mw.loaded_file.data.im_array).all()

        biexp_hw_rises = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("half_width", "rise_mid_im"))
        biexp_hw_decays = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("half_width", "decay_mid_im"))
        assert np.isin(biexp_hw_rises, biexp_fit_ims).all()
        assert np.isin(biexp_hw_decays, biexp_fit_ims).all()

        # Check HW and decay % are calculated on the data and this changes when smoothed (the smoothing doesn't work
        # very well on this artificial data)
        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 2)
        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=False)

        decay_perc_ims = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("decay_point", "im"))
        assert np.isin(decay_perc_ims, tgui.mw.loaded_file.data.im_array).all()

        hw_rises = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("half_width", "rise_mid_im"))
        hw_decays = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("half_width", "decay_mid_im"))

        assert np.isin(hw_decays, tgui.mw.loaded_file.data.im_array).all()
        assert np.isin(hw_rises, tgui.mw.loaded_file.data.im_array).all()

        tgui.enter_number_into_spinbox(options_dialog.dia.decay_period_smooth_spinbox, 3)
        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=False)

        decay_perc_ims = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("decay_perc", "im"))
        assert not np.isin(decay_perc_ims, tgui.mw.loaded_file.data.im_array).all()  # very basic, this is also tested in test_calclate_decay_percentage_peak_from_smoothed_decay()

        del options_dialog

    @pytest.mark.parametrize("rise_perc_low_high", [[0, 10], [25, 75], [33, 66], [55, 81], [80, 99]])
    def test_rise_times_percent_change(self, tgui, rise_perc_low_high):
        """
        The easiest way to test is to calculate expected rise time as a percentage of amplitude
        and then get the nearest true values from the actual data.
        """
        low, high = rise_perc_low_high

        # Set options and run
        tgui.update_events_to_varying_amplitude_and_tau()
        options_dialog = tgui.open_analysis_options_dialog(tgui)

        tgui.enter_number_into_spinbox(options_dialog.dia.rise_time_cutoff_low,
                                       low)
        tgui.enter_number_into_spinbox(options_dialog.dia.rise_time_cutoff_high,
                                       high)

        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=False)

        # calculate expected rise times
        amplitudes = tgui.get_data_from_qtable("amplitude", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        test_low_im, test_low_time = self.get_nearest_datapoints_to_rise_time_im_percent(tgui, amplitudes * (low / 100) + tgui.adata.resting_im)
        test_high_im, test_high_time = self.get_nearest_datapoints_to_rise_time_im_percent(tgui, amplitudes * (high / 100) + tgui.adata.resting_im)
        test_rise_times = (test_high_time - test_low_time) * 1000

        # check against table and plot
        qtable_rise_times = tgui.get_data_from_qtable("rise", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")
        assert utils.allclose(test_rise_times, qtable_rise_times, 1e-10)

        low_plot = np.array([])
        high_plot = np.array([])
        for rec in range(tgui.adata.num_recs):

            tgui.mw.update_displayed_rec(rec)

            rec_low_plot = tgui.mw.loaded_file_plot.rise_time_plot.yData[0::2]
            rec_high_plot = tgui.mw.loaded_file_plot.rise_time_plot.yData[1::2]

            low_plot = np.hstack([low_plot, rec_low_plot])
            high_plot = np.hstack([high_plot, rec_high_plot])

        assert utils.allclose(low_plot, test_low_im, 1e-10)
        assert utils.allclose(high_plot, test_high_im, 1e-10)

        del options_dialog

    def get_nearest_datapoints_to_rise_time_im_percent(self, tgui, theoretical_rise_points):
        """
        Find the nearest point to every theoretical rise point that is in the data
        """
        bl_idx = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("baseline", "idx"))
        peak_idx = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("peak", "idx"))
        recs = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("record_num", "rec_idx"))

        nearest_datapoint = []
        nearest_timepoint = []
        for i in range(len(bl_idx)):

            data_range = tgui.adata.im_array[recs[i]][bl_idx[i]:peak_idx[i] + 1]
            nearest_idx = np.argmin(np.abs(data_range - theoretical_rise_points[i]))

            nearest_datapoint.append(data_range[nearest_idx])
            nearest_timepoint.append(tgui.adata.time_array[recs[i]][bl_idx[i] + nearest_idx])

        return np.array(nearest_datapoint), np.array(nearest_timepoint)

    def test_auc_against_axograph(self, tgui):
        """
        It is not possible to match exactly as depends very precisely on baseline, start and
        end positions which is not possible to match percetly. However if it is very close
        it is working well, this is just a sanity check.
        """
        axograph_first_event_auc = -57.205
        tgui.load_a_filetype("cell_5")
        dialog = tgui.switch_to_threshold_and_open_analsis_dialog(tgui)

        tgui.switch_checkbox(tgui.mw.mw.events_threshold_average_baseline_checkbox, on=False)

        tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_decay_search_period_spinbox, 6.1)

        self.set_omit_times_for_axograph_first_events(tgui)

        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, -135)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        first_auc = tgui.mw.loaded_file.make_list_from_event_info_all_recs("area_under_curve", "im")[0]

        assert utils.allclose(axograph_first_event_auc, first_auc, 1e-1)

        del dialog

    def setup_for_event_interpolation_test(self, tgui):
        tgui.update_events_to_varying_amplitude_and_tau()
        tgui.mw.loaded_file.downsample_data(8, "bessel", 8)
        tgui.mw.reset_after_data_manipulation()

        tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_amplitude_threshold_spinbox, 5)
        dialog = tgui.switch_to_threshold_and_open_analsis_dialog(tgui)
        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, -65)

        options_dialog = tgui.open_analysis_options_dialog(tgui)
        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 2)

        return dialog, options_dialog

    def test_interpolation_for_hw_decay(self, tgui):
        """
        Quick check that interpolate is working as expected for events, in that after noise addition
        the measured rise, hw and decay percent point are closer to the true values.

        Rise cannot be tested with straight line - monoexpoential as there are no real gains as it is a straight
        line. Test rise on a biexponential in the function below.

        The underlying functions are more precisely unit-tested elsewhere.
        """
        dialog, options_dialog = self.setup_for_event_interpolation_test(tgui)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        rise_ss, decay_perc_ss, half_width_ss = self.get_ss_difference_true_and_measured_params(tgui)

        tgui.switch_checkbox(options_dialog.dia.interp_200khz_checkbox, on=True)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        rise_ss_interp, decay_perc_ss_interp, half_width_ss_interp = self.get_ss_difference_true_and_measured_params(tgui)

        assert decay_perc_ss > decay_perc_ss_interp
        assert half_width_ss > half_width_ss_interp

    def get_ss_difference_true_and_measured_params(self, tgui):
        """
        """
        test_rise_times = ((tgui.adata.rise_samples * 0.79) * tgui.adata.ts) * 1000
        test_rise_times = np.repeat(test_rise_times, tgui.adata.num_events())
        test_decay_perc = tgui.adata.get_all_decay_times()
        test_half_width = tgui.adata.get_all_half_widths()

        qtable_rise_times = tgui.get_data_from_qtable("rise", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")
        qtable_decay_perc = tgui.get_data_from_qtable("decay_perc", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")
        qtable_half_width = tgui.get_data_from_qtable("half_width", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        rise_ss = np.sum((test_rise_times - qtable_rise_times)**2)
        decay_perc_ss = np.sum((test_decay_perc - qtable_decay_perc)**2)
        half_width_ss = np.sum((test_half_width - qtable_half_width)**2)

        return rise_ss, decay_perc_ss, half_width_ss

    def test_interpolate_rise_times(self, tgui):
        """
        see test_interpolation_for_rise_hw_decay()
        """
        tgui.setup_artificial_data(norm_or_cumu_time="cumulative", analysis_type="events_multi_record_biexp_7500")
        dialog, options_dialog = self.setup_for_event_interpolation_test(tgui)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        qtable_rise_times_no_interp = tgui.get_data_from_qtable("rise", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        tgui.switch_checkbox(options_dialog.dia.interp_200khz_checkbox, on=True)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)
        qtable_rise_times_interp = tgui.get_data_from_qtable("rise", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        test_rise = (tgui.adata.rise * np.squeeze(tgui.adata.rise_offsets)) * tgui.adata.ts * 1000
        ss_rise = np.sum((test_rise - qtable_rise_times_no_interp)**2)
        ss_rise_interp = np.sum((test_rise - qtable_rise_times_interp)**2)

        assert ss_rise > ss_rise_interp

        del dialog
        del options_dialog

    def test_individual_events_panel(self, tgui):
        """
        Check that the individual event plot and GUI update properly
        """
        num_clicks = 4
        tgui.update_events_to_varying_amplitude_and_tau()

        tgui.run_artificial_events_analysis(tgui, "threshold")
        dialog = tgui.mw.dialogs["events_threshold_analyse_events"]

        assert self.displayed_event_num(dialog) == 1
        assert self.highlighted_peak_idx(tgui) == 0

        # Cycle through and check GUI updates
        for click_idx in range(num_clicks):

            self.check_disp_event_info(click_idx, dialog, tgui)

            tgui.left_mouse_click(dialog.dia.scroll_right_individual_traces_button)

        # go back one, check updates
        tgui.left_mouse_click(dialog.dia.scroll_left_individual_traces_button)

        self.check_disp_event_info(num_clicks - 1, dialog, tgui)

        # save table, delete event, check table and plot update
        qtable_amplitudes = tgui.get_data_from_qtable("amplitude", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        tgui.left_mouse_click(dialog.dia.delete_individual_trace_button)

        tgui.switch_mw_tab(1)  # need to update table
        qtable_amplitudes_del = tgui.get_data_from_qtable("amplitude", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        self.check_disp_event_info(num_clicks - 1, dialog, tgui)

        assert len(qtable_amplitudes_del) == len(qtable_amplitudes) - 1
        assert utils.allclose(qtable_amplitudes_del, np.delete(qtable_amplitudes, num_clicks - 1), 1e-10)

        del dialog

    def test_individual_events_across_recs(self, tgui):
        """
        Check that individual events GUI and delete works well across the recs boundary. Move to a
        event in new rec and check 'Go To Event' shifts to the new rec.  Also check the
        highlighted event moves one to the right when deleted.
        """
        if tgui.analysis_type == "events_one_record":
            return

        tgui.update_events_to_varying_amplitude_and_tau()

        tgui.run_artificial_events_analysis(tgui, "threshold")
        dialog = tgui.mw.dialogs["events_threshold_analyse_events"]

        num_events_first_rec = tgui.adata.num_events(rec=0)

        # move to first event on second rec, go to it and check rec changes
        tgui.enter_number_into_spinbox(dialog.dia.individual_event_number_spinbox, num_events_first_rec + 1)

        assert tgui.mw.cfgs.main["displayed_rec"] == 0
        tgui.left_mouse_click(dialog.dia.go_to_event_button)
        assert tgui.mw.cfgs.main["displayed_rec"] == 1

        # go to last event on first rec, check go to event changes back
        tgui.left_mouse_click(dialog.dia.scroll_left_individual_traces_button)
        tgui.left_mouse_click(dialog.dia.go_to_event_button)
        assert tgui.mw.cfgs.main["displayed_rec"] == 0

        # delete selected event and check individual event moves one to the right
        tgui.left_mouse_click(dialog.dia.delete_individual_trace_button)
        tgui.left_mouse_click(dialog.dia.go_to_event_button)
        assert tgui.mw.cfgs.main["displayed_rec"] == 1

        del dialog

    def check_disp_event_info(self, idx, dialog, tgui):
        assert self.displayed_event_num(dialog) == idx + 1
        assert self.highlighted_peak_idx(tgui) == idx

    def displayed_event_num(self, dialog):

        ev_num = dialog.dia.individual_event_number_spinbox.text().split(" / ")[0]

        return int(ev_num)

    def highlighted_peak_idx(self, tgui):
        """
        Updated to cycle through all recs and keep track of the absolute
        idx of the highlighted peak rather than relative to rec
        """
        green = (0, 204, 0, 255)

        track_idx = -1
        for rec in range(tgui.adata.num_recs):
            tgui.mw.update_displayed_rec(rec)
            points_on_rec = tgui.mw.loaded_file_plot.peak_plot.scatter.points()

            for point in points_on_rec:
                track_idx += 1
                if point.brush().color().getRgb() == green:
                    return track_idx

        return "none_in_file"

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Manual Event and Kinetics Selection
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_manual_event_selection(self, tgui, analyse_specific_recs, template_or_threshold):
        """
        Remove and then manually select events, checking fs latency is properly updated (see
        tgui function for more details
        """
        for filenum in range(3):

            tgui.update_events_to_varying_amplitude_and_tau()

            tgui.test_manual_spike_selection(tgui,
                                             filenum,
                                             analyse_specific_recs,
                                             run_analysis_function=lambda x: self.run_for_event_selection_function(x, template_or_threshold),
                                             spikes_to_delete_function=self.events_spikes_to_delete_function,
                                             test_function=self.switch_table_tab_and_check_event_times)

        tgui.shutdown()

    def switch_table_tab_and_check_event_times(self, tgui, filenum, rec_from, rec_to):
        tgui.switch_mw_tab(1)
        self.check_event_times(tgui, filenum, rec_from, rec_to)
        tgui.switch_mw_tab(0)

    def events_spikes_to_delete_function(self, tgui, rec_from, rec_to):

        if tgui.analysis_type == "events_one_record":  # adapt for one rec
            spikes_to_delete = [[0, 4, "m_one"], [0, 3, "m_two"],  # must be in reverse order if on the same rec
                                [0, 2, "m_three"], [0, 0, "m_four"]]
            rec_from = rec_to = None  # required format for self.check_event_times()
        else:
            spikes_to_delete = [[rec_from, 2, "m_one"], [rec_from + 6, 0, "m_two"]]

        return spikes_to_delete, rec_from, rec_to

    @staticmethod
    def run_for_event_selection_function(tgui, template_or_threshold):

        tgui.run_artificial_events_analysis(tgui, template_or_threshold)

        if template_or_threshold == "threshold":
            tgui.left_mouse_click(tgui.mw.mw.events_threshold_manually_select_button)
        elif template_or_threshold == "template":
            tgui.left_mouse_click(tgui.mw.mw.events_template_manually_select_button)

    def check_baseline_against_qtable(self, tgui, rec, rec_ev_idx, new_bl_im, avg_dialog=None):

        if avg_dialog:
            tgui.left_mouse_click(avg_dialog.dia.display_results_table_button)
            table = avg_dialog.results_table_dialog.dia.average_events_tablewidget
            assert float(table.item(2, 4).data(0)) == new_bl_im  # baseline
            avg_dialog.results_table_dialog.close()
        else:
            num_evs_per_rec = core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info, return_per_rec=True)
            abs_ev_idx = int(np.cumsum(num_evs_per_rec)[rec - 1] + rec_ev_idx) if rec > 0 else rec_ev_idx
            tgui.switch_mw_tab(1)
            qtable_baseline = tgui.get_data_from_qtable("baseline", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")[abs_ev_idx]
            assert new_bl_im == qtable_baseline
            tgui.switch_mw_tab(0)

    def get_test_edit_kinetics_objects_for_average_vs_not(self, tgui, dialog, analyse_average):
        """
        Get the objects for testing event kinetics on the average event vs. mainwindow results. These tests are very similar
        but the average event info is all on the dialog class not mainwindow class.

        Here the average event kinetic is analysed first before objects are returned.
        """
        if analyse_average:
            tgui.left_mouse_click(dialog.dia.average_all_events_button)
            avg_dialog = dialog.average_all_events_dialog
            tgui.switch_groupbox(avg_dialog.dia.calculate_event_kinetics_groupbox, on=True)
            avg_dialog.curve_fitting_region.bounds["upper_exp_lr"].setRegion((0, 100))
            tgui.left_mouse_click(avg_dialog.dia.calculate_kinetics_button)

            rec = 0
            rec_ev_idx = 0
            time_multiplier = 1000
            bl_plot = avg_dialog.event_kinetics_plots["bl_plot"]
            decay_point_plot = avg_dialog.event_kinetics_plots["decay_point_plot"]
            plot = avg_dialog.plot
            event_info = avg_dialog.event_info
            im_array = avg_dialog.average_event_y
            time_array = avg_dialog.average_event_x
        else:
            rec = 2 if tgui.analysis_type in ["events_multi_record", "events_multi_record_norm"] else 0
            rec_ev_idx = 2
            time_multiplier = 1
            bl_plot = tgui.mw.loaded_file_plot.bl_plot
            decay_point_plot = tgui.mw.loaded_file_plot.decay_point_plot
            plot = tgui.mw.loaded_file_plot.upperplot
            event_info = tgui.mw.loaded_file.event_info
            im_array = tgui.mw.loaded_file.data.im_array[rec]
            time_array = tgui.mw.loaded_file.data.time_array[rec]
            avg_dialog = None

        return bl_plot, decay_point_plot, plot, event_info, im_array, time_array, avg_dialog, rec, rec_ev_idx, time_multiplier

    @pytest.mark.parametrize("template_or_threshold", ["threshold", "template"])
    @pytest.mark.parametrize("fit_type", [["monoexp", 0], ["biexp", 1], ["no_fit", 2]])
    @pytest.mark.parametrize("analyse_average", [False, True])
    def test_manual_edit_kinetics(self, tgui, fit_type, template_or_threshold, analyse_average):
        """
        Test editing of manual kinetics. First pick an event and move the baseline, check it updates correctly and the rise re-calculated.
        Then move the decay. Check the decay is updated correctly and baseline is not changed back to origonal.

        Test for average event and normal events.
        """
        self.add_noise_to_loaded_file_traces(tgui, noise_divisor=10)  # need to add noise or euclidean distance function for click position doesn't work

        # Run Analysis
        fit_type, fit_idx = fit_type
        options_dialog = tgui.open_analysis_options_dialog(tgui)
        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, fit_idx)

        tgui.run_artificial_events_analysis(tgui, template_or_threshold)
        dialog = self.get_events_analysis_dialog(tgui, template_or_threshold)

        if template_or_threshold == "threshold":
            tgui.left_mouse_click(tgui.mw.mw.events_threshold_manually_edit_kinetics_button)

        elif template_or_threshold == "template":
            tgui.left_mouse_click(tgui.mw.mw.events_template_manually_edit_kinetics_button)

        bl_plot, decay_point_plot, plot, event_info, \
            im_array, time_array, avg_dialog, rec, rec_ev_idx, time_multiplier = self.get_test_edit_kinetics_objects_for_average_vs_not(tgui,
                                                                                                                                        dialog,
                                                                                                                                        analyse_average)
        tgui.mw.update_displayed_rec(rec)

        # move basleine on an event, check it is moved
        ev_time = list(event_info[rec].keys())[rec_ev_idx]
        starting_event_info = copy.deepcopy(event_info[rec][ev_time])

        tgui.click_upperplot_spotitem(plot=bl_plot, spotitem_idx=rec_ev_idx)

        new_bl_idx = starting_event_info["baseline"]["idx"] + 10
        new_bl_time = time_array[new_bl_idx]
        new_bl_im = im_array[new_bl_idx]

        ax = QtCore.QPointF(new_bl_time, new_bl_im)
        plot.vb.sig_events_manually_edit_kinetics_datapoint_selected.emit(ax)

        changed_bl_event_info = copy.deepcopy(event_info[rec][ev_time])
        assert new_bl_idx == changed_bl_event_info["baseline"]["idx"]
        assert new_bl_im == changed_bl_event_info["baseline"]["im"]
        assert new_bl_time == changed_bl_event_info["baseline"]["time"] * time_multiplier
        self.check_baseline_against_qtable(tgui, rec, rec_ev_idx, new_bl_im, avg_dialog)

        assert changed_bl_event_info["rise"] != starting_event_info["rise"]

        # Now change the decay point, check it is moved and check the moved baseline is not changed back to the origonal.
        tgui.click_upperplot_spotitem(plot=decay_point_plot, spotitem_idx=rec_ev_idx)

        new_decay_idx = starting_event_info["decay_point"]["idx"] - 50
        new_decay_time = time_array[new_decay_idx]
        new_decay_im = im_array[new_decay_idx]

        ax = QtCore.QPointF(new_decay_time, new_decay_im)
        plot.vb.sig_events_manually_edit_kinetics_datapoint_selected.emit(ax)

        changed_decay_event_info = copy.deepcopy(event_info[rec][ev_time])

        assert new_decay_idx == changed_decay_event_info["decay_point"]["idx"]
        assert new_decay_time == changed_decay_event_info["decay_point"]["time"] * time_multiplier

        if fit_type in ["monoexp", "biexp"]:
            assert changed_decay_event_info[fit_type + "_fit"]["fit_im"] != starting_event_info[fit_type + "_fit"]["fit_im"]
        else:
            assert new_decay_im == changed_decay_event_info["decay_point"]["im"]  # only tested without fit as with fit it is taken as the last point of fit

        assert changed_decay_event_info["baseline"]["idx"] == changed_bl_event_info["baseline"]["idx"]
        assert changed_decay_event_info["baseline"]["im"] == changed_bl_event_info["baseline"]["im"]
        assert changed_decay_event_info["baseline"]["time"] == changed_bl_event_info["baseline"]["time"]
        self.check_baseline_against_qtable(tgui, rec, rec_ev_idx, changed_bl_event_info["baseline"]["im"], avg_dialog)

        del options_dialog
        del dialog

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Average Events - Kinetics and Load, Save Event
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def setup_and_run_average_event_kinetics(self, tgui, max_slope=True):
        options_dialog = tgui.open_analysis_options_dialog(tgui)
        if max_slope:
            tgui.switch_groupbox(options_dialog.dia.max_slope_groupbox, on=True)

        tgui.run_artificial_events_analysis(tgui, "threshold")

        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.average_all_events_button)
        avg_dialog = tgui.mw.dialogs["events_threshold_analyse_events"].average_all_events_dialog
        tgui.switch_groupbox(avg_dialog.dia.calculate_event_kinetics_groupbox, on=True)
        avg_dialog.curve_fitting_region.bounds["upper_exp_lr"].setRegion((0, 50))

        tgui.left_mouse_click(avg_dialog.dia.calculate_kinetics_button)

        return options_dialog, avg_dialog

    @pytest.mark.parametrize("biexp", [False, True])
    def test_average_event_kinetics(self, tgui, biexp):
        """

        """
        __, avg_dialog = self.setup_and_run_average_event_kinetics(tgui)

        self.check_all_average_kinetics_against_adata(tgui, avg_dialog, ev_data=avg_dialog.average_event_y, monoexp=True)

        tgui.set_combobox(avg_dialog.dia.baseline_method_combobox, 1)
        self.check_all_average_kinetics_against_adata(tgui, avg_dialog, ev_data=avg_dialog.average_event_y, monoexp=True)
        tgui.left_mouse_click(avg_dialog.dia.calculate_kinetics_button)

        # calc on biexp fit and check all kinetics are on the fit
        tgui.switch_checkbox(avg_dialog.dia.fit_biexp_curve_kinetics_from_fit_checkbox, on=True)
        tgui.left_mouse_click(avg_dialog.dia.calculate_kinetics_button)
        self.check_kinetics_are_on_biexp_fit(event_info=avg_dialog.event_info[0],
                                             ev_time=list(avg_dialog.event_info[0].keys())[0],
                                             fit_data=avg_dialog.biexp_event_fit_plot.yData)

        del avg_dialog

    def run_avg_event_kinetics_analysis(self, tgui, avg_dialog, region_bounds_upper=50):
        tgui.set_combobox(avg_dialog.dia.alignment_method_combobox, 1)  # need this or it align HW and the average event changes slightly as HW on biexp is not the same as no fit
        tgui.switch_groupbox(avg_dialog.dia.calculate_event_kinetics_groupbox, on=True)
        avg_dialog.curve_fitting_region.bounds["upper_exp_lr"].setRegion((0, region_bounds_upper))
        tgui.left_mouse_click(avg_dialog.dia.calculate_kinetics_button)

    def compare_event_info_between_fit_analysis(self, tgui, stored_event_info_biexp_fit, stored_event_info_no_fit):  # TODO: tidy!

        ev_time = list(stored_event_info_no_fit.keys())[0]
        for parameter in ["peak", "baseline", "amplitude", "rise", "decay_perc", "half_width", "area_under_curve",  "event_period", "max_rise", "max_decay"]:
            QtWidgets.QApplication.processEvents()

            for key in stored_event_info_biexp_fit[ev_time][parameter].keys():

                if "idx" not in key and stored_event_info_biexp_fit[ev_time][parameter][key] is not None and \
                        stored_event_info_no_fit[ev_time][parameter][key] is not None:  # some fits

                    if type(stored_event_info_biexp_fit[ev_time][parameter][key]) != str:
                        assert utils.allclose(stored_event_info_biexp_fit[ev_time][parameter][key], stored_event_info_no_fit[ev_time][parameter][key], 1e-8)

    def run_avg_kinetics_save_event_info(self, tgui, dialog, close):
        tgui.left_mouse_click(dialog.dia.average_all_events_button)
        self.run_avg_event_kinetics_analysis(tgui, dialog.average_all_events_dialog, region_bounds_upper=500)
        stored_event_info = copy.deepcopy(dialog.average_all_events_dialog.event_info[0])
        if close:
            dialog.average_all_events_dialog.close()
        return stored_event_info

    def setup_curve_fitting_biexp_event(self, tgui):
        time_type = tgui.time_type
        tgui.shutdown()

        tgui = GuiTestSetup("artificial_events_one_record")
        tgui.setup_mainwindow(show=True)
        tgui.speed = SPEED
        tgui.test_update_fileinfo()
        tgui.setup_artificial_data(time_type, analysis_type="curve_fitting")

        tgui.update_curve_fitting_function(vary_coefs=False,
                                           insert_function="biexp_event",
                                           norm_or_cumu_time=tgui.time_type)
        return tgui

    def test_average_events_analyse_on_biexp(self, tgui):
        """
        """
        tgui = self.setup_curve_fitting_biexp_event(tgui)

        # set options for fitting curve fitting biexponential
        dialog = tgui.switch_to_threshold_and_open_analsis_dialog(tgui)
        tgui.set_combobox(tgui.mw.mw.events_threshold_peak_direction_combobox, 0)
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_decay_search_period_spinbox, get_settings(tgui.speed,
                                                                                                "curve_fitting")["decay_search_period"])
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_amplitude_threshold_spinbox, 2)
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_baseline_search_period_spinbox, 100)
        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, -59)

        # Run analysis with no fit and store the average event
        options_dialog = tgui.open_analysis_options_dialog(tgui)
        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 2)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)
        stored_event_info_no_fit = self.run_avg_kinetics_save_event_info(tgui, dialog, close=True)

        # run analysis with biexp fit and check the results match exactly the version with no fit. This works because we have a perfect
        # biexponential as the test event and so the kinetics calculated on the biexp fit should be identical to no fit.
        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 1)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        stored_event_info_biexp_fit = self.run_avg_kinetics_save_event_info(tgui, dialog, close=False)
        self.compare_event_info_between_fit_analysis(tgui, stored_event_info_biexp_fit, stored_event_info_no_fit)

        # Finally, run an event analysis with the biexponential fit options. Check that each kinetic is on the biexp fit (rather than the raw data),
        # that the results match the pervious perfectly (they should as calculated on the same function), and that the fit is perfect.
        avg_dialog = dialog.average_all_events_dialog
        self.run_avg_event_kinetics_analysis(tgui, avg_dialog, region_bounds_upper=500)

        ev_time = list(avg_dialog.event_info[0].keys())[0]
        avg_dialog.curve_fitting_region.bounds["upper_exp_lr"].setRegion((avg_dialog.event_info[0][ev_time]["baseline"]["time"] * 1000,
                                                                          avg_dialog.event_info[0][ev_time]["decay_point"]["time"] * 1000))
        tgui.switch_checkbox(avg_dialog.dia.fit_biexp_curve_kinetics_from_fit_checkbox, on=True)
        tgui.left_mouse_click(avg_dialog.dia.calculate_kinetics_button)

        self.compare_event_info_between_fit_analysis(tgui, stored_event_info_biexp_fit, avg_dialog.event_info[0])
        self.check_kinetics_are_on_biexp_fit(event_info=avg_dialog.event_info[0],
                                             ev_time=ev_time,
                                             fit_data=avg_dialog.biexp_event_fit_plot.yData)

        tgui.left_mouse_click(avg_dialog.dia.display_results_table_button)
        table = avg_dialog.results_table_dialog.dia.average_events_tablewidget
        assert float(table.item(2, 16).data(0)) == 1
        avg_dialog.results_table_dialog.close()

        del dialog
        del options_dialog
        del avg_dialog
        tgui.shutdown()

    def check_kinetics_are_on_biexp_fit(self, event_info, ev_time, fit_data):
        """
        """
        for kinetic in [["peak"], ["baseline"], ["rise", "min_"], ["rise", "max_"], ["decay_perc"], ["half_width", "rise_mid_"], ["half_width", "decay_mid_"]]:  # decay point is on fit measure

            key_1 = kinetic[0]
            key_2 = "im" if len(kinetic) == 1 else kinetic[1] + "im"

            assert event_info[ev_time][key_1][key_2] in fit_data

    def check_all_average_kinetics_against_adata(self, tgui, avg_dialog, ev_data, monoexp):

        tgui.left_mouse_click(avg_dialog.dia.display_results_table_button)
        table = avg_dialog.results_table_dialog.dia.average_events_tablewidget

        assert table.item(2, 0).data(0) == "1"                                                                                                              # event num
        assert table.item(2, 1).data(0) == "Thr."                                                                                                           # template
        assert table.item(2, 2).data(0) == "1"                                                                                                              # record
        assert table.item(2, 3).data(0) == "n/a"                                                                                                            # event time
        assert float(table.item(2, 4).data(0)) == tgui.adata.resting_im                                                                                     # baseline
        assert float(table.item(2, 5).data(0)) == tgui.adata.peak_im                                                                                        # peak
        assert float(table.item(2, 6).data(0)) == tgui.adata.b1                                                                                             # amplitude
        assert utils.allclose(float(table.item(2, 7).data(0)), ((tgui.adata.rise_samples * 0.79) * tgui.adata.ts) * 1000, 1e-10)                            # note rise times are all the same even if adjusting so this is not a great test   MAKE OWN FUNCTION                     # rise times
        assert utils.allclose(float(table.item(2, 8).data(0)), tgui.adata.get_all_half_widths()[0], 1e-10)                                                  # half width
        assert utils.allclose(float(table.item(2, 9).data(0)), tgui.adata.get_all_decay_times()[0], 1e-10)                                                  # decay %

        assert utils.allclose(float(table.item(2, 10).data(0)), tgui.adata.area_under_curves[0] * 1000, 1e-10)
        assert utils.allclose(float(table.item(2, 11).data(0)), self.get_auc_event_periods(tgui)[0], 1e-10)

        max_rise = np.min(np.diff(ev_data)) / (tgui.adata.ts * 1000)
        max_decay = np.max(np.diff(ev_data)) / (tgui.adata.ts * 1000)
        assert utils.allclose(float(table.item(2, 12).data(0)), max_rise, 1e-10)                                                                            # decay_fit b0
        assert utils.allclose(float(table.item(2, 13).data(0)), max_decay, 1e-10)

        assert utils.allclose(float(table.item(2, 14).data(0)), tgui.adata.resting_im, 1e-10)                                                               # decay_fit b0
        assert utils.allclose(float(table.item(2, 15).data(0)), tgui.adata.b1, 1e-10)                                                                       # decay_fit b1

        test_tau_ms = self.get_test_tau_ms(tgui)
        assert utils.allclose(float(table.item(2, 16).data(0)), test_tau_ms, 1e-10)                                                                         # decay_fit tau

        if monoexp:                                                                                                                                         # decay_fit R2
            assert float(table.item(2, 17).data(0)) == 1

        avg_dialog.results_table_dialog.close()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Average Events - Kinetics
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_curve_fitting_kinetics_table(self, tgui):

        tgui.set_analysis_type("curve_fitting")
        tgui.left_mouse_click(tgui.mw.mw.curve_fitting_show_dialog_button)
        first_peak_time = tgui.adata.peak_times[tgui.time_type][0][0]

        tgui.switch_checkbox(tgui.mw.dialogs["curve_fitting"].dia.hide_baseline_radiobutton, on=True)

        ev_stop_time = first_peak_time + tgui.adata.ts * 550

        tgui.mw.curve_fitting_regions["reg_1"].bounds["upper_exp_lr"].setRegion((first_peak_time - 0.005,
                                                                                 ev_stop_time))

        tgui.set_combobox(tgui.mw.dialogs["curve_fitting"].dia.fit_type_combobox, 8)
        tgui.switch_checkbox(tgui.mw.dialogs["curve_fitting"].dia.max_slope_direction_pos_radiobutton, on=True)

        tgui.mw.cfgs.curve_fitting["analysis"]["reg_1"]["biexp_event"]["direction"] = -1

        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.curve_fitting_event_kinetics_options)
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.baseline_search_period_spinbox, "25")  # TODO: DO SOMETHING ABOUT DRY WITH ABOVE
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)

        table = tgui.mw.mw.table_tab_tablewidget
        assert float(table.item(2, 6).data(0)) == tgui.adata.resting_im                                                                             # baseline
        assert float(table.item(2, 7).data(0)) == tgui.adata.peak_im                                                                                # peak
        assert float(table.item(2, 8).data(0)) == tgui.adata.b1                                                                                     # amplitude
        assert utils.allclose(float(table.item(2, 9).data(0)), ((tgui.adata.rise_samples * 0.79) * tgui.adata.ts) * 1000, 1e-10)                    # note rise times are all the same even if adjusting so this is not a great test   MAKE OWN FUNCTION                     # rise times
        assert utils.allclose(float(table.item(2, 10).data(0)), tgui.adata.get_all_half_widths()[0], 1e-10)                                         # half width
        assert utils.allclose(float(table.item(2, 11).data(0)), tgui.adata.get_all_decay_times(), 1e-10)                                            # decay %

        assert utils.allclose(float(table.item(2, 12).data(0)), tgui.adata.area_under_curves[0] * 1000, 1e-4)                                       # area under curve
        ev_bl_time = self.get_cf_event_info(tgui, rec=0, region="reg_1")["baseline"]["time"]
        test_auc_time = (ev_stop_time - ev_bl_time) * 1000
        assert utils.allclose(float(table.item(2, 13).data(0)), test_auc_time, 1e-10)                                                               # area under curve time

        assert table.item(2, 14).data(0) == "off"                                                                                                   # max slope
        assert table.item(2, 15).data(0) == "off"                                                                                                   # max slope

        assert utils.allclose(float(table.item(2, 16).data(0)), tgui.adata.resting_im, 1e-6)                                                        # decay_fit b0
        assert utils.allclose(float(table.item(2, 17).data(0)), tgui.adata.b1, 1e-6)                                                                # decay_fit b1

        test_tau_ms = self.get_test_tau_ms(tgui)
        assert utils.allclose(float(table.item(2, 18).data(0)), test_tau_ms, 1e-6)                                                                  # decay_fit tau
        assert utils.allclose(float(table.item(2, 19).data(0)), 1, 1e-6)                                                                            # r2

    def get_cf_event_info(self, tgui, rec, region, extract_time=True):
        event_info = tgui.mw.loaded_file.curve_fitting_results[region]["data"][rec]["event_info"]  # OWN FUNCTION
        if extract_time and event_info:
            ev_time = list(event_info.keys())[0]  # own function and above
            event_info = event_info[ev_time]
        return event_info

    def interleave_same_size_arrays(self, array_1, array_2):
        new_array = np.empty(array_1.size * 2, dtype=array_1.dtype)
        new_array[1::2] = array_1
        new_array[0::2] = array_2
        return new_array

    # analyse specific recs
    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_event_kinetics_plots(self, tgui, analyse_specific_recs):
        """
        peak plots tested elsewhere
        """
        tgui.update_events_to_varying_amplitude_and_tau()
        self.setup_max_slope_events(tgui)
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_amplitude_threshold_spinbox, 1)

        analyse_specific_recs = analyse_specific_recs if tgui.analysis_type in ["events_multi_record", "events_multi_record_norm"] else False  # wasteful
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)


        tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_threshold_analyse_events"].dia.threshold_lower_spinbox, -65)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        main_plot = tgui.mw.loaded_file_plot
        for rec in range(tgui.adata.num_recs):

            tgui.mw.update_displayed_rec(rec)

            if not rec_from <= rec <= rec_to:
                for plot in [main_plot.max_rise_plot, main_plot.max_decay_plot, main_plot.half_width_plot, main_plot.rise_time_plot,
                             main_plot.bl_plot, main_plot.decay_fit_plot, main_plot.decay_percent_plot, main_plot.decay_point_plot,
                             main_plot.peak_plot]:  # MOVE / TIDY

                    assert not np.any(plot.xData)
                    assert not np.any(plot.yData)
            else:
                for info in [[main_plot.max_rise_plot, "max_rise", "fit_time", "fit_data", False],
                             [main_plot.max_decay_plot, "max_decay", "fit_time", "fit_data", False],
                             [main_plot.half_width_plot , "half_width", "rise_midtime", "rise_mid_im", ["decay_midtime", "decay_mid_im"]],
                             [main_plot.rise_time_plot, "rise", "min_time", "min_im", ["max_time", "max_im"]],
                             [main_plot.bl_plot , "baseline", "time", "im", False],
                             [main_plot.decay_fit_plot , "monoexp_fit", "fit_time", "fit_im", False],
                             [main_plot.decay_percent_plot, "decay_perc", "time", "im", False],
                             [main_plot.decay_point_plot, "decay_point", "time", "im", False],
                             [main_plot.peak_plot, "peak", "time", "im", False]]:

                    plot, key_1, time_key, data_key, combine = info
                    self.check_kinetics_event_info_against_plots(tgui, rec, plot, key_1, time_key, data_key, combine)

    def check_kinetics_event_info_against_plots(self, tgui, rec, plot, key_1, time_key, data_key, combine, event_info=None, convert_to_ms=False):

        time_, im = self.get_all_rec_data(tgui, rec, key_1, time_key, data_key, event_info)

        if combine:
            time_2, im_2 = self.get_all_rec_data(tgui, rec, key_1, combine[0], combine[1], event_info)
            im = self.interleave_same_size_arrays(im_2, im)
            time_ = self.interleave_same_size_arrays(time_2, time_)

        if convert_to_ms:
            time_ *= 1000

        assert np.array_equal(time_, plot.xData)
        assert np.array_equal(im, plot.yData)

    def get_all_rec_data(self, tgui, rec, key_1, key_time, key_data, event_info=None):  # TODO: use hstack in actual software!

        data = []
        times = []

        if not event_info:
            event_info = tgui.mw.loaded_file.event_info[rec]

        for ev in event_info.values():

            data.append(ev[key_1][key_data])
            times.append(ev[key_1][key_time])

        return np.hstack(times), np.hstack(data)

    def test_average_events_plots(self, tgui):
        """
        """
        __, avg_dialog = self.setup_and_run_average_event_kinetics(tgui)

        ev_plots = avg_dialog.event_kinetics_plots
        rec = 0

        for info in [[ev_plots["max_rise_plot"], "max_rise", "fit_time", "fit_data", False],  # COMBINE WITH ABOVE!!
                     [ev_plots["max_decay_plot"], "max_decay", "fit_time", "fit_data", False],
                     [ev_plots["half_width_plot"], "half_width", "rise_midtime", "rise_mid_im", ["decay_midtime", "decay_mid_im"]],
                     [ev_plots["rise_time_plot"], "rise", "min_time", "min_im", ["max_time", "max_im"]],
                     [ev_plots["bl_plot"], "baseline", "time", "im", False],
                     [ev_plots["decay_fit_plot"], "monoexp_fit", "fit_time", "fit_im", False],
                     [ev_plots["decay_percent_plot"], "decay_perc", "time", "im", False],
                     [ev_plots["decay_point_plot"], "decay_point", "time", "im", False],
                     [ev_plots["peak_plot"], "peak", "time", "im", False]]:

            plot, key_1, time_key, data_key, combine = info
            self.check_kinetics_event_info_against_plots(tgui, rec, plot, key_1, time_key, data_key, combine, event_info=avg_dialog.event_info[0], convert_to_ms=True)

        del avg_dialog

    def test_curve_fitting_event_kinetics_plots(self, tgui):

        tgui = self.setup_curve_fitting_biexp_event(tgui)

        for region in ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]:  # ADD

            test_curve_fitting.setup_and_run_curve_fitting_analysis(tgui, func_type="biexp_event", region_name=region, rec_from=0, rec_to=tgui.adata.num_recs, set_options_only=False)

            tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.curve_fitting_event_kinetics_options)
            ev_opts_dialog = tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog
            tgui.enter_number_into_spinbox(ev_opts_dialog.dia.baseline_search_period_spinbox, 100, setValue=True)
            tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)

            ev_plots = tgui.mw.loaded_file_plot.curve_fitting_event_kinetics_plots[region]
            for rec in range(tgui.adata.num_recs):

                tgui.mw.update_displayed_rec(rec)
                event_info = tgui.mw.loaded_file.curve_fitting_results[region]["data"][rec]["event_info"]

                for info in [
                             [ev_plots["half_width_plot"], "half_width", "rise_midtime", "rise_mid_im", ["decay_midtime", "decay_mid_im"]],
                             [ev_plots["rise_time_plot"], "rise", "min_time", "min_im", ["max_time", "max_im"]],
                             [ev_plots["bl_plot"], "baseline", "time", "im", False],
                             [ev_plots["decay_fit_plot"], "monoexp_fit", "fit_time", "fit_im", False],
                             [ev_plots["decay_percent_plot"], "decay_perc", "time", "im", False],
                             [ev_plots["decay_point_plot"], "decay_point", "time", "im", False],
                             [ev_plots["peak_plot"], "peak", "time", "im", False]]:
                    plot, key_1, time_key, data_key, combine = info
                    self.check_kinetics_event_info_against_plots(tgui, rec, plot, key_1, time_key, data_key, combine, event_info=event_info)

            del ev_opts_dialog

        tgui.shutdown()

    @pytest.mark.parametrize("decay_percent", [25, 50, 75])  # tested in more detail in test_calclate_decay_percentage_peak_from_smoothed_decay()
    def test_decay_percents(self, tgui, decay_percent):
        """
        """
        tgui.update_events_to_varying_amplitude_and_tau()
        options_dialog = tgui.open_analysis_options_dialog(tgui)

        tgui.enter_number_into_spinbox(options_dialog.dia.decay_amplitude_perc_spinbox,
                                       decay_percent)

        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=False)

        test_decay_perc = tgui.adata.get_all_decay_times(decay_percent)

        qtable_decay_perc = tgui.get_data_from_qtable("decay_perc", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert utils.allclose(test_decay_perc, qtable_decay_perc, 1e-10)

        del options_dialog

    def get_quick_half_width(self, tgui, rec, event_info):
        mid = event_info["baseline"]["im"] + (event_info["peak"]["im"] - event_info["baseline"]["im"]) / 2
        rise_idx = np.argmin(np.abs(tgui.adata.im_array[rec][:event_info["peak"]["idx"]] - mid))
        decay_idx = event_info["peak"]["idx"] + np.argmin(np.abs(tgui.adata.im_array[rec][event_info["peak"]["idx"]:] - mid))
        return decay_idx - rise_idx

    @pytest.mark.parametrize("region", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_kinetics_all_options(self, tgui, region):  # dont check manual baseline
        """
        """
        tgui = self.setup_curve_fitting_biexp_event(tgui)
        self.add_noise_to_loaded_file_traces(tgui, 25)

        test_curve_fitting.setup_and_run_curve_fitting_analysis(tgui, func_type="biexp_event", region_name=region, rec_from=0, rec_to=tgui.adata.num_recs, set_options_only=False)

        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.curve_fitting_event_kinetics_options)
        ev_opts_dialog = tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog

        tgui.enter_number_into_spinbox(ev_opts_dialog.dia.baseline_search_period_spinbox, 100, setValue=True)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)

        for rec in range(tgui.adata.num_recs):
            event_info = self.get_cf_event_info(tgui, rec, region)

            quick_half_width = self.get_quick_half_width(tgui, rec, event_info)
            measured_half_width = event_info["half_width"]["decay_mid_idx"] - event_info["half_width"]["rise_mid_idx"]
            assert(measured_half_width < quick_half_width and measured_half_width > quick_half_width - 5)  # the half width is measured on monoexp rather
                                                                                                           # that true data so just give an arbitary range.

            tgui.switch_checkbox(ev_opts_dialog.dia.average_baseline_checkbox, on=True)
            tgui.enter_number_into_spinbox(ev_opts_dialog.dia.average_baseline_spinbox, 10, setValue=True)
            tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)
            event_info = self.get_cf_event_info(tgui, rec, region)


            bl_search_samples = core_analysis_methods.quick_get_time_in_samples(tgui.adata.ts, 10 / 1000)  # check
            bl_idx = event_info["baseline"]["idx"]
            test_avg_baseline = np.mean(tgui.mw.loaded_file.data.im_array[rec][bl_idx - bl_search_samples:bl_idx + 1])

            assert test_avg_baseline == event_info["baseline"]["im"]

            peak_idx = event_info["peak"]["idx"]
            assert tgui.mw.loaded_file.data.im_array[rec][peak_idx] == event_info["peak"]["im"]

        three_samples_in_ms = (tgui.adata.ts * 3) * 1000
        tgui.switch_checkbox(ev_opts_dialog.dia.average_peak_checkbox, on=True)
        tgui.enter_number_into_spinbox(ev_opts_dialog.dia.average_peak_spinbox, three_samples_in_ms, setValue=True)  # round(three_samples_in_ms, 2)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)

        for rec in range(tgui.adata.num_recs):
            event_info = self.get_cf_event_info(tgui, rec, region)

            window = np.floor(core_analysis_methods.quick_get_time_in_samples(tgui.adata.ts, tgui.mw.cfgs.events["decay_search_period_s"]) / 4).astype(int)

            __, __, test_avg_peak = voltage_calc.find_event_peak_after_smoothing(tgui.mw.loaded_file.data.time_array[rec], tgui.mw.loaded_file.data.im_array[rec], peak_idx, window, samples_to_smooth=3, direction=1)

            assert utils.allclose(test_avg_peak, event_info["peak"]["im"])

        tgui.switch_checkbox(ev_opts_dialog.dia.calculate_kinetics_from_fit_not_data_checkbox, on=True)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)

        for rec in range(tgui.adata.num_recs):
            event_info = self.get_cf_event_info(tgui, rec, region, extract_time=False)

            self.check_kinetics_are_on_biexp_fit(event_info=event_info,
                                                 ev_time=list(event_info.keys())[0],
                                                 fit_data=tgui.mw.loaded_file.curve_fitting_results[region]["data"][rec]["fit"])

        del ev_opts_dialog
        tgui.shutdown()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Max Slope Tests
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def setup_max_slope_events(self, tgui, smooth=False, use_baseline_crossing=False):
        options_dialog = tgui.setup_max_slope_events(tgui, smooth, use_baseline_crossing)
        return options_dialog

    def set_omit_times_for_axograph_first_events(self, tgui):
        """
        We only need the first couple of events for the axograph
        """
        dialog = self.get_events_analysis_dialog(tgui, "threshold")
        tgui.enter_numbers_into_omit_times_table(dialog, [[1, tgui.mw.loaded_file.data.t_stop]])

    def test_max_slope_events(self, tgui):
        """
        Tested against Axograph implementation
        """
        axograph_first_event_rise_3, axograph_first_event_decay_3, \
            axograph_first_event_rise_5, axograph_first_event_decay_7 = tgui.get_axograph_max_slope_to_test_against()

        tgui.load_a_filetype("cell_5")

        self.setup_max_slope_events(tgui, use_baseline_crossing=True)
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_threshold_analyse_events"].dia.threshold_lower_spinbox, -180)

        self.set_omit_times_for_axograph_first_events(tgui)

        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        np.round(tgui.mw.loaded_file.make_list_from_event_info_all_recs("max_rise", "max_slope_ms")[0], 3)

        assert axograph_first_event_rise_3 == np.round(tgui.mw.loaded_file.make_list_from_event_info_all_recs("max_rise", "max_slope_ms")[0], 3)
        assert axograph_first_event_decay_3 == np.round(tgui.mw.loaded_file.make_list_from_event_info_all_recs("max_decay", "max_slope_ms")[0], 4)

        tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_analysis_options"].dia.max_slope_num_samples_rise_spinbox, 5)
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_analysis_options"].dia.max_slope_num_samples_decay_spinbox, 7)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        assert axograph_first_event_rise_5 == np.round(tgui.mw.loaded_file.make_list_from_event_info_all_recs("max_rise", "max_slope_ms")[0], 3)
        assert axograph_first_event_decay_7 == np.round(tgui.mw.loaded_file.make_list_from_event_info_all_recs("max_decay", "max_slope_ms")[0], 4)

    @pytest.mark.parametrize("use_first_baseline_crossing", [False, True])
    def test_max_slope_with_smoothing(self, tgui, use_first_baseline_crossing):
        """
        Check the GUI side for max slope with smoothing. Run analysis with max rise / decay slope on smoothing
        then re-run here and check results match. Also tests 'Always use baseline crossing as max slope search endpoint'
        in a convenient way. If working correctly, max slope here will match the GUI because the decay_point idx will match
        as in setup_max_slope_events(), decay_endpoint_search_method is set to match the max slope decay method.
        """
        tgui.update_events_to_varying_amplitude_and_tau()
        self.setup_max_slope_events(tgui, smooth=True, use_baseline_crossing=use_first_baseline_crossing)

        tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_threshold_analyse_events"].dia.threshold_lower_spinbox, -65)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        test_max_rise, test_max_decay = self.get_max_slope_test_results(tgui)

        model_max_rise = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "max_rise", "max_slope_ms")
        tabledata_max_rise = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[0], "max_rise", "max_slope_ms")
        qtable_max_rise = tgui.get_data_from_qtable("max_rise", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert utils.allclose(test_max_rise, model_max_rise, 1e-10)
        assert utils.allclose(test_max_rise, tabledata_max_rise, 1e-10)
        assert utils.allclose(test_max_rise, qtable_max_rise, 1e-10)

        model_max_decay = self.unpack_event_info_values(tgui.mw.loaded_file.event_info, "max_decay", "max_slope_ms")
        tabledata_max_decay = self.unpack_event_info_values(tgui.mw.stored_tabledata.event_info[0], "max_decay", "max_slope_ms")
        qtable_max_decay = tgui.get_data_from_qtable("max_decay", row_from=0, row_to=tgui.adata.num_events(), analysis_type="events")

        assert utils.allclose(test_max_decay, model_max_decay, 1e-3)
        assert utils.allclose(test_max_decay, tabledata_max_decay, 1e-3)
        assert utils.allclose(test_max_decay, qtable_max_decay, 1e-3)

    def get_max_slope_test_results(self, tgui):

        bl_idx = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("baseline", "idx"))
        decay_idx = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("decay_point", "idx"))
        peak_idx = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("peak", "idx"))
        recs = np.hstack(tgui.mw.loaded_file.make_list_from_event_info_all_recs("record_num", "rec_idx"))

        test_max_rise = []
        test_max_decay = []
        for i in range(len(bl_idx)):

            rise_max_slope, __, __ = core_analysis_methods.calculate_max_slope_rise_or_decay(tgui.adata.time_array[recs[i]],
                                                                                             tgui.adata.im_array[recs[i]],
                                                                                             bl_idx[i],
                                                                                             peak_idx[i],
                                                                                             window_samples=3,
                                                                                             ts=tgui.adata.ts,
                                                                                             smooth_settings={"on": True, "num_samples": 2},
                                                                                             argmax_func=np.argmin)
            test_max_rise.append(rise_max_slope)

            decay_max_slope, __, __ = core_analysis_methods.calculate_max_slope_rise_or_decay(tgui.adata.time_array[recs[i]],
                                                                                              tgui.adata.im_array[recs[i]],
                                                                                              peak_idx[i],
                                                                                              decay_idx[i],
                                                                                              window_samples=3,
                                                                                              ts=tgui.adata.ts,
                                                                                              smooth_settings={"on": True, "num_samples": 2},
                                                                                              argmax_func=np.argmax)
            test_max_decay.append(decay_max_slope)

        return np.array(test_max_rise), np.array(test_max_decay)

    @pytest.mark.parametrize("monoexp_or_biexp", ["monoexp", "biexp"])
    def test_adjust_start_point_to_improve_fit(self, tgui, monoexp_or_biexp):

        param_key = monoexp_or_biexp + "_fit"
        tgui.update_events_to_varying_amplitude_and_tau()

        self.add_noise_to_loaded_file_traces(tgui, noise_divisor=2)
        options_dialog = tgui.open_analysis_options_dialog(tgui)

        if monoexp_or_biexp == "monoexp":
            idx = 0
            checkbox = options_dialog.dia.monoexp_fit_adjust_start_point_checkbox
            spinbox = options_dialog.dia.monoexp_fit_adjust_start_point_spinbox

        elif monoexp_or_biexp == "biexp":
            idx = 1
            checkbox = options_dialog.dia.monoexp_fit_adjust_start_point_checkbox
            spinbox = options_dialog.dia.biexp_fit_adjust_start_point_spinbox

        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, idx)

        biexp = True if monoexp_or_biexp == "biexp" else False
        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=biexp, overide_biexp_adjust_start_point=True)

        all_fit_times = self.get_list_of_fit_times(tgui, monoexp_or_biexp)
        all_r2 = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs(param_key, "r2"))

        tgui.switch_checkbox(checkbox, on=True)
        tgui.enter_number_into_spinbox(spinbox, "3")

        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=biexp)

        new_fit_times = self.get_list_of_fit_times(tgui, monoexp_or_biexp)
        new_r2 = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs(param_key, "r2"))

        idx_same = np.where(all_fit_times == new_fit_times)
        idx_less = np.where(all_fit_times < new_fit_times)
        idx_more = np.where(all_fit_times > new_fit_times)

        assert (new_r2[idx_same] == all_r2[idx_same]).all()
        assert (new_r2[idx_less] > all_r2[idx_less]).all()
        assert (new_r2[idx_more] > all_r2[idx_more]).all()

        if monoexp_or_biexp == "biexp":
            assert (new_fit_times > all_fit_times).any()

        del options_dialog

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    def test_lower_threshold_gui(self, tgui, template_or_threshold):

        # Run and setup
        tgui.run_artificial_events_analysis(tgui, template_or_threshold)
        num_events = core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info)
        assert any(tgui.mw.loaded_file.event_info) is True

        peak_val = tgui.adata.peak_im
        dialog_key = "template_analyse_events" if template_or_threshold == "template" else "events_threshold_analyse_events"  # OWN FUNCTION
        dialog = tgui.mw.dialogs[dialog_key]

        # Set the linear threshold to just below the peak val - should be no spikes
        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, peak_val - 0.01)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)
        assert any(tgui.mw.loaded_file.event_info) is False

        # Set the spinbox to curved. find the minimum values of the curve and set it so this is close to the peak val but number.
        # Analyse and check all are within threshold, then set just over threshold and check none are found. Tried to make it a
        # bit more specific per-spike but its too complex with different curves per rec. This is unit tested more thoroughly.
        tgui.set_combobox(dialog.dia.threshold_lower_combobox, 1)
        min_curve_pos = np.min(np.min(dialog.curved_threshold_lower_w_displacement, axis=0))
        offset = peak_val - min_curve_pos
        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, offset)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)
        new_num_events = core_analysis_methods.total_num_events(tgui.mw.loaded_file.event_info)
        assert new_num_events == num_events

        tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox, offset - 5)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)
        assert any(tgui.mw.loaded_file.event_info) is False

        del dialog

    @pytest.mark.parametrize("template_or_threshold", ["template", "threshold"])
    def test_area_under_curve_threshold(self, tgui, template_or_threshold):
        """
        TODO: very similar to above
        """
        tgui.update_events_to_varying_amplitude_and_tau()
        tgui.run_artificial_events_analysis(tgui, template_or_threshold)

        first_event_auc = self.data_first_event_param(tgui, "area_under_curve", "im")

        threshold = abs(first_event_auc) + 0.01

        checkbox = tgui.mw.mw.events_threshold_area_under_curve_checkbox if template_or_threshold == "threshold" else tgui.mw.mw.events_template_area_under_curve_checkbox
        spinbox = tgui.mw.mw.events_threshold_area_under_curve_spinbox if template_or_threshold == "threshold" else tgui.mw.mw.events_template_area_under_curve_spinbox

        tgui.switch_checkbox(checkbox, on=True)
        tgui.enter_number_into_spinbox(spinbox, threshold)

        dialog = self.get_events_analysis_dialog(tgui, template_or_threshold)

        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        new_first_event_auc = self.data_first_event_param(tgui, "area_under_curve", "im")
        assert first_event_auc != new_first_event_auc
        all_amplitudes = np.array(tgui.mw.loaded_file.make_list_from_event_info_all_recs("area_under_curve", "im"))
        assert (abs(all_amplitudes) >= threshold).all()

        tgui.enter_number_into_spinbox(spinbox, 1)
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        assert first_event_auc == self.data_first_event_param(tgui, "area_under_curve", "im")

        del dialog
