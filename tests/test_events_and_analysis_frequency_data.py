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
from ephys_data_methods import core_analysis_methods, event_analysis_master
from ephys_data_methods_private import curve_fitting_master
from utils import utils
from setup_test_suite import GuiTestSetup
import copy
from sys import platform
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
os.environ["PYTEST_QT_API"] = "pyside2"

SPEED = "fast"

class TestEvents:

    @pytest.fixture(scope="function", params=["multi_record", "one_record"], ids=["multi_record", "one_record"])
    def tgui(test, request):
        tgui = GuiTestSetup("artificial_events_" + request.param)
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = SPEED
        tgui.setup_artificial_data("cumulative", analysis_type="events_" + request.param)
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    # --------------------------------------------------------------------------------------------------------------------------------------
    # Test Kolmogorov-Smirnov
    # --------------------------------------------------------------------------------------------------------------------------------------

    def test_ks_test_gui(self, tgui):

        dialog, __ = self.setup_and_fill_ks_test_one_dataset(tgui)

        assert dialog.user_input_mean == 0
        assert dialog.user_input_stdev == 0.0001
        assert dialog.analyse_one_or_two_columns == "one"
        assert dialog.one_sample_method == "lilliefors"
        assert dialog.alternative_hypothesis == "two-sided"
        assert not dialog.dia.one_dataset_sample_mean_spinbox.isEnabled()
        assert not dialog.dia.one_dataset_sample_stdev_spinbox.isEnabled()
        assert dialog.dia.analysis_table.isColumnHidden(1)

        tgui.set_combobox(dialog.dia.one_dataset_method_combobox, idx=1)

        assert dialog.one_sample_method == "user_input_population"
        assert dialog.dia.one_dataset_sample_mean_spinbox.isEnabled()
        assert dialog.dia.one_dataset_sample_stdev_spinbox.isEnabled()

        tgui.enter_number_into_spinbox(dialog.dia.one_dataset_sample_mean_spinbox, 5)
        tgui.enter_number_into_spinbox(dialog.dia.one_dataset_sample_stdev_spinbox, 10)

        tgui.switch_groupbox(dialog.dia.two_datasets_groupbox, on=True)

        assert not dialog.dia.one_dataset_groupbox.isChecked()
        assert not dialog.dia.analysis_table.isColumnHidden(1)

        for idx, option in enumerate(["two-sided", "greater", "less"]):
            tgui.set_combobox(dialog.dia.alternative_hypothesis_combobox, idx=idx)
            assert dialog.alternative_hypothesis == option

        del dialog

    def test_one_dataset_ks_test_lillifores(self, tgui):
        """
        p is derived from intepolating the distribution from monte-carlo simulations
        as such p values do not match exactly.

        SPSS: D = 0.012 p = 0.983
        MATLAB: D = 0.0122 p = 0.9837
        """
        dialog, data = self.setup_and_fill_ks_test_one_dataset(tgui, provide_parameters=False)

        results = self.get_ks_test_results_from_gui(dialog)

        assert results["name_1"] == "Dataset 1"
        assert results["name_2"] == "Normal CDF"
        assert results["hypothesis"] == "Two-Tailed"

        assert results["n"] == "1000"
        assert results["D"] == "0.0122"
        assert results["p"] == "0.98894247"

        del dialog

    def test_one_dataset_ks_test_provide_parameters(self, tgui):
        """
        see test_one_dataset_ks_test_lillifores()

        SPSS: D = 0.021, p = 0.751

        MATLAB: D = 0.214 p = 0.7427
        """
        dialog, data = self.setup_and_fill_ks_test_one_dataset(tgui, provide_parameters=True)

        results = self.get_ks_test_results_from_gui(dialog)

        assert results["name_1"] == "Dataset 1"
        assert results["name_2"] == "Normal CDF"
        assert results["hypothesis"] == "Two-Tailed"

        tgui.enter_number_into_spinbox(dialog.dia.one_dataset_sample_mean_spinbox, 0)
        tgui.enter_number_into_spinbox(dialog.dia.one_dataset_sample_stdev_spinbox, 1)
        tgui.left_mouse_click(dialog.dia.run_analysis_button)

        results = self.get_ks_test_results_from_gui(dialog)

        assert results["n"] == "1000"
        assert results["D"] == "0.0214"
        assert results["p"] == "0.74268437"

        del dialog

    @pytest.mark.parametrize("hypothesis", ["two-sided", "greater", "less"])  # TODO: all these tests will run twice with cumulative vs. normalised - is this necessary?
    def test_two_dataset_ks_test(self, tgui, hypothesis):
        """
        SPSS unequal:  D = 0.029 p = 0.794
             larger:   D = 0.029
             smaller:  D = -0.021
        MATLAB unequal: D = 0.029, p = 0.7888
               larger:  D = 0.029, p = 0. 0.4272
               smaller: D = 0.021, p = 0.6402
        """
        dialog, data = self.setup_and_fill_ks_test_two_dataset(tgui, hypothesis=hypothesis)
        D, p, name = self.get_hypothesis_results(hypothesis)

        results = self.get_ks_test_results_from_gui(dialog)

        assert results["name_1"] == "Dataset 1"
        assert results["name_2"] == "Dataset 2"
        assert results["hypothesis"] == name

        p, D = self.get_two_ks_test_result(hypothesis)

        assert results["n"] == "1000"
        assert results["D"] == p
        assert results["p"] == D

        del dialog

    def test_ks_test_not_enough_variance_1d(self, tgui):

        QtCore.QTimer.singleShot(1000, lambda: self.check_data_enough_variance(tgui))
        self.setup_and_fill_ks_test_one_dataset(tgui,
                                                user_data=np.atleast_2d([0, 0, 0, 0]).T)

# KS Test Utils ----------------------------------------------------------------------------------------------------------------------

    def load_ks_test_data(self, tgui):
        data_path = tgui.test_base_dir + "/ks_test_data.csv"
        data = pd.read_csv(data_path)
        return data.to_numpy()

    def get_ks_test_results_from_gui(self, dialog):
        """
        """
        results = {}
        results["name_1"] = dialog.dia.results_list_widget.item(1).text().split(": ")[1]
        results["name_2"] = dialog.dia.results_list_widget.item(2).text().split(": ")[1]
        results["hypothesis"] = dialog.dia.results_list_widget.item(3).text().split(": ")[1]

        results["n"] = dialog.dia.results_list_widget.item(5).text().split("= ")[1]
        results["D"] = dialog.dia.results_list_widget.item(6).text().split("= ")[1]
        results["p"] = dialog.dia.results_list_widget.item(7).text().split("= ")[1]

        return results

    def setup_ks_test_widgets_and_data(self, tgui):
        """
        """
        tgui.mw.mw.actionKolmogorov_Smirnoff.trigger()
        dialog = tgui.mw.dialogs["analysis_grouped_cum_prob"]
        table = dialog.dia.analysis_table

        data = self.load_ks_test_data(tgui)

        return dialog, data, table

    def setup_and_fill_ks_test_two_dataset(self, tgui, hypothesis="two-sided", user_data=None):
        """
        """
        dialog, data, table = self.setup_ks_test_widgets_and_data(tgui)

        if user_data is not None:
            data = user_data

        tgui.switch_groupbox(dialog.dia.two_datasets_groupbox, on=True)

        idx = ["two-sided", "greater", "less"].index(hypothesis)
        tgui.set_combobox(dialog.dia.alternative_hypothesis_combobox, idx=idx)

        tgui.fill_tablewidget_with_items(table, data)

        tgui.left_mouse_click(dialog.dia.run_analysis_button)

        return dialog, data

    def setup_and_fill_ks_test_one_dataset(self, tgui, provide_parameters=False, user_data=None):
        """
        """
        dialog, data, table = self.setup_ks_test_widgets_and_data(tgui)

        if user_data is not None:
            data = user_data  # must be 2d

        tgui.fill_tablewidget_with_items_1d(table, data[:, 0])

        if provide_parameters:
            tgui.set_combobox(dialog.dia.one_dataset_method_combobox, idx=1)
            tgui.enter_number_into_spinbox(dialog.dia.one_dataset_sample_mean_spinbox, np.mean(data[:, 0]))
            tgui.enter_number_into_spinbox(dialog.dia.one_dataset_sample_stdev_spinbox, np.std(data[:, 0], ddof=1))
        else:
            tgui.set_combobox(dialog.dia.one_dataset_method_combobox, idx=0)

        tgui.left_mouse_click(dialog.dia.run_analysis_button)

        return dialog, data

    def get_hypothesis_results(self, hypothesis):
        """
        """
        if hypothesis == "two-sided":
            D, p, name = [1, 1, "Two-Tailed"]
        elif hypothesis == "greater":
            D, p, name = [1, 1, "Dataset 1 Greater"]
        elif hypothesis == "less":
            D, p, name = [1, 1, "Dataset 1 Less"]

        return D, p, name

    def get_two_ks_test_result(self, hypothesis):

        results = {
            "two-sided": ["0.0290", "0.79466374"],
            "greater": ["0.0290", "0.43140956"],
            "less": ["0.0210", "0.64351371"],
        }

        p, D = results[hypothesis]
        return p, D

    def check_not_enough_samples_1d(self, tgui):
        assert "Ensure data has at least 4 observations" in tgui.mw.messagebox.text()
        tgui.mw.messagebox.close()

    def check_data_enough_variance(self, tgui):
        assert "Could not run KS test, check inputs are valid." in tgui.mw.messagebox.text()
        tgui.mw.messagebox.close()

    def check_not_enough_samples_2d(self, tgui):
        assert "Ensure data has at least 4 observations." in tgui.mw.messagebox.text()
        tgui.mw.messagebox.close()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Frequency Data Tests - Unit Tests
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("cum_prob_or_hist", ["cum_prob", "hist"])
    @pytest.mark.parametrize("one_or_two_dataset", ["one_dataset", "two_dataset_1", "two_dataset_2"])
    def test_ks_plot_copy_data(self, tgui, cum_prob_or_hist, one_or_two_dataset):
        """
        """
        if "one_dataset" in one_or_two_dataset:
            dialog, data = self.setup_and_fill_ks_test_one_dataset(tgui)
        else:
            dialog, data = self.setup_and_fill_ks_test_two_dataset(tgui)

        plot_dialog, plot_x, plot_y = self.open_plot_and_get_data(tgui, dialog, one_or_two_dataset, plot_type=cum_prob_or_hist)

        dataset = "2" if one_or_two_dataset == "two_dataset_2" else "1"
        plot_dialog.handle_copy_dataset(dataset)
        clipboard_data = pd.read_clipboard()

        assert utils.allclose(plot_x, clipboard_data.iloc[:, 0], 1e-10)
        assert utils.allclose(plot_y, clipboard_data.iloc[:, 1], 1e-10)

        del dialog

    def open_plot_and_get_data(self, tgui, dialog, one_or_two_dataset, plot_type="cum_prob"):

        tgui.left_mouse_click(dialog.dia.plot_data_button)
        plot_dialog = dialog.plot_dialog

        idx = 0 if plot_type == "cum_prob" else 1
        tgui.set_combobox(plot_dialog.dia.plot_type_combobox, idx)

        if one_or_two_dataset == "two_dataset_2":
            tgui.set_combobox(plot_dialog.dia.data_to_show_combobox, 1)

        if plot_type == "cum_prob":
            plot_x = plot_dialog.cum_prob_plot.xData
            plot_y = plot_dialog.cum_prob_plot.yData

        elif plot_type == "hist":
            plot_x = plot_dialog.histplot.getData()[0]
            plot_y = plot_dialog.histplot.getData()[1]

        return plot_dialog, plot_x, plot_y

    def run_test_ks_frequency_data(self, data, binning_method, plot_type, x_axis_display, custom_binnum=None, custom_binsize=None, divisor_num=None):

        settings = {"binning_method": binning_method,
                    "custom_binnum": custom_binnum,
                    "plot_type": plot_type,
                    "x_axis_display": x_axis_display,
                    "custom_binsize": {"grouped_ks_analysis": custom_binsize},
                    "divide_by_number": divisor_num}

        y_values, x_values, binsize, num_bins = core_analysis_methods.calc_cumulative_probability_or_histogram(data,
                                                                                                               settings,
                                                                                                               parameter="grouped_ks_analysis",
                                                                                                               legacy_bin_sizes=False)  # these functions already unit tested below, now just check the GUI is correct
        return y_values, x_values, binsize, num_bins

    def plot_label(self, plot_dialog):
        label = plot_dialog.dia.bin_size_label.text()
        bin_size = label.split("Bin Size: ")[1]
        return float(bin_size)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Frequency Data Tests - Unit Tests
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_process_frequency_data_for_cum_prob(self):
        """
        """
        event_times = np.array([1.1, 2.2, 3.3, 11.1, 12.2, 13.3, 21.1, 22.2, 23.3])

        for start_rec_idx in [5]:
            data, sort_idx = core_analysis_methods.process_frequency_data_for_cum_prob(event_times, start_rec_idx)

        test_isi = np.diff(event_times)

        assert np.array_equal(data, test_isi)
        start_rec = start_rec_idx + 1
        test_sort_idx = [str(rec) + "-" + str(rec + 1) for rec in np.arange(start_rec,
                                                                            start_rec + len(test_isi))]
        assert sort_idx == test_sort_idx

    def test_voltageclampdatamodel_process_frequency_data_for_cum_prob(self, tgui):
        """
        """
        peak_times = np.array([
            [1, 2, 3],
            [10, 12, 14],
            [20, 23, 26],
        ])

        event_info = self.make_peak_time_event_info(peak_times)

        tgui.mw.loaded_file.data.num_recs = 3
        processed_data, sort_idx = tgui.mw.loaded_file.process_frequency_data_for_cum_prob(event_info)

        assert np.array_equal(processed_data, np.array([1, 1, 2, 2, 3, 3]))
        assert sort_idx == ["1-2", "2-3", "4-5", "5-6", "7-8", "8-9"]

    def test_process_amplitude_for_frequency_table(self):
        """
        """
        amplitudes = [-1, 2.34, 1234, -5.432, -500, -600, 10000.1234, 10000.1235, -10000.1234]
        ordered_idx = np.array([1, 2, 4, 5, 6, 3, 7, 9, 8])

        sorted_amplitudes, sort_idx = core_analysis_methods.process_amplitude_for_frequency_table(amplitudes, sort_method=False)

        assert utils.allclose(sorted_amplitudes,
                              np.array([1, 2.34, 5.432, 500, 600, 1234, 10000.1234, 10000.1234, 10000.1235]),  # not tested at rec level UNTIL LATER
                              1e-10)
        assert np.array_equal(sort_idx, ordered_idx)

    def test_process_non_negative_param_for_frequency_table(self):
        """
        """
        parameters = np.array([0, 10, 5, 100, 23, 22, 21])
        sorted_paramters, sort_idx = core_analysis_methods.process_amplitude_for_frequency_table(parameters, sort_method=False)

        assert np.array_equal(sorted_paramters,
                              np.array([0, 5, 10, 21, 22, 23, 100]))
        assert np.array_equal(sort_idx,
                              np.array([1, 3, 2, 7, 6, 5, 4]))

    def test_get_num_bins_from_settings_custom_binnum(self):
        """
        """
        data = np.random.uniform(1, 1000, 100)

        settings = {"binning_method": "custom_binnum",
                    "custom_binnum": 10}

        assert 10 == core_analysis_methods.get_num_bins_from_settings(data, settings, parameter=None)

        settings["custom_binnum"] = 1000
        assert 100 == core_analysis_methods.get_num_bins_from_settings(data, settings, parameter=None)

        settings["custom_binnum"] = 0
        assert 100 == core_analysis_methods.get_num_bins_from_settings(data, settings, parameter=None)

    def test_get_num_bins_from_settings_custom_binsize(self):
        data = np.random.uniform(1, 1000, 1000)

        settings = {"binning_method": "custom_binsize",
                    "custom_binsize": {"test_parameter": 10}}

        assert 100 == core_analysis_methods.get_num_bins_from_settings(data, settings, parameter="test_parameter")

        settings["custom_binsize"]["test_parameter"] = 10000
        assert 1000 == core_analysis_methods.get_num_bins_from_settings(data, settings, parameter="test_parameter")

    def test_get_num_bins_from_settings_auto(self):
        data = np.random.uniform(1, 1000, 1000)

        settings = {"binning_method": "auto"}

        test_num_bins = len(np.histogram_bin_edges(data, bins="auto"))
        assert np.array_equal(test_num_bins,
                              core_analysis_methods.get_num_bins_from_settings(data, settings, parameter=None))

    def test_get_num_bins_from_settings(self):
        """
        Divide by zero is not permitted as widget is set to > 1
        """
        data = np.random.uniform(1, 1000, 1000)

        settings = {"binning_method": "num_events_divided_by",
                    "divide_by_number": 100}

        assert 10 == core_analysis_methods.get_num_bins_from_settings(data, settings, parameter="test_parameter")

        settings["divide_by_number"] = 10000
        assert 1000 == core_analysis_methods.get_num_bins_from_settings(data, settings, parameter="test_parameter")

    def test_calc_cumulative_probability_or_histogram(self):
        data = np.arange(1, 11)
        settings = {"binning_method": "custom_binnum",
                    "custom_binnum": 10,
                    "plot_type": "hist",
                    "x_axis_display": "left_edge"}

        y_values, x_values, binsize, num_bins = core_analysis_methods.calc_cumulative_probability_or_histogram(data,
                                                                                                               settings,
                                                                                                               parameter=None,
                                                                                                               legacy_bin_sizes=False)
        assert np.array_equal(np.ones(10),
                              y_values)
        assert np.array_equal(x_values,
                              np.histogram(data)[1][:-1])
        assert binsize == np.diff(np.histogram(data)[1])[0]
        assert num_bins == 10

        settings["plot_type"] = "cum_prob"
        y_values, x_values, binsize, num_bins = core_analysis_methods.calc_cumulative_probability_or_histogram(data,
                                                                                                               settings,
                                                                                                               parameter=None,
                                                                                                               legacy_bin_sizes=False)
        info = scipy.stats.cumfreq(data,
                                   numbins=settings["custom_binnum"],
                                   defaultreallimits=(np.min(data), np.max(data)))
        test_cum_prob = info.cumcount / len(data)

        assert utils.allclose(y_values,
                              test_cum_prob, 1e-10)
        assert np.array_equal(x_values,
                              np.histogram(data)[1][:-1])
        assert binsize == np.diff(np.histogram(data)[1])[0]
        assert num_bins == 10

    def test_format_bin_edges(self):
        bin_edges = np.array([0, 1, 2, 3, 4, 5, 6])

        left_bin_edges = core_analysis_methods.format_bin_edges(bin_edges, "left_edge")
        assert np.array_equal(left_bin_edges,
                              np.array([0, 1, 2, 3, 4, 5]))

        right_bin_edges = core_analysis_methods.format_bin_edges(bin_edges, "right_edge")
        assert np.array_equal(right_bin_edges,
                              np.array([1, 2, 3, 4, 5, 6]))

        center_bin_edges = core_analysis_methods.format_bin_edges(bin_edges, "bin_centre")
        np.array_equal(center_bin_edges,
                       np.array([np.mean([0, 1]), np.mean([1, 2]), np.mean([2, 3]), np.mean([3, 4]), np.mean([4, 5]), np.mean([5, 6])]))

# Frequency Data Test Utils ----------------------------------------------------------------------------------------------------------------------------------------

    def make_peak_time_event_info(self, peak_times, peak_idxs=False):
        event_info = [{} for rec in range(len(peak_times))]
        for rec in range(len(peak_times)):
            for idx in range(len(peak_times[rec])):

                peak_time = peak_times[rec][idx]

                if ~np.isnan(peak_time):
                    event_info[rec][str(peak_time)] = {"peak": {"time": peak_time}}
                    if np.any(peak_idxs):
                        event_info[rec][str(peak_time)]["peak"]["idx"] = peak_idxs[rec][idx]

        return event_info

    def get_test_frequency_data(self, tgui, analysis_type):
        if analysis_type == "frequency":
            peak_times = tgui.adata.peak_times["cumulative"]

            start_ev_num = 1
            all_params = []
            all_labels = []
            for rec in range(peak_times.shape[0]):
                rec_param = np.diff(peak_times[rec][~np.isnan(peak_times[rec])])
                peak_labels = [str(peak_idx) + "-" + str(peak_idx + 1) for peak_idx in np.arange(start_ev_num, start_ev_num + rec_param.size)]
                start_ev_num += rec_param.size + 1

                all_params.append(rec_param)
                all_labels.append(peak_labels)

            params = np.array(utils.flatten_list(all_params))
            labels = utils.flatten_list(all_labels)

        else:
            if analysis_type == "amplitude":
                params = np.abs(tgui.adata.b1 * tgui.adata.b1_offsets)

            elif analysis_type == "rise_time":
                params = ((tgui.adata.rise_samples * 0.79) * tgui.adata.ts) * 1000  # note rise times are all the same even if adjusting so this is not a great test   MAKE OWN FUNCTION
                params = np.repeat(params, tgui.adata.num_events())  # but will be correct if all others are, all other params are variable

            elif analysis_type == "decay_amplitude_percent":
                params = tgui.adata.get_all_decay_times()  # COMBEBINE WITH ABOVE?

            elif analysis_type == "event_time":
                peak_times = tgui.adata.peak_times["cumulative"]
                params = peak_times[~np.isnan(peak_times)]

            elif analysis_type == "decay_tau":
                test_tau_samples = tgui.adata.tau * tgui.adata.tau_offsets  # COMBEBINE WITH ABOVE?
                params = test_tau_samples * tgui.adata.ts * 1000

            if analysis_type == "biexp_rise":
                biexp_rise_samples = (tgui.adata.rise * np.squeeze(tgui.adata.rise_offsets)) # TODO: copied directly from test_events()
                params = biexp_rise_samples* tgui.adata.ts * 1000

            if analysis_type == "biexp_decay":
                biexp_decay_samples = tgui.adata.decay * np.squeeze(tgui.adata.decay_offsets)
                params = biexp_decay_samples * tgui.adata.ts * 1000

            labels = [ev_idx + 1 for ev_idx in range(tgui.adata.num_events())]

        return params, labels

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Frequency Data Tests - Gui and Plots
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def check_summary_statistics(self, tgui, start_col, test_data):

        tgui.switch_mw_tab(1)

        table = tgui.mw.mw.table_tab_tablewidget
        M_label = table.item(table.rowCount() - 3, start_col).text()
        SD_label = table.item(table.rowCount() - 2, start_col).text()
        SE_label = table.item(table.rowCount() - 1, start_col).text()

        assert M_label.strip() == "M"
        assert SD_label.strip() == "SD"
        assert SE_label.strip() == "SE"

        M = float(table.item(table.rowCount() - 3, start_col + 1).text())
        SD = float(table.item(table.rowCount() - 2, start_col + 1).text())
        SE = float(table.item(table.rowCount() - 1, start_col + 1).text())

        assert utils.allclose(M, np.mean(test_data), 1e-8)
        assert utils.allclose(SD, np.std(test_data, ddof=1), 1e-8)
        assert utils.allclose(SE, np.std(test_data, ddof=1) / np.sqrt(test_data.size), 1e-8)

    def run_tests_frequency_data_plot(self, tgui, bin_edges, cum_probs, analysis_type, cum_prob_or_hist):
        """
        """
        tgui.left_mouse_click(tgui.mw.mw.plot_cumulative_probability_button)

        analysis_idx = {"frequency": 0, "amplitude": 1, "rise_time": 2, "decay_amplitude_percent": 3, "event_time": 4, "decay_tau": 5, "biexp_rise": 6, "biexp_decay": 7}
        idx = analysis_idx[analysis_type]

        tgui.mw.loaded_file.cumulative_frequency_plot_dialog.dia.data_to_show_combobox.setCurrentIndex(idx)

        plot_x, plot_y = self.get_frequency_plot_data(tgui, cum_prob_or_hist, "x_and_y")

        assert np.array_equal(bin_edges, plot_x)
        assert np.array_equal(cum_probs, plot_y)

        tgui.mw.loaded_file.cumulative_frequency_plot_dialog.close()

    def run_tests_all_frequency_data(self, tgui, analysis_type, frequency_data, flattened_params, flattened_labels, custom_binnum, cum_prob_or_hist):
        """
        For rise time, decay perc, decay tau the test values are ~1e-10 different from the values in ee. This is fine normally
        as tested with utils.allclose but for sorting it messes everything up. As such in this case we need to get the true values from
        easy electrophysiollgy, although this is fine because we know they match to 1e-10 from the first test.
        """
        sort_idx = np.argsort(flattened_params)
        sorted_param = flattened_params[sort_idx]
        sorted_labels = [flattened_labels[i] for i in sort_idx]

        # Check data used to calculate frequencies is correct
        assert utils.allclose(frequency_data[1],
                              sorted_param,
                              1e-8), "parameter data " + analysis_type

        bin_edges = np.array([num for num in frequency_data[3] if num != ""])
        test_bin_edges = np.histogram(frequency_data[1], bins=custom_binnum)[1]

        test_bin_edges_center = (np.cumsum(test_bin_edges)[1:] - np.hstack([0, np.cumsum(test_bin_edges[:-2])])) / 2

        # Test Bin Edges
        assert utils.allclose(bin_edges,
                              test_bin_edges_center,
                              1e-10), \
            "bin edges " + analysis_type

        # Test Labels
        if analysis_type in ["rise_time", "decay_amplitude_percent", "event_time", "decay_tau", "biexp_rise", "biexp_decay"]:
            if analysis_type == "rise_time":
                true_param = tgui.mw.loaded_file.make_list_from_event_info_all_recs("rise", "rise_time_ms")
            elif analysis_type == "decay_amplitude_percent":
                true_param = tgui.mw.loaded_file.make_list_from_event_info_all_recs("decay_perc", "decay_time_ms")
            elif analysis_type == "event_time":
                true_param = tgui.mw.loaded_file.make_list_from_event_info_all_recs("peak", "time")
            elif analysis_type == "decay_tau":
                true_param = tgui.mw.loaded_file.make_list_from_event_info_all_recs("monoexp_fit", "tau_ms")
            elif analysis_type == "biexp_rise":
                true_param = tgui.mw.loaded_file.make_list_from_event_info_all_recs("biexp_fit", "rise_ms")
            elif analysis_type == "biexp_decay":
                true_param = tgui.mw.loaded_file.make_list_from_event_info_all_recs("biexp_fit", "decay_ms")

            sort_idx = np.argsort(true_param)
            sorted_labels = [flattened_labels[i] for i in sort_idx]

        assert sorted_labels == frequency_data[0], "sorted labels " + analysis_type

        # Test cumulative probabilities / frequencies
        # critical to use easy e frequency[1] output as e1-12 rounding errors will make this fail is using sorted_param (this is tested above so we can be sure htey are same > 1e-10
        third_col = np.array([num for num in frequency_data[2] if num != ""])
        if cum_prob_or_hist == "cum_prob":
            info = scipy.stats.cumfreq(frequency_data[1], numbins=custom_binnum, defaultreallimits=(np.min(frequency_data[1]), np.max(frequency_data[1])))
            test_cum_prob = info.cumcount / len(sorted_param)
            assert utils.allclose(third_col, test_cum_prob, 1e-10), "cum_prob " + analysis_type
        else:
            assert np.array_equal(third_col,
                                  np.histogram(frequency_data[1], bins=custom_binnum)[0])
        return third_col, bin_edges

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Sort  by event number
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("biexp", [True, False])
    @pytest.mark.parametrize("cum_prob_or_hist", ["cum_prob", "hist"])
    def test_sort_by_event_number(self, tgui, cum_prob_or_hist, biexp):
        """
        Run events and check cum prob for all events matches main table when both sorted
        and unsorted
        """
        if tgui.test_filetype == "artificial_events_one_record":
            return

        tgui.mw.cfgs.events["frequency_data_options"]["plot_type"] = cum_prob_or_hist

        num_recs = self.run_three_events_files_for_frequency_data_tests(tgui, biexp)

        if biexp:
            analysis_type_info = {
                "biexp_rise": [14, 31, 48],
                "biexp_decay": [15, 32, 49],
            }

        else:
            analysis_type_info = {
                "event_time": [3, 19, 35],  # same as frequency, not subtracted, easier to have as separate key
                "frequency": [3, 19, 35],  # hard coded
                "amplitude": [6, 22, 38],
                "rise_time": [7, 23, 39],
                "decay_amplitude_percent": [9, 25, 41],
                "decay_tau": [14, 30, 46],
            }

        files = ["file_0", "file_1", "file_2"]

        main_table_data = {}
        for analysis_type, table_col in analysis_type_info.items():

            # Save the main table
            main_table_data[analysis_type] = {}
            for file, col in zip(files, table_col):

                table_data = []
                for row in range(2, tgui.mw.mw.table_tab_tablewidget.rowCount()):
                    try:
                        data = tgui.mw.mw.table_tab_tablewidget.item(row, col).data(0)  # skip empty cells
                    except:
                        continue
                    table_data.append(float(data))

                main_table_data[analysis_type][file] = np.array(table_data)

        for analysis_type in analysis_type_info.keys():
            # for each analysis type, check that the cum prob matches the main table data, when both
            # sorted and unsorted.
            tgui.set_frequency_data_table(tgui, analysis_type)

            frequency_data_table = self.get_frequency_data_table(tgui, files, analysis_type)
            unsorted_cum_prob_columns = frequency_data_table

            self.check_main_table_vs_frequency_data(tgui, files, num_recs, analysis_type, main_table_data, frequency_data_table, sort_main_table=True)

            tgui.mw.update_table_sort_cum_prob("event_num")

            frequency_data_table = self.get_frequency_data_table(tgui, files, analysis_type)

            self.check_main_table_vs_frequency_data(tgui, files, num_recs, analysis_type, main_table_data, frequency_data_table, sort_main_table=False)

            for file in files:
                assert np.array_equal(unsorted_cum_prob_columns[file][:, 2:], frequency_data_table[file][:, 2:], equal_nan=True)  # check cum prob and bins remain unchanged
                assert np.array_equal(unsorted_cum_prob_columns[file][:, 3:], frequency_data_table[file][:, 3:], equal_nan=True)
            tgui.mw.update_table_sort_cum_prob("parameter")

    def get_frequency_data_table(self, tgui, files, analysis_type):  # TODO: this is similar to setup_test_suite.get_frequency_data_from_qtable()
        """
        """
        frequency_data_table = {}
        start_col = 0
        for file in files:
            file_table_data = utils.np_empty_nan((tgui.mw.mw.table_tab_tablewidget.rowCount() - 2, 4))  # TODO: table var
            for col_idx, col in enumerate(range(start_col, start_col + 4)):
                for row_idx, row in enumerate(range(2, tgui.mw.mw.table_tab_tablewidget.rowCount() - 4)):  # ignore summary statistcs

                    try:
                        data = tgui.mw.mw.table_tab_tablewidget.item(row, col).data(0)
                        if analysis_type == "frequency" and "-" in data:
                            data = data.split("-")[0]
                        else:
                            data = tgui.mw.mw.table_tab_tablewidget.item(row, col).data(0)

                            if data == "":
                                data = "NaN"
                    except:
                        data = "NaN"

                    file_table_data[row_idx, col_idx] = float(data)

            frequency_data_table[file] = file_table_data

            start_col += 4

        return frequency_data_table

    def check_main_table_vs_frequency_data(self, tgui, files, num_recs, analysis_type, main_table_data, frequency_data_table, sort_main_table):

        # check stuff
        for idx, file in enumerate(files):

            main_data = np.abs(main_table_data[analysis_type][file])  # must be abs
            frequency_data = frequency_data_table[file]

            if analysis_type == "frequency":

                main_table_iei = np.diff(main_data)
                event_nums = np.arange(1, len(main_table_iei) + 2)
                rec_overlap_idx = np.where(np.isin(main_table_iei, tgui.clean(frequency_data[:, 1])))[0]

                sort_iei_idx = np.argsort(main_table_iei[rec_overlap_idx]) if sort_main_table else np.arange(len(main_table_iei[rec_overlap_idx]))  # TODO: tidy all this up!
                test_iei = main_table_iei[rec_overlap_idx][sort_iei_idx]
                event_nums = event_nums[rec_overlap_idx][sort_iei_idx]

                discarded_recs = np.where(~np.isin(main_table_iei, tgui.clean(frequency_data[:, 1])))[0]

                assert len(tgui.clean(frequency_data[:, 0])) == len(main_data) - num_recs[idx]
                assert len(discarded_recs) == (num_recs[idx] - 1)
                assert np.array_equal(tgui.clean(frequency_data[:, 1]), test_iei)
                assert np.array_equal(tgui.clean(frequency_data[:, 0]), event_nums)

            else:
                assert len(tgui.clean(frequency_data[:, 0])) == len(main_data)

                num_recs = np.arange(1, len(main_data) + 1)
                sort_idx = np.argsort(main_data) if sort_main_table else np.arange(len(main_data))

                assert np.array_equal(tgui.clean(frequency_data[:, 1]), np.abs(main_data[sort_idx]))
                assert np.array_equal(tgui.clean(frequency_data[:, 0]), num_recs[sort_idx])

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Frequency Data Tests - plot batch file checks
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("cum_prob_or_hist", ["cum_prob", "hist"])
    def test_batch_mode_frequency_data_plot_filecheck(self, tgui, cum_prob_or_hist):  # TODO: can test bin size here too
        """
        """
        if tgui.test_filetype == "artificial_events_one_record":
            return

        tgui.mw.cfgs.events["frequency_data_options"]["plot_type"] = cum_prob_or_hist

        self.run_three_events_files_for_frequency_data_tests(tgui)

        for analysis_type in ["frequency", "amplitude", "rise_time", "decay_amplitude_percent", "event_time", "decay_tau"]:

            tgui.set_frequency_data_table(tgui, analysis_type)
            tgui.left_mouse_click(tgui.mw.mw.plot_cumulative_probability_button)

            analysis_idx = {"frequency": 0, "amplitude": 1, "rise_time": 2, "decay_amplitude_percent": 3, "event_time": 4, "decay_tau": 5}
            idx = analysis_idx[analysis_type]
            tgui.set_combobox(tgui.mw.loaded_file.cumulative_frequency_plot_dialog.dia.data_to_show_combobox, idx)

            if analysis_type != "frequency":  # dont set the combobox for the first loop to check the last file analyised is automatically shown
                tgui.set_combobox(dialog.dia.file_to_plot_combobox, 2)

            dialog = tgui.mw.loaded_file.cumulative_frequency_plot_dialog
            assert dialog.dia.file_to_plot_combobox.currentText() == "high_freq_events_1"
            assert np.array_equal(self.get_frequency_plot_data(tgui, cum_prob_or_hist, "y"), self.quick_get_table_column_data(tgui, 10)), "yData 2 " + analysis_type
            assert np.array_equal(self.get_frequency_plot_data(tgui, cum_prob_or_hist, "x"), self.quick_get_table_column_data(tgui, 11)), "xData 2 " + analysis_type
            QtWidgets.QApplication.processEvents()

            tgui.set_combobox(dialog.dia.file_to_plot_combobox, 1)
            assert dialog.dia.file_to_plot_combobox.currentText() == "vc_events_one_record"
            assert np.array_equal(self.get_frequency_plot_data(tgui, cum_prob_or_hist, "y"), self.quick_get_table_column_data(tgui, 6)), "yData 1 " + analysis_type
            assert np.array_equal(self.get_frequency_plot_data(tgui, cum_prob_or_hist, "x"), self.quick_get_table_column_data(tgui, 7)), "xData 1 " + analysis_type
            QtWidgets.QApplication.processEvents()

            tgui.set_combobox(dialog.dia.file_to_plot_combobox, 0)
            assert dialog.dia.file_to_plot_combobox.currentText() == "fake_data_name"
            assert np.array_equal(self.get_frequency_plot_data(tgui, cum_prob_or_hist, "y"), self.quick_get_table_column_data(tgui, 2)), "yData 0 " + analysis_type
            assert np.array_equal(self.get_frequency_plot_data(tgui, cum_prob_or_hist, "x"), self.quick_get_table_column_data(tgui, 3)), "xData 0 " + analysis_type
            QtWidgets.QApplication.processEvents()

        del dialog

    def run_three_events_files_for_frequency_data_tests(self, tgui, biexp=False):
        """
        """
        num_recs = []

        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=biexp)
        num_recs.append(tgui.mw.loaded_file.data.num_recs)

        tgui.load_a_filetype("voltage_clamp_1_record")
        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=biexp)
        num_recs.append(tgui.mw.loaded_file.data.num_recs)

        tgui.load_a_filetype("voltage_clamp_multi_record_events")
        tgui.run_artificial_events_analysis(tgui, "threshold", biexp=biexp)
        num_recs.append(tgui.mw.loaded_file.data.num_recs)

        tgui.switch_mw_tab(1)

        return num_recs

    def get_frequency_plot_data(self, tgui, cum_prob_or_hist, x_or_y):
        """
        """
        if cum_prob_or_hist == "cum_prob":
            plot_x = tgui.mw.loaded_file.cumulative_frequency_plot_dialog.cum_prob_plot.xData
            plot_y = tgui.mw.loaded_file.cumulative_frequency_plot_dialog.cum_prob_plot.yData
        elif cum_prob_or_hist == "hist":
            plot_x = tgui.mw.loaded_file.cumulative_frequency_plot_dialog.histplot.getData()[0]
            plot_y = tgui.mw.loaded_file.cumulative_frequency_plot_dialog.histplot.getData()[1]

        if x_or_y in ["x", "X"]:
            return plot_x
        elif x_or_y in ["y", "Y"]:
            return plot_y
        elif x_or_y == "x_and_y":
            return plot_x, plot_y

    def quick_get_table_column_data(self, tgui, col_idx):
        col_items = []
        for i in range(tgui.mw.mw.table_tab_tablewidget.rowCount()):
            item_ = tgui.mw.mw.table_tab_tablewidget.item(i, col_idx)
            try:
                col_items.append(float(item_.data(0)))
            except:
                pass
        return np.array(col_items)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Tests that can crash due to undiagnossed memory access tests
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @pytest.mark.may_crash
    @pytest.mark.parametrize("cum_prob_or_hist", ["cum_prob", "hist"])
    @pytest.mark.parametrize("bin_edge_setting", [["bin_centre", 0], ["left_edge", 1], ["right_edge", 2]])
    @pytest.mark.parametrize("one_or_two_dataset", ["one_dataset", "two_dataset_1", "two_dataset_2"])
    def test_ks_plot_cum_prob_and_hist(self, tgui, cum_prob_or_hist, bin_edge_setting, one_or_two_dataset):

        # setup plot details
        if "one_dataset" in one_or_two_dataset:
            dialog, data = self.setup_and_fill_ks_test_one_dataset(tgui)
            data = data[:, 0]
        else:
            dialog, data = self.setup_and_fill_ks_test_two_dataset(tgui)
            col = 0 if one_or_two_dataset == "two_dataset_1" else 1
            data = data[:, col]

        # options
        tgui.mw.mw.actionEvents_Analyis_Options.trigger()
        tgui.left_mouse_click(tgui.mw.dialogs["events_analysis_options"].dia.frequency_data_more_options_button)
        options_dialog = tgui.mw.dialogs["events_analysis_options"].events_frequency_data_options_dialog

        bin_edge, bin_edge_idx = bin_edge_setting
        tgui.set_combobox(options_dialog.dia.x_axis_display_combobox, bin_edge_idx)

        # Auto ----------------------------------------------------------------------------------------------------------------------------------------------------------

        plot_dialog, plot_x, plot_y = self.open_plot_and_get_data(tgui, dialog, one_or_two_dataset, plot_type=cum_prob_or_hist)
        y_values, x_values, bin_size, __ = self.run_test_ks_frequency_data(data, "auto", cum_prob_or_hist, bin_edge)  # these functions already unit tested below, now just check the GUI is correct

        test_num_bins = len(np.histogram_bin_edges(data, bins="auto"))
        assert test_num_bins == len(plot_x)
        assert np.isclose(self.plot_label(plot_dialog), bin_size, 1e-2)
        assert np.array_equal(y_values, plot_y)
        assert np.array_equal(x_values, plot_x)

        # Custom Bin Number   ------------------------------------------------------------------------------------------------------------------------------------------

        plot_dialog.close()
        tgui.set_combobox(options_dialog.dia.binning_method_combobox, 1)
        tgui.enter_number_into_spinbox(options_dialog.dia.custom_bin_number_spinbox, 50)

        plot_dialog, plot_x, plot_y = self.open_plot_and_get_data(tgui, dialog, one_or_two_dataset, plot_type=cum_prob_or_hist)
        y_values, x_values, bin_size, __ = self.run_test_ks_frequency_data(data, "custom_binnum", cum_prob_or_hist, bin_edge, custom_binnum=50)

        assert np.isclose(self.plot_label(plot_dialog), bin_size, 1e-2)
        assert len(plot_x) == 50
        assert np.array_equal(y_values, plot_y)
        assert np.array_equal(x_values, plot_x)

        # Custom Bin Size  ---------------------------------------------------------------------------------------------------------------------------------------------

        plot_dialog.close()
        test_interval = 0.1
        tgui.set_combobox(options_dialog.dia.binning_method_combobox, 2)
        tgui.set_combobox(options_dialog.dia.custom_binsizes_combobox, 8)
        tgui.enter_number_into_spinbox(options_dialog.dia.custom_binsizes_spinbox, test_interval)

        plot_dialog, plot_x, plot_y = self.open_plot_and_get_data(tgui, dialog, one_or_two_dataset, plot_type=cum_prob_or_hist)
        y_values, x_values, __, num_bins = self.run_test_ks_frequency_data(data, "custom_binsize", cum_prob_or_hist, bin_edge, custom_binsize=0.1)

        x_inter_bin_interval = scipy.stats.mode(np.diff(plot_x))[0]
        assert np.isclose(x_inter_bin_interval, test_interval, 1e-2)
        assert np.isclose(self.plot_label(plot_dialog), test_interval, 1e-2)
        assert np.array_equal(y_values, plot_y)
        assert np.array_equal(x_values, plot_x)

        # Num Events Divided by ----------------------------------------------------------------------------------------------------------------------------------------

        plot_dialog.close()
        test_divisor = 4
        tgui.set_combobox(options_dialog.dia.binning_method_combobox, 3)
        tgui.enter_number_into_spinbox(options_dialog.dia.custom_bin_number_spinbox, test_divisor)

        plot_dialog, plot_x, plot_y = self.open_plot_and_get_data(tgui, dialog, one_or_two_dataset, plot_type=cum_prob_or_hist)
        y_values, x_values, bin_size, num_bins = self.run_test_ks_frequency_data(data, "num_events_divided_by", cum_prob_or_hist, bin_edge, divisor_num=test_divisor)

        assert np.isclose(self.plot_label(plot_dialog), bin_size, 1e-2)
        assert np.array_equal(y_values, plot_y)
        assert np.array_equal(x_values, plot_x)

        plot_dialog.close()
        del dialog
        del plot_dialog
        del options_dialog
        tgui.shutdown()
        
        # Plots -
        # is there a way to intergrate below to check all binning methods?
        # do plot options already exist? if not, these can be checked manually. Are the configs shared!??! (yes) (so just need to check 'KS test' options - is this done already?
    @pytest.mark.may_crash
    @pytest.mark.parametrize("analysis_type", ["frequency", "amplitude", "rise_time", "decay_amplitude_percent", "event_time", "decay_tau", "biexp_rise", "biexp_decay"])
    @pytest.mark.parametrize("cum_prob_or_hist", ["cum_prob", "hist"])
    def test_cumulative_frequency_for_all_params(self, tgui, analysis_type, cum_prob_or_hist):
        """
        """
        for filenum in range(1):

            if "biexp" in analysis_type:
                tgui.setup_artificial_data(norm_or_cumu_time="cumulative", analysis_type="events_multi_record_biexp_7500")
            tgui.update_events_to_varying_amplitude_and_tau()

            custom_binnum = int(tgui.adata.num_events() / 2)
            tgui.mw.cfgs.events["frequency_data_options"]["plot_type"] = cum_prob_or_hist
            tgui.mw.cfgs.events["frequency_data_options"]["binning_method"] = "custom_binnum"
            tgui.mw.cfgs.events["frequency_data_options"]["custom_binnum"] = custom_binnum

            if "biexp" in analysis_type:
                tgui.set_widgets_for_artificial_event(tgui, run=True)
            else:
                tgui.run_artificial_events_analysis(tgui, "threshold")

            tgui.switch_mw_tab(1)
            tgui.set_frequency_data_table(tgui, analysis_type)

            num_rows = tgui.adata.num_events() - tgui.adata.num_recs if analysis_type == "frequency" else tgui.adata.num_events()

            frequency_data = tgui.get_frequency_data_from_qtable(analysis_type, 0, num_rows)  # remove 3 from the bottom - the summary statistics

            flattened_params, flattened_labels = self.get_test_frequency_data(tgui, analysis_type)

            self.check_summary_statistics(tgui, start_col=filenum * 4, test_data=flattened_params)

            cum_probs, bin_edges = self.run_tests_all_frequency_data(tgui, analysis_type, frequency_data,
                                                                     flattened_params, flattened_labels,
                                                                     custom_binnum, cum_prob_or_hist)

            self.run_tests_frequency_data_plot(tgui, bin_edges, cum_probs, analysis_type, cum_prob_or_hist)

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data("cumulative", analysis_type=tgui.analysis_type)
        tgui.shutdown()

    @pytest.mark.may_crash
    def test_ks_test_not_enough_samples_1d(self, tgui):

        QtCore.QTimer.singleShot(1000, lambda: self.check_not_enough_samples_1d(tgui))
        self.setup_and_fill_ks_test_one_dataset(tgui,
                                                user_data=np.atleast_2d([0, 1, 2]).T)

    @pytest.mark.may_crash
    @pytest.mark.parametrize("uneven_data", [[[0, 0, 0, 0], [0, 0, 0]],
                                             [[0, 0, 0], [0, 0, 0, 0]],
                                             [[0, 0, 0], [0, 0, 0]]])
    def test_ks_test_not_enough_samples_2d(self, tgui, uneven_data):
        """
        """
        dialog, data, table = self.setup_ks_test_widgets_and_data(tgui)

        tgui.switch_groupbox(dialog.dia.two_datasets_groupbox, on=True)

        for row, input in enumerate(uneven_data[0]):  # RENAME
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(input)))

        for row, input in enumerate(uneven_data[1]):
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(input)))

        QtCore.QTimer.singleShot(2000, lambda: self.check_not_enough_samples_2d(tgui))
        tgui.left_mouse_click(dialog.dia.run_analysis_button)

        del dialog