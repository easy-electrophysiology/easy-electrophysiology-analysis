from PySide6 import QtWidgets, QtCore, QtGui
from PySide6 import QtTest
from PySide6.QtTest import QTest
import pytest
import sys
import os
import numpy as np
import copy

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology.mainwindow.mainwindow import (
    MainWindow,
)
from ephys_data_methods import current_calc, core_analysis_methods
from utils import utils
import utils_for_testing as test_utils
from setup_test_suite import GuiTestSetup
from slow_vs_fast_settings import get_settings
from PySide6.QtCore import Signal

keyClicks = QTest.keyClicks
os.environ["PYTEST_QT_API"] = "PySide6"

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------

SPEED = "fast"


class TestSpikeCountGui:
    """
    Test all AP Counting analysis through the GUI
    """

    @pytest.fixture(
        scope="function",
        params=["normalised", "cumulative"],
        ids=["normalised", "cumulative"],
    )
    def tgui(test, request):
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = SPEED
        tgui.setup_artificial_data(request.param)
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_spiketimes_from_qtable(tgui, rec_from, rec_to, filenum):
        tgui.switch_mw_tab(1)
        tgui.switch_checkbox(tgui.mw.mw.show_all_spiketimes_checkbox, on=True)
        table_data = utils.np_empty_nan((tgui.adata.num_recs, tgui.adata.max_num_spikes))

        start_col = tgui.mw.mw.table_tab_tablewidget.findItems(" Spike 1:    ", QtGui.Qt.MatchExactly)[
            -1
        ].column()  # must get the last entry dynamically as each file can have different number of spikes

        for tab_idx, rec in enumerate(range(rec_from, rec_to)):
            row_table_data = []
            for ispike in range(start_col, start_col + tgui.adata.max_num_spikes):
                try:  # ignore None and "" entries in table.
                    table_cell_data = float(tgui.mw.mw.table_tab_tablewidget.item(tab_idx + 2, ispike).data(0))
                except:
                    table_cell_data = np.nan
                row_table_data.append(table_cell_data)
            table_data[rec] = row_table_data

        return table_data

    @staticmethod
    def overwrite_modeldata_to_test_rheobase(tgui, test_rheobase_rec):
        """
        Blank out Vm for first records to artificially create a rheobase (so first spike is not always rec 1)
        """
        tgui.mw.loaded_file.data.vm_array[0:test_rheobase_rec] = np.zeros(
            (test_rheobase_rec - 0, tgui.adata.num_samples)
        )
        tgui.mw.loaded_file.init_analysis_results_tables()

    @staticmethod
    def get_first_spikes_within_bounds(tgui, bounds_vm):
        """ """
        num_recs = tgui.mw.loaded_file.data.num_recs

        start_times = np.array(bounds_vm["exp"][0])
        stop_times = np.array(bounds_vm["exp"][1])
        start_times = np.atleast_2d(start_times).T

        first_spikes_in_bounds = utils.np_empty_nan(num_recs)
        for rec in range(num_recs):
            try:
                first_spike_idx = np.where(tgui.adata.peak_times[tgui.time_type][rec] > start_times[rec])[0][0]
                spike_time = tgui.adata.peak_times[tgui.time_type][rec][first_spike_idx]
                if spike_time < stop_times[rec]:  # ensure spike is within both bounds
                    first_spikes_in_bounds[rec] = spike_time
                else:
                    first_spikes_in_bounds[rec] = 0
            except IndexError:
                first_spikes_in_bounds[rec] = 0

        return first_spikes_in_bounds

    # Rheobase Helpers
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def overwrite_artificial_im_array_with_ramp_protocol(tgui):
        """
        Replace im array with a sloping injection
        """
        num_recs = tgui.mw.loaded_file.data.num_recs
        im_injection = np.linspace(0, 1, tgui.adata.num_samples)
        im_array = np.tile(im_injection, [num_recs, 1]) * np.random.randint(1, 100, num_recs)[:, None]
        tgui.mw.loaded_file.data.im_array = im_array
        return im_array

    @staticmethod
    def fill_im_protocol_with_default_ramp_step(tgui, accept=False):
        """ """
        step = 10
        tgui.switch_groupbox(
            tgui.mw.dialogs["user_im_entry"].dia.step_ramp_start_stop_im_groupbox,
            on=True,
        )
        tgui.mw.dialogs["user_im_entry"].dia.step_ramp_start_stop_im_spinbox.clear()
        keyClicks(
            tgui.mw.dialogs["user_im_entry"].dia.step_ramp_start_stop_im_spinbox,
            str(step),
        )
        tgui.left_mouse_click(tgui.mw.dialogs["user_im_entry"].dia.fill_protocol_button)

        if accept:
            QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
            tgui.mw.dialogs["user_im_entry"].handle_ramp_accept()

        return step

    @staticmethod
    def get_rheobase_accounting_for_different_bounds_per_rec(tgui, rec, bounds_im):
        """ """
        results = {"exp": None, "bl": None}

        for key in ["exp", "bl"]:
            start_time = bounds_im[key][0][rec] - tgui.mw.loaded_file.data.min_max_time[rec][0]
            stop_time = bounds_im[key][1][rec] - tgui.mw.loaded_file.data.min_max_time[rec][0]  # FIX NAMING WRONG !?

            start_idx = current_calc.convert_time_to_samples(
                timepoint=start_time,
                start_or_stop="start",
                time_array=tgui.adata.time_array,
                min_max_time=tgui.adata.min_max_time,
                base_rec=rec,
                add_offset_back=True,
            )

            stop_idx = current_calc.convert_time_to_samples(
                timepoint=stop_time,
                start_or_stop="stop",
                time_array=tgui.adata.time_array,
                min_max_time=tgui.adata.min_max_time,
                base_rec=rec,
                add_offset_back=True,
            )

            results[key] = np.mean(tgui.adata.im_array[rec][start_idx : stop_idx + 1])

        im_injection_between_bounds_at_rec = results["exp"] - results["bl"]

        return im_injection_between_bounds_at_rec

    @staticmethod
    def find_true_rheobase_and_exact_im(
        tgui,
        im_ramp_array,
        test_rheobase_rec,
        bounds_im=False,
        provide_baselines_instead_or_measure_from_data=False,
    ):
        """
        find true rheobase and exact im and test
        """
        rheobase_spike_idx = int(
            tgui.adata.spike_sample_idx[test_rheobase_rec][0] + tgui.adata.samples_from_start_to_peak
        )

        if provide_baselines_instead_or_measure_from_data is False:
            bl_upper = current_calc.convert_time_to_samples(
                timepoint=bounds_im["bl"][0][test_rheobase_rec],
                start_or_stop="start",
                time_array=tgui.adata.time_array,
                min_max_time=tgui.adata.min_max_time,
                base_rec=test_rheobase_rec,
                add_offset_back=False,
            )

            bl_lower = current_calc.convert_time_to_samples(
                timepoint=bounds_im["bl"][1][test_rheobase_rec],
                start_or_stop="stop",
                time_array=tgui.adata.time_array,
                min_max_time=tgui.adata.min_max_time,
                base_rec=test_rheobase_rec,
                add_offset_back=False,
            )

            baseline = np.mean(im_ramp_array[test_rheobase_rec][bl_upper : bl_lower + 1])

        else:
            baseline = provide_baselines_instead_or_measure_from_data[test_rheobase_rec]

        test_rheobase = im_ramp_array[test_rheobase_rec][rheobase_spike_idx] - baseline

        return test_rheobase

    def check_rheobase(self, tgui, num_recs, rec_from, rec_to, test_rheobase_rec, filenum=0):
        baselines = np.zeros((num_recs, 1))
        saved_data = tgui.mw.loaded_file.saved_rheobase_settings["im_array"]
        test_rheobase = self.find_true_rheobase_and_exact_im(
            tgui,
            saved_data,
            test_rheobase_rec,
            provide_baselines_instead_or_measure_from_data=baselines,
        )
        assert (
            tgui.mw.loaded_file.spkcnt_data.loc[0, "rheobase"] == test_rheobase
        ), "model data rheobase is incorrect after ramp user Im protocol"
        assert (
            tgui.mw.stored_tabledata.spkcnt_data[filenum].loc[0, "rheobase"] == test_rheobase
        ), "stored table data rheobase is incorrect after ramp user Im protocol"
        assert tgui.eq(
            tgui.get_data_from_qtable("rheobase", rec_from, rec_to)[0], test_rheobase[0]
        ), "table rheobase is incorrect after ramp user Im protocol"

    # Other Parameter Test Utils
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def check_fs_latency(tgui, rec_from, rec_to, filenum=0, first_spikes=None):
        if not np.any(first_spikes):
            first_spikes = tgui.adata.peak_times[tgui.time_type][:, 0]

        test_fs_latency_unprocessed = first_spikes - tgui.adata.min_max_time[:, 0]
        test_fs_latency_unprocessed[np.where(test_fs_latency_unprocessed < 0)] = 0  # handle recs with zero spikes
        test_fs_latency = tgui.process_test_data_for_analyse_recs(test_fs_latency_unprocessed, rec_from, rec_to)
        test_fs_latency_ms = test_fs_latency * 1000
        test_fs_latency_ms[rec_from:rec_to][np.isnan(test_fs_latency_ms[rec_from:rec_to])] = 0

        assert utils.allclose(tgui.mw.loaded_file.spkcnt_data["fs_latency_ms"], test_fs_latency_ms, 1e-10)
        assert utils.allclose(
            tgui.mw.stored_tabledata.spkcnt_data[filenum]["fs_latency_ms"],
            test_fs_latency_ms,
            1e-10,
        )
        assert utils.allclose(
            tgui.get_data_from_qtable("fs_latency_ms", rec_from, rec_to),
            tgui.clean(test_fs_latency_ms),
            1e-10,
        )

    @staticmethod
    def remove_spike_from_adata(tgui, rec, spike_idx, adata_array):
        """ """
        adata_array[rec][spike_idx] = np.nan

        cleaned_peak_data = tgui.clean(adata_array[rec])
        new_peak_times = utils.np_empty_nan(tgui.adata.max_num_spikes)

        new_peak_times[0 : len(cleaned_peak_data)] = cleaned_peak_data
        adata_array[rec, :] = new_peak_times

    def calculate_fa(self, tgui, spike_times, rec_from, rec_to, method):
        """ """
        test_fa_divisor = []
        for rec in range(rec_from, rec_to + 1):
            clean_row = tgui.clean(spike_times[rec])

            if len(clean_row) <= 2:
                test_fa_divisor.append(0)
                continue

            if method == "spike_fa_divisor":
                rec_fa = self.calculate_test_divisor_method(clean_row)

            elif method == "spike_fa_local_variance":
                rec_fa = self.calculate_test_local_variance(clean_row)

            test_fa_divisor.append(rec_fa)

        test_fa_divisor = np.array(test_fa_divisor)

        return test_fa_divisor

    @staticmethod
    def calculate_test_divisor_method(data):
        """ """
        if len(data) > 1:
            divisor_method = (data[1] - data[0]) / (data[-1] - data[-2])
        else:
            divisor_method = 0

        return divisor_method

    @staticmethod
    def calculate_test_local_variance(data):
        """
        Re-calculate in a loop here (vectorised in EE) as is clearer.
        """
        isi = np.diff(data)
        n = len(isi)

        sum_ = 0
        for i in range(n - 1):
            Ti = isi[i]
            Ti_p1 = isi[i + 1]
            sum_ += (3 * (Ti - Ti_p1) ** 2) / (Ti + Ti_p1) ** 2

        local_variance = sum_ / (n - 1)

        return local_variance

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Tests
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("spike_detection_method", ["auto_record", "auto_spike", "manual"])
    def test_spikecount_model_data_all_spike_detection_methods(
        self, tgui, analyse_specific_recs, spike_detection_method
    ):
        """
        For these tests, only the default spike detection method is tested Auto. Threshold Record.
        """
        test_spike_counts, rec_from, rec_to = tgui.handle_analyse_specific_recs(
            analyse_specific_recs, tgui.adata.spikes_per_rec
        )
        tgui.run_spikecount_analysis(spike_detection_method=spike_detection_method)

        assert tgui.eq(tgui.mw.loaded_file.spkcnt_data["num_spikes"], test_spike_counts), (
            "Check Model Spike Number: " + spike_detection_method
        )

        assert tgui.eq(tgui.mw.stored_tabledata.spkcnt_data[0]["num_spikes"], test_spike_counts), (
            "Check Stored Tabledata Spike Number: " + spike_detection_method
        )
        assert tgui.eq(
            tgui.get_data_from_qtable("num_spikes", rec_from, rec_to),
            tgui.clean(test_spike_counts),
        ), (
            "Check QTable Spike Number " + spike_detection_method
        )

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_current_calc_spike_info(self, tgui, analyse_specific_recs):
        """
        Check spike times from spikecounts.
        Check against: 1) spike times found in the model spkcnt_spike_info
                       2) spike times found in the stored tabledata
                       3) spike times found in the qtable
        """
        for filenum in range(3):
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)
            tgui.run_spikecount_analysis()

            test_spike_times = tgui.set_out_of_rec_to_nan(
                rec_from,
                rec_to,
                array_size=(tgui.adata.num_recs, tgui.adata.max_num_spikes),
                data=tgui.adata.peak_times_,
            )

            spike_times = test_utils.get_spike_times_from_spike_info(tgui.adata, tgui.mw.loaded_file.spkcnt_spike_info)
            tgui.check_spiketimes(tgui, spike_times, test_spike_times, "model spiketimes")

            tabledata_spike_times = test_utils.get_spike_times_from_spike_info(
                tgui.adata, tgui.mw.stored_tabledata.spkcnt_spike_info[filenum]
            )
            tgui.check_spiketimes(tgui, tabledata_spike_times, test_spike_times, "stored data spiketimes")

            table_data = self.get_spiketimes_from_qtable(tgui, rec_from, rec_to, filenum)
            tgui.check_spiketimes(tgui, table_data, test_spike_times, "stored data qtable")

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data(tgui.time_type, "spkcnt")
        tgui.shutdown()

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_current_calc_spike_info_idx(self, tgui, analyse_specific_recs):
        """
        The spike_info data store also contains the index of all spikes, check these too on the model and stored tabledata.
        """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        tgui.run_spikecount_analysis()
        spike_idx = test_utils.get_spike_times_from_spike_info(
            tgui.adata, tgui.mw.loaded_file.spkcnt_spike_info, param_type="idx"
        )
        assert tgui.eq(spike_idx[rec_from:rec_to], tgui.adata.spike_peak_idx[rec_from:rec_to])

        tabledata_spike_idx = test_utils.get_spike_times_from_spike_info(
            tgui.adata, tgui.mw.stored_tabledata.spkcnt_spike_info[0], param_type="idx"
        )
        assert tgui.eq(
            tabledata_spike_idx[rec_from:rec_to],
            tgui.adata.spike_peak_idx[rec_from:rec_to],
        )

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_spkcnt_counted_recs(self, tgui, analyse_specific_recs):
        test_counted_recs = np.array([rec for rec in range(1, tgui.adata.num_recs + 1)])
        test_counted_recs, rec_from, rec_to = tgui.handle_analyse_specific_recs(
            analyse_specific_recs, test_counted_recs
        )
        tgui.run_spikecount_analysis()

        assert tgui.eq(
            tgui.mw.loaded_file.spkcnt_data["record_num"], test_counted_recs
        ), "Check Model Spike Counted Recs"
        assert tgui.eq(
            tgui.mw.stored_tabledata.spkcnt_data[0]["record_num"], test_counted_recs
        ), "Check Stored Tableddata Counted Recs"
        assert tgui.eq(
            tgui.get_data_from_qtable("record_num", rec_from, rec_to),
            tgui.clean(test_counted_recs),
        ), "Check QTable Spike Number"

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("align_bounds_across_recs", [True, False])
    def test_current_calc_model_data_with_bounds(self, tgui, analyse_specific_recs, align_bounds_across_recs):
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        bounds_vm = tgui.get_analyse_across_recs_or_not_boundaries_dict_for_spikes(align_bounds_across_recs)

        tgui.run_spikecount_analysis(["bounds"], bounds_vm=bounds_vm),  # need to set bounds before they are moved

        (
            spike_info_test_model,
            spike_count_test_model,
        ) = tgui.adata.subtract_results_from_data(
            tgui.adata,
            tgui.mw.loaded_file.spkcnt_spike_info,
            tgui.mw.loaded_file.spkcnt_data["num_spikes"],
            rec_from,
            rec_to,
            tgui.time_type,
            bounds=[
                bounds_vm["exp"][0]
                - tgui.mw.loaded_file.data.min_max_time[
                    :, 0
                ],  # format required for function to align with test_spikecalc
                bounds_vm["exp"][1] - tgui.mw.loaded_file.data.min_max_time[:, 0],
            ],
        )
        assert spike_info_test_model
        assert spike_count_test_model

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("align_bounds_across_recs", [True, False])
    def test_current_calc_tabledata_and_table_with_bounds(self, tgui, analyse_specific_recs, align_bounds_across_recs):
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        bounds_vm = tgui.get_analyse_across_recs_or_not_boundaries_dict_for_spikes(align_bounds_across_recs)

        tgui.run_spikecount_analysis(["bounds"], bounds_vm=bounds_vm)

        (
            spike_info_test_tabledata,
            spike_time_test_tabledata,
        ) = tgui.adata.subtract_results_from_data(
            tgui.adata,
            tgui.mw.stored_tabledata.spkcnt_spike_info[0],
            tgui.mw.stored_tabledata.spkcnt_data[0]["num_spikes"],
            rec_from,
            rec_to,
            tgui.time_type,
            [
                bounds_vm["exp"][0] - tgui.mw.loaded_file.data.min_max_time[:, 0],
                bounds_vm["exp"][1] - tgui.mw.loaded_file.data.min_max_time[:, 0],
            ],
        )

        assert spike_info_test_tabledata
        assert spike_time_test_tabledata
        assert tgui.eq(
            tgui.get_data_from_qtable("num_spikes", rec_from, rec_to),
            tgui.clean(tgui.mw.loaded_file.spkcnt_data["num_spikes"]),
        ), "Check QTable Spike Number"

    def test_current_calc_plotgraphs_time(self, tgui):
        tgui.run_spikecount_analysis()
        for rec in range(0, tgui.adata.num_recs):
            test_rec_peak_times = tgui.adata.peak_times[tgui.time_type][rec]
            assert tgui.eq(
                tgui.mw.loaded_file_plot.spkcnt_plot.xData,
                tgui.clean(test_rec_peak_times),
            )
            tgui.left_mouse_click(tgui.mw.mw.current_rec_rightbutton)

    def test_current_calc_plotgraphs_amplitudes(self, tgui):
        tgui.run_spikecount_analysis()
        test_ampltudes = test_utils.get_spike_times_from_spike_info(
            tgui.adata, tgui.mw.loaded_file.spkcnt_spike_info, param_type="amplitude"
        )
        for rec in range(0, tgui.adata.num_recs):
            rec_amplitudes = test_ampltudes[rec]
            assert tgui.eq(tgui.mw.loaded_file_plot.spkcnt_plot.yData, tgui.clean(rec_amplitudes))
            tgui.left_mouse_click(tgui.mw.mw.current_rec_rightbutton)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Test Spikecount Params
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("align_bounds_across_recs", ["no_bounds", True, False])
    def test_first_spike_latency(self, tgui, analyse_specific_recs, align_bounds_across_recs):
        """
        Need to reset the recs to analyse per file as the underlying file is only 12 recs which changes
        the recs_to from 50 to 12.
        """
        for filenum in range(3):
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

            if align_bounds_across_recs == "no_bounds":
                tgui.run_spikecount_analysis(["fs_latency_ms"])
                first_spikes = tgui.adata.peak_times[tgui.time_type][:, 0]
            else:
                bounds_vm = tgui.get_analyse_across_recs_or_not_boundaries_dict_for_spikes(align_bounds_across_recs)
                tgui.run_spikecount_analysis(["fs_latency_ms", "bounds"], bounds_vm=bounds_vm)
                first_spikes = self.get_first_spikes_within_bounds(tgui, bounds_vm)

            self.check_fs_latency(tgui, rec_from, rec_to, filenum, first_spikes)

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data(tgui.time_type, "spkcnt")
        tgui.shutdown()

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_mean_isi(self, tgui, analyse_specific_recs):
        for filenum in range(3):
            _, rec_from, rec_to = tgui.handle_analyse_specific_recs(
                analyse_specific_recs, tgui.adata.peak_times["normalised"][:, 0]
            )

            tgui.run_spikecount_analysis(["mean_isi_ms"])
            test_mean_isi_ms = tgui.calculate_mean_isi()
            test_mean_isi_ms = tgui.process_test_data_for_analyse_recs(test_mean_isi_ms, rec_from, rec_to)

            assert utils.allclose(tgui.mw.loaded_file.spkcnt_data["mean_isi_ms"], test_mean_isi_ms, 1e-10), "model"

            assert utils.allclose(
                tgui.mw.stored_tabledata.spkcnt_data[filenum]["mean_isi_ms"],
                test_mean_isi_ms,
                1e-10,
            ), "tabledata"
            assert utils.allclose(
                tgui.get_data_from_qtable("mean_isi_ms", rec_from, rec_to),
                tgui.clean(test_mean_isi_ms),
                1e-10,
            ), "qtable"

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data(tgui.time_type, "spkcnt")
        tgui.shutdown()

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("mode", ["dont_align_across_recs", "dont_align_across_recs"])
    def test_spikecount_im_bounds_analysis_im_bounds(self, tgui, analyse_specific_recs, mode):
        """
        Check the Im bounds is set correctly when moving bounds across every single record, and try
        loading a file in batch mode in between to check it doesn't mess anything up.

        See tset_input_resistance.test_ir_bounds_not_linked_across_recs() for more details.
        """
        tgui.set_link_across_recs(tgui, mode)

        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False, im_setting="bounds", im_groupbox_on=True)
        for filenum in range(2):
            # Get and set every region randomly across all recs. Get the bounds first otherwise
            # the boundary region will not exist but will try to be moved and create problems in the test environment
            tgui.set_recs_to_analyse_spinboxes_checked(
                on=False
            )  # we need to turn this off for assign_random_boundary_position_for_every_rec_and_test() to work
            all_start_stop_times = tgui.assign_random_boundary_position_for_every_rec_and_test(
                tgui, tgui.mw.spkcnt_bounds, mode
            )
            _, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)
            tgui.left_mouse_click(tgui.mw.mw.spike_count_button)

            # Convert all boundary times to sample index
            all_start_stop_times = tgui.convert_random_boundary_positions_from_time_to_samples(
                tgui, all_start_stop_times
            )

            # Convert the times to indicies and index out the relevant data. Calculate additional params
            # from this indexed data (deta im / vm and input resistance)
            (
                test_results,
                test_delta_im_pa,
                __,
                __,
            ) = tgui.calculate_test_measures_from_boundary_start_stop_indicies(
                all_start_stop_times,
                ["lower_bl_lr", "lower_exp_lr"],
                ["im_baseline", "im_steady_state"],
                rec_from,
                rec_to,
            )

            # Test they match for Loaded File, Stored Tabledata and the analysis results table
            for test_dataset in [
                tgui.mw.loaded_file.spkcnt_data,
                tgui.mw.stored_tabledata.spkcnt_data[filenum],
            ]:
                assert np.array_equal(
                    test_dataset["im_baseline"],
                    test_results["im_baseline"],
                    equal_nan=True,
                )
                assert np.array_equal(
                    test_dataset["im_steady_state"],
                    test_results["im_steady_state"],
                    equal_nan=True,
                )
                assert np.array_equal(test_dataset["im_delta"], test_delta_im_pa, equal_nan=True)

            # this always gets the last file on the table
            assert np.array_equal(
                tgui.get_data_from_qtable("im_baseline", rec_from, rec_to, analysis_type="Ri"),
                test_results["im_baseline"][rec_from : rec_to + 1],
            )
            assert np.array_equal(
                tgui.get_data_from_qtable("im_steady_state", rec_from, rec_to, analysis_type="Ri"),
                test_results["im_steady_state"][rec_from : rec_to + 1],
            )
            assert np.array_equal(
                tgui.get_data_from_qtable("im_delta", rec_from, rec_to, analysis_type="Ri"),
                test_delta_im_pa[rec_from : rec_to + 1],
            )

            # load a new file
            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data(tgui.time_type, "spkcnt")
        tgui.shutdown()

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize(
        "im_analysis_type",
        ["bounds_align_recs", "bounds_not_align_recs", "im_protocol"],
    )
    def test_rheobase(self, tgui, im_analysis_type, analyse_specific_recs):
        for filenum in range(3):
            if "bounds" in im_analysis_type:
                im_setting = "bounds"
                align_bounds_across_recs = True if im_analysis_type == "bounds_align_recs" else False
                bounds_im = tgui.get_analyse_across_recs_or_not_boundaries_dict_for_spikes(align_bounds_across_recs)
            else:
                im_setting = "im_protocol"
                bounds_im = {"start": 0.2, "stop": 0.8}

            tgui.switch_to_spikecounts_and_set_im_combobox(
                spike_bounds_on=False, im_setting=im_setting, im_groupbox_on=True
            )

            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(
                analyse_specific_recs, tgui.adata.current_injection_amplitude
            )
            (
                __,
                test_rheobase_rec,
                test_rheobase,
            ) = tgui.adata.generate_test_rheobase_data_from_spikeinfo(
                rec_from, rec_to, tgui.adata.spikes_per_rec, change_spikeinfo=False
            )  #
            self.overwrite_modeldata_to_test_rheobase(tgui, test_rheobase_rec)
            tgui.run_spikecount_analysis(["rheobase_record"], im_setting=im_setting, bounds_im=bounds_im)

            assert tgui.eq(
                tgui.mw.loaded_file.spkcnt_data.loc[0, "rheobase_rec"],
                test_rheobase_rec,
            )
            assert tgui.eq(
                tgui.mw.stored_tabledata.spkcnt_data[filenum].loc[0, "rheobase_rec"],
                test_rheobase_rec,
            )

            if im_analysis_type == "im_protocol":
                assert tgui.eq(
                    tgui.get_data_from_qtable("rheobase", rec_from, rec_to)[0],
                    tgui.adata.current_injection_amplitude[test_rheobase_rec],
                )
            else:
                assert tgui.eq(
                    tgui.get_data_from_qtable("rheobase", rec_from, rec_to)[0],
                    self.get_rheobase_accounting_for_different_bounds_per_rec(tgui, test_rheobase_rec, bounds_im),
                )

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data(tgui.time_type, "spkcnt")
        tgui.shutdown()

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_user_im_input_rheobase(self, tgui, analyse_specific_recs):
        """ """
        tgui.switch_to_spikecounts_and_set_im_combobox(
            spike_bounds_on=False, im_setting="user_input_im", im_groupbox_on=True
        )

        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)
        (
            __,
            test_rheobase_rec,
            __,
        ) = tgui.adata.generate_test_rheobase_data_from_spikeinfo(
            rec_from, rec_to, tgui.adata.spikes_per_rec, change_spikeinfo=False
        )

        self.overwrite_modeldata_to_test_rheobase(tgui, test_rheobase_rec)
        rows_to_fill_in = rec_to - rec_from + 1
        tgui.fill_user_im_input_widget(rows_to_fill_in, tgui.mw.mw.spkcnt_set_im_button)
        tgui.run_spikecount_analysis(["rheobase_record"], im_setting="user_input_im")

        assert tgui.eq(tgui.mw.loaded_file.spkcnt_data.loc[0, "rheobase_rec"], test_rheobase_rec)
        assert tgui.eq(
            tgui.mw.stored_tabledata.spkcnt_data[0].loc[0, "rheobase_rec"],
            test_rheobase_rec,
        )
        assert tgui.eq(
            tgui.get_data_from_qtable("rheobase", rec_from, rec_to)[0],
            tgui.mw.cfgs.spkcnt["user_input_im"]["step"][test_rheobase_rec - rec_from][0],
        )  # the user input Im is indexed from the rec_from

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("align_bounds_across_recs", [True, False])
    def test_rheobase_exact(self, tgui, analyse_specific_recs, align_bounds_across_recs):
        """ """
        bounds_im = tgui.get_analyse_across_recs_or_not_boundaries_dict_for_spikes(align_bounds_across_recs)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False, im_groupbox_on=True)
        im_ramp_array = self.overwrite_artificial_im_array_with_ramp_protocol(tgui)

        # zero out vm array to create a rheobase
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(
            analyse_specific_recs, tgui.adata.current_injection_amplitude
        )
        (
            __,
            test_rheobase_rec,
            __,
        ) = tgui.adata.generate_test_rheobase_data_from_spikeinfo(
            rec_from, rec_to, tgui.adata.spikes_per_rec, change_spikeinfo=False
        )
        self.overwrite_modeldata_to_test_rheobase(tgui, test_rheobase_rec)
        tgui.run_spikecount_analysis(["rheobase_exact"], im_setting="bounds", bounds_im=bounds_im)

        test_rheobase = self.find_true_rheobase_and_exact_im(
            tgui, im_ramp_array, test_rheobase_rec, bounds_im=bounds_im
        )

        assert utils.allclose(tgui.mw.loaded_file.spkcnt_data.loc[0, "rheobase"], test_rheobase, 1e-10)

        assert utils.allclose(
            tgui.mw.stored_tabledata.spkcnt_data[0].loc[0, "rheobase"],
            test_rheobase,
            1e-10,
        )
        assert utils.allclose(
            tgui.get_data_from_qtable("rheobase", rec_from, rec_to)[0],
            test_rheobase,
            1e-10,
        )

    def test_rheobase_exact_1_rec(self, tgui):
        """v2.4.0 had a bug in which 1 rec rheobase analysis would crashs"""
        tgui.setup_artificial_data(tgui.time_type, "spkcnt_1_rec")

        (
            num_recs,
            __,
            rec_from,
            rec_to,
        ) = tgui.setup_spkcnt_ramp_protocol_with_filled_protocol(tgui, analyse_specific_recs=False)
        __ = self.fill_im_protocol_with_default_ramp_step(tgui, accept=True)

        tgui.run_spikecount_analysis(["rheobase_exact"], im_setting="user_input_im")

        ramp = tgui.mw.loaded_file.make_im_protocol_from_user_input(
            tgui.mw.cfgs.spkcnt["user_input_im"]["ramp"]["protocol"]
        )

        test_rheobase = ramp[0][tgui.adata.spike_peak_idx[0][0].astype("int")]

        assert utils.allclose(tgui.mw.loaded_file.spkcnt_data.loc[0, "rheobase"], test_rheobase, 1e-10)

        assert utils.allclose(
            tgui.mw.stored_tabledata.spkcnt_data[0].loc[0, "rheobase"],
            test_rheobase,
            1e-10,
        )
        assert utils.allclose(
            tgui.get_data_from_qtable("rheobase", rec_from, rec_to)[0],
            test_rheobase,
            1e-10,
        )

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("fill_without_auto_ramping", [True, False])
    def test_user_ramp_protocol_generation(self, tgui, analyse_specific_recs, fill_without_auto_ramping):
        """ """
        num_recs, table, __, __ = tgui.setup_spkcnt_ramp_protocol_with_filled_protocol(tgui, analyse_specific_recs)

        data = table.dataframe_generation_from_table()
        assert data.shape == (
            1,
            4,
        ), "single record user-ramp protocol is not input correctly"

        if fill_without_auto_ramping:
            tgui.left_mouse_click(tgui.mw.dialogs["user_im_entry"].dia.fill_protocol_button)
            data = table.dataframe_generation_from_table()
            assert data.shape == (
                num_recs,
                4,
            ), "num recs does not match without auto ramping"
            for col in range(4):
                assert np.unique(data[:, col]).size == 1, "filled protocol columns are not all the same "

        else:
            default_step = self.fill_im_protocol_with_default_ramp_step(tgui)
            data = table.dataframe_generation_from_table()
            assert data.shape == (
                num_recs,
                4,
            ), "num recs does not match with auto ramping"
            for col in [0, 2]:
                assert (
                    np.unique(data[:, col]).size == 1
                ), "filled baseline and time protocol for ramp Im are not the same"
            assert (
                np.mean(np.diff(data[:, 1])) == default_step
            ), "filled ramp start Im protocol is not increasing exactly by 10"
            assert (
                np.mean(np.diff(data[:, 3])) == default_step
            ), "filled ramp stop Im protocol is not increasing exactly by 10"

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_rheobase_exact_with_user_ramp_protocol_generation_and_analysis(self, tgui, analyse_specific_recs):
        """ """
        (
            num_recs,
            __,
            rec_from,
            rec_to,
        ) = tgui.setup_spkcnt_ramp_protocol_with_filled_protocol(tgui, analyse_specific_recs)
        __ = self.fill_im_protocol_with_default_ramp_step(tgui, accept=True)

        (
            __,
            test_rheobase_rec,
            __,
        ) = tgui.adata.generate_test_rheobase_data_from_spikeinfo(
            rec_from, rec_to, tgui.adata.spikes_per_rec, change_spikeinfo=False
        )
        self.overwrite_modeldata_to_test_rheobase(tgui, test_rheobase_rec)
        tgui.run_spikecount_analysis(["rheobase_exact"], im_setting="user_input_im")

        saved_data = tgui.mw.loaded_file.saved_rheobase_settings["im_array"]
        assert saved_data.shape == (
            tgui.adata.num_recs,
            tgui.adata.num_samples,
        ), "full user Im ramp protocol is not padded"
        assert (
            saved_data[~np.isnan(saved_data[:, 0]), :].shape[0] == num_recs
        ), "full user Im ramp protocol specified recs do not match recs to analyse"

        baselines = np.zeros((num_recs, 1))
        test_rheobase = self.find_true_rheobase_and_exact_im(
            tgui,
            saved_data,
            test_rheobase_rec,
            provide_baselines_instead_or_measure_from_data=baselines,
        )

        assert (
            tgui.mw.loaded_file.spkcnt_data.loc[0, "rheobase"] == test_rheobase
        ), "model data rheobase is incorrect after ramp user Im protocol"
        assert (
            tgui.mw.stored_tabledata.spkcnt_data[0].loc[0, "rheobase"] == test_rheobase
        ), "stored table data rheobase is incorrect after ramp user Im protocol"
        assert tgui.eq(
            tgui.get_data_from_qtable("rheobase", rec_from, rec_to)[0], test_rheobase[0]
        ), "table rheobase is incorrect after ramp user Im protocol"

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Manual Selection
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # TODO: there is some DRY on these two tests, but they have some subtle differences. Still, could combine.

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_spike_removed_after_analysis(self, tgui, analyse_specific_recs):
        """
        Remove and then manually select spikes, checking fs latency is properly updated (see
        tgui function for more details).
        """
        for filenum in range(3):
            tgui.test_manual_spike_selection(
                tgui,
                filenum,
                analyse_specific_recs,
                run_analysis_function=self.run_spkcnt_for_delete_spike_function,
                spikes_to_delete_function=self.spkcnt_spikes_to_delete_function,
                test_function=self.event_selection_test_fs_latency,
            )
        tgui.shutdown()

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_spike_manual_remove_rheobase(self, tgui, analyse_specific_recs):
        """
        Remove and manually select spikes and check rheobase is properly updated.

        TODO: this is very similar to tgui.test_manual_spike_selection(), which has already been factored to
              incorporate test_spike_removed_after_analysis() and the test_events.py function test_event_selection().
              It should be refactored to support this function also.
        """
        (
            num_recs,
            __,
            rec_from,
            rec_to,
        ) = tgui.setup_spkcnt_ramp_protocol_with_filled_protocol(tgui, analyse_specific_recs)
        __ = self.fill_im_protocol_with_default_ramp_step(tgui, accept=True)

        (
            __,
            test_rheobase_rec,
            __,
        ) = tgui.adata.generate_test_rheobase_data_from_spikeinfo(
            rec_from, rec_to, tgui.adata.spikes_per_rec, change_spikeinfo=False
        )
        rec_from = test_rheobase_rec

        self.overwrite_modeldata_to_test_rheobase(tgui, test_rheobase_rec)

        tgui.run_spikecount_analysis(["rheobase_exact"], im_setting="user_input_im")

        spikes_to_delete, rec_from, rec_to = self.spkcnt_spikes_to_delete_function(tgui, rec_from, rec_to)

        peak_times = dict(
            all={"data": copy.deepcopy(tgui.adata.spike_sample_idx)},
            m_one=dict(data=[], time=[]),
            m_two=dict(data=[], time=[]),
            m_three=dict(data=[], time=[]),
            m_four=dict(data=[], time=[]),
        )

        for rec, spike_rec_idx, dict_key in spikes_to_delete:
            tgui.mw.update_displayed_rec(rec)
            deleted_spike_time = tgui.adata.peak_times[tgui.time_type][rec][spike_rec_idx]

            deleted_spike_peak = np.max(tgui.adata.cannonical_spike)

            tgui.click_upperplot_spotitem(
                tgui.mw.loaded_file_plot.spkcnt_plot,
                spike_rec_idx,
                doubleclick_to_delete=True,
            )

            self.remove_spike_from_adata(tgui, rec, spike_rec_idx, tgui.adata.spike_sample_idx)

            self.check_rheobase(tgui, num_recs, rec_from, rec_to, test_rheobase_rec)

            peak_times[dict_key]["data"] = copy.deepcopy(copy.deepcopy(tgui.adata.spike_sample_idx))
            peak_times[dict_key]["time"] = deleted_spike_time
            peak_times[dict_key]["amplitude"] = deleted_spike_peak
            peak_times[dict_key]["rec"] = rec

        tgui.left_mouse_click(tgui.mw.mw.spkcnt_click_mode_button)
        deleted_spike_keys = list(reversed([ele[2] for ele in spikes_to_delete]))

        for idx, deleted_spike_key in enumerate(deleted_spike_keys):
            rec = peak_times[deleted_spike_key]["rec"]
            tgui.mw.update_displayed_rec(rec)

            tgui.expand_xaxis_around_peak(tgui, peak_times[deleted_spike_key]["time"])

            tgui.manually_select_spike(
                rec,
                spike_num=None,
                overide_time_and_amplitude=peak_times[deleted_spike_key],
            )  # , rect_size_as_perc=0.02)

            level_up_data = "all" if deleted_spike_key == "m_one" else deleted_spike_keys[idx + 1]

            tgui.adata.spike_sample_idx = peak_times[level_up_data]["data"]
            self.check_rheobase(tgui, num_recs, rec_from, rec_to, test_rheobase_rec)

    def run_spkcnt_for_delete_spike_function(self, tgui):
        tgui.run_spikecount_analysis(["mean_isi_ms", "fs_latency_ms"])

    def spkcnt_spikes_to_delete_function(self, tgui, rec_from, rec_to):
        spikes_to_delete = get_settings(tgui.speed, tgui.analysis_type, num_recs=rec_from)["manually_del"]
        return spikes_to_delete, rec_from, rec_to

    def event_selection_test_fs_latency(self, tgui, filenum, rec_from, rec_to):
        tgui.switch_mw_tab(1)
        self.check_fs_latency(tgui, rec_from, rec_to, filenum)
        tgui.switch_mw_tab(0)

    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("align_bounds_across_recs", ["no_bounds", False, True])
    @pytest.mark.parametrize("spike_fa_type", ["spike_fa_local_variance", "spike_fa_divisor"])
    def test_spike_fa_divisor(self, tgui, analyse_specific_recs, align_bounds_across_recs, spike_fa_type):
        """ """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)

        spike_times = tgui.adata.peak_times[tgui.time_type]

        if align_bounds_across_recs == "no_bounds":
            tgui.run_spikecount_analysis(["spike_fa"])
        else:
            bounds_vm = tgui.get_analyse_across_recs_or_not_boundaries_dict_for_spikes(align_bounds_across_recs)
            tgui.run_spikecount_analysis(["spike_fa", "bounds"], bounds_vm=bounds_vm)
            spike_times = test_utils.vals_within_bounds(spike_times, bounds_vm["exp"][0], bounds_vm["exp"][1])

        test_fa = self.calculate_fa(tgui, spike_times, rec_from, rec_to, spike_fa_type)

        assert utils.allclose(
            tgui.mw.loaded_file.spkcnt_data[spike_fa_type][rec_from : rec_to + 1],
            test_fa,
            1e-10,
        )
        assert utils.allclose(
            tgui.mw.stored_tabledata.spkcnt_data[0][spike_fa_type][rec_from : rec_to + 1],
            test_fa,
            1e-10,
        )
        assert utils.allclose(tgui.get_data_from_qtable(spike_fa_type, rec_from, rec_to), test_fa, 1e-10)
