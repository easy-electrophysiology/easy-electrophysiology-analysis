from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtTest import QTest
import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
import easy_electrophysiology.easy_electrophysiology

from easy_electrophysiology.mainwindow.mainwindow import MainWindow
from ephys_data_methods import core_analysis_methods
from utils import utils

keyClick = QTest.keyClick
from setup_test_suite import GuiTestSetup
from slow_vs_fast_settings import get_settings
import scipy.signal
import copy
import peakutils
from sys import platform
from scipy import stats

SPEED = "fast"
os.environ["PYTEST_QT_API"] = "PySide6"


class TestDataToolsGui:
    """
    Test all data tools (settings, changes to data) through the GUI
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
        tgui.setup_artificial_data(request.param, analysis_type="data_tools")
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    def generate_random_data_in_tgui(self, tgui):
        new_vm_array = np.random.random((tgui.adata.num_recs, tgui.adata.num_samples)) * 200
        new_im_array = np.random.random((tgui.adata.num_recs, tgui.adata.num_samples)) * 200

        tgui.mw.loaded_file.data.vm_array = new_vm_array
        tgui.mw.loaded_file.data.im_array = new_im_array

        return new_vm_array, new_im_array

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Test Widgets (separate from configs because these are not saved)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_upsample_widgets(self, tgui):
        """
        Test all the widgets are responding correctly on the Upsample GUI. The interpolation factor should be between 1-15,
        and interpolation options are "linear", "nearest" and "cubic" (see scipy interp1d).
        """
        tgui.mw.mw.actionUpsample.trigger()
        upsample_dialog = tgui.mw.dialogs["upsample"]

        assert (
            upsample_dialog.dia.upsample_dialog_spinbox.value() == upsample_dialog.interp_factor
        ), "default init value"
        upsample_dialog.dia.upsample_dialog_spinbox.setValue(10),
        assert upsample_dialog.interp_factor == 10, "no change response"
        upsample_dialog.dia.upsample_dialog_spinbox.setValue(0)
        assert upsample_dialog.interp_factor == 1, "min bound incorrect"
        upsample_dialog.dia.upsample_dialog_spinbox.setValue(16)
        assert upsample_dialog.interp_factor == 15, "max bound incorrect"

        assert upsample_dialog.interp_type.lower() in upsample_dialog.dia.upsample_dialog_combobox.currentText().lower()
        tgui.set_combobox(upsample_dialog.dia.upsample_dialog_combobox, 0)
        assert "cubic" in upsample_dialog.dia.upsample_dialog_combobox.currentText().lower(), "cubic combobox text"
        assert "cubic" in upsample_dialog.interp_type.lower(), "cubic saved var"
        tgui.set_combobox(upsample_dialog.dia.upsample_dialog_combobox, 1)
        assert "linear" in upsample_dialog.dia.upsample_dialog_combobox.currentText().lower(), "linear combobox text"
        assert "linear" in upsample_dialog.interp_type.lower(), "linear saved var"
        tgui.set_combobox(upsample_dialog.dia.upsample_dialog_combobox, 2)
        assert "nearest" in upsample_dialog.dia.upsample_dialog_combobox.currentText().lower(), "nearest combobox text"
        assert "nearest" in upsample_dialog.interp_type.lower(), "nearest saved var"

    def test_downsample_widgets(self, tgui):
        """
        Test the widgets in the downsample gui are correct. Downsample factor should be between 1-12 (this is lower than interpolate
        due to filtering needs to be performed in two steps if > 13. Filter order should be under 40, effective filter order is half
        this due to use of filtfilt. Current filter options are butter and bessel.
        """
        tgui.mw.mw.actionDownsample.trigger()
        downsample_dialog = tgui.mw.dialogs["downsample"]

        assert (
            downsample_dialog.dia.downsample_factor_spinbox.value() == downsample_dialog.downsample_factor
        ), "factor default init value"
        downsample_dialog.dia.downsample_factor_spinbox.setValue(10)
        assert downsample_dialog.downsample_factor == 10, "factor no change response"
        downsample_dialog.dia.downsample_factor_spinbox.setValue(
            1
        )  # permitted upsampling factor should be between 1 - 15
        assert downsample_dialog.downsample_factor == 2, "factor min bound incorrect"
        downsample_dialog.dia.downsample_factor_spinbox.setValue(13)
        assert downsample_dialog.downsample_factor == 12, "factor max bound incorrect"

        assert (
            downsample_dialog.filter_.lower() in downsample_dialog.dia.filter_combobox.currentText().lower()
        ), "filter default init value"
        tgui.set_combobox(downsample_dialog.dia.filter_combobox, 0)
        assert "bessel" in downsample_dialog.dia.filter_combobox.currentText().lower(), "bessel combobox text"
        assert "bessel" == downsample_dialog.filter_.lower(), "bessel saved var"
        tgui.set_combobox(downsample_dialog.dia.filter_combobox, 1)
        assert "butter" in downsample_dialog.dia.filter_combobox.currentText().lower(), "butter combobox text"
        assert "butter" == downsample_dialog.filter_.lower(), "butter saved bar"

        assert (
            downsample_dialog.dia.filter_order_spinbox.value() / 2 == downsample_dialog.filter_order
        ), "order default init value"
        downsample_dialog.dia.filter_order_spinbox.setValue(1)
        assert downsample_dialog.dia.filter_order_spinbox.value() == 2, "order min bound incorrect"
        assert downsample_dialog.filter_order == 1, "order var min bound incorrect"
        keyClick(downsample_dialog.dia.filter_order_spinbox, QtGui.Qt.Key_Up)
        assert downsample_dialog.dia.filter_order_spinbox.value() == 4, "order change bound incorrect"
        assert downsample_dialog.filter_order == 2, "order var change bound incorrect"
        downsample_dialog.dia.filter_order_spinbox.setValue(41), "order min bound incorrect"
        assert downsample_dialog.dia.filter_order_spinbox.value() == 40, "order var min bound incorrect"
        assert downsample_dialog.filter_order == 20

    def test_filter_widgets(self, tgui):
        """
        Test Fiter gui options are working correctly. Filter order should be 1-40 (see filter_data or above).
        """
        tgui.mw.mw.actionFilter.trigger()
        filter_dialog = tgui.mw.dialogs["filter_data"]

        # filter type combobox
        assert filter_dialog.filter_.lower() in filter_dialog.dia.filter_combobox.currentText().lower()

        tgui.set_combobox(filter_dialog.dia.filter_combobox, 0)
        assert "bessel" in filter_dialog.dia.filter_combobox.currentText().lower(), "bessel combobox text"
        assert "bessel" == filter_dialog.filter_.lower(), "bessel saved var"
        tgui.set_combobox(filter_dialog.dia.filter_combobox, 1)
        assert "butter" in filter_dialog.dia.filter_combobox.currentText().lower(), "butter combobox text"
        assert "butter" == filter_dialog.filter_.lower(), "butter saved var"

        # filter_order_spinbox
        assert (
            filter_dialog.dia.filter_order_spinbox.value() / 2 == filter_dialog.filter_order
        ), "order default init value"
        filter_dialog.dia.filter_order_spinbox.setValue(1)
        assert filter_dialog.dia.filter_order_spinbox.value() == 2, "order min bound incorrect"
        assert filter_dialog.filter_order == 1, "order var min bound incorrect"
        keyClick(filter_dialog.dia.filter_order_spinbox, QtGui.Qt.Key_Up)
        assert filter_dialog.dia.filter_order_spinbox.value() == 4, "order change  incorrect"
        assert filter_dialog.filter_order == 2, "order var change incorrect"
        filter_dialog.dia.filter_order_spinbox.setValue(41)
        assert filter_dialog.dia.filter_order_spinbox.value() == 40, "order min bound incorrect"
        assert filter_dialog.filter_order == 20, "order var min bound incorrect"

        # filter_lowpass_radiobutton
        test_low_or_highpass = "lowpass" if filter_dialog.dia.filter_lowpass_radiobutton.isChecked() else "highpass"
        assert filter_dialog.low_or_highpass == test_low_or_highpass, "lowpass init"
        tgui.left_mouse_click(filter_dialog.dia.filter_highpass_radiobutton)
        assert filter_dialog.low_or_highpass == "highpass", "toggle highpass"
        tgui.left_mouse_click(filter_dialog.dia.filter_lowpass_radiobutton)
        assert filter_dialog.low_or_highpass == "lowpass", "toggle lowpass"

        # filter cutoff spinbox
        assert filter_dialog.dia.filter_cutoff_spinbox.value() == np.floor(tgui.adata.fs / 2) - 1, "filter cutoff init"
        filter_dialog.dia.filter_cutoff_spinbox.setValue(0)
        assert filter_dialog.dia.filter_cutoff_spinbox.value() == 1, "filter cutoff min"

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        filter_dialog.dia.filter_cutoff_spinbox.setValue(np.floor(tgui.adata.fs / 2))
        assert (
            filter_dialog.dia.filter_cutoff_spinbox.value() == np.floor(tgui.adata.fs / 2) - 1
        ), "filter cutoff over nyq"

    def test_cut_trace_length_spinboxes(self, tgui):
        """
        Test time spinboxes match data time and just change and try again for fun

        NOTE: Don't change tgui.mw.dialogs["cut_trace"] to dialog holding var, need to keep reference live after close / reopen.
        """
        tgui.mw.mw.actionCut_Down_Trace_Time.trigger()

        assert tgui.mw.dialogs["cut_trace"].dia.new_start_time_spinbox.maximum() == np.round(
            tgui.mw.loaded_file.data.min_max_time[0][-1], 6
        )
        assert tgui.mw.dialogs["cut_trace"].dia.new_start_time_spinbox.minimum() == np.round(
            tgui.mw.loaded_file.data.min_max_time[0][0], 6
        )
        assert tgui.mw.dialogs["cut_trace"].dia.new_stop_time_spinbox.maximum() == np.round(
            tgui.mw.loaded_file.data.min_max_time[0][-1], 6
        )
        assert tgui.mw.dialogs["cut_trace"].dia.new_stop_time_spinbox.minimum() == np.round(
            tgui.mw.loaded_file.data.min_max_time[0][0], 6
        )

        tgui.left_mouse_click(
            tgui.mw.dialogs["cut_trace"].dia.cut_trace_buttonbox.button(QtWidgets.QDialogButtonBox.Cancel)
        )
        tgui.mw.loaded_file.data.min_max_time = np.array([[1, 5]])

        tgui.mw.mw.actionCut_Down_Trace_Time.trigger()

        assert tgui.mw.dialogs["cut_trace"].dia.new_start_time_spinbox.maximum() == 5
        assert tgui.mw.dialogs["cut_trace"].dia.new_start_time_spinbox.minimum() == 1
        assert tgui.mw.dialogs["cut_trace"].dia.new_stop_time_spinbox.maximum() == 5
        assert tgui.mw.dialogs["cut_trace"].dia.new_stop_time_spinbox.minimum() == 1

    def test_cut_trace_length_radiobuttons(self, tgui):
        tgui.mw.mw.actionCut_Down_Trace_Time.trigger()
        dialog = tgui.mw.dialogs["cut_trace"]

        assert dialog.time_method == "cumulative"

        tgui.switch_checkbox(dialog.dia.normalise_radiobutton, on=True)
        assert dialog.time_method == "normalised"

        tgui.switch_checkbox(dialog.dia.cumulative_radiobutton, on=True)
        assert dialog.time_method == "cumulative"

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Functional Tests of Data Manipulation
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_upsample_data(self, tgui):
        """
        Upsample factor 10, uses default (cubic spline).
        See test in test_core_analysis_methods interpolate for details.
        To avoid modal dialog problems, the gui is tested above and the method call used here. Just the
        default is run through gui to check config change.
        """
        tgui.mw.loaded_file.upsample_data("cubic", 10)
        assert (
            tgui.mw.loaded_file.data.vm_array[:, 0] == tgui.mw.loaded_file.raw_data.vm_array[:, 0]
        ).all(), "first sample changed "
        assert utils.allclose(
            tgui.mw.loaded_file.data.vm_array[0, 0::10],
            tgui.mw.loaded_file.raw_data.vm_array[0, :],
            1e-10,
        ), "every nth smapled does not match"
        assert not (
            tgui.mw.loaded_file.data.vm_array[:, 3] == tgui.mw.loaded_file.raw_data.vm_array[:, 3]
        ).all(), "interpolated sample matches original (?)"
        assert (
            tgui.mw.loaded_file.data.vm_array[:, -1] == tgui.mw.loaded_file.raw_data.vm_array[:, -1]
        ).all(), "last sample changed"
        assert (
            np.shape(tgui.mw.loaded_file.data.vm_array)[1]
            == 10 * (np.shape(tgui.mw.loaded_file.raw_data.vm_array)[1] - 1) + 1
        ), "new num samples wrong "

        tgui.mw.mw.actionUpsample.trigger()
        QtWidgets.QApplication.processEvents()

        tgui.left_mouse_click(
            tgui.mw.dialogs["upsample"].dia.upsample_dialog_buttonbox.button(QtWidgets.QDialogButtonBox.Apply)
        )
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    def test_downsample_data(self, tgui):
        """
        Cannot test against raw_data because of filtering. Here downsample and check signal is downsampled + against
        scipy decimate function with same filter provided.
        """
        # run test
        tgui.mw.loaded_file.downsample_data(10, "bessel", 8)
        x_ = copy.copy(tgui.mw.loaded_file.raw_data.vm_array)
        nyq = tgui.mw.loaded_file.raw_data.fs / 2
        cutoff = np.floor((tgui.mw.loaded_file.raw_data.fs / 10) / 2)
        test = scipy.signal.decimate(
            x_,
            q=10,
            n=1,  # uses EE defaults - will fail if changed. These settings are not saved in file cfg so cannot access.
            ftype=scipy.signal.dlti(*scipy.signal.bessel(8, cutoff / nyq)),
            axis=1,
        )

        assert np.shape(test)[1] == np.shape(tgui.mw.loaded_file.data.vm_array)[1], "new shape incorrect"
        assert (
            np.shape(tgui.mw.loaded_file.raw_data.vm_array[:, 0::10])[1]
            == np.shape(tgui.mw.loaded_file.data.vm_array)[1]
        ), "orig points dont match"

        assert utils.allclose(test, tgui.mw.loaded_file.data.vm_array), "does not match direct scipy analyis"

        tgui.mw.mw.actionDownsample.trigger()
        QtWidgets.QApplication.processEvents()

        tgui.left_mouse_click(
            tgui.mw.dialogs["downsample"].dia.downsample_dialog_buttonbox.button(QtWidgets.QDialogButtonBox.Apply)
        )
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    def test_filter_data(self, tgui):
        """
        Filter the artificual data (added sine waves) and check the remaining sine waves are correct.
        See test_core_analysis_methods fitler tests for more detail.
        """
        tgui.mw.loaded_file.filter_data("bessel", 10, 2000, "lowpass")

        for rec in range(tgui.adata.num_recs):
            tgui.mw.update_displayed_rec(rec)
            fft_results = core_analysis_methods.get_fft(
                tgui.mw.loaded_file.data.vm_array[rec],
                False,
                tgui.adata.num_samples,
                True,
            )
            peaks_idx = peakutils.indexes(fft_results["Y"], thres=0.05, min_dist=5)
            hz_to_add = tgui.adata.hz_to_add[rec]

            hz_to_add = hz_to_add[
                (fft_results["N"] < hz_to_add) & (hz_to_add < 500)
            ]  # account for cut down freqs returned (done in EE for plot appearance)
            if hz_to_add:
                assert np.array_equal(fft_results["freqs"][peaks_idx], hz_to_add), "Failed FFT Test"

        if (
            not platform == "darwin"
        ):  # for some reason this gives a segmentation fault in test environment, works fine in EE
            tgui.mw.mw.actionFilter.trigger()
            QtWidgets.QApplication.processEvents()
            tgui.mw.dialogs["filter_data"].handle_apply()

        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    def test_reshape_concatenate_data(self, tgui):
        """
        Refactor on revisit - quick function to check that concatenating records works.
        """
        tgui.shutdown()
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = SPEED
        tgui.setup_artificial_data("cumulative", analysis_type="spkcnt")
        tgui.mw.raise_()

        initial_time_array = copy.copy(tgui.mw.loaded_file.data.time_array)
        initial_time_array -= np.atleast_2d(tgui.mw.loaded_file.data.min_max_time[:, 0]).T
        num_recs = copy.copy(tgui.mw.loaded_file.data.num_recs)

        total_time = (tgui.mw.loaded_file.data.num_recs * tgui.mw.loaded_file.data.min_max_time[0][1]) + (
            (tgui.mw.loaded_file.data.num_recs - 1) * tgui.mw.loaded_file.data.ts
        )
        tmp_time_1 = np.linspace(
            0,
            total_time,
            tgui.mw.loaded_file.data.num_samples * tgui.mw.loaded_file.data.num_recs,
        )

        offsets = (np.arange(num_recs) * tgui.mw.loaded_file.data.min_max_time[0][1]) + (
            np.arange(tgui.mw.loaded_file.data.num_recs) * tgui.mw.loaded_file.data.ts
        )
        tmp_time_2 = (initial_time_array + np.atleast_2d(offsets).T).flatten()

        tgui.mw.mw.actionReshape_Records.trigger()
        tgui.left_mouse_click(
            tgui.mw.dialogs["reshape"].dia.reshape_data_button_box.button(QtWidgets.QDialogButtonBox.Apply)
        )

        new_time = tgui.mw.loaded_file.data.time_array

        assert utils.allclose(tmp_time_1, new_time, 1e-10)
        assert utils.allclose(tmp_time_2, new_time, 1e-10)

        tgui.mw.mw.actionReshape_Records.trigger()
        tgui.left_mouse_click(tgui.mw.dialogs["reshape"].dia.reshape_to_multiple_records_radiobutton)
        tgui.switch_checkbox(tgui.mw.dialogs["reshape"].dia.norm_time_checkbox, on=True)
        tgui.enter_number_into_spinbox(
            tgui.mw.dialogs["reshape"].dia.reshape_data_dialog_spinbox,
            str(num_recs),
            setValue=True,
        )

        tgui.left_mouse_click(
            tgui.mw.dialogs["reshape"].dia.reshape_data_button_box.button(QtWidgets.QDialogButtonBox.Apply)
        )

        assert utils.allclose(initial_time_array, tgui.mw.loaded_file.data.time_array, 1e-10)
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

        tgui.shutdown()

    def test_normalise_time(self, tgui):
        """
        Test normalise time. Load a cumulative time trace and check that after normalisation the time is no longer cumulative, but
        the same as the first rec across all records.
        """
        tgui.shutdown()
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = SPEED
        tgui.setup_artificial_data("cumulative", analysis_type="spkcnt")
        tgui.mw.raise_()
        rec = 3
        tgui.mw.update_displayed_rec(rec)

        tmp_time = tgui.mw.loaded_file.data.min_max_time
        tmp_x_axis = copy.copy(tgui.mw.loaded_file_plot.curve_upper.xData)

        tgui.mw.mw.actionNormalise_Timescale.trigger()
        tgui.left_mouse_click(
            tgui.mw.dialogs["normalise_time"].dia.normalise_time_buttonbox.button(QtWidgets.QDialogButtonBox.Apply)
        )
        tgui.mw.update_displayed_rec(rec)

        assert (
            tmp_time[0][0] == tgui.mw.loaded_file.data.min_max_time[:, 0]
        ).all(), "new time does not match orig file rec time - start "
        assert (
            tmp_time[0][1] == tgui.mw.loaded_file.data.min_max_time[:, 1]
        ).all(), "new time does not match orig file rec time - stop"
        assert tgui.eq(
            tmp_time[1][1],
            (tgui.mw.loaded_file.data.min_max_time[0][1] * 2) + tgui.adata.ts,
        ), "second rec is not normalised first rec * 2"
        assert tgui.eq(
            tgui.mw.loaded_file.data.min_max_time[-1],
            tgui.mw.loaded_file.data.min_max_time[0],
        ), "first and last rec time last sample do not match"
        assert (tmp_x_axis != tgui.mw.loaded_file_plot.curve_upper.xData).all(), "x axis plot incorrect"

        assert tgui.eq(
            tgui.mw.loaded_file_plot.curve_upper.xData,
            tgui.mw.loaded_file.data.time_array[rec],
        ), "plot incorrect"
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

        tgui.shutdown()

    def fill_tgui_loaded_file_data_with_half_step_protocol(self, tgui, num_samples, half_num_samples):
        """ """
        new_data_array = utils.np_empty_nan((tgui.adata.num_recs, tgui.adata.num_samples))
        new_time_array = utils.np_empty_nan((tgui.adata.num_recs, tgui.adata.num_samples))

        for rec in range(tgui.adata.num_recs):
            new_time_array[rec, :] = (
                np.arange(num_samples)
                if tgui.time_type == "normalised"
                else np.arange(rec * num_samples, rec * num_samples + num_samples)
            )
            y = np.hstack(
                [
                    np.ones(half_num_samples) * np.random.randint(0, 5000),
                    np.ones(half_num_samples) * np.random.randint(0, 5000),
                ]
            )
            new_data_array[rec, :] = y

        tgui.mw.loaded_file.data.vm_array = new_data_array
        tgui.mw.loaded_file.data.im_array = new_data_array
        tgui.mw.loaded_file.data.time_array = new_time_array
        tgui.mw.loaded_file.data.min_max_time[:, 0] = new_time_array[:, 0]
        tgui.mw.loaded_file.data.min_max_time[:, 1] = new_time_array[:, -1]

        return new_data_array

    @pytest.mark.parametrize("mode", ["dont_align_across_recs", "align_across_recs"])
    @pytest.mark.parametrize("match_im", [True, False])
    def test_remove_baseline(self, tgui, match_im, mode):
        """
        Make a new data array which is a steady signal with a single step halfway through
        Remove baseline from the first half, and test that the first half of the step is now zero
        and the second half is the offset from zero.

        If bounds are aligned across recs, set the bounds to the first half and subtract it from the data. Then test
        the first half is zero and the second half the pre-determined data offset.

        Otherwise, if the bounds are not aligned across recs, set all even recs to subtract the first half
        of the data and all off recs to subtract the second.
        """
        # Setup boundary data and analysis
        num_samples = tgui.adata.num_samples
        half_num_samples = int(num_samples / 2)
        new_data_array = self.fill_tgui_loaded_file_data_with_half_step_protocol(tgui, num_samples, half_num_samples)
        tgui.set_link_across_recs(tgui, mode)

        tgui.mw.mw.actionRemove_Baseline.trigger()

        # Iterate through all recs and set the boundaries, either to the normalised idx for the record
        # or adjusted for cumulatively increasing records
        for rec in range(tgui.adata.num_recs):
            tgui.mw.update_displayed_rec(rec)

            if mode == "align_across_recs" or rec % 2 == 0:
                bounds = (
                    (0, int(num_samples / 4))
                    if tgui.time_type == "normalised"
                    else np.array([0, int(num_samples / 4)]) + (num_samples * rec)
                )
            else:
                bounds = (
                    (int(num_samples * 0.75), num_samples)
                    if tgui.time_type == "normalised"
                    else np.array([int(num_samples * 0.75), num_samples]) + (num_samples * rec)
                )

            tgui.mw.dialogs["remove_baseline"].bounds.bounds["upper_bl_lr"].setRegion(tuple(bounds))

        # Set Im on / off and run
        tgui.switch_checkbox(
            tgui.mw.dialogs["remove_baseline"].dia.remove_from_other_data_checkbox,
            on=match_im,
        )

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(
            tgui.mw.dialogs["remove_baseline"].dia.remove_baseline_buttonbox.button(QtWidgets.QDialogButtonBox.Apply)
        )

        # Iterate through recs and check that the data has been offset correctly. If the recs are not aligned or
        # recs are even, the first half of the data is subtract. Otherwise, the second half of the data is subtracted.
        second_array_half_offset = new_data_array[:, -1] - new_data_array[:, 0]
        for rec in range(tgui.adata.num_recs):
            if mode == "align_across_recs" or rec % 2 == 0:
                assert (tgui.mw.loaded_file.data.vm_array[rec, 0:half_num_samples] == 0).all()
                assert (
                    tgui.mw.loaded_file.data.vm_array[rec, half_num_samples:] == second_array_half_offset[rec]
                ).all()
                if match_im:
                    assert (tgui.mw.loaded_file.data.im_array[rec, 0:half_num_samples] == 0).all()
                    assert (
                        tgui.mw.loaded_file.data.im_array[rec, half_num_samples:] == second_array_half_offset[rec]
                    ).all()
                else:
                    assert np.array_equal(tgui.mw.loaded_file.data.im_array, new_data_array)
            else:
                assert (
                    tgui.mw.loaded_file.data.vm_array[rec, 0:half_num_samples] == -second_array_half_offset[rec]
                ).all()
                assert (tgui.mw.loaded_file.data.vm_array[rec, half_num_samples:] == 0).all()
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    @pytest.mark.parametrize("poly_order", [*range(1, 20)])
    def test_detrend(self, tgui, poly_order):
        """
        Make a random nth order polynomail per rec, insert it at the data, run.
        This implicitly checks the key widgets for this dialog too.
        """
        new_vm_array = utils.np_empty_nan((tgui.adata.num_recs, tgui.adata.num_samples))
        for rec in range(tgui.adata.num_recs):
            coefs = np.random.uniform(-0.01, 0.01, poly_order) * 100
            x = np.linspace(0, 1, tgui.adata.num_samples)
            y = np.polyval(coefs, x)
            new_vm_array[rec, :] = y

        save_mean = np.atleast_2d(np.mean(new_vm_array, axis=1))
        tgui.mw.loaded_file.raw_data.vm_array = new_vm_array
        tgui.mw.loaded_file.raw_data.im_array = np.zeros((tgui.adata.num_recs, tgui.adata.num_samples))
        tgui.mw.loaded_file.data.vm_array = new_vm_array
        tgui.mw.loaded_file.data.im_array = np.zeros((tgui.adata.num_recs, tgui.adata.num_samples))

        tgui.mw.mw.actionDetrend.trigger()
        tgui.enter_number_into_spinbox(
            tgui.mw.dialogs["detrend"].dia.sepsc_detrend_spinbox,
            poly_order,
            setValue=True,
        )

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(tgui.mw.dialogs["detrend"].dia.apply_button)
        assert utils.allclose(tgui.mw.loaded_file.data.vm_array - save_mean.T, 0, 1e-10)
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    def test_average__all_records(self, tgui):
        new_vm_array, new_im_array = self.generate_random_data_in_tgui(tgui)

        tgui.mw.mw.actionAverage_Records.trigger()

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(
            tgui.mw.dialogs["average_records"].dia.average_records_buttonbox.button(QtWidgets.QDialogButtonBox.Apply)
        )

        assert utils.allclose(np.mean(new_vm_array, axis=0), tgui.mw.loaded_file.data.vm_array, 1e-10)
        assert utils.allclose(np.mean(new_im_array, axis=0), tgui.mw.loaded_file.data.im_array, 1e-10)
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    def test_average_individual_records(self, tgui):
        """ """
        new_vm_array, new_im_array = self.generate_random_data_in_tgui(tgui)

        tgui.mw.mw.actionAverage_Records.trigger()
        tgui.set_combobox(tgui.mw.dialogs["average_records"].dia.average_records_spinbox, 1)

        # randomly split the record numbers into chunks of random size to average
        permute_records = np.random.permutation(range(1, tgui.adata.num_recs + 1))
        split_idx = np.sort(
            np.random.choice(
                range(1, tgui.adata.num_recs),
                get_settings(tgui.speed, "data_tools")["recs_to_split"],
                replace=False,
            )
        )
        recs_to_average = np.split(permute_records, split_idx)

        # save the average of the recs to average, and input them into the table and run in EE and test
        avg_vm = utils.np_empty_nan((len(recs_to_average), tgui.adata.num_samples))
        avg_im = utils.np_empty_nan((len(recs_to_average), tgui.adata.num_samples))
        for idx, recs in enumerate(recs_to_average):
            avg_vm[idx, :] = np.mean(new_vm_array[recs - 1], axis=0)
            avg_im[idx, :] = np.mean(new_im_array[recs - 1], axis=0)

            text_input = ",".join(recs.astype(str))
            num = QtWidgets.QTableWidgetItem(text_input)
            tgui.mw.dialogs["average_records"].average_records_table_dialog.dia.average_records_table.setItem(
                idx, 0, num
            )

        tgui.left_mouse_click(
            tgui.mw.dialogs["average_records"].average_records_table_dialog.dia.average_records_buttonbox.button(
                QtWidgets.QDialogButtonBox.Ok
            )
        )

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(
            tgui.mw.dialogs["average_records"].dia.average_records_buttonbox.button(QtWidgets.QDialogButtonBox.Apply)
        )

        assert np.array_equal(avg_vm, tgui.mw.loaded_file.data.vm_array)
        assert np.array_equal(avg_im, tgui.mw.loaded_file.data.im_array)
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    def get_nearest_time(self, time_array, time_, round_dp=False, force_lower=False):
        nearest_time = time_array[np.argmin(abs(time_array - time_))]
        if force_lower:
            round_nearest_time = (
                np.floor(nearest_time * 10**round_dp) / 10**round_dp if round_dp is not False else nearest_time
            )
        else:
            round_nearest_time = round(nearest_time, round_dp) if round_dp is not False else nearest_time
        return round_nearest_time

    def setup_and_run_cut_trace_length(self, tgui, time_method, init_start_time, init_stop_time):
        start_time = self.get_nearest_time(tgui.mw.loaded_file.data.time_array[0], init_start_time)
        stop_time = self.get_nearest_time(tgui.mw.loaded_file.data.time_array[0], init_stop_time, force_lower=True)
        new_time = self.get_nearest_time(tgui.mw.loaded_file.data.time_array[0], stop_time - start_time)

        # copy and run
        save_start_times = copy.deepcopy(tgui.mw.loaded_file.data.time_array[:, 0])
        save_fs = copy.deepcopy(tgui.mw.loaded_file.data.fs)

        tgui.mw.mw.actionCut_Down_Trace_Time.trigger()
        dialog = tgui.mw.dialogs["cut_trace"]

        self.set_cut_trace_length_radiobutton(tgui, dialog, time_method)

        tgui.enter_number_into_spinbox(dialog.dia.new_start_time_spinbox, round(start_time, 6), setValue=True)
        tgui.enter_number_into_spinbox(dialog.dia.new_stop_time_spinbox, round(stop_time, 6), setValue=True)

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(dialog.dia.cut_trace_buttonbox.button(QtWidgets.QDialogButtonBox.Apply))

        return save_fs, save_start_times, new_time, start_time, stop_time

    def set_cut_trace_length_radiobutton(self, tgui, dialog, time_method):
        radiobuttons = {
            "cumulative": dialog.dia.cumulative_radiobutton,
            "normalised": dialog.dia.normalise_radiobutton,
        }
        tgui.switch_checkbox(radiobuttons[time_method], on=True)

    @pytest.mark.parametrize(
        "start_stop_times",
        [
            [0, "full"],
            [0, 0.5],
            [0.05, 0.1],
            [0.1, 0.5],
            [0.5, 1],
            [1, 1.25],
            [1.5, "full"],
        ],
    )
    def test_cut_trace_length_on_data_normalised(self, tgui, start_stop_times):
        """
        The cut trace length cuts the trace to the time period specified and the time
        then becomes the specified stop time - start time (e.g. 1.0 - 1.5 s becomes 0 - 0.5 s). If data is cumulative, it is
        normalised. The time is upper-bound inclusive.

        get_nearest_time() returns input round to 3dp for input into spinbox

        Similar to _raw_times and _cumulative version below, had all in one function but was too messy
        """
        init_start_time, init_stop_time = start_stop_times
        init_stop_time = (
            copy.deepcopy(tgui.mw.loaded_file.data.min_max_time[0][-1]) if init_stop_time == "full" else init_stop_time
        )

        (
            save_fs,
            save_start_times,
            new_time,
            start_time,
            stop_time,
        ) = self.setup_and_run_cut_trace_length(tgui, "normalised", init_start_time, init_stop_time)

        assert utils.allclose(save_fs, tgui.mw.loaded_file.data.fs, 1e-10)
        assert utils.allclose(
            new_time - tgui.mw.loaded_file.data.ts,
            tgui.mw.loaded_file.data.time_array[:, -1],
            1e-10,
        )
        assert utils.allclose(0, tgui.mw.loaded_file.data.time_array[:, 0], 1e-10)
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    @pytest.mark.parametrize(
        "start_stop_times",
        [
            [0, "full"],
            [0, 0.5],
            [0.05, 0.1],
            [0.1, 0.5],
            [0.5, 1],
            [1, 1.25],
            [1.5, "full"],
        ],
    )
    def test_cut_trace_length_on_data_cumulative(self, tgui, start_stop_times):
        """ """
        init_start_time, init_stop_time = start_stop_times
        init_stop_time = (
            copy.deepcopy(tgui.mw.loaded_file.data.min_max_time[0][-1]) if init_stop_time == "full" else init_stop_time
        )

        (
            save_fs,
            save_start_times,
            new_time,
            start_time,
            stop_time,
        ) = self.setup_and_run_cut_trace_length(tgui, "cumulative", init_start_time, init_stop_time)

        cum_new_time = new_time * np.arange(tgui.mw.loaded_file.data.num_recs)
        last_datapoint_on_recs = cum_new_time + new_time - tgui.mw.loaded_file.data.ts

        assert utils.allclose(save_fs, tgui.mw.loaded_file.data.fs, 1e-10)
        assert utils.allclose(last_datapoint_on_recs, tgui.mw.loaded_file.data.time_array[:, -1], 1e-10)
        assert utils.allclose(cum_new_time, tgui.mw.loaded_file.data.time_array[:, 0], 1e-10)
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    @pytest.mark.parametrize(
        "start_stop_samples",
        [
            [0, 50],
            [0, 10000],
            [0, 11131],
            [1, "full"],
            [0, "full"],
            [1000, 2000],
            [1500, 5000],
            [2500, 5000],
            [10000, 12500],
            [12500, 15000],
            [15000, "full"],
        ],
    )
    @pytest.mark.parametrize("time_method", ["cumulative", "cumulative"])
    def test_cut_trace_length_sample(self, tgui, start_stop_samples, time_method):
        saved_vm_array = copy.deepcopy(tgui.mw.loaded_file.data.vm_array)
        saved_im_array = copy.deepcopy(tgui.mw.loaded_file.data.im_array)

        start_time, stop_time = start_stop_samples
        stop_time = tgui.adata.num_samples - 1 if stop_time == "full" else stop_time

        # overwrite time data with indexes and ts = 1 so that time and sample is equivalent (quick_get_time_in_samples())
        new_time_array = np.tile(np.arange(tgui.adata.num_samples), (tgui.adata.num_recs, 1))
        tgui.mw.loaded_file.data.time_array = new_time_array
        tgui.mw.loaded_file.data.min_max_time = np.vstack([new_time_array[:, 0], new_time_array[:, -1]]).T
        tgui.mw.loaded_file.data.ts = 1

        # run cut trace length
        tgui.mw.mw.actionCut_Down_Trace_Time.trigger()
        dialog = tgui.mw.dialogs["cut_trace"]

        dialog.dia.new_start_time_spinbox.setMaximum(tgui.adata.num_samples)
        dialog.dia.new_stop_time_spinbox.setMaximum(tgui.adata.num_samples)

        tgui.enter_number_into_spinbox(dialog.dia.new_start_time_spinbox, start_time, setValue=True)
        tgui.enter_number_into_spinbox(dialog.dia.new_stop_time_spinbox, stop_time, setValue=True)

        self.set_cut_trace_length_radiobutton(tgui, dialog, time_method)

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(dialog.dia.cut_trace_buttonbox.button(QtWidgets.QDialogButtonBox.Apply))

        assert np.array_equal(tgui.mw.loaded_file.data.vm_array, saved_vm_array[:, start_time:stop_time])
        assert np.array_equal(tgui.mw.loaded_file.data.im_array, saved_im_array[:, start_time:stop_time])

        if time_method == "normalised":
            assert np.array_equal(
                tgui.mw.loaded_file.data.time_array,
                new_time_array[:, 0 : stop_time - start_time],
            )

        elif time_method == "cumulative":
            stop_time_increasing_array = np.atleast_2d(np.arange(tgui.mw.loaded_file.data.num_recs)).T * (
                stop_time - start_time
            )
            cum_times_array = new_time_array[:, 0 : stop_time - start_time] + stop_time_increasing_array
            assert np.array_equal(tgui.mw.loaded_file.data.time_array, cum_times_array)
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    def test_normalise_time_gui(self, tgui):
        save_fs = copy.deepcopy(tgui.mw.loaded_file.data.fs)
        save_ts = copy.deepcopy(tgui.mw.loaded_file.data.ts)

        tgui.mw.mw.actionNormalise_Timescale.trigger()
        tgui.left_mouse_click(
            tgui.mw.dialogs["normalise_time"].dia.normalise_time_buttonbox.button(QtWidgets.QDialogButtonBox.Apply)
        )

        assert (tgui.mw.loaded_file.data.time_array[0, :] == tgui.mw.loaded_file.data.time_array[:, :]).all()
        assert tgui.mw.loaded_file.data.fs == save_fs
        assert tgui.mw.loaded_file.data.ts == save_ts
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    def test_reset_to_raw_data(self, tgui):
        tgui.mw.loaded_file.raw_data.vm_array = np.random.random((tgui.adata.num_recs, tgui.adata.num_samples))
        tgui.mw.loaded_file.raw_data.im_array = np.random.random((tgui.adata.num_recs, tgui.adata.num_samples))
        tgui.mw.loaded_file.raw_data.time_array = np.random.random((tgui.adata.num_recs, tgui.adata.num_samples))
        tgui.mw.loaded_file.raw_data.fs = np.random.random(1)
        tgui.mw.loaded_file.raw_data.ts = np.random.random(1)

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.mw.mw.actionReset_to_Raw_Data.trigger()

        assert np.array_equal(tgui.mw.loaded_file.raw_data.vm_array, tgui.mw.loaded_file.data.vm_array)
        assert np.array_equal(tgui.mw.loaded_file.raw_data.im_array, tgui.mw.loaded_file.data.im_array)
        assert np.array_equal(tgui.mw.loaded_file.raw_data.time_array, tgui.mw.loaded_file.data.time_array)
        self.check_plot_equals_data(tgui, "current_clamp", one_channel=True)

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Stimulus Artefact Plots
    # --------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("data_type", ["voltage_clamp", "current_clamp"])
    def test_record_arithmatic_two_channels(self, tgui, data_type):
        """
        Test record arithmatic tool, by repeatedly performing record arithmatic operations.
        Enter these operations per-row in the GUI. Then click Apply to look at the results.
        Check the results against performing the operation in python.
        Test both on voltage clamp and current clamp.
        """
        if data_type == "voltage_clamp":
            tgui = self.get_voltage_clamp_tgui(tgui, tgui.time_type)
        tgui.mw.mw.actionRecord_Arithmatic.trigger()

        self.get_primary_data(tgui, data_type)
        self.get_secondary_data(tgui, data_type)

        # First set of record arithmatic --------------------------------------------------------------------------------------------------------

        tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_table.setItem(
            0, 0, QtWidgets.QTableWidgetItem("(r1 + r2) * (r4 / 2) + r5")
        )
        tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_table.setItem(
            0, 1, QtWidgets.QTableWidgetItem("(r5 - r1) / (r3-1000) * 0.1")
        )
        tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_table.setItem(
            0, 2, QtWidgets.QTableWidgetItem("r1 + r2 - r1 - r2")
        )

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(
            tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_buttonbox.button(
                QtWidgets.QDialogButtonBox.Apply
            )
        )

        test_vm = self.get_primary_adata(tgui, data_type)
        np.array_equal(
            self.get_primary_data(tgui, data_type)[0],
            (test_vm[0] + test_vm[1]) * (test_vm[3] / 2) + test_vm[4],
        )
        np.array_equal(
            self.get_primary_data(tgui, data_type)[1],
            (test_vm[4] - test_vm[0]) / (test_vm[2] - 1000) * 0.1,
        )
        np.allclose(
            self.get_primary_data(tgui, data_type)[2],
            np.zeros(test_vm[0].size),
            rtol=0,
            atol=1e-15,
        )

        test_im = self.get_secondary_adata(tgui, data_type)
        np.array_equal(
            self.get_secondary_data(tgui, data_type)[0],
            (test_im[0] + test_im[1]) * (test_im[3] / 2) + test_im[4],
        )
        np.array_equal(
            self.get_secondary_data(tgui, data_type)[1],
            (test_im[4] - test_im[0]) / (test_im[2] - 1000) * 0.1,
        )
        np.allclose(
            self.get_secondary_data(tgui, data_type)[2],
            np.zeros(test_im[0].size),
            rtol=0,
            atol=1e-15,
        )
        self.check_plot_equals_data(tgui, data_type)

        # Second set of record arithmatic --------------------------------------------------------------------------------------------------------

        first_edit_vm_data = copy.deepcopy(self.get_primary_adata(tgui, data_type))
        first_edit_im_data = copy.deepcopy(self.get_secondary_adata(tgui, data_type))
        tgui.mw.mw.actionRecord_Arithmatic.trigger()

        tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_table.setItem(
            0,
            0,
            QtWidgets.QTableWidgetItem("r1**2 + r3**1 - ((r2 + 100)**(-1)) - (r1 + 100)**2"),
        )

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(
            tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_buttonbox.button(
                QtWidgets.QDialogButtonBox.Apply
            )
        )

        test_vm = first_edit_vm_data
        np.array_equal(
            self.get_primary_data(tgui, data_type)[0],
            test_vm[0] ** 2 + test_vm[2] ** 1 - ((test_vm[1] + 100) ** (-1)) - (test_vm[0] + 100) ** 2,
        )

        test_im = first_edit_im_data
        np.array_equal(
            self.get_secondary_data(tgui, data_type)[0],
            test_im[0] ** 2 + test_im[2] ** 1 - ((test_im[1] + 100) ** (-1)) - (test_im[0] + 100) ** 2,
        )
        self.check_plot_equals_data(tgui, data_type)

        # Third set of record arithmatic --------------------------------------------------------------------------------------------------------

        second_edit_vm_data = copy.deepcopy(self.get_primary_data(tgui, data_type))
        second_edit_im_data = copy.deepcopy(self.get_secondary_data(tgui, data_type))
        tgui.mw.mw.actionRecord_Arithmatic.trigger()

        tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_table.setItem(0, 0, QtWidgets.QTableWidgetItem("2"))
        tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_table.setItem(
            0, 1, QtWidgets.QTableWidgetItem("(r1/r1)**2")
        )
        tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_table.setItem(
            0, 2, QtWidgets.QTableWidgetItem("(r1+r1-r1)*2 - r1")
        )

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(
            tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_buttonbox.button(
                QtWidgets.QDialogButtonBox.Apply
            )
        )

        test_vm = second_edit_vm_data
        np.array_equal(self.get_primary_data(tgui, data_type)[0], np.ones(test_vm[0].size) * 2)
        np.array_equal(self.get_primary_data(tgui, data_type)[1], test_vm[0] ** 2 + 1.012312)
        np.allclose(
            self.get_primary_data(tgui, data_type)[2],
            (test_vm[0] + test_vm[0] - test_vm[0]) * 2 - test_vm[0],
        )

        test_im = second_edit_im_data
        np.array_equal(self.get_secondary_data(tgui, data_type)[0], np.ones(test_im[0].size) * 2)
        np.array_equal(self.get_secondary_data(tgui, data_type)[1], test_im[0] ** 2 + 1.012312)
        np.allclose(
            self.get_secondary_data(tgui, data_type)[2],
            (test_im[0] + test_im[0] - test_im[0]) * 2 - test_im[0],
        )
        self.check_plot_equals_data(tgui, data_type)

        if data_type == "voltage_clamp":
            tgui.shutdown()

    @pytest.mark.parametrize("data_type", ["voltage_clamp", "current_clamp"])
    def test_record_arithmatic_one_channel(self, tgui, data_type):
        """
        Perform record arithmatic again, this time on a single record file
        as a sanity check.
        """
        if data_type == "voltage_clamp":
            tgui = self.get_voltage_clamp_tgui(tgui, tgui.time_type, one_channel=True)

        tgui.mw.mw.actionRecord_Arithmatic.trigger()

        self.get_primary_data(tgui, data_type)
        self.get_secondary_data(tgui, data_type)

        tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_table.setItem(
            0, 0, QtWidgets.QTableWidgetItem("r1 * 0 + r1 / 2 - 2**4")
        )
        tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_table.setItem(
            0, 1, QtWidgets.QTableWidgetItem("r1 + r1 / 2 + r1 / 3")
        )

        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(
            tgui.mw.dialogs["record_arithmatic"].dia.record_arithmatic_buttonbox.button(
                QtWidgets.QDialogButtonBox.Apply
            )
        )

        test_vm = self.get_primary_adata(tgui, data_type)
        np.array_equal(
            self.get_primary_data(tgui, data_type)[0],
            test_vm[0] * 8 + test_vm[0] / 2 - 2**4,
        )
        np.array_equal(
            self.get_primary_data(tgui, data_type)[0],
            test_vm[0] + test_vm[0] / 2 + test_vm[0] / 3,
        )

        self.check_plot_equals_data(tgui, data_type, one_channel=True)

        if data_type == "voltage_clamp":
            tgui.shutdown()

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Stimulus Artefact Plots
    # --------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("data_type", ["current_clamp", "voltage_clamp"])
    def test_stimulus_removal_tool(self, tgui, data_type):
        """
        Test the stimulus removal tool by making a defined
        stimulus and checking each parameter change changes the
        result as expected. Caclulate across all recs, changing
        the baseline by 1 each rec just to ensure the recs are
        not all the same.

        Beause the stimulus is defined, it was defined on
        the current clamp data first. When testing voltage clamp, the file
        did not have the correct time, and so all test were broken.
        To remedy this, current clamp time data are copied
        over to voltage clamp before stimulus insertion.
        """
        if data_type == "voltage_clamp":
            current_clamp_data = {
                "vm": copy.deepcopy(tgui.mw.loaded_file.data.vm_array),
                "im": copy.deepcopy(tgui.mw.loaded_file.data.im_array),
                "time": copy.deepcopy(tgui.mw.loaded_file.data.time_array),
                "ts": copy.deepcopy(tgui.mw.loaded_file.data.ts),
            }
            tgui = self.get_voltage_clamp_tgui(tgui, tgui.time_type, one_channel=True)
        else:
            current_clamp_data = None

        for rec in range(5):
            baseline = rec

            tgui.mw.loaded_file.data.vm_array.setflags(write=True)
            tgui.mw.loaded_file.data.im_array.setflags(write=True)

            max_vm, min_vm, max_diff, mid_idx = self.make_stimulus(
                tgui,
                pos_peak=100,
                neg_peak=-50,
                forward_time_s=0.5,
                backward_time_s=0.02,
                rec=rec,
                baseline=baseline,
                current_clamp_data=current_clamp_data,
            )

            tgui.mw.mw.actionRemove_Stim_Artefacts.trigger()
            dialog = tgui.mw.dialogs["stimulus_artefact_removal"]
            tgui.left_mouse_click(dialog.dia.reset_options_pushbutton)
            tgui.mw.update_displayed_rec(rec)

            units = " (mV/ms)" if data_type == "current_clamp" else " (pA/ms)"
            assert dialog.dia.backward_search_threshold_spinbox.suffix() == units
            assert dialog.dia.derivative_threshold_spinbox.suffix() == units

            data = self.get_stim_data(tgui, level="apply", rec=rec, data_type=data_type)

            # Check derivative threshold --------------------------------------------------------
            tgui.enter_number_into_spinbox(dialog.dia.end_of_stimulus_perc_median_spinbox, 1)
            tgui.enter_number_into_spinbox(dialog.dia.derivative_threshold_spinbox, max_diff + 1)
            tgui.left_mouse_click(dialog.dia.preview_button)
            try:
                assert not self.check_stim_deleted(tgui, baseline, "neg")
            except:
                breakpoint()
            tgui.enter_number_into_spinbox(dialog.dia.derivative_threshold_spinbox, max_diff - 1)
            tgui.left_mouse_click(dialog.dia.preview_button)

            assert self.check_stim_deleted(tgui, baseline, "neg")

            # Check absolute threshold --------------------------------------------------------

            # Negative
            tgui.set_combobox(dialog.dia.threshold_method_combobox, 1)
            tgui.set_combobox(dialog.dia.abs_threshold_combobox, 0)
            tgui.enter_number_into_spinbox(dialog.dia.backward_search_period_spinbox, 0.2)

            tgui.enter_number_into_spinbox(dialog.dia.abs_threshold_spinbox, min_vm + 1)
            tgui.left_mouse_click(dialog.dia.preview_button)
            assert self.check_stim_deleted(tgui, baseline, "neg")

            tgui.enter_number_into_spinbox(dialog.dia.abs_threshold_spinbox, min_vm - 1)
            tgui.left_mouse_click(dialog.dia.preview_button)
            assert not self.check_stim_deleted(tgui, baseline, "neg")

            # Positive
            tgui.set_combobox(dialog.dia.abs_threshold_combobox, 1)

            tgui.enter_number_into_spinbox(dialog.dia.abs_threshold_spinbox, max_vm - 1)
            tgui.left_mouse_click(dialog.dia.preview_button)
            assert self.check_stim_deleted(tgui, baseline, "neg")

            tgui.enter_number_into_spinbox(dialog.dia.abs_threshold_spinbox, max_vm + 1)
            tgui.left_mouse_click(dialog.dia.preview_button)
            assert not self.check_stim_deleted(tgui, baseline, "neg")

            # Backwards Threshold --------------------------------------------------------

            tgui.enter_number_into_spinbox(dialog.dia.abs_threshold_spinbox, max_vm - 1)

            backward_deriv = stats.mode(np.abs(np.diff(data[mid_idx : mid_idx + 20])))[0]
            backward_deriv /= tgui.mw.loaded_file.data.ts * 1000

            tgui.enter_number_into_spinbox(dialog.dia.backward_search_threshold_spinbox, backward_deriv + 0.01)
            tgui.left_mouse_click(dialog.dia.preview_button)
            assert not self.check_stim_deleted(tgui, baseline, "pos")

            tgui.enter_number_into_spinbox(dialog.dia.backward_search_threshold_spinbox, backward_deriv - 0.01)
            tgui.left_mouse_click(dialog.dia.preview_button)
            assert self.check_stim_deleted(tgui, baseline, "pos")

            # Backwards Threshold Search Period -------------------------------------------
            # move the backward search only 100 samples from the peak. This will cut the rise of
            # the stimulus, not fully removing it. Check that the peak of the new half-removed
            # stimlus goes as far back as expected.
            N = 100
            time_ = tgui.mw.loaded_file.data.ts * N

            tgui.enter_number_into_spinbox(dialog.dia.backward_search_period_spinbox, time_)
            tgui.left_mouse_click(dialog.dia.preview_button)

            peak_idx = np.argmax(data)
            new_max = data[peak_idx - N - 2]  # -2 not -1 due to rounding issues

            assert new_max == np.max(self.get_stim_data(tgui))

            # Check Perc Rec
            # This is the level that the forward search must reduce to to
            # end the detected stimulus.
            # -------------------------------------------------------------------------------

            tgui.enter_number_into_spinbox(dialog.dia.end_of_stimulus_perc_median_spinbox, 2000)
            tgui.left_mouse_click(dialog.dia.preview_button)

            # Beware - hard coded. First calculate the median from the region
            # that matches the algorithm region (this is not the full data trace but
            # local subsection. Next find the nearest negative point of the stimulus
            # upswing (after 12000 idx) to the percentage cutoff. This should match
            # the preview.
            min_peak_idx = np.argmin(data)
            diff_median_ = np.median(np.abs(np.diff((data[min_peak_idx - 3 :]))))  # made to match window_pad
            median_ = np.median(data[min_peak_idx - 3 :])

            test_perc_median = np.abs(diff_median_ * 20)
            test_new_min_idx = np.where(np.abs(data[8200:] - median_) < test_perc_median)[0]

            test_new_min_idx = np.delete(test_new_min_idx, test_new_min_idx.size - 1)

            test_new_min = np.min(data[8200 + test_new_min_idx + 2])  # +2 not + 1 due to roudning issue
            new_min = np.min(self.get_stim_data(tgui))

            assert test_new_min == new_min

            if rec != 4:
                tgui.mw.dialogs["stimulus_artefact_removal"].close()

        # At the very end, apply to the data and check it matches preview
        preview = self.get_stim_data(tgui)
        QtCore.QTimer.singleShot(2000, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(dialog.dia.apply_button)

        assert np.array_equal(
            preview,
            self.get_stim_data(tgui, level="apply", rec=rec, data_type=data_type),
        )

        self.check_plot_equals_data(tgui, data_type, one_channel=True)

        if data_type == "voltage_clamp":
            tgui.shutdown()

    def make_stimulus(
        self,
        tgui,
        pos_peak,
        neg_peak,
        forward_time_s,
        backward_time_s,
        rec,
        baseline,
        current_clamp_data=None,
    ):
        """"""
        if current_clamp_data is not None:
            tgui.mw.loaded_file.data.vm_array = current_clamp_data["vm"]
            tgui.mw.loaded_file.data.im_array = current_clamp_data["im"]
            tgui.mw.loaded_file.data.time_array = current_clamp_data["time"]
            tgui.mw.loaded_file.data.ts = current_clamp_data[
                "ts"
            ]  # use so same as was designed for initial time params. Can't copy whole data as contas cc info
            tgui.mw.loaded_file.data.fs = 1 / current_clamp_data["ts"]

        tgui.mw.loaded_file.data.vm_array.fill(baseline)

        rise1 = np.linspace(baseline, pos_peak, int(backward_time_s * tgui.mw.loaded_file.data.fs))
        rise2 = np.linspace(neg_peak, baseline, int(forward_time_s * tgui.mw.loaded_file.data.fs))
        stim = np.hstack([rise1, rise2])

        mid_idx = int(np.shape(tgui.mw.loaded_file.data.vm_array)[1] / 2)
        data = tgui.mw.loaded_file.data.vm_array[rec]
        data[mid_idx : mid_idx + stim.size] = stim
        tgui.mw.loaded_file.data.im_array = copy.deepcopy(tgui.mw.loaded_file.data.vm_array)
        tgui.mw.loaded_file.set_data_params(tgui.mw.loaded_file.data)

        ts = tgui.mw.loaded_file.data.ts
        max_diff = np.max(np.abs(np.diff(data / (ts * 1000))))
        tgui.mw.reset_after_data_manipulation()
        tgui.mw.loaded_file_plot.init_axis_scaling(tgui.mw.cfgs.main["displayed_rec"])

        return np.max(data), np.min(data), max_diff, mid_idx

    def check_plot_equals_data(self, tgui, data_type, one_channel=False):
        """
        Generaic test that everything shown on the mainwindow plot equals
        what is in the loaded_data model.
        """
        rec = tgui.mw.cfgs.main["displayed_rec"]

        if data_type == "current_clamp":
            assert np.array_equal(
                tgui.mw.loaded_file.data.vm_array[rec],
                tgui.mw.loaded_file_plot.curve_upper.yData,
            )
            if not one_channel:
                assert np.array_equal(
                    tgui.mw.loaded_file.data.im_array[rec],
                    tgui.mw.loaded_file_plot.curve_lower.yData,
                )
        else:
            assert np.array_equal(
                tgui.mw.loaded_file.data.im_array[rec],
                tgui.mw.loaded_file_plot.curve_upper.yData,
            )
            if not one_channel:
                assert np.array_equal(
                    tgui.mw.loaded_file.data.vm_array[rec],
                    tgui.mw.loaded_file_plot.curve_lower.yData,
                )

        assert np.array_equal(
            tgui.mw.loaded_file.data.time_array[rec],
            tgui.mw.loaded_file_plot.curve_upper.xData,
        )
        if not one_channel:
            assert np.array_equal(
                tgui.mw.loaded_file.data.time_array[rec],
                tgui.mw.loaded_file_plot.curve_lower.xData,
            )

    def get_stim_data(self, tgui, level="preview", rec=None, data_type=None):
        if level == "preview":
            return tgui.mw.loaded_file_plot.stim_artefact_plot.yData
        else:
            if data_type == "voltage_clamp":
                return tgui.mw.loaded_file.data.im_array[rec]
            elif data_type == "current_clamp":
                return tgui.mw.loaded_file.data.vm_array[rec]

    def check_stim_deleted(self, tgui, baseline, pos_or_neg):
        data = self.get_stim_data(tgui, level="preview")
        if pos_or_neg == "pos":
            deleted = not np.any(data > baseline + 2)
        else:
            deleted = not np.any(data < -baseline - 2)

        return deleted

    # --------------------------------------------------------------------------------------------------------------------------------------------
    # Utils
    # --------------------------------------------------------------------------------------------------------------------------------------------

    def get_voltage_clamp_tgui(self, tgui, time_type, one_channel=False):
        tgui.shutdown()

        if one_channel:
            file_type = "events_one_record"
        else:
            file_type = "events_multi_record_cont"

        tgui = GuiTestSetup("artificial_" + file_type)
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = SPEED
        tgui.setup_artificial_data(time_type, analysis_type=file_type)
        tgui.raise_mw_and_give_focus()
        return tgui

    def get_primary_adata(self, tgui, data_type):
        if data_type == "current_clamp":
            return tgui.adata.vm_array
        else:
            return tgui.adata.im_array

    def get_secondary_adata(self, tgui, data_type):
        if data_type == "current_clamp":
            return tgui.adata.im_array
        else:
            return tgui.adata.vm_array

    def get_primary_data(self, tgui, data_type):
        if data_type == "current_clamp":
            return tgui.mw.loaded_file.data.vm_array
        else:
            return tgui.mw.loaded_file.data.im_array

    def get_secondary_data(self, tgui, data_type):
        if data_type == "current_clamp":
            return tgui.mw.loaded_file.data.im_array
        else:
            return tgui.mw.loaded_file.data.vm_array
