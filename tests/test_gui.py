from PySide6 import QtWidgets, QtCore, QtGui
from PySide6 import QtTest
from PySide6.QtTest import QTest
import pytest
import sys
import os
import pandas as pd
import numpy as np
import time
import keyboard

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
import easy_electrophysiology.easy_electrophysiology

from easy_electrophysiology.mainwindow.mainwindow import MainWindow
from dialog_menus.file_details_diaclass import FileDetails

mouseClick = QTest.mouseClick  # There must be a better way!
keyPress = QTest.keyPress
keyClick = QTest.keyClick
keyClicks = QTest.keyClicks
from utils import utils
from setup_test_suite import GuiTestSetup

os.environ["PYTEST_QT_API"] = "PySide6"


class TestGui:
    """
    Basic checks to navigation GUI widgets and save / copy data utilities:

    Tests:
    test_current_rec_spinbox_lowerbound/upperbound:    Ensure current rec boundary cannot exceed number of records in file
    check_upper/lowerbound_recs_to_analyse_spinboxes:  Ensure upper/lower bounds on analyise speecific rec spinboxes cannot exceed number of records on file
    test_analysis_option_spkcnt/ri/skinetics_button:   Check that correct panels are displayed when Analysis Options Widgets are clicked

    @pytest.mark.save_or_copy_test
    test copy / save Im and Vm: =                      Check proper data is loaded to clipboard / saved when menubar actions are pressed
                                                       Unfortunately QTest cannot access filedialogs. As such keyboard module is used here but it means
                                                       you canot use the mouse / keyboard during test.

    Loads the setup_test_suite module to init mainwindow and load file. This inherently tests loading file and setup widgets.

    """

    @pytest.fixture(scope="function", autouse=True)
    def tgui(test):
        tgui = GuiTestSetup("wcp")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.test_load_norm_time_file()
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    # Test Current Record Swidgets -----------------------------------------------------------------------------------------------------------------------

    def test_current_rec_spinbox_lowerbound(self, tgui):
        keyClick(tgui.mw.mw.current_rec_spinbox, "0")
        keyClick(tgui.mw.mw.current_rec_spinbox, QtGui.Qt.Key_Enter)
        assert tgui.mw.mw.current_rec_spinbox.value() == 1, " wrong rec press"

    def test_current_rec_spinbox_upperbound(self, tgui):
        tgui.mw.mw.current_rec_spinbox.clear()
        keyClicks(tgui.mw.mw.current_rec_spinbox, str(tgui.num_recs))
        keyClick(tgui.mw.mw.current_rec_spinbox, QtGui.Qt.Key_Enter)
        assert tgui.mw.mw.current_rec_spinbox.value() == tgui.num_recs

    def test_current_rec_spinbox_left_button(self, tgui):
        tgui.repeat_mouse_click(tgui.mw.mw.current_rec_leftbutton, tgui.num_recs * 2, 0.001)
        assert tgui.mw.mw.current_rec_spinbox.value() == 1
        assert tgui.mw.cfgs.main["displayed_rec"] == 0

    def test_current_rec_spinbox_right_button(self, tgui):
        tgui.repeat_mouse_click(tgui.mw.mw.current_rec_rightbutton, tgui.num_recs * 2, 0.001)
        assert tgui.mw.mw.current_rec_spinbox.value() == tgui.num_recs
        assert tgui.mw.cfgs.main["displayed_rec"] == tgui.num_recs - 1

    def test_current_rec_slider_right_button(self, tgui):
        scrollbar = tgui.mw.mw.horizontal_scroll_bar
        tgui.repeat_mouse_click(
            scrollbar,
            tgui.num_recs * 2,
            0.001,
            pos=QtCore.QPoint(scrollbar.width() - 1, 2),
        )
        assert tgui.mw.cfgs.main["displayed_rec"] == tgui.num_recs - 1

    def test_current_rec_slider_left_button(self, tgui):
        scrollbar = tgui.mw.mw.horizontal_scroll_bar
        tgui.repeat_mouse_click(scrollbar, tgui.num_recs * 2, 0.001, pos=QtCore.QPoint(1, 2))
        assert tgui.mw.cfgs.main["displayed_rec"] == 0

    def test_analyse_specific_recs_from_spinbox(self, tgui):
        tgui.set_analyse_specific_recs_rec_from_spinboxes(["0"])
        for spinbox in [
            tgui.mw.mw.spkcnt_spike_recs_from_spinbox,
            tgui.mw.mw.ir_recs_from_spinbox,
            tgui.mw.mw.curve_fitting_recs_from_spinbox,
        ]:
            assert spinbox.value() == 1, str(spinbox)

    def test_analyse_specific_recs_to_spinbox(self, tgui):
        """
        check upper bound holds on rec to spinbox
        """
        tgui.set_analyse_specific_recs_rec_to_spinboxes([num for num in str(tgui.num_recs)])  # set higher
        for spinbox in [
            tgui.mw.mw.spkcnt_spike_recs_to_spinbox,
            tgui.mw.mw.ir_recs_to_spinbox,
            tgui.mw.mw.curve_fitting_recs_to_spinbox,
        ]:
            keyPress(spinbox, QtGui.Qt.Key_Up)
            assert spinbox.value() == tgui.num_recs

    # Check stackwidgets on analysis type change (configs testing in test_configs)
    # ----------------------------------------------------------------------------------------------------------------------

    def test_analysis_option_curve_fitting_button(self, tgui):
        tgui.mw.dialog_manager.open_analysis_options_menu()
        tgui.left_mouse_click(tgui.mw.dialogs["analysis_options"].dia.curve_fitting_button)
        assert tgui.mw.mw.apanel_stackwidget.currentIndex() == 1, "Stackwidget wrong curve fitting"
        assert tgui.mw.mw.table_data_to_show_stackwidget.currentIndex() == 1

    def test_analysis_option_events_template_button(self, tgui):
        tgui.mw.handle_load_file([os.path.join(tgui.test_base_dir, "vc_events_one_record.abf")])
        tgui.mw.dialog_manager.open_analysis_options_menu()
        tgui.left_mouse_click(tgui.mw.dialogs["analysis_options"].dia.events_template_matching_button)
        assert tgui.mw.mw.apanel_stackwidget.currentIndex() == 3, "Stackwidget wrong Events threshold"
        assert tgui.mw.mw.table_data_to_show_stackwidget.currentIndex() == 3

    def test_analysis_option_events_threshold_button(self, tgui):
        tgui.mw.handle_load_file([os.path.join(tgui.test_base_dir, "vc_two_channels.wcp")])
        tgui.mw.dialog_manager.open_analysis_options_menu()
        tgui.left_mouse_click(tgui.mw.dialogs["analysis_options"].dia.events_thresholding_button)
        assert tgui.mw.mw.apanel_stackwidget.currentIndex() == 4, "anal. panel. Events template"
        assert tgui.mw.mw.table_data_to_show_stackwidget.currentIndex() == 3

    def test_analysis_option_spkcnt_button(self, tgui):
        tgui.mw.dialog_manager.open_analysis_options_menu()
        tgui.left_mouse_click(
            tgui.mw.dialogs["analysis_options"].dia.spike_count_button,
        )
        assert tgui.mw.mw.apanel_stackwidget.currentIndex() == 5, "Stackwidget wrong AP"
        assert tgui.mw.mw.table_data_to_show_stackwidget.currentIndex() == 4

    def test_analysis_option_skinetics_button(self, tgui):
        tgui.mw.dialog_manager.open_analysis_options_menu()
        tgui.left_mouse_click(tgui.mw.dialogs["analysis_options"].dia.spike_kinetics_button)
        assert tgui.mw.mw.apanel_stackwidget.currentIndex() == 6, "Stackwidget wrong skinetics"
        assert tgui.mw.mw.table_data_to_show_stackwidget.currentIndex() == 5

    def test_analysis_option_ir_button(self, tgui):
        tgui.mw.dialog_manager.open_analysis_options_menu()
        tgui.left_mouse_click(tgui.mw.dialogs["analysis_options"].dia.inp_resistance_button)
        assert tgui.mw.mw.apanel_stackwidget.currentIndex() == 2, "Stackwidget wrong Ri"
        assert tgui.mw.mw.table_data_to_show_stackwidget.currentIndex() == 2

    def test_analysis_option_clear_button(self, tgui):
        tgui.mw.dialog_manager.open_analysis_options_menu()
        tgui.left_mouse_click(tgui.mw.dialogs["analysis_options"].dia.clear_analysis_panel)
        assert tgui.mw.mw.apanel_stackwidget.currentIndex() == 0, "Stackwidget wrong clear panel"
        assert tgui.mw.mw.table_data_to_show_stackwidget.currentIndex() == 0

    def test_file_details_dialog(self, tgui):
        """
        Test file dialog matches data. Order based on index (stupid).

        Note: tags tested in test_importdata
        """
        file_details_dialog = FileDetails(tgui.mw, tgui.mw.loaded_file)
        filename = file_details_dialog.items[0].split(": ")[1]
        num_recs = file_details_dialog.items[1].split(": ")[1]
        num_samples = file_details_dialog.items[2].split(": ")[1]
        fs = file_details_dialog.items[3].split(": ")[1]
        ts_ms = file_details_dialog.items[4].split(": ")[1]
        start_time = file_details_dialog.items[5].split(": ")[1]
        stop_time = file_details_dialog.items[6].split(": ")[1]

        assert filename == tgui.mw.loaded_file.fileinfo["filename"] + tgui.mw.loaded_file.fileinfo["file_ext"]
        assert float(num_recs) == tgui.mw.loaded_file.data.num_recs
        assert float(num_samples) == tgui.mw.loaded_file.data.num_samples
        assert float(fs) == np.round(tgui.mw.loaded_file.data.fs, 4)
        assert float(ts_ms) == round(tgui.mw.loaded_file.data.ts * 1000, 4)
        assert float(start_time) == round(tgui.mw.loaded_file.data.min_max_time[0][0], 4)
        assert float(stop_time) == round(tgui.mw.loaded_file.data.min_max_time[0][1], 4)

    def test_copy_vm_data(self, tgui):
        tgui.mw.mw.actionCopy_Vm_Plot.trigger()
        data = pd.read_clipboard()
        assert utils.allclose(
            data["Vm"].to_numpy(),
            tgui.mw.loaded_file.data.vm_array[tgui.mw.cfgs.main["displayed_rec"]],
            1e-10,
        )

        assert utils.allclose(
            data["Time(s)"].to_numpy(),
            tgui.mw.loaded_file.data.time_array[tgui.mw.cfgs.main["displayed_rec"]],
            1e-10,
        )

    def test_copy_im_data(self, tgui):
        tgui.mw.mw.actionCopy_Im_Plot.trigger()
        data = pd.read_clipboard()
        assert utils.allclose(
            data["Im"].to_numpy(),
            tgui.mw.loaded_file.data.im_array[tgui.mw.cfgs.main["displayed_rec"]],
            10e-10,
        )
        assert utils.allclose(
            data["Time(s)"].to_numpy(),
            tgui.mw.loaded_file.data.time_array[tgui.mw.cfgs.main["displayed_rec"]],
            10e-10,
        )

    def test_about_dialog(self, tgui):
        tgui.mw.mw.actionInfo.trigger()
        label_text = tgui.mw.dialogs["about_dialog"].dia.about_label.text()

        version = tgui.mw.cfgs.main["version"]
        version_on_label = label_text.split("version ")[1][0 : len(version)]

        assert version == version_on_label, "version is incorrect"
