from PySide2 import QtWidgets, QtCore, QtGui
from PySide2 import QtTest
from PySide2.QtTest import QTest
import pytest
import sys
import os
import pandas as pd
import numpy as np
import time
import keyboard
import copy
import logging
import scipy.stats
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology import easy_electrophysiology
try:
    MainWindow = easy_electrophysiology.MainWindow  # this is for running test_artificial_data.py TODO move
except:
    MainWindow = easy_electrophysiology.easy_electrophysiology.MainWindow  # for all other tests

from ephys_data_methods import current_calc, core_analysis_methods
from generate_artificial_data import TestArtificialSkCntData, TestArtificialRiData, TestDataTools, TestArtificialEventData, ArtificialCurveFitting, TestArtificialsKinetics
from utils import utils
import utils_for_testing as test_utils
from sys import platform
mouseClick = QTest.mouseClick
keyPress = QTest.keyPress
keyClick = QTest.keyClick
keyClicks = QTest.keyClicks
import string
import random

class GuiTestSetup:
    def __init__(self, test_filetype):
        """
        Main test class that contains all functions for loading artificial data and functions for interacting with GUI.
        TODO: quite large, can split into functions for interacting with GUI and functions to setup articifial data. Also,
        offloading some of this functionality to the new and improved pytest-qt would be worthwhile.
        """
        if platform == "darwin":
            self.test_base_dir = "/Users/easyelectrophysiology/git-repos/easy_electrophysiology/tests/test_data/importdata_tests"
        else:
            self.test_base_dir = "C:/fMRIData/git-repo/easy_electrophysiology/tests/test_data/importdata_tests"

        self.mw = None
        self.num_recs = None
        self.rec_from_value = 4
        self.rec_to_value = 50
        self.test_filetype = test_filetype
        self.fake_filename = None

        if test_filetype == "cc_two_channel_abf":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "cc_two_channels.wcp")
            self.cumu_time_data_path = os.path.join(self.test_base_dir, "cc_two_channels_cumu.abf")
            self.num_recs = 12
        elif test_filetype == "wcp":
            self.file_ext = ".wcp"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "cc_two_channels.wcp")
            self.cumu_time_data_path = os.path.join(self.test_base_dir, "cc_one_channel.wcp")
        elif test_filetype == "once_rec_test":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "reshape_records_example_data.abf")
        elif test_filetype in ["artificial", "artificial_skinetics"]:
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "cc_two_channels_cumu.abf")  # this is just a placeholder, the data is overwritten with artificial data
        elif test_filetype == "artificial_events_one_record":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "vc_events_one_record.abf")
        elif test_filetype in ["artificial_events_multi_record", "artificial_events_multi_record_norm"]:
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "light_events_per_rec.abf")
        elif test_filetype == "with_time_offset":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "data_with_time_offset.abf")
        elif test_filetype == "four_channel_input":
            self.file_ext = ".wcp"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "four_channel_input.wcp")
        elif test_filetype == "test_tags":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "mels_tagged_file_1709.abf")

        self.qt_buttons = {
            "0": QtGui.Qt.Key_0,
            "1": QtGui.Qt.Key_1,
            "2": QtGui.Qt.Key_2,
            "3": QtGui.Qt.Key_3,
            "4": QtGui.Qt.Key_4,
            "5": QtGui.Qt.Key_5,
            "6": QtGui.Qt.Key_6,
            "7": QtGui.Qt.Key_7,
            "8": QtGui.Qt.Key_8,
            "9": QtGui.Qt.Key_9,
        }

    def shutdown(self):
        """
        You cannot have 2 QApplication instances at once, so need to gracefully shut then down when tearing down tests.
        However, a strange error occurs if a file is loaded more than once, C++ memory error. It is completely
        harmless but makes a huge mess of the test results. As such stop all logging and exceptions
        during tear down.
        """
        logger = logging.getLogger("my-logger")
        logger.propagate = False
        try:
            tmp = sys.excepthook
            sys.excepthook = None
            self.app.shutdown()
            sys.excepthook = tmp
        except:
            pass
        logger.propagate = True

    def load_a_filetype(self, filetype):

        if filetype == "current_clamp":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "cc_one_channel.abf")

        elif filetype == "voltage_clamp_1_record":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "vc_events_one_record.abf")

        elif filetype == "voltage_clamp_multi_record_events":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "high_freq_events_1.abf")

        elif filetype == "voltage_clamp_multi_record":
            self.file_ext = ".wcp"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "vc_two_channels.wcp")

        elif filetype == "current_clamp_cumu":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "cc_two_channels_cumu.abf")

        elif filetype == "cell_5":
            self.file_ext = ".abf"
            self.norm_time_data_path = os.path.join(self.test_base_dir, "cell5-predrug.abf")

        else:
            error()

        self.mw.update_fileinfo(self.norm_time_data_path)
        self.test_load_norm_time_file()

    def setup_artificial_data(self, norm_or_cumu_time, analysis_type="spkcnt", negative_events=True):
        """
        """
        if analysis_type == "spkcnt":
            self.adata = TestArtificialSkCntData()
        elif analysis_type == "Ri":
            self.adata = TestArtificialRiData()
        elif analysis_type == "skinetics":
            self.adata = TestArtificialsKinetics()
        elif analysis_type == "data_tools":
            self.adata = TestDataTools()
        elif analysis_type == "events_one_record":
            self.adata = TestArtificialEventData(num_recs=1, num_samples=1200000, time_stop=120, negative_events=negative_events)
        elif analysis_type in ["events_multi_record", "events_multi_record_norm"]:
            self.mw.cfgs.file_load_options["select_channels_to_load"]["on"] = True  # the initial file only has 1 channel
            self.mw.cfgs.file_load_options["select_channels_to_load"]["channel_1_idx"] = 0
            self.mw.cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"] = None
            self.adata = TestArtificialEventData(num_recs=14, num_samples=800000, time_stop=15, negative_events=negative_events)
        elif "events_multi_record_biexp" in analysis_type:
            self.mw.cfgs.file_load_options["select_channels_to_load"]["on"] = True  # the initial file only has 1 channel DRY ABOVE FIX
            self.mw.cfgs.file_load_options["select_channels_to_load"]["channel_1_idx"] = 0
            self.mw.cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"] = None
            if analysis_type == "events_multi_record_biexp_7500":
                self.adata = TestArtificialEventData(num_recs=14, num_samples=1200000, time_stop=15, event_type="biexp", event_samples=7500)  # need higher sampling to get correct rise on the biexp
            else:
                self.adata = TestArtificialEventData(num_recs=14, num_samples=1200000, time_stop=15, event_type="biexp")  # need higher sampling to get correct rise on the biexp
        elif analysis_type == "curve_fitting":
            self.adata = ArtificialCurveFitting()

        self.load_artificial_file_from_adata(norm_or_cumu_time, analysis_type)

    def load_artificial_file_from_adata(self, norm_or_cumu_time, analysis_type):
        self.mw.load_file(self.norm_time_data_path)

        if norm_or_cumu_time == "normalised":
            self.mw.loaded_file.raw_data.time_array = self.adata.norm_time_array
            self.adata.min_max_time = self.adata.norm_min_max_time
            self.adata.time_array = self.adata.norm_time_array
            self.time_type = "normalised"
            if analysis_type in ["spkcnt", "skinetics", "events_multi_record_norm"]:
                self.adata.peak_times_ = self.adata.peak_times["normalised"]
        elif norm_or_cumu_time == "cumulative":
            self.mw.loaded_file.raw_data.time_array = self.adata.cum_time_array
            self.adata.min_max_time = self.adata.cum_min_max_time
            self.adata.time_array = self.adata.cum_time_array
            self.time_type = "cumulative"
            if analysis_type in ["spkcnt", "skinetics", "events_one_record", "events_multi_record"]:
                self.adata.peak_times_ = self.adata.peak_times["cumulative"]

        # reset and make mw have fkae data
        self.time_type = norm_or_cumu_time
        self.analysis_type = analysis_type
        self.mw.loaded_file.raw_data.vm_array = self.adata.vm_array
        self.mw.loaded_file.raw_data.im_array = self.adata.im_array
        self.mw.loaded_file.set_data_params(self.mw.loaded_file.raw_data)
        self.mw.loaded_file.set_data_params(self.mw.loaded_file.data)
        self.mw.loaded_file.data = copy.deepcopy(self.mw.loaded_file.raw_data)
        self.mw.loaded_file.init_analysis_results_tables()
        self.mw.clear_and_reset_widgets_for_new_file()

    def load_file(self, filetype):
        """
        Quick convenience function to load a file
        """
        if filetype is None:
            self.test_update_fileinfo()
            self.setup_artificial_data("normalised")
            self.setup_file_details()
        else:
            self.load_a_filetype(filetype)

    def update_events_to_varying_amplitude_and_tau(self):
        """
        """
        self.adata.update_with_varying_amplitudes_and_tau()
        self.load_artificial_file_from_adata(self.time_type,
                                             self.analysis_type)

    def update_events_time_to_irregularly_spaced(self):
        self.adata.update_events_time_to_irregularly_spaced()
        self.load_artificial_file_from_adata("cumulative",
                                             "events_multi_record")

    def update_curve_fitting_function(self, vary_coefs, insert_function, norm_or_cumu_time, pos_or_neg="pos"):
        """
        min/max times are updated here and then again in load_artificial. The second is redundant, but we need the first
        updated so that the times are probably set in insert_function_to_data().

        For basic, make a monoexponential to take measure of.
        """
        if norm_or_cumu_time == "normalised":
            self.adata.min_max_time = self.adata.norm_min_max_time
        elif norm_or_cumu_time == "cumulative":
            self.adata.min_max_time = self.adata.cum_min_max_time

        if insert_function == "slope":
            self.adata.update_vm_im_with_slope_injection()
        else:
            self.adata.insert_function_to_data(vary_coefs,
                                               insert_function,
                                               pos_or_neg)

        self.load_artificial_file_from_adata(norm_or_cumu_time,
                                             "curve_fitting")

    # Setup Gui Tests
    # revisit these, static and should reference a variable! not really any reason to randomise this rec_to / rec_from
    def rec_from(self):
        """
        Curve fitting has much fewer recs due to computationally heavy (rec to / from is zero idx)
        """
        if self.analysis_type == "curve_fitting":
            rec_from = 1
        elif "event" in self.analysis_type:
            rec_from = 4
        elif self.analysis_type == "skinetics":
            rec_from = 2
        else:
            rec_from = self.rec_from_value
        return rec_from

    def rec_to(self):
        """
        Curve fitting has much fewer recs due to computationally heavy (rec to / from is zero idx)
        """
        if self.analysis_type == "curve_fitting":
            rec_to = 3
        elif "event" in self.analysis_type:
            rec_to = 10
        elif self.analysis_type == "skinetics":
            rec_to = 7
        else:
            rec_to = self.rec_to_value
        return rec_to

    def setup_mainwindow(self, show, app=None, reset_all_configs=True, dont_freeze_gui_for_test=True):
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        self.app = app
        self.mw = MainWindow(self.app,
                             reset_all_configs=reset_all_configs,
                             dont_freeze_gui_for_test=dont_freeze_gui_for_test)
        if show:
            self.mw.show()

        if platform == "darwin":
            app.setStyle("Fusion")

        QTest.qWaitForWindowExposed(self.mw.mw.mainwindow_tabwidget)

    def test_update_fileinfo(self, norm=True):
        if norm:
            self.mw.update_fileinfo(self.norm_time_data_path)
        else:
            self.mw.update_fileinfo(self.cumu_time_data_path)

        assert self.mw.cfgs.main["base_dir"] == self.test_base_dir
        assert self.mw.cfgs.main["base_file_ext"] == self.file_ext

    def test_load_norm_time_file(self):
        self.mw.load_file(self.norm_time_data_path)
        self.setup_file_details()

    def test_load_cumu_time_file(self):
        self.mw.load_file(self.cumu_time_data_path)
        self.setup_file_details()

    def setup_file_details(self):
        self.num_recs = self.mw.loaded_file.raw_data.num_recs
        if self.mw.loaded_file.raw_data.num_samples > 5000:
            self.test_bl_lr_start_idx = 1
            self.test_bl_lr_stop_idx = 5000
            self.test_bl_lr_start_time = self.mw.loaded_file.data.time_array[self.mw.cfgs.main["displayed_rec"]][self.test_bl_lr_start_idx]
            self.test_bl_lr_stop_time = self.mw.loaded_file.data.time_array[self.mw.cfgs.main["displayed_rec"]][self.test_bl_lr_stop_idx]

    def set_fake_filename(self):
        self.fake_filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        self.mw.loaded_file.fileinfo["filename"] = self.fake_filename

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# GUI Interaction Convenience Methods
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def click_checkbox(self, checkbox):
        """
        convenience function for clicking a checkbox
        """
        mouseClick(checkbox,
                   QtGui.Qt.MouseButton.LeftButton,
                   pos=QtCore.QPoint(2, checkbox.height() / 2))

    def switch_checkbox(self, checkbox, on):
        """
        click a checkbox ensuring it is on / off. This is used rather than direct control of the checkbox with setChecked()
        to fully replicate. Not certain this is actaully necessary as might be sufficient to check directly, but this way fully
        replicates user use.
        """
        checkbox.blockSignals(True)
        if on:
            checkbox.setChecked(False)  # very similar to directly below can combine but would add another level indirection
        else:
            checkbox.setChecked(True)
        checkbox.blockSignals(False)
        self.click_checkbox(checkbox)

    def switch_groupbox(self, groupbox, on):
        """
        switch_checkbox for groupboxes
        """
        groupbox.blockSignals(True)
        if on:
            groupbox.setChecked(False)
        else:
            groupbox.setChecked(True)
        groupbox.blockSignals(False)
        keyClick(groupbox, QtGui.Qt.Key_Space)

    def repeat_mouse_click(self, widget, n_clicks, delay, pos=None):  # remove update plot, could be the culprit of an issue but doubt it
        for i in range(n_clicks):
            if pos:
                mouseClick(widget, QtGui.Qt.MouseButton.LeftButton, pos=pos)
            else:
                mouseClick(widget, QtGui.Qt.MouseButton.LeftButton)
            QtWidgets.QApplication.processEvents()
            QTest.qWait(delay)
           # time.sleep(delay)

    def repeat_key_click(self, widget, key, n_clicks, delay):
        for i in range(n_clicks):
            keyClick(widget, key)
            QtWidgets.QApplication.processEvents()
            QTest.qWait(delay)
            #time.sleep(delay)

    def single_shot_keyboard_sequence_to_save_csv(self):
        QtCore.QTimer.singleShot(1000, lambda: keyboard.press("tab"))
        QtCore.QTimer.singleShot(1100, lambda: keyboard.press("down"))
        QtCore.QTimer.singleShot(1200, lambda: keyboard.press("down"))
        QtCore.QTimer.singleShot(1230, lambda: keyboard.press("tab"))
        QtCore.QTimer.singleShot(1240, lambda: keyboard.press("enter"))
        QtCore.QTimer.singleShot(1800, lambda: keyboard.press("left"))
        QtCore.QTimer.singleShot(1900, lambda: keyboard.press("enter"))

    def save_mainwindow_table_excel(self):
        """
        Save the table then re-load it to check if it matches
        """
        self.mw.mw.mainwindow_tabwidget.setCurrentIndex(1)
        os.chdir(self.test_base_dir)
        if os.path.isfile("test_save_analysis_excel.xlsx"):
            os.remove("test_save_analysis_excel.xlsx")

        QtCore.QTimer.singleShot(1000, lambda: keyboard.write("test_save_analysis" + "_excel.xlsx"))
        QtCore.QTimer.singleShot(1200, lambda: keyboard.press("enter"))
        QtCore.QTimer.singleShot(1310, lambda: keyboard.press("left"))
        QtCore.QTimer.singleShot(1420, lambda: keyboard.press("enter"))
        self.left_mouse_click(self.mw.mw.save_table_button)
        QTest.qWait(5000)

        data = pd.read_excel(self.test_base_dir + "/test_save_analysis" + "_excel.xlsx")

        return data

    def save_mainwindow_table_csv(self):
        self.mw.mw.mainwindow_tabwidget.setCurrentIndex(1)
        os.chdir(self.test_base_dir)
        if os.path.isfile("test_save_analysis_csv.csv"):
            os.remove("test_save_analysis_csv.csv")

        self.reset_combobox_to_first_index(self.mw.mw.save_table_combobox)
        keyPress(self.mw.mw.save_table_combobox, QtGui.Qt.Key_Down)
        QtCore.QTimer.singleShot(1000, lambda: keyboard.write("test_save_analysis" + "_csv.csv"))
        QtCore.QTimer.singleShot(1200, lambda: keyboard.press("enter"))
        QtCore.QTimer.singleShot(1310, lambda: keyboard.press("left"))
        QtCore.QTimer.singleShot(1420, lambda: keyboard.press("enter"))
        self.left_mouse_click(self.mw.mw.save_table_button)
        QTest.qWait(5000)
        # time.sleep(5)
        data = pd.read_csv(self.test_base_dir + "/test_save_analysis" + "_csv.csv")
        return data

    def saving_traces_excel(self, Im_or_Vm):
        """
        load and save, inpuit as str.

        What a pain this is, cannot seem to save file in EE and then open here as cause segmentation
        fault, even though the qThread is finished and file closed. After returning from this function,
        data can be reloaded in parent functions (test_save_raw_data.py).
        """
        wd = os.getcwd()
        os.chdir(self.test_base_dir)

        excel_filename = self.test_base_dir + "/test_save_" + Im_or_Vm + "_excel.xlsx"
        if os.path.isfile(excel_filename):
            os.remove(excel_filename)

        QtCore.QTimer.singleShot(1000, lambda: keyboard.write("test_save_" + Im_or_Vm + "_excel.xlsx"))
        QtCore.QTimer.singleShot(1200, lambda: keyboard.press("enter"))
        if platform != "darwin":
            QtCore.QTimer.singleShot(1310, lambda: keyboard.press("left"))
            QtCore.QTimer.singleShot(1420, lambda: keyboard.press("enter"))

        self.mw.save_all_records(Im_or_Vm)

        self.wait_for_other_thread(120)   # need a lot of time to drop the link to the saved file even though Qthread closed.

        return excel_filename

    def wait_for_other_thread(self, time_s):
        t = time.time()
        while time.time() < t + time_s:
            QtWidgets.QApplication.processEvents()

    def saving_traces_csv(self, Im_or_Vm):
        """
        Save and re-load data by .xlsx or csv and check it is the same as it was previously in case
        something goes wrong in the saving process.

        For mac, tabbing through to change .xlsx to .csv by keystroke didn't work and I could not find a replacement.
        So copy and paste here and run the core function. This is not ideal but because the overall function is tested
        for excel it should be okay.
        """
        wd = os.getcwd()
        os.chdir(self.test_base_dir)
        csv_filename = self.test_base_dir + "/test_save_" + Im_or_Vm + "_csv.csv"
        if os.path.isfile(csv_filename):
            os.remove(csv_filename)

        if not platform == "darwin":
            QtCore.QTimer.singleShot(500, lambda: keyboard.write("test_save_" + Im_or_Vm + "_csv.csv"))
            QtCore.QTimer.singleShot(1420, lambda: keyboard.press("enter"))
            QtCore.QTimer.singleShot(1310, lambda: keyboard.press("left"))
            self.single_shot_keyboard_sequence_to_save_csv()
            self.mw.save_all_records(Im_or_Vm)
            self.wait_for_other_thread(120)

        else:
            data = self.mw.loaded_file.data.im_array if Im_or_Vm == "Im" else self.mw.loaded_file.data.vm_array
            all_data_voltage_as_pandas_df = pd.DataFrame(data.transpose())  # Max sheet size is: 1048576, 16384
            all_data_voltage_as_pandas_df.to_csv(csv_filename, index=False, header=False)


        return csv_filename

    def set_recs_to_analyse_spinboxes_checked(self, on=True):
        for groupbox in [self.mw.mw.spkcnt_recs_to_analyse_groupbox, self.mw.mw.skinetics_recs_to_analyse_groupbox,
                         self.mw.mw.ir_recs_to_analyse_groupbox, self.mw.mw.curve_fitting_recs_to_analyse_groupbox,
                         self.mw.mw.events_template_recs_to_analyse_groupbox, self.mw.mw.events_threshold_recs_to_analyse_groupbox]:
            self.switch_groupbox(groupbox, on)

    def enter_number_into_spinbox(self, spinbox, number, setValue=False):
        """
        Enter number into a spinbox. By default, press enter to register the input number
        which will update configs.
        Otherwise, provide the dialog and click on it to change focus and update the cfgs
        in the case that enter is pressing OK by default. It is annoying that is even happening.
        """
        spinbox.clear()
        if setValue:
            spinbox.setValue(float(number))
        else:
            keyClicks(spinbox, str(number))
            keyClick(spinbox, QtGui.Qt.Key_Return)

    def set_analyse_specific_recs_rec_from_spinboxes(self, from_):
        self.set_recs_to_analyse_spinboxes_checked()
        for spinbox in [self.mw.mw.spkcnt_spike_recs_from_spinbox, self.mw.mw.skinetics_recs_from_spinbox,
                        self.mw.mw.ir_recs_from_spinbox, self.mw.mw.curve_fitting_recs_from_spinbox,
                        self.mw.mw.events_template_recs_from_spinbox, self.mw.mw.events_threshold_recs_from_spinbox]:
            spinbox.clear()
            for num in from_:
                keyClick(spinbox, self.qt_buttons[num])
            keyClick(spinbox, QtGui.Qt.Key_Enter)

    def set_analyse_specific_recs_rec_to_spinboxes(self, to):
        """
        Set a spinbox to the number from a list of single digits e.g. to = ["1", "2", "5"]
        will set the spinbox to 125.
        """
        self.set_recs_to_analyse_spinboxes_checked()
        for spinbox in [self.mw.mw.spkcnt_spike_recs_to_spinbox, self.mw.mw.skinetics_recs_to_spinbox,
                        self.mw.mw.ir_recs_to_spinbox, self.mw.mw.curve_fitting_recs_to_spinbox,
                        self.mw.mw.events_template_recs_to_spinbox, self.mw.mw.events_threshold_recs_to_spinbox]:

            spinbox.clear()
            for num in to:
                keyClick(spinbox, self.qt_buttons[num])
            keyClick(spinbox, QtGui.Qt.Key_Enter)

    def get_test_idx_from_time_bounds(self, analysis_cfg, lowerbound, upperbound, rec):
        """
        INPUT example (for lowerplot):
               lowerbound - "lower_bl_lr_lowerbound"
               upperbound - "lower_bl_lr_upperbound"
        """
        start_time, stop_time = current_calc.get_bound_times_in_sample_units([analysis_cfg[lowerbound][rec],
                                                                             analysis_cfg[upperbound][rec]],
                                                                             ["start", "stop"],
                                                                             self.mw.loaded_file.data,
                                                                             self.mw.cfgs.main["displayed_rec"])
        return start_time, stop_time

    def get_analysis_im_and_run_buttons(self, analysis):
        if analysis == "spkcnt":
            set_im_button = self.mw.mw.spkcnt_set_im_button
            run_analysis_button = self.mw.mw.spike_count_button
        elif analysis == "Ri":
            set_im_button = self.mw.mw.ir_set_im_button
            run_analysis_button = self.mw.mw.ir_calc_button
        return set_im_button, run_analysis_button

    def set_analyse_specific_recs(self, rec_from, rec_to):
        self.set_analyse_specific_recs_rec_from_spinboxes(list(str(rec_from + 1)))
        self.set_analyse_specific_recs_rec_to_spinboxes(list(str(rec_to + 1)))
        num_recs = rec_to - rec_from + 1
        return num_recs

    def get_analysis_bounds_object(self, analysis):
        if analysis == "spkcnt":
            bounds = self.mw.spkcnt_bounds
        if analysis == "Ri":
            bounds = self.mw.ir_bounds
        if analysis == "skinetics":
            bounds = self.mw.skinetics_bounds
        return bounds

    def get_analysis_dataframe(self, analysis):
        if analysis == "spkcnt":
            loaded_file_analysis_df = self.mw.loaded_file.spkcnt_data
        elif analysis == "Ri":
            loaded_file_analysis_df = self.mw.loaded_file.ir_data
        return loaded_file_analysis_df

    def left_mouse_click(self, widget, pos=None):
        if pos:
            mouseClick(widget, QtGui.Qt.MouseButton.LeftButton, pos=pos)
        else:
            mouseClick(widget,
                       QtGui.Qt.MouseButton.LeftButton)

    def right_mouse_click(self, widget, pos=None):
        if pos:
            mouseClick(widget, QtGui.Qt.MouseButton.RightButton, pos=pos)
        else:
            mouseClick(widget,
                       QtGui.Qt.MouseButton.RightButton)

    def reset_combobox_to_first_index(self, combobox):
        """
        Necessary to activate "activated" signal in mainwindow.
        Qt very buggy on macos with this interaction
        """
        keyPress(combobox, QtGui.Qt.Key_Up)  # extra for macos
        keyPress(combobox, QtGui.Qt.Key_Up)
        keyPress(combobox, QtGui.Qt.Key_Up)
        keyPress(combobox, QtGui.Qt.Key_Up)
        combobox.setCurrentIndex(0)
        if platform != "darwin":
            keyPress(combobox, QtGui.Qt.Key_Down)
        keyPress(combobox, QtGui.Qt.Key_Up)
        keyPress(combobox, QtGui.Qt.Key_Up)
        keyPress(combobox, QtGui.Qt.Key_Up)

        combobox.activated.emit(0)

    def set_analysis_type(self, analysis_type):
        self.mw.mw.actionSelect_Analysis_Window.trigger()
        if analysis_type == "spkcnt":
            self.left_mouse_click(self.mw.dialogs["analysis_options"].dia.spike_count_button)
        elif analysis_type == "Ri":
            self.left_mouse_click(self.mw.dialogs["analysis_options"].dia.inp_resistance_button)
        elif analysis_type == "skinetics":
            self.left_mouse_click(self.mw.dialogs["analysis_options"].dia.spike_kinetics_button)
        elif analysis_type == "curve_fitting":
            self.left_mouse_click(self.mw.dialogs["analysis_options"].dia.curve_fitting_button)
        elif analysis_type == "events_template_matching":
            self.left_mouse_click(self.mw.dialogs["analysis_options"].dia.events_template_matching_button)
        elif analysis_type == "events_thresholding":
            self.left_mouse_click(self.mw.dialogs["analysis_options"].dia.events_thresholding_button)
        elif analysis_type == "curve_fitting":

            self.left_mouse_click(self.mw.dialogs["analysis_options"].dia.curve_fitting_button)
        else:
            BaseException("Wrong analysis type")
        self.mw.dialogs["analysis_options"].close()
        QtWidgets.QApplication.processEvents()

    def switch_to_spikecounts_and_set_im_combobox(self, spike_bounds_on, im_groupbox_on, im_setting="bounds", more_analysis=False):
        """
        """
        self.set_analysis_type("spkcnt")
        self.reset_combobox_to_first_index(self.mw.mw.spkcnt_im_combobox)
        self.switch_checkbox(self.mw.mw.spkcnt_set_bounds_checkbox,
                             on=spike_bounds_on)
        self.switch_groupbox(self.mw.mw.spkcnt_im_groupbox,
                             on=im_groupbox_on)

        if im_setting == "im_protocol":
            keyPress(self.mw.mw.spkcnt_im_combobox, QtGui.Qt.Key_Down)

        elif im_setting == "user_input_im":
            keyPress(self.mw.mw.spkcnt_im_combobox, QtGui.Qt.Key_Down)
            keyPress(self.mw.mw.spkcnt_im_combobox, QtGui.Qt.Key_Down)

        if more_analysis:
            self.left_mouse_click(self.mw.mw.spkcnt_more_analyses_button)

    def switch_to_input_resistance_and_set_im_combobox(self, im_setting="bounds"):
        self.set_analysis_type("Ri")
        self.reset_combobox_to_first_index(self.mw.mw.ir_im_combobox)
        if im_setting == "im_protocol":
            keyPress(self.mw.mw.ir_im_combobox, QtGui.Qt.Key_Down)
        if im_setting == "user_input_im":
            keyPress(self.mw.mw.ir_im_combobox, QtGui.Qt.Key_Down)
            keyPress(self.mw.mw.ir_im_combobox, QtGui.Qt.Key_Down)

    def switch_to_skinetics_and_set_bound(self, skinetics_bounds_on):
        self.set_analysis_type("skinetics")
        self.switch_checkbox(self.mw.mw.skinetics_set_bounds_checkbox,
                             on=skinetics_bounds_on)

    def set_combobox(self, combobox, idx):
        self.reset_combobox_to_first_index(combobox)

        for i in range(idx):
            keyPress(combobox, QtGui.Qt.Key_Down)

    def fill_user_im_input_widget(self, rows_to_fill_in, analysis_set_im_button, all_numbers_the_same=False):
        QtWidgets.QApplication.processEvents()  # this and sleep very important, allows test to catch up so click activates dialog.
        QTest.qWait(1000)

        self.left_mouse_click(analysis_set_im_button)
        self.mw.dialogs["user_im_entry"].dia.step_table.clear()

        for i in range(rows_to_fill_in):
            num_to_fill = "0" if all_numbers_the_same else str(i)
            num = QtWidgets.QTableWidgetItem(num_to_fill)
            self.mw.dialogs["user_im_entry"].dia.step_table.setItem(i, 0, num)
            QtWidgets.QApplication.processEvents()

        QtCore.QTimer.singleShot(50, lambda: self.mw.messagebox.close())
        keyClick(self.mw.dialogs["user_im_entry"].dia.step_tab_buttonbox.buttons()[0],  # OK button
                 QtGui.Qt.Key_Enter)

    def fill_im_injection_protocol_dialog(self, analysis_set_im_button, start_time, stop_time):
        """
        start_time and stop_time must be str.
        """
        # analysis_set_im_button.setEnabled(True)
        self.left_mouse_click(analysis_set_im_button)
        self.mw.dialogs["im_inj_protocol"].dia.im_injprot_start_spinbox.setValue(float(start_time))
        self.mw.dialogs["im_inj_protocol"].dia.im_injprot_stop_spinbox.setValue(float(stop_time))
        self.left_mouse_click(
                              self.mw.dialogs["im_inj_protocol"].dia.im_injprot_buttonbox.button(QtGui.QDialogButtonBox.Apply))

    def calculate_mean_isi(self):
        spiketimes = self.adata.peak_times[self.time_type]
        test_mean_isi = np.nanmean(np.diff(spiketimes, axis=1), axis=1)
        test_mean_isi[np.isnan(test_mean_isi)] = 0
        test_mean_isi *= 1000  # ms
        return test_mean_isi

    def run_skinetics_analysis(self,
                               spike_detection_method,
                               bounds_vm=False,
                               max_slope=False):
        """
        """
        self.reset_combobox_to_first_index(self.mw.mw.skinetics_thr_combobox)
        self.set_analysis_type("skinetics")

        self.left_mouse_click(self.mw.mw.skinetics_options_button)
        skinetics_dia = self.mw.dialogs["skinetics_options"].dia
        self.enter_number_into_spinbox(skinetics_dia.skinetics_search_region_min, 1)

        if spike_detection_method == "auto_record":
            pass
        elif spike_detection_method == "auto_spike":
            keyPress(self.mw.mw.skinetics_thr_combobox, QtGui.Qt.Key_Down)
            self.mw.cfgs.spkcnt["auto_thr_width"] = 1
        elif spike_detection_method == "manual":
            keyPress(self.mw.mw.skinetics_thr_combobox, QtGui.Qt.Key_Down)
            keyPress(self.mw.mw.skinetics_thr_combobox, QtGui.Qt.Key_Down)
            self.enter_number_into_spinbox(self.mw.mw.skinetics_man_thr_spinbox,
                                           -30)

        if max_slope:
            self.switch_groupbox(skinetics_dia.max_slope_groupbox, on=True)

            self.enter_number_into_spinbox(skinetics_dia.max_slope_num_samples_rise_spinbox,
                                           max_slope["n_samples_rise"])
            self.enter_number_into_spinbox(skinetics_dia.max_slope_num_samples_decay_spinbox,
                                           max_slope["n_samples_decay"])

        if bounds_vm:

            self.switch_checkbox(self.mw.mw.skinetics_set_bounds_checkbox,
                                 on=True)
            if "exp" in bounds_vm.keys():
                for rec in range(self.mw.loaded_file.data.num_recs):
                    self.mw.update_displayed_rec(rec)
                    if self.mw.cfgs.rec_within_analysis_range("skinetics",
                                                              rec):
                        self.mw.skinetics_bounds.bounds["upper_exp_lr"].setRegion((bounds_vm["exp"][0][rec],
                                                                                   bounds_vm["exp"][1][rec]))
        self.left_mouse_click(self.mw.mw.skinetics_auto_count_spikes_button)

    def set_skinetics_ahp_spinboxes(self, skinetics_dia, fahp_start, fahp_stop, mahp_start, mahp_stop):
        """
        skinetics spinbox wont let higher less than lower / lower less than higher,
        so first reset to 0 then set higher first
        """
        self.enter_number_into_spinbox(skinetics_dia.fahp_start, 0)
        self.enter_number_into_spinbox(skinetics_dia.fahp_stop, 1)
        self.enter_number_into_spinbox(skinetics_dia.mahp_start, 0)
        self.enter_number_into_spinbox(skinetics_dia.mahp_stop, 1)

        self.enter_number_into_spinbox(skinetics_dia.fahp_stop, fahp_stop)
        self.enter_number_into_spinbox(skinetics_dia.fahp_start, fahp_start)

        self.enter_number_into_spinbox(skinetics_dia.mahp_stop, mahp_stop)
        self.enter_number_into_spinbox(skinetics_dia.mahp_start, mahp_start)

    def manually_select_spike(self, rec, spike_num, overide_time_and_amplitude=False, rect_size_as_perc=0.001):  # reworked but should work fine for spikecalc

        if overide_time_and_amplitude:
            time_ = overide_time_and_amplitude["time"]
            amplitude = overide_time_and_amplitude["amplitude"]
        else:
            time_ = self.adata.peak_times_[rec][spike_num]
            amplitude = self.adata.all_true_peaks[rec][spike_num]

        x_axis_view = self.mw.loaded_file_plot.upperplot.vb.state["limits"]["xLimits"]
        x_offset = (x_axis_view[1] - x_axis_view[0]) * 0.001

        y_axis_view = self.mw.loaded_file_plot.upperplot.vb.state["limits"]["yLimits"]
        y_offset = (y_axis_view[1] - y_axis_view[0]) * 0.005

        [x_start, delta_x] = [time_ - x_offset, x_offset * 2]
        [y_start, delta_y] = [amplitude - np.abs(y_offset), np.abs(y_offset) * 2]

        ax = QtCore.QRectF(x_start, y_start, delta_x, delta_y)
        self.mw.loaded_file_plot.upperplot.vb.sig_plot_click_event.emit(ax)

    def run_spikecount_analysis(self, analysis_to_run=False, im_setting=False,
                                bounds_vm=False, bounds_im=False,
                                spike_detection_method="auto_record",
                                run_=True):
        """
        Run spikecount with selected parameters. All heled in analysis_to_run
        except for im_setting, vm and im bounds which is specified seperately (kind of)

                       tgui.run_spikecount_analysis(["fs_latency_ms", "bounds"],
                                             bounds_vm=bounds_vm)


        auto_record  auto_spike  manual

        bounds
        im_protocol
        user_input_im
        fs_latency_ms
        mean_isi_ms
        rheobase_record
        rheobase_exact
        spike_fa

        """
        if not analysis_to_run:  # cannot use mutable default argument
            analysis_to_run = []

        bounds = True if "bounds" in analysis_to_run else False

        im_groupbox_on = True if im_setting else False
        self.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=bounds,
                                                       im_setting=im_setting,
                                                       im_groupbox_on=im_groupbox_on)

        self.reset_combobox_to_first_index(self.mw.mw.spikecnt_thr_combobox)
        if spike_detection_method == "auto_record":
            pass
        elif spike_detection_method == "auto_spike":
            self.set_combobox(self.mw.mw.spikecnt_thr_combobox, 1)
            self.mw.mw.actionSpike_Detection_Options.trigger()
            self.enter_number_into_spinbox(self.mw.dialogs["spike_counting_options"].dia.width_spinbox, 2.00)
        elif spike_detection_method == "manual":
            self.set_combobox(self.mw.mw.spikecnt_thr_combobox, 2)
            self.enter_number_into_spinbox(self.mw.mw.spkcnt_man_thr_spinbox, -40)

        # im protocol
        if im_setting == "bounds" and bounds_im:
            for rec in range(self.mw.loaded_file.data.num_recs):
                self.mw.update_displayed_rec(rec)
                if self.mw.cfgs.rec_within_analysis_range("spkcnt",  # only move if bounds are visible otherwise causes problems (and is not realistic)
                                                          rec):
                    if "bl" in bounds_im.keys():
                        self.mw.spkcnt_bounds.bounds["lower_bl_lr"].setRegion((bounds_im["bl"][0][rec],
                                                                               bounds_im["bl"][1][rec]))
                    if "exp" in bounds_im.keys():
                        self.mw.spkcnt_bounds.bounds["lower_exp_lr"].setRegion((bounds_im["exp"][0][rec],
                                                                                bounds_im["exp"][1][rec]))
                    QtWidgets.QApplication.processEvents()

        elif im_setting == "im_protocol":
            self.fill_im_injection_protocol_dialog(self.mw.mw.spkcnt_set_im_button,
                                                   str(bounds_im["start"]),
                                                   str(bounds_im["stop"]))
        elif im_setting == "user_input_im":
            pass

        # vm bounds
        if bounds_vm:
            for rec in range(self.mw.loaded_file.data.num_recs):
                self.mw.update_displayed_rec(rec)
                if self.mw.cfgs.rec_within_analysis_range("spkcnt",
                                                          rec):
                    if "exp" in bounds_vm.keys():
                        self.mw.spkcnt_bounds.bounds["upper_exp_lr"].setRegion((bounds_vm["exp"][0][rec],
                                                                                bounds_vm["exp"][1][rec]))
                        QtWidgets.QApplication.processEvents()

        # more analysis
        self.left_mouse_click(self.mw.mw.spkcnt_more_analyses_button)

        if "fs_latency_ms" in analysis_to_run:
            self.switch_checkbox(self.mw.spkcnt_popup.dia.fs_latency_checkbox, on=True)
            apply_im_setting_button = self.mw.dialogs["im_inj_protocol"].dia.im_injprot_buttonbox.button(QtWidgets.QDialogButtonBox.Apply)
            self.left_mouse_click(apply_im_setting_button)

        if "mean_isi_ms" in analysis_to_run:
            self.switch_checkbox(self.mw.mw.spkcnt_more_analyses_button, on=True)
            self.switch_checkbox(self.mw.spkcnt_popup.dia.mean_isi_checkbox, on=True)
            self.switch_checkbox(self.mw.mw.spike_count_button, on=True)

        if "spike_fa" in analysis_to_run:
            self.switch_checkbox(self.mw.spkcnt_popup.dia.spike_freq_accommodation_checkbox, on=True)

        if "rheobase_record" in analysis_to_run:
            self.switch_groupbox(self.mw.spkcnt_popup.dia.spkcnt_rheobase_groupbox, on=True)
            self.switch_checkbox(self.mw.spkcnt_popup.dia.spkcnt_rheobase_record_radiobutton, on=True)

        elif "rheobase_exact" in analysis_to_run:
            self.switch_groupbox(self.mw.spkcnt_popup.dia.spkcnt_rheobase_groupbox, on=True)
            self.switch_checkbox(self.mw.spkcnt_popup.dia.spkcnt_rheobase_exact_radiobutton, on=True)

        if run_:
            self.left_mouse_click(self.mw.mw.spike_count_button)

    def click_upperplot_spotitem(self, plot, spotitem_idx, doubleclick_to_delete=False):
        spotitem = plot.allChildItems()[1].points()[spotitem_idx]  # use signal rather than click through GUI as tough to map clicks
        plot.sigPointsClicked.emit(None, [spotitem])

        if doubleclick_to_delete:
            plot.sigPointsClicked.emit(None, [spotitem])

    def get_spotitem_color(self, plot, spotitem_idx):
        spot_item_color = plot.allChildItems()[1].points()[spotitem_idx].brush().color().name()
        return spot_item_color

    @staticmethod
    def eq(arg1, arg2):
        return np.array_equal(arg1,
                              arg2,
                              equal_nan=True)

    def get_frequency_data_from_qtable(self, analysis_df_colname, rec_from, rec_to):

        headers = self.mw.cfgs.get_events_frequency_table_col_headers(analysis_df_colname)

        all_data = []
        for header in headers:
            header_items = self.mw.mw.table_tab_tablewidget.findItems(header + "   ", QtGui.Qt.MatchExactly)
            num_recs = rec_to - rec_from + 1
            table_data, __ = self.get_all_item_data_from_qtable(header_items, num_recs,  return_str=True)
            all_data.append(table_data)

        return all_data

    def get_data_from_qtable(self, analysis_df_colname, rec_from, rec_to, analysis_type="spkcnt", return_regions=False):
        """
        """
        if analysis_type in ["spkcnt", "Ri"]:
            __, header = self.mw.cfgs.get_table_col_headers("spkcnt_and_input_resistance", analysis_df_colname)
        elif analysis_type == "skinetics":
            __, header = self.mw.cfgs.get_table_col_headers("skinetics", analysis_df_colname)
        else:
            __, header = self.mw.cfgs.get_table_col_headers(analysis_type,
                                                            analysis_df_colname)
        header_items = self.mw.mw.table_tab_tablewidget.findItems(header + "   ", QtGui.Qt.MatchExactly)

        num_recs = rec_to - rec_from + 1

        table_data, regions = self.get_all_item_data_from_qtable(header_items, num_recs)

        if return_regions:
            return regions
        else:
            return np.array(table_data)

    def get_all_item_data_from_qtable(self, header_items, num_recs, return_str=False):
        """
        return str: Will always try and convert item from str to float. return None if item cannot be converted to float if False, otherwise string
        """
        regions = []
        for header_item in header_items:
            col_idx = header_item.column()
            table_data = []

            for i in range(2, num_recs + 2):  # account for first 2 rows are cell name and title

                if not self.mw.mw.table_tab_tablewidget.item(i, col_idx):
                    continue

                table_cell_data = self.mw.mw.table_tab_tablewidget.item(i, col_idx).data(0)
                if table_cell_data.strip() == "n/a":
                    table_cell_data = 0
                else:
                    try:
                        table_cell_data = float(self.mw.mw.table_tab_tablewidget.item(i, col_idx).data(0))  # int() catch rheobase text etc
                    except:
                        if return_str:
                            table_cell_data = self.mw.mw.table_tab_tablewidget.item(i, col_idx).data(0)
                        else:
                            table_cell_data = None

                table_data.append(table_cell_data)
            regions.append(table_data)

        return table_data, regions

    def handle_analyse_specific_recs(self, analyse_specific_recs, data=False):
        """
        set_records_and_get_data_for_analyse_specific_recs
        """
        if analyse_specific_recs:
            self.set_analyse_specific_recs(self.rec_from(),
                                           self.rec_to())
            if np.any(data):
                data = self.process_test_data_for_analyse_recs(data, self.rec_from(), self.rec_to())

            return data, self.rec_from(), self.rec_to()
        else:
            return data, 0, self.adata.num_recs - 1

    def process_test_data_for_analyse_recs(self, adata, rec_from, rec_to):
        """
        replace parts of data with nan that are not within analysed record range for comparison
        with analysed data in tests.
        """
        processed_test_data = utils.np_empty_nan(self.adata.num_recs)
        if len(adata) == self.adata.num_recs:
            if adata.ndim == 2:
                processed_test_data = utils.np_empty_nan((self.adata.num_recs,
                                                         np.shape(adata)[1]))
                processed_test_data[rec_from:rec_to + 1, :] = adata[rec_from:rec_to+1, :]
            else:
                processed_test_data[rec_from:rec_to + 1] = adata[rec_from:rec_to+1]
        else:
            processed_test_data[rec_from:rec_to + 1] = adata
        return processed_test_data

    @staticmethod
    def clean(test_data, make_1d=False):
        """
        remove nans, TODO: must be a better way to do this
        """
        clean_data = []
        for data in test_data:
            if np.isnan(data):
                continue
            clean_data.append(data)

        if make_1d:
            clean_data = np.reshape(clean_data, (len(clean_data), 1))

        return np.array(clean_data)

    def switch_mw_tab(self, idx):
        """
        Convenience function change tab while testing (close analysis option
        as it obscures view).
        """
        self.mw.mw.mainwindow_tabwidget.setCurrentIndex(idx)
        if self.mw.dialogs["analysis_options"]:
            self.mw.dialogs["analysis_options"].close()
        QtWidgets.QApplication.processEvents()

    def check_spiketimes(self, tgui, spikeinfo_to_test, test_spike_times, description):
        for i in range(tgui.adata.num_recs):
            if not np.isnan(spikeinfo_to_test[i]).all() and not np.isnan(test_spike_times[i]).all():  # TODO: use equal_nan=True
                assert tgui.eq(spikeinfo_to_test[i], test_spike_times[i]), "record " + str(i) + " spiketimes does not match " + description

    @staticmethod
    def set_out_of_rec_to_nan(rec_from, rec_to, array_size, data):
        test_data = utils.np_empty_nan(array_size)
        test_data[rec_from:rec_to+1, :] = data[rec_from:rec_to+1, :]
        return test_data

    def calculate_percent_isclose(self, array1, array2, tolerance):
        isclose_bool = np.isclose(array1, array2, atol=tolerance, rtol=0)
        percent_close = np.count_nonzero(isclose_bool) / len(array1)
        return percent_close

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Testing Linear Regions across Recs
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def generate_random_boundary_positions(self, tgui, avoid_spikes=False):
        """
        Generate random times to move bounds to.
        If avoid spikes is on, don"t let any bound be within 50 samples of a spike peak because
        this can cause problems in estimating the number of counted spikes
        if a boundary slices a spike in half
        """
        all_start = []
        all_stop = []
        for rec in range(self.mw.loaded_file.data.num_recs):

            while True:
                start_time, stop_time = self.gen_random_times_for_bounds(tgui,
                                                                         rec)
                if avoid_spikes:
                    pad_samples = self.adata.ts * 50
                    peak_times = self.adata.peak_times_[rec]
                    peak_times = peak_times[~np.isnan(peak_times)]
                    if (abs(start_time - peak_times) < pad_samples).any() or \
                            (abs(stop_time - peak_times) < pad_samples).any():
                        continue
                    else:
                        break
                else:
                    break

            all_start.append(start_time)
            all_stop.append(stop_time)

        return all_start, all_stop

    def assign_random_boundary_position_for_every_rec_and_test(self, tgui, bounds, mode):
        """
        mode = "align_across_recs" or "dont_align_across_recs"
        """
        start_stop_times = {}

        for key in ["lower_bl_lr", "lower_exp_lr", "upper_bl_lr", "upper_exp_lr"]:

            start_stop_times[key] = {"all_start": None,
                                     "all_stop": None}

            if ("upper" in key and bounds.bounds[key] in tgui.mw.loaded_file_plot.upperplot.items) or \
                    ("lower" in key and key in bounds.bounds and bounds.bounds[key] in tgui.mw.loaded_file_plot.lowerplot.items):  # check boundary is on plot, curve_fitting_has_no_lower_cfgs

                all_start = []
                all_stop = []
                for rec in range(tgui.mw.loaded_file.data.num_recs):
                    QtWidgets.QApplication.processEvents()
                    tgui.mw.update_displayed_rec(rec)

                    start_time, stop_time = self.gen_random_times_for_bounds(tgui, rec)
                    bounds.bounds[key].setRegion((start_time, stop_time))

                    all_start.append(start_time - tgui.mw.loaded_file.data.min_max_time[rec][0])
                    all_stop.append(stop_time - tgui.mw.loaded_file.data.min_max_time[rec][0])

                    if mode == "align_across_recs":
                        assert all(all_start[rec] == bounds.analysis_cfg[key + "_lowerbound"]), key
                        assert all(all_stop[rec] == bounds.analysis_cfg[key + "_upperbound"]), key

                if mode == "dont_align_across_recs":
                    assert all_start == bounds.analysis_cfg[key + "_lowerbound"], key
                    assert all_stop == bounds.analysis_cfg[key + "_upperbound"], key

                start_stop_times[key]["all_start"] = all_start
                start_stop_times[key]["all_stop"] = all_stop

        return start_stop_times

    @staticmethod
    def gen_random_times_for_bounds(tgui, rec):
        """
        get some random times to generate to position the boundarys. Make sure they are not too near the edge of the record
        or to eachother.
        """
        start_idx, stop_idx = test_utils.random_int_with_minimum_distance(min_val=10,
                                                                          max_val=tgui.mw.loaded_file.data.num_samples - 10,
                                                                          n=2,
                                                                          min_distance=20)

        start_time = tgui.mw.loaded_file.data.time_array[rec][start_idx]
        stop_time = tgui.mw.loaded_file.data.time_array[rec][stop_idx]

        return start_time, stop_time

    def convert_random_boundary_positions_from_time_to_samples(self, tgui, all_start_stop_times):
        """
        """
        # Convert all boundary times to sample indx
        for bounds_key in ["lower_bl_lr", "lower_exp_lr", "upper_bl_lr", "upper_exp_lr"]:
            for start_stop_key in ["all_start", "all_stop"]:
                if all_start_stop_times[bounds_key][start_stop_key] is not None:
                    start_stop_times = all_start_stop_times[bounds_key][start_stop_key]
                    start_stop_idxs = []
                    for rec in range(len(start_stop_times)):
                        start_stop_idx = current_calc.convert_time_to_samples(timepoint=start_stop_times[rec],
                                                                              start_or_stop=start_stop_key.split("_")[1],
                                                                              time_array=tgui.adata.time_array,
                                                                              min_max_time=self.adata.min_max_time,
                                                                              base_rec=rec,
                                                                              add_offset_back=True)

                        start_stop_idxs.append(start_stop_idx)

                    all_start_stop_times[bounds_key][start_stop_key + "_idx"] = start_stop_idxs

        return all_start_stop_times

    def calculate_test_measures_from_boundary_start_stop_indicies(self, all_start_stop_times, boundary_keys, analysis_names, rec_from, rec_to):
        """
        # boundary_keys: ["upper_bl_lr", "upper_exp_lr", "lower_bl_lr", "lower_exp_lr"]
        # ["vm_baseline", "vm_steady_state", "im_baseline", "im_steady_state"]

        assumes Im allways passed (spkcnt and Ri) but Vm / input resistance only used in Ri
        """
        test_results = {}
        for bound, result in zip(boundary_keys,
                                 analysis_names):
            test_results[result] = utils.np_empty_nan(self.adata.num_recs)
            for rec in range(rec_from, rec_to+1):
                data = self.adata.vm_array if result[0:3] == "vm_" else self.adata.im_array
                test_results[result][rec] = np.mean(data[rec][all_start_stop_times[bound]["all_start_idx"][rec]:all_start_stop_times[bound]["all_stop_idx"][rec]])

        test_delta_im_pa = test_results["im_steady_state"] - test_results["im_baseline"]
        test_delta_vm_mv = test_ir = None
        if "vm_baseline" in analysis_names:
            test_delta_vm_mv = test_results["vm_steady_state"] - test_results["vm_baseline"]
            test_ir = scipy.stats.linregress(test_delta_im_pa[rec_from:rec_to+1] / 1000, test_delta_vm_mv[rec_from:rec_to+1])

        return test_results, test_delta_im_pa, test_delta_vm_mv, test_ir

    def set_link_across_recs(self, tgui, mode):
        """
        Test IR link across records
        """
        if mode == "dont_align_across_recs":
            tgui.mw.mw.actionLink_Across_Records_off.trigger()

        elif mode == "align_across_recs":
            tgui.mw.mw.actionLink_Across_Records_on.trigger()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Run artificial events analysis
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def run_threshold_for_artificial_event_data(self, tgui, biexp, overide_biexp_adjust_start_point=False, negative_events=True):
        tgui.set_analysis_type("events_thresholding")
        tgui.left_mouse_click(tgui.mw.mw.events_threshold_analyse_events_button)
        self.set_widgets_for_artificial_event_data(tgui, "threshold", biexp, overide_biexp_adjust_start_point, negative_events)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

    def run_template_for_aritificial_event_data(self, tgui, biexp, overide_biexp_adjust_start_point=False, negative_events=True):
        tgui.set_analysis_type("events_template_matching")
        tgui.left_mouse_click(tgui.mw.mw.events_template_analyse_all_button)
        self.set_widgets_for_artificial_event_data(tgui, "template", biexp, overide_biexp_adjust_start_point, negative_events)
        tgui.left_mouse_click(tgui.mw.dialogs["template_analyse_events"].dia.fit_all_events_button)

    def run_artificial_events_analysis(self, tgui, template_or_threshold, biexp=False, overide_biexp_adjust_start_point=False, negative_events=True):
        if template_or_threshold == "template":
            self.run_template_for_aritificial_event_data(tgui, biexp, overide_biexp_adjust_start_point, negative_events)
        elif template_or_threshold == "threshold":
            self.run_threshold_for_artificial_event_data(tgui, biexp, overide_biexp_adjust_start_point, negative_events)

    def set_widgets_for_artificial_event_data(self, tgui, template_or_threshold, biexp, overide_biexp_adjust_start_point=False, negative_events=True):
        """
        The dialogs are distcinct, fopr the panel widgets although threshold is changed
        these are linked so it will change for templaet also.

        This as such acts as an implicit test for these widgets also, although it will
        tested explicitly elsewhere.
        """
        if template_or_threshold == "template":

            threshold_lower = "-61" if negative_events else "-58"
            tgui.enter_number_into_spinbox(tgui.mw.dialogs["template_analyse_events"].dia.threshold_lower_spinbox,
                                           threshold_lower)
            tgui.enter_number_into_spinbox(tgui.mw.dialogs["template_analyse_events"].dia.detection_threshold_spinbox,
                                           0.20)

            if not negative_events:
                tgui.mw.cfgs.save_direction(1)

        elif template_or_threshold == "threshold":

            if not negative_events:
                tgui.set_combobox(tgui.mw.mw.events_threshold_peak_direction_combobox, 0)

            threshold_lower = "-65" if negative_events else "-58"
            tgui.enter_number_into_spinbox(tgui.mw.dialogs["events_threshold_analyse_events"].dia.threshold_lower_spinbox,
                                           threshold_lower)

        tgui.switch_checkbox(tgui.mw.mw.events_threshold_average_baseline_checkbox,
                             on=False)
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_amplitude_threshold_spinbox,
                                       "1")
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_baseline_search_period_spinbox,
                                       "25")

        if biexp:
            event_time = (tgui.adata.event_samples - 1) * tgui.adata.ts * 1000
            tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_decay_search_period_spinbox, str(event_time))
            tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_baseline_search_period_spinbox,
                                           "5")

            tgui.mw.mw.actionEvents_Analyis_Options.trigger()
            tgui.switch_checkbox(tgui.mw.dialogs["events_analysis_options"].dia.biexp_fit_adjust_start_point_checkbox,
                                 on=(not overide_biexp_adjust_start_point))

        else:
            time_of_monoexp_decay = self.get_time_of_monoexp_decay(tgui)
            tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_decay_search_period_spinbox,  # measure decay exp exactly
                                           str(time_of_monoexp_decay))

    def get_time_of_monoexp_decay(self, tgui):
        """
        see test_events, test_decay_endpoint_method_and_auc_time(). This is used to
        test decay_search_period_ms() matched the length set here.
        """
        return (tgui.adata.decay_samples - 1) * tgui.adata.ts * 1000

    def raise_mw_and_give_focus(self, for_gui_interaction=False):  # TODO: False options depreciated
        """
        """
        if for_gui_interaction:
     #       self.mw.setWindowState((self.mw.windowState() & ~QtCore.Qt.WindowMinimized) | QtCore.Qt.WindowActive)  # (self.mw.windowState() & ~QtCore.Qt.WindowMinimized) |
      #      self.mw.hide()
       #     self.mw.show()
            self.mw.raise_()
        #    self.mw.activateWindow()

    def get_control(self):
        import matplotlib
        from matplotlib import pyplot as plt
        plt.plot()
        plt.show()

    def enter_numbers_into_omit_times_table(self, dialog, list_of_start_stop_times):
        """
        list_of_start_stop = list of start_stop_times e.g. [ [0, 1], [2, 3] ]
        """
        self.left_mouse_click(dialog.dia.omit_time_periods_button)
        table = dialog.omit_time_periods_dialog.dia.omit_times_table

        for i, times in enumerate(list_of_start_stop_times):
            start, stop = times

            table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(start)))  # TODO: factor out filling to new function below
            table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(stop)))

        self.left_mouse_click(
                               dialog.omit_time_periods_dialog.dia.omit_times_buttonbox.button(QtWidgets.QDialogButtonBox.Ok))


    def fill_tablewidget_with_items(self, tablewidget, array_to_fill):

        for i in range(array_to_fill.shape[0]):
            for j in range(array_to_fill.shape[1]):

                item = array_to_fill[i][j]
                cell_data = QtWidgets.QTableWidgetItem(str(item))
                tablewidget.setItem(i, j, cell_data)

    def fill_tablewidget_with_items_1d(self, tablewidget, array_to_fill):

        for i in range(array_to_fill.shape[0]):
            item = array_to_fill[i]
            cell_data = QtWidgets.QTableWidgetItem(str(item))
            tablewidget.setItem(i, 0, cell_data)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Input Resistance
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def run_ri_analysis_bounds(self, set_sag_analysis=False):
        """
        Run input resistance analysis and set im bounds. Bounds are set to incorporate the pre-current injection baseline
        and current injection (see baseline_time().
        If set_sag_anaylsis is true, sag analysis will be set and times set to incorporate sag delta function added in
        artificial data generation.
        """
        self.switch_to_input_resistance_and_set_im_combobox()

        self.mw.mw.actionLink_im_vm_on.trigger()
        self.mw.ir_bounds.bounds["upper_bl_lr"].setRegion([self.baseline_time()[0],
                                                           self.baseline_time()[1]])
        self.mw.ir_bounds.bounds["upper_exp_lr"].setRegion([self.exp_time()[0],
                                                            self.exp_time()[1]])
        if set_sag_analysis:
            self.turn_on_set_sag_analysis()

        self.left_mouse_click(self.mw.mw.ir_calc_button)

    def run_ri_analysis_user_input_im(self, rec_from, rec_to, set_sag_analysis=False):
        """
        """
        self.switch_to_input_resistance_and_set_im_combobox(im_setting="user_input_im")

        # fill in the user-im table with int 0:num_records (simply for convenience) and run analysis
        rows_to_fill_in = rec_to - rec_from + 1
        self.fill_user_im_input_widget(rows_to_fill_in, self.mw.mw.ir_set_im_button)

        if set_sag_analysis:
            self.turn_on_set_sag_analysis()

        self.left_mouse_click(self.mw.mw.ir_calc_button)
        
        return rows_to_fill_in

    def run_input_resistance_im_protocol(self, set_sag_analysis=False):
        self.switch_to_input_resistance_and_set_im_combobox(im_setting="im_protocol")

        self.fill_im_injection_protocol_dialog(self.mw.mw.ir_set_im_button,
                                               str(self.adata.start_time)[0:4],
                                               str(self.adata.stop_time)[0:4])

        if set_sag_analysis:
            self.turn_on_set_sag_analysis()

        self.left_mouse_click(self.mw.mw.ir_calc_button)

    def turn_on_set_sag_analysis(self):
        self.switch_groupbox(self.mw.mw.ir_calculate_sag_groupbox, on=True)
        half_time = (self.adata.time_stop - self.adata.time_start) / 2
        self.mw.mw.ir_sag_hump_start_spinbox.setValue(0)
        self.mw.mw.ir_sag_hump_stop_spinbox.setValue(half_time)

    def baseline_time(self):
        return [0,
                self.adata.start_time * .90]

    def exp_time(self):
        length = self.adata.stop_time - self.adata.start_time
        return [self.adata.start_time + length*0.5,
                self.adata.start_time + length*0.9]

    def get_test_ir(self, test_vm, user_test_im=False, round_im=False):
        """
        Calculate input resistance to test against output (which is calc with scipy)
        Here if no test_im provided, assume it is the same as test_vm because the im and vim injection
        is the same for artificial generated data
        """
        test_vm = self.clean(test_vm, make_1d=True)

        if np.any(user_test_im):
            test_im = self.clean(user_test_im / 1000, make_1d=True)
        else:
            test_im = test_vm / 1000

        if round_im is not False:
            test_im = current_calc.round_im_injection_to_user_stepsize(pd.Series(test_im.squeeze()),
                                                                       round_im / 1000,
                                                                       "increasing")
            test_im = np.atleast_2d(test_im).T

        test_im = np.concatenate((test_im,
                                  np.ones((len(test_im), 1))),
                                 1)
        # calculate OLS
        I = np.linalg.inv
        test_input_resistance = I(test_im.T @ test_im) @ test_im.T @ test_vm
        return test_input_resistance[0]

    def get_artificial_data_ir_parameters(self, rec_from, rec_to):

        test_counted_recs = np.array([rec for rec in range(1, self.adata.num_recs + 1)])

        test_delta_im = test_delta_vm = self.adata.current_injection_amplitude[rec_from:rec_to + 1]
        test_sag_hump = self.adata.sag_hump_peaks[:, 0][rec_from:rec_to + 1]

        results = {

            "test_counted_recs": test_counted_recs,
            "test_delta_im": test_delta_im,
            "test_delta_vm": test_delta_vm,
            "im_baseline": self.adata.im_offset,
            "im_steady_state": self.adata.im_offset + test_delta_im,
            "vm_baseline": self.adata.resting_vm,
            "vm_steady_state": self.adata.resting_vm + test_delta_vm,
            "test_input_resistance": self.get_test_ir(test_delta_vm),
            "test_sag_hump": test_sag_hump,
            "max_sag_hump_deflection": (test_sag_hump + test_delta_vm),
        }

        return results

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Skinetics
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def run_artificial_skinetics_analysis(self, spike_detection_method="auto_record", interp=False, max_cutoff=90,
                                          min_cutoff=10, run_with_bounds=False, override_mahp_fahp_defaults=False, max_slope=False):

        self.setup_configs_for_test_spike_detection(interp, max_cutoff, min_cutoff, override_mahp_fahp_defaults)

        if run_with_bounds:
            bounds_vm = self.get_analyse_across_recs_or_not_boundaries_dict_for_spikes(align_bounds_across_recs=True)
        else:
            bounds_vm = False

        self.run_skinetics_analysis(spike_detection_method, bounds_vm=bounds_vm, max_slope=max_slope)

        return bounds_vm

    def setup_configs_for_test_spike_detection(self, interp, max_cutoff, min_cutoff, override_mahp_fahp_defaults):

        self.mw.mw.actionSpike_Kinetics_Options_2.trigger()

        self.left_mouse_click(self.mw.dialogs["skinetics_options"].dia.first_deriv_cutoff_radiobutton)
        self.enter_number_into_spinbox(self.mw.dialogs["skinetics_options"].dia.first_deriv_cutoff_spinbox,
                                       5)

        self.enter_number_into_spinbox(self.mw.dialogs["skinetics_options"].dia.skinetics_search_region_min, 2)  # the theoretical min is ts * (self.spike_width / 4) = 0.0007

        if override_mahp_fahp_defaults:
            self.set_skinetics_ahp_spinboxes(self.mw.dialogs["skinetics_options"].dia,
                                             0, 3, 0, 3)

        self.switch_checkbox(self.mw.dialogs["skinetics_options"].dia.interp_200khz_checkbox,
                             on=interp)

        self.enter_number_into_spinbox(self.mw.dialogs["skinetics_options"].dia.rise_time_cutoff_low, min_cutoff)
        self.enter_number_into_spinbox(self.mw.dialogs["skinetics_options"].dia.rise_time_cutoff_high, max_cutoff)
        self.enter_number_into_spinbox(self.mw.dialogs["skinetics_options"].dia.decay_time_cutoff_low, min_cutoff)
        self.enter_number_into_spinbox(self.mw.dialogs["skinetics_options"].dia.decay_time_cutoff_high, max_cutoff)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
#
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_saved_spkcnt_to_loaded_file_data(self, saved_data, loaded_file_data, check_im=True):

        assert (saved_data.iloc[1:, 0].to_numpy().astype(float) == loaded_file_data["record_num"]).all(), "record_num"
        assert (saved_data.iloc[1:, 1].to_numpy().astype(float) == loaded_file_data["num_spikes"]).all(), "num_spikes"

        if check_im:
            assert utils.allclose(saved_data.iloc[1:, 2].to_numpy().astype(float), loaded_file_data["im_baseline"], 1e-10), "im_baseline"
            assert utils.allclose(saved_data.iloc[1:, 3].to_numpy().astype(float), loaded_file_data["im_steady_state"], 1e-10), "im_steady_state"
            assert utils.allclose(saved_data.iloc[1:, 4].to_numpy().astype(float), loaded_file_data["im_delta"], 1e-10), "im_delta"

    def test_saved_input_resistance_to_loaded_file_data(self, saved_data, d, rec_from, rec_to):
        """
        """
        assert np.array_equal(saved_data.iloc[1:, 0].to_numpy(dtype=float), d["test_counted_recs"][rec_from:rec_to + 1]), "test_counted_recs"
        assert (saved_data.iloc[1:, 1].to_numpy(dtype=float) == d["im_baseline"]).all(), "im_baseline"
        assert np.array_equal(saved_data.iloc[1:, 2].to_numpy(dtype=float), d["im_steady_state"]), "im_steady_state"
        assert np.array_equal(saved_data.iloc[1:, 3].to_numpy(dtype=float), d["test_delta_im"]), "test_delta_im"
        assert (saved_data.iloc[1:, 4].to_numpy(dtype=float) == d["vm_baseline"]).all(), "vm_baseline"
        assert np.array_equal(saved_data.iloc[1:, 5].to_numpy(dtype=float), d["vm_steady_state"]), "vm_steady_state"
        assert np.array_equal(saved_data.iloc[1:, 6].to_numpy(dtype=float), d["test_delta_vm"]), "test_delta_vm"
        assert np.isclose(float(saved_data.iloc[1,  7]), d["test_input_resistance"], atol=1e-10, rtol=0), "test_input_resistance"
        assert np.array_equal(saved_data.iloc[1:, 8].to_numpy(dtype=float), d["test_sag_hump"]), "test_sag_hump"
        assert utils.allclose(saved_data.iloc[1:, 9].to_numpy(dtype=float), d["test_sag_hump"] / d["max_sag_hump_deflection"], 1e-10), "test_sag_hump"

    def get_skinetics_param_from_skinetics_data(self, skinetics_data, main_key, param_key, rec):
        return np.array([dict_[param_key] for dict_ in utils.flatten_dict(skinetics_data, rec, main_key)])

# Skinetics ------------------------------------------------------------------------------------------------------------------------------------------

    def get_analyse_across_recs_or_not_boundaries_dict_for_spikes(self, align_bounds_across_recs):
        bounds_vm = {"exp": None, "bl": None}

        if not align_bounds_across_recs:
            self.mw.mw.actionLink_Across_Records_off.trigger()

            for key in ["exp", "bl"]:
                time_starts, time_stops = self.generate_random_boundary_positions(self,
                                                                                  avoid_spikes=True)  #

                bounds_vm[key] = [time_starts,
                                  time_stops]
        else:
            bounds_vm["exp"] = [self.get_test_bound("lower"),
                                self.get_test_bound("upper")]

            bounds_vm["bl"] = [self.get_test_bound("lower", exp_or_bl="bl"),
                               self.get_test_bound("upper", exp_or_bl="bl")]

        return bounds_vm

    def get_test_bound(self, bound_type, rec_from=None, rec_to=None, exp_or_bl="exp"):
        """
        explain for cumu.
        """
        bounds = self.adata.generate_bounds() if exp_or_bl == "exp" else self.adata.generate_baseline_bounds()

        if bound_type == "upper":
            upper_bound = bounds[1]  # * self.mw.loaded_file.data.num_recs
            if rec_from is not None and rec_to is not None:
                upper_bound += self.adata.min_max_time[rec_from:rec_to+1, 0]
            else:
                upper_bound += self.adata.min_max_time[:, 0]
            return upper_bound

        if bound_type == "lower":
            lower_bound = bounds[0]
            if rec_from is not None and rec_to is not None:
                lower_bound += self.adata.min_max_time[rec_from:rec_to+1, 0]
            else:
                lower_bound += self.adata.min_max_time[:, 0]
            return lower_bound

    def reshape_skinetics_data_into_table(self, skinetics_data):  # TODO: MESSY NOW !
        """
        Reshape skinetics_data into table for testing. Ignore record and skinetics num as hard to reshape
        """
        skinetics_data_table = np.full([self.calc_num_spikes(skinetics_data), 13],
                                       np.nan)

        for col, (main_key, param_key) in enumerate(zip(["peak", "peak", "amplitude", "thr", "rise_time", "decay_time", "fwhm", "fahp", "mahp", "max_rise", "max_decay"],
                                                        ["time", "vm", "vm", "vm", "rise_time_ms", "decay_time_ms", "fwhm_ms", "value", "value", "max_slope_ms", "max_slope_ms"])):
            i = 0
            for rec in range(len(skinetics_data)):

                rec_data = self.get_skinetics_param_from_skinetics_data(skinetics_data, main_key, param_key, rec)
                num_spikes = len(rec_data)
                skinetics_data_table[i:i + num_spikes, 0] = np.tile(rec + 1, num_spikes)
                skinetics_data_table[i:i + num_spikes, 1] = np.arange(1, num_spikes + 1)

                if not (param_key == "max_slope_ms" and "off" in rec_data):
                    skinetics_data_table[i:i + num_spikes, col + 2] = rec_data

                i += num_spikes

        return skinetics_data_table

    def reshape_events_into_table(self, event_info):
        """
        """
        event_info_table = np.full([self.calc_num_spikes(event_info), 23],
                                    np.nan)

        for col, (main_key, param_key) in enumerate(zip(["peak",         "record_num", "peak", "baseline", "peak", "amplitude", "rise",         "half_width", "decay_perc",   "area_under_curve", "event_period", "max_rise",     "max_decay",    "monoexp_fit", "monoexp_fit", "monoexp_fit", "monoexp_fit", "biexp_fit", "biexp_fit", "biexp_fit", "biexp_fit", "biexp_fit"],
                                                        ["template_num", "rec_idx",    "time", "im",       "im",   "im",        "rise_time_ms", "fwhm_ms",    "decay_time_ms", "im",               "time_ms",     "max_slope_ms", "max_slope_ms", "b0",          "b1",          "tau_ms",      "r2",          "b0",        "b1",        "rise_ms",   "decay_ms",  "r2"])):
            i = 0
            for rec in range(len(event_info)):

                rec_data = self.get_skinetics_param_from_skinetics_data(event_info, main_key, param_key, rec)
                num_spikes = len(rec_data)

                if param_key == "rec_idx":
                    event_info_table[i:i + num_spikes, col + 1] = rec_data + 1

                elif not ((param_key == "max_slope_ms" and "off" in rec_data) or (param_key == "template_num")):
                    event_info_table[i:i + num_spikes, col + 1] = rec_data

                i += num_spikes

        event_info_table[:, 0] = np.arange(1, self.calc_num_spikes(event_info) + 1)  # TODO: FIX UP

        return event_info_table

    def calc_num_spikes(self, skinetics_data):
        num_spikes = 0
        for rec_dict in skinetics_data:
            if rec_dict:
                num_spikes += len(rec_dict.keys())
        return num_spikes

    def get_entire_qtable(self):
        row_num = self.mw.mw.table_tab_tablewidget.rowCount()
        col_num = self.mw.mw.table_tab_tablewidget.columnCount()
        entire_qtable = utils.np_empty_nan((row_num - 2,
                                            col_num))
        for row in range(2, row_num):
            for col in range(col_num):

                try:
                    data = np.float(self.mw.mw.table_tab_tablewidget.item(row, col).text())
                except (AttributeError, ValueError):
                    data = np.nan
                entire_qtable[row - 2, col] = data

        return entire_qtable

    def save_events_analysis(self, tgui):
        """
        """
        full_filepath = os.path.join(tgui.test_base_dir, "test_save_events_json.json")
        if os.path.isfile(full_filepath):
            os.remove(full_filepath)

        tgui.mw.loaded_file.save_event_analysis(full_filepath)

        return full_filepath

    def delete_event_from_gui_and_artificial_data(self, tgui, template_or_threshold):

        dialog = "events_threshold_analyse_events" if template_or_threshold == "threshold" else "template_analyse_events"

        num_to_del = 5 if tgui.adata.num_events() > 10 else 3
        ev_nums_to_del = np.random.choice(np.arange(1, tgui.adata.num_events()), num_to_del, replace=False)

        ev_nums_to_del = np.flip(np.sort(ev_nums_to_del))   # reverse or deleting low events will change number of high events on next loop

        for num in ev_nums_to_del:
            idx = num - 1

            tgui.enter_number_into_spinbox(tgui.mw.dialogs[dialog].dia.individual_event_number_spinbox,
                                           str(num))

            tgui.left_mouse_click(tgui.mw.dialogs[dialog].dia.delete_individual_trace_button)
            QtWidgets.QApplication.processEvents()

            tgui.adata.b1_offsets = np.delete(tgui.adata.b1_offsets, idx)
            tgui.adata.tau_offsets = np.delete(tgui.adata.tau_offsets, idx)
            tgui.adata.area_under_curves = np.delete(tgui.adata.area_under_curves, idx)

            test_ = self.get_rec_and_idx_from_idx(idx, tgui.adata.peak_times[self.time_type])
            tgui.adata.peak_times[self.time_type][test_[0], test_[1]] = np.nan

            QtWidgets.QApplication.processEvents()
        tgui.switch_mw_tab(1)  # need to switch to re-load table after event deletion
        tgui.switch_mw_tab(0)

    def get_rec_and_idx_from_idx(self, idx, peak_times):
        cnt = 0
        for rec in range(peak_times.shape[0]):
            for ev_idx, ev in enumerate(peak_times[rec]):
                if np.isnan(ev):
                    continue
                else:
                    if idx == cnt:
                        return rec, ev_idx
                    else:
                        cnt += 1

    def get_axograph_max_slope_to_test_against(self):
        axograph_first_event_rise_3 = -331.258
        axograph_first_event_decay_3 = 54.8199
        axograph_first_event_rise_5 = -264.902
        axograph_first_event_decay_7 = 33.4598

        return axograph_first_event_rise_3, axograph_first_event_decay_3, axograph_first_event_rise_5, axograph_first_event_decay_7

    def get_spike_selection_info_for_analysis_type(self, tgui, rec, spike_rec_idx):

        if tgui.analysis_type in ["events_multi_record", "events_one_record"]:

            deleted_spike_peak = tgui.adata.all_rec_b1_offsets[rec][spike_rec_idx] * tgui.adata.b1 + tgui.adata.resting_im
            peak_plot = tgui.mw.loaded_file_plot.peak_plot

        elif tgui.analysis_type == "spkcnt":

            deleted_spike_peak = np.max(tgui.adata.cannonical_spike)
            peak_plot = tgui.mw.loaded_file_plot.spkcnt_plot

        return deleted_spike_peak, peak_plot

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Manual Selection
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_manual_spike_selection(self, tgui, filenum, analyse_specific_recs, run_analysis_function, spikes_to_delete_function, test_function):
        """
        Function to test manaul spike (spikecount analysis) or events (event analyis; herby events are referred to as spikes for convenience).
        First spikes are manually deleted, then manually selected. At each stage a test function is run to ensure the analysis is correctly
        updated. This requires updating the EE results (by analysing in EE) and the test results (by manipulating / saving at each stage
        the data in tgui.adata.

        filenum - idx of file analysed

        run_analysis_function - function to run the analysis, typically a holding function that runs tgui conveinnce functions
                                but only accepts tgui as an argument

        spikes_to_delete_function - a dict including spikes to delete, and their record. These should be specified in reverse order
                                    (i.e. highest to lowest) so that deleting earlier spikes does not change the index of the later spikes

        test_function - function to run to test the analysis is correctly updated after spike deletion / manual selection. It is important
                        to switch the mainwindow tab to table so that the table is updated correctly.

        see test_events.test_manual_event_selection()  and test_spikecount_gui_analysis.test_spike_removed_after_analysis() for examples

        """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=analyse_specific_recs)

        spikes_to_delete, rec_from, rec_to = spikes_to_delete_function(tgui, rec_from, rec_to)

        run_analysis_function(tgui)

        test_function(tgui, filenum, rec_from, rec_to)

        peak_times = dict(all={"data": copy.deepcopy(tgui.adata.peak_times)},
                          m_one=[], m_two=[], m_three=[], m_four=[])
        for rec, spike_rec_idx, dict_key in spikes_to_delete:

            tgui.mw.update_displayed_rec(rec)
            deleted_spike_time = tgui.adata.peak_times[tgui.time_type][rec][spike_rec_idx]

            deleted_spike_peak, peak_plot = self.get_spike_selection_info_for_analysis_type(tgui, rec, spike_rec_idx)

            tgui.click_upperplot_spotitem(peak_plot, spike_rec_idx, doubleclick_to_delete=True)

            self.remove_spike_from_adata(tgui, rec, spike_rec_idx, tgui.adata.peak_times[tgui.time_type])

            peak_times[dict_key] = copy.deepcopy(copy.deepcopy(tgui.adata.peak_times))

            test_function(tgui, filenum, rec_from, rec_to)

            peak_times[dict_key]["data"] = copy.deepcopy(copy.deepcopy(tgui.adata.peak_times))
            peak_times[dict_key]["time"] = deleted_spike_time
            peak_times[dict_key]["amplitude"] = deleted_spike_peak
            peak_times[dict_key]["rec"] = rec

        deleted_spike_keys = ["m_four", "m_three", "m_two", "m_one"]
        for idx, deleted_spike_key in enumerate(deleted_spike_keys):
            rec = peak_times[deleted_spike_key]["rec"]
            tgui.mw.update_displayed_rec(rec)

            self.expand_xaxis_around_peak(tgui, peak_times[deleted_spike_key]["time"])

            tgui.manually_select_spike(rec, spike_num=None, overide_time_and_amplitude=peak_times[deleted_spike_key], rect_size_as_perc=0.02)

            level_up_data = "all" if deleted_spike_key == "m_one" else deleted_spike_keys[idx + 1]
            tgui.adata.peak_times = peak_times[level_up_data]["data"]

            test_function(tgui, filenum, rec_from, rec_to)

        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        tgui.setup_artificial_data(tgui.time_type, tgui.analysis_type)

    @staticmethod
    def remove_spike_from_adata(tgui, rec, spike_idx, adata_array):
        """
        """
        adata_array[rec][spike_idx] = np.nan

        cleaned_peak_data = tgui.clean(adata_array[rec])
        new_peak_times = utils.np_empty_nan(tgui.adata.max_num_spikes)

        new_peak_times[0:len(cleaned_peak_data)] = cleaned_peak_data
        adata_array[rec, :] = new_peak_times

    @staticmethod
    def expand_xaxis_around_peak(tgui, peak_time):

        tgui.mw.loaded_file_plot.upperplot.setXRange(peak_time - 1,
                                                     peak_time + 1)

    def setup_max_slope_events(self, tgui, smooth=False, use_baseline_crossing=False):

        self.switch_to_threshold_and_open_analsis_dialog(tgui)

        options_dialog = self.open_analysis_options_dialog(tgui)

        tgui.switch_groupbox(options_dialog.dia.max_slope_groupbox, on=True)
        tgui.enter_number_into_spinbox(options_dialog.dia.max_slope_num_samples_rise_spinbox, 3)
        tgui.enter_number_into_spinbox(options_dialog.dia.max_slope_num_samples_decay_spinbox, 3)

        if smooth:
            tgui.switch_checkbox(options_dialog.dia.max_slope_smooth_checkbox, on=True)
            tgui.enter_number_into_spinbox(options_dialog.dia.max_slope_smooth_spinbox, 2)

        tgui.switch_checkbox(options_dialog.dia.max_slope_use_first_baseline_crossing_checkbox, on=use_baseline_crossing)
        if use_baseline_crossing:
            tgui.set_combobox(options_dialog.dia.decay_endpoint_search_method_combobox, idx=1)

        return options_dialog

    def switch_to_threshold_and_open_analsis_dialog(self, tgui):
        """
        """
        tgui.set_analysis_type("events_thresholding")  # TODO: own function?
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_threshold_amplitude_threshold_spinbox, 1)

        tgui.left_mouse_click(tgui.mw.mw.events_threshold_analyse_events_button)
        dialog = tgui.mw.dialogs["events_threshold_analyse_events"]

        return dialog

    def open_analysis_options_dialog(self, tgui):

        tgui.mw.mw.actionEvents_Analyis_Options.trigger()
        options_dialog = tgui.mw.dialogs["events_analysis_options"]

        return options_dialog
