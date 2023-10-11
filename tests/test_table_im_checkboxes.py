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
from utils import utils

mouseClick = QTest.mouseClick  # There must be a better way!
keyPress = QTest.keyPress
keyClick = QTest.keyClick
from setup_test_suite import GuiTestSetup
from PySide6.QtCore import Signal

SPEED = "slow"


class TestTableCheckboxes:
    """
    Check the correct table checkboxes are enabled / checked depending on the spikecount analysis that was
    carried out.
    """

    @pytest.fixture(scope="function", params=["normalised", "cumulative"], ids=["1", "2"])
    def tgui(test, request):
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = SPEED
        tgui.setup_artificial_data(request.param)  # UPDATE GEENRATOR TO USE A DICT
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def check_spikecount_table_checkboxes_checked(self, tgui, param_bool, spiketimes_bool):
        assert tgui.mw.mw.num_spikes_checkbox.isChecked() is param_bool
        assert tgui.mw.mw.show_im_checkbox.isChecked() is param_bool
        assert tgui.mw.mw.show_rheobase_checkbox.isChecked() is param_bool
        assert tgui.mw.mw.show_fs_latency_checkbox.isChecked() is param_bool
        assert tgui.mw.mw.show_mean_isi_checkbox.isChecked() is param_bool
        assert tgui.mw.mw.show_all_spiketimes_checkbox.isChecked() is spiketimes_bool

    def check_spikecount_table_checkboxes_enabled(self, tgui, param_bool, spiketimes_bool):
        assert tgui.mw.mw.num_spikes_checkbox.isEnabled() is param_bool
        assert tgui.mw.mw.show_im_checkbox.isEnabled() is param_bool
        assert tgui.mw.mw.show_rheobase_checkbox.isEnabled() is param_bool
        assert tgui.mw.mw.show_fs_latency_checkbox.isEnabled() is param_bool
        assert tgui.mw.mw.show_mean_isi_checkbox.isEnabled() is param_bool
        assert tgui.mw.mw.show_all_spiketimes_checkbox.isEnabled() is spiketimes_bool

    def switch_spikecount_table_checkboxes(self, tgui, param_bool, spiketimes_bool=None):
        tgui.switch_checkbox(tgui.mw.mw.num_spikes_checkbox, on=param_bool)
        tgui.switch_checkbox(tgui.mw.mw.show_im_checkbox, on=param_bool)
        tgui.switch_checkbox(tgui.mw.mw.show_rheobase_checkbox, on=param_bool)
        tgui.switch_checkbox(tgui.mw.mw.show_fs_latency_checkbox, on=param_bool)
        tgui.switch_checkbox(tgui.mw.mw.show_mean_isi_checkbox, on=param_bool)
        if spiketimes_bool is not None:
            tgui.switch_checkbox(tgui.mw.mw.show_all_spiketimes_checkbox, on=spiketimes_bool)

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Test spike calcs
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_spkcnt_table_options(self, tgui):  # check that checkboxes are kept the same after analysis wrong
        """ """
        tgui.run_spikecount_analysis(["fs_latency_ms", "mean_isi_ms", "rheobase_exact"], im_setting="bounds")
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=True, spiketimes_bool=False)
        self.switch_spikecount_table_checkboxes(tgui, param_bool=False)
        tgui.run_spikecount_analysis(
            ["fs_latency_ms", "mean_isi_ms", "rheobase_exact"], im_setting="bounds"
        )  # re-run and check settings are maintained
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=False, spiketimes_bool=False)

    def test_spkcnt_table_options_spiketimes_checkbox(self, tgui):
        """ """
        tgui.run_spikecount_analysis(["fs_latency_ms", "mean_isi_ms", "rheobase_exact"], im_setting="bounds")
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=True, spiketimes_bool=False)
        tgui.switch_checkbox(tgui.mw.mw.show_all_spiketimes_checkbox, on=True)
        self.check_spikecount_table_checkboxes_enabled(tgui, param_bool=False, spiketimes_bool=True)
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=True, spiketimes_bool=True)
        tgui.switch_checkbox(tgui.mw.mw.show_all_spiketimes_checkbox, on=False)
        self.check_spikecount_table_checkboxes_enabled(tgui, param_bool=True, spiketimes_bool=True)
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=True, spiketimes_bool=False)
        self.switch_spikecount_table_checkboxes(tgui, param_bool=False)
        tgui.switch_checkbox(tgui.mw.mw.show_all_spiketimes_checkbox, on=True)
        self.check_spikecount_table_checkboxes_enabled(tgui, param_bool=False, spiketimes_bool=True)
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=False, spiketimes_bool=True)

    def test_loading_file_with_checkboxes_batch_mode_off(self, tgui):
        tgui.run_spikecount_analysis(["fs_latency_ms", "mean_isi_ms", "rheobase_exact"], im_setting="bounds")
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=True, spiketimes_bool=False)
        tgui.test_load_norm_time_file()
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=True, spiketimes_bool=False)

    def test_loading_file_with_spiketimmes_checkbox_batch_mode_off(self, tgui):
        tgui.run_spikecount_analysis(["fs_latency_ms", "mean_isi_ms", "rheobase_exact"], im_setting="bounds")
        tgui.switch_checkbox(tgui.mw.mw.show_all_spiketimes_checkbox, on=True)
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=True, spiketimes_bool=True)
        self.check_spikecount_table_checkboxes_enabled(tgui, param_bool=False, spiketimes_bool=True)
        tgui.test_load_norm_time_file()
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=True, spiketimes_bool=True)
        self.check_spikecount_table_checkboxes_enabled(tgui, param_bool=False, spiketimes_bool=True)
        tgui.run_spikecount_analysis(["fs_latency_ms", "mean_isi_ms", "rheobase_exact"], im_setting="bounds")
        self.check_spikecount_table_checkboxes_checked(tgui, param_bool=True, spiketimes_bool=True)
        self.check_spikecount_table_checkboxes_enabled(tgui, param_bool=False, spiketimes_bool=True)

    def test_error_when_no_spikes_detected_and_checkbox_is_spiketimes(self, tgui):
        tgui.run_spikecount_analysis(["fs_latency_ms"])
        QtWidgets.QApplication.processEvents()
        tgui.switch_checkbox(tgui.mw.mw.show_all_spiketimes_checkbox, on=True)
        tgui.mw.cfgs.spkcnt["auto_thr_width"] = 0.01
        tgui.run_spikecount_analysis(["fs_latency_ms"])
