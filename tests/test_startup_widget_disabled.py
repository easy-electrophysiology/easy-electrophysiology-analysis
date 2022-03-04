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
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), 'easy_electrophysiology'))
from ..easy_electrophysiology import easy_electrophysiology
MainWindow = easy_electrophysiology.MainWindow
from setup_test_suite import GuiTestSetup


class TestConfigs:
    """
    Check the relevant widgets are properly disabled on startup (before a file is loaded).
    """
    @pytest.fixture(scope="function", autouse=True)
    def tgui(test):
        tgui = GuiTestSetup('artificial')
        tgui.setup_mainwindow(show=True)
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    def test_startup_pages_are_correct(self, tgui):
        """
        Check that the software displayed the correct widgets on startup and change on file loading (e.g. splash screen, plot tab etc).
        """
        assert tgui.mw.mw.mainwindow_tabwidget.currentIndex() == 0
        assert tgui.mw.mw.graphics_stackwidget.currentIndex() == 1
        assert tgui.mw.mw.table_im_opts_stackwidget.currentIndex() == 0
        assert tgui.mw.mw.table_data_to_show_stackwidget.currentIndex() == 0
        assert tgui.mw.mw.apanel_stackwidget.currentIndex() == 0

        assert tgui.mw.mw.current_rec_or_plot_window_stackedwidget.currentIndex() == 0

        tgui.load_a_filetype("voltage_clamp_1_record")

        assert tgui.mw.mw.graphics_stackwidget.currentIndex() == 0
        assert tgui.mw.mw.current_rec_or_plot_window_stackedwidget.currentIndex() == 1

    @staticmethod
    def check_all_startup_widgets(tgui, bool_):
        """
        see test_initial_widgets_turned_off()
        """
        assert tgui.mw.mw.spkcnt_groupbox.isEnabled() is bool_
        assert tgui.mw.mw.ir_groupbox.isEnabled() is bool_
        assert tgui.mw.mw.skinetics_groupbox.isEnabled() is bool_
        assert tgui.mw.mw.curve_fitting_groupbox.isEnabled() is bool_
        assert tgui.mw.mw.events_threshold_groupbox.isEnabled() is bool_
        assert tgui.mw.mw.events_template_groupbox.isEnabled() is bool_

        assert tgui.mw.mw.spkcnt_table_groupbox.isEnabled() is bool_
        assert tgui.mw.mw.ir_table_groupbox.isEnabled() is bool_
        assert tgui.mw.mw.events_table_groupbox.isEnabled() is bool_

        assert tgui.mw.mw.actionFile_Details.isEnabled() is bool_

        assert tgui.mw.mw.actionUpsample.isEnabled() is bool_
        assert tgui.mw.mw.actionDownsample.isEnabled() is bool_
        assert tgui.mw.mw.actionFilter.isEnabled() is bool_
        assert tgui.mw.mw.actionDetrend.isEnabled() is bool_
        assert tgui.mw.mw.actionRemove_Baseline.isEnabled() is bool_

        assert tgui.mw.mw.actionAverage_Records.isEnabled() is bool_
        assert tgui.mw.mw.actionCut_Down_Trace_Time.isEnabled() is bool_
        assert tgui.mw.mw.actionNormalise_Timescale.isEnabled() is bool_
        assert tgui.mw.mw.actionReshape_Records.isEnabled() is bool_
        assert tgui.mw.mw.actionReset_to_Raw_Data.isEnabled() is bool_

        assert tgui.mw.mw.actionCopy_Vm_Plot.isEnabled() is bool_
        assert tgui.mw.mw.actionCopy_Im_Plot.isEnabled() is bool_
        assert tgui.mw.mw.actionSave_All_Records_Vm.isEnabled() is bool_
        assert tgui.mw.mw.actionSave_All_Records_Im.isEnabled() is bool_

        assert tgui.mw.mw.actionGraph_Options.isEnabled() is bool_

    def test_initial_widgets_turned_off(self, tgui):
        """
        Some widgets are disabled if file is not loaded, to avoid calling connected methods and crashing the window.
        Test that these are disabled when EE is started and enabled when a file is loaded.
        See in activate_widgets_on_first_file_loaded() in MainWindow code.
        """
        self.check_all_startup_widgets(tgui, False)

        tgui.test_update_fileinfo()
        tgui.setup_artificial_data('normalised')

        self.check_all_startup_widgets(tgui, True)

    def test_im_round_combobox(self, tgui):
        """
        Check that the Im rounding combobox for spkcnt and RI analysis are enabled  disabled correctl
        """
        assert not tgui.mw.mw.spkcnt_im_opts_combobox.isEnabled()
        assert not tgui.mw.mw.ir_im_opts_combobox.isEnabled()
        assert not tgui.mw.mw.fake_im_round_combobox.isEnabled()

        tgui.test_update_fileinfo()
        tgui.setup_artificial_data('normalised')

        assert not tgui.mw.mw.spkcnt_im_opts_combobox.isEnabled()

        tgui.run_spikecount_analysis()

        assert tgui.mw.mw.spkcnt_im_opts_combobox.isEnabled()
        assert not tgui.mw.mw.fake_im_round_combobox.isEnabled()
        assert not tgui.mw.mw.ir_im_opts_combobox.isEnabled()

        tgui.run_ri_analysis_bounds()

        assert tgui.mw.mw.spkcnt_im_opts_combobox.isEnabled()
        assert not tgui.mw.mw.fake_im_round_combobox.isEnabled()
        assert tgui.mw.mw.ir_im_opts_combobox.isEnabled()

    @pytest.mark.parametrize("analysis_type", ["events_template_matching", "events_thresholding"])
    def test_load_save_events_only_enabled_on_events(self, tgui, analysis_type):
        """
        Check Save Events menu is only disabled when file is loaded and events analysis selected
        """
        assert not tgui.mw.mw.actionSave_Events_Analysis.isEnabled()
        assert not tgui.mw.mw.actionLoad_Events_Analysis.isEnabled()

        tgui.load_a_filetype("voltage_clamp_1_record")

        assert not tgui.mw.mw.actionSave_Events_Analysis.isEnabled()
        assert not tgui.mw.mw.actionLoad_Events_Analysis.isEnabled()

        tgui.set_analysis_type(analysis_type)

        assert tgui.mw.mw.actionSave_Events_Analysis.isEnabled()
        assert tgui.mw.mw.actionLoad_Events_Analysis.isEnabled()

        tgui.set_analysis_type("curve_fitting")

        assert not tgui.mw.mw.actionSave_Events_Analysis.isEnabled()
        assert not tgui.mw.mw.actionLoad_Events_Analysis.isEnabled()
