from sys import platform

if platform != "darwin":
    import mouse  # not supported on macos
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
from pathlib import Path
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from ephys_data_methods import core_analysis_methods, event_analysis_master
from ephys_data_methods_private import curve_fitting_master
from easy_electrophysiology.easy_electrophysiology.easy_electrophysiology import MainWindow
from utils import utils
keyPress = QTest.keyPress
keyClick = QTest.keyClick
keyClicks = QTest.keyClicks
from setup_test_suite import GuiTestSetup
import time
import copy
os.environ["PYTEST_QT_API"] = "pyside2"

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Functions used mainly for this test class but also called from other test classes
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def close_and_reload_for_defaults_check(tgui, save_="", region_name=None, force_current_clamp=False):
    """
    For the selected analysis window, save the setttings to default, close and re-start the software and
    open the window back up again (for checks that options are saved across sessions).
    """
    if save_ == "graph_view_options":
        tgui.left_mouse_click(tgui.mw.dialogs["graph_view_options"].dia.save_as_default_options_pushbutton)
    if save_ == "spkcnt_options":
        tgui.left_mouse_click(tgui.mw.dialogs["spike_counting_options"].dia.save_as_default_options_pushbutton)
    if save_ == "skinetics_options":
        tgui.left_mouse_click(tgui.mw.dialogs["skinetics_options"].dia.save_as_default_options_pushbutton)
    if save_ == "curve_fitting_options":
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.curve_fitting_save_as_default_options_pushbutton)
    if save_ == "curve_fitting_options_event_kinetics":
        tgui.left_mouse_click(
                              tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.buttonBox.button(QtWidgets.QDialogButtonBox.Ok))
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.curve_fitting_save_as_default_options_pushbutton)
    if save_ == "events_generate":
        tgui.left_mouse_click(tgui.mw.dialogs["events_template_generate"].dia.save_as_default_options_pushbutton)
    if save_ == "events_refine":
        tgui.left_mouse_click(tgui.mw.dialogs["events_template_refine"].dia.save_as_default_options_pushbutton)
    if save_ == "events_analyse":
        tgui.left_mouse_click(tgui.mw.dialogs["template_analyse_events"].dia.save_as_default_options_pushbutton)
    if save_ == "events_threshold":
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.save_as_default_options_pushbutton)
    if save_ == "events_template_panel":
        tgui.left_mouse_click(tgui.mw.mw.events_template_dialog_save_as_default)
    if save_ == "events_threshold_panel":
        tgui.left_mouse_click(tgui.mw.mw.events_threshold_panel_save_as_default)
    if save_ == "events_misc_options":
        tgui.left_mouse_click(tgui.mw.dialogs["events_analysis_options"].dia.save_as_default_options_pushbutton)
    if save_ == "file_loading_options":
        tgui.left_mouse_click(tgui.mw.dialogs["force_load_options"].dia.save_as_default_options_pushbutton)
    if save_ == "events_frequency_plot_options":
        tgui.left_mouse_click(tgui.mw.loaded_file.cumulative_frequency_plot_dialog.frequency_plot_options_dialog.dia.save_as_default_options_pushbutton)
    if save_ == "legacy_options":  # saved automatically on close
        tgui.mw.dialogs["legacy_options"].close()
    if save_ == "misc_options":
        tgui.mw.dialogs["misc_options_dialog"].close()
    if save_ == "table_analysis_options":
        tgui.left_mouse_click(tgui.mw.dialogs["analysis_statistics_options_dialog"].dia.save_as_default_options_pushbutton)

    if save_ == "events_frequency_plot_options":
        tgui.shutdown()
        tgui = setup_artificial_event(tgui, reset_all_configs=False)
    else:
        tgui.shutdown()
        to_load = "test_tags" if force_current_clamp else "artificial"
        tgui = GuiTestSetup(to_load)
        tgui.setup_mainwindow(show=True, reset_all_configs=False)
        if not force_current_clamp:
            load_file(tgui)

    if save_ == "graph_view_options":
        tgui.mw.mw.actionGraph_Options.trigger()
    if save_ == "misc_options":
        tgui.mw.mw.actionMisc_Options.trigger()
    if save_ == "spkcnt_options":
        tgui.mw.mw.actionSpike_Detection_Options.trigger()
    if save_ == "skinetics_options":
        tgui.mw.mw.actionSpike_Kinetics_Options_2.trigger()
    if save_ == "curve_fitting_options":
        setup_curve_fitting_options_dialog(tgui)
        curve_fitting_switch_to_region(tgui, region_name)
    if save_ == "curve_fitting_options_event_kinetics":
        setup_curve_fitting_options_dialog(tgui)
        curve_fitting_switch_to_region(tgui, region_name)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.curve_fitting_event_kinetics_options)
    if save_ == "file_loading_options":
        tgui.mw.mw.actionForce_Load_Options.trigger()
    if save_ == "legacy_options":
        tgui.mw.mw.actionLegacy.trigger()
    if save_ == "table_analysis_options":
        tgui.mw.mw.actionTable_Options.trigger()

    return tgui

def setup_curve_fitting_options_dialog(tgui):
    load_file(tgui)
    tgui.set_analysis_type("curve_fitting")
    tgui.left_mouse_click(tgui.mw.mw.curve_fitting_show_dialog_button)

def curve_fitting_switch_to_region(tgui, region_name):

    if region_name == "reg_1":
        pass
    else:
        for __ in range(int(region_name[-1]) - 1):  # e.g. for reg_4 click 3 times
            tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.change_region_right_button)

def load_file(tgui, filetype=None):
    tgui.load_file(filetype)

def setup_artificial_event(tgui, reset_all_configs=True):
    tgui.shutdown()
    tgui = GuiTestSetup("artificial_events_one_record")
    tgui.setup_mainwindow(show=True,
                          reset_all_configs=reset_all_configs)
    tgui.test_update_fileinfo()
    tgui.setup_artificial_data("normalised", analysis_type="events_one_record")
    tgui.mw.raise_()

    return tgui
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class TestConfigs:

    @pytest.fixture(scope="function")
    def tgui(test):
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Helper Functions
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def get_groupbox_to_switch_from_analysis_type(tgui, groupbox_type):
        if groupbox_type == "spkcnt":
            groupbox_to_switch = tgui.mw.mw.spkcnt_recs_to_analyse_groupbox
        elif groupbox_type == "Ri":
            groupbox_to_switch = tgui.mw.mw.ir_recs_to_analyse_groupbox
        elif groupbox_type == "curve_fitting":
            groupbox_to_switch = tgui.mw.mw.curve_fitting_recs_to_analyse_groupbox
        return groupbox_to_switch

    @staticmethod
    def click_in_middle_of_layout_widget(graphics_layout_widget, plot): 
        point_ = QtCore.QPointF(0, 0)
        scene_point = graphics_layout_widget.ci.mapFromItem(plot,
                                                            point_)  # https://stackoverflow.com/questions/52429399/pyside2-how-to-get-mouse-position, https://stackoverflow.com/questions/16138040/how-to-convert-qpointf-to-qpoint-in-pyqt
        x_move = scene_point.x() + graphics_layout_widget.ci.width() / 2
        y_move = scene_point.y() + graphics_layout_widget.ci.height() / 2
        view_point = graphics_layout_widget.mapFromScene(QtCore.QPoint(x_move, y_move))
        glob_point = graphics_layout_widget.mapToGlobal(view_point)
        mouse.move(glob_point.x(), glob_point.y())
        mouse.click(button="left")
        QtWidgets.QApplication.processEvents() 
        QtWidgets.QApplication.processEvents()

    @staticmethod
    def get_thr_method_from_combobox_idx(tgui):
        thr_method_idx = tgui.mw.dialogs["skinetics_options"].dia.skinetics_method_combobox.currentIndex()
        if thr_method_idx == 0:
            thr_method = "first_deriv"
        elif thr_method_idx == 1:
            thr_method = "third_deriv"
        elif thr_method_idx == 2:
            thr_method = "method_I"
        elif thr_method_idx == 3:
            thr_method = "method_II"
        elif thr_method_idx == 4:
            thr_method = "leading_inflection"
        elif thr_method_idx == 5:
            thr_method = "max_curvature"
        return thr_method

    def check_spikecount_threshold_combobox_spinbox_configs(self, tgui, analysis_cfg, analysis_combobox, spinbox):
        """
        Test that configs match spikecount threshold comobox and spinbox for spikecounts and skinetics.
        """
        tgui.set_combobox(analysis_combobox, idx=0)
        tgui.set_combobox(analysis_combobox, idx=0)
        assert analysis_cfg["threshold_type"] == "auto_record"
        tgui.set_combobox(analysis_combobox, idx=1)
        assert analysis_cfg["threshold_type"] == "auto_spike"
        tgui.set_combobox(analysis_combobox, idx=2)
        assert analysis_cfg["threshold_type"] == "manual"
        spinbox.setValue(1.01)
        keyPress(spinbox,
                 QtGui.Qt.Key_Enter)
        assert analysis_cfg["man_thr_value"] == 1.01

    def click_user_im_input_toggle(self, tgui, protocol_type):
        if protocol_type == "increasing":
            tgui.left_mouse_click(tgui.mw.dialogs["user_im_round"].dia.current_protocol_increasing_radiobutton)
        elif protocol_type == "repeating":
            tgui.left_mouse_click(tgui.mw.dialogs["user_im_round"].dia.current_protocol_repeating_radiobutton)
        elif protocol_type == "decreasing":
            tgui.left_mouse_click(tgui.mw.dialogs["user_im_round"].dia.current_protocol_decreasing_radiobutton)

    def check_spike_count_options_dialog(self, tgui, slider, spinbox, cfg_dict_key, fall=False):
        invert = -1 if fall else 1
        slider.setValue(25 * invert)
        assert spinbox.value() == 2.5 * invert
        assert tgui.mw.cfgs.spkcnt[cfg_dict_key] == 2.5 * invert

        spinbox.setValue(5.0 * invert)
        assert slider.value() == 50 * invert
        assert tgui.mw.cfgs.spkcnt[cfg_dict_key] == 5.0 * invert

        return invert

    def check_baseline_spinbox(self, text_, dialog, idx, tgui):
        if text_ == "Auto.":
            assert not dialog.dia.baseline_spinbox.isEnabled(), "auto spinbox"
        elif text_ == "Draw":
            assert dialog.dia.baseline_stackwidget.currentIndex() == 1, "draw spinbox"
        else:
            assert dialog.dia.baseline_spinbox.value() == idx + 1, "linear or curved spinbox"

        show_plot = not dialog.dia.hide_baseline_from_plot_checkbox.isChecked()
        loaded_file_plot = tgui.mw.loaded_file_plot
        if text_ == "Linear":
            assert (dialog.baseline_axline.axline in loaded_file_plot.upperplot.items) == show_plot
        elif text_ == "Curve":
            assert (loaded_file_plot.curved_baseline_plot in loaded_file_plot.upperplot.items) == show_plot

    def check_threshold_lower_spinbox(self, text_, dialog, idx, tgui):
        if text_ == "Draw":
            assert dialog.dia.threshold_lower_stackwidget.currentIndex() == 1, "draw spinbox"
        else:
            assert dialog.dia.threshold_lower_spinbox.value() == idx + 1, "linear or curved spinbox"

        show_plot = not dialog.dia.hide_threshold_lower_from_plot_checkbox.isChecked()
        loaded_file_plot = tgui.mw.loaded_file_plot
        if text_ == "Linear":
            if show_plot:
                assert (dialog.man_thr_axline.axline in loaded_file_plot.upperplot.items) == show_plot
        elif text_ == "Curved":
            assert (loaded_file_plot.curved_threshold_lower_plot in loaded_file_plot.upperplot.items) == show_plot

    def run_test_analysis_panel_clear_after_different_filetype_loaded(self, tgui, filetype_1, filetype_2, button):
        """
        Check that analysis panel defaults to blank when an unallowed analysis type is loaded (e.g. events thresholding
        panel shown and a current clamp file loaded).

        Had to run like this as loading the same file twice was leading to c++ internal memory error
        """
        load_file(tgui, filetype=filetype_1)
        tgui.left_mouse_click(button)
        load_file(tgui, filetype=filetype_2)
        return tgui.mw.mw.apanel_stackwidget.currentIndex() == 0, \
               tgui.mw.mw.table_data_to_show_stackwidget.currentIndex() == 0

    def setup_events_with_dialog(self, tgui, dialog_type, load_file=True):

        if load_file:
            tgui.load_file("voltage_clamp_1_record")

        if "threshold" in dialog_type:
            tgui.set_analysis_type("events_thresholding")
        else:
            tgui.set_analysis_type("events_template_matching")

        if dialog_type == "generate":
            tgui.left_mouse_click(tgui.mw.mw.events_template_generate_button)
            dialog = tgui.mw.dialogs["events_template_generate"]
        elif dialog_type == "refine":
            tgui.left_mouse_click(tgui.mw.mw.events_template_refine_button)
            dialog = tgui.mw.dialogs["events_template_refine"]
        elif dialog_type == "analyse":
            tgui.left_mouse_click(tgui.mw.mw.events_template_analyse_all_button)
            dialog = tgui.mw.dialogs["template_analyse_events"]
        elif dialog_type == "threshold":
            tgui.left_mouse_click(tgui.mw.mw.events_threshold_analyse_events_button)
            dialog = tgui.mw.dialogs["events_threshold_analyse_events"]
        elif dialog_type == "misc_options":
            tgui.mw.mw.actionEvents_Analyis_Options.trigger()
            dialog = tgui.mw.dialogs["events_analysis_options"]
        elif dialog_type == "frequency_data_options":
            tgui.mw.mw.actionEvents_Analyis_Options.trigger()
            tgui.left_mouse_click(tgui.mw.dialogs["events_analysis_options"].dia.frequency_data_more_options_button)
            dialog = tgui.mw.dialogs["events_analysis_options"].events_frequency_data_options_dialog
        else:
            dialog = None

        config_dict = tgui.mw.cfgs.events
        return dialog, config_dict

    def get_widget(self, t, template_or_threshold, dict_key):

        mw = t.mw.mw

        widgets = {

            "template": {
                "decay_search_period_s": mw.events_template_decay_search_period_spinbox,
                "amplitude_threshold": mw.events_template_amplitude_threshold_spinbox,
                "average_peak_points_on": mw.events_template_average_peak_checkbox,
                "average_peak_points_value_s": mw.events_template_average_peak_spinbox,
                "area_under_curve_on": mw.events_template_area_under_curve_checkbox,
                "area_under_curve_value_pa_ms": mw.events_template_area_under_curve_spinbox,
                "baseline_search_period_s": mw.events_template_baseline_search_period_spinbox,
                "average_baseline_points_on": mw.events_template_average_baseline_checkbox,
                "average_baseline_points_value_s": mw.events_template_average_baseline_spinbox,
            },
            "threshold": {
                "decay_search_period_s": mw.events_threshold_decay_search_period_spinbox,
                "amplitude_threshold": mw.events_threshold_amplitude_threshold_spinbox,
                "average_peak_points_on": mw.events_threshold_average_peak_checkbox,
                "average_peak_points_value_s": mw.events_threshold_average_peak_spinbox,
                "area_under_curve_on": mw.events_threshold_area_under_curve_checkbox,
                "area_under_curve_value_pa_ms": mw.events_threshold_area_under_curve_spinbox,
                "baseline_search_period_s": mw.events_threshold_baseline_search_period_spinbox,
                "average_baseline_points_on": mw.events_threshold_average_baseline_checkbox,
                "average_baseline_points_value_s": mw.events_threshold_average_baseline_spinbox,
            }
        }

        return widgets[template_or_threshold][dict_key]

    def misc_event_widgets(self, dialog, key_1, key_2):

        widgets = {
            "monoexp": {"fit_exclude_r2_checkbox": dialog.dia.monoexp_fit_exclude_r2_checkbox,
                        "fit_exclude_r2_spinbox": dialog.dia.monoexp_fit_exclude_r2_spinbox,
                        "fit_adjust_start_point_checkbox": dialog.dia.monoexp_fit_adjust_start_point_checkbox,
                        "fit_adjust_start_point_spinbox": dialog.dia.monoexp_fit_adjust_start_point_spinbox,
                        "adjust_start_point_for_bounds_checkbox": dialog.dia.monoexp_adjust_start_point_for_bounds_checkbox,
                        "adjust_start_point_for_bounds_spinbox": dialog.dia.monoexp_adjust_start_point_for_bounds_spinbox,
                        "exclude_outside_of_bounds_checkbox": dialog.dia.monoexp_exclude_outside_of_bounds_checkbox,
                        "min_tau_spinbox": dialog.dia.monoexp_min_tau_spinbox,
                        "max_tau_spinbox": dialog.dia.monoexp_max_tau_spinbox,
                        },

            "biexp": {"fit_exclude_r2_checkbox": dialog.dia.biexp_fit_exclude_r2_checkbox,
                      "fit_exclude_r2_spinbox": dialog.dia.biexp_fit_exclude_r2_spinbox,
                      "fit_adjust_start_point_checkbox": dialog.dia.biexp_fit_adjust_start_point_checkbox,
                      "fit_adjust_start_point_spinbox": dialog.dia.biexp_fit_adjust_start_point_spinbox,
                      "adjust_start_point_for_bounds_checkbox": dialog.dia.biexp_adjust_start_point_for_bounds_checkbox,
                      "adjust_start_point_for_bounds_spinbox": dialog.dia.biexp_adjust_start_point_for_bounds_spinbox,
                      "exclude_outside_of_bounds_checkbox": dialog.dia.biexp_exclude_outside_of_bounds_checkbox,
                      },
        }
        return widgets[key_1][key_2]

    def setup_and_show_frequency_plot_options(self, tgui, reload_tgui=True, reset_all_configs=True):

        if reload_tgui:
            tgui = setup_artificial_event(tgui, reset_all_configs)

        tgui.run_threshold_for_artificial_event_data(tgui, biexp=False)
        tgui.mw.loaded_file.plot_cum_frequency()
        tgui.left_mouse_click(tgui.mw.loaded_file.cumulative_frequency_plot_dialog.dia.plot_options_button)
        dialog = tgui.mw.loaded_file.cumulative_frequency_plot_dialog.frequency_plot_options_dialog
        config_dict = tgui.mw.cfgs.dialog_plot_options["events_frequency_plots"]

        return tgui, dialog, config_dict

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test On File Load / Widgets Enabled
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_correct_widgets_are_disabled_before_file_load(self, tgui):
        """
        Some widgets are disabled if file is not loaded, to avoid calling connected methods and crashing the window.
        Test that these are disabled when EE is started and enabled when a file is loaded.
        See in activate_widgets_on_first_file_loaded() in MainWindow code.
        """
        # Analysis Panels
        assert not tgui.mw.mw.spkcnt_groupbox.isEnabled(), "spkcnt_groupbox is enabled on EE start"
        assert not tgui.mw.mw.ir_groupbox.isEnabled(), "ir_groupbox  is enabled on EE start"
        assert not tgui.mw.mw.skinetics_groupbox.isEnabled(), "skinetics_groupbox  is enabled on EE start"
        assert not tgui.mw.mw.spkcnt_table_groupbox.isEnabled(), "spkcnt_table_groupbox  is enabled on EE start"
        assert not tgui.mw.mw.ir_table_groupbox.isEnabled(), "ir_table_groupbox  is enabled on EE start"

        assert not tgui.mw.mw.curve_fitting_groupbox.isEnabled(), "curve_fitting groupbox is enabled on EE start"
        assert not tgui.mw.mw.events_template_groupbox.isEnabled(), "events_template groupbox is enabled on EE start"
        assert not tgui.mw.mw.events_threshold_groupbox.isEnabled(), "events_threshold groupbox is enabled on EE start"
        assert not tgui.mw.mw.events_table_groupbox.isEnabled(), "events_table_groupbox is enabled on EE start"

        # Data manipulation and Save Data
        assert not tgui.mw.mw.actionCopy_Vm_Plot.isEnabled(), "copy Vm is enabled on EE start "
        assert not tgui.mw.mw.actionCopy_Im_Plot.isEnabled(), "copy Im is enabled on EE start "
        assert not tgui.mw.mw.actionSave_All_Records_Vm.isEnabled(), "save Vm is enabled on EE start "
        assert not tgui.mw.mw.actionSave_All_Records_Im.isEnabled(), "save Im is enabled on EE start "
        assert not tgui.mw.mw.actionSave_Events_Analysis.isEnabled(), "save events is enabled on EE start"
        assert not tgui.mw.mw.actionLoad_Events_Analysis.isEnabled(), "load events is enabled on EE start "

        assert not tgui.mw.mw.actionUpsample.isEnabled(), "Upsample is enabled on EE start "
        assert not tgui.mw.mw.actionDownsample.isEnabled(), "Downsample is enabled on EE start "
        assert not tgui.mw.mw.actionFilter.isEnabled(), "Filter is enabled on EE start "
        assert not tgui.mw.mw.actionDetrend.isEnabled(), "Detrend is enabled on EE start "
        assert not tgui.mw.mw.actionRemove_Baseline.isEnabled(), "Remove Baseline is enabled on EE start "
        assert not tgui.mw.mw.actionAverage_Records.isEnabled(), "Average Records is enabled on EE start "
        assert not tgui.mw.mw.actionCut_Down_Trace_Time.isEnabled(), "Cut Down Trace is enabled on EE start "
        assert not tgui.mw.mw.actionNormalise_Timescale.isEnabled(), "Normalise is enabled on EE start "
        assert not tgui.mw.mw.actionReshape_Records.isEnabled(), "Cut Records is enabled on EE start "
        assert not tgui.mw.mw.actionReset_to_Raw_Data.isEnabled(), "Reset to raw Data is enabled on EE start "

        # Options
        assert not tgui.mw.mw.actionGraph_Options.isEnabled(), "Graph options is enabled on EE start "
        assert not tgui.mw.mw.actionFile_Details.isEnabled(), "File Details is enabled on EE start "

    def test_data_manipulation_and_options_are_enabled_on_load(self, tgui):
        load_file(tgui)

        assert tgui.mw.mw.actionCopy_Vm_Plot.isEnabled(), "copy Vm is disabled after file load"
        assert tgui.mw.mw.actionCopy_Im_Plot.isEnabled(), "copy Im is disabled after file load"
        assert tgui.mw.mw.actionSave_All_Records_Vm.isEnabled(), "save Vm is disabled after file load"
        assert tgui.mw.mw.actionSave_All_Records_Im.isEnabled(), "copy Im is disabled after file load"

        assert tgui.mw.mw.actionUpsample.isEnabled(), "Upsample is disabled after file load"
        assert tgui.mw.mw.actionDownsample.isEnabled(), "Downsample is disabled after file load"
        assert tgui.mw.mw.actionFilter.isEnabled(), "Filter is disabled after file load"
        assert tgui.mw.mw.actionDetrend.isEnabled(), "Detrend is disabled after file load"
        assert tgui.mw.mw.actionRemove_Baseline.isEnabled(), "Remove Baseline is disabled after file load"
        assert tgui.mw.mw.actionAverage_Records.isEnabled(), "Average record is disabled after file load"
        assert tgui.mw.mw.actionCut_Down_Trace_Time.isEnabled(), "Cut Down Trace is disabled after file load"
        assert tgui.mw.mw.actionNormalise_Timescale.isEnabled(), "Normalise Timescale is disabled after file load"
        assert tgui.mw.mw.actionReshape_Records.isEnabled(), "Cut (reshape) records is disabled after file load"
        assert tgui.mw.mw.actionReset_to_Raw_Data.isEnabled(), "Reset to raw data is disabled after file load"

        assert tgui.mw.mw.actionGraph_Options.isEnabled(), "Graph options data is disabled after file load"
        assert tgui.mw.mw.actionFile_Details.isEnabled(), "File Details is disabled after file load"

    def test_current_clamp_widgets_are_enabled_on_file_load(self, tgui):

        load_file(tgui, filetype="current_clamp")
        tgui.mw.mw.actionSelect_Analysis_Window.trigger()
        assert tgui.mw.dialogs["analysis_options"].dia.curve_fitting_button.isEnabled(), "curve fitting not active on cc file load "
        assert not tgui.mw.dialogs["analysis_options"].dia.events_template_matching_button.isEnabled(), "events template active on cc file load"
        assert not tgui.mw.dialogs["analysis_options"].dia.events_thresholding_button.isEnabled(), "events_threshold active on cc file load"
        assert tgui.mw.dialogs["analysis_options"].dia.inp_resistance_button.isEnabled(), "Ri not active on cc file load"
        assert tgui.mw.dialogs["analysis_options"].dia.spike_count_button.isEnabled(), "spike counting not active on cc file load"
        assert tgui.mw.dialogs["analysis_options"].dia.spike_kinetics_button.isEnabled(), "spike kinetics not active on cc file load"

        assert tgui.mw.mw.actionAP_Counting.isEnabled()
        assert tgui.mw.mw.actionAP_Kinetics.isEnabled()
        assert tgui.mw.mw.actionInput_Resistance.isEnabled()
        assert not tgui.mw.mw.actionEvents_Template.isEnabled()
        assert not tgui.mw.mw.actionEvents_Thresholding.isEnabled()

    def test_voltage_clamp_1_record_are_enabled_on_file_load(self, tgui):
        load_file(tgui, filetype="voltage_clamp_1_record")
        tgui.mw.mw.actionSelect_Analysis_Window.trigger()
        assert tgui.mw.dialogs["analysis_options"].dia.curve_fitting_button.isEnabled(), "curve fitting is not active on vc 1 channel file load"
        assert tgui.mw.dialogs[
            "analysis_options"].dia.events_template_matching_button.isEnabled(), "events template is not active on vc 1 channel file load"
        assert tgui.mw.dialogs[
            "analysis_options"].dia.events_thresholding_button.isEnabled(), "events thresholding is not active on vc 1 channel file load"
        assert not tgui.mw.dialogs["analysis_options"].dia.inp_resistance_button.isEnabled(), "Ri is active on vc 1 channel file load"
        assert not tgui.mw.dialogs["analysis_options"].dia.spike_count_button.isEnabled(), "Ri is active on vc 1 channel file load"
        assert not tgui.mw.dialogs["analysis_options"].dia.spike_kinetics_button.isEnabled(), "Ri is active on vc 1 channel file load"

        assert not tgui.mw.mw.actionAP_Counting.isEnabled()
        assert not tgui.mw.mw.actionAP_Kinetics.isEnabled()
        assert not tgui.mw.mw.actionInput_Resistance.isEnabled()
        assert tgui.mw.mw.actionEvents_Template.isEnabled()
        assert tgui.mw.mw.actionEvents_Thresholding.isEnabled()

    def test_voltage_clamp_multi_record_widgets_are_enabled_on_load(self, tgui):
        load_file(tgui, filetype="voltage_clamp_multi_record")
        tgui.mw.mw.actionSelect_Analysis_Window.trigger()
        assert tgui.mw.dialogs["analysis_options"].dia.curve_fitting_button.isEnabled(), "curve fitting is not active on vc multi channel file load"
        assert tgui.mw.dialogs[
            "analysis_options"].dia.events_template_matching_button.isEnabled(), "events template is not active on vc multi channel file load"
        assert tgui.mw.dialogs[
            "analysis_options"].dia.events_thresholding_button.isEnabled(), "events thresholding is not active on vc multi channel file load"
        assert not tgui.mw.dialogs["analysis_options"].dia.inp_resistance_button.isEnabled(), "Ri  is active on vc multi channel file load"
        assert not tgui.mw.dialogs["analysis_options"].dia.spike_count_button.isEnabled(), "spike count is active on vc multi channel file load"
        assert not tgui.mw.dialogs["analysis_options"].dia.spike_kinetics_button.isEnabled(), "skinetics is active on vc multi channel file load"

        assert not tgui.mw.mw.actionAP_Counting.isEnabled()
        assert not tgui.mw.mw.actionAP_Kinetics.isEnabled()
        assert not tgui.mw.mw.actionInput_Resistance.isEnabled()
        assert tgui.mw.mw.actionEvents_Template.isEnabled()
        assert tgui.mw.mw.actionEvents_Thresholding.isEnabled()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Tests to check that analysis panel for not-allowed analysis is hidden on file load (e.g. had voltage clamp file now current clamp, hide events panel)
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_analysis_panel_is_switched_off_voltage_clamp_events_tempate_on_current_clamp_load(self, tgui):
        """
        Check that analysis panel defaults to blank when an unallowed analysis type is loaded (e.g. events thresholding
        panel shown and a current clamp file loaded).
        """
        tgui.mw.mw.actionSelect_Analysis_Window.trigger()
        analysis_panel, table_panel = self.run_test_analysis_panel_clear_after_different_filetype_loaded(tgui,
                                                                                                         "voltage_clamp_1_record",
                                                                                                         "current_clamp",
                                                                                                         tgui.mw.dialogs[
                                                                                                             "analysis_options"].dia.events_template_matching_button)
        assert analysis_panel, "template matching panel not hidden after current clamp loaded"
        assert table_panel, "template matching table panel not hidden after current clamp loaded"

    def test_analysis_panel_is_switched_off_voltage_clamp_events_threshold_on_current_clamp_load(self, tgui):
        tgui.mw.mw.actionSelect_Analysis_Window.trigger()
        analysis_panel, table_panel = self.run_test_analysis_panel_clear_after_different_filetype_loaded(tgui,
                                                                                                         "voltage_clamp_1_record",
                                                                                                         "current_clamp",
                                                                                                         tgui.mw.dialogs[
                                                                                                             "analysis_options"].dia.events_thresholding_button)
        assert analysis_panel, "events thresholding panel not hidden after current clamp loaded"
        assert table_panel, "events thresholding table panel not hidden after current clamp loaded"

    @pytest.mark.parametrize("filetype", ["voltage_clamp_1_record", "voltage_clamp_multi_record"])
    def test_input_resistance_panel_is_switched_off_current_clamp_on_voltage_clamp_load(self, tgui, filetype):
        """
        Same as above but for current clamp then loading voltage clamp.
        Looping or parametrising button does not work, results in C++ interval Qt memory error
        """
        tgui.mw.mw.actionSelect_Analysis_Window.trigger()
        analysis_panel, table_panel = self.run_test_analysis_panel_clear_after_different_filetype_loaded(tgui,
                                                                                                         "current_clamp",
                                                                                                         filetype,
                                                                                                         tgui.mw.dialogs[
                                                                                                             "analysis_options"].dia.inp_resistance_button)
        assert analysis_panel, "Ri panel not hidden after voltage clamp loaded"
        assert table_panel, "Ri events table panel not hidden after voltage clamp loaded"

    @pytest.mark.parametrize("filetype", ["voltage_clamp_1_record", "voltage_clamp_multi_record"])
    def test_spkcnt_panel_is_switched_off_current_clamp_on_voltage_clamp_load(self, tgui, filetype):
        """
        Same as above but for current clamp then loading voltage clamp.
        Looping or parametrizing button does not work, results in C++ interval Qt memory error
        """
        tgui.mw.mw.actionSelect_Analysis_Window.trigger()
        analysis_panel, table_panel = self.run_test_analysis_panel_clear_after_different_filetype_loaded(tgui,
                                                                                                         "current_clamp",
                                                                                                         filetype,
                                                                                                         tgui.mw.dialogs[
                                                                                                             "analysis_options"].dia.spike_count_button)
        assert analysis_panel, "spike counting panel not hidden after voltage clamp loaded"
        assert table_panel, "spike counting table panel not hidden after voltage clamp loaded"

    @pytest.mark.parametrize("filetype", ["voltage_clamp_1_record", "voltage_clamp_multi_record"])
    def test_skinetics_panel_is_switched_off_current_clamp_on_voltage_clamp_load(self, tgui, filetype):
        tgui.mw.mw.actionSelect_Analysis_Window.trigger()
        analysis_panel, table_panel = self.run_test_analysis_panel_clear_after_different_filetype_loaded(tgui,
                                                                                                         "current_clamp",
                                                                                                         filetype,
                                                                                                         tgui.mw.dialogs[
                                                                                                             "analysis_options"].dia.spike_kinetics_button)
        assert analysis_panel, "skinetics panel not hidden after voltage clamp loaded"
        assert table_panel, "skinetics table panel not hidden after voltage clamp loaded"

    def test_change_analysis_triggers(self, tgui):

        tgui.mw.mw.actionCurve_Fitting.trigger()
        assert tgui.mw.cfgs.main["current_analysis"] == "curve_fitting"

        tgui.mw.mw.actionEvents_Template.trigger()
        assert tgui.mw.cfgs.main["current_analysis"] == "events_template_matching"

        tgui.mw.mw.actionEvents_Thresholding.trigger()
        assert tgui.mw.cfgs.main["current_analysis"] == "events_thresholding"

        tgui.mw.mw.actionAP_Counting.trigger()
        assert tgui.mw.cfgs.main["current_analysis"] == "spkcnt"

        tgui.mw.mw.actionAP_Kinetics.trigger()
        assert tgui.mw.cfgs.main["current_analysis"] == "skinetics"

        tgui.mw.mw.actionInput_Resistance.trigger()
        assert tgui.mw.cfgs.main["current_analysis"] == "Ri"

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Graph View Options
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_graph_view_options_default_config(self, tgui):
        """
        note that these are saved in csv file so not hard coded - just a check that the configs match the GUI
        """
        load_file(tgui)
        tgui.mw.mw.actionGraph_Options.trigger()

        assert tgui.mw.cfgs.graph_view_options["upperplot"] == "all_recs" if tgui.mw.dialogs[
            "graph_view_options"].dia.upper_scale_all_rec_radiobutton.isChecked() else "dispalyed_rec"
        assert tgui.mw.cfgs.graph_view_options["lowerplot"] == "all_recs" if tgui.mw.dialogs[
            "graph_view_options"].dia.lower_scale_all_rec_radiobutton.isChecked() else "dispalyed_rec"
        assert tgui.mw.cfgs.graph_view_options["limit_view_range_for_performance"] is True
        assert tgui.mw.cfgs.graph_view_options["voltage_clamp_one_record_view_limit_for_performance"] == 500000
        assert tgui.mw.cfgs.graph_view_options["upperplot_stretchfactor"] == 18  # hard coded for load
        assert tgui.mw.cfgs.graph_view_options["lowerplot_stretchfactor"] == 1
        assert tgui.mw.dialogs["graph_view_options"].dia.size_ratio_slider.minimum() == -20
        assert tgui.mw.dialogs["graph_view_options"].dia.size_ratio_slider.maximum() == 20
        assert tgui.mw.cfgs.graph_view_options["upperplot_grid"] == tgui.mw.dialogs["graph_view_options"].dia.upper_plot_gridlines.isChecked()
        assert tgui.mw.cfgs.graph_view_options["lowerplot_grid"] == tgui.mw.dialogs["graph_view_options"].dia.lower_plot_gridlines.isChecked()
        assert tgui.mw.cfgs.graph_view_options["link_upper_lower_lr"] == tgui.mw.mw.actionLink_im_vm_on.isChecked()
        assert tgui.mw.cfgs.graph_view_options["link_lr_across_recs"] == tgui.mw.mw.actionLink_Across_Records_on.isChecked()
        assert tgui.mw.cfgs.graph_view_options["plotpen_thickness"] == tgui.mw.dialogs["graph_view_options"].dia.plotpen_thickness_spinbox.value()

    def test_link_im_vm_config(self, tgui):
        """
        Check the config and checks on the Link boundary widgets. These determine whether linear regions are linked between
        Im and Vm, or across records.

        Change them to both possible settings, load a new file and check the config is saved (it should be saved automatically
        by the software across sessions).
        """
        load_file(tgui)
        tgui.mw.mw.actionLink_im_vm_on.trigger()
        assert tgui.mw.cfgs.graph_view_options["link_upper_lower_lr"] is True
        assert tgui.mw.cfgs.graph_view_options["link_upper_lower_lr"] == tgui.mw.mw.actionLink_im_vm_on.isChecked()

        tgui.mw.mw.actionLink_Across_Records_on.trigger()
        assert tgui.mw.cfgs.graph_view_options["link_lr_across_recs"] is True
        assert tgui.mw.cfgs.graph_view_options["link_lr_across_recs"] == tgui.mw.mw.actionLink_Across_Records_on.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui)
        assert tgui.mw.cfgs.graph_view_options["link_upper_lower_lr"] is True
        assert tgui.mw.cfgs.graph_view_options["link_upper_lower_lr"] == tgui.mw.mw.actionLink_im_vm_on.isChecked()

        assert tgui.mw.cfgs.graph_view_options["link_lr_across_recs"] is True
        assert tgui.mw.cfgs.graph_view_options["link_lr_across_recs"] == tgui.mw.mw.actionLink_Across_Records_on.isChecked()

        tgui.mw.mw.actionLink_im_vm_off.trigger()
        assert tgui.mw.cfgs.graph_view_options["link_upper_lower_lr"] is False
        assert tgui.mw.cfgs.graph_view_options["link_upper_lower_lr"] == tgui.mw.mw.actionLink_im_vm_on.isChecked()
        assert tgui.mw.mw.actionLink_im_vm_off.isChecked()

        tgui.mw.mw.actionLink_Across_Records_off.trigger()
        assert tgui.mw.cfgs.graph_view_options["link_lr_across_recs"] is False
        assert tgui.mw.cfgs.graph_view_options["link_lr_across_recs"] == tgui.mw.mw.actionLink_Across_Records_on.isChecked()
        assert tgui.mw.mw.actionLink_Across_Records_off.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui)
        assert tgui.mw.cfgs.graph_view_options["link_upper_lower_lr"] is False
        assert tgui.mw.cfgs.graph_view_options["link_upper_lower_lr"] == tgui.mw.mw.actionLink_im_vm_on.isChecked()
        assert tgui.mw.mw.actionLink_im_vm_off.isChecked()

        assert tgui.mw.cfgs.graph_view_options["link_lr_across_recs"] is False
        assert tgui.mw.cfgs.graph_view_options["link_lr_across_recs"] == tgui.mw.mw.actionLink_Across_Records_on.isChecked()
        assert tgui.mw.mw.actionLink_Across_Records_off.isChecked()

        tgui.shutdown()

    def test_graph_view_option_slider_configs(self, tgui):
        load_file(tgui)
        tgui.mw.mw.actionGraph_Options.trigger()
        for slider_val in range(-20, 21):
            tgui.mw.dialogs["graph_view_options"].dia.size_ratio_slider.setValue(slider_val)
            assert tgui.mw.cfgs.graph_view_options["upperplot_stretchfactor"] == 50 - (2.5 * (slider_val * -1))
            assert tgui.mw.cfgs.graph_view_options["lowerplot_stretchfactor"] == 50 + (2.5 * (slider_val * -1))
            assert tgui.mw.cfgs.graph_view_options["slider_val"] == tgui.mw.dialogs["graph_view_options"].dia.size_ratio_slider.value()

        # Reload for Configs Check
        tgui.mw.dialogs["graph_view_options"].dia.size_ratio_slider.setValue(4)
        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["upperplot_stretchfactor"] == 50 - (2.5 * (4 * -1))
        assert tgui.mw.cfgs.graph_view_options["lowerplot_stretchfactor"] == 50 + (2.5 * (4 * -1))
        assert tgui.mw.cfgs.graph_view_options["slider_val"] == tgui.mw.dialogs["graph_view_options"].dia.size_ratio_slider.value()

        tgui.shutdown()

    def test_display_rec_radiobuttons_upperplot(self, tgui):
        load_file(tgui)
        tgui.mw.mw.actionGraph_Options.trigger()

        tgui.left_mouse_click(tgui.mw.dialogs["graph_view_options"].dia.upper_scale_all_rec_radiobutton)
        assert tgui.mw.cfgs.graph_view_options["upperplot"] == "all_recs"
        assert tgui.mw.dialogs["graph_view_options"].dia.upper_scale_all_rec_radiobutton.isChecked()
        assert not tgui.mw.dialogs["graph_view_options"].dia.upper_scale_display_rec_radiobutton.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["upperplot"] == "all_recs"
        assert tgui.mw.dialogs["graph_view_options"].dia.upper_scale_all_rec_radiobutton.isChecked()
        assert not tgui.mw.dialogs["graph_view_options"].dia.upper_scale_display_rec_radiobutton.isChecked()

        tgui.left_mouse_click(tgui.mw.dialogs["graph_view_options"].dia.upper_scale_display_rec_radiobutton)
        assert tgui.mw.cfgs.graph_view_options["upperplot"] == "displayed_rec"
        assert not tgui.mw.dialogs["graph_view_options"].dia.upper_scale_all_rec_radiobutton.isChecked()
        assert tgui.mw.dialogs["graph_view_options"].dia.upper_scale_display_rec_radiobutton.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["upperplot"] == "displayed_rec"
        assert not tgui.mw.dialogs["graph_view_options"].dia.upper_scale_all_rec_radiobutton.isChecked()
        assert tgui.mw.dialogs["graph_view_options"].dia.upper_scale_display_rec_radiobutton.isChecked()

        tgui.shutdown()

    def test_display_rec_radiobuttons_lowerplot(self, tgui):  # must use full names or get c++ error
        load_file(tgui)
        tgui.mw.mw.actionGraph_Options.trigger()

        tgui.left_mouse_click(tgui.mw.dialogs["graph_view_options"].dia.lower_scale_all_rec_radiobutton)
        assert tgui.mw.cfgs.graph_view_options["lowerplot"] == "all_recs"
        assert tgui.mw.dialogs["graph_view_options"].dia.lower_scale_all_rec_radiobutton.isChecked()
        assert not tgui.mw.dialogs["graph_view_options"].dia.lower_scale_display_rec_radiobutton.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["lowerplot"] == "all_recs"
        assert tgui.mw.dialogs["graph_view_options"].dia.lower_scale_all_rec_radiobutton.isChecked()
        assert not tgui.mw.dialogs["graph_view_options"].dia.lower_scale_display_rec_radiobutton.isChecked()

        tgui.left_mouse_click(tgui.mw.dialogs["graph_view_options"].dia.lower_scale_display_rec_radiobutton)
        assert tgui.mw.cfgs.graph_view_options["lowerplot"] == "displayed_rec"
        assert not tgui.mw.dialogs["graph_view_options"].dia.lower_scale_all_rec_radiobutton.isChecked()
        assert tgui.mw.dialogs["graph_view_options"].dia.lower_scale_display_rec_radiobutton.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["lowerplot"] == "displayed_rec"
        assert not tgui.mw.dialogs["graph_view_options"].dia.lower_scale_all_rec_radiobutton.isChecked()
        assert tgui.mw.dialogs["graph_view_options"].dia.lower_scale_display_rec_radiobutton.isChecked()

        tgui.shutdown()

    def test_upperplot_grid_config(self, tgui):
        load_file(tgui)
        tgui.mw.mw.actionGraph_Options.trigger()

        tgui.switch_checkbox(tgui.mw.dialogs["graph_view_options"].dia.upper_plot_gridlines, on=False)
        assert tgui.mw.cfgs.graph_view_options["upperplot_grid"] is False
        assert not tgui.mw.dialogs["graph_view_options"].dia.upper_plot_gridlines.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["upperplot_grid"] is False
        assert not tgui.mw.dialogs["graph_view_options"].dia.upper_plot_gridlines.isChecked()

        tgui.switch_checkbox(tgui.mw.dialogs["graph_view_options"].dia.upper_plot_gridlines, on=True)
        assert tgui.mw.cfgs.graph_view_options["upperplot_grid"] is True
        assert tgui.mw.dialogs["graph_view_options"].dia.upper_plot_gridlines.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["upperplot_grid"] is True
        assert tgui.mw.dialogs["graph_view_options"].dia.upper_plot_gridlines.isChecked()

        tgui.shutdown()

    def test_lowerplot_grid_config(self, tgui):
        load_file(tgui)

        tgui.mw.mw.actionGraph_Options.trigger()
        tgui.switch_checkbox(tgui.mw.dialogs["graph_view_options"].dia.lower_plot_gridlines, on=False)
        assert tgui.mw.cfgs.graph_view_options["lowerplot_grid"] is False
        assert not tgui.mw.dialogs["graph_view_options"].dia.lower_plot_gridlines.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["lowerplot_grid"] is False
        assert not tgui.mw.dialogs["graph_view_options"].dia.lower_plot_gridlines.isChecked()

        tgui.switch_checkbox(tgui.mw.dialogs["graph_view_options"].dia.lower_plot_gridlines, on=True)
        assert tgui.mw.cfgs.graph_view_options["lowerplot_grid"] is True
        assert tgui.mw.dialogs["graph_view_options"].dia.lower_plot_gridlines.isChecked()

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["lowerplot_grid"] is True
        assert tgui.mw.dialogs["graph_view_options"].dia.lower_plot_gridlines.isChecked()

        tgui.shutdown()

    def test_plotpen_spinbox(self, tgui):
        load_file(tgui)
        tgui.mw.mw.actionGraph_Options.trigger()
        assert tgui.mw.cfgs.graph_view_options["plotpen_thickness"] == tgui.mw.dialogs["graph_view_options"].dia.plotpen_thickness_spinbox.value()
        for slider_val in np.linspace(0, 8,
                                      81):  # THESE SHOULD BUT SOMEWHERE IN CONFIGS ALSO, A SETTING OR SET OF SETTINGS THAT DONT CHANGE. E.G. WIDGET MIN/MAX VALUES.
            tgui.mw.dialogs["graph_view_options"].dia.plotpen_thickness_spinbox.setValue(slider_val)
            assert tgui.mw.cfgs.graph_view_options["plotpen_thickness"] == tgui.mw.dialogs["graph_view_options"].dia.plotpen_thickness_spinbox.value()

        tgui.mw.dialogs["graph_view_options"].dia.plotpen_thickness_spinbox.setValue(4)
        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert 4 == tgui.mw.cfgs.graph_view_options["plotpen_thickness"] == tgui.mw.dialogs[
            "graph_view_options"].dia.plotpen_thickness_spinbox.value()

        tgui.shutdown()

    def test_limit_graph_view(self, tgui):
        load_file(tgui)
        tgui.mw.mw.actionGraph_Options.trigger()

        # Test groupbox is correct and default saves
        assert tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_groupbox.isChecked()

        tgui.switch_groupbox(tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_groupbox,
                             on=False)

        assert not tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_groupbox.isChecked()
        assert tgui.mw.cfgs.graph_view_options["limit_view_range_for_performance"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert not tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_groupbox.isChecked()
        assert tgui.mw.cfgs.graph_view_options["limit_view_range_for_performance"] is False

        tgui.switch_groupbox(tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_groupbox,
                             on=True)

        # check spinbox value is correct and saves
        assert tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_spinbox.value() == 500
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_spinbox,
                                       250)
        assert tgui.mw.cfgs.graph_view_options["voltage_clamp_one_record_view_limit_for_performance"] == 250000

        tgui = close_and_reload_for_defaults_check(tgui, save_="graph_view_options")
        assert tgui.mw.cfgs.graph_view_options["voltage_clamp_one_record_view_limit_for_performance"] == 250000
        assert tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_spinbox.value() == 250

        # check spinbox suffix and min
        assert tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_spinbox.suffix() == "k samples"
        assert tgui.mw.dialogs["graph_view_options"].dia.limit_view_range_for_performance_spinbox.minimum() == 1

        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Main Configs
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_main_config_default_setting(self, tgui):
        """
        Test default main configs - this is mainly linked to the state of the GUI. This is a little pointless, most of these are
        updated on file load, and this is when it is important what their state is (these tests are below).
        But good to check default cfgs are not accidently changed in case it has unintended consequences.
        """
        assert tgui.mw.cfgs.main["batch_mode"] is tgui.mw.mw.actionBatch_Mode_ON.isChecked()
        assert tgui.mw.cfgs.main["batch_mode"] is (tgui.mw.mw.splash_batch_off_button.graphicsEffect() is None)
        assert tgui.mw.cfgs.main["base_dir"] is None
        assert tgui.mw.cfgs.main["base_file_ext"] is None
        assert tgui.mw.cfgs.main["analyse_specific_recs"] is False
        assert tgui.mw.cfgs.main["rec_from"] is None
        assert tgui.mw.cfgs.main["rec_to"] is None
        assert tgui.mw.cfgs.main["is_file_loaded"] is False
        assert tgui.mw.cfgs.main["displayed_rec"] == 0
        assert tgui.mw.cfgs.main["current_analysis"] == "None"
        assert tgui.mw.cfgs.main["click_spike_mode"] is False
        assert tgui.mw.cfgs.main["spkcnt_more_analyses_popup_shown"] is False
        assert tgui.mw.cfgs.main["one_data_channel"] is False
        assert tgui.mw.cfgs.main["data_manipulated"] == []
        assert tgui.mw.cfgs.main["freeze_all_boundary_regions"] is False
        assert tgui.mw.cfgs.main["file_first_analysis"]["spkcnt"] is True
        assert tgui.mw.cfgs.main["file_first_analysis"]["Ri"] is True
        assert tgui.mw.cfgs.main["file_first_analysis"]["skinetics"] is True
        assert tgui.mw.cfgs.main["file_first_analysis"]["events"] is True
        assert tgui.mw.cfgs.main["file_first_analysis"]["curve_fitting"] is True
        assert tgui.mw.cfgs.main["im_inj_protocol_padding"] == 10
        assert tgui.mw.cfgs.file_load_options["importdata_always_normalise_time"] is False
        assert tgui.mw.cfgs.file_load_options["importdata_always_offset_time"] is False
        assert tgui.mw.cfgs.file_load_options["show_large_file_warning"] is True
        assert tgui.mw.cfgs.file_load_options["force_load_options"] is None
        assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["on"] is False
        assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["channel_1_idx"] == 0
        assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"] == 1
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["on"] is False
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["assume_pa"] is True
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["pa_unit_to_convert"] == "fA"
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["on"] is False
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["assume_mv"] is True
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["mv_unit_to_convert"] == "pV"
        assert tgui.mw.cfgs.file_load_options["generate_axon_protocol"] is False

    def test_batch_mode_splash_screen_widgets_connected(self, tgui):
        """
        Load File not tested
        """
        tgui.left_mouse_click(tgui.mw.mw.splash_batch_on_button)
        assert tgui.mw.cfgs.main["batch_mode"] is True
        tgui.left_mouse_click(tgui.mw.mw.splash_batch_off_button)
        assert tgui.mw.cfgs.main["batch_mode"] is False

        assert tgui.mw.dialogs["analysis_options"] is None
        tgui.left_mouse_click(tgui.mw.mw.splash_analysis_panel_button)
        assert tgui.mw.dialogs["analysis_options"] is not None

    def test_batch_mode_menubar_widgets_connected(self, tgui):
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        assert tgui.mw.cfgs.main["batch_mode"] is True
        tgui.mw.mw.actionBatch_Mode_OFF.trigger()
        assert tgui.mw.cfgs.main["batch_mode"] is False

    def test_base_dir_and_base_file_ext_config(self, tgui):
        tgui.test_update_fileinfo()

    def test_analyse_specific_recs_spkcnt_config(self, tgui):
        """
        Test all recs to analyse spinboxes are off, turn them on and check config,
        change the recs_to / recs_from for each analysis and check the configs.
        Very long, could do this with loops but will probably become convoluted
        """
        load_file(tgui)
        assert tgui.mw.cfgs.spkcnt["analyse_specific_recs"] is False
        assert tgui.mw.mw.spkcnt_recs_to_analyse_groupbox.isChecked() is False
        assert tgui.mw.cfgs.ir["analyse_specific_recs"] is False
        assert tgui.mw.mw.ir_recs_to_analyse_groupbox.isChecked() is False
        assert tgui.mw.cfgs.skinetics["analyse_specific_recs"] is False
        assert tgui.mw.mw.skinetics_recs_to_analyse_groupbox.isChecked() is False
        assert tgui.mw.cfgs.curve_fitting["analyse_specific_recs"] is False
        assert tgui.mw.mw.curve_fitting_recs_to_analyse_groupbox.isChecked() is False

        # Test Turning groupbox on
        tgui.switch_groupbox(tgui.mw.mw.spkcnt_recs_to_analyse_groupbox,
                             on=True)
        assert tgui.mw.cfgs.spkcnt["analyse_specific_recs"] is True

        tgui.switch_groupbox(tgui.mw.mw.ir_recs_to_analyse_groupbox,
                             on=True)
        assert tgui.mw.cfgs.ir["analyse_specific_recs"] is True

        tgui.switch_groupbox(tgui.mw.mw.skinetics_recs_to_analyse_groupbox,
                             on=True)
        assert tgui.mw.cfgs.skinetics["analyse_specific_recs"] is True

        tgui.switch_groupbox(tgui.mw.mw.curve_fitting_recs_to_analyse_groupbox,
                             on=True)
        assert tgui.mw.cfgs.curve_fitting["analyse_specific_recs"] is True

        # Switch Rec to / Rec from
        tgui.enter_number_into_spinbox(tgui.mw.mw.spkcnt_spike_recs_from_spinbox,
                                       number=5)
        tgui.enter_number_into_spinbox(tgui.mw.mw.spkcnt_spike_recs_to_spinbox,
                                       number=65)
        assert tgui.mw.cfgs.spkcnt["rec_from"] == 5 - 1
        assert tgui.mw.cfgs.spkcnt["rec_to"] == 65 - 1

        tgui.enter_number_into_spinbox(tgui.mw.mw.skinetics_recs_from_spinbox,
                                       number=10)
        tgui.enter_number_into_spinbox(tgui.mw.mw.skinetics_recs_to_spinbox,
                                       number=60)
        assert tgui.mw.cfgs.skinetics["rec_from"] == 10 - 1
        assert tgui.mw.cfgs.skinetics["rec_to"] == 60 - 1

        tgui.enter_number_into_spinbox(tgui.mw.mw.ir_recs_from_spinbox,
                                       number=15)
        tgui.enter_number_into_spinbox(tgui.mw.mw.ir_recs_to_spinbox,
                                       number=55)
        assert tgui.mw.cfgs.ir["rec_from"] == 15 - 1
        assert tgui.mw.cfgs.ir["rec_to"] == 55 - 1

        tgui.enter_number_into_spinbox(tgui.mw.mw.curve_fitting_recs_from_spinbox,
                                       number=20)
        tgui.enter_number_into_spinbox(tgui.mw.mw.curve_fitting_recs_to_spinbox,
                                       number=50)
        assert tgui.mw.cfgs.curve_fitting["rec_from"] == 20 - 1
        assert tgui.mw.cfgs.curve_fitting["rec_to"] == 50 - 1

    def test_load_file_config(self, tgui):
        assert tgui.mw.cfgs.main["is_file_loaded"] is False
        load_file(tgui)
        assert tgui.mw.cfgs.main["is_file_loaded"] is True

    def test_displayed_rec_config(self, tgui):
        tgui.mw.update_displayed_rec(0)
        for i in range(20):
            tgui.left_mouse_click(tgui.mw.mw.current_rec_rightbutton)
            assert tgui.mw.cfgs.main["displayed_rec"] == tgui.mw.mw.current_rec_spinbox.value() - 1

    def test_current_displayed_analysis_current_clamp_config(self, tgui):
        tgui.set_analysis_type("spkcnt")
        assert tgui.mw.cfgs.main["current_analysis"] == "spkcnt"
        tgui.set_analysis_type("Ri")
        assert tgui.mw.cfgs.main["current_analysis"] == "Ri"
        tgui.set_analysis_type("skinetics")
        assert tgui.mw.cfgs.main["current_analysis"] == "skinetics"
        tgui.set_analysis_type("curve_fitting")
        assert tgui.mw.cfgs.main["current_analysis"] == "curve_fitting"
        tgui.set_analysis_type("events_template_matching")
        assert tgui.mw.cfgs.main["current_analysis"] == "events_template_matching"
        tgui.set_analysis_type("events_thresholding")
        assert tgui.mw.cfgs.main["current_analysis"] == "events_thresholding"

    def test_spkcnt_analysis_click_spike_mode_button(self, tgui):
        """
        Currently click-spike mode is not available until action potential counting analysis has been run
        """
        load_file(tgui)
        tgui.left_mouse_click(tgui.mw.mw.spike_count_button)
        assert tgui.mw.cfgs.main["click_spike_mode"] is False, "click mode is on after initial analysis conducted"
        tgui.left_mouse_click(tgui.mw.mw.spkcnt_click_mode_button)
        assert tgui.mw.cfgs.main["click_spike_mode"] is True, "click mode is not turned on after button pressed"

    def test_spkcnt_analysis_popup_config(self, tgui):
        load_file(tgui)
        assert tgui.mw.cfgs.main["spkcnt_more_analyses_popup_shown"] is False
        tgui.left_mouse_click(tgui.mw.mw.spkcnt_more_analyses_button)
        assert tgui.mw.cfgs.main["spkcnt_more_analyses_popup_shown"] is True
        tgui.left_mouse_click(tgui.mw.mw.spkcnt_more_analyses_button)
        assert tgui.mw.cfgs.main["spkcnt_more_analyses_popup_shown"] is False

    def test_one_data_channel_config(self, tgui):
        load_file(tgui, filetype="current_clamp")
        assert tgui.mw.cfgs.main["one_data_channel"] is True
        load_file(tgui, filetype="voltage_clamp_multi_record")
        assert tgui.mw.cfgs.main["one_data_channel"] is False
        load_file(tgui, filetype="current_clamp")
        assert tgui.mw.cfgs.main["one_data_channel"] is True

    def test_data_manipulation_config(self, tgui):
        """
        These check whether these data steps have already been run so they cannot be run repeatedly
        per file.
        """
        load_file(tgui)
        assert tgui.mw.cfgs.main["data_manipulated"] == []

        tgui.mw.mw.actionUpsample.trigger()
        tgui.mw.dialogs["upsample"].accept()
        assert tgui.mw.cfgs.main["data_manipulated"] == ["upsampled"]

        tgui.mw.mw.actionDownsample.trigger()
        tgui.mw.dialogs["downsample"].accept()
        assert tgui.mw.cfgs.main["data_manipulated"] == ["upsampled", "downsampled"]

        tgui.mw.mw.actionFilter.trigger()
        tgui.mw.dialogs["filter_data"].accept()
        assert tgui.mw.cfgs.main["data_manipulated"] == ["upsampled", "downsampled", "filtered"]

        load_file(tgui)
        assert tgui.mw.cfgs.main["data_manipulated"] == []

    def test_freeze_all_bounary_regions_config(self, tgui):
        load_file(tgui)
        assert tgui.mw.cfgs.main["freeze_all_boundary_regions"] is False
        tgui.mw.loaded_file_plot.upperplot.vb.menu.hold_bounds_action.trigger()
        assert tgui.mw.cfgs.main["freeze_all_boundary_regions"] is True
        tgui.mw.loaded_file_plot.upperplot.vb.menu.hold_bounds_action.trigger()
        assert tgui.mw.cfgs.main["freeze_all_boundary_regions"] is False

    # Legacy Options
    # ------------------------------------------------------------------------------------------------------------------------------------------------

    def test_legacy_options(self, tgui):

        assert tgui.mw.cfgs.main["legacy_options"]["baseline_method"] is False
        assert tgui.mw.cfgs.main["legacy_options"]["decay_detection_method"] is False
        assert tgui.mw.cfgs.main["legacy_options"]["cum_prob_bin_limits"] is False
        assert tgui.mw.cfgs.main["legacy_options"]["baseline_enhanced_position"] is False

        tgui.mw.mw.actionLegacy.trigger()

        tgui.switch_checkbox(tgui.mw.dialogs["legacy_options"].dia.legacy_baseline_no_enhanced_position_checkbox, on=True)
        tgui.switch_checkbox(tgui.mw.dialogs["legacy_options"].dia.legacy_baseline_checkbox, on=True)
        tgui.switch_checkbox(tgui.mw.dialogs["legacy_options"].dia.legacy_decay_detection_checkbox, on=True)
        tgui.switch_checkbox(tgui.mw.dialogs["legacy_options"].dia.legacy_cumulative_probabiltiy_bin_limits, on=True)

        assert tgui.mw.cfgs.main["legacy_options"]["baseline_method"] is True
        assert tgui.mw.cfgs.main["legacy_options"]["decay_detection_method"] is True
        assert tgui.mw.cfgs.main["legacy_options"]["cum_prob_bin_limits"] is True
        assert tgui.mw.cfgs.main["legacy_options"]["baseline_enhanced_position"] is True

        tgui = close_and_reload_for_defaults_check(tgui, save_="legacy_options")

        assert tgui.mw.dialogs["legacy_options"].dia.legacy_baseline_no_enhanced_position_checkbox.isChecked()
        assert tgui.mw.dialogs["legacy_options"].dia.legacy_baseline_checkbox.isChecked()
        assert tgui.mw.dialogs["legacy_options"].dia.legacy_decay_detection_checkbox.isChecked()
        assert tgui.mw.dialogs["legacy_options"].dia.legacy_cumulative_probabiltiy_bin_limits.isChecked()

        assert tgui.mw.cfgs.main["legacy_options"]["baseline_method"] is True
        assert tgui.mw.cfgs.main["legacy_options"]["decay_detection_method"] is True
        assert tgui.mw.cfgs.main["legacy_options"]["cum_prob_bin_limits"] is True
        assert tgui.mw.cfgs.main["legacy_options"]["baseline_enhanced_position"] is True

        tgui.shutdown()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Force Load Dialog
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_default_units_load_options_config_on_init(self, tgui):
        """
        Check all options change the correct config. Probs uncessaryt as done below.
        """
        tgui.mw.mw.actionForce_Load_Options.trigger()

        assert tgui.mw.dialogs["force_load_options"].dia.default_im_units_groupbox.isChecked() == tgui.mw.cfgs.file_load_options["default_im_units"][
            "on"]
        assert tgui.mw.dialogs["force_load_options"].dia.default_vm_units_groupbox.isChecked() == tgui.mw.cfgs.file_load_options["default_vm_units"][
            "on"]
        assert tgui.mw.dialogs["force_load_options"].dia.assume_pa_radiobutton.isChecked() == tgui.mw.cfgs.file_load_options["default_im_units"][
            "assume_pa"]
        assert tgui.mw.dialogs["force_load_options"].dia.convert_pa_combobox.currentText() == tgui.mw.cfgs.file_load_options["default_im_units"][
            "pa_unit_to_convert"]
        assert tgui.mw.dialogs["force_load_options"].dia.convert_to_pa_radiobutton.isChecked() == (
            not tgui.mw.cfgs.file_load_options["default_im_units"]["assume_pa"])
        assert tgui.mw.dialogs["force_load_options"].dia.assume_mv_radiobutton.isChecked() == tgui.mw.cfgs.file_load_options["default_vm_units"][
            "assume_mv"]
        assert tgui.mw.dialogs["force_load_options"].dia.convert_to_mv_radiobutton.isChecked() == (
            not tgui.mw.cfgs.file_load_options["default_vm_units"]["assume_mv"])
        assert tgui.mw.dialogs["force_load_options"].dia.convert_mv_combobox.currentText() == tgui.mw.cfgs.file_load_options["default_vm_units"][
            "mv_unit_to_convert"]
        assert tgui.mw.dialogs["force_load_options"].dia.generate_axon_protocol_checkbox.isChecked() == tgui.mw.cfgs.file_load_options["generate_axon_protocol"]

    def test_default_units_load_options_config(self, tgui):

        tgui.mw.mw.actionForce_Load_Options.trigger()

        assert tgui.mw.dialogs["force_load_options"].dia.default_im_units_groupbox.isChecked() is False
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["on"] is False
        tgui.switch_groupbox(tgui.mw.dialogs["force_load_options"].dia.default_im_units_groupbox, on=True)
        assert tgui.mw.dialogs["force_load_options"].dia.default_im_units_groupbox.isChecked()
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["on"]

        assert tgui.mw.dialogs["force_load_options"].dia.default_vm_units_groupbox.isChecked() is False
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["on"] is False
        tgui.switch_groupbox(tgui.mw.dialogs["force_load_options"].dia.default_vm_units_groupbox, on=True)
        assert tgui.mw.dialogs["force_load_options"].dia.default_vm_units_groupbox.isChecked()
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["on"]

        assert tgui.mw.dialogs["force_load_options"].dia.assume_pa_radiobutton.isChecked()
        assert tgui.mw.dialogs["force_load_options"].dia.convert_to_pa_radiobutton.isChecked() is False
        assert tgui.mw.dialogs["force_load_options"].dia.convert_pa_combobox.isEnabled() is False
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["assume_pa"] is True
        tgui.left_mouse_click(tgui.mw.dialogs["force_load_options"].dia.convert_to_pa_radiobutton)
        assert tgui.mw.dialogs["force_load_options"].dia.assume_pa_radiobutton.isChecked() is False
        assert tgui.mw.dialogs["force_load_options"].dia.convert_to_pa_radiobutton.isChecked()
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["assume_pa"] is False
        assert tgui.mw.dialogs["force_load_options"].dia.convert_pa_combobox.isEnabled() is True

        assert tgui.mw.dialogs["force_load_options"].dia.assume_mv_radiobutton.isChecked()
        assert tgui.mw.dialogs["force_load_options"].dia.convert_to_mv_radiobutton.isChecked() is False
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["assume_mv"] is True
        assert tgui.mw.dialogs["force_load_options"].dia.convert_mv_combobox.isEnabled() is False
        tgui.left_mouse_click(tgui.mw.dialogs["force_load_options"].dia.convert_to_mv_radiobutton)
        assert tgui.mw.dialogs["force_load_options"].dia.assume_mv_radiobutton.isChecked() is False
        assert tgui.mw.dialogs["force_load_options"].dia.convert_to_mv_radiobutton.isChecked()
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["assume_mv"] is False
        assert tgui.mw.dialogs["force_load_options"].dia.convert_mv_combobox.isEnabled() is True

        for cnt, unit in enumerate(["fA", "nA", "uA", "mA", "A"]):
            tgui.set_combobox(tgui.mw.dialogs["force_load_options"].dia.convert_pa_combobox,
                              idx=cnt)

            assert tgui.mw.dialogs["force_load_options"].dia.convert_pa_combobox.currentText() == unit
            assert tgui.mw.cfgs.file_load_options["default_im_units"]["pa_unit_to_convert"] == unit

        for cnt, unit in enumerate(["pV", "nV", "uV", "V"]):
            tgui.set_combobox(tgui.mw.dialogs["force_load_options"].dia.convert_mv_combobox,
                              idx=cnt)
            assert tgui.mw.dialogs["force_load_options"].dia.convert_mv_combobox.currentText() == unit
            assert tgui.mw.cfgs.file_load_options["default_vm_units"]["mv_unit_to_convert"] == unit

        # Save, quick and check correct are still shown.
        tgui = close_and_reload_for_defaults_check(tgui, save_="file_loading_options")

        assert tgui.mw.dialogs["force_load_options"].dia.default_im_units_groupbox.isChecked()
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["on"]

        assert tgui.mw.dialogs["force_load_options"].dia.default_vm_units_groupbox.isChecked()
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["on"]

        assert tgui.mw.dialogs["force_load_options"].dia.assume_pa_radiobutton.isChecked() is False
        assert tgui.mw.dialogs["force_load_options"].dia.convert_to_pa_radiobutton.isChecked()
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["assume_pa"] is False
        assert tgui.mw.dialogs["force_load_options"].dia.convert_pa_combobox.isEnabled() is True

        assert tgui.mw.dialogs["force_load_options"].dia.assume_mv_radiobutton.isChecked() is False
        assert tgui.mw.dialogs["force_load_options"].dia.convert_to_mv_radiobutton.isChecked()
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["assume_mv"] is False
        assert tgui.mw.dialogs["force_load_options"].dia.convert_mv_combobox.isEnabled() is True

        assert tgui.mw.dialogs["force_load_options"].dia.convert_pa_combobox.currentText() == "A"
        assert tgui.mw.cfgs.file_load_options["default_im_units"]["pa_unit_to_convert"] == "A"
        assert tgui.mw.dialogs["force_load_options"].dia.convert_mv_combobox.currentText() == "V"
        assert tgui.mw.cfgs.file_load_options["default_vm_units"]["mv_unit_to_convert"] == "V"

        tgui.shutdown()

    def test_force_load_options_config(self, tgui):
        """
        Check all options for forcing load are connected to the correct config.
        """
        tgui.load_a_filetype("current_clamp")  # must be current clamp for generate axon protocol test

        tgui.mw.mw.actionForce_Load_Options.trigger()
        assert tgui.mw.dialogs[
            "force_load_options"].dia.do_not_force_load_radiobutton.isChecked(), "do not load togglebox is not selected on dialog load"

        tgui.click_checkbox(tgui.mw.dialogs["force_load_options"].dia.do_not_force_load_radiobutton)
        assert tgui.mw.cfgs.file_load_options["force_load_options"] is None

        tgui.click_checkbox(tgui.mw.dialogs["force_load_options"].dia.force_voltage_clamp_radiobutton)
        assert tgui.mw.cfgs.file_load_options["force_load_options"] == "voltage_clamp"

        tgui.click_checkbox(tgui.mw.dialogs["force_load_options"].dia.force_current_clamp_radiobutton)
        assert tgui.mw.cfgs.file_load_options["force_load_options"] == "current_clamp"

        tgui.switch_groupbox(tgui.mw.dialogs["force_load_options"].dia.select_channels_to_load_groupbox, on=True)
        assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["on"]

        tgui.click_checkbox(tgui.mw.dialogs["force_load_options"].dia.generate_axon_protocol_checkbox)
        QtWidgets.QApplication.processEvents()
        assert tgui.mw.cfgs.file_load_options["generate_axon_protocol"] is True
        assert tgui.mw.dialogs["force_load_options"].dia.secondary_data_channel_override_combobox.isEnabled() is False
        assert tgui.mw.dialogs["force_load_options"].dia.secondary_data_channel_override_combobox.currentIndex() == 0
        assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"] is None

        tgui.click_checkbox(tgui.mw.dialogs["force_load_options"].dia.generate_axon_protocol_checkbox)
        assert tgui.mw.cfgs.file_load_options["generate_axon_protocol"] is False
        assert tgui.mw.dialogs["force_load_options"].dia.secondary_data_channel_override_combobox.isEnabled() is True
        tgui.click_checkbox(tgui.mw.dialogs["force_load_options"].dia.generate_axon_protocol_checkbox)  # for reload test

        assert tgui.mw.cfgs.file_load_options["importdata_always_normalise_time"] == tgui.mw.dialogs[
                                                                                                     "force_load_options"].dia.always_normalise_time_checkbox.isChecked()

        tgui.switch_checkbox(tgui.mw.dialogs["force_load_options"].dia.always_normalise_time_checkbox, on=False)
        assert tgui.mw.cfgs.file_load_options["importdata_always_normalise_time"] is False
        tgui.switch_checkbox(tgui.mw.dialogs["force_load_options"].dia.always_normalise_time_checkbox, on=True)
        assert tgui.mw.cfgs.file_load_options["importdata_always_normalise_time"] is True

        tgui.switch_checkbox(tgui.mw.dialogs["force_load_options"].dia.show_large_file_warning_checkbox, on=False)  # default is true
        assert tgui.mw.cfgs.file_load_options["show_large_file_warning"] is False
        tgui.switch_checkbox(tgui.mw.dialogs["force_load_options"].dia.show_large_file_warning_checkbox, on=True)
        assert tgui.mw.cfgs.file_load_options["show_large_file_warning"] is True
        tgui.switch_checkbox(tgui.mw.dialogs["force_load_options"].dia.show_large_file_warning_checkbox, on=False)
        assert tgui.mw.cfgs.file_load_options["show_large_file_warning"] is False

        assert tgui.mw.cfgs.file_load_options["importdata_always_offset_time"] == tgui.mw.dialogs[
            "force_load_options"].dia.always_correct_offset_checkbox.isChecked()
        tgui.switch_checkbox(tgui.mw.dialogs["force_load_options"].dia.always_correct_offset_checkbox, on=False)
        assert tgui.mw.cfgs.file_load_options["importdata_always_offset_time"] is False
        tgui.switch_checkbox(tgui.mw.dialogs["force_load_options"].dia.always_correct_offset_checkbox, on=True)
        assert tgui.mw.cfgs.file_load_options["importdata_always_offset_time"] is True

        # Save, quick and check correct are still shown.
        tgui = close_and_reload_for_defaults_check(tgui, save_="file_loading_options", force_current_clamp=True)

        assert tgui.mw.cfgs.file_load_options["force_load_options"] == "current_clamp"
        assert tgui.mw.dialogs["force_load_options"].dia.force_current_clamp_radiobutton.isChecked()

        assert tgui.mw.dialogs["force_load_options"].dia.select_channels_to_load_groupbox.isChecked()
        assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["on"]

        assert tgui.mw.cfgs.file_load_options["generate_axon_protocol"] is True
        assert tgui.mw.dialogs["force_load_options"].dia.generate_axon_protocol_checkbox.isChecked()
        assert tgui.mw.dialogs["force_load_options"].dia.secondary_data_channel_override_combobox.isEnabled() is False
        assert tgui.mw.dialogs["force_load_options"].dia.secondary_data_channel_override_combobox.currentIndex() == 0

        assert tgui.mw.cfgs.file_load_options["importdata_always_normalise_time"] is True
        assert tgui.mw.dialogs["force_load_options"].dia.always_normalise_time_checkbox.isChecked()

        assert tgui.mw.cfgs.file_load_options["show_large_file_warning"] is False
        assert not tgui.mw.dialogs["force_load_options"].dia.show_large_file_warning_checkbox.isChecked()

        assert tgui.mw.cfgs.file_load_options["importdata_always_offset_time"] is True
        assert tgui.mw.dialogs["force_load_options"].dia.always_correct_offset_checkbox.isChecked()

        tgui.shutdown()

    def test_select_channels_to_load(self, tgui):
        """
        Check that the widgets select the right channels to load.
        By default the user is prompted (when select channels to load is False).
        Otherwise the input channels to load are used. For primary this ranges from 1-10, for
        secondary there is also the option to not load a second channel ("None").
        """
        tgui.mw.mw.actionForce_Load_Options.trigger()
        assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["on"] is False

        channels_to_load_groupbox = tgui.mw.dialogs["force_load_options"].dia.select_channels_to_load_groupbox
        assert not channels_to_load_groupbox.isChecked()
        tgui.switch_groupbox(channels_to_load_groupbox, on=True)
        assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["on"]

        for idx in range(10):
            tgui.set_combobox(tgui.mw.dialogs["force_load_options"].dia.primary_data_chanel_override_combobox,
                              idx)
            assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["channel_1_idx"] == idx, str(idx) + " index first channel is wrong"

        for idx in range(11):
            tgui.set_combobox(tgui.mw.dialogs["force_load_options"].dia.secondary_data_channel_override_combobox,
                              idx)
            channel_idx = idx - 1 if idx != 0 else None
            assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"] == channel_idx, str(
                channel_idx) + " index second channel is wrong"

        tgui.switch_groupbox(channels_to_load_groupbox, on=False)
        assert tgui.mw.cfgs.file_load_options["select_channels_to_load"]["on"] is False

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # First file load configs - key config for proper table plotting
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_file_first_analysis_config(self, tgui):
        load_file(tgui)
        tgui.left_mouse_click(tgui.mw.mw.spike_count_button)
        assert tgui.mw.cfgs.main["file_first_analysis"]["spkcnt"] is False

        tgui.set_analysis_type("Ri")
        tgui.left_mouse_click(tgui.mw.mw.ir_calc_button)
        assert tgui.mw.cfgs.main["file_first_analysis"]["Ri"] is False

        tgui.set_analysis_type("skinetics")
        tgui.left_mouse_click(tgui.mw.mw.skinetics_auto_count_spikes_button)
        assert tgui.mw.cfgs.main["file_first_analysis"]["skinetics"] is False

        tgui.set_analysis_type("curve_fitting")
        tgui.left_mouse_click(tgui.mw.mw.curve_fitting_fit_selected_region_button)
        assert tgui.mw.cfgs.main["file_first_analysis"]["curve_fitting"] is False

        load_file(tgui, "voltage_clamp_1_record")
        assert tgui.mw.cfgs.main["file_first_analysis"]["spkcnt"] is True
        assert tgui.mw.cfgs.main["file_first_analysis"]["Ri"] is True
        assert tgui.mw.cfgs.main["file_first_analysis"]["skinetics"] is True
        assert tgui.mw.cfgs.main["file_first_analysis"]["curve_fitting"] is True

    def test_file_first_analysis_events_thresholding_config(self, tgui):
        """
        Need to do separately or previous test results in a C++ error...
        """
        load_file(tgui, "voltage_clamp_1_record")
        tgui.set_analysis_type("events_thresholding")
        tgui.left_mouse_click(tgui.mw.mw.events_threshold_analyse_events_button)
        tgui.left_mouse_click(tgui.mw.dialogs["events_threshold_analyse_events"].dia.fit_all_events_button)

        assert tgui.mw.cfgs.main["file_first_analysis"]["events"] is False

        load_file(tgui, "current_clamp")
        assert tgui.mw.cfgs.main["file_first_analysis"]["events"] is True

    def test_file_first_analysis_events_template_config(self, tgui):
        """
        Need to do separately or previous test results in a C++ error...
        """
        load_file(tgui, "voltage_clamp_1_record")
        tgui.set_analysis_type("events_template_matching")
        tgui.left_mouse_click(tgui.mw.mw.events_template_analyse_all_button)

        tgui.left_mouse_click(tgui.mw.dialogs["template_analyse_events"].dia.fit_all_events_button)

        QtWidgets.QApplication.processEvents()
        assert tgui.mw.cfgs.main["file_first_analysis"]["events"] is False

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    #  Misc Dialogs check - time_offset_dialog and misc_options_dialog
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_misc_options_dialog(self, tgui):
        tgui.mw.mw.actionMisc_Options.trigger()
        pass

    def test_pad_im_injection_protocol(self, tgui):
        """
        test above, below default and that if set to zero it will not allow
        """
        assert tgui.mw.cfgs.main["im_inj_protocol_padding"] == 10

        tgui.mw.mw.actionMisc_Options.trigger()
        assert tgui.mw.cfgs.main["im_inj_protocol_padding"] == tgui.mw.dialogs[
                                                                               "misc_options_dialog"].dia.pad_im_injection_protocol_spinbox.value()
        tgui.mw.dialogs["misc_options_dialog"].dia.pad_im_injection_protocol_spinbox.setValue(100)
        assert tgui.mw.cfgs.main["im_inj_protocol_padding"] == 100

        tgui = close_and_reload_for_defaults_check(tgui, save_="misc_options")
        assert tgui.mw.dialogs["misc_options_dialog"].dia.pad_im_injection_protocol_spinbox.value() == 100
        assert tgui.mw.cfgs.main["im_inj_protocol_padding"] == 100

        tgui.mw.dialogs["misc_options_dialog"].dia.pad_im_injection_protocol_spinbox.setValue(1)
        assert tgui.mw.cfgs.main["im_inj_protocol_padding"] == 1
        tgui.mw.dialogs["misc_options_dialog"].dia.pad_im_injection_protocol_spinbox.setValue(0)
        assert tgui.mw.cfgs.main["im_inj_protocol_padding"] == 1

        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Spike Count Configs
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_spkcnt_config_default_setting(self, tgui):
        """
        """
        assert tgui.mw.cfgs.spkcnt["analyse_within_bounds"] is False
        assert tgui.mw.cfgs.spkcnt["im_injection_protocol_start"] is None
        assert tgui.mw.cfgs.spkcnt["im_injection_protocol_stop"] is None
        assert tgui.mw.cfgs.spkcnt["threshold_type"] == "auto_record"
        assert tgui.mw.cfgs.spkcnt[
                   "threshold_type"] == "auto_record" if tgui.mw.mw.spikecnt_thr_combobox.currentIndex() == 0 else "manual" "ACTUAL TEST"
        assert tgui.mw.cfgs.spkcnt["table_im_combobox"] == "im_delta"
        assert tgui.mw.cfgs.spkcnt["table_im_combobox"] == "im_delta" if tgui.mw.mw.ir_im_opts_combobox.currentIndex == 0 else "im_delta_round"
        assert tgui.mw.cfgs.spkcnt["user_im_step"] == 0
        assert tgui.mw.cfgs.spkcnt["user_im_step_protocol_type"] == "increasing"
        assert tgui.mw.cfgs.spkcnt["man_thr_value"] is None
        assert tgui.mw.cfgs.spkcnt["im_groupbox_active"] is False
        assert tgui.mw.cfgs.spkcnt[
                   "im_on_for_most_recent_analysis"] is False  # important cfg, used to save im setting for most recent analysis for tabledata switching
        assert tgui.mw.cfgs.spkcnt["user_input_im"]["step"] is None
        assert tgui.mw.cfgs.spkcnt["im_opt_setting"] == "bounds"
        assert tgui.mw.cfgs.spkcnt["calc_fs_latency"] is False
        assert tgui.mw.cfgs.spkcnt["calc_mean_isi"] is False
        assert tgui.mw.cfgs.spkcnt["calc_spike_fa"] is False
        assert tgui.mw.cfgs.spkcnt["calc_rheobase"] is False
        assert tgui.mw.cfgs.spkcnt["calc_rheobase_method"] is "record"
        assert tgui.mw.cfgs.spkcnt["auto_thr_amp"] == 20
        assert tgui.mw.cfgs.spkcnt["auto_thr_rise"] == 20
        assert tgui.mw.cfgs.spkcnt["auto_thr_fall"] == -1
        assert tgui.mw.cfgs.spkcnt["auto_thr_width"] == 5

    def test_spkcnt_analyse_within_bounds_checkbox_config(self, tgui):
        load_file(tgui)
        tgui.switch_checkbox(tgui.mw.mw.spkcnt_set_bounds_checkbox,
                             on=False)
        assert tgui.mw.cfgs.spkcnt["analyse_within_bounds"] is False
        tgui.switch_checkbox(tgui.mw.mw.spkcnt_set_bounds_checkbox,
                             on=True)
        assert tgui.mw.cfgs.spkcnt["analyse_within_bounds"] is True

    def test_im_injection_protocol_config(self, tgui):
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=True,
                                                       im_setting="im_protocol")
        tgui.fill_im_injection_protocol_dialog(tgui.mw.mw.spkcnt_set_im_button,
                                               "0.25",
                                               "0.75")
        assert tgui.mw.cfgs.spkcnt["im_injection_protocol_start"] == 0.25
        assert tgui.mw.cfgs.spkcnt["im_injection_protocol_stop"] == 0.75

    def test_spkcnt_spike_threshold_widgets_config(self, tgui):
        """
        For spkcnt threshold type and "man_thr_value"
        """
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True)
        self.check_spikecount_threshold_combobox_spinbox_configs(tgui,
                                                                 analysis_cfg=tgui.mw.cfgs.spkcnt,
                                                                 analysis_combobox=tgui.mw.mw.spikecnt_thr_combobox,
                                                                 spinbox=tgui.mw.mw.spkcnt_man_thr_spinbox)

        assert tgui.mw.mw.spkcnt_im_opts_combobox.isEnabled() is False, "W"
        tgui.left_mouse_click(
            tgui.mw.mw.spike_count_button)
        assert tgui.mw.mw.spkcnt_im_opts_combobox.isEnabled() is True, "Y"

    def test_spkcnt_table_im_options_config(self, tgui):
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True)
        tgui.left_mouse_click(tgui.mw.mw.spike_count_button)
        tgui.set_combobox(tgui.mw.mw.spkcnt_im_opts_combobox, idx=0)
        assert tgui.mw.cfgs.spkcnt["table_im_combobox"] == "im_delta"

    @pytest.mark.parametrize("protocol_type", ["decreasing", "increasing", "repeating"])
    def test_spkcnt_table_im_round_config(self, tgui, protocol_type):
        """
        Setup user im round dialog on tablewidget that has user specify the Im input for rounding.
        Check protocol type (increasing, repeating, decreasing) and inputting a value.
        """
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True)
        tgui.left_mouse_click(tgui.mw.mw.spike_count_button)
        tgui.set_combobox(tgui.mw.mw.spkcnt_im_opts_combobox,
                          idx=1)

        self.click_user_im_input_toggle(tgui, protocol_type)
        tgui.mw.dialogs["user_im_round"].dia.im_round_input.setValue(10)

        tgui.mw.dialogs["user_im_round"].accept()
        assert tgui.mw.cfgs.spkcnt["user_im_step_protocol_type"] == protocol_type
        assert tgui.mw.cfgs.spkcnt["table_im_combobox"] == "im_delta_round"
        assert tgui.mw.cfgs.spkcnt["user_im_step"] == 10

    def test_spkcnt_im_groupbox_active_config(self, tgui):
        assert tgui.mw.cfgs.spkcnt["im_groupbox_active"] is False
        tgui.switch_groupbox(tgui.mw.mw.spkcnt_im_groupbox,
                             on=True)
        assert tgui.mw.cfgs.spkcnt["im_groupbox_active"] is True

    def test_im_on_for_most_recent_analysis_config(self, tgui):
        """
        """
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=False)
        tgui.left_mouse_click(tgui.mw.mw.spike_count_button)
        assert tgui.mw.cfgs.spkcnt["im_on_for_most_recent_analysis"] is False
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=True)
        tgui.left_mouse_click(tgui.mw.mw.spike_count_button)
        assert tgui.mw.cfgs.spkcnt["im_on_for_most_recent_analysis"] is True

    def test_spkcnt_user_input_im_config(self, tgui):
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True,
                                                       im_setting="user_input_im")
        assert tgui.mw.cfgs.spkcnt["user_input_im"]["step"] is None
        tgui.fill_user_im_input_widget(tgui.num_recs,
                                       tgui.mw.mw.spkcnt_set_im_button)
        assert np.all(tgui.mw.cfgs.spkcnt["user_input_im"]["step"] == [[x] for x in range(tgui.num_recs)])

    def test_spkcnt_im_combobox_configs(self, tgui):
        load_file(tgui)
        for bound_setting in ["bounds", "im_protocol", "user_input_im"]:
            tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                           im_groupbox_on=True,
                                                           im_setting=bound_setting)
            assert tgui.mw.cfgs.spkcnt["im_opt_setting"] == bound_setting

    def test_fs_latency_checkbox_config(self, tgui):
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False, im_groupbox_on=False, more_analysis=True)
        tgui.switch_checkbox(tgui.mw.spkcnt_popup.dia.fs_latency_checkbox, on=True)
        tgui.mw.dialogs["im_inj_protocol"].accept()
        assert tgui.mw.cfgs.spkcnt["calc_fs_latency"] is True
        tgui.switch_checkbox(tgui.mw.spkcnt_popup.dia.fs_latency_checkbox, on=False)
        assert tgui.mw.cfgs.spkcnt["calc_fs_latency"] is False

    def test_mean_isi_checkbox_config(self, tgui):
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False, im_groupbox_on=False, more_analysis=True)
        tgui.switch_checkbox(tgui.mw.spkcnt_popup.dia.mean_isi_checkbox, on=True)
        assert tgui.mw.cfgs.spkcnt["calc_mean_isi"] is True
        tgui.switch_checkbox(tgui.mw.spkcnt_popup.dia.mean_isi_checkbox, on=False)
        assert tgui.mw.cfgs.spkcnt["calc_mean_isi"] is False

    def test_calc_spike_fa(self, tgui):
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False, im_groupbox_on=False, more_analysis=True)
        assert tgui.mw.cfgs.spkcnt["calc_spike_fa"] is False
        tgui.left_mouse_click(tgui.mw.spkcnt_popup.dia.spike_freq_accommodation_checkbox)
        assert tgui.mw.cfgs.spkcnt["calc_spike_fa"] is True
        tgui.left_mouse_click(tgui.mw.spkcnt_popup.dia.spike_freq_accommodation_checkbox)
        assert tgui.mw.cfgs.spkcnt["calc_spike_fa"] is False

    def test_rheobase_configs(self, tgui):
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False, im_groupbox_on=True, more_analysis=True)
        assert tgui.mw.cfgs.spkcnt["calc_rheobase"] is False
        tgui.switch_groupbox(tgui.mw.spkcnt_popup.dia.spkcnt_rheobase_groupbox, on=True)
        assert tgui.mw.cfgs.spkcnt["calc_rheobase"] is True
        tgui.left_mouse_click(tgui.mw.spkcnt_popup.dia.spkcnt_rheobase_record_radiobutton)
        assert tgui.mw.cfgs.spkcnt["calc_rheobase_method"] == "record"
        tgui.switch_checkbox(tgui.mw.spkcnt_popup.dia.spkcnt_rheobase_exact_radiobutton, on=True)
        assert tgui.mw.cfgs.spkcnt["calc_rheobase_method"] == "exact"

    def test_spikecount_autodetect_parameters(self, tgui):
        """
        Hardcoded. Note nice because slider is c++ error must be called fully so be outside of this old function.
        """
        tgui.mw.mw.actionSpike_Detection_Options.trigger()
        invert = self.check_spike_count_options_dialog(tgui,
                                                       slider=tgui.mw.dialogs["spike_counting_options"].dia.spkcnt_amp_slider,
                                                       spinbox=tgui.mw.dialogs["spike_counting_options"].dia.amp_spinbox,
                                                       cfg_dict_key="auto_thr_amp")

        tgui = close_and_reload_for_defaults_check(tgui, save_="spkcnt_options")
        assert tgui.mw.dialogs["spike_counting_options"].dia.spkcnt_amp_slider.value() == 50 * invert
        assert tgui.mw.cfgs.spkcnt["auto_thr_amp"] == 5.0 * invert

        invert = self.check_spike_count_options_dialog(tgui,
                                                       slider=tgui.mw.dialogs["spike_counting_options"].dia.spkcnt_rise_slider,
                                                       spinbox=tgui.mw.dialogs["spike_counting_options"].dia.rise_spinbox,
                                                       cfg_dict_key="auto_thr_rise")

        tgui = close_and_reload_for_defaults_check(tgui, save_="spkcnt_options")
        assert tgui.mw.dialogs["spike_counting_options"].dia.spkcnt_rise_slider.value() == 50 * invert
        assert tgui.mw.cfgs.spkcnt["auto_thr_rise"] == 5.0 * invert

        invert = self.check_spike_count_options_dialog(tgui,
                                                       slider=tgui.mw.dialogs["spike_counting_options"].dia.spkcnt_fall_slider,
                                                       spinbox=tgui.mw.dialogs["spike_counting_options"].dia.fall_spinbox,
                                                       cfg_dict_key="auto_thr_fall",
                                                       fall=True)

        tgui = close_and_reload_for_defaults_check(tgui, save_="spkcnt_options")
        assert tgui.mw.dialogs["spike_counting_options"].dia.spkcnt_fall_slider.value() == 50 * invert
        assert tgui.mw.cfgs.spkcnt["auto_thr_fall"] == 5.0 * invert

        invert = self.check_spike_count_options_dialog(tgui,
                                                       slider=tgui.mw.dialogs["spike_counting_options"].dia.spkcnt_width_slider,
                                                       spinbox=tgui.mw.dialogs["spike_counting_options"].dia.width_spinbox,
                                                       cfg_dict_key="auto_thr_width")

        tgui = close_and_reload_for_defaults_check(tgui, save_="spkcnt_options")
        assert tgui.mw.dialogs["spike_counting_options"].dia.spkcnt_width_slider.value() == 50 * invert
        assert tgui.mw.cfgs.spkcnt["auto_thr_width"] == 5.0 * invert

        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Input Resistance Configs
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # many of these test structures are similar to spkcnt but kept seperate for now rather than explicity for readability and flexibility at the expense of verbosity.

    def test_ir_config_default_setting(self, tgui):  # and from widget
        """
        """
        assert tgui.mw.cfgs.ir["im_injection_protocol_start"] is None
        assert tgui.mw.cfgs.ir["im_injection_protocol_stop"] is None
        assert tgui.mw.cfgs.ir["im_opt_setting"] == "bounds"
        assert tgui.mw.cfgs.ir["user_input_im"]["step"] is None
        assert tgui.mw.cfgs.ir["table_im_combobox"] == "im_delta"
        assert tgui.mw.cfgs.ir["user_im_step"] == 0
        assert tgui.mw.cfgs.ir["user_im_step_protocol_type"] == "increasing"
        assert tgui.mw.cfgs.ir["calculate_sag"] is False
        assert tgui.mw.cfgs.ir["calculate_sag"] == tgui.mw.mw.ir_calculate_sag_groupbox.isChecked()
        assert tgui.mw.cfgs.ir["sag_hump_start_time"] == 0.00
        assert tgui.mw.cfgs.ir["sag_hump_start_time"] == tgui.mw.mw.ir_sag_hump_start_spinbox.value()
        assert tgui.mw.cfgs.ir["sag_hump_stop_time"] == 0.00
        assert tgui.mw.cfgs.ir["sag_hump_stop_time"] == tgui.mw.mw.ir_sag_hump_stop_spinbox.value()
        assert tgui.mw.cfgs.ir["sag_hump_peak_direction"] == "follow_im"

    def test_ir_im_protocol_config(self, tgui):
        load_file(tgui)
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting="im_protocol")
        tgui.fill_im_injection_protocol_dialog(tgui.mw.mw.ir_set_im_button, "0.25", "0.75")
        assert tgui.mw.cfgs.ir["im_injection_protocol_start"] == 0.25
        assert tgui.mw.cfgs.ir["im_injection_protocol_stop"] == 0.75

    def test_ir_im_setting_config(self, tgui):
        load_file(tgui)
        for bound_setting in ["bounds", "im_protocol", "user_input_im"]:
            tgui.switch_to_input_resistance_and_set_im_combobox(im_setting=bound_setting)
            assert tgui.mw.cfgs.ir["im_opt_setting"] == bound_setting

    def test_ir_im_user_input_protocol(self, tgui):
        load_file(tgui)
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting="user_input_im")
        assert tgui.mw.cfgs.ir["user_input_im"]["step"] is None
        tgui.fill_user_im_input_widget(tgui.num_recs, tgui.mw.mw.ir_set_im_button)
        assert np.all(tgui.mw.cfgs.ir["user_input_im"]["step"] == [[x] for x in range(tgui.num_recs)])

    def test_ir_table_im_avg_config(self, tgui):
        load_file(tgui)
        tgui.switch_to_input_resistance_and_set_im_combobox()
        tgui.left_mouse_click(tgui.mw.mw.ir_calc_button)
        tgui.set_combobox(tgui.mw.mw.ir_im_opts_combobox, idx=0)
        assert tgui.mw.cfgs.ir["table_im_combobox"] == "im_delta"

    @pytest.mark.parametrize("protocol_type", ["increasing", "decreasing", "repeating"])
    def test_ir_table_im_round_config(self, tgui, protocol_type):
        load_file(tgui)
        tgui.switch_to_input_resistance_and_set_im_combobox()
        tgui.left_mouse_click(tgui.mw.mw.ir_calc_button)
        tgui.set_combobox(tgui.mw.mw.ir_im_opts_combobox, idx=1)

        self.click_user_im_input_toggle(tgui, protocol_type)
        tgui.mw.dialogs["user_im_round"].dia.im_round_input.setValue(10)

        tgui.mw.dialogs["user_im_round"].accept()
        assert tgui.mw.cfgs.ir["table_im_combobox"] == "im_delta_round"
        assert tgui.mw.cfgs.ir["user_im_step_protocol_type"] == protocol_type
        assert tgui.mw.cfgs.ir["user_im_step"] == 10

    def test_ir_calculate_sag_configs(self, tgui):
        load_file(tgui)
        tgui.switch_groupbox(tgui.mw.mw.ir_calculate_sag_groupbox, on=True)
        assert tgui.mw.cfgs.ir["calculate_sag"] is True

        tgui.mw.mw.ir_sag_hump_start_spinbox.setValue(0.50)
        assert tgui.mw.cfgs.ir["sag_hump_start_time"] == 0.50

        tgui.mw.mw.ir_sag_hump_stop_spinbox.setValue(1.00)
        assert tgui.mw.cfgs.ir["sag_hump_stop_time"] == 1.00

    def test_sag_hump_peak_direction(self, tgui):
        load_file(tgui)
        tgui.switch_to_input_resistance_and_set_im_combobox()
        tgui.switch_groupbox(tgui.mw.mw.ir_calculate_sag_groupbox, on=True)
        tgui.set_combobox(tgui.mw.mw.ri_sag_hump_combobox, 1)
        assert tgui.mw.cfgs.ir["sag_hump_peak_direction"] == "min"
        tgui.set_combobox(tgui.mw.mw.ri_sag_hump_combobox, 2)
        assert tgui.mw.cfgs.ir["sag_hump_peak_direction"] == "max"
        keyPress(tgui.mw.mw.ri_sag_hump_combobox, QtGui.Qt.Key_Up)
        keyPress(tgui.mw.mw.ri_sag_hump_combobox, QtGui.Qt.Key_Up)
        assert tgui.mw.cfgs.ir["sag_hump_peak_direction"] == "follow_im"

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # sKinetics Configs
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_skinetics_config_default_setting(self, tgui):
        assert tgui.mw.cfgs.skinetics["analyse_within_bounds"] is False
        assert tgui.mw.cfgs.skinetics["analyse_within_bounds"] == tgui.mw.mw.skinetics_set_bounds_checkbox.isChecked()
        assert tgui.mw.cfgs.skinetics["hide_label"] is False
        assert tgui.mw.cfgs.skinetics["hide_label"] == tgui.mw.mw.skinetics_hide_label.isChecked()
        assert tgui.mw.cfgs.skinetics["man_thr_value"] is None
        assert tgui.mw.mw.skinetics_man_thr_spinbox.value() == 0.00
        assert tgui.mw.cfgs.skinetics["threshold_type"] == "auto_record"
        assert tgui.mw.cfgs.skinetics[
                   "threshold_type"] == "auto_record" if tgui.mw.mw.skinetics_thr_combobox.currentIndex() == 0 else "manual" "ACTUAL TEST"

    def test_skinetics_panel_checkbox_config(self, tgui):
        load_file(tgui)
        tgui.set_analysis_type("skinetics")
        assert tgui.mw.cfgs.skinetics["analyse_within_bounds"] is False
        tgui.switch_checkbox(tgui.mw.mw.skinetics_set_bounds_checkbox, on=True)
        assert tgui.mw.cfgs.skinetics["analyse_within_bounds"] is True

        assert tgui.mw.cfgs.skinetics["hide_label"] is False
        tgui.switch_checkbox(tgui.mw.mw.skinetics_hide_label,
                             on=True)
        assert tgui.mw.cfgs.skinetics["hide_label"] is True

    def test_skinetics_click_spike_mode_button(self, tgui):
        load_file(tgui)
        assert tgui.mw.cfgs.main["click_spike_mode"] is False
        tgui.left_mouse_click(tgui.mw.mw.skinetics_click_mode_button)
        assert tgui.mw.cfgs.main["click_spike_mode"] is True
        tgui.left_mouse_click(tgui.mw.mw.skinetics_click_mode_button)
        assert tgui.mw.cfgs.main["click_spike_mode"] is False

    def test_skinetics_spike_threshold_widgets_config(self, tgui):
        """
        Tests "threshold_type" and "man_thr_value"
        """
        load_file(tgui)
        tgui.set_analysis_type("skinetics")
        self.check_spikecount_threshold_combobox_spinbox_configs(tgui,
                                                                 analysis_cfg=tgui.mw.cfgs.skinetics,
                                                                 analysis_combobox=tgui.mw.mw.skinetics_thr_combobox,
                                                                 spinbox=tgui.mw.mw.skinetics_man_thr_spinbox)

    def test_skinetics_options_gui_match_config(self, tgui):
        load_file(tgui)
        tgui.set_analysis_type("skinetics")
        tgui.left_mouse_click(tgui.mw.mw.skinetics_options_button)

        thr_method = self.get_thr_method_from_combobox_idx(tgui)
        assert tgui.mw.cfgs.skinetics["thr_method"] == thr_method
        cutoff = "max" if tgui.mw.dialogs["skinetics_options"].dia.first_deriv_max_radiobutton.isChecked() else "cutoff"
        assert tgui.mw.cfgs.skinetics["first_deriv_max_or_cutoff"] == cutoff
        max = "max" if tgui.mw.dialogs["skinetics_options"].dia.third_deriv_max_radiobutton.isChecked() else "cutoff"
        assert tgui.mw.cfgs.skinetics["third_deriv_max_or_cutoff"] == max
        assert tgui.mw.cfgs.skinetics["first_deriv_cutoff"] == tgui.mw.dialogs["skinetics_options"].dia.first_deriv_cutoff_spinbox.value()
        assert tgui.mw.cfgs.skinetics["third_deriv_cutoff"] == tgui.mw.dialogs["skinetics_options"].dia.third_deriv_cutoff_spinbox.value()
        assert tgui.mw.cfgs.skinetics["method_I_lower_bound"] == tgui.mw.dialogs["skinetics_options"].dia.method_I_lower_bound_spinbox.value()
        assert tgui.mw.cfgs.skinetics["method_II_lower_bound"] == tgui.mw.dialogs["skinetics_options"].dia.method_II_lower_bound_spinbox.value()
        assert tgui.mw.cfgs.skinetics["interp_200khz"] == tgui.mw.dialogs["skinetics_options"].dia.interp_200khz_checkbox.isChecked()
        assert tgui.mw.cfgs.skinetics["decay_to_thr_not_fahp"] == tgui.mw.dialogs[
            "skinetics_options"].dia.decay_time_from_threshold_checkbox.isChecked()
        assert tgui.mw.cfgs.skinetics["rise_cutoff_low"] == tgui.mw.dialogs["skinetics_options"].dia.rise_time_cutoff_low.value()
        assert tgui.mw.cfgs.skinetics["rise_cutoff_high"] == tgui.mw.dialogs["skinetics_options"].dia.rise_time_cutoff_high.value()
        assert tgui.mw.cfgs.skinetics["decay_cutoff_low"] == tgui.mw.dialogs["skinetics_options"].dia.decay_time_cutoff_low.value()
        assert tgui.mw.cfgs.skinetics["decay_cutoff_high"] == tgui.mw.dialogs["skinetics_options"].dia.decay_time_cutoff_high.value()
        assert tgui.mw.cfgs.skinetics["fahp_start"] == tgui.mw.dialogs["skinetics_options"].dia.fahp_start.value()
        assert tgui.mw.cfgs.skinetics["fahp_stop"] == tgui.mw.dialogs["skinetics_options"].dia.fahp_stop.value() / 1000
        assert tgui.mw.cfgs.skinetics["mahp_start"] == tgui.mw.dialogs["skinetics_options"].dia.mahp_start.value() / 1000
        assert tgui.mw.cfgs.skinetics["mahp_stop"] == tgui.mw.dialogs["skinetics_options"].dia.mahp_stop.value() / 1000
        assert tgui.mw.cfgs.skinetics["search_region_min"] == tgui.mw.dialogs["skinetics_options"].dia.skinetics_search_region_min.value() / 1000

    def test_skinetics_deriv_options_config(self, tgui):
        load_file(tgui)
        tgui.set_analysis_type("skinetics")
        tgui.left_mouse_click(tgui.mw.mw.skinetics_options_button)  # no defaults because user can override - just check against gui

        tgui.mw.dialogs["skinetics_options"].dia.skinetics_search_region_min.setValue(5)
        assert tgui.mw.cfgs.skinetics["search_region_min"] == 5 / 1000

        tgui.set_combobox(tgui.mw.dialogs["skinetics_options"].dia.skinetics_method_combobox, idx=0)
        assert tgui.mw.cfgs.skinetics["thr_method"] == "first_deriv"
        tgui.mw.dialogs["skinetics_options"].dia.first_deriv_max_radiobutton.setChecked(True)
        assert tgui.mw.cfgs.skinetics["first_deriv_max_or_cutoff"] == "max"
        tgui.mw.dialogs["skinetics_options"].dia.first_deriv_cutoff_radiobutton.setChecked(True)
        assert tgui.mw.cfgs.skinetics["first_deriv_max_or_cutoff"] == "cutoff"
        tgui.mw.dialogs["skinetics_options"].dia.first_deriv_cutoff_spinbox.setValue(0.01)
        assert tgui.mw.cfgs.skinetics["first_deriv_cutoff"] == 0.01
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 0

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert not tgui.mw.dialogs["skinetics_options"].dia.first_deriv_max_radiobutton.isChecked()
        assert tgui.mw.dialogs["skinetics_options"].dia.first_deriv_cutoff_radiobutton.isChecked()
        assert tgui.mw.cfgs.skinetics["first_deriv_max_or_cutoff"] == "cutoff"
        assert tgui.mw.cfgs.skinetics["first_deriv_cutoff"] == 0.01
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 0

        tgui.set_combobox(tgui.mw.dialogs["skinetics_options"].dia.skinetics_method_combobox, idx=1)
        assert tgui.mw.cfgs.skinetics["thr_method"] == "third_deriv"  # USEFUL FOR SWITCHING PUT INTO OWN METHOD?
        tgui.mw.dialogs["skinetics_options"].dia.third_deriv_max_radiobutton.setChecked(True)
        assert tgui.mw.cfgs.skinetics["third_deriv_max_or_cutoff"] == "max"
        tgui.mw.dialogs["skinetics_options"].dia.third_deriv_cutoff_radiobutton.setChecked(True)
        assert tgui.mw.cfgs.skinetics["third_deriv_max_or_cutoff"] == "cutoff"
        tgui.mw.dialogs["skinetics_options"].dia.third_deriv_cutoff_spinbox.setValue(0.01)
        assert tgui.mw.cfgs.skinetics["third_deriv_cutoff"] == 0.01
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 1

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")

        assert tgui.mw.cfgs.skinetics["search_region_min"] == 5 / 1000
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_search_region_min.value() == 5

        assert not tgui.mw.dialogs["skinetics_options"].dia.third_deriv_max_radiobutton.isChecked()
        assert tgui.mw.dialogs["skinetics_options"].dia.third_deriv_cutoff_radiobutton.isChecked()
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 1
        assert tgui.mw.dialogs["skinetics_options"].dia.third_deriv_cutoff_spinbox.value() == 0.01
        assert tgui.mw.cfgs.skinetics["third_deriv_max_or_cutoff"] == "cutoff"
        assert tgui.mw.cfgs.skinetics["third_deriv_cutoff"] == 0.01

        tgui.set_combobox(tgui.mw.dialogs["skinetics_options"].dia.skinetics_method_combobox, idx=2)
        assert tgui.mw.cfgs.skinetics["thr_method"] == "method_I"
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 2
        tgui.mw.dialogs["skinetics_options"].dia.method_I_lower_bound_spinbox.setValue(0.01)
        assert tgui.mw.cfgs.skinetics["method_I_lower_bound"] == 0.01

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 2
        assert tgui.mw.dialogs["skinetics_options"].dia.method_I_lower_bound_spinbox.value() == 0.01
        assert tgui.mw.cfgs.skinetics["thr_method"] == "method_I"
        assert tgui.mw.cfgs.skinetics["method_I_lower_bound"] == 0.01

        tgui.set_combobox(tgui.mw.dialogs["skinetics_options"].dia.skinetics_method_combobox, idx=3)
        assert tgui.mw.cfgs.skinetics["thr_method"] == "method_II"
        tgui.mw.dialogs["skinetics_options"].dia.method_II_lower_bound_spinbox.setValue(0.01)
        assert tgui.mw.cfgs.skinetics["method_II_lower_bound"] == 0.01
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 3

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 3
        assert tgui.mw.dialogs["skinetics_options"].dia.method_II_lower_bound_spinbox.value() == 0.01
        assert tgui.mw.cfgs.skinetics["thr_method"] == "method_II"
        assert tgui.mw.cfgs.skinetics["method_II_lower_bound"] == 0.01

        tgui.set_combobox(tgui.mw.dialogs["skinetics_options"].dia.skinetics_method_combobox, idx=4)
        assert tgui.mw.cfgs.skinetics["thr_method"] == "leading_inflection"
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 4

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.cfgs.skinetics["thr_method"] == "leading_inflection"
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 4

        tgui.set_combobox(tgui.mw.dialogs["skinetics_options"].dia.skinetics_method_combobox, idx=5)
        assert tgui.mw.cfgs.skinetics["thr_method"] == "max_curvature"
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 4

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.cfgs.skinetics["thr_method"] == "max_curvature"
        assert tgui.mw.dialogs["skinetics_options"].dia.skinetics_options_stackwidget.currentIndex() == 4

        tgui.shutdown()

    def test_interp_200khz_checkbox(self, tgui):
        load_file(tgui)
        tgui.set_analysis_type("skinetics")
        tgui.left_mouse_click(tgui.mw.mw.skinetics_options_button)  # no defaults because user can override - just check against gui

        tgui.switch_checkbox(tgui.mw.dialogs["skinetics_options"].dia.interp_200khz_checkbox, on=True)
        assert tgui.mw.cfgs.skinetics["interp_200khz"] is True

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.interp_200khz_checkbox.isChecked()
        assert tgui.mw.cfgs.skinetics["interp_200khz"] is True

        tgui.switch_checkbox(tgui.mw.dialogs["skinetics_options"].dia.interp_200khz_checkbox, on=False)
        assert tgui.mw.cfgs.skinetics["interp_200khz"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert not tgui.mw.dialogs["skinetics_options"].dia.interp_200khz_checkbox.isChecked()
        assert tgui.mw.cfgs.skinetics["interp_200khz"] is False

        tgui.shutdown()

    def test_decay_to_thr_not_fahp(self, tgui):
        load_file(tgui)
        tgui.set_analysis_type("skinetics")
        tgui.left_mouse_click(tgui.mw.mw.skinetics_options_button)  # no defaults because user can override - just check against gui

        tgui.switch_checkbox(tgui.mw.dialogs["skinetics_options"].dia.decay_time_from_threshold_checkbox, on=True)
        assert tgui.mw.cfgs.skinetics["decay_to_thr_not_fahp"] is True

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.decay_time_from_threshold_checkbox.isChecked()
        assert tgui.mw.cfgs.skinetics["decay_to_thr_not_fahp"] is True

        tgui.switch_checkbox(tgui.mw.dialogs["skinetics_options"].dia.decay_time_from_threshold_checkbox, on=False)
        assert tgui.mw.cfgs.skinetics["decay_to_thr_not_fahp"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert not tgui.mw.dialogs["skinetics_options"].dia.decay_time_from_threshold_checkbox.isChecked()
        assert tgui.mw.cfgs.skinetics["decay_to_thr_not_fahp"] is False

        tgui.shutdown()

    def test_rise_time_spinbox_configs(self, tgui):
        load_file(tgui)
        tgui.set_analysis_type("skinetics")
        tgui.left_mouse_click(tgui.mw.mw.skinetics_options_button)

        tgui.mw.dialogs["skinetics_options"].dia.rise_time_cutoff_low.setValue(5)
        assert tgui.mw.cfgs.skinetics["rise_cutoff_low"] == 5

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.rise_time_cutoff_low.value() == 5
        assert tgui.mw.cfgs.skinetics["rise_cutoff_low"] == 5

        tgui.mw.dialogs["skinetics_options"].dia.rise_time_cutoff_high.setValue(6)
        assert tgui.mw.cfgs.skinetics["rise_cutoff_high"] == 6

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.rise_time_cutoff_high.value() == 6
        assert tgui.mw.cfgs.skinetics["rise_cutoff_high"] == 6

        tgui.shutdown()

    def test_decay_time_spinbox_configs(self, tgui):
        load_file(tgui)
        tgui.set_analysis_type("skinetics")  # THIS INTO OWN METHOD
        tgui.left_mouse_click(tgui.mw.mw.skinetics_options_button)

        tgui.mw.dialogs["skinetics_options"].dia.decay_time_cutoff_low.setValue(5)
        assert tgui.mw.cfgs.skinetics["decay_cutoff_low"] == 5

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.decay_time_cutoff_low.value() == 5
        assert tgui.mw.cfgs.skinetics["decay_cutoff_low"] == 5

        tgui.mw.dialogs["skinetics_options"].dia.decay_time_cutoff_high.setValue(6)
        assert tgui.mw.cfgs.skinetics["decay_cutoff_high"] == 6

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.decay_time_cutoff_high.value() == 6
        assert tgui.mw.cfgs.skinetics["decay_cutoff_high"] == 6

        tgui.shutdown()

    def test_fahp_spinbox_configs(self, tgui):
        load_file(tgui)
        tgui.set_analysis_type("skinetics")
        tgui.left_mouse_click(tgui.mw.mw.skinetics_options_button)

        tgui.mw.dialogs["skinetics_options"].dia.fahp_start.setValue(2)  # set stop first as start must be before stop
        assert tgui.mw.cfgs.skinetics["fahp_start"] == 2 / 1000

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.fahp_start.value() == 2
        assert tgui.mw.cfgs.skinetics["fahp_start"] == 2 / 1000

        tgui.mw.dialogs["skinetics_options"].dia.fahp_stop.setValue(99)
        assert tgui.mw.cfgs.skinetics["fahp_stop"] == 99 / 1000

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.fahp_stop.value() == 99
        assert tgui.mw.cfgs.skinetics["fahp_stop"] == 99 / 1000

        tgui.shutdown()

    def test_mahp_spinbox_configs(self, tgui):
        load_file(tgui)
        tgui.set_analysis_type("skinetics")
        tgui.left_mouse_click(tgui.mw.mw.skinetics_options_button)

        tgui.mw.dialogs["skinetics_options"].dia.mahp_stop.setValue(99)
        assert tgui.mw.cfgs.skinetics["mahp_stop"] == 99 / 1000

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.mahp_stop.value() == 99
        assert tgui.mw.cfgs.skinetics["mahp_stop"] == 99 / 1000

        tgui.mw.dialogs["skinetics_options"].dia.mahp_start.setValue(1)
        assert tgui.mw.cfgs.skinetics["mahp_start"] == 1 / 1000

        tgui = close_and_reload_for_defaults_check(tgui, save_="skinetics_options")
        assert tgui.mw.dialogs["skinetics_options"].dia.mahp_start.value() == 1
        assert tgui.mw.cfgs.skinetics["mahp_start"] == 1 / 1000

        tgui.shutdown()

    def test_skinetics_params(self, tgui):
        """
        quick check to confirm all skinetic params are present in base dict
        """
        skinetics_params = tgui.mw.cfgs.skinetics_params()
        assert list(skinetics_params.keys()) == ["peak", "amplitude", "thr", "rise_time", "decay_time", "fwhm", "fahp", "mahp", "max_rise", "max_decay"]
        for key in skinetics_params.keys():
            assert skinetics_params[key] is None, "error in skinetics_params - " + key

        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    #  Track Plot Widgets Test
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_track_plot_widgets_cfgs(self, tgui):
        assert tgui.mw.cfgs.track_plot_widgets["spktcnt_counted_spikes"] is False
        assert tgui.mw.cfgs.track_plot_widgets["upperplot_background_click"] == 0

    def test_track_plot_widgets_cfgs_moving(self, tgui):
        load_file(tgui)
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=False)
        tgui.left_mouse_click(tgui.mw.mw.spike_count_button)
        assert tgui.mw.cfgs.track_plot_widgets["spktcnt_counted_spikes"] is True

    def test_upperplot_background_click(self, tgui):
        load_file(tgui)  # https://stackoverflow.com/questions/52429399/pyside2-how-to-get-mouse-position, https://stackoverflow.com/questions/16138040/how-to-convert-qpointf-to-qpoint-in-pyqt

        while True:  # switch recs until more than two spikes are found
            rec = tgui.mw.cfgs.main["displayed_rec"]
            if tgui.adata.spikes_per_rec[rec] < 2:
                tgui.mw.update_displayed_rec(rec + 1)
            else:
                break

        tgui.left_mouse_click(tgui.mw.mw.spike_count_button)
        plot = tgui.mw.loaded_file_plot.spkcnt_plot
        blue = "#0000ff"
        red = "#ff0000"

        # check color of first spike is red
        assert tgui.get_spotitem_color(plot, 0) == red

        # click on first spike and check color turns blue
        tgui.click_upperplot_spotitem(plot, 0)
        assert tgui.get_spotitem_color(plot, 0) == blue
        assert tgui.mw.cfgs.track_plot_widgets["upperplot_background_click"] == 1

        # click second spike and check color of first spike turns red and second blue
        tgui.click_upperplot_spotitem(plot, 1)
        assert tgui.get_spotitem_color(plot, 0) == red
        assert tgui.mw.cfgs.track_plot_widgets["upperplot_background_click"] == 1
        assert tgui.get_spotitem_color(plot, 1) == blue

        # click in the centre of the plot and check all colors dissapear (mouse not supported on macos)
        if platform != "darwin":  # mouse doesn't support macos
            self.click_in_middle_of_layout_widget(tgui.mw.loaded_file_plot.graphics_layout_widget,
                                                  tgui.mw.loaded_file_plot.upperplot)
            assert tgui.mw.cfgs.track_plot_widgets["upperplot_background_click"] == 0  # CLICK ON PLOT AND CHECK - GOOD TIME TO EMULATE
            assert tgui.get_spotitem_color(plot, 1) == red

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Test Curve Fitting
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_curve_fitting_config_default_setting(self, tgui):
        """
        Test all default values on curve fitting configs
        """
        assert tgui.mw.cfgs.curve_fitting_region_info["names"] == ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]
        assert tgui.mw.cfgs.curve_fitting_region_info["colors"] == ["red", "blue", "green", "orange", "pink", "teal"]

        assert tgui.mw.cfgs.curve_fitting["currently_selected_region"] == "reg_1"
        assert tgui.mw.cfgs.curve_fitting["region_idx"] == 0

        for region_name in ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]:
            assert tgui.mw.cfgs.curve_fitting["analysis_type_stackedwidget_idx"][region_name] == 0
            assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["region"] == "show_region"
            assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["hide_event_fit"] is False
            assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["link_baseline"] is True
        assert tgui.mw.cfgs.curve_fitting["region_display"]["show_only_selected_region"] is False

        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_peak_points"]["on"] is False
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_peak_points"]["value_s"] == 0.0005
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_baseline_points"]["on"] is False
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_baseline_points"]["value_s"] == 0.005
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["baseline_search_period_s"] == 0.01
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["baseline_type"] == "per_event"
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["from_fit_not_data"] is False
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["decay_period_type"] == "use_end_of_region"

        for region_name in ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]:

            for analysis in ["monoexp", "biexp_decay", "biexp_event", "triexp"]:
                assert tgui.mw.cfgs.curve_fitting["analysis"][region_name][analysis]["direction"] == 1
                assert tgui.mw.cfgs.curve_fitting["analysis"][region_name][analysis]["set_coefficients"] is False
                assert tgui.mw.cfgs.curve_fitting["analysis"][region_name][analysis]["set_bounds"] is False

            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["initial_est"]["b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["initial_est"]["b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["initial_est"]["tau"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["start_b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["start_b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["start_tau"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["stop_b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["stop_b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["stop_tau"] == 0

            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["tau1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["b2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["tau2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_tau1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_b2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_tau2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_tau1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_b2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_tau2"] == 0

            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["initial_est"]["b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["initial_est"]["b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["initial_est"]["rise"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["initial_est"]["decay"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["start_b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["start_b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["start_rise"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["start_decay"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["stop_b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["stop_b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["stop_rise"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["stop_decay"] == 0

            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["tau1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["b2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["tau2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["b3"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["tau3"] == 0

            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_tau1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_b2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_tau2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_b3"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_tau3"] == 0

            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_b0"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_b1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_tau1"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_b2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_tau2"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_b3"] == 0
            assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_tau3"] == 0

            assert tgui.mw.cfgs.curve_fitting_region_position_configs[region_name]["upper_bl_lr_lowerbound"] is None
            assert tgui.mw.cfgs.curve_fitting_region_position_configs[region_name]["upper_bl_lr_upperbound"] is None
            assert tgui.mw.cfgs.curve_fitting_region_position_configs[region_name]["upper_exp_lr_lowerbound"] is None
            assert tgui.mw.cfgs.curve_fitting_region_position_configs[region_name]["upper_exp_lr_upperbound"] is None

    @pytest.mark.parametrize("dialog", [True, False])
    def test_currently_selected_region_buttons(self, tgui, dialog):
        """
        First open the dialog and switch through all regions. Then go back the opposite way.
        Use the buttons on the dialog if set, otherwise the buttons on mainwindow
        """
        setup_curve_fitting_options_dialog(tgui)
        if dialog:
            button_left = tgui.mw.dialogs["curve_fitting"].dia.change_region_left_button
            button_right = tgui.mw.dialogs["curve_fitting"].dia.change_region_right_button
        else:
            button_left = tgui.mw.mw.curve_fitting_scroll_region_left_button
            button_right = tgui.mw.mw.curve_fitting_scroll_region_right_button

        regions = ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]
        for region_idx, region_name in enumerate(regions):
            assert tgui.mw.cfgs.curve_fitting["currently_selected_region"] == region_name
            assert tgui.mw.cfgs.curve_fitting["region_idx"] == region_idx
            tgui.left_mouse_click(button_right)

        for region_idx, region_name in zip([5, 4, 3, 2, 1],
                                           reversed(regions)):
            assert tgui.mw.cfgs.curve_fitting["currently_selected_region"] == region_name
            assert tgui.mw.cfgs.curve_fitting["region_idx"] == region_idx
            tgui.left_mouse_click(button_left)

    def test_show_selected_region_only(self, tgui):
        """
        Check show only selected region, same button across all regions but just scroll through them all
        just in case
        """
        setup_curve_fitting_options_dialog(tgui)
        config_dict = tgui.mw.cfgs.curve_fitting["region_display"]
        dialog = tgui.mw.dialogs["curve_fitting"].dia

        for __ in ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]:
            tgui.left_mouse_click(dialog.show_selected_region_only_checkbox)
            assert config_dict["show_only_selected_region"] is True
            tgui.left_mouse_click(dialog.show_selected_region_only_checkbox)
            assert config_dict["show_only_selected_region"] is False
            tgui.left_mouse_click(dialog.change_region_right_button)

    def test_show_selected_region_only_defaults(self, tgui):

        setup_curve_fitting_options_dialog(tgui)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.show_selected_region_only_checkbox)
        assert tgui.mw.cfgs.curve_fitting["region_display"]["show_only_selected_region"] is True

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name="reg_1")
        assert tgui.mw.dialogs["curve_fitting"].dia.show_selected_region_only_checkbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["region_display"]["show_only_selected_region"] is True

        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.show_selected_region_only_checkbox)

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name="reg_1")
        assert not tgui.mw.dialogs["curve_fitting"].dia.show_selected_region_only_checkbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["region_display"]["show_only_selected_region"] is False
        tgui.shutdown()

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_display_options(self, tgui, region_name):
        """
        """
        setup_curve_fitting_options_dialog(tgui)
        curve_fitting_switch_to_region(tgui, region_name)

        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.hide_baseline_radiobutton)
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["region"] == "hide_baseline"

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.hide_baseline_radiobutton.isChecked()
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["region"] == "hide_baseline"

        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.hide_region_radiobutton)
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["region"] == "hide_region"

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.hide_region_radiobutton.isChecked()
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["region"] == "hide_region"

        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.show_region_radiobutton)
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["region"] == "show_region"

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.show_region_radiobutton.isChecked()
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["region"] == "show_region"

        tgui.click_checkbox(tgui.mw.dialogs["curve_fitting"].dia.link_and_unlink_curve_fit_region_checkbox)
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["link_baseline"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert not tgui.mw.dialogs["curve_fitting"].dia.link_and_unlink_curve_fit_region_checkbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["link_baseline"] is False

        tgui.click_checkbox(tgui.mw.dialogs["curve_fitting"].dia.link_and_unlink_curve_fit_region_checkbox)
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["link_baseline"] is True

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.link_and_unlink_curve_fit_region_checkbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["link_baseline"] is True

        tgui.shutdown()

    # Monoexpoential
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    # MOVE
    def check_curve_fitting_coefficient_widget(self, dict_, spinbox, input_, analysis_type, est_or_bound, coeff):
        first_check = dict_[analysis_type][est_or_bound][coeff] == 0.000
        spinbox.clear()
        keyClicks(spinbox, str(input_))
        input_ = input_ / 1000 if "tau" in coeff or "rise" in coeff or "decay" in coeff else input_  # convert taus / rise /decay/ to ms
        second_check = dict_[analysis_type][est_or_bound][coeff] == input_
        assert first_check, analysis_type + " " + est_or_bound + " " + coeff + " initial entry is not zero at start"
        assert second_check, analysis_type + " " + est_or_bound + " " + coeff + " configs is not changed after spinbox input"

    def check_curve_fitting_coefficient_pos_or_neg_radiobuttons(self, tgui, dict_, pos_radiobutton, neg_radiobutton, analysis_type):
        assert pos_radiobutton.isChecked(), "pos radiobutton is not checked on init " + analysis_type
        tgui.click_checkbox(neg_radiobutton)
        assert dict_[analysis_type]["direction"] == -1, "dict direction is not updated to negative " + analysis_type
        tgui.click_checkbox(pos_radiobutton)
        assert dict_[analysis_type]["direction"] == 1, "dict direction is not updated to positive " + analysis_type

    def setup_curve_fitting_on_analysis_page(self, tgui, region_name, combobox_idx):  # dont need full curve fitting for this.
        load_file(tgui)
        tgui.left_mouse_click(tgui.mw.mw.curve_fitting_show_dialog_button)
        curve_fitting_switch_to_region(tgui, region_name)

        # Set and Check analysis combobox
        tgui.set_combobox(tgui.mw.dialogs["curve_fitting"].dia.fit_type_combobox,
                          combobox_idx)

    def get_radiobutton_widgets(self, tgui, analysis_type, pos_or_neg):

        radiobuttons = {

            "monoexp": {"pos": tgui.mw.dialogs["curve_fitting"].dia.monoexp_direction_pos_radiobutton,
                        "neg": tgui.mw.dialogs["curve_fitting"].dia.monoexp_direction_neg_radiobutton,
                        },

            "biexp_decay": {"pos": tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_direction_pos_radiobutton,
                            "neg": tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_direction_neg_radiobutton,
                            },

            "biexp_event": {"pos": tgui.mw.dialogs["curve_fitting"].dia.biexp_event_direction_pos_radiobutton,
                            "neg": tgui.mw.dialogs["curve_fitting"].dia.biexp_event_direction_neg_radiobutton,
                            },
            "triexp": {"pos": tgui.mw.dialogs["curve_fitting"].dia.triexp_direction_pos_radiobutton,
                       "neg": tgui.mw.dialogs["curve_fitting"].dia.triexp_direction_neg_radiobutton,
                       },
        }
        return radiobuttons[analysis_type][pos_or_neg]

    # Curve Fitting Direction Radiobuttons -------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    @pytest.mark.parametrize("combobox_idx", [4, 5, 6, 7])
    def test_curve_fitting_monoexp_widgets_configs_stackewidget(self, tgui, region_name, combobox_idx):

        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx)

        assert tgui.mw.cfgs.curve_fitting["analysis_type_stackedwidget_idx"][
                   region_name] == combobox_idx, region_name + " stackwidget cfg is incorrect for " + combobox_idx

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.fit_type_combobox.currentIndex() == combobox_idx
        assert tgui.mw.cfgs.curve_fitting["analysis_type_stackedwidget_idx"][
                   region_name] == combobox_idx, region_name + " stackwidget cfg is incorrect after reload"

        tgui.shutdown()

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    @pytest.mark.parametrize("idx_analysis_type", [[5, "monoexp"], [6, "biexp_decay"], [7, "biexp_event"], [8, "triexp"]])
    def test_curve_fitting_monoexp_widgets_configs_radiobuttons(self, tgui, idx_analysis_type, region_name):
        combobox_idx, analysis_type = idx_analysis_type
        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx=combobox_idx)
        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_direction_pos_radiobutton.isChecked(), "pos radiobutton is not checked on init"

        tgui.click_checkbox(self.get_radiobutton_widgets(tgui, analysis_type, "neg"))
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name][analysis_type]["direction"] == -1, "dict direction is not updated to negative"

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert self.get_radiobutton_widgets(tgui, analysis_type, "neg").isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name][analysis_type]["direction"] == -1

        tgui.click_checkbox(self.get_radiobutton_widgets(tgui, analysis_type, "pos"))
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name][analysis_type]["direction"] == 1, "dict direction is not updated to positive"

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert self.get_radiobutton_widgets(tgui, analysis_type, "pos").isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name][analysis_type]["direction"] == 1

        tgui.shutdown()

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_monoexp_widgets_configs_initial_est(self, tgui, region_name):
        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx=5)

        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.monoexp_coefficients_groupbox,
                             on=True)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["set_coefficients"] is True

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.monoexp_initial_est_b0_spinbox,
                                                    0.001,
                                                    "monoexp", "initial_est", "b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.monoexp_initial_est_b1_spinbox,
                                                    0.002,
                                                    "monoexp", "initial_est", "b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.monoexp_initial_est_tau_spinbox,
                                                    0.003,
                                                    "monoexp", "initial_est", "tau")

        # Save, reload and check settings remained
        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_coefficients_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["set_coefficients"] is True
        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_initial_est_b0_spinbox.value() == 0.001
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["initial_est"]["b0"] == 0.001

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_initial_est_b1_spinbox.value() == 0.002
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["initial_est"]["b1"] == 0.002

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_initial_est_tau_spinbox.value() == 0.003
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["initial_est"]["tau"] == 0.003 / 1000

        # turn off box and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.monoexp_coefficients_groupbox,
                             on=False)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["set_coefficients"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert not tgui.mw.dialogs["curve_fitting"].dia.monoexp_coefficients_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["set_coefficients"] is False

        tgui.shutdown()

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_monoexp_widgets_configs_bounds(self, tgui, region_name):
        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx=5)

        # Turn out bounds and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_groupbox,
                             on=True)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["set_bounds"] is True

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_start_b0_spinbox,
                                                    0.004,
                                                    "monoexp", "bounds", "start_b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_start_b1_spinbox,
                                                    0.005,
                                                    "monoexp", "bounds", "start_b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_start_tau_spinbox,
                                                    0.006,
                                                    "monoexp", "bounds", "start_tau")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_stop_b0_spinbox,
                                                    0.007,
                                                    "monoexp", "bounds", "stop_b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_stop_b1_spinbox,
                                                    0.008,
                                                    "monoexp", "bounds", "stop_b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_stop_tau_spinbox,
                                                    0.009,
                                                    "monoexp", "bounds", "stop_tau")

        # Save, reload and check settings remained
        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["set_bounds"] is True
        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_groupbox.isChecked()
        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_start_b0_spinbox.value() == 0.004
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["start_b0"] == 0.004

        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_start_b1_spinbox.value() == 0.005
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["start_b1"] == 0.005

        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_start_tau_spinbox.value() == 0.006
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["start_tau"] == 0.006 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_stop_b0_spinbox.value() == 0.007
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["stop_b0"] == 0.007

        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_stop_b1_spinbox.value() == 0.008
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["stop_b1"] == 0.008

        assert tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_stop_tau_spinbox.value() == 0.009
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["bounds"]["stop_tau"] == 0.009 / 1000

        # Turn off box and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_groupbox,
                             on=False)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["set_bounds"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert not tgui.mw.dialogs["curve_fitting"].dia.monoexp_bounds_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["monoexp"]["set_bounds"] is False

        tgui.shutdown()

    # Test Biexp Decay
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_biexp_decay_widgets_configs_set_estimates(self, tgui, region_name):
        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx=6)

        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_coefficients_groupbox,
                             on=True)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["set_coefficients"] is True

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_b0_spinbox,
                                                    0.010,
                                                    "biexp_decay", "initial_est", "b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_b1_spinbox,
                                                    0.011,
                                                    "biexp_decay", "initial_est", "b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_tau1_spinbox,
                                                    0.012,
                                                    "biexp_decay", "initial_est", "tau1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_b2_spinbox,
                                                    0.013,
                                                    "biexp_decay", "initial_est", "b2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_tau2_spinbox,
                                                    0.014,
                                                    "biexp_decay", "initial_est", "tau2")

        # Save, reload and check settings remained
        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["set_coefficients"] is True
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_coefficients_groupbox.isChecked()
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_b0_spinbox.value() == 0.010
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["b0"] == 0.010

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_b1_spinbox.value() == 0.011
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["b1"] == 0.011

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_tau1_spinbox.value() == 0.012
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["tau1"] == 0.012 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_b2_spinbox.value() == 0.013
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["b2"] == 0.013

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_initial_est_tau2_spinbox.value() == 0.014
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["initial_est"]["tau2"] == 0.014 / 1000

        # Turn off boxes and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_coefficients_groupbox,
                             on=False)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["set_coefficients"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert not tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_coefficients_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["set_coefficients"] is False

        tgui.shutdown()

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_biexp_decay_widgets_configs_bounds(self, tgui, region_name):
        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx=6)

        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_groupbox,
                             on=True)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["set_bounds"] is True

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_b0_spinbox,
                                                    0.015,
                                                    "biexp_decay", "bounds", "start_b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_b1_spinbox,
                                                    0.016,
                                                    "biexp_decay", "bounds", "start_b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_tau1_spinbox,
                                                    0.017,
                                                    "biexp_decay", "bounds", "start_tau1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_b2_spinbox,
                                                    0.018,
                                                    "biexp_decay", "bounds", "start_b2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_tau2_spinbox,
                                                    0.019,
                                                    "biexp_decay", "bounds", "start_tau2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_b0_spinbox,
                                                    0.020,
                                                    "biexp_decay", "bounds", "stop_b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_b1_spinbox,
                                                    0.021,
                                                    "biexp_decay", "bounds", "stop_b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_tau1_spinbox,
                                                    0.022,
                                                    "biexp_decay", "bounds", "stop_tau1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_b2_spinbox,
                                                    0.023,
                                                    "biexp_decay", "bounds", "stop_b2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_tau2_spinbox,
                                                    0.024,
                                                    "biexp_decay", "bounds", "stop_tau2")

        # Save, reload and check settings remained
        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["set_bounds"] is True
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_groupbox.isChecked()
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_b0_spinbox.value() == 0.015
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_b0"] == 0.015

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_b1_spinbox.value() == 0.016
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_b1"] == 0.016

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_tau1_spinbox.value() == 0.017
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_tau1"] == 0.017 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_b2_spinbox.value() == 0.018
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_b2"] == 0.018

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_start_tau2_spinbox.value() == 0.019
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["start_tau2"] == 0.019 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_b0_spinbox.value() == 0.020
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_b0"] == 0.020

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_b1_spinbox.value() == 0.021
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_b1"] == 0.021

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_tau1_spinbox.value() == 0.022
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_tau1"] == 0.022 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_b2_spinbox.value() == 0.023
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_b2"] == 0.023

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_stop_tau2_spinbox.value() == 0.024
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["bounds"]["stop_tau2"] == 0.024 / 1000

        # Turn off boxes and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_groupbox,
                             on=False)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["set_bounds"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert not tgui.mw.dialogs["curve_fitting"].dia.biexp_decay_bounds_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_decay"]["set_bounds"] is False

        tgui.shutdown()

    # Test Biexp Event
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_biexp_event_widgets_configs_initial_est(self, tgui, region_name):
        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx=7)

        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_event_coefficients_groupbox,
                             on=True)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["set_coefficients"] is True

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_initial_est_b0_spinbox,
                                                    0.030,
                                                    "biexp_event", "initial_est", "b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_initial_est_b1_spinbox,
                                                    0.031,
                                                    "biexp_event", "initial_est", "b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_initial_est_rise_spinbox,
                                                    0.032,
                                                    "biexp_event", "initial_est", "rise")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_initial_est_decay_spinbox,
                                                    0.033,
                                                    "biexp_event", "initial_est", "decay")

        # Save, reload and check settings remained
        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_coefficients_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["set_coefficients"] is True
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_initial_est_b0_spinbox.value() == 0.030
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["initial_est"]["b0"] == 0.030

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_initial_est_b1_spinbox.value() == 0.031
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["initial_est"]["b1"] == 0.031

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_initial_est_rise_spinbox.value() == 0.032
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["initial_est"]["rise"] == 0.032 / 1000

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_initial_est_decay_spinbox.value() == 0.033
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["initial_est"]["decay"] == 0.033 / 1000

        # Biexp Event Show Fit
        tgui.click_checkbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_event_hide_event_fit_button)  # TODO: this is a checkbox
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["hide_event_fit"] is True

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_hide_event_fit_button.isChecked()
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["hide_event_fit"] is True

        tgui.click_checkbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_event_hide_event_fit_button)  # TODO: this is a checkbox
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["hide_event_fit"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert not tgui.mw.dialogs["curve_fitting"].dia.biexp_event_hide_event_fit_button.isChecked()
        assert tgui.mw.cfgs.curve_fitting["region_display"][region_name]["hide_event_fit"] is False

        # Turn off boxes and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_event_coefficients_groupbox,
                             on=False)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["set_coefficients"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert not tgui.mw.dialogs["curve_fitting"].dia.biexp_event_coefficients_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["set_coefficients"] is False

        tgui.shutdown()

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_biexp_event_widgets_configs_bounds(self, tgui, region_name):
        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx=7)

        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_groupbox,
                             on=True)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["set_bounds"] is True

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_start_b0_spinbox,
                                                    0.040,
                                                    "biexp_event", "bounds", "start_b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_start_b1_spinbox,
                                                    0.041,
                                                    "biexp_event", "bounds", "start_b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_start_rise_spinbox,
                                                    0.042,
                                                    "biexp_event", "bounds", "start_rise")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_start_decay_spinbox,
                                                    0.043,
                                                    "biexp_event", "bounds", "start_decay")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_stop_b0_spinbox,
                                                    0.044,
                                                    "biexp_event", "bounds", "stop_b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_stop_b1_spinbox,
                                                    0.045,
                                                    "biexp_event", "bounds", "stop_b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_stop_rise_spinbox,
                                                    0.046,
                                                    "biexp_event", "bounds", "stop_rise")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_stop_decay_spinbox,
                                                    0.047,
                                                    "biexp_event", "bounds", "stop_decay")

        # Save, reload and check settings remained
        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["set_bounds"] is True
        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_start_b0_spinbox.value() == 0.040
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["start_b0"] == 0.040

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_start_b1_spinbox.value() == 0.041
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["start_b1"] == 0.041

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_start_rise_spinbox.value() == 0.042
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["start_rise"] == 0.042 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_start_decay_spinbox.value() == 0.043
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["start_decay"] == 0.043 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_stop_b0_spinbox.value() == 0.044
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["stop_b0"] == 0.044

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_stop_b1_spinbox.value() == 0.045
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["stop_b1"] == 0.045

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_stop_rise_spinbox.value() == 0.046
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["stop_rise"] == 0.046 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_stop_decay_spinbox.value() == 0.047
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["bounds"]["stop_decay"] == 0.047 / 1000

        # Turn off box, reload and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_groupbox,
                             on=False)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["set_bounds"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert not tgui.mw.dialogs["curve_fitting"].dia.biexp_event_bounds_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["biexp_event"]["set_bounds"] is False

        tgui.shutdown()

    def bring_up_curve_fitting_events_dialog(self, tgui):
        setup_curve_fitting_options_dialog(tgui)
        tgui.set_combobox(tgui.mw.dialogs["curve_fitting"].dia.fit_type_combobox,
                          7)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.curve_fitting_event_kinetics_options)

    def test_curve_fitting_biexp_event_matches_configs_on_init(self, tgui):
        self.bring_up_curve_fitting_events_dialog(tgui)
        events_dialog = tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia
        events_config = tgui.mw.cfgs.curve_fitting["event_kinetics"]

        assert events_dialog.average_peak_checkbox.isChecked() is events_config["average_peak_points"]["on"]
        assert events_dialog.average_peak_spinbox.value() == events_config["average_peak_points"]["value_s"] * 1000
        assert events_dialog.average_baseline_checkbox.isChecked() is events_config["average_baseline_points"]["on"]
        assert events_dialog.average_baseline_spinbox.value() == events_config["average_baseline_points"]["value_s"] * 1000
        assert events_dialog.baseline_search_period_spinbox.value() == events_config["baseline_search_period_s"] * 1000
        assert events_dialog.baseline_per_event_combobox.currentIndex() == 0 if events_config["baseline_type"] == "per_event" else 1
        assert events_dialog.calculate_kinetics_from_fit_not_data_checkbox.isChecked() is events_config["from_fit_not_data"]

    def test_biexp_event_event_configs(self, tgui):
        self.bring_up_curve_fitting_events_dialog(tgui)

        tgui.click_checkbox(tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_peak_checkbox)
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_peak_points"]["on"]

        tgui.enter_number_into_spinbox(tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_peak_spinbox,
                                       5,
                                       setValue=True)
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_peak_points"]["value_s"] == 0.005

        tgui.set_combobox(tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.baseline_per_event_combobox,
                          1)
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["baseline_type"] == "manual"

        tgui.enter_number_into_spinbox(tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.baseline_search_period_spinbox, 10,
                                       setValue=True)
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["baseline_search_period_s"] == 0.010

        tgui.click_checkbox(tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_baseline_checkbox)
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_baseline_points"]["on"]

        tgui.enter_number_into_spinbox(tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_baseline_spinbox, 6,
                                       setValue=True)
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_baseline_points"]["value_s"] == 0.006

        # Reload file and check widgets / configs
        tgui = close_and_reload_for_defaults_check(tgui,
                                                        save_="curve_fitting_options_event_kinetics",
                                                        region_name="reg_1")

        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_peak_points"]["on"]
        assert tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_peak_checkbox.isChecked()

        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_peak_points"]["value_s"] == 0.005
        assert tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_peak_spinbox.value() == 5

        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["baseline_type"] == "manual"
        assert tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.baseline_per_event_combobox.currentIndex() == 1

        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["baseline_search_period_s"] == 0.010
        assert tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.baseline_search_period_spinbox.value() == 10

        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_baseline_points"]["on"]
        assert tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_baseline_checkbox.isChecked()

        assert tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_baseline_spinbox.value() == 6
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["average_baseline_points"]["value_s"] == 0.006

        tgui.set_combobox(tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.baseline_per_event_combobox,
                          0)
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["baseline_type"] == "per_event"

        # Test Combobox and Fit from Curve checkbox
        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options_event_kinetics", region_name="reg_1")
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["baseline_type"] == "per_event"

        tgui.click_checkbox(tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.calculate_kinetics_from_fit_not_data_checkbox)
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["from_fit_not_data"] is True

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options_event_kinetics", region_name="reg_1")
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["from_fit_not_data"] is True

        tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_peak_checkbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["event_kinetics"]["from_fit_not_data"] is True

        assert not tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_peak_spinbox.isEnabled()
        assert not tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.baseline_per_event_combobox.isEnabled()
        assert not tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.baseline_search_period_spinbox.isEnabled()
        assert not tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_baseline_checkbox.isEnabled()
        assert not tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.average_baseline_spinbox.isEnabled()
        assert not tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog.dia.baseline_per_event_combobox.isEnabled()

        tgui.shutdown()

    # Test Triexponential
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_triexp_widgets_configs_initial_est(self, tgui, region_name):
        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx=8)

        # Turn out initial estimate bounds and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.triexp_coefficients_groupbox,
                             on=True)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["set_coefficients"] is True

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_b0_spinbox,
                                                    0.050,
                                                    "triexp", "initial_est", "b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_b1_spinbox,
                                                    0.051,
                                                    "triexp", "initial_est", "b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_tau1_spinbox,
                                                    0.052,
                                                    "triexp", "initial_est", "tau1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_b2_spinbox,
                                                    0.053,
                                                    "triexp", "initial_est", "b2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_tau2_spinbox,
                                                    0.054,
                                                    "triexp", "initial_est", "tau2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_b3_spinbox,
                                                    0.055,
                                                    "triexp", "initial_est", "b3")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_tau3_spinbox,
                                                    0.056,
                                                    "triexp", "initial_est", "tau3")

        # Reload and check settings remain
        tgui = close_and_reload_for_defaults_check(tgui,
                                                        save_="curve_fitting_options",
                                                        region_name=region_name)

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_coefficients_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["set_coefficients"] is True
        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_b0_spinbox.value() == 0.050
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["b0"] == 0.050

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_b1_spinbox.value() == 0.051
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["b1"] == 0.051

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_tau1_spinbox.value() == 0.052
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["tau1"] == 0.052 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_b2_spinbox.value() == 0.053
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["b2"] == 0.053

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_tau2_spinbox.value() == 0.054
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["tau2"] == 0.054 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_b3_spinbox.value() == 0.055
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["b3"] == 0.055

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_initial_est_tau3_spinbox.value() == 0.056
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["initial_est"]["tau3"] == 0.056 / 1000

        # Turn off box and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.triexp_coefficients_groupbox,
                             on=False)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["set_coefficients"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert not tgui.mw.dialogs["curve_fitting"].dia.triexp_coefficients_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["set_coefficients"] is False

        tgui.shutdown()

    @pytest.mark.parametrize("region_name", ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_triexp_widgets_configs_bounds(self, tgui, region_name):
        self.setup_curve_fitting_on_analysis_page(tgui, region_name, combobox_idx=8)

        # Turn out bounds and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_groupbox,
                             on=True)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["set_bounds"] is True

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_b0_spinbox,
                                                    0.060,
                                                    "triexp", "bounds", "start_b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_b1_spinbox,
                                                    0.061,
                                                    "triexp", "bounds", "start_b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_tau1_spinbox,
                                                    0.062,
                                                    "triexp", "bounds", "start_tau1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_b2_spinbox,
                                                    0.063,
                                                    "triexp", "bounds", "start_b2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_tau2_spinbox,
                                                    0.064,
                                                    "triexp", "bounds", "start_tau2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_b3_spinbox,
                                                    0.065,
                                                    "triexp", "bounds", "start_b3")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_tau3_spinbox,
                                                    0.066,
                                                    "triexp", "bounds", "start_tau3")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_b0_spinbox,
                                                    0.067,
                                                    "triexp", "bounds", "stop_b0")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_b1_spinbox,
                                                    0.068,
                                                    "triexp", "bounds", "stop_b1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_tau1_spinbox,
                                                    0.069,
                                                    "triexp", "bounds", "stop_tau1")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_b2_spinbox,
                                                    0.070,
                                                    "triexp", "bounds", "stop_b2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_tau2_spinbox,
                                                    0.071,
                                                    "triexp", "bounds", "stop_tau2")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_b3_spinbox,
                                                    0.072,
                                                    "triexp", "bounds", "stop_b3")

        self.check_curve_fitting_coefficient_widget(tgui.mw.cfgs.curve_fitting["analysis"][region_name],
                                                    tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_tau3_spinbox,
                                                    0.073,
                                                    "triexp", "bounds", "stop_tau3")
        # Save and Check
        tgui = close_and_reload_for_defaults_check(tgui, save_="curve_fitting_options", region_name=region_name)
        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_groupbox.isChecked()
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["set_bounds"] is True
        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_b0_spinbox.value() == 0.060
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_b0"] == 0.060

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_b1_spinbox.value() == 0.061
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_b1"] == 0.061

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_tau1_spinbox.value() == 0.062
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_tau1"] == 0.062 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_b2_spinbox.value() == 0.063
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_b2"] == 0.063

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_tau2_spinbox.value() == 0.064
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_tau2"] == 0.064 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_b3_spinbox.value() == 0.065
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_b3"] == 0.065

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_start_tau3_spinbox.value() == 0.066
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["start_tau3"] == 0.066 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_b0_spinbox.value() == 0.067
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_b0"] == 0.067

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_b1_spinbox.value() == 0.068
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_b1"] == 0.068

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_tau1_spinbox.value() == 0.069
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_tau1"] == 0.069 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_b2_spinbox.value() == 0.070
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_b2"] == 0.070

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_tau2_spinbox.value() == 0.071
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_tau2"] == 0.071 / 1000

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_b3_spinbox.value() == 0.072
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_b3"] == 0.072

        assert tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_stop_tau3_spinbox.value() == 0.073
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["bounds"]["stop_tau3"] == 0.073 / 1000

        # Turn off boxes and check
        tgui.switch_groupbox(tgui.mw.dialogs["curve_fitting"].dia.triexp_bounds_groupbox,
                             on=False)
        assert tgui.mw.cfgs.curve_fitting["analysis"][region_name]["triexp"]["set_bounds"] is False

        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Events Configs
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    # Defaults
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_events_config_defaults(self, tgui):

        # Always default options
        assert tgui.mw.cfgs.events["plot_view_window_size"] is None, "plot_view_window_size"
        assert tgui.mw.cfgs.events["template_data_first_selection"]["x"] is None, "template_data_first_selection"
        assert tgui.mw.cfgs.events["template_data_first_selection"]["y"] is None, "template_data_first_selection"
        assert tgui.mw.cfgs.events["select_template_spike_mode"] is False, "select_template_spike_mode"
        assert tgui.mw.cfgs.events["draw_line_mode"] is False, "draw_line_mode"
        assert tgui.mw.cfgs.events["refine_template_betas"] is None, "refine_template_betas"
        assert tgui.mw.cfgs.events["refine_template_detection_coefs"] is None, "refine_template_detection_coefs"
        assert tgui.mw.cfgs.events["refine_template_user_events_average"] is None, "refine_template_user_events_average"
        assert tgui.mw.cfgs.events["refine_template_filter_user_average"] is None, "refine_template_filter_user_average"
        assert tgui.mw.cfgs.events["decay_period_type"] == "auto_search_data", "decay_period_type"

        # Current Coefficients, shared between generate, refine and template + detect all events.
        for template in ["1", "2", "3"]:
            assert tgui.mw.cfgs.events["templates"][template]["b0_ms"] == 0, "b0_ms"
            assert tgui.mw.cfgs.events["templates"][template]["b1_ms"] == -1, "b1_ms"
            assert tgui.mw.cfgs.events["templates"][template]["rise_s"] == 0.0005, "rise_s"
            assert tgui.mw.cfgs.events["templates"][template]["decay_s"] == 0.005, "decay_s"
            assert tgui.mw.cfgs.events["templates"][template]["window_len_s"] == 0.02, "window_len_s"
            assert tgui.mw.cfgs.events["templates"][template]["window_len_samples"] is None, "window_len_samples"
            assert tgui.mw.cfgs.events["templates"][template]["direction"] == -1, "template_direction"

        assert tgui.mw.cfgs.events["currently_selected_template"] == "1", "currently_selected_template"
        assert tgui.mw.cfgs.events["template_to_use"] == "selected_template", "template_to_use"
        assert tgui.mw.cfgs.events["cannonical_initial_biexp_coefficients"]["rise"] == 0.5, "cannonical_initial_biexp_coefficients"
        assert tgui.mw.cfgs.events["cannonical_initial_biexp_coefficients"]["decay"] == 5, "cannonical_initial_biexp_coefficients"

        # Refine Template
        assert tgui.mw.cfgs.events["refine_template_show_sliding_window_fit_plot"] is False, "refine_template_show_sliding_window_fit_plot"
        assert tgui.mw.cfgs.events["refine_template_show_corr_plot"] is False, "refine_template_show_corr_plot"

        # Shared between Refine and Analyse
        assert tgui.mw.cfgs.events["corr_cutoff"] == 0.4, "corr_cutoff"
        assert tgui.mw.cfgs.events["detection_threshold_type"] == "correlation", "detection_threshold_type"
        assert tgui.mw.cfgs.events["detection_criterion"] == 4, "detection_criterion"

        assert tgui.mw.cfgs.events["deconv_options"]["filt_low_hz"] == 0.1, "filt_low_hz"
        assert tgui.mw.cfgs.events["deconv_options"]["filt_high_hz"] == 200, "filt_high_hz"
        assert tgui.mw.cfgs.events["deconv_options"]["n_times_std"] == 3.5, "n_times_std"
        assert tgui.mw.cfgs.events["deconv_options"]["detection_threshold"] is None, "detection_threshold"

        assert tgui.mw.cfgs.events["curved_threshold_lower_displacement"] == 0, "curved_threshold_lower_displacement"
        assert tgui.mw.cfgs.events["rms_lower_threshold_n_times"] == 2, "rms_lower_threshold_n_times"
        assert tgui.mw.cfgs.events["threshold_upper_limit_on"] is False, "threshold_upper_limit_on"
        assert tgui.mw.cfgs.events["threshold_upper_limit_value"] == 0, "threshold_upper_limit_value"
        assert tgui.mw.cfgs.events["show_threshold_lower_on_plot"] is True, "show_threshold_lower_on_plot"
        assert tgui.mw.cfgs.events["linear_threshold_lower_value"] == -52, "linear_threshold_lower_value"
        assert tgui.mw.cfgs.events["threshold_type"] == "manual", "threshold_type"

        assert tgui.mw.cfgs.events["drawn_threshold_lower_points"] == [], "drawn_threshold_lower_points"
        assert tgui.mw.cfgs.events["omit_start_stop_times"] is None, "omit_start_stop_times"

        assert tgui.mw.cfgs.events["show_baseline_on_plot"] is True, "show_baseline_on_plot"
        assert tgui.mw.cfgs.events["baseline_type"] == "per_event", "baseline_type"
        assert tgui.mw.cfgs.events["curved_baseline_displacement"] == 0, "curved_baseline_displacement"
        assert tgui.mw.cfgs.events["baseline_axline_settings"]["linear_baseline_value"] is None, "baseline_axline_settings"
        assert tgui.mw.cfgs.events["drawn_baseline_points"] == [], "drawn_baseline_points"

        # Misc.
        assert tgui.mw.cfgs.events["endpoint_search_method"] == "entire_search_region"
        assert tgui.mw.cfgs.events["interp_200khz"] is False, "interp_200khz"
        assert tgui.mw.cfgs.events["rise_cutoff_high"] == 90, "rise_cutoff_high"
        assert tgui.mw.cfgs.events["rise_cutoff_low"] == 10, "rise_cutoff_low"
        assert tgui.mw.cfgs.events["decay_amplitude_percent"] == 37, "decay_amplitude_percent"
        assert tgui.mw.cfgs.events["threshold_direction"] == -1, "threshold_direction"
        assert tgui.mw.cfgs.events["dynamic_curve_polynomial_order"] == 5, "dynamic_curve_polynomial_order"
        assert tgui.mw.cfgs.events["amplitude_threshold"] == 10, "amplitude_threshold"
        assert tgui.mw.cfgs.events["threshold_local_maximum_period_s"] == 0.010, "threshold_local_maximum_period_s"
        assert tgui.mw.cfgs.events["decay_search_period_s"] == 0.030, "decay_search_period_s"
        assert tgui.mw.cfgs.events["threshold_manual_selected_event"] is True
        assert tgui.mw.cfgs.events["show_auc_plot"] is False
        assert tgui.mw.cfgs.events["max_slope"]["on"] is False
        assert tgui.mw.cfgs.events["max_slope"]["rise_num_samples"] == 2
        assert tgui.mw.cfgs.events["max_slope"]["decay_num_samples"] == 2
        assert tgui.mw.cfgs.events["max_slope"]["smooth"]["on"] is False
        assert tgui.mw.cfgs.events["max_slope"]["smooth"]["num_samples"] == 2
        assert tgui.mw.cfgs.events["max_slope"]["use_baseline_crossing_endpoint"] is True

        for key in ["monoexp_fit", "biexp_fit"]:
            assert tgui.mw.cfgs.events[key]["exclude_from_r2_on"] is False
            assert tgui.mw.cfgs.events[key]["exclude_from_r2_value"] == 0
            assert tgui.mw.cfgs.events[key]["exclude_if_params_not_in_bounds"] is False
            bool_ = True if key == "biexp_fit" else False
            assert tgui.mw.cfgs.events[key]["adjust_startpoint_r2_on"] is bool_
            val = 3 if key == "biexp_fit" else 1
            assert tgui.mw.cfgs.events[key]["adjust_startpoint_r2_value"] == val
            assert tgui.mw.cfgs.events[key]["adjust_startpoint_bounds_on"] is False
            assert tgui.mw.cfgs.events[key]["adjust_startpoint_bounds_value"] == 1

        assert tgui.mw.cfgs.events["monoexp_fit"]["tau_cutoff_min"] == 0
        assert tgui.mw.cfgs.events["monoexp_fit"]["tau_cutoff_max"] == 60
        assert tgui.mw.cfgs.events["biexp_fit"]["rise_cutoff_min"] == 0
        assert tgui.mw.cfgs.events["biexp_fit"]["rise_cutoff_max"] == 60
        assert tgui.mw.cfgs.events["biexp_fit"]["decay_cutoff_min"] == 0
        assert tgui.mw.cfgs.events["biexp_fit"]["decay_cutoff_max"] == 60

        # Frequency data options
        config_dict = tgui.mw.cfgs.events["frequency_data_options"]

        for parameter in ["frequency", "amplitude", "rise_time", "decay_amplitude_percent", "decay_tau", "biexp_rise", "biexp_decay"]:
            assert config_dict["custom_binsize"][parameter] == 0

        assert config_dict["x_axis_display"] == "bin_centre"
        assert config_dict["custom_binnum"] == 2
        assert config_dict["binning_method"] == "auto"
        assert config_dict["divide_by_number"] == 1
        assert config_dict["plot_type"] == "cum_prob"

        # From shared widgets (template and threshold)
        assert tgui.mw.cfgs.events["baseline_search_period_s"] == 0.01, "baseline_search_period_s"
        assert tgui.mw.cfgs.events["average_peak_points"]["on"] is False, "average_peak_points"
        assert tgui.mw.cfgs.events["average_peak_points"]["value_s"] == 0.0005, "average_peak_points"
        assert tgui.mw.cfgs.events["area_under_curve"]["on"] is False, "area_under_curve"
        assert tgui.mw.cfgs.events["area_under_curve"]["value_pa_ms"] == 80, "area_under_curve"
        assert tgui.mw.cfgs.events["average_baseline_points"]["on"], "average_baseline_points"
        assert tgui.mw.cfgs.events["average_baseline_points"]["value_s"] == 0.001, "average_baseline_points"

    # Dialogs on init
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_events_generate_configs_on_init(self, tgui):
        """
        Check Settings Match Default
        """
        dialog, config_dict = self.setup_events_with_dialog(tgui, "generate")

        for idx in range(3):
            template = str(idx + 1)

            dialog.dia.choose_template_combobox.setCurrentIndex(idx)
            assert config_dict["currently_selected_template"] == template

            assert dialog.dia.b0_spinbox.value() == config_dict["templates"][template]["b0_ms"]
            assert dialog.dia.b1_spinbox.value() == config_dict["templates"][template]["b1_ms"]
            assert dialog.dia.rise_spinbox.value() == config_dict["templates"][template]["rise_s"] * 1000
            assert dialog.dia.decay_spinbox.value() == config_dict["templates"][template]["decay_s"] * 1000
            assert dialog.dia.width_spinbox.value() == config_dict["templates"][template]["window_len_s"] * 1000
            assert dialog.dia.width_spinbox.value() == (config_dict["templates"][template]["window_len_samples"] / tgui.mw.loaded_file.data.fs) * 1000
            direction = True if config_dict["templates"][template]["direction"] == 1 else False
            assert dialog.dia.event_direction_is_positive_checkbox.isChecked() == direction

    def test_events_refine_configs_on_init(self, tgui):

        dialog, config_dict = self.setup_events_with_dialog(tgui, "refine")
        assert config_dict["refine_template_show_sliding_window_fit_plot"] == dialog.dia.show_sliding_window_fit_checkbox.isChecked()
        assert config_dict["refine_template_show_corr_plot"] == dialog.dia.show_detection_threshold_checkbox.isChecked()

    @pytest.mark.parametrize("dialog_type", ["refine", "analyse"])
    def test_shared_events_template_refine_analyse_configs_on_init(self, tgui, dialog_type):

        dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)

        if config_dict["detection_threshold_type"] == "correlation":  # will always be as configs reset
            assert dialog.dia.detection_cutoff_combobox.currentIndex() == 0
            assert dialog.dia.detection_threshold_spinbox.value() == config_dict["corr_cutoff"]

        idx = 0 if config_dict["threshold_type"] == "manual" else "error!"
        assert dialog.dia.threshold_lower_combobox.currentIndex() == idx
        assert dialog.dia.threshold_lower_spinbox.value() == config_dict["linear_threshold_lower_value"]
        assert dialog.dia.hide_threshold_lower_from_plot_checkbox.isChecked() != config_dict["show_threshold_lower_on_plot"]

        assert dialog.dia.threshold_upper_groupbox.isChecked() == config_dict["threshold_upper_limit_on"]
        assert dialog.dia.threshold_upper_spinbox.value() == config_dict["threshold_upper_limit_value"]

    @pytest.mark.parametrize("dialog_type", ["refine", "analyse", "threshold"])
    def test_shared_refine_analyse_threshold_baseline_on_init(self, tgui, dialog_type):

        dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)

        idx = 0 if config_dict["baseline_type"] == "per_event" else "error!"
        assert dialog.dia.baseline_combobox.currentIndex() == idx
        assert dialog.dia.baseline_stackwidget.currentIndex() == idx
        assert dialog.dia.hide_baseline_from_plot_checkbox.isChecked() != config_dict["show_baseline_on_plot"]

    # Test all events dialog widges / configs
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_generate_template_store_templates(self, tgui):
        """
        Delete template not tested as requires focus.

        Load templates, store, reload and check they remain in store, load into GUI and check all spinboxes are updated
        """
        dialog, config_dict = self.setup_events_with_dialog(tgui, "generate")
        tgui.left_mouse_click(dialog.dia.load_save_template_button)
        store_dialog = dialog.load_save_template_dialog

        template_1 = [-100, 101, 51, 55,  100]
        template_2 = [-5,   8.3, 5,  99,  120]
        template_3 = [1.5,  2.5, 10, 20,  25]

        for idx, template_params in enumerate([template_1, template_2, template_3]):

            dialog.dia.choose_template_combobox.setCurrentIndex(idx)
            self.set_template_generate(tgui, dialog, *template_params)

            tgui.enter_number_into_spinbox(store_dialog.dia.save_template_name_lineedit,
                                           "template_" + str(idx + 1))

            tgui.left_mouse_click(store_dialog.dia.save_current_template_button)

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_generate")  # TODO: own function
        dialog, config_dict = self.setup_events_with_dialog(tgui, "generate")
        tgui.left_mouse_click(dialog.dia.load_save_template_button)
        store_dialog = dialog.load_save_template_dialog

        model_data = store_dialog.model.model_data
        assert model_data[0] == ["template_1  ", "template_2  ", "template_3  "]
        assert model_data[1] == [-100, -5, 1.5]
        assert model_data[2] == [101, 8.3, 2.5]
        assert model_data[3] == [51, 5, 10]
        assert model_data[4] == [55, 99, 20]
        assert model_data[5] == [100, 120, 25]

        for idx, template_params in enumerate([template_1, template_2, template_3]):

            store_dialog.dia.tableView.selectRow(idx)

            QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
            tgui.left_mouse_click(store_dialog.dia.load_selected_template_button)

            assert dialog.dia.b0_spinbox.value() == template_params[0]
            assert dialog.dia.b1_spinbox.value() == template_params[1]
            assert dialog.dia.rise_spinbox.value() == template_params[2]
            assert dialog.dia.decay_spinbox.value() == template_params[3]
            assert dialog.dia.width_spinbox.value() == template_params[4]

        tgui.shutdown()

    def set_template_generate(self, tgui, dialog, b0=None, b1=None, rise=None, decay=None, width=None):
        """
        """
        if b0 is not None:
            tgui.enter_number_into_spinbox(dialog.dia.b0_spinbox, b0)

        if b1 is not None:
            tgui.enter_number_into_spinbox(dialog.dia.b1_spinbox, b1)

        if rise is not None:
            tgui.enter_number_into_spinbox(dialog.dia.rise_spinbox, rise)

        if decay is not None:
            tgui.enter_number_into_spinbox(dialog.dia.decay_spinbox, decay)

        if width is not None:
            tgui.enter_number_into_spinbox(dialog.dia.width_spinbox, width)

    def test_coefficient_boxes_on_event_template_dialogs(self, tgui):

        for template in ["1", "2", "3"]:
            dialog, __ = self.setup_events_with_dialog(tgui, "generate", load_file=True)
            dialog.dia.choose_template_combobox.setCurrentIndex(int(template) - 1)

            dialog, __ = self.setup_events_with_dialog(tgui, "refine")
            assert dialog.dia.direction_label.text() == "Selected Template: {0}. Direction: Negative".format(template)
            dialog, __ = self.setup_events_with_dialog(tgui, "analyse", load_file=False)
            assert dialog.dia.direction_label.text() == "Selected Template: {0}. Direction: Negative".format(template)

            dialog, __ = self.setup_events_with_dialog(tgui, "generate", load_file=False)
            tgui.click_checkbox(dialog.dia.event_direction_is_positive_checkbox)
            self.set_template_generate(tgui, dialog, rise=101, decay=100, width=500)

            dialog, __ = self.setup_events_with_dialog(tgui, "refine", load_file=False)
            assert dialog.dia.direction_label.text() == "Selected Template: {0}. Direction: Positive".format(template)
            assert float(dialog.dia.rise_lineedit.text()) == 101
            assert float(dialog.dia.decay_lineedit.text()) == 100
            assert float(dialog.dia.window_length_lineedit.text()) == 500

            dialog, __ = self.setup_events_with_dialog(tgui, "analyse", load_file=False)
            assert dialog.dia.direction_label.text() == "Selected Template: {0}. Direction: Positive".format(template)
            assert float(dialog.dia.rise_lineedit.text()) == 101
            assert float(dialog.dia.decay_lineedit.text()) == 100
            assert float(dialog.dia.window_length_lineedit.text()) == 500

    def test_events_refine_configs(self, tgui):

        dialog, config_dict = self.setup_events_with_dialog(tgui, "refine")
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        tgui.click_checkbox(dialog.dia.show_sliding_window_fit_checkbox)
        assert config_dict["refine_template_show_sliding_window_fit_plot"] is True
        tgui.click_checkbox(dialog.dia.show_detection_threshold_checkbox)
        assert config_dict["refine_template_show_corr_plot"] is True

        # Reload and Check
        tgui = close_and_reload_for_defaults_check(tgui, save_="events_refine")
        dialog, config_dict = self.setup_events_with_dialog(tgui, "refine")
        tgui.left_mouse_click(dialog.dia.fit_all_events_button)

        assert dialog.dia.show_sliding_window_fit_checkbox.isChecked()
        assert config_dict["refine_template_show_sliding_window_fit_plot"] is True
        assert dialog.dia.show_detection_threshold_checkbox.isChecked()
        assert config_dict["refine_template_show_corr_plot"] is True

        # Change, reload and check checkboxes
        tgui.click_checkbox(dialog.dia.show_sliding_window_fit_checkbox)
        assert config_dict["refine_template_show_sliding_window_fit_plot"] is False
        tgui.click_checkbox(dialog.dia.show_detection_threshold_checkbox)
        assert config_dict["refine_template_show_corr_plot"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_refine")
        dialog, config_dict = self.setup_events_with_dialog(tgui, "refine")

        assert not dialog.dia.show_sliding_window_fit_checkbox.isChecked()
        assert config_dict["refine_template_show_sliding_window_fit_plot"] is False
        assert not dialog.dia.show_detection_threshold_checkbox.isChecked()
        assert config_dict["refine_template_show_corr_plot"] is False

        tgui.shutdown()

    @pytest.mark.parametrize("dialog_type", ["refine", "analyse", "threshold"])
    def test_omit_start_stop_times(self, tgui, dialog_type):

        dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)

        assert config_dict["omit_start_stop_times"] is None

        tgui.enter_numbers_into_omit_times_table(dialog, [[0.5, 0.7], [2.5, 5.5]])  # TODO: make times a var

        assert len(config_dict["omit_start_stop_times"]) == 2
        assert config_dict["omit_start_stop_times"][0] == [0.5, 0.7]
        assert config_dict["omit_start_stop_times"][1] == [2.5, 5.5]

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_" + dialog_type)
        dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)

        assert len(config_dict["omit_start_stop_times"]) == 2
        assert config_dict["omit_start_stop_times"][0] == [0.5, 0.7]
        assert config_dict["omit_start_stop_times"][1] == [2.5, 5.5]

        tgui.shutdown()

    # Events Panels
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_panel_widgets_on_init(self, tgui):

        self.setup_events_with_dialog(tgui, dialog_type=[])

        assert self.get_widget(tgui, "template", "decay_search_period_s").value() == self.get_widget(tgui, "threshold", "decay_search_period_s").value() == \
               tgui.mw.cfgs.events["decay_search_period_s"] * 1000
        assert self.get_widget(tgui, "template", "amplitude_threshold").value() == self.get_widget(tgui, "threshold", "amplitude_threshold").value() == \
               tgui.mw.cfgs.events["amplitude_threshold"]

        assert self.get_widget(tgui, "template", "average_peak_points_on").isChecked() == self.get_widget(tgui, "threshold", "average_peak_points_on").isChecked() == tgui.mw.cfgs.events["average_peak_points"]["on"]
        
        assert self.get_widget(tgui, "template", "average_peak_points_value_s").value() == self.get_widget(tgui, "threshold", "average_peak_points_value_s").value() == tgui.mw.cfgs.events["average_peak_points"]["value_s"] * 1000

        assert self.get_widget(tgui, "template", "area_under_curve_on").isChecked() == self.get_widget(tgui, "threshold", "area_under_curve_on").isChecked() == tgui.mw.cfgs.events["area_under_curve"]["on"]
     
        assert  self.get_widget(tgui, "template", "area_under_curve_value_pa_ms").value() == self.get_widget(tgui, "threshold", "area_under_curve_value_pa_ms").value() == tgui.mw.cfgs.events["area_under_curve"]["value_pa_ms"]

        assert self.get_widget(tgui, "template", "baseline_search_period_s").value() == self.get_widget(tgui, "threshold", "baseline_search_period_s").value() ==  tgui.mw.cfgs.events["baseline_search_period_s"] * 1000

        assert self.get_widget(tgui, "template", "average_baseline_points_on").isChecked() == self.get_widget(tgui, "threshold", "average_baseline_points_on").isChecked() == tgui.mw.cfgs.events["average_baseline_points"]["on"]

        assert self.get_widget(tgui, "template", "average_baseline_points_value_s").value() == self.get_widget(tgui, "threshold", "average_baseline_points_value_s").value() == tgui.mw.cfgs.events["average_baseline_points"]["value_s"] * 1000

        assert tgui.mw.mw.events_threshold_peak_direction_combobox.currentIndex() == 1
        assert tgui.mw.mw.events_threshold_local_maximum_period_spinbox.value() == tgui.mw.cfgs.events["threshold_local_maximum_period_s"] * 1000

        # tgui.shutdown()

    @pytest.mark.parametrize("ordered_widgets", [["template", "threshold"], ["template", "template"]])
    def test_events_panel_widgets(self, tgui, ordered_widgets):

        first, second = ordered_widgets
        self.setup_events_with_dialog(tgui,
                                      dialog_type="analyse" if first == "template" else "threshold")

        tgui.enter_number_into_spinbox(self.get_widget(tgui, first, "decay_search_period_s"),
                                       2)
        assert self.get_widget(tgui, second, "decay_search_period_s").value() == 2, "decay search first spinbox"
        assert tgui.mw.cfgs.events["decay_search_period_s"] * 1000 == 2, "deca search first cfg"

        tgui.enter_number_into_spinbox(self.get_widget(tgui, first, "amplitude_threshold"),
                                       3)

        assert self.get_widget(tgui, second, "amplitude_threshold").value() == 3, "amplitude_threshold first spinbox"
        assert tgui.mw.cfgs.events["amplitude_threshold"] == 3, "amplitude_threshold first cfg"

        tgui.switch_checkbox(self.get_widget(tgui, first, "average_peak_points_on"),
                             on=True)
        assert self.get_widget(tgui, second, "average_peak_points_on").isChecked()
        assert tgui.mw.cfgs.events["average_peak_points"]["on"]

        tgui.enter_number_into_spinbox(self.get_widget(tgui, first, "average_peak_points_value_s"),
                                       4)

        assert self.get_widget(tgui, second, "average_peak_points_value_s").value() == 4, "average_peak_points first spinbox"
        assert tgui.mw.cfgs.events["average_peak_points"]["value_s"] * 1000 == 4, "average_peak_points first cfg"

        tgui.switch_checkbox(self.get_widget(tgui, first, "area_under_curve_on"), on=True)
        assert tgui.mw.cfgs.events["area_under_curve"]["on"]

        tgui.enter_number_into_spinbox(self.get_widget(tgui, first, "area_under_curve_value_pa_ms"),
                                       165)
        assert self.get_widget(tgui, second, "area_under_curve_value_pa_ms").value() == 165, "area_under_curve first spinbox"
        assert tgui.mw.cfgs.events["area_under_curve"]["value_pa_ms"] == 165

        tgui.enter_number_into_spinbox(self.get_widget(tgui, first, "baseline_search_period_s"),
                                       5)
        assert self.get_widget(tgui, second, "baseline_search_period_s").value() == 5, "baseline_search_period_s first spinbox"
        assert tgui.mw.cfgs.events["baseline_search_period_s"] * 1000 == 5, "baseline_search_period_s first cfg"

        tgui.switch_checkbox(self.get_widget(tgui, first, "average_baseline_points_on"),
                             on=True)
        assert self.get_widget(tgui, second, "average_baseline_points_on").isChecked()
        assert tgui.mw.cfgs.events["average_baseline_points"]["on"]

        tgui.enter_number_into_spinbox(self.get_widget(tgui, first, "average_baseline_points_value_s"),
                                       6)
        assert self.get_widget(tgui, second, "average_baseline_points_value_s").value() == 6, "average_baseline_points_value_s first spinbox"
        assert tgui.mw.cfgs.events["average_baseline_points"]["value_s"] * 1000 == 6, "average_baseline_points_value_s first cfg"

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_" + first + "_panel")

        self.setup_events_with_dialog(tgui,
                                      dialog_type="analyse" if first == "template" else "threshold")

        assert self.get_widget(tgui, first, "decay_search_period_s").value() == 2, "decay search first spinbox"
        assert self.get_widget(tgui, second, "decay_search_period_s").value() == 2, "decay search second spinbox"
        assert tgui.mw.cfgs.events["decay_search_period_s"] * 1000 == 2, "deca search first cfg"

        assert self.get_widget(tgui, first, "amplitude_threshold").value() == 3, "amplitude_threshold first spinbox"
        assert self.get_widget(tgui, second, "amplitude_threshold").value() == 3, "amplitude_threshold second spinbox"
        assert tgui.mw.cfgs.events["amplitude_threshold"] == 3, "amplitude_threshold first cfg"

        assert self.get_widget(tgui, first, "average_peak_points_on").isChecked()
        assert self.get_widget(tgui, second, "average_peak_points_on").isChecked()
        assert tgui.mw.cfgs.events["average_peak_points"]["on"]

        assert self.get_widget(tgui, first, "average_peak_points_value_s").value() == 4, "average_peak_points first spinbox"
        assert self.get_widget(tgui, second, "average_peak_points_value_s").value() == 4, "average_peak_points second spinbox"
        assert tgui.mw.cfgs.events["average_peak_points"]["value_s"] * 1000 == 4, "average_peak_points first cfg"

        assert self.get_widget(tgui, first, "area_under_curve_on").isChecked()
        assert self.get_widget(tgui, second, "area_under_curve_on").isChecked()
        assert tgui.mw.cfgs.events["area_under_curve"]["on"]

        assert self.get_widget(tgui, second, "area_under_curve_value_pa_ms").value() == 165, "area_under_curve first spinbox"
        assert self.get_widget(tgui, second, "area_under_curve_value_pa_ms").value() == 165, "area_under_curve first spinbox"
        assert tgui.mw.cfgs.events["area_under_curve"]["value_pa_ms"] == 165,  "area_under_curve first cfgs"

        assert self.get_widget(tgui, first, "baseline_search_period_s").value() == 5, "baseline_search_period_s first spinbox"
        assert self.get_widget(tgui, second, "baseline_search_period_s").value() == 5, "baseline_search_period_s first spinbox"
        assert tgui.mw.cfgs.events["baseline_search_period_s"] * 1000 == 5, "baseline_search_period_s first cfg"

        assert self.get_widget(tgui, first, "average_baseline_points_on").isChecked()
        assert self.get_widget(tgui, second, "average_baseline_points_on").isChecked()
        assert tgui.mw.cfgs.events["average_baseline_points"]["on"]

        assert self.get_widget(tgui, first, "average_baseline_points_value_s").value() == 6, "average_baseline_points_value_s first spinbox"
        assert self.get_widget(tgui, second, "average_baseline_points_value_s").value() == 6, "average_baseline_points_value_s second spinbox"
        assert tgui.mw.cfgs.events["average_baseline_points"]["value_s"] * 1000 == 6, "average_baseline_points_value_s first cfg"

        # Flip checkboxes to False, reload for final check
        tgui.switch_checkbox(self.get_widget(tgui, first, "average_peak_points_on"),
                             on=False)
        assert self.get_widget(tgui, second, "average_peak_points_on").isChecked() is False
        assert tgui.mw.cfgs.events["average_peak_points"]["on"] is False

        tgui.switch_checkbox(self.get_widget(tgui, first, "area_under_curve_on"), on=False)
        assert self.get_widget(tgui, second, "area_under_curve_on").isChecked() is False
        assert tgui.mw.cfgs.events["area_under_curve"]["on"] is False

        tgui.switch_checkbox(self.get_widget(tgui, first, "average_baseline_points_on"),
                             on=False)
        assert self.get_widget(tgui, second, "average_baseline_points_on").isChecked() is False
        assert tgui.mw.cfgs.events["average_baseline_points"]["on"] is False

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_" + first + "_panel")

        self.setup_events_with_dialog(tgui,
                                      dialog_type="analyse" if first == "template" else "threshold")

        assert self.get_widget(tgui, first, "average_peak_points_on").isChecked() is False
        assert self.get_widget(tgui, second, "average_peak_points_on").isChecked() is False
        assert tgui.mw.cfgs.events["average_peak_points"]["on"] is False

        assert self.get_widget(tgui, first, "area_under_curve_on").isChecked() is False
        assert self.get_widget(tgui, second, "area_under_curve_on").isChecked() is False
        assert tgui.mw.cfgs.events["area_under_curve"]["on"] is False

        assert self.get_widget(tgui, first, "average_baseline_points_on").isChecked() is False
        assert self.get_widget(tgui, second, "average_baseline_points_on").isChecked() is False
        assert tgui.mw.cfgs.events["average_baseline_points"]["on"] is False

        tgui.shutdown()

    def test_misc_events_analysis_matched_config_on_init(self, tgui):
        dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                            dialog_type="misc_options")

        assert dialog.dia.interp_200khz_checkbox.isChecked() is config_dict["interp_200khz"]
        assert dialog.dia.rise_time_cutoff_low.value() == config_dict["rise_cutoff_low"]
        assert dialog.dia.rise_time_cutoff_high.value() == config_dict["rise_cutoff_high"]
        assert dialog.dia.decay_amplitude_perc_spinbox.value() == config_dict["decay_amplitude_percent"]
        assert dialog.dia.decay_period_smooth_spinbox.value() == config_dict["decay_period_smooth_s"]
        assert dialog.dia.dynamic_baseline_polynomial_order_spinbox.value() == config_dict["dynamic_curve_polynomial_order"]

        assert dialog.dia.template_matching_rise_coef_spinbox.value() == config_dict["cannonical_initial_biexp_coefficients"]["rise"]
        assert dialog.dia.template_matching_decay_coef_spinbox.value() == config_dict["cannonical_initial_biexp_coefficients"]["decay"]

    def test_misc_events_analysis_widgets(self, tgui):

        dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                            dialog_type="misc_options")

        assert config_dict["endpoint_search_method"] != "first_baseline_cross"
        tgui.set_combobox(dialog.dia.decay_endpoint_search_method_combobox, 1)
        assert config_dict["endpoint_search_method"] == "first_baseline_cross"

        assert config_dict["interp_200khz"] is False
        tgui.switch_checkbox(dialog.dia.interp_200khz_checkbox,
                             on=True)
        assert config_dict["interp_200khz"] is True

        assert config_dict["rise_cutoff_low"] != 15
        tgui.enter_number_into_spinbox(dialog.dia.rise_time_cutoff_low,
                                       15)
        assert config_dict["rise_cutoff_low"] == 15

        assert config_dict["rise_cutoff_high"] != 25
        tgui.enter_number_into_spinbox(dialog.dia.rise_time_cutoff_high,
                                       25)
        assert config_dict["rise_cutoff_high"] == 25

        assert config_dict["decay_amplitude_percent"] != 35
        tgui.enter_number_into_spinbox(dialog.dia.decay_amplitude_perc_spinbox,
                                       35)
        assert config_dict["decay_amplitude_percent"] == 35

        assert config_dict["max_slope"]["on"] is False
        tgui.switch_groupbox(dialog.dia.max_slope_groupbox,
                             on=True)
        assert config_dict["max_slope"]["on"]

        assert config_dict["max_slope"]["rise_num_samples"] != 45
        tgui.enter_number_into_spinbox(dialog.dia.max_slope_num_samples_rise_spinbox,
                                       45)
        assert config_dict["max_slope"]["rise_num_samples"] == 45

        assert config_dict["max_slope"]["decay_num_samples"] != 55
        tgui.enter_number_into_spinbox(dialog.dia.max_slope_num_samples_decay_spinbox,
                                       55)
        assert config_dict["max_slope"]["decay_num_samples"] == 55

        assert config_dict["max_slope"]["smooth"]["on"] is False
        tgui.switch_checkbox(dialog.dia.max_slope_smooth_checkbox,
                             on=True)
        assert config_dict["max_slope"]["smooth"]["on"]

        assert config_dict["max_slope"]["smooth"]["num_samples"] != 65
        tgui.enter_number_into_spinbox(dialog.dia.max_slope_smooth_spinbox,
                                       65)
        assert config_dict["max_slope"]["smooth"]["num_samples"] == 65

        assert config_dict["max_slope"]["use_baseline_crossing_endpoint"] is True
        tgui.switch_checkbox(dialog.dia.max_slope_use_first_baseline_crossing_checkbox,
                             on=False)
        assert config_dict["max_slope"]["use_baseline_crossing_endpoint"] is False

        assert config_dict["dynamic_curve_polynomial_order"] != 6
        tgui.enter_number_into_spinbox(dialog.dia.dynamic_baseline_polynomial_order_spinbox,
                                       6)
        assert config_dict["dynamic_curve_polynomial_order"] == 6

        assert config_dict["cannonical_initial_biexp_coefficients"]["rise"] != 65
        tgui.enter_number_into_spinbox(dialog.dia.template_matching_rise_coef_spinbox,
                                       65)
        assert config_dict["cannonical_initial_biexp_coefficients"]["rise"] == 65

        assert config_dict["cannonical_initial_biexp_coefficients"]["decay"] != 75
        tgui.enter_number_into_spinbox(dialog.dia.template_matching_decay_coef_spinbox,
                                       75)
        assert config_dict["cannonical_initial_biexp_coefficients"]["decay"] == 75

        assert config_dict["threshold_manual_selected_event"] is True
        tgui.switch_checkbox(dialog.dia.threshold_manually_selected_events_checkbox,
                             on=False)
        assert config_dict["threshold_manual_selected_event"] is False

        assert config_dict["show_auc_plot"] is False
        tgui.switch_checkbox(dialog.dia.show_area_under_curve_checkbox,
                             on=True)
        assert config_dict["show_auc_plot"] is True

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_misc_options")
        dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                            dialog_type="misc_options")

        assert dialog.dia.decay_endpoint_search_method_combobox.currentIndex() == 1
        assert config_dict["endpoint_search_method"] == "first_baseline_cross"

        assert dialog.dia.interp_200khz_checkbox.isChecked()
        assert config_dict["interp_200khz"] is True

        assert dialog.dia.rise_time_cutoff_low.value() == 15
        assert config_dict["rise_cutoff_low"] == 15

        assert dialog.dia.rise_time_cutoff_high.value() == 25
        assert config_dict["rise_cutoff_high"] == 25

        assert dialog.dia.decay_amplitude_perc_spinbox.value() == 35
        assert config_dict["decay_amplitude_percent"] == 35

        assert dialog.dia.max_slope_groupbox.isChecked()
        assert config_dict["max_slope"]["on"]

        assert dialog.dia.max_slope_num_samples_rise_spinbox.value()
        assert config_dict["max_slope"]["rise_num_samples"] == 45

        assert dialog.dia.max_slope_num_samples_decay_spinbox.value() == 55
        assert config_dict["max_slope"]["decay_num_samples"] == 55

        assert dialog.dia.max_slope_smooth_checkbox.isChecked()
        assert config_dict["max_slope"]["smooth"]["on"]

        assert dialog.dia.max_slope_smooth_spinbox.value() == 65
        assert config_dict["max_slope"]["smooth"]["num_samples"] == 65

        assert not dialog.dia.max_slope_use_first_baseline_crossing_checkbox.isChecked()
        assert config_dict["max_slope"]["use_baseline_crossing_endpoint"] is False

        assert not dialog.dia.threshold_manually_selected_events_checkbox.isChecked()
        assert config_dict["threshold_manual_selected_event"] is False

        assert dialog.dia.show_area_under_curve_checkbox.isChecked()
        assert config_dict["show_auc_plot"] is True

        assert dialog.dia.dynamic_baseline_polynomial_order_spinbox.value() == 6
        assert config_dict["dynamic_curve_polynomial_order"] == 6

        assert dialog.dia.template_matching_rise_coef_spinbox.value() == 65
        assert config_dict["cannonical_initial_biexp_coefficients"]["rise"] == 65

        assert dialog.dia.template_matching_decay_coef_spinbox.value() == 75
        assert config_dict["cannonical_initial_biexp_coefficients"]["decay"] == 75

        tgui.shutdown()

    def test_misc_events_do_not_fit_widgets(self, tgui):

        dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                            dialog_type="misc_options")

        tgui.set_combobox(dialog.dia.event_fit_method_combobox, 2)
        assert config_dict["decay_or_biexp_fit_method"] == "do_not_fit"
        tgui.enter_number_into_spinbox(dialog.dia.decay_period_smooth_spinbox, 15)
        assert config_dict["decay_period_smooth_s"] == 15 / 1000

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_misc_options")
        dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                            dialog_type="misc_options")

        assert dialog.dia.event_fit_method_combobox.currentIndex() == 2
        assert config_dict["decay_or_biexp_fit_method"] == "do_not_fit"
        assert dialog.dia.decay_period_smooth_spinbox.value() == 15
        assert config_dict["decay_period_smooth_s"] == 15 / 1000

        tgui.shutdown()

# Frequency Data Options
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_events_frequency_data_options_on_init(self, tgui):
        dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                            dialog_type="frequency_data_options")
        config_dict = config_dict["frequency_data_options"]

        idx = 0 if config_dict["x_axis_display"] == "bin_centre" else "error"
        assert dialog.dia.binning_method_combobox.currentIndex() == idx

        assert dialog.dia.custom_bin_number_spinbox.value() == 1

        idx = 0 if config_dict["binning_method"] == "auto" else "error"
        assert dialog.dia.custom_binsizes_combobox.currentIndex() == idx

        assert dialog.dia.custom_binsizes_spinbox.value() == 0

        idx = 0 if config_dict["plot_type"] == "cum_prob" else "error"
        assert dialog.dia.x_axis_display_combobox.currentIndex() == idx

    def test_events_frequency_data_options_all(self, tgui):

        dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                            dialog_type="frequency_data_options")
        config_dict = config_dict["frequency_data_options"]

        params = ["frequency", "amplitude", "rise_time", "decay_amplitude_percent", "decay_tau", "biexp_rise", "biexp_decay", "grouped_ks_analysis"]

        # Binning Method Combobox:
        tgui.set_combobox(dialog.dia.binning_method_combobox, 0)
        assert config_dict["binning_method"] == "auto"
        assert not dialog.dia.custom_bin_number_spinbox.isEnabled()

        tgui.set_combobox(dialog.dia.binning_method_combobox, 1)
        assert config_dict["binning_method"] == "custom_binnum"
        assert dialog.dia.custom_bin_number_spinbox.isEnabled()
        tgui.enter_number_into_spinbox(dialog.dia.custom_bin_number_spinbox, 5)
        assert config_dict["custom_binnum"] == 5

        tgui.set_combobox(dialog.dia.binning_method_combobox, 2)
        assert config_dict["binning_method"] == "custom_binsize"
        assert not dialog.dia.custom_bin_number_spinbox.isEnabled()
        assert dialog.dia.custom_binsizes_combobox.isEnabled()
        assert dialog.dia.custom_binsizes_spinbox.isEnabled()

        for idx, parameter in zip(range(len(params)),
                                  params):

            tgui.set_combobox(dialog.dia.custom_binsizes_combobox, idx)
            tgui.enter_number_into_spinbox(dialog.dia.custom_binsizes_spinbox, idx + 1)
            assert config_dict["custom_binsize"][parameter] == idx + 1

            key = list(config_dict["custom_binsize"].keys())[idx]
            assert key == parameter

        tgui.set_combobox(dialog.dia.binning_method_combobox, 3)
        assert config_dict["binning_method"] == "num_events_divided_by"
        assert dialog.dia.custom_bin_number_spinbox.isEnabled()
        assert not dialog.dia.custom_binsizes_combobox.isEnabled()
        assert not dialog.dia.custom_binsizes_spinbox.isEnabled()
        tgui.enter_number_into_spinbox(dialog.dia.custom_bin_number_spinbox, 6)
        assert config_dict["divide_by_number"] == 6

        for idx, display_ in zip(range(3),
                                ["bin_centre", "left_edge", "right_edge"]):
            tgui.set_combobox(dialog.dia.x_axis_display_combobox, idx)
            assert config_dict["x_axis_display"] == display_

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_misc_options")
        dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                            dialog_type="frequency_data_options")
        config_dict = config_dict["frequency_data_options"]

        assert dialog.dia.binning_method_combobox.currentIndex() == 3
        assert config_dict["binning_method"] == "num_events_divided_by"

        assert dialog.dia.x_axis_display_combobox.currentIndex() == 2
        assert config_dict["x_axis_display"] == "right_edge"

        tgui.set_combobox(dialog.dia.binning_method_combobox, 2)

        for idx, parameter in zip(range(len(params)),
                                  params):

            tgui.set_combobox(dialog.dia.custom_binsizes_combobox, idx)
            assert dialog.dia.custom_binsizes_spinbox.value() == idx + 1
            assert config_dict["custom_binsize"][parameter] == idx + 1

        tgui.set_combobox(dialog.dia.binning_method_combobox, 1)
        assert dialog.dia.custom_bin_number_spinbox.value() == 5
        assert config_dict["custom_binnum"] == 5

        tgui.shutdown()

# Frequency Plot Options
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_frequency_plot_configs_defaults(self, tgui):
        """
        """
        tgui, dialog, config_dict = self.setup_and_show_frequency_plot_options(tgui)

        assert config_dict["marker_color"] == "blue"
        assert config_dict["marker_line_color"] == "light_blue"
        assert config_dict["line_color"] == "light_blue"
        assert config_dict["marker_size_cum_prob"] == 8
        assert config_dict["marker_size_histogram"] == 0.02
        assert config_dict["marker_line_width"] == 1
        assert config_dict["line_width"] == 2
        assert config_dict["show_gridlines"] is True

        for parameter in ["frequency", "amplitude", "rise_time", "decay_amplitude_percent", "decay_tau", "biexp_rise", "biexp_decay"]:

            assert config_dict[parameter]["set_x_limits"] == 0
            assert config_dict[parameter]["x_limits_min"] == 0
            assert config_dict[parameter]["x_limits_max"] == 0
            assert config_dict[parameter]["set_y_limits"] == 0
            assert config_dict[parameter]["y_limits_min"] == 0
            assert config_dict[parameter]["y_limits_max"] == 0

        tgui.shutdown()

    def test_frequency_plot_configs_on_init(self, tgui):
        """
        Because there is a limit on min 5 events, to avoid running analysis open the dialog backend.
        """
        tgui, dialog, config_dict = self.setup_and_show_frequency_plot_options(tgui)

        idx = 0 if config_dict["marker_color"] == "blue" else "error"
        assert dialog.dia.marker_color_combobox.currentIndex() == idx
        idx = 1 if config_dict["marker_line_color"] == "light_blue" else "error"
        assert dialog.dia.marker_line_color_combobox.currentIndex() == idx
        idx = 1 if config_dict["line_color"] == "light_blue" else "error"
        assert dialog.dia.line_color_combobox.currentIndex() == idx
        assert dialog.dia.marker_size_cum_prob_spinbox.value() == config_dict["marker_size_cum_prob"]
        assert dialog.dia.marker_size_histogram_spinbox.value() == config_dict["marker_size_histogram"]
        assert dialog.dia.marker_line_width_spinbox.value() == config_dict["marker_line_width"]
        assert dialog.dia.line_width_spinbox.value() == config_dict["line_width"]
        assert dialog.dia.show_gridlines_checkbox.isChecked() is config_dict["show_gridlines"]

        for combobox_idx, parameter in zip(range(8),
                                           ["frequency", "amplitude", "rise_time", "decay_amplitude_percent", "decay_tau", "biexp_rise", "biexp_decay"]):

            tgui.set_combobox(dialog.dia.axis_limits_parameter_combobox,
                              combobox_idx)
            assert dialog.dia.set_x_limits_checkbox.isChecked() is config_dict[parameter]["set_x_limits"]
            assert dialog.dia.min_x_limits_spinbox.value() == config_dict[parameter]["x_limits_min"]
            assert dialog.dia.max_x_limits_spinbox.value() == config_dict[parameter]["x_limits_max"]
            assert dialog.dia.set_y_limits_checkbox.isChecked() is config_dict[parameter]["set_y_limits"]
            assert dialog.dia.min_y_limits_spinbox.value() == config_dict[parameter]["y_limits_min"]
            assert dialog.dia.max_y_limits_spinbox.value() == config_dict[parameter]["y_limits_max"]

        tgui.shutdown()

    def test_frequency_plot_configs_all(self, tgui):

        tgui, dialog, config_dict = self.setup_and_show_frequency_plot_options(tgui)

        # Color comboboxes
        for idx, color in zip(range(9),
                              dialog.marker_color_order):

            tgui.set_combobox(dialog.dia.marker_color_combobox, idx)
            assert config_dict["marker_color"] == color

            tgui.set_combobox(dialog.dia.marker_line_color_combobox, idx)
            assert config_dict["marker_line_color"] == color

            tgui.set_combobox(dialog.dia.line_color_combobox, idx)
            assert config_dict["line_color"] == color

        # Marker sizes
        tgui.enter_number_into_spinbox(dialog.dia.marker_size_cum_prob_spinbox, 4)
        assert config_dict["marker_size_cum_prob"] == 4

        tgui.enter_number_into_spinbox(dialog.dia.marker_size_histogram_spinbox, 5)
        assert config_dict["marker_size_histogram"] == 5

        tgui.enter_number_into_spinbox(dialog.dia.marker_line_width_spinbox, 6)
        assert config_dict["marker_line_width"] == 6

        tgui.enter_number_into_spinbox(dialog.dia.line_width_spinbox, 7)
        assert config_dict["line_width"] == 7

        tgui.switch_checkbox(dialog.dia.show_gridlines_checkbox, on=False)
        assert config_dict["show_gridlines"] is False

        # set limits checkboxes
        for combobox_idx, parameter in zip(range(7),
                                           ["frequency", "amplitude", "rise_time", "decay_amplitude_percent", "decay_tau", "biexp_rise", "biexp_decay"]):

            tgui.set_combobox(dialog.dia.axis_limits_parameter_combobox,
                              combobox_idx)
            tgui.switch_checkbox(dialog.dia.set_x_limits_checkbox, on=True)
            assert config_dict[parameter]["set_x_limits"] is True
            tgui.enter_number_into_spinbox(dialog.dia.min_x_limits_spinbox, 1 + combobox_idx)
            assert config_dict[parameter]["x_limits_min"] == 1 + combobox_idx
            tgui.enter_number_into_spinbox(dialog.dia.max_x_limits_spinbox, 2 + combobox_idx)
            assert config_dict[parameter]["x_limits_max"] == 2 + combobox_idx

            tgui.switch_checkbox(dialog.dia.set_y_limits_checkbox, on=True)
            assert config_dict[parameter]["set_y_limits"] is True
            tgui.enter_number_into_spinbox(dialog.dia.min_y_limits_spinbox, 4 + combobox_idx)
            assert config_dict[parameter]["y_limits_min"] == 4 + combobox_idx
            tgui.enter_number_into_spinbox(dialog.dia.max_y_limits_spinbox, 5 + combobox_idx)
            assert config_dict[parameter]["y_limits_max"] == 5 + combobox_idx

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_frequency_plot_options")
        __, dialog, config_dict = self.setup_and_show_frequency_plot_options(tgui,
                                                                             reload_tgui=False,
                                                                             reset_all_configs=False)

        # Color comboboxes, left at the last idx point from the above loop
        assert dialog.dia.marker_color_combobox.currentIndex() == 8
        assert config_dict["marker_color"] == "no_fill"

        assert dialog.dia.marker_line_color_combobox.currentIndex() == 8
        assert config_dict["marker_line_color"] == "no_fill"

        assert dialog.dia.marker_line_color_combobox.currentIndex() == 8
        assert config_dict["line_color"] == "no_fill"

        # Marker sizes
        assert dialog.dia.marker_size_cum_prob_spinbox.value() == 4
        assert config_dict["marker_size_cum_prob"] == 4

        assert dialog.dia.marker_size_histogram_spinbox.value() == 5
        assert config_dict["marker_size_histogram"] == 5

        assert dialog.dia.marker_line_width_spinbox.value() == 6
        assert config_dict["marker_line_width"] == 6

        assert dialog.dia.line_width_spinbox.value() == 7
        assert config_dict["line_width"] == 7

        assert dialog.dia.show_gridlines_checkbox.isChecked() is False
        assert config_dict["show_gridlines"] is False

        # set limits checkboxes
        for combobox_idx, parameter in zip(range(7),
                                           ["frequency", "amplitude", "rise_time", "decay_amplitude_percent", "decay_tau", "biexp_rise", "biexp_decay"]):

            tgui.set_combobox(dialog.dia.axis_limits_parameter_combobox,
                              combobox_idx)
            tgui.switch_checkbox(dialog.dia.set_x_limits_checkbox, on=True)
            assert config_dict[parameter]["set_x_limits"] is True

            assert dialog.dia.min_x_limits_spinbox.value() == 1 + combobox_idx
            assert config_dict[parameter]["x_limits_min"] == 1 + combobox_idx

            assert dialog.dia.max_x_limits_spinbox.value() == 2 + combobox_idx
            assert config_dict[parameter]["x_limits_max"] == 2 + combobox_idx

            assert dialog.dia.set_y_limits_checkbox.isChecked()
            assert config_dict[parameter]["set_y_limits"] is True

            assert dialog.dia.min_y_limits_spinbox.value() == 4 + combobox_idx
            assert config_dict[parameter]["y_limits_min"] == 4 + combobox_idx

            assert dialog.dia.max_y_limits_spinbox.value() == 5 + combobox_idx
            assert config_dict[parameter]["y_limits_max"] == 5 + combobox_idx

        tgui.shutdown()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Events Run Settings
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_event_template_kinetics_run_settings(self, tgui):
        """
        Key test of all run settings. TODO: the whole run settings system can removed, it is vestigal from when
        these analysis were run in a qThread before it was removed due to bugs with threading and scipy.
        
        Only test template panel as we know threshold works the same as above
        """
        tgui = setup_artificial_event(tgui)
        tgui.set_analysis_type("events_template_matching")

#       Set All Options
#       ----------------------------------------------------------------------------------------------------------------------------------------------
        
        tgui.mw.mw.actionEvents_Analyis_Options.trigger()
        options_dialog = tgui.mw.dialogs["events_analysis_options"]

        # Misc
        tgui.set_combobox(options_dialog.dia.decay_endpoint_search_method_combobox, 1) #
        tgui.switch_checkbox(options_dialog.dia.interp_200khz_checkbox, on=True)
        tgui.enter_number_into_spinbox(options_dialog.dia.rise_time_cutoff_low, "30")
        tgui.enter_number_into_spinbox(options_dialog.dia.rise_time_cutoff_high, "65")
        tgui.enter_number_into_spinbox(options_dialog.dia.decay_amplitude_perc_spinbox, "50")
        tgui.switch_groupbox(options_dialog.dia.max_slope_groupbox, on=True)
        tgui.enter_number_into_spinbox(options_dialog.dia.max_slope_num_samples_rise_spinbox, "11")
        tgui.enter_number_into_spinbox(options_dialog.dia.max_slope_num_samples_decay_spinbox, "12")
        tgui.switch_checkbox(options_dialog.dia.max_slope_smooth_checkbox, on=True)
        tgui.enter_number_into_spinbox(options_dialog.dia.max_slope_smooth_spinbox, "14")
        tgui.switch_checkbox(options_dialog.dia.max_slope_use_first_baseline_crossing_checkbox, on=True)
        tgui.switch_checkbox(options_dialog.dia.threshold_manually_selected_events_checkbox, on=False)

        tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 1)
        tgui.switch_checkbox(options_dialog.dia.biexp_adjust_start_point_for_bounds_checkbox, on=True)
        tgui.enter_number_into_spinbox(options_dialog.dia.biexp_adjust_start_point_for_bounds_spinbox, 55)
        tgui.switch_checkbox(options_dialog.dia.biexp_exclude_outside_of_bounds_checkbox, on=True)
        tgui.enter_number_into_spinbox(options_dialog.dia.biexp_min_rise_spinbox, 0.001)
        tgui.enter_number_into_spinbox(options_dialog.dia.biexp_max_rise_spinbox, 100)
        tgui.enter_number_into_spinbox(options_dialog.dia.biexp_min_decay_spinbox, 0.002)
        tgui.enter_number_into_spinbox(options_dialog.dia.biexp_max_decay_spinbox, 200)

        # Panel
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_decay_search_period_spinbox, "10")  # decay_search_period_s
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_amplitude_threshold_spinbox, "5")  # amplitude_threshold
        tgui.switch_checkbox(tgui.mw.mw.events_template_average_peak_checkbox, on=True)  # average_peak_points on
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_average_peak_spinbox, "20")  # average_peak_points value_s
        tgui.switch_checkbox(tgui.mw.mw.events_template_area_under_curve_checkbox, on=True)  # area_under_curve on
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_area_under_curve_spinbox, "3")  # area_under_curve value_s
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_baseline_search_period_spinbox, "25")  # baseline_search_period_s
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_average_baseline_spinbox, "30")  # average_baseline_points value_s
        tgui.switch_checkbox(tgui.mw.mw.events_template_average_baseline_checkbox, on=False)  # average_baseline_points on

        # Dialog
        tgui.left_mouse_click(tgui.mw.mw.events_template_analyse_all_button)
        analyse_dialog = tgui.mw.dialogs["template_analyse_events"]
        tgui.set_combobox(analyse_dialog.dia.detection_cutoff_combobox, 1)  # event detection type == detection criteria
        tgui.enter_number_into_spinbox(analyse_dialog.dia.detection_threshold_spinbox, 1.4)
        tgui.set_combobox(analyse_dialog.dia.threshold_lower_combobox, 3)  # threshold_type == rms
        tgui.enter_number_into_spinbox(analyse_dialog.dia.threshold_lower_spinbox, 0.2)  # rms_lower_threshold_n_times
        tgui.switch_groupbox(analyse_dialog.dia.threshold_upper_groupbox, on=True)  # threshold_upper_limit_on
        tgui.enter_number_into_spinbox(analyse_dialog.dia.threshold_upper_spinbox, -72)  # threshold_upper_limit_value
        tgui.set_combobox(analyse_dialog.dia.baseline_combobox, 2)  # baseline_type
        tgui.enter_number_into_spinbox(analyse_dialog.dia.baseline_spinbox, "-0.05")  # curved_baseline_displacement

        curved_baseline = analyse_dialog.curved_baseline_w_displacement
        analyse_dialog.process_rms_threshold_lower(curved_baseline)
        run_settings = tgui.mw.loaded_file.get_event_detection_run_settings("template_analyse")

#       Test All Options
#       ----------------------------------------------------------------------------------------------------------------------------------------------

        # Analysis Options
        assert run_settings["endpoint_search_method"] == "first_baseline_cross"
        assert run_settings["interp_200khz"] is True
        assert run_settings["rise_cutoff_low"] == 30
        assert run_settings["rise_cutoff_high"] == 65
        assert run_settings["decay_amplitude_percent"] == 50

        assert run_settings["max_slope"]["on"]
        assert run_settings["max_slope"]["rise_num_samples"] == 11
        assert run_settings["max_slope"]["decay_num_samples"] == 12
        assert run_settings["max_slope"]["smooth"]["on"]
        assert run_settings["max_slope"]["smooth"]["num_samples"] == 14
        assert run_settings["max_slope"]["use_baseline_crossing_endpoint"] is True
        assert "threshold_manual_selected_event" not in run_settings

        assert run_settings["decay_or_biexp_fit_method"] == "biexp"
        assert run_settings["biexp_fit"]["adjust_startpoint_bounds_on"] is False  # this is turned off by exclude bounds
        assert run_settings["biexp_fit"]["adjust_startpoint_bounds_value"] == 55
        assert run_settings["biexp_fit"]["exclude_if_params_not_in_bounds"] is True
        assert run_settings["biexp_fit"]["rise_cutoff_min"] == 0.001
        assert run_settings["biexp_fit"]["rise_cutoff_max"] == 100
        assert run_settings["biexp_fit"]["decay_cutoff_min"] == 0.002
        assert run_settings["biexp_fit"]["decay_cutoff_max"] == 200

        # Panel
        assert run_settings["decay_search_period_s"] == 10 / 1000, "decay_search_period_s"
        assert run_settings["amplitude_threshold"] == 5, "amplitude_threshold"
        assert run_settings["average_peak_points"]["on"], "average_peak_points_on"
        assert run_settings["average_peak_points"]["value_s"] == 20 / 1000, "average_peak_points_value"
        assert run_settings["area_under_curve"]["on"], "area_under_curve_on"
        assert run_settings["area_under_curve"]["value_pa_ms"] == 3

        tgui.switch_checkbox(tgui.mw.mw.events_template_area_under_curve_checkbox, on=True)  # area_under_curve on
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_area_under_curve_spinbox, "3")  # area_under_curve value_s
        assert run_settings["baseline_search_period_s"] == 25 / 1000, "baseline_search_period_s"
        assert run_settings["average_baseline_points"]["on"] is False, "average_baseline_points_on"
        assert run_settings["average_baseline_points"]["value_s"] == 30 / 1000, "average_baseline_points_value"

        # Events Analyse
        assert run_settings["detection_threshold_type"] == "detection_criterion"
        assert run_settings["detection_criterion"] == 1.4
        assert run_settings["threshold_type"] == "rms", "threshold_type"
        assert run_settings["threshold_lower"]["rms"] == event_analysis_master.calculate_rms(tgui.adata.im_array[0], 0.2, curved_baseline)[0]
        assert run_settings["threshold_lower"]["n_times_rms"] == event_analysis_master.calculate_rms(tgui.adata.im_array[0], 0.2, curved_baseline)[1]
        assert np.array_equal(run_settings["threshold_lower"]["baseline"],
                              curved_baseline)
        assert run_settings["threshold_upper_limit_on"] is True, "threshold_upper_limit_on"
        assert run_settings["threshold_upper_limit_value"] == -72, "threshold_upper_limit_value"
        assert run_settings["baseline_type"] == "curved", "baseline_type"
        assert np.array_equal(run_settings["baseline"],
                              curved_baseline), "curved_baseline_displacement"
        assert run_settings["ts"] == tgui.adata.ts, "ts"

        self.check_extra_run_settings(tgui, run_settings)

#       Average Event - Check set options are correct
#       ----------------------------------------------------------------------------------------------------------------------------------------------

        tgui.left_mouse_click(analyse_dialog.dia.fit_all_events_button)
        tgui.left_mouse_click(analyse_dialog.dia.average_all_events_button)

        analysis_cfgs = analyse_dialog.average_all_events_dialog.make_average_event_kinetics_analysis_cfg(1, 2)

        shared_run_settings = tgui.mw.loaded_file.get_event_kinetics_run_settings_shared_between_all_analyses()
        run_settings = curve_fitting_master.get_curve_fitting_and_average_event_run_settings(shared_run_settings, analysis_cfgs, direction=-1)

        # Analysis Options
        assert run_settings["interp_200khz"] is True  # this test insanely pointless because all of these opts are taken from the same function call. I guess you can never be too careful!
        assert run_settings["rise_cutoff_low"] == 30
        assert run_settings["rise_cutoff_high"] == 65
        assert run_settings["decay_amplitude_percent"] == 50
        assert run_settings["max_slope"]["on"]
        assert run_settings["max_slope"]["rise_num_samples"] == 11
        assert run_settings["max_slope"]["decay_num_samples"] == 12
        assert run_settings["max_slope"]["smooth"]["on"]
        assert run_settings["max_slope"]["smooth"]["num_samples"] == 14
        assert run_settings["max_slope"]["use_baseline_crossing_endpoint"] is True
        assert run_settings["threshold_manual_selected_event"] is False
        assert run_settings["decay_or_biexp_fit_method"] == "biexp"
        assert run_settings["biexp_fit"]["adjust_startpoint_bounds_on"] is False
        assert run_settings["biexp_fit"]["adjust_startpoint_bounds_value"] == 55
        assert run_settings["biexp_fit"]["exclude_if_params_not_in_bounds"] is True
        assert run_settings["biexp_fit"]["rise_cutoff_min"] == 0.001
        assert run_settings["biexp_fit"]["rise_cutoff_max"] == 100
        assert run_settings["biexp_fit"]["decay_cutoff_min"] == 0.002
        assert run_settings["biexp_fit"]["decay_cutoff_max"] == 200

        # Panel
        assert run_settings["decay_search_period_s"] == 10 / 1000, "decay_search_period_s"
        assert run_settings["amplitude_threshold"] == 1e-15, "amplitude_threshold"  # no amplitude threshold applied for average event
        assert run_settings["average_peak_points"]["on"], "average_peak_points_on"
        assert run_settings["average_peak_points"]["value_s"] == 20 / 1000, "average_peak_points_value"
        assert run_settings["area_under_curve"]["on"] is False, "area_under_curve_on"  # area under curve thresholding turned off for average event kinetics

        assert run_settings["baseline_search_period_s"] == 25 / 1000, "baseline_search_period_s"
        assert run_settings["average_baseline_points"]["on"] is False, "average_baseline_points_on"
        assert run_settings["average_baseline_points"]["value_s"] == 30 / 1000, "average_baseline_points_value"

        # Events Analyse (no threshold lower entry for average events
        assert run_settings["detection_threshold_type"] == "detection_criterion"
        assert run_settings["detection_criterion"] == 1.4
        assert run_settings["threshold_type"] == "rms", "threshold_type"
        assert run_settings["threshold_upper_limit_on"] is True, "threshold_upper_limit_on"
        assert run_settings["threshold_upper_limit_value"] == -72, "threshold_upper_limit_value"
        assert run_settings["baseline_type"] == "per_event", "baseline_type"
        assert run_settings["baseline"] == analyse_dialog.average_all_events_dialog.curve_fitting_region.baseline, "average events baseline"
        assert run_settings["ts"] == tgui.adata.ts, "ts"
        self.check_extra_run_settings(tgui, run_settings)
        assert run_settings["endpoint_search_method"] == "first_baseline_cross"
        assert "analysis_type" not in run_settings
        assert run_settings["name"] == "average_event_kinetics"

#       Average Event - Set new options and check they are configured correctly
#       ----------------------------------------------------------------------------------------------------------------------------------------------

        analyse_dialog.average_all_events_dialog.close()
        tgui.left_mouse_click(analyse_dialog.dia.average_all_events_button)

        tgui.switch_checkbox(options_dialog.dia.interp_200khz_checkbox, on=False)
        tgui.enter_number_into_spinbox(options_dialog.dia.rise_time_cutoff_low, "50")
        tgui.enter_number_into_spinbox(options_dialog.dia.rise_time_cutoff_high, "86")
        tgui.enter_number_into_spinbox(options_dialog.dia.decay_amplitude_perc_spinbox, "70")
        tgui.switch_groupbox(analyse_dialog.average_all_events_dialog.dia.calculate_event_kinetics_groupbox, on=True)
        tgui.set_combobox(analyse_dialog.average_all_events_dialog.dia.baseline_method_combobox, 1)
        tgui.left_mouse_click(analyse_dialog.average_all_events_dialog.dia.calculate_kinetics_button)

        analysis_cfgs = analyse_dialog.average_all_events_dialog.make_average_event_kinetics_analysis_cfg(1, 2)
        shared_run_settings = tgui.mw.loaded_file.get_event_kinetics_run_settings_shared_between_all_analyses()
        run_settings = curve_fitting_master.get_curve_fitting_and_average_event_run_settings(shared_run_settings, analysis_cfgs, direction=-1)

        # Analysis Options
        assert run_settings["interp_200khz"] is False
        assert run_settings["rise_cutoff_high"] == 86
        assert run_settings["rise_cutoff_low"] == 50
        assert run_settings["decay_amplitude_percent"] == 70
        assert run_settings["max_slope"]["on"]
        assert run_settings["max_slope"]["rise_num_samples"] == 11
        assert run_settings["max_slope"]["decay_num_samples"] == 12
        assert run_settings["max_slope"]["smooth"]["on"]
        assert run_settings["max_slope"]["smooth"]["num_samples"] == 14
        assert run_settings["max_slope"]["use_baseline_crossing_endpoint"] is True
        assert run_settings["threshold_manual_selected_event"] is False
        assert run_settings["decay_or_biexp_fit_method"] == "biexp"
        assert run_settings["biexp_fit"]["adjust_startpoint_bounds_on"] is False
        assert run_settings["biexp_fit"]["adjust_startpoint_bounds_value"] == 55
        assert run_settings["biexp_fit"]["exclude_if_params_not_in_bounds"] is True
        assert run_settings["biexp_fit"]["rise_cutoff_min"] == 0.001
        assert run_settings["biexp_fit"]["rise_cutoff_max"] == 100
        assert run_settings["biexp_fit"]["decay_cutoff_min"] == 0.002
        assert run_settings["biexp_fit"]["decay_cutoff_max"] == 200

        # Panel
        assert run_settings["decay_search_period_s"] == 10 / 1000, "decay_search_period_s"
        assert run_settings["amplitude_threshold"] == 1e-15, "amplitude_threshold"
        assert run_settings["average_peak_points"]["on"], "average_peak_points_on"
        assert run_settings["average_peak_points"]["value_s"] == 20 / 1000, "average_peak_points_value"
        assert run_settings["area_under_curve"]["on"] is False, "area_under_curve_on"  # area under curve thresholding turned off for curve fitting kinetics
        assert run_settings["baseline_search_period_s"] == 25 / 1000, "baseline_search_period_s"
        assert run_settings["average_baseline_points"]["on"] is False, "average_baseline_points_on"
        assert run_settings["average_baseline_points"]["value_s"] == 30 / 1000, "average_baseline_points_value"

        # Events Analyse (no threshold lower entry)
        assert run_settings["detection_threshold_type"] == "detection_criterion"
        assert run_settings["detection_criterion"] == 1.4
        assert run_settings["threshold_type"] == "rms", "threshold_type"
        assert run_settings["threshold_upper_limit_on"] is True, "threshold_upper_limit_on"
        assert run_settings["threshold_upper_limit_value"] == -72, "threshold_upper_limit_value"
        assert run_settings["baseline_type"] == "manual", "baseline_type"
        assert run_settings["baseline"] == analyse_dialog.average_all_events_dialog.curve_fitting_region.baseline, "average events baseline"
        assert run_settings["ts"] == tgui.adata.ts, "ts"

#       Curve Fitting Event - Change these settings and check they match
#       ----------------------------------------------------------------------------------------------------------------------------------------------

        analyse_dialog.average_all_events_dialog.close()
        tgui.set_analysis_type("curve_fitting")

        tgui.left_mouse_click(tgui.mw.mw.curve_fitting_show_dialog_button)

        tgui.set_combobox(tgui.mw.dialogs["curve_fitting"].dia.fit_type_combobox, 6)

        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.curve_fitting_event_kinetics_options)

        cf_event_dialog = tgui.mw.dialogs["curve_fitting"].curve_fitting_events_kinetics_dialog
        tgui.switch_checkbox(cf_event_dialog.dia.average_peak_checkbox, on=True)
        tgui.enter_number_into_spinbox(cf_event_dialog.dia.average_peak_spinbox, "7", setValue=True)
        tgui.enter_number_into_spinbox(tgui.mw.mw.events_template_area_under_curve_spinbox, "2")
        tgui.set_combobox(cf_event_dialog.dia.baseline_per_event_combobox, 1)
        tgui.enter_number_into_spinbox(cf_event_dialog.dia.baseline_search_period_spinbox, "14", setValue=True)
        tgui.switch_checkbox(cf_event_dialog.dia.average_baseline_checkbox, on=True)
        tgui.enter_number_into_spinbox(cf_event_dialog.dia.average_baseline_spinbox, "21", setValue=True)
        tgui.switch_checkbox(cf_event_dialog.dia.calculate_kinetics_from_fit_not_data_checkbox, on=False)

        analysis_cfgs = curve_fitting_master.generate_curve_fitting_event_kinetics_analysis_cfgs(
            copy.deepcopy(tgui.mw.cfgs.curve_fitting["event_kinetics"]), 0, 3, -60, tgui.mw.loaded_file.data.ts)
        shared_run_settings = tgui.mw.loaded_file.get_event_kinetics_run_settings_shared_between_all_analyses()
        run_settings = curve_fitting_master.get_curve_fitting_and_average_event_run_settings(shared_run_settings, analysis_cfgs, direction=-1)

        # decay_search_period_s added in later in method run_peak_detection_and_kinetics_fit_for_curve_fitting_and_average_events()
        assert run_settings["interp_200khz"] is False
        assert run_settings["rise_cutoff_high"] == 86
        assert run_settings["rise_cutoff_low"] == 50
        assert run_settings["decay_amplitude_percent"] == 70
        assert run_settings["max_slope"]["on"]
        assert run_settings["max_slope"]["rise_num_samples"] == 11
        assert run_settings["max_slope"]["decay_num_samples"] == 12
        assert run_settings["max_slope"]["smooth"]["on"]
        assert run_settings["max_slope"]["smooth"]["num_samples"] == 14
        assert run_settings["max_slope"]["use_baseline_crossing_endpoint"] is True
        assert "threshold_manual_selected_event" not in run_settings
        assert run_settings["decay_or_biexp_fit_method"] == "biexp"
        assert run_settings["biexp_fit"]["adjust_startpoint_bounds_on"] is False
        assert run_settings["biexp_fit"]["adjust_startpoint_bounds_value"] == 55
        assert run_settings["biexp_fit"]["exclude_if_params_not_in_bounds"] is True
        assert run_settings["biexp_fit"]["rise_cutoff_min"] == 0.001
        assert run_settings["biexp_fit"]["rise_cutoff_max"] == 100
        assert run_settings["biexp_fit"]["decay_cutoff_min"] == 0.002
        assert run_settings["biexp_fit"]["decay_cutoff_max"] == 200

        # Panel
        assert run_settings["amplitude_threshold"] == 1e-15, "amplitude_threshold"
        assert run_settings["baseline_type"] == "manual"
        assert run_settings["baseline"] == -60
        assert run_settings["area_under_curve"]["on"] is False, "area_under_curve_on"

        # cf settings type
        assert run_settings["decay_period_type"] == "use_end_of_region"
        assert run_settings["average_peak_points"]["on"], "average_peak_points_on"
        assert run_settings["average_peak_points"]["value_s"] == 7 / 1000, "average_peak_points_value"
        assert run_settings["baseline_search_period_s"] == 14 / 1000, "baseline_search_period_s"
        assert run_settings["average_baseline_points"]["on"], "average_baseline_points_on"
        assert run_settings["average_baseline_points"]["value_s"] == 21 / 1000, "average_baseline_points_value"
        assert run_settings["from_fit_not_data"] is False

        assert run_settings["endpoint_search_method"] == "first_baseline_cross"
        assert "analysis_type" not in run_settings
        assert run_settings["name"] == "curve_fit_event_kinetics"

        tgui.switch_checkbox(cf_event_dialog.dia.calculate_kinetics_from_fit_not_data_checkbox, on=True)
        analysis_cfgs = curve_fitting_master.generate_curve_fitting_event_kinetics_analysis_cfgs(
            copy.deepcopy(tgui.mw.cfgs.curve_fitting["event_kinetics"]), 0, 3, -60, tgui.mw.loaded_file.data.ts)
        shared_run_settings = tgui.mw.loaded_file.get_event_kinetics_run_settings_shared_between_all_analyses()
        run_settings = curve_fitting_master.get_curve_fitting_and_average_event_run_settings(shared_run_settings, analysis_cfgs, direction=-1)

        assert run_settings["from_fit_not_data"] is True
        assert run_settings["average_peak_points"]["on"] is False
        assert run_settings["average_baseline_points"]["on"] is False
        assert run_settings["baseline_type"] is None

        tgui.shutdown()

    def test_deconvolution_run_settings(self, tgui):

        tgui = setup_artificial_event(tgui)
        dialog, config_dict = self.setup_events_with_dialog(tgui, "analyse")

        tgui.set_combobox(dialog.dia.detection_cutoff_combobox, 2)
        tgui.left_mouse_click(dialog.dia.deconvolution_options_button)

        tgui.enter_number_into_spinbox(dialog.deconvolution_options_dialog.dia.low_filter_cutoff_spinbox, 55)
        tgui.enter_number_into_spinbox(dialog.deconvolution_options_dialog.dia.high_filter_cutoff_spinbox, 155)
        tgui.enter_number_into_spinbox(dialog.deconvolution_options_dialog.dia.standard_deviation_spinbox, 5.5)

        run_settings = tgui.mw.loaded_file.get_event_detection_run_settings("template_analyse")
        assert run_settings["deconv_options"]["filt_low_hz"] == 55  # detection_threshold is tested on test_events.py
        assert run_settings["deconv_options"]["filt_high_hz"] == 155
        assert run_settings["deconv_options"]["n_times_std"] == 5.5

        tgui.shutdown()

    def test_quick_get_run_settings_for_event_insertion_or_deletion(self, tgui):
        """
        DataModel.quick_get_run_settings_for_event_insertion_or_deletion()
        """
        tgui = setup_artificial_event(tgui, reset_all_configs=True)
        tgui.run_artificial_events_analysis(tgui, "template")

        event_info = tgui.mw.loaded_file.event_info
        ev_times = list(event_info[0].keys())
        previous_event_time = ev_times[0]
        next_event_time = ev_times[2]

        run_settings = tgui.mw.loaded_file.quick_get_run_settings_for_event_insertion_or_deletion("template_analyse", event_info, 0, next_event_time, previous_event_time)

        assert run_settings["next_event_baseline_idx"] == event_info[0][next_event_time]["baseline"]["idx"]
        assert run_settings["previous_event_idx"] == event_info[0][previous_event_time]["peak"]["idx"]
        assert run_settings["decay_or_biexp_fit_method"] == "monoexp"
        assert run_settings["template_num"] == tgui.mw.loaded_file.get_template_num(run_settings) == "1"
        assert run_settings["rise_s"] == tgui.mw.loaded_file.cfgs.get_rise("s") == 0.0005
        assert run_settings["decay_s"] ==  tgui.mw.loaded_file.cfgs.get_decay("s") == 0.005
        assert run_settings["direction"] ==  tgui.mw.loaded_file.cfgs.direction() == -1
        assert run_settings["window_len_s"] == tgui.mw.loaded_file.cfgs.get_window_len("s") == 0.02
        assert run_settings["rec"] == 0
        assert run_settings["manual_select"]["use_thresholding"] ==  tgui.mw.loaded_file.cfgs.events["threshold_manual_selected_event"]

        # switch some settings and check again
        tgui.mw.mw.actionEvents_Analyis_Options.trigger()
        tgui.set_combobox(tgui.mw.dialogs["events_analysis_options"].dia.event_fit_method_combobox, 1)

        tgui.left_mouse_click(tgui.mw.mw.events_template_generate_button)
        tgui.set_combobox(tgui.mw.dialogs["events_template_generate"].dia.choose_template_combobox, 1)

        run_settings = tgui.mw.loaded_file.quick_get_run_settings_for_event_insertion_or_deletion("template_analyse", event_info, 0, next_event_time=None, previous_event_time=None)

        assert run_settings["template_num"] == "2"
        assert run_settings["decay_or_biexp_fit_method"] == "monoexp"  # this is always consistent with the actual analysis, not the current settings
        assert run_settings["next_event_baseline_idx"] is None
        assert run_settings["previous_event_idx"] is None

        tgui.set_combobox(tgui.mw.dialogs["events_template_generate"].dia.choose_template_combobox, 2)

        run_settings = tgui.mw.loaded_file.quick_get_run_settings_for_event_insertion_or_deletion("template_analyse", event_info, 0, next_event_time=None, previous_event_time=None, previous_event_idx=40)

        assert run_settings["template_num"] == "3"
        assert run_settings["previous_event_idx"] == 40

        tgui.shutdown()

    def check_extra_run_settings(self, tgui, run_settings):

        assert run_settings["edit_kinetics_mode"] is None
        assert run_settings["next_event_idx"] is None
        assert run_settings["previous_event_idx"] is None
        assert run_settings["legacy_options"] == tgui.mw.cfgs.main["legacy_options"]
        assert run_settings["cannonical_initial_biexp_coefficients"]  == {"rise": 0.5, "decay": 5}
        assert run_settings["analyse_specific_recs"] is None
        assert run_settings["rec_from"] == 0
        assert run_settings["rec_to"] == 0

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Table Col Headers
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_all_table_col_headers(self, tgui):
        load_file(tgui)
        table_col_headers = tgui.mw.cfgs.get_table_col_headers("spkcnt_and_input_resistance")[0]

        assert table_col_headers["num_spikes"] == "Num Spikes"
        assert table_col_headers["fs_latency_ms"] == "FS Latency (ms)"
        assert table_col_headers["mean_isi_ms"] == "Mean ISI (ms)"
        assert table_col_headers["spike_fa_divisor"] == "SFA (divisor method)"
        assert table_col_headers["spike_fa_local_variance"] == "SFA (local variance)"
        assert table_col_headers["rheobase"] == "Rheobase (pA)"
        assert table_col_headers["record_num"] == "Record"
        assert table_col_headers["im_baseline"] == "Baseline Im (pA)"
        assert table_col_headers["im_steady_state"] == "Steady State Im (pA)"
        assert table_col_headers["im_delta"] == "ΔIm (pA)"
        assert table_col_headers["im_delta_round"] == "ΔIm (pA)"
        assert table_col_headers["user_input_im"] == "User Input Im (pA)"
        assert table_col_headers["input_resistance"] == "Ri (MOhms)"
        assert table_col_headers["vm_baseline"] == "Baseline Vm (mV)"
        assert table_col_headers["vm_steady_state"] == "Steady State Vm (mV)"
        assert table_col_headers["vm_delta"] == "ΔVm (mV)"
        assert table_col_headers["sag_hump_peaks"] == "Sag (or Hump) (mV)"
        assert table_col_headers["sag_hump_ratio"] == "Sag (or Hump) Ratio"

        table_col_headers = tgui.mw.cfgs.get_table_col_headers("skinetics")[0]
        assert table_col_headers["record_num"] == "Record"
        assert table_col_headers["spike_number"] == "Spike"
        assert table_col_headers["spike_time"] == "Spike Time (s)"
        assert table_col_headers["peak"] == "Peak (mV)"
        assert table_col_headers["amplitude"] == "Amplitude (mV)"
        assert table_col_headers["thr"] == "Threshold (mV)"
        assert table_col_headers["rise_time"] == "Rise Time (ms)"
        assert table_col_headers["decay_time"] == "Decay Time (ms)"
        assert table_col_headers["fwhm"] == "Half-Width (ms)"
        assert table_col_headers["fahp"] == "fAHP (mV)"
        assert table_col_headers["mahp"] == "mAHP (mV)"
        assert table_col_headers["max_rise"] == "Max Rise Slope (ΔmV/Δms)"
        assert table_col_headers["max_decay"] == "Max Decay Slope (ΔmV/Δms)"

        table_col_headers = tgui.mw.cfgs.get_table_col_headers("events")[0]
        assert table_col_headers["event_num"] == "Event Num."
        assert table_col_headers["template_num"] == "Template"
        assert table_col_headers["record_num"] == "Record"
        assert table_col_headers["event_time"] == "Event Time (s)"
        assert table_col_headers["baseline"] == "Baseline (pA)"
        assert table_col_headers["peak"] == "Peak (pA)"
        assert table_col_headers["amplitude"] == "Amplitude (pA)"
        assert table_col_headers["rise"] == "Rise Time (ms)"
        assert table_col_headers["half_width"] == "Half-Width (ms)"
        assert table_col_headers["decay_perc"] == "Decay % (ms)"

        assert table_col_headers["area_under_curve"] == "AUC (pA ms)"
        assert table_col_headers["event_period"]  == "AUC Time (ms)"
        assert table_col_headers["max_rise"] == "Max Rise Slope (ΔpA/Δms)"
        assert table_col_headers["max_decay"] == "Max Decay Slope (ΔpA/Δms)"

        assert table_col_headers["monoexp_fit_b0"] == "Decay Fit b0"
        assert table_col_headers["monoexp_fit_b1"] == "Decay Fit b1"
        assert table_col_headers["monoexp_fit_tau"] == "Decay Fit Tau (ms)"
        assert table_col_headers["events_monoexp_r2"] == "Decay Fit R\u00B2"
        assert table_col_headers["biexp_fit_b0"] == "Biexp Fit b0"
        assert table_col_headers["biexp_fit_b1"] == "Biexp Fit b1"
        assert table_col_headers["biexp_fit_rise"] == "Biexp Fit Rise (ms)"
        assert table_col_headers["biexp_fit_decay"] == "Biexp Fit Decay (ms)"
        assert table_col_headers["events_biexp_r2"] == "Biexp Fit R\u00B2"

        table_col_headers = tgui.mw.cfgs.get_table_col_headers("curve_fitting")[0]
        mV_or_pA = "(mV)"
        assert table_col_headers["record_num"] == "Record"
        assert table_col_headers["event_time"] == "Event Time (s)"
        assert table_col_headers["event_number"] == "Event Num."
        assert table_col_headers["min"] == "Minimum " + mV_or_pA
        assert table_col_headers["max"] == "Maximum " + mV_or_pA
        assert table_col_headers["mean"] == "Mean " + mV_or_pA
        assert table_col_headers["median"] == "Median " + mV_or_pA
        assert table_col_headers["max_slope_ms"] == "Max Slope (Δ" + mV_or_pA[1:-1] + "/Δms)"
        assert table_col_headers["baseline"] == "Baseline " + mV_or_pA
        assert table_col_headers["peak"] == "Peak " + mV_or_pA
        assert table_col_headers["amplitude"] == "Amplitude " + mV_or_pA

        assert table_col_headers["rise"] == "Rise Time (ms)"
        assert table_col_headers["half_width"] == "Half-Width (ms)"
        assert table_col_headers["decay_perc"] == "Decay Ampitude % (ms)"

        assert table_col_headers["area_under_curve"] == "AUC (pA ms)"
        assert table_col_headers["event_period"] == "AUC Time (ms)"
        assert table_col_headers["max_rise"] == "Max Rise Slope (Δ" + mV_or_pA[1:-1] + "/Δms)"
        assert table_col_headers["max_decay"] == "Max Decay Slope (Δ" + mV_or_pA[1:-1] + "/Δms)"

        assert table_col_headers["monoexp_fit_b0"] == "Decay Fit b0"
        assert table_col_headers["monoexp_fit_b1"] == "Decay Fit b1"
        assert table_col_headers["monoexp_fit_tau"] == "Decay Fit Tau (ms)"
        assert table_col_headers["events_monoexp_r2"] == "Decay Fit R\u00B2"
        assert table_col_headers["biexp_fit_b0"] == "Biexp Fit b0"
        assert table_col_headers["biexp_fit_b1"] == "Biexp Fit b1"
        assert table_col_headers["biexp_fit_rise"] == "Biexp Fit Rise (ms)"
        assert table_col_headers["biexp_fit_decay"] == "Biexp Fit Decay (ms)"
        assert table_col_headers["events_biexp_r2"] == "Biexp Fit R\u00B2"

        assert table_col_headers["b0"] == "b0 " + mV_or_pA
        assert table_col_headers["b1"] == "b1 " + mV_or_pA
        assert table_col_headers["b2"] == "b2 " + mV_or_pA
        assert table_col_headers["b3"] == "b3 " + mV_or_pA
        assert table_col_headers["fit_rise"] == "Rise (fit) (ms)"
        assert table_col_headers["fit_decay"] == "Decay (fit) (ms)"
        assert table_col_headers["tau"] == "Tau (ms)"
        assert table_col_headers["tau1"] == "Tau1 (ms)"
        assert table_col_headers["tau2"] == "Tau2 (ms)"
        assert table_col_headers["tau3"] == "Tau3 (ms)"

        assert table_col_headers["r2"] == "R\u00B2"

    def test_voltage_calc_table_col_headers(self, tgui):
        load_file(tgui, "voltage_clamp_1_record")
        table_col_headers = tgui.mw.cfgs.get_table_col_headers("curve_fitting")[0]
        mV_or_pA = "(pA)"
        assert table_col_headers["min"] == "Minimum " + mV_or_pA
        assert table_col_headers["max"] == "Maximum " + mV_or_pA
        assert table_col_headers["mean"] == "Mean " + mV_or_pA
        assert table_col_headers["median"] == "Median " + mV_or_pA
        assert table_col_headers["baseline"] == "Baseline " + mV_or_pA
        assert table_col_headers["peak"] == "Peak " + mV_or_pA
        assert table_col_headers["amplitude"] == "Amplitude " + mV_or_pA

# ----------------------------------------------------------------------------------------------------------------------------------------------------
#  Analysis Table Options
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_analysis_table_options(self, tgui):
        """
        Check defaults are correct, change all options, close and reopen software, check options are all saved 
        """
        analysis_cfgs = tgui.mw.cfgs.analysis

        assert analysis_cfgs["curve_fitting_summary_statistics_region"] == "reg_1"
        assert analysis_cfgs["table_file_view"] == "per_column"
        assert analysis_cfgs["show_filename_once_row_mode"] is False
        assert analysis_cfgs["summary_stats_within_rec"] is False
        assert analysis_cfgs["summary_stats_params_to_show"] == ["M", "SD", "SE"]

        for ireg in range(6):
            tgui.set_combobox(tgui.mw.mw.curve_fitting_summary_statistics_combobox, ireg)
            assert analysis_cfgs["curve_fitting_summary_statistics_region"] == "reg_" + str(ireg + 1)

        tgui.mw.mw.actionTable_Options.trigger()
        dialog = tgui.mw.dialogs["analysis_statistics_options_dialog"]

        tgui.switch_checkbox(dialog.dia.file_per_row_radiobutton, on=True)
        assert analysis_cfgs["table_file_view"] == "per_row"

        tgui.switch_checkbox(dialog.dia.show_filename_once_checkbox, on=True)
        assert analysis_cfgs["show_filename_once_row_mode"] is True

        tgui.set_combobox(dialog.dia.calculate_across_records_combobox, 1)
        assert analysis_cfgs["summary_stats_within_rec"] is True

        tgui.switch_checkbox(dialog.dia.mean_checkbox, on=False)
        assert analysis_cfgs["summary_stats_params_to_show"] == ["SD", "SE"]

        tgui.switch_checkbox(dialog.dia.standard_error_checkbox, on=False)
        assert analysis_cfgs["summary_stats_params_to_show"] == ["SD"]

        tgui = close_and_reload_for_defaults_check(tgui, save_="table_analysis_options")
        dialog = tgui.mw.dialogs["analysis_statistics_options_dialog"]
        analysis_cfgs = tgui.mw.cfgs.analysis

        assert analysis_cfgs["curve_fitting_summary_statistics_region"] == "reg_1"  # this is not saved

        assert dialog.dia.file_per_row_radiobutton.isChecked()
        assert not dialog.dia.file_per_column_radiobutton.isChecked()
        assert analysis_cfgs["table_file_view"] == "per_row"

        assert dialog.dia.show_filename_once_checkbox.isChecked()
        assert analysis_cfgs["show_filename_once_row_mode"] is True

        assert dialog.dia.calculate_across_records_combobox.currentIndex() == 1
        assert analysis_cfgs["summary_stats_within_rec"] is True

        assert not dialog.dia.mean_checkbox.isChecked()
        assert not dialog.dia.standard_error_checkbox.isChecked()
        assert dialog.dia.standard_deviation_checkbox.isChecked()
        assert analysis_cfgs["summary_stats_params_to_show"] == ["SD"]

        # if no stats attempted to be shown, will give warning and revert to mean
        QtCore.QTimer.singleShot(500, lambda: self.check_warning_at_least_one_statistic_messagebox(tgui.mw.messagebox))
        tgui.switch_checkbox(dialog.dia.standard_deviation_checkbox, on=False)

        assert dialog.dia.mean_checkbox.isChecked()
        assert analysis_cfgs["summary_stats_params_to_show"] == ["M"]

        tgui.shutdown()

    def check_warning_at_least_one_statistic_messagebox(self, messagebox):
        assert "Must choose at least one statistic to calculate." in messagebox.text()
        messagebox.close()

    def test_kolmogorov_statistics(self, tgui):
        """
        These options are not stored in configs. Change all options and check the dialog settings are correct.
        The data entered into the table is stored for the session. Enter some numbers into the table,
        check they are stored in configs, close and reopen the window and check they are still down on the table.
        """
        tgui.mw.mw.actionKolmogorov_Smirnoff.trigger()
        dialog = tgui.mw.dialogs["analysis_grouped_cum_prob"]

        assert dialog.dia.one_dataset_method_combobox.currentIndex() == 0
        assert not dialog.dia.one_dataset_sample_mean_spinbox.isEnabled()
        assert not dialog.dia.one_dataset_sample_stdev_spinbox.isEnabled()
        assert dialog.one_sample_method == "lilliefors"

        tgui.set_combobox(dialog.dia.one_dataset_method_combobox, 1)
        assert dialog.one_sample_method == "user_input_population"
        assert dialog.dia.one_dataset_sample_mean_spinbox.isEnabled()
        assert dialog.dia.one_dataset_sample_stdev_spinbox.isEnabled()

        assert dialog.user_input_mean == 0
        assert dialog.user_input_stdev == 0.0001
        assert dialog.dia.one_dataset_sample_mean_spinbox.value() == 0
        assert dialog.dia.one_dataset_sample_stdev_spinbox.value() == 0.0001

        tgui.enter_number_into_spinbox(dialog.dia.one_dataset_sample_mean_spinbox, 10)
        tgui.enter_number_into_spinbox(dialog.dia.one_dataset_sample_stdev_spinbox, 20)
        assert dialog.user_input_mean == 10
        assert dialog.user_input_stdev == 20

        assert dialog.analyse_one_or_two_columns == "one"
        tgui.switch_groupbox(dialog.dia.two_datasets_groupbox, on=True)
        assert dialog.alternative_hypothesis == "two-sided"

        for idx, text in enumerate(["two-sided", "greater", "less"]):
            tgui.set_combobox(dialog.dia.alternative_hypothesis_combobox, idx)
            assert text == dialog.alternative_hypothesis

        tgui.fill_tablewidget_with_items(dialog.dia.analysis_table, np.array([[1, 2, 3], [4, 5, 6]]).T)

        analysis_configs = tgui.mw.cfgs.analysis["temp_stored_analysis"]["ks_test"]

        dialog.close()
        assert np.array_equal(analysis_configs["column_1"], np.atleast_2d([[1, 2, 3]]).T)
        assert np.array_equal(analysis_configs["column_2"], np.atleast_2d([[4, 5, 6]]).T)

        tgui.mw.mw.actionKolmogorov_Smirnoff.trigger()
        dialog = tgui.mw.dialogs["analysis_grouped_cum_prob"]

        first_col = [dialog.dia.analysis_table.item(row, 0).data(0) for row in range(3)]
        assert first_col == ['1.0', '2.0', '3.0']

        tgui.switch_groupbox(dialog.dia.two_datasets_groupbox, on=True)
        second_col = [dialog.dia.analysis_table.item(row, 1).data(0) for row in range(3)]
        assert second_col == ['4.0', '5.0', '6.0']

# Processor Problems Test !! ------------------------------------------------------------------------------------------

# had to refactor most of these tests (I think for most recent computer version with new processor)
# as loading mulitple tgui in loop was causing segmentation faults, so had to use parameterise even
# thoughthe test is not as neat.

# ---------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("dialog_type", ["refine", "analyse", "threshold"])
    @pytest.mark.parametrize("combobox_idx_name", [[0, "Linear"], [1, "Curved"], [2, "Draw"], [3, None]])
    def test_threshold_lower_combobox_and_widgets(self, tgui, dialog_type, combobox_idx_name):
        """
        """
        def wrap_to_avoid_pytest_seg_fault(tgui, dialog_type, combobox_idx_name):

            dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)
            idx, text_ = combobox_idx_name

            tgui.set_combobox(dialog.dia.threshold_lower_combobox,
                              idx)
            tgui.enter_number_into_spinbox(dialog.dia.threshold_lower_spinbox,
                                           idx + 1)
            tgui.switch_checkbox(dialog.dia.hide_threshold_lower_from_plot_checkbox,
                                 on=True)

            if idx == 3:
                text_ = "RMS of mean" if dialog_type == "refine" else "RMS of baseline"

            assert dialog.dia.threshold_lower_combobox.currentIndex() == idx
            assert dialog.dia.threshold_lower_combobox.currentText() == text_
            assert dialog.dia.hide_threshold_lower_from_plot_checkbox.isChecked()
            assert not tgui.mw.cfgs.events["show_threshold_lower_on_plot"]
            self.check_threshold_lower_spinbox(text_, dialog, idx, tgui)

            tgui.switch_checkbox(dialog.dia.hide_threshold_lower_from_plot_checkbox,
                                 on=False)

            tgui = close_and_reload_for_defaults_check(tgui, save_="events_" + dialog_type)
            dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)

            assert dialog.dia.threshold_lower_combobox.currentIndex() == idx
            assert dialog.dia.threshold_lower_combobox.currentText() == text_
            assert not dialog.dia.hide_threshold_lower_from_plot_checkbox.isChecked()
            assert tgui.mw.cfgs.events["show_threshold_lower_on_plot"]
            self.check_threshold_lower_spinbox(text_, dialog, idx, tgui)

            tgui.set_combobox(dialog.dia.threshold_lower_combobox, idx)
            self.check_threshold_lower_spinbox(text_, dialog, idx, tgui)

            tgui.shutdown()
        wrap_to_avoid_pytest_seg_fault(tgui, dialog_type, combobox_idx_name)

    def test_deconvolution_options_dialog_widgets(self, tgui):

        def wrap_to_avoid_pytest_seg_fault(tgui):
            dialog, config_dict = self.setup_events_with_dialog(tgui, "refine")

            tgui.set_combobox(dialog.dia.detection_cutoff_combobox, 2)
            tgui.left_mouse_click(dialog.dia.deconvolution_options_button)

            tgui.enter_number_into_spinbox(dialog.deconvolution_options_dialog.dia.low_filter_cutoff_spinbox, 55)
            tgui.enter_number_into_spinbox(dialog.deconvolution_options_dialog.dia.high_filter_cutoff_spinbox, 155)
            tgui.enter_number_into_spinbox(dialog.deconvolution_options_dialog.dia.standard_deviation_spinbox, 5.5)

            assert config_dict["deconv_options"]["filt_low_hz"] == 55  # detection_threshold is tested on test_events.py
            assert config_dict["deconv_options"]["filt_high_hz"] == 155
            assert config_dict["deconv_options"]["n_times_std"] == 5.5

            tgui = close_and_reload_for_defaults_check(tgui, save_="events_refine")
            dialog, config_dict = self.setup_events_with_dialog(tgui, "refine")

            tgui.set_combobox(dialog.dia.detection_cutoff_combobox, 2)
            tgui.left_mouse_click(dialog.dia.deconvolution_options_button)

            assert dialog.deconvolution_options_dialog.dia.low_filter_cutoff_spinbox.value() == 55
            assert dialog.deconvolution_options_dialog.dia.high_filter_cutoff_spinbox.value() == 155
            assert dialog.deconvolution_options_dialog.dia.standard_deviation_spinbox.value() == 5.5

            assert config_dict["deconv_options"]["filt_low_hz"] == 55  # detection_threshold is tested on test_events.py
            assert config_dict["deconv_options"]["filt_high_hz"] == 155
            assert config_dict["deconv_options"]["n_times_std"] == 5.5

            tgui.shutdown()
        wrap_to_avoid_pytest_seg_fault(tgui)

    @pytest.mark.parametrize("fit_type", ["monoexp", "biexp"])
    def test_decay_fit_type_widgets(self, tgui,
                                    fit_type):

        def wrap_to_avoid_pytest_seg_fault(tgui, fit_type):

            dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                                dialog_type="misc_options")
            key = fit_type + "_fit"

            combobox_idx = 0 if fit_type == "monoexp" else 1
            tgui.set_combobox(dialog.dia.event_fit_method_combobox, combobox_idx)
            assert config_dict["decay_or_biexp_fit_method"] == fit_type

            tgui.switch_checkbox(self.misc_event_widgets(dialog, fit_type, "fit_exclude_r2_checkbox"),
                                 on=True)
            assert config_dict[key]["exclude_from_r2_on"] is True
            tgui.enter_number_into_spinbox(self.misc_event_widgets(dialog, fit_type, "fit_exclude_r2_spinbox"),
                                           0.25)
            assert config_dict[key]["exclude_from_r2_value"] == 0.25

            tgui.switch_checkbox(self.misc_event_widgets(dialog, fit_type, "fit_adjust_start_point_checkbox"),
                                 on=True)
            assert config_dict[key]["adjust_startpoint_r2_on"] is True

            tgui.enter_number_into_spinbox(self.misc_event_widgets(dialog, fit_type, "fit_adjust_start_point_spinbox"),
                                           35)
            assert config_dict[key]["adjust_startpoint_r2_value"] == 35

            tgui = close_and_reload_for_defaults_check(tgui, save_="events_misc_options")
            dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                                dialog_type="misc_options")
            # Reopen and check
            assert dialog.dia.event_fit_method_combobox.currentIndex() == combobox_idx
            assert config_dict["decay_or_biexp_fit_method"] == fit_type

            assert self.misc_event_widgets(dialog, fit_type, "fit_exclude_r2_checkbox").isChecked()
            assert config_dict[key]["exclude_from_r2_on"] is True

            assert self.misc_event_widgets(dialog, fit_type, "fit_exclude_r2_spinbox").value() == 0.25
            assert config_dict[key]["exclude_from_r2_value"] == 0.25

            assert self.misc_event_widgets(dialog, fit_type, "fit_adjust_start_point_checkbox").isChecked()
            assert config_dict[key]["adjust_startpoint_r2_on"] is True

            assert self.misc_event_widgets(dialog, fit_type, "fit_adjust_start_point_spinbox").value() == 35
            assert config_dict[key]["adjust_startpoint_r2_value"] == 35

            tgui.switch_checkbox(self.misc_event_widgets(dialog, fit_type, "adjust_start_point_for_bounds_checkbox"),
                                 # switching this on turns adjust_startpoint_r2_on off
                                 on=True)
            assert config_dict[key]["adjust_startpoint_bounds_on"] is True
            assert not self.misc_event_widgets(dialog, fit_type, "fit_adjust_start_point_checkbox").isChecked()
            assert not self.misc_event_widgets(dialog, fit_type, "fit_adjust_start_point_spinbox").isEnabled()

            tgui.enter_number_into_spinbox(self.misc_event_widgets(dialog, fit_type, "adjust_start_point_for_bounds_spinbox"),
                                           45)
            assert config_dict[key]["adjust_startpoint_bounds_value"] == 45

            tgui = close_and_reload_for_defaults_check(tgui, save_="events_misc_options")
            dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                                dialog_type="misc_options")

            assert self.misc_event_widgets(dialog, fit_type, "adjust_start_point_for_bounds_checkbox").isChecked()
            assert config_dict[key]["adjust_startpoint_bounds_on"] is True
            assert self.misc_event_widgets(dialog, fit_type, "adjust_start_point_for_bounds_spinbox").value() == 45
            assert config_dict[key]["adjust_startpoint_bounds_value"] == 45

            if fit_type == "monoexp":  # messy, but cannot split into function or get access violation that shutdown() does not fix
                tgui.switch_checkbox(dialog.dia.monoexp_exclude_outside_of_bounds_checkbox,  # switching this turns off adjust_startpoint_bounds_on
                                     on=True)
                assert config_dict["monoexp_fit"]["exclude_if_params_not_in_bounds"] is True
                assert not dialog.dia.monoexp_adjust_start_point_for_bounds_checkbox.isChecked()
                assert not dialog.dia.monoexp_adjust_start_point_for_bounds_spinbox.isEnabled()

                tgui.enter_number_into_spinbox(dialog.dia.monoexp_min_tau_spinbox,
                                               55)
                assert config_dict["monoexp_fit"]["tau_cutoff_min"] == 55
                tgui.enter_number_into_spinbox(dialog.dia.monoexp_max_tau_spinbox,
                                               65)
                assert config_dict["monoexp_fit"]["tau_cutoff_max"] == 65

                tgui = close_and_reload_for_defaults_check(tgui, save_="events_misc_options")
                dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                                    dialog_type="misc_options")

                assert dialog.dia.monoexp_exclude_outside_of_bounds_checkbox.isChecked()
                assert config_dict["monoexp_fit"]["exclude_if_params_not_in_bounds"] is True

                assert dialog.dia.monoexp_min_tau_spinbox.value() == 55
                assert config_dict["monoexp_fit"]["tau_cutoff_min"] == 55
                assert dialog.dia.monoexp_max_tau_spinbox.value() == 65
                assert config_dict["monoexp_fit"]["tau_cutoff_max"] == 65

            elif fit_type == "biexp":
                tgui.switch_checkbox(dialog.dia.biexp_exclude_outside_of_bounds_checkbox,
                                     on=True)
                assert config_dict["biexp_fit"]["exclude_if_params_not_in_bounds"] is True
                assert not dialog.dia.biexp_adjust_start_point_for_bounds_checkbox.isChecked()
                assert not dialog.dia.biexp_adjust_start_point_for_bounds_spinbox.isEnabled()

                tgui.enter_number_into_spinbox(dialog.dia.biexp_min_rise_spinbox, 55)
                assert config_dict["biexp_fit"]["rise_cutoff_min"] == 55

                tgui.enter_number_into_spinbox(dialog.dia.biexp_min_decay_spinbox, 65)
                assert config_dict["biexp_fit"]["decay_cutoff_min"] == 65

                tgui.enter_number_into_spinbox(dialog.dia.biexp_max_rise_spinbox, 75)
                assert config_dict["biexp_fit"]["rise_cutoff_max"] == 75

                tgui.enter_number_into_spinbox(dialog.dia.biexp_max_decay_spinbox, 85)
                assert config_dict["biexp_fit"]["decay_cutoff_max"] == 85

                tgui = close_and_reload_for_defaults_check(tgui, save_="events_misc_options")
                dialog, config_dict = self.setup_events_with_dialog(tgui,
                                                                    dialog_type="misc_options")

                assert dialog.dia.biexp_exclude_outside_of_bounds_checkbox.isChecked()
                assert config_dict["biexp_fit"]["exclude_if_params_not_in_bounds"] is True

                assert dialog.dia.biexp_min_rise_spinbox.value() == 55
                assert config_dict["biexp_fit"]["rise_cutoff_min"] == 55

                assert dialog.dia.biexp_min_decay_spinbox.value() == 65
                assert config_dict["biexp_fit"]["decay_cutoff_min"] == 65

                assert dialog.dia.biexp_max_rise_spinbox.value() == 75
                assert config_dict["biexp_fit"]["rise_cutoff_max"] == 75

                assert dialog.dia.biexp_max_decay_spinbox.value() == 85
                assert config_dict["biexp_fit"]["decay_cutoff_max"] == 85

            tgui.shutdown()

        wrap_to_avoid_pytest_seg_fault(tgui, fit_type)

    @pytest.mark.parametrize("template", ["1", "2", "3"])
    def test_events_generate_configs(self, tgui, template):

        def wrap_to_avoid_pytest_seg_fault(tgui, template):
            dialog, config_dict = self.setup_events_with_dialog(tgui, "generate")

            offset = int(template)

            dialog.dia.choose_template_combobox.setCurrentIndex(int(template) - 1)

            self.set_template_generate(tgui, dialog, b0=900 + offset)
            assert tgui.mw.cfgs.events["templates"][template]["b0_ms"] == 900 + offset

            self.set_template_generate(tgui, dialog, b1=-900 + offset)
            assert tgui.mw.cfgs.events["templates"][template]["b1_ms"] == -900 + offset

            self.set_template_generate(tgui, dialog, rise=10 + offset)
            assert np.isclose(tgui.mw.cfgs.events["templates"][template]["rise_s"],
                              0.010 + (offset / 1000))

            self.set_template_generate(tgui, dialog, decay=100 + offset)
            assert np.isclose(tgui.mw.cfgs.events["templates"][template]["decay_s"],
                              0.100 + (offset / 1000))  # annoying rounding issue

            self.set_template_generate(tgui, dialog, width=500 + offset)
            assert config_dict["templates"][template]["window_len_s"] == 0.5 + (offset / 1000)
            assert config_dict["templates"][template]["window_len_samples"] == ((0.5 + (offset / 1000)) * tgui.mw.loaded_file.data.fs)

            tgui.click_checkbox(dialog.dia.event_direction_is_positive_checkbox)
            assert config_dict["templates"][template]["direction"] == 1

            tgui.enter_number_into_spinbox(dialog.dia.b1_spinbox, 900 + offset)
            assert tgui.mw.cfgs.events["templates"][template]["b1_ms"] == 900 + offset

            # Reload and Check
            tgui = close_and_reload_for_defaults_check(tgui, save_="events_generate")
            dialog, config_dict = self.setup_events_with_dialog(tgui, "generate")

            assert dialog.dia.b0_spinbox.value() == 900 + offset
            assert tgui.mw.cfgs.events["templates"][template]["b0_ms"] == 900 + offset

            assert dialog.dia.b1_spinbox.value() == 900 + offset
            assert tgui.mw.cfgs.events["templates"][template]["b1_ms"] == 900 + offset

            assert dialog.dia.rise_spinbox.value() == 10 + offset
            assert np.isclose(tgui.mw.cfgs.events["templates"][template]["rise_s"],
                              0.010 + (offset / 1000))

            assert dialog.dia.decay_spinbox.value() == 100 + offset
            assert np.isclose(tgui.mw.cfgs.events["templates"][template]["decay_s"],
                              0.100 + (offset / 1000))

            assert dialog.dia.width_spinbox.value() == 500 + offset
            assert config_dict["templates"][template]["window_len_s"] == 0.5 + (offset / 1000)
            assert config_dict["templates"][template]["window_len_samples"] == ((0.5 + (offset / 1000)) * tgui.mw.loaded_file.data.fs)

            assert dialog.dia.event_direction_is_positive_checkbox.isChecked()
            assert config_dict["templates"][template]["direction"] == 1

            # Switch Direction and reload
            assert not tgui.click_checkbox(dialog.dia.event_direction_is_positive_checkbox)
            assert config_dict["templates"][template]["direction"] == -1

            tgui = close_and_reload_for_defaults_check(tgui, save_="events_generate")
            dialog, config_dict = self.setup_events_with_dialog(tgui, "generate")

            assert not dialog.dia.event_direction_is_positive_checkbox.isChecked()
            assert config_dict["templates"][template]["direction"] == -1

            tgui.shutdown()
        wrap_to_avoid_pytest_seg_fault(tgui, template)

    @pytest.mark.parametrize("dialog_type", ["refine", "analyse", "threshold"])
    @pytest.mark.parametrize("combobox_idx_name", [[0, "Auto."], [1, "Linear"], [2, "Curve"], [3, "Draw"]])
    def test_baseline_combobox_and_widgets(self, tgui, dialog_type, combobox_idx_name):
        """
        Cannot run the last test for macos due to the weird CPU issues / threading running
        these test on macos. Changing spinbox is not currently updating, either with
        tgui fill function or .setValue. Only works if break in and run processEvents()
        like 10 times - very strange macOS behaviour, or Qt bug.
        """

        def wrap_to_avoid_pytest_seg_fault(tgui, dialog_type, combobox_idx_name):
            dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)
            idx, text_ = combobox_idx_name

            tgui.set_combobox(dialog.dia.baseline_combobox, idx)

            tgui.enter_number_into_spinbox(dialog.dia.baseline_spinbox, idx + 1)

            if text_ != "Auto.":
                tgui.switch_checkbox(dialog.dia.hide_baseline_from_plot_checkbox, on=True)

            assert dialog.dia.baseline_combobox.currentIndex() == idx
            assert dialog.dia.baseline_combobox.currentText() == text_
            self.check_baseline_spinbox(text_, dialog, idx, tgui)

            if text_ != "Auto.":
                assert dialog.dia.hide_baseline_from_plot_checkbox.isChecked()
                assert not tgui.mw.cfgs.events["show_baseline_on_plot"]

            tgui.switch_checkbox(dialog.dia.hide_baseline_from_plot_checkbox,
                                 on=False)

            tgui = close_and_reload_for_defaults_check(tgui, save_="events_" + dialog_type)
            dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)

            assert dialog.dia.baseline_combobox.currentIndex() == idx
            assert tgui.mw.cfgs.events["show_baseline_on_plot"]
            self.check_baseline_spinbox(text_, dialog, idx, tgui)

            if text_ != "Auto.":
                assert not dialog.dia.hide_baseline_from_plot_checkbox.isChecked()
                assert tgui.mw.cfgs.events["show_baseline_on_plot"]

            if platform == "darwin":  # see doc
                tgui.shutdown()
                return

            tgui.set_combobox(dialog.dia.baseline_combobox, idx)
            self.check_baseline_spinbox(text_, dialog, idx, tgui)

            QtWidgets.QApplication.closingDown()
            tgui.shutdown()
        wrap_to_avoid_pytest_seg_fault(tgui, dialog_type, combobox_idx_name)

    @pytest.mark.parametrize("dialog_type", ["refine", "analyse", "threshold"])
    def test_upper_threshold(self, tgui, dialog_type):
        
        def wrap_to_avoid_pytest_seg_fault(tgui, dialog_type):
            dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)
    
            tgui.switch_groupbox(dialog.dia.threshold_upper_groupbox, on=True)
            tgui.enter_number_into_spinbox(dialog.dia.threshold_upper_spinbox,
                                           100)
    
            assert config_dict["threshold_upper_limit_on"] is True
            assert config_dict["threshold_upper_limit_value"] == 100
    
            tgui = close_and_reload_for_defaults_check(tgui, save_="events_" + dialog_type)
            dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)
    
            assert dialog.dia.threshold_upper_groupbox.isChecked()
            assert dialog.dia.threshold_upper_spinbox.value() == 100
            assert config_dict["threshold_upper_limit_on"] is True
            assert config_dict["threshold_upper_limit_value"] == 100

            tgui.shutdown()
        wrap_to_avoid_pytest_seg_fault(tgui, dialog_type)


    @pytest.mark.parametrize("dialog_type", ["refine", "analyse"])
    def test_template_detection_method_widgets(self, tgui, dialog_type):

        dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)

        # Correlation
        tgui.set_combobox(dialog.dia.detection_cutoff_combobox, 0)
        assert config_dict["detection_threshold_type"] == "correlation"
        tgui.enter_number_into_spinbox(dialog.dia.detection_threshold_spinbox, 0.85)
        assert config_dict["corr_cutoff"] == 0.85

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_" + dialog_type)
        dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)

        assert dialog.dia.detection_cutoff_combobox.currentIndex() == 0
        assert config_dict["detection_threshold_type"] == "correlation"
        assert dialog.dia.detection_threshold_spinbox.value() == 0.85
        assert config_dict["corr_cutoff"] == 0.85

        # Detection Criteria
        tgui.set_combobox(dialog.dia.detection_cutoff_combobox, 1)
        assert config_dict["detection_threshold_type"] == "detection_criterion"
        tgui.enter_number_into_spinbox(dialog.dia.detection_threshold_spinbox, 12.34)
        assert config_dict["detection_criterion"] == 12.34

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_" + dialog_type)
        dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)

        assert dialog.dia.detection_cutoff_combobox.currentIndex() == 1
        assert config_dict["detection_threshold_type"] == "detection_criterion"
        assert dialog.dia.detection_threshold_spinbox.value() == 12.34
        assert config_dict["detection_criterion"] == 12.34

        # Deconvolution
        tgui.set_combobox(dialog.dia.detection_cutoff_combobox, 2)
        assert config_dict["detection_threshold_type"] == "deconvolution"

        tgui = close_and_reload_for_defaults_check(tgui, save_="events_" + dialog_type)
        dialog, config_dict = self.setup_events_with_dialog(tgui, dialog_type)
        assert dialog.dia.detection_cutoff_combobox.currentIndex() == 2
        assert config_dict["detection_threshold_type"] == "deconvolution"

        tgui.shutdown()
