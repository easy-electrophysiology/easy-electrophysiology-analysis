from PySide2.QtTest import QTest
from PySide2 import QtWidgets, QtCore
import pytest
import sys
import os
import copy
import pandas as pd
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology.easy_electrophysiology.easy_electrophysiology import MainWindow, dump_analysis_after_software_crash
from ephys_data_methods import core_analysis_methods, event_analysis_master
from utils import utils
from setup_test_suite import GuiTestSetup
from sys import platform
os.environ["PYTEST_QT_API"] = "pyside2"
import keyboard
import glob
import test_curve_fitting
import shutil

class TestBackupOptions:

    @pytest.fixture(scope="function")
    def tgui(test):
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.setup_artificial_data("normalised")
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    def get_test_curve_fitting_data(self, tgui, rec_from, rec_to):

        results = {
            "event_num": tgui.get_data_from_qtable("record_num", rec_from, rec_to, "curve_fitting"),
            "baseline": tgui.get_data_from_qtable("baseline", rec_from, rec_to, "curve_fitting"),
            "max": tgui.get_data_from_qtable("max", rec_from, rec_to, "curve_fitting"),
            "amplitude": tgui.get_data_from_qtable("amplitude", rec_from, rec_to, "curve_fitting"),
        }
        return results

    def get_test_events_data(self, tgui, rec_from, rec_to):

        results = {
                "event_num": tgui.get_data_from_qtable("event_num", rec_from, rec_to, "events"),
                "template_num": tgui.get_data_from_qtable("template_num", rec_from, rec_to, "events"),
                "record_num": tgui.get_data_from_qtable("record_num", rec_from, rec_to, "events"),
                "event_time": tgui.get_data_from_qtable("event_time", rec_from, rec_to, "events"),
                "baseline": tgui.get_data_from_qtable("baseline", rec_from, rec_to, "events"),
                "peak": tgui.get_data_from_qtable("peak", rec_from, rec_to, "events"),
                "amplitude": tgui.get_data_from_qtable("amplitude", rec_from, rec_to, "events"),
                "rise_time": tgui.get_data_from_qtable("rise", rec_from, rec_to, "events"),
                "half_width": tgui.get_data_from_qtable("half_width", rec_from, rec_to, "events"),
                "decay_perc": tgui.get_data_from_qtable("decay_perc", rec_from, rec_to, "events"),
                "area_under_curve": tgui.get_data_from_qtable("area_under_curve", rec_from, rec_to, "events"),
                "event_period": tgui.get_data_from_qtable("event_period", rec_from, rec_to, "events"),
                "monoexp_fit_b0": tgui.get_data_from_qtable("monoexp_fit_b0", rec_from, rec_to, "events"),
                "monoexp_fit_b1": tgui.get_data_from_qtable("monoexp_fit_b1", rec_from, rec_to, "events"),
                "monoexp_fit_tau": tgui.get_data_from_qtable("monoexp_fit_tau", rec_from, rec_to, "events"),
        }

        return results

    def check_events_analysis(self, saved_events, events_data):  # TODO: break into this for final double check
        for col, key in enumerate(["event_num", "template_num", "record_num", "event_time", "baseline", "peak", "amplitude", "rise_time",
                                   "half_width", "decay_perc", "area_under_curve", "event_period", "monoexp_fit_b0", "monoexp_fit_b1", "monoexp_fit_tau"]):
            if key == "template_num":
                continue

            assert np.array_equal(saved_events.iloc[1:, col].to_numpy().astype(np.float64), events_data[key])

# Test -----------------------------------------------------------------------------------------------------------------------------------------------

    def test_all_analysis_are_saved_correctly(self, tgui):
        """
        NOTE: previous versions of the dump files are deleted before test starts otherwise test will fail
        """
        test_backup_dir = tgui.test_base_dir + "/test_backups"
        if os.path.isdir(test_backup_dir):
            shutil.rmtree(test_backup_dir)
        os.mkdir(test_backup_dir)

        tgui.mw.mw.actionBatch_Mode_ON.trigger()

        # Run all analysis and save results ----------------------------------------------------------------------------------------------------------

        tgui.run_spikecount_analysis(spike_detection_method="auto_record")
        spkcnt_data = copy.deepcopy(tgui.mw.loaded_file.spkcnt_data)

        tgui.setup_artificial_data("normalised", analysis_type="Ri")
        tgui.switch_to_input_resistance_and_set_im_combobox()
        tgui.run_ri_analysis_bounds(set_sag_analysis=True)
        ir_rec_from, ir_rec_to = [0, tgui.adata.num_recs - 1]  # save for the checking function later
        input_resistance_data = tgui.get_artificial_data_ir_parameters(ir_rec_from, ir_rec_to)

        tgui.setup_artificial_data("normalised", analysis_type="skinetics")
        tgui.run_artificial_skinetics_analysis()
        skinetics_data = copy.deepcopy(tgui.mw.loaded_file.skinetics_data)
        skinetics_data_table = tgui.reshape_skinetics_data_into_table(skinetics_data)
        skinetics_data_table = skinetics_data_table[:, 2:-2]

        tgui.setup_artificial_data("normalised", analysis_type="curve_fitting")
        test_curve_fitting.run_curve_fitting(tgui, vary_coefs=False, func_type="max", region_name="reg_1", norm_or_cumu_time="normalised", analyse_specific_recs=False, slope_override=False)
        curve_fitting_data = self.get_test_curve_fitting_data(tgui, 0, tgui.adata.num_recs)

        tgui.file_ext = ".abf"
        tgui.norm_time_data_path = os.path.join(tgui.test_base_dir, "vc_events_one_record.abf")
        tgui.setup_artificial_data("cumulative", analysis_type="events_one_record")
        tgui.run_artificial_events_analysis(tgui, "threshold")
        events_data = self.get_test_events_data(tgui, 0, tgui.adata.num_events())

        # stimulate crash and test -------------------------------------------------------------------------------------------------------------------

        tgui.mw.cfgs.main["crash_backup_path"] = test_backup_dir
        dump_analysis_after_software_crash(tgui.mw, test_backup_dir, "Test Error")

        # Load all tables and check
        backup_path = glob.glob(test_backup_dir + "/*")[0]

        saved_spkcnt_data = pd.read_csv(backup_path + "/ap_counting.csv")
        tgui.test_saved_spkcnt_to_loaded_file_data(saved_spkcnt_data, spkcnt_data, check_im=False)

        saved_input_resistance_data = pd.read_csv(backup_path + "/input_resistance.csv")
        tgui.test_saved_input_resistance_to_loaded_file_data(saved_input_resistance_data, input_resistance_data,
                                                             ir_rec_from, ir_rec_to)

        saved_skinetics_data = pd.read_csv(backup_path + "/ap_kinetics.csv")
        for col in range(skinetics_data_table.shape[1]):
            assert np.array_equal(saved_skinetics_data.iloc[1:, 2 + col].to_numpy().astype(np.float64), skinetics_data_table[:, col])

        saved_curve_fitting = pd.read_csv(backup_path + "/curve_fitting.csv")
        for col, key in enumerate(["event_num", "baseline", "max", "amplitude"]):
            assert np.array_equal(saved_curve_fitting.iloc[1:, col].to_numpy().astype(np.float64), curve_fitting_data[key])

        saved_events = pd.read_csv(backup_path + "/events.csv")
        self.check_events_analysis(saved_events, events_data)

        # check error log
        assert os.path.isfile(backup_path + "/error_log.json")

        # check save events_data, cannot use tgui.mw.mw.actionReset_to_Raw_Data.trigger() with new computer processor, getting
        # random seg fault that does not occur in real software.
        tgui.shutdown()
        tgui = GuiTestSetup("artificial_events_one_record")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.setup_artificial_data("cumulative", analysis_type="events_one_record")
        tgui.raise_mw_and_give_focus()

        # compared saved data with saved events
        tgui.mw.loaded_file.load_event_analysis(backup_path + "/saved_events_analysis.json")
        tgui.set_analysis_type("events_template_matching")
        events_data = self.get_test_events_data(tgui, 0, saved_events.shape[0])  # use num events from old analysis
        tgui.switch_mw_tab(1)
        self.check_events_analysis(saved_events, events_data)

        tgui.shutdown()