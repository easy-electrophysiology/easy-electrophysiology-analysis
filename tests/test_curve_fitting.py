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
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology.easy_electrophysiology.easy_electrophysiology import MainWindow
from ephys_data_methods import core_analysis_methods
from utils import utils
mouseClick = QTest.mouseClick  # There must be a better way!
keyPress = QTest.keyPress
keyClick = QTest.keyClick
from setup_test_suite import GuiTestSetup
from PySide2.QtCore import Signal
from dialog_menus.upsample_diaclass import UpSample
from dialog_menus.downsample_diaclass import DownSample
from dialog_menus.filter_data_diaclass import FilterData
import scipy.signal
import copy
import peakutils
from sys import platform
os.environ["PYTEST_QT_API"] = "pyside2"

"""
Some helper functions are accessible outside of the class so they can be used in other modules. However, they are not added
to tgui because there is too many of them. 
"""
REGIONS_TO_RUN = ["reg_1"]  # , "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"

def run_curve_fitting(tgui, vary_coefs, func_type, region_name, norm_or_cumu_time, analyse_specific_recs,
                      slope_override=False, set_options_only=False, pos_or_neg="pos"):
    """
    Coordinate inserting the specified functions into the data, coordinate switching to curve fitting on GUI
    and analysis setup and running analysis.

    Added slope override to insert sloping Im and Vm to check analysing across recs. Not the most intuitive for args but is v neat for code
    """
    insert_function = "slope" if slope_override else func_type
    tgui.update_curve_fitting_function(vary_coefs, insert_function,
                                       norm_or_cumu_time, pos_or_neg)  # order important this will overwrite analyse specific recs
    __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)
    setup_and_run_curve_fitting_analysis(tgui, func_type, region_name, rec_from, rec_to, set_options_only, pos_or_neg)

    return rec_from, rec_to


def setup_and_run_curve_fitting_analysis(tgui, func_type, region_name, rec_from, rec_to, set_options_only, pos_or_neg="pos"):
    """
    Setup the GUI for analysis. If region 1 is selected, no need to scroll through as it is initialised on
    curve fitting dialog open.

    If a single reg 2-6 is selected, click n times to select the region of interest.

    If all regions are selected to run, scroll through and run every region.

    Open the dialog box and click the region buttons to bring up the specified region. Set the combobox to the correct
    analysis, set initial est if required and run analysis.
    """
    tgui.set_analysis_type("curve_fitting")
    tgui.left_mouse_click(tgui.mw.mw.curve_fitting_show_dialog_button)

    if region_name == "all":
        for reg in ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]:

            if reg != "reg_1":
                tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.change_region_right_button)
            setup_gui_for_analysis_and_run(tgui, reg, func_type, rec_from, rec_to, set_options_only=set_options_only, pos_or_neg=pos_or_neg)

    else:
        quick_switch_to_region(tgui, tgui.mw.dialogs["curve_fitting"], int(region_name[-1]))
        setup_gui_for_analysis_and_run(tgui, region_name, func_type, rec_from, rec_to, set_options_only=set_options_only, pos_or_neg=pos_or_neg)

    tgui.mw.dialogs["curve_fitting"].hide()

def setup_gui_for_analysis_and_run(tgui, region_name, func_type, rec_from, rec_to, rec=0, set_options_only=False, pos_or_neg="pos"):
    """
    """
    dialog = tgui.mw.dialogs["curve_fitting"]

    tgui.set_combobox(dialog.dia.fit_type_combobox,
                      idx=get_combobox_idx_from_analysis_type(func_type))
    tgui.switch_checkbox(dialog.dia.link_and_unlink_curve_fit_region_checkbox,
                         on=False)

    if func_type in ["triexp", "biexp_event", "biexp_decay", "monoexp"]:
        set_direction_radiobutton(tgui, func_type, dialog, pos_or_neg)

    # Set initial est if required, this is an implicit test the bounds are set correctly.
    if func_type == "biexp_decay":
        override_biexp_decay_coefs(tgui, rec_from, rec_to, rec)

    if func_type == "triexp":
        override_tripex_coefs(tgui, rec_from, rec_to, rec)

    # Set the position of the bounds and run analyis across all recs. If recs are linked, only set the first rec
    # and the rest will follow)
    for rec in range(tgui.adata.num_recs):
        tgui.mw.update_displayed_rec(rec)
        tgui.mw.curve_fitting_regions[region_name].bounds["upper_exp_lr"].setRegion((tgui.adata.start_times[rec],  # confirmed LR is not on plot, but the bounds are still updated because this is not a condition
                                                                                     tgui.adata.stop_times[rec]))  # in curve_fitting_linear_regions.py update_bounds_after_mw_change() wheras it is is linear_regions.py for spkcnt, skinetics and Ri
        tgui.mw.curve_fitting_regions[region_name].bounds["upper_bl_lr"].setRegion((0,
                                                                                    tgui.adata.start_times[rec] - 1))
        if rec == 0 and tgui.mw.cfgs.graph_view_options["link_lr_across_recs"]:
            break

    if not set_options_only:
        tgui.left_mouse_click(dialog.dia.fit_button)
    QtWidgets.QApplication.processEvents()

def set_direction_radiobutton(tgui, func_type, dialog, pos_or_neg):

    radiobuttons = {
        "monoexp": {"pos": dialog.dia.monoexp_direction_pos_radiobutton,
                    "neg":  dialog.dia.monoexp_direction_neg_radiobutton},
        "biexp_decay": {"pos": dialog.dia.biexp_decay_direction_pos_radiobutton,
                        "neg":  dialog.dia.biexp_decay_direction_neg_radiobutton},
        "biexp_event": {"pos": dialog.dia.biexp_event_direction_pos_radiobutton,
                        "neg":  dialog.dia.biexp_event_direction_neg_radiobutton},
        "triexp": {"pos": dialog.dia.triexp_direction_pos_radiobutton,
                   "neg":  dialog.dia.triexp_direction_neg_radiobutton}
    }
    tgui.switch_checkbox(radiobuttons[func_type][pos_or_neg], on=True)

def get_combobox_idx_from_analysis_type(analysis_type):
    idx = {"min": 0,
           "max": 1,
           "mean": 2,
           "median": 3,
           "monoexp": 5,
           "biexp_decay": 6,
           "biexp_event": 7,
           "triexp": 8}

    return idx[analysis_type]


def override_biexp_decay_coefs(tgui, rec_from, rec_to, rec=0):
    """
    For biexp_decay and triexp, if the initial est is not set near the true bounds the algorithm will
    go to local minimum. This could yield a function that is extremely close to the original,
    but will not be with the exact coefs we used to generate hte function and test against.

    As such we override the initial est here to match the true coefs of the inserted fucntion. Note that
    this is only for the first record - for testing coefficients changing across records this creates
    some issues.
    """
    dialog = tgui.mw.dialogs["curve_fitting"].dia

    tgui.switch_groupbox(dialog.biexp_decay_coefficients_groupbox,
                         on=True)
    tgui.enter_number_into_spinbox(dialog.biexp_decay_initial_est_b0_spinbox,
                                   str(tgui.adata.offset))
    tgui.enter_number_into_spinbox(dialog.biexp_decay_initial_est_b1_spinbox,
                                   str(tgui.adata.get_b_coef("biexp_decay", "b1", rec_from, rec_to, rec)))
    tgui.enter_number_into_spinbox(dialog.biexp_decay_initial_est_tau1_spinbox,
                                   str(tgui.adata.get_tau("biexp_decay", "tau1", rec_from, rec_to, rec)))
    tgui.enter_number_into_spinbox(dialog.biexp_decay_initial_est_b2_spinbox,
                                   str(tgui.adata.get_b_coef("biexp_decay", "b2", rec_from, rec_to, rec)))
    tgui.enter_number_into_spinbox(dialog.biexp_decay_initial_est_tau2_spinbox,
                                   str(tgui.adata.get_tau("biexp_decay", "tau2", rec_from, rec_to, rec)))


def override_tripex_coefs(tgui, rec_from, rec_to, rec=0):
    """
    See override_biexp_decay_coefs.
    """
    dialog = tgui.mw.dialogs["curve_fitting"].dia

    tgui.switch_groupbox(dialog.triexp_coefficients_groupbox,
                         on=True)

    tgui.enter_number_into_spinbox(dialog.triexp_initial_est_b0_spinbox,
                                   str(tgui.adata.offset))
    tgui.enter_number_into_spinbox(dialog.triexp_initial_est_b1_spinbox,
                                   str(tgui.adata.get_b_coef("triexp", "b1", rec_from, rec_to, rec)))
    tgui.enter_number_into_spinbox(dialog.triexp_initial_est_tau1_spinbox,
                                   str(tgui.adata.get_tau("triexp", "tau1", rec_from, rec_to, rec)))
    tgui.enter_number_into_spinbox(dialog.triexp_initial_est_b2_spinbox,
                                   str(tgui.adata.get_b_coef("triexp", "b2", rec_from, rec_to, rec)))
    tgui.enter_number_into_spinbox(dialog.triexp_initial_est_tau2_spinbox,
                                   str(tgui.adata.get_tau("triexp", "tau2", rec_from, rec_to, rec)))
    tgui.enter_number_into_spinbox(dialog.triexp_initial_est_b3_spinbox,
                                   str(tgui.adata.get_b_coef("triexp", "b3", rec_from, rec_to, rec)))
    tgui.enter_number_into_spinbox(dialog.triexp_initial_est_tau3_spinbox,
                                   str(tgui.adata.get_tau("triexp", "tau3", rec_from, rec_to, rec)))

def quick_switch_to_region(tgui, curve_fitting_dialog, region):

    for region_idx in range(6):
        tgui.left_mouse_click(curve_fitting_dialog.dia.change_region_left_button)

    for region_idx in range(region - 1):
        tgui.left_mouse_click(curve_fitting_dialog.dia.change_region_right_button)

def fill_in_starting_estimate(tgui, func_type, rec_from=None, rec_to=None, rec=0, analyse_one_rec_only=False):
    """
    """
    if analyse_one_rec_only:
        tgui.set_analyse_specific_recs(rec,
                                       rec + 1)

    if func_type == "biexp_decay":
        override_biexp_decay_coefs(tgui, rec_from, rec_to, rec)

    if func_type == "triexp":
        override_tripex_coefs(tgui, rec_from, rec_to, rec)

class TestCurveFitting:

    @pytest.fixture(scope="function", params=["normalised", "cumulative"],
                    ids=["normalised", "cumulative"])
    def tgui(test, request):
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.setup_artificial_data(request.param, analysis_type="curve_fitting")
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    # Helpers
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def check_all_sublists_are_equal(self, parent_lists):
        evaluate = []
        for list_ in parent_lists:
            evaluate.append(utils.allclose(list_, parent_lists[0], 1e-08))

        return all(evaluate)

    # Testing Convenience Function
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def check_qtable_peak_baseline_amplitude(self, tgui, test_peak_ims, func_type, rec_from, rec_to):
        """
        Convenience function for checking peak, amplitude and baseline
        for test_qtable_num_recs_amplitude_baseline_for_all_funs()
        """
        assert np.array_equal(test_peak_ims,
                              tgui.get_data_from_qtable("peak", rec_from, rec_to, "curve_fitting")), "peak " + func_type

        assert utils.allclose(tgui.adata.get_b0(rec_from, rec_to),
                              tgui.get_data_from_qtable("baseline", rec_from, rec_to, "curve_fitting"),
                              1e-08), "baseline " + func_type

        assert utils.allclose(test_peak_ims - tgui.adata.offset,
                              tgui.get_data_from_qtable("amplitude", rec_from, rec_to, "curve_fitting"),
                              1e-08), "amplitude " + func_type

    def quick_override_starting_estimate_for_testing_different_curves_across_records(self, tgui, func_type, rec):
        fill_in_starting_estimate(tgui, func_type, rec=rec)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)
        QtWidgets.QApplication.processEvents()

    def check_triexp_coefficients_across_recs(self, tgui, data, func_type, rec_from, rec_to, rec):
        """
        Checking for triexp across tests is a complete nightmare as the fitting
        procedure is very sensitive and any coefficient can change. Here at least one
        of these should be correct if everything is setup probably. Remember, the purpose of this
        test is not to check the output of the fitting funtion itself but ensure that
        all coefficients are analysed and displayed correctly across records.
        """
        peak_im = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec) + tgui.adata.get_b_coef(func_type, "b2", rec_from, rec_to, rec) + tgui.adata.get_b_coef(func_type, "b3", rec_from, rec_to, rec)
        sum_b = data["b1"] + data["b2"] + data["b3"]
        test_sum_b = peak_im

        sum_tau = data["tau1"] + data["tau2"] + data["tau3"]
        test_sum_tau = tgui.adata.get_tau(func_type, "tau1", rec_from, rec_to, rec) + \
                       tgui.adata.get_tau(func_type, "tau2", rec_from, rec_to, rec) + \
                       tgui.adata.get_tau(func_type, "tau3", rec_from, rec_to, rec)

        test_fit = core_analysis_methods.triexp_decay_function(np.arange(tgui.adata.function_samples),
                                                               tgui.adata.cannonical_coefs[func_type] * tgui.adata.coef_offsets[rec][
                                                                   func_type])

        test = utils.allclose(sum_b, test_sum_b, 1e-05) or utils.allclose(sum_tau, test_sum_tau, 1e-05) or utils.allclose(data["fit"][:-1],
                                                                                                                          test_fit, 1e-05)

        return test, peak_im

    def check_plots(self, tgui, region_name, y_line=None, x_line=None, y_scatter=None, x_scatter=None):

        if np.any(y_line) and np.any(x_line):
            self.check_data_against_plot(region_name, tgui.mw.loaded_file_plot.curve_fitting_line_plots, y_line, x_line)

        if np.any(y_scatter) and np.any(x_scatter):
            self.check_data_against_plot(region_name, tgui.mw.loaded_file_plot.curve_fitting_scatter_plots, y_scatter, x_scatter)

    def check_data_against_plot(self, region_name, plot, y, x):

        for reg in ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]:
            if region_name == reg:
                assert (plot[reg].xData == x).all()
                assert (plot[reg].yData == y).all()
            else:
                assert plot[reg].xData[0] is None
                assert plot[reg].yData[0] is None

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Fits while varying size of coefficients within a single function (e.g. b1 <
# b2 in event_decay)
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("vary_coefs", [False, "within_function"])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("region_name", REGIONS_TO_RUN)
    @pytest.mark.parametrize("func_type", ["monoexp", "biexp_event", "biexp_decay", "triexp"])
    @pytest.mark.parametrize("pos_or_neg", ["neg", "pos"])
    def test_curve_fitting_model_with_changing_coefficients_within_a_function(self, tgui, func_type, vary_coefs, analyse_specific_recs, region_name, pos_or_neg):
        """
        Major test for all curve fitting possibilities and settings. Coefficients may be varied on a coefficient by coefficient
        bassis (varied "within_function") or not at all (all beta, tau are exactly as specified in tgui.adata.cannonical coefs).
        Test every region, and with / without selecting a subset of records to analyse (analyse sepcific recs).
        Every function type is checked here because making functions for each led to massive code re-use
        setting up the record / data model loop and checking the peak / amplitude / baseine.

        It is also useful to revise the form of the functions we fit. They are typically made up of a liear combination
        of beta values which are scaled as a function of e^x. Thus the amplitude of a biexpoentnial decay
        function will be b1 + b2.

        In general, the fitting is not 100% perfect to the true coefficients. So sometimes we test only to 0.1 or 0 dp. The main
        purpose of these tests it to check all results are calculated and displayed correctly.

        Tests:

        "vary coefs": False, "within_function", "across_records",
                      The setup for this is that the functions inserted into the data by the generate class
                      are all based on a set of canonical coefficients for beta and tau. However, to model
                      realistic data these can be varied, so each coefficient with a function is different
                      e.g. b1 != b2. Here the coefficients are the same for every record, but different within a function.

        "analyse_specific_recs": Analyse a subset of recs here. only 5 are used with low number of samples because
                                 the analysis takes so long to run.

         "region_name": Test every region (1-6). This is a bit redundant as they all use very similar back end
                        but worth checking just in case.

        Issues:
                To get the measured coefficients to exactly match the specified coefficients, the initial est must be
                very close to the true coefficients or an alternative, almost equally accurate local minimum will be found.
                This is only an issue with functions with mulitple beta / tau params (e.g. triexp).
                The bounds are set only based on the first record in the data e.g. override_biexp_decay_coefs()
                This means we cannot test these functions accurately when changing coefficients across records because the initial
                est will not be good. As such the coefficient are varied only within a function in these tests and varying
                across records is done separately.
        """
        tgui.mw.mw.actionBatch_Mode_ON.trigger()

        for filenum in range(3):

            rec_from, rec_to = run_curve_fitting(tgui, vary_coefs, func_type, region_name, tgui.time_type, analyse_specific_recs, pos_or_neg=pos_or_neg)

            # Test model and tabldata by rec (cannot slice into the dict structure)
            for rec in range(tgui.adata.num_recs):

                tgui.mw.update_displayed_rec(rec)

                for data in [tgui.mw.loaded_file.curve_fitting_results[region_name]["data"][rec],
                             tgui.mw.stored_tabledata.curve_fitting[filenum][region_name]["data"][rec]]:

                    if not tgui.mw.cfgs.rec_within_analysis_range("curve_fitting",
                                                                  rec):
                        assert data == {}
                        continue

                    assert utils.allclose(data["b0"], tgui.adata.get_b0(rec_from, rec_to, rec), 1e-08), "b0"  # actually first not used

                    # Monoexp ----------------------------------------------------------------------------------------------------------------------------

                    if func_type == "monoexp":
                        peak_im = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec)

                        assert utils.allclose(data["b1"], peak_im, 1e-08), "b1"
                        assert utils.allclose(data["tau"], tgui.adata.get_tau(func_type, "tau", rec_from, rec_to, rec), 1e-08), "tau"
                        assert data["r2"] == 1

                    # Biexp decay ------------------------------------------------------------------------------------------------------------------------

                    if func_type == "biexp_decay":
                        peak_im = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec) + \
                                  tgui.adata.get_b_coef(func_type, "b2", rec_from, rec_to, rec)

                        assert utils.allclose(data["b1"],
                                              tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec),
                                              1e-00), "b1"
                        assert utils.allclose(data["b2"],
                                              tgui.adata.get_b_coef(func_type, "b2", rec_from, rec_to, rec),
                                              1e-00), "b2"

                        assert utils.allclose(data["tau1"], tgui.adata.get_tau(func_type, "tau1", rec_from, rec_to, rec), 1e-02), "tau1"
                        assert utils.allclose(data["tau2"], tgui.adata.get_tau(func_type, "tau2", rec_from, rec_to, rec), 1e-02), "tau2"

                        assert data["r2"] == 1

                    # Biexp Event ------------------------------------------------------------------------------------------------------------------------

                    if func_type == "biexp_event":
                        peak_im = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec)

                        assert utils.allclose(data["b1"], peak_im, 1e-05), "b1"
                        assert utils.allclose(data["fit_rise"], tgui.adata.get_tau(func_type, "fit_rise", rec_from, rec_to, rec), 1e-05), "fit rise"
                        assert utils.allclose(data["fit_decay"], tgui.adata.get_tau(func_type, "fit_decay", rec_from, rec_to, rec), 1e-05), "fit decay"

                        assert data["r2"] == 1

                        continue  # peak is nan for biexp event and calculated from the event_info

                    # Tripexonential ---------------------------------------------------------------------------------------------------------------------

                    if func_type == "triexp":
                        peak_im = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec) + \
                                  tgui.adata.get_b_coef(func_type, "b2", rec_from, rec_to, rec) + \
                                  tgui.adata.get_b_coef(func_type, "b3", rec_from, rec_to, rec)

                        assert utils.allclose(data["b1"],
                                              tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec),
                                              1e-00), "b1"

                        assert utils.allclose(data["b2"], tgui.adata.get_b_coef(func_type, "b2", rec_from, rec_to, rec), 1e-00), "b2"

                        assert utils.allclose(data["b3"], tgui.adata.get_b_coef(func_type, "b3", rec_from, rec_to, rec), 1e-00), "b3"
                        assert utils.allclose(data["tau1"], tgui.adata.get_tau(func_type, "tau1", rec_from, rec_to, rec), 1e-02), "tau1"

                        assert utils.allclose(data["tau2"], tgui.adata.get_tau(func_type, "tau2", rec_from, rec_to, rec), 1e-02), "tau2"
                        assert utils.allclose(data["tau3"], tgui.adata.get_tau(func_type, "tau3", rec_from, rec_to, rec), 1e-02), "tau3"

                        assert data["r2"] == 1

                    test_peak_im = peak_im + tgui.adata.offset
                    assert data["peak"] == test_peak_im, "peak"
                    assert data["peak_time"] == tgui.adata.start_times[rec], "peak_time"  # not actually on the table at this time.

                    assert utils.allclose(data["baseline"], tgui.adata.offset, 1e-08), "baseline"
                    assert utils.allclose(data["amplitude"], test_peak_im - tgui.adata.offset, 1e-08), "amplitude"

            tgui.setup_artificial_data(tgui.time_type, analysis_type="curve_fitting")
        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Test while varying coefficients across records
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("region_name", REGIONS_TO_RUN)
    @pytest.mark.parametrize("func_type", ["monoexp", "biexp_event", "biexp_decay", "triexp"])
    @pytest.mark.parametrize("pos_or_neg", ["pos", "neg"])
    def test_curve_fitting_difference_in_amplitude_across_records(self, tgui, func_type, analyse_specific_recs, region_name, pos_or_neg):
        """

        This function tests results when the inserted functions coefficients vary across records.
        Very simialr to "test_curve_fitting_model_with_changing_coefficients_within_a_function", see that function for more details.

        This is slightly more complicated as initial estimates are set before analysis.
        This means only one set of initial estimates (set based on the coefficients of the first record) are used and so they will be less
        approrpiate for the functions later in the record. This is mainly a problem for biexp_decay and triexp function which multiple b or tau.

        The main focus of this is just to check that the measured function params are measured and shown in the correct order on the table.
        As such a fairly loose method of checking the biexp_decay and tripexp where the coefficients are summed, and either b, tau or
        the fit matches the inserted function / coefficients by 0 dp. This works well enough to acheive our aims. In practice, better
        fitting would be obtained by setting the bounds. Additionally, the fits still look prety good and it is usually just 1 sample / 2000
        that is out higher dp. resolution.

        """
        vary_coefs = "across_records"
        tgui.mw.mw.actionBatch_Mode_ON.trigger()

        for filenum in range(3):

            rec_from, rec_to = run_curve_fitting(tgui, vary_coefs, func_type, region_name, tgui.time_type, analyse_specific_recs, pos_or_neg=pos_or_neg)

            # Test model and tabldata by rec (cannot slice into the dict structure)
            for rec in range(tgui.adata.num_recs):

                self.quick_override_starting_estimate_for_testing_different_curves_across_records(tgui, func_type, rec)
                tgui.mw.update_displayed_rec(rec)

                for data in [tgui.mw.loaded_file.curve_fitting_results[region_name]["data"][rec],
                             tgui.mw.stored_tabledata.curve_fitting[filenum][region_name]["data"][rec]]:

                    if not tgui.mw.cfgs.rec_within_analysis_range("curve_fitting", rec):
                        assert data == {}
                        continue

                    # Monoexp --------------------------------------------------------------------------------------------------------------------------------

                    if func_type == "monoexp":
                        peak_im = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec)

                        assert utils.allclose(data["b1"], peak_im, 1e-08), "b1"
                        assert utils.allclose(data["tau"], tgui.adata.get_tau(func_type, "tau", rec_from, rec_to, rec), 1e-08), "tau"

                    # Biexp Decay  ---------------------------------------------------------------------------------------------------------------------------

                    if func_type == "biexp_decay":
                        peak_im = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec) + tgui.adata.get_b_coef(func_type, "b2", rec_from,
                                                                                                                        rec_to,
                                                                                                                        rec)
                        sum_b = data["b1"] + data["b2"]
                        sum_test_b = peak_im
                        sum_tau = data["tau1"] + data["tau2"]
                        sum_test_tau = tgui.adata.get_tau(func_type, "tau1", rec_from, rec_to, rec) + tgui.adata.get_tau(func_type, "tau2", rec_from,
                                                                                                                         rec_to, rec)

                        test_fit = core_analysis_methods.biexp_decay_function(np.arange(tgui.adata.function_samples),
                                                                              tgui.adata.cannonical_coefs["biexp_decay"] *
                                                                              tgui.adata.coef_offsets[rec][
                                                                                  func_type])
                        test = utils.allclose(sum_b, sum_test_b, 1e-05) or utils.allclose(sum_tau, sum_test_tau, 1e-05) or utils.allclose(
                            data["fit"][:-1],
                            test_fit, 1e-05)
                        assert test, "biexp decay coeffs are completely wrong"

                    # Biexp Event  ---------------------------------------------------------------------------------------------------------------------------

                    if func_type == "biexp_event":
                        peak_im = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to, rec)
                        assert utils.allclose(data["b1"], peak_im, 1e-05), "b1"
                        assert utils.allclose(data["fit_rise"], tgui.adata.get_tau(func_type, "fit_rise", rec_from, rec_to, rec), 1e-05), "fit rise"
                        assert utils.allclose(data["fit_decay"], tgui.adata.get_tau(func_type, "fit_decay", rec_from, rec_to, rec),
                                              1e-05), "fit decay"

                        self.check_plots(tgui, region_name,
                                         y_line=data["fit"], x_line=data["fit_time"])
                        continue  # peak is nan for biexp event and calculated from the event_info

                    # Tripexonential  ------------------------------------------------------------------------------------------------------------------------

                    if func_type == "triexp":
                        test, peak_im = self.check_triexp_coefficients_across_recs(tgui, data, func_type, rec_from, rec_to, rec)
                        assert test, "triexp coeffs are completely wrong"

                    # Peak, Amplitude, Baseline
                    test_peak_im = peak_im + tgui.adata.offset
                    assert data["peak"] == test_peak_im, "peak"
                    assert data["peak_time"] == tgui.adata.start_times[rec], "peak time"

                    assert utils.allclose(data["baseline"], tgui.adata.offset, 1e-08), "baseline"
                    assert utils.allclose(data["amplitude"], test_peak_im - tgui.adata.offset, 1e-08), "amplitude"

                    self.check_plots(tgui, region_name,
                                     y_line=data["fit"], x_line=data["fit_time"],
                                     y_scatter=data["peak"], x_scatter=data["peak_time"])

            tgui.setup_artificial_data(tgui.time_type, analysis_type="curve_fitting")
        tgui.shutdown()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test varying coefficients within function - QTable results
# ----------------------------------------------------------------------------------------------------------------------------------------------------


    def test_analyse_all_vs_analyse_single_region(self, tgui):
        """
        run_curve_fitting() runs through the dialog, so check the panel buttons
        """
        rec = 0
        rec_from, rec_to = run_curve_fitting(tgui,
                                             vary_coefs=False,
                                             func_type="monoexp",
                                             region_name="all",
                                             norm_or_cumu_time="normalised",
                                             analyse_specific_recs=False,
                                             set_options_only=True)

        tgui.left_mouse_click(tgui.mw.mw.curve_fitting_fit_selected_region_button)

        data = tgui.mw.loaded_file.curve_fitting_results["reg_6"]["data"][rec]
        peak_im = tgui.adata.get_b_coef("monoexp", "b1", rec_from, rec_to, rec)
        assert utils.allclose(data["b1"], peak_im, 1e-08), "b1"
        assert utils.allclose(data["tau"], tgui.adata.get_tau("monoexp", "tau", rec_from, rec_to, rec), 1e-08), "tau"

        for reg in ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5"]:
            data = tgui.mw.loaded_file.curve_fitting_results[reg]["data"]
            assert data is None, reg

        tgui.left_mouse_click(tgui.mw.mw.curve_fitting_fit_all_regions_button)
        for reg in ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]:
            data = tgui.mw.loaded_file.curve_fitting_results["reg_6"]["data"][rec]
            peak_im = tgui.adata.get_b_coef("monoexp", "b1", rec_from, rec_to, rec)
            assert utils.allclose(data["b1"], peak_im, 1e-08), "b1"
            assert utils.allclose(data["tau"], tgui.adata.get_tau("monoexp", "tau", rec_from, rec_to, rec), 1e-08), "tau"

    @pytest.mark.parametrize("vary_coefs", [False, "within_function", "across_records"])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("region_name", REGIONS_TO_RUN)
    @pytest.mark.parametrize("func_type", ["monoexp", "biexp_decay", "triexp"])
    def test_qtable_num_recs_peak_amplitude_baseline_for_all_func_types(self, tgui, func_type, vary_coefs, analyse_specific_recs, region_name):
        """
        event not tested as stored in event info and calc differencely
        Only num recs are all equal are tested across all regions as these are checked below.
        Tests whether data are shown on graph corrected if plot individually or all together.
        """
        rec_from, rec_to = run_curve_fitting(tgui, vary_coefs, func_type, region_name, tgui.time_type, analyse_specific_recs)

        # Test Num Recs
        table_recs = tgui.get_data_from_qtable("record_num", rec_from, rec_to, "curve_fitting")
        assert np.array_equal(table_recs,
                              (np.arange(rec_from, rec_to + 1) + 1)), "table recs"

        # Test Peaks / Amplitude / Baseline
        if func_type == "monoexp":
            peak_ims = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to)
        elif func_type == "biexp_decay":
            peak_ims = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to) + \
                       tgui.adata.get_b_coef(func_type, "b2", rec_from, rec_to)
        elif func_type == "triexp":
            peak_ims = tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to) + \
                       tgui.adata.get_b_coef(func_type, "b2", rec_from, rec_to) + \
                       tgui.adata.get_b_coef(func_type, "b3", rec_from, rec_to)

        test_peak_ims = peak_ims + tgui.adata.offset
        self.check_qtable_peak_baseline_amplitude(tgui, test_peak_ims, func_type, rec_from, rec_to)

    @pytest.mark.parametrize("vary_coefs", [False, "within_function"])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("region_name", REGIONS_TO_RUN)
    def test_monoexp_qtable_with_changing_coefficients_within_a_function(self, tgui, vary_coefs, analyse_specific_recs, region_name):
        """
        """
        rec_from, rec_to = run_curve_fitting(tgui, vary_coefs, "monoexp", region_name, tgui.time_type, analyse_specific_recs)

        assert utils.allclose(tgui.adata.get_b0(rec_from, rec_to),
                              tgui.get_data_from_qtable("b0", rec_from, rec_to, "curve_fitting"),
                              1e-08), "b0"

        assert utils.allclose(tgui.adata.get_b_coef("monoexp", "b1", rec_from, rec_to),
                              tgui.get_data_from_qtable("b1", rec_from, rec_to, "curve_fitting"),
                              1e-08), "b1"

        assert utils.allclose(tgui.adata.get_tau("monoexp", "tau", rec_from, rec_to),
                              tgui.get_data_from_qtable("tau", rec_from, rec_to, "curve_fitting"),
                              1e-08), "tau"

    @pytest.mark.parametrize("vary_coefs", [False, "within_function"])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("region_name", REGIONS_TO_RUN)
    def test_biexp_decay_qtable_with_changing_coefficients_within_a_function(self, tgui, vary_coefs, analyse_specific_recs, region_name):
        """
        """
        rec_from, rec_to = run_curve_fitting(tgui, vary_coefs, "biexp_decay", region_name, tgui.time_type, analyse_specific_recs)

        assert utils.allclose(tgui.adata.get_b0(rec_from, rec_to),
                              tgui.get_data_from_qtable("b0", rec_from, rec_to, "curve_fitting"),
                              1e-2), "b0"

        assert utils.allclose(tgui.adata.get_b_coef("biexp_decay", "b1", rec_from, rec_to),
                              tgui.get_data_from_qtable("b1", rec_from, rec_to, "curve_fitting"),
                              1e-0), "b1"

        assert utils.allclose(tgui.adata.get_tau("biexp_decay", "tau1", rec_from, rec_to),
                              tgui.get_data_from_qtable("tau1", rec_from, rec_to, "curve_fitting"),
                              1e-2), "tau1"

        assert utils.allclose(tgui.adata.get_b_coef("biexp_decay", "b2", rec_from, rec_to),
                              tgui.get_data_from_qtable("b2", rec_from, rec_to, "curve_fitting"),
                              1e-0), "b2"

        assert utils.allclose(tgui.adata.get_tau("biexp_decay", "tau2", rec_from, rec_to),
                              tgui.get_data_from_qtable("tau2", rec_from, rec_to, "curve_fitting"),
                              1e-2), "tau2"

    @pytest.mark.parametrize("vary_coefs", [False, "within_function"])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("region_name", REGIONS_TO_RUN)
    def test_biexp_event_qtable_with_changing_coefficients_within_a_function(self, tgui, vary_coefs, analyse_specific_recs, region_name):
        """
        """
        rec_from, rec_to = run_curve_fitting(tgui, vary_coefs, "biexp_event", region_name, tgui.time_type, analyse_specific_recs)

        assert utils.allclose(tgui.adata.get_b0(rec_from, rec_to),
                              tgui.get_data_from_qtable("b0", rec_from, rec_to, "curve_fitting"),
                              1e-05), "b0"

        assert utils.allclose(tgui.adata.get_b_coef("biexp_event", "b1", rec_from, rec_to),
                              tgui.get_data_from_qtable("b1", rec_from, rec_to, "curve_fitting"),
                              1e-05), "b1"

        assert utils.allclose(tgui.adata.get_tau("biexp_event", "fit_rise", rec_from, rec_to),
                              tgui.get_data_from_qtable("fit_rise", rec_from, rec_to, "curve_fitting"),
                              1e-05), "fit rise"

        assert utils.allclose(tgui.adata.get_tau("biexp_event", "fit_decay", rec_from, rec_to),
                              tgui.get_data_from_qtable("fit_decay", rec_from, rec_to, "curve_fitting"),
                              1e-05), "fit_decay"

    @pytest.mark.parametrize("vary_coefs", [False, "within_function"])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("region_name", REGIONS_TO_RUN)
    def test_triexp_qtable_with_changing_coefficients_within_a_function(self, tgui, vary_coefs, analyse_specific_recs, region_name):
        """
        """
        rec_from, rec_to = run_curve_fitting(tgui, vary_coefs, "triexp", region_name, tgui.time_type, analyse_specific_recs)

        assert utils.allclose(tgui.adata.get_b0(rec_from, rec_to),
                              tgui.get_data_from_qtable("b0", rec_from, rec_to, "curve_fitting"),
                              1e-02), "b0"

        assert utils.allclose(tgui.adata.get_b_coef("triexp", "b1", rec_from, rec_to),
                              tgui.get_data_from_qtable("b1", rec_from, rec_to, "curve_fitting"),
                              1e-00), "b1"

        assert utils.allclose(tgui.adata.get_tau("triexp", "tau1", rec_from, rec_to),
                              tgui.get_data_from_qtable("tau1", rec_from, rec_to, "curve_fitting"),
                              1e-02), "tau1"

        assert utils.allclose(tgui.adata.get_b_coef("triexp", "b2", rec_from, rec_to),
                              tgui.get_data_from_qtable("b2", rec_from, rec_to, "curve_fitting"),
                              1e-00), "b2"  # 1dp for triexp coefs

        assert utils.allclose(tgui.adata.get_tau("triexp", "tau2", rec_from, rec_to),
                              tgui.get_data_from_qtable("tau2", rec_from, rec_to, "curve_fitting"),
                              1e-02), "tau2"

        assert utils.allclose(tgui.adata.get_b_coef("triexp", "b3", rec_from, rec_to),
                              tgui.get_data_from_qtable("b3", rec_from, rec_to, "curve_fitting"),
                              1e-00), "b3"

        assert utils.allclose(tgui.adata.get_tau("triexp", "tau3", rec_from, rec_to),
                              tgui.get_data_from_qtable("tau3", rec_from, rec_to, "curve_fitting"),
                              1e-02), "tau3"

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test table output when running every region at once
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("func_type", ["monoexp", "biexp_event", "biexp_decay", "triexp"])
    def test_all_recs_analysed_together_curve_fit(self, tgui, func_type, analyse_specific_recs):
        """
        Test that when all regions are analysed the data is still shown correctly on the Table.

        A little bit messy as B0, B1, are shared between all, tau1, tau2 are shared between triexp
        and biexp-decay, and monoexp tau1 is called tau. However done in this way to avoid extensive code
        re-use.
        """
        rec_from, rec_to = run_curve_fitting(tgui, False, func_type, "all", tgui.time_type, analyse_specific_recs)

        # Table Recs
        table_recs = tgui.get_data_from_qtable("record_num", rec_from, rec_to, "curve_fitting",
                                               return_regions=True)
        assert self.check_all_sublists_are_equal(table_recs), "table recs all regions equal"
        assert np.array_equal(table_recs[0],
                              (np.arange(rec_from, rec_to + 1) + 1)), "table recs"

        # B0 -----------------------------------------------------------------------------------------------------------------------------------------

        b0_results = tgui.get_data_from_qtable("b0", rec_from, rec_to, "curve_fitting",
                                               return_regions=True)
        assert self.check_all_sublists_are_equal(b0_results), "b0 all regions equal"
        assert utils.allclose(b0_results[0], tgui.adata.get_b0(rec_from, rec_to), 1e-02), "b0"

        # B1 -----------------------------------------------------------------------------------------------------------------------------------------

        b1_results = tgui.get_data_from_qtable("b1", rec_from, rec_to, "curve_fitting",
                                               return_regions=True)

        assert self.check_all_sublists_are_equal(b1_results), "b1 all regions equal"
        assert utils.allclose(b1_results[0],
                              tgui.adata.get_b_coef(func_type, "b1", rec_from, rec_to),
                              1e-00), "b1"

        # Tau - Monoexonential -----------------------------------------------------------------------------------------------------------------------

        if func_type == "monoexp":
            tau_results = tgui.get_data_from_qtable("tau", rec_from, rec_to, "curve_fitting",
                                                    return_regions=True)
            assert self.check_all_sublists_are_equal(tau_results), "tau all regions equal"
            assert utils.allclose(tau_results[0],
                                  tgui.adata.get_tau(func_type, "tau", rec_from, rec_to),
                                  1e-08), "tau"

        # Biexponential Event ------------------------------------------------------------------------------------------------------------------------

        if func_type == "biexp_event":
            rise_results = tgui.get_data_from_qtable("fit_rise", rec_from, rec_to, "curve_fitting",
                                                     return_regions=True)
            assert self.check_all_sublists_are_equal(rise_results), "rise all regions equal"
            assert utils.allclose(rise_results[0],
                                  tgui.adata.get_tau(func_type, "fit_rise", rec_from, rec_to),
                                  1e-05), "rise"

            decay_results = tgui.get_data_from_qtable("fit_decay", rec_from, rec_to, "curve_fitting",
                                                      return_regions=True)
            assert self.check_all_sublists_are_equal(decay_results), "decay all regions equal"
            assert utils.allclose(decay_results[0],
                                  tgui.adata.get_tau(func_type, "fit_decay", rec_from, rec_to),
                                  1e-05), "decay"

        # Biexponential Decay ------------------------------------------------------------------------------------------------------------------------

        if func_type == "biexp_decay" or func_type == "triexp":
            tau1_results = tgui.get_data_from_qtable("tau1", rec_from, rec_to, "curve_fitting",
                                                     return_regions=True)
            assert self.check_all_sublists_are_equal(tau1_results), "tau1 all regions equal"
            assert utils.allclose(tau1_results[0],
                                  tgui.adata.get_tau(func_type, "tau1", rec_from, rec_to),
                                  1e-02), "tau1"

            b2_results = tgui.get_data_from_qtable("b2", rec_from, rec_to, "curve_fitting",
                                                   return_regions=True)
            assert self.check_all_sublists_are_equal(b2_results), "b2 all regions equal"
            assert utils.allclose(b2_results[0], tgui.adata.get_b_coef(func_type, "b2", rec_from, rec_to),
                                  1e-00), "b2"  # not sure why this only 1 dp for triexp

            tau2_results = tgui.get_data_from_qtable("tau2", rec_from, rec_to, "curve_fitting",
                                                     return_regions=True)
            assert self.check_all_sublists_are_equal(tau2_results), "tau2 all regions equal"
            assert utils.allclose(tau2_results[0],
                                  tgui.adata.get_tau(func_type, "tau2", rec_from, rec_to),
                                  1e-02), "tau2"

        # Biexponential Decay ------------------------------------------------------------------------------------------------------------------------

        if func_type == "triexp":
            b3_results = tgui.get_data_from_qtable("b3", rec_from, rec_to, "curve_fitting",
                                                   return_regions=True)
            assert self.check_all_sublists_are_equal(b3_results), "b3 all regions equal"
            assert utils.allclose(b3_results[0],
                                  tgui.adata.get_b_coef(func_type, "b3", rec_from, rec_to),
                                  1e-00), "b3"

            tau3_results = tgui.get_data_from_qtable("tau3", rec_from, rec_to, "curve_fitting",
                                                     return_regions=True)
            assert self.check_all_sublists_are_equal(tau3_results), "tau3 all regions equal"
            assert utils.allclose(tau3_results[0],
                                  tgui.adata.get_tau(func_type, "tau3", rec_from, rec_to),
                                  1e-02), "tau3"

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Min, Max, Mean, Min
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("vary_coefs", [False, "within_function", "across_records"])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("region_name", REGIONS_TO_RUN)
    @pytest.mark.parametrize("func_type", ["min", "max", "mean", "median"])
    def test_min_max_mean_median_and_plots(self, tgui, func_type, vary_coefs, analyse_specific_recs, region_name):
        """
        See test_curve_fitting_model_with_changing_coefficients_within_a_function(), same idea but
        easier because testing min, max, mean, median
        """
        __, __ = run_curve_fitting(tgui, vary_coefs, func_type, region_name, tgui.time_type, analyse_specific_recs)

        # Test model and tabldata by rec (cannot slice into the dict structure)
        for rec in range(tgui.adata.num_recs):

            tgui.mw.update_displayed_rec(rec)

            for data in [tgui.mw.loaded_file.curve_fitting_results[region_name]["data"][rec],
                         tgui.mw.stored_tabledata.curve_fitting[0][region_name]["data"][rec]]:

                if not tgui.mw.cfgs.rec_within_analysis_range("curve_fitting",
                                                              rec):
                    assert data == {}
                    continue

                inserted_function = tgui.mw.loaded_file.data.vm_array[rec][tgui.adata.start_sample:tgui.adata.stop_sample + 1]

                if func_type == "min":
                    assert data["min"] == np.min(inserted_function), "min"
                    self.check_plots(tgui, region_name, y_scatter=data["min"], x_scatter=data["time"])

                if func_type == "max":
                    assert data["max"] == np.max(inserted_function), "max"
                    self.check_plots(tgui, region_name, y_scatter=data["max"], x_scatter=data["time"])

                if func_type == "mean":
                    assert data["mean"] == np.mean(inserted_function), "mean"
                    self.check_plots(tgui, region_name, y_line=np.tile(data["mean"], len(data["fit_time"])), x_line=data["fit_time"])

                if func_type == "median":
                    assert utils.allclose(data["median"], np.median(inserted_function)), "median"
                    self.check_plots(tgui, region_name, y_line=np.tile(data["median"], len(data["fit_time"])), x_line=data["fit_time"])

                assert utils.allclose(data["baseline"], tgui.adata.offset, 1e-08), "baseline"

                if func_type in ["min", "max"]:
                    peak_im = np.max(inserted_function) if func_type != "min" else np.min(inserted_function)
                else:
                    peak_im = np.mean(inserted_function) if func_type == "mean" else np.median(inserted_function)
                assert data["amplitude"] == peak_im - data["baseline"], "amplitude"

    @pytest.mark.parametrize("vary_coefs", [False, "within_function"])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("region_name", REGIONS_TO_RUN)
    @pytest.mark.parametrize("func_type", ["min", "max", "mean", "median"])
    def test_qtable_min_max_mean_median(self, tgui, func_type, vary_coefs, analyse_specific_recs, region_name):
        """
        Test the results shown on the table for the min, max, mean, median measures
        """
        rec_from, rec_to = run_curve_fitting(tgui, vary_coefs, func_type, region_name, tgui.time_type, analyse_specific_recs)

        num_recs = rec_to - rec_from + 1
        data = utils.np_empty_nan((num_recs, tgui.adata.function_samples + 1))
        for idx, rec in enumerate(range(rec_from, rec_to + 1)):
            data[idx, :] = tgui.mw.loaded_file.data.vm_array[rec][tgui.adata.start_sample:tgui.adata.stop_sample + 1]

        if func_type == "min":
            assert utils.allclose(np.min(data, axis=1),
                                  tgui.get_data_from_qtable("min", rec_from, rec_to, "curve_fitting"),
                                  1e-08), "min"

        if func_type == "max":
            assert utils.allclose(np.max(data, axis=1),
                                  tgui.get_data_from_qtable("max", rec_from, rec_to, "curve_fitting"),
                                  1e-08), "max"

        if func_type == "mean":
            assert utils.allclose(np.mean(data, axis=1),
                                  tgui.get_data_from_qtable("mean", rec_from, rec_to, "curve_fitting"),
                                  1e-08), "mean"

        if func_type == "median":
            assert utils.allclose(np.median(data, axis=1),
                                  tgui.get_data_from_qtable("median", rec_from, rec_to, "curve_fitting"),
                                  1e-08), "median"

    @pytest.mark.parametrize("vary_coefs", [False, "within_function"])
    @pytest.mark.parametrize("analyse_specific_recs", [False, True])
    @pytest.mark.parametrize("func_type", ["min", "max", "mean", "median"])
    def test_all_recs_analysed_together_curve_fit_min_max_mean_median(self, tgui, func_type, vary_coefs,
                                                                      analyse_specific_recs):
        """
        see test_all_recs_analysed_together_curve_fit()
        """
        rec_from, rec_to = run_curve_fitting(tgui, vary_coefs, func_type, "all", tgui.time_type, analyse_specific_recs)

        num_recs = rec_to - rec_from + 1
        data = utils.np_empty_nan((num_recs, tgui.adata.function_samples + 1))
        for idx, rec in enumerate(range(rec_from, rec_to + 1)):
            data[idx, :] = tgui.mw.loaded_file.data.vm_array[rec][tgui.adata.start_sample:tgui.adata.stop_sample + 1]

        if func_type == "min":
            min_results = tgui.get_data_from_qtable("min", rec_from, rec_to, "curve_fitting",
                                                    return_regions=True)
            assert self.check_all_sublists_are_equal(min_results), "min all regions equal"
            assert utils.allclose(np.min(data, axis=1),
                                  min_results[0],
                                  1e-08), "min"

        elif func_type == "max":
            max_results = tgui.get_data_from_qtable("max", rec_from, rec_to, "curve_fitting",
                                                    return_regions=True)
            self.check_all_sublists_are_equal(max_results), "max all regions equal"
            assert utils.allclose(np.max(data, axis=1),
                                  max_results[0],
                                  1e-08), "b0"

        elif func_type == "mean":
            mean_results = tgui.get_data_from_qtable("mean", rec_from, rec_to, "curve_fitting",
                                                     return_regions=True)
            self.check_all_sublists_are_equal(mean_results), "mean all regions equal"
            assert utils.allclose(np.mean(data, axis=1),
                                  mean_results[0],
                                  1e-08), "mean"

        elif func_type == "median":
            median_results = tgui.get_data_from_qtable("median", rec_from, rec_to, "curve_fitting",
                                                       return_regions=True)
            self.check_all_sublists_are_equal(median_results), "median"
            assert utils.allclose(np.median(data, axis=1),
                                  median_results[0],
                                  1e-08), "median"

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Bounds Varying Across Recs
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("func_type", ["min", "max", "mean", "median", "monoexp"])
    @pytest.mark.parametrize("mode", ["dont_align_across_recs",
                                      "align_across_recs"])
    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_varying_boundaries_across_recs(self, tgui, mode, func_type, analyse_specific_recs):
        """
        Test curve fitting analysis when linking or unlinking recs across records. First insert ramping protocols with
        random amplitude to Im and Vm. Then set bounds to random position held in tgui.start_times / stop_times and
        compare the results from EE between the bounds with test analysis between the same bounds.

        func_type == monoexp only due to issues with not-exactly-replicable fitting procedures for
        other curve fitting types.
        """
        tgui.setup_artificial_data(tgui.time_type, analysis_type="curve_fitting_slope")
        tgui.set_link_across_recs(tgui, mode)

        rec_from, rec_to = run_curve_fitting(tgui, False, func_type, "all", tgui.time_type, analyse_specific_recs=analyse_specific_recs,
                                             slope_override=True)

        for region_name in REGIONS_TO_RUN:

            for rec in range(rec_from, rec_to + 1):

                set_rec = 0 if mode == "align_across_recs" else rec

                results_key = func_type if func_type in ["min", "max", "median", "mean"] else "fit"
                ee_data = tgui.mw.loaded_file.curve_fitting_results[region_name]["data"][rec][results_key]

                data_range = tgui.adata.vm_array[rec][tgui.adata.start_sample[set_rec]:
                                                      tgui.adata.stop_sample[
                                                          set_rec] + 1]

                if func_type == "min":
                    test_data = np.min(data_range)
                elif func_type == "max":
                    test_data = np.max(data_range)
                elif func_type == "mean":
                    test_data = np.mean(data_range)
                elif func_type == "median":
                    test_data = np.median(data_range)
                elif func_type == "monoexp":
                    time_range = tgui.adata.time_array[rec][tgui.adata.start_sample[set_rec]: tgui.adata.stop_sample[set_rec] + 1]
                    __, test_data, __ = core_analysis_methods.fit_curve(func_type, time_range, data_range, direction=-1)

                assert np.array_equal(ee_data, test_data), " ".join([func_type, region_name, str(rec)])

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Max Slope
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_max_slope_curve_fitting(self, tgui):
        """
        Test against events analysed in axograph. It is the 3rd, 5th and 7th events in the file, these were analysed with
        cursors in axograph.
        """
        axograph_first_event_rise_3, axograph_first_event_decay_3, \
            axograph_first_event_rise_5, axograph_first_event_decay_7 = tgui.get_axograph_max_slope_to_test_against()
        tgui.load_a_filetype("cell_5")

        tgui.set_analysis_type("curve_fitting")
        tgui.left_mouse_click(tgui.mw.mw.curve_fitting_show_dialog_button)
        tgui.switch_checkbox(tgui.mw.dialogs["curve_fitting"].dia.hide_baseline_radiobutton, on=True)
        tgui.mw.curve_fitting_regions["reg_1"].bounds["upper_exp_lr"].setRegion((0.2626, 0.2632))
        tgui.set_combobox(tgui.mw.dialogs["curve_fitting"].dia.fit_type_combobox, 4)

        # Rise sample 3 and 5 search region
        tgui.switch_checkbox(tgui.mw.dialogs["curve_fitting"].dia.max_slope_direction_neg_radiobutton, on=True)
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["curve_fitting"].dia.max_slope_num_samples, 3)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)
        assert np.round(float(tgui.mw.mw.table_tab_tablewidget.item(2, 1).data(0)), 3) == axograph_first_event_rise_3

        tgui.enter_number_into_spinbox(tgui.mw.dialogs["curve_fitting"].dia.max_slope_num_samples, 5)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)
        assert np.round(float(tgui.mw.mw.table_tab_tablewidget.item(2, 1).data(0)), 3) == axograph_first_event_rise_5

        # Decay sample 3 and 7 search regionq
        tgui.switch_checkbox(tgui.mw.dialogs["curve_fitting"].dia.max_slope_direction_pos_radiobutton, on=True)

        tgui.mw.curve_fitting_regions["reg_1"].bounds["upper_exp_lr"].setRegion((0.2632, 0.2700))
        tgui.enter_number_into_spinbox(tgui.mw.dialogs["curve_fitting"].dia.max_slope_num_samples, 3)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)
        assert np.round(float(tgui.mw.mw.table_tab_tablewidget.item(2, 1).data(0)), 4) == axograph_first_event_decay_3

        tgui.enter_number_into_spinbox(tgui.mw.dialogs["curve_fitting"].dia.max_slope_num_samples, 7)
        tgui.left_mouse_click(tgui.mw.dialogs["curve_fitting"].dia.fit_button)
        assert np.round(float(tgui.mw.mw.table_tab_tablewidget.item(2, 1).data(0)), 4) == axograph_first_event_decay_7

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Region Options
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_hide_baseline_and_hide_region(self, tgui, analyse_specific_recs):

        rec_from, rec_to = run_curve_fitting(tgui,
                                             vary_coefs=False,
                                             func_type="monoexp",
                                             region_name="all",
                                             norm_or_cumu_time=tgui.time_type,
                                             analyse_specific_recs=analyse_specific_recs,
                                             set_options_only=True)

        plot_items = tgui.mw.loaded_file_plot.upperplot.items
        curve_fitting_dialog = tgui.mw.dialogs["curve_fitting"]

        for rec in range(0, tgui.adata.num_recs):

            tgui.mw.update_displayed_rec(rec)

            if analyse_specific_recs:
                if rec in np.hstack([np.arange(0, rec_from),
                                     np.arange(rec_to + 1, tgui.adata.num_recs)]):  # excluded recs
                    for region_name in REGIONS_TO_RUN:

                        assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_exp_lr"] not in plot_items
                        assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_bl_lr"] not in plot_items
                    continue

            for region_name in REGIONS_TO_RUN:

                assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_exp_lr"] in plot_items
                assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_bl_lr"] in plot_items

            for region_name in REGIONS_TO_RUN:

                quick_switch_to_region(tgui, curve_fitting_dialog, int(region_name[-1]))

                tgui.switch_checkbox(curve_fitting_dialog.dia.hide_baseline_radiobutton, on=True)
                assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_bl_lr"] not in plot_items

            for region_name in REGIONS_TO_RUN:

                quick_switch_to_region(tgui, curve_fitting_dialog, int(region_name[-1]))
                tgui.switch_checkbox(curve_fitting_dialog.dia.hide_region_radiobutton, on=True)

                assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_exp_lr"] not in plot_items
                assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_bl_lr"] not in plot_items

            for region_name in REGIONS_TO_RUN:

                quick_switch_to_region(tgui, curve_fitting_dialog, int(region_name[-1]))
                tgui.switch_checkbox(curve_fitting_dialog.dia.hide_baseline_radiobutton, on=True)

                assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_exp_lr"] in plot_items
                assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_bl_lr"] not in plot_items

            for region_name in REGIONS_TO_RUN:

                quick_switch_to_region(tgui, curve_fitting_dialog, int(region_name[-1]))
                tgui.switch_checkbox(curve_fitting_dialog.dia.show_region_radiobutton, on=True)

                assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_exp_lr"] in plot_items
                assert tgui.mw.curve_fitting_regions[region_name].bounds["upper_bl_lr"] in plot_items

    def test_differet_analysis_per_region(self, tgui):
        """
        Final sanity check, run different types of analysis in each region.

        A monoexp function is plot for all recs. For biexp_event detection, it won't fit so just check the R2 is bad
        and the max matches peak.

        For biexp and triexp curves, these are fit to a monoexp function so one beta value will be modelled with the tau,
        and the remaining b2 are modelled with 0 amplitude so the tau can be ignored.
        """
        __, rec_from, rec_to = tgui.handle_analyse_specific_recs(False)
        tgui.update_curve_fitting_function("across_records",
                                           "monoexp",
                                           tgui.time_type)  # order important this will overwrite analyse specific recs

        for region_name, func_type in zip(["reg_1", "reg_2", "reg_3",   "reg_4",       "reg_5",       "reg_6"],
                                          ["max",   "mean",  "monoexp", "biexp_event", "biexp_decay", "triexp"]):

            setup_and_run_curve_fitting_analysis(tgui,
                                                 func_type=func_type,
                                                 region_name=region_name,
                                                 rec_from=0,
                                                 rec_to=tgui.adata.num_recs,
                                                 set_options_only=True)

        tgui.left_mouse_click(tgui.mw.mw.curve_fitting_fit_all_regions_button)

        for region_name, func_type in zip(["reg_1", "reg_2", "reg_3",   "reg_4",       "reg_5",       "reg_6"],
                                          ["max",   "mean",  "monoexp", "biexp_event", "biexp_decay", "triexp"]):

            for rec in range(tgui.adata.num_recs):

                if func_type in ["biexp_decay", "triexp"]:
                    self.quick_override_starting_estimate_for_testing_different_curves_across_records(tgui, func_type, rec)

                data = tgui.mw.loaded_file.curve_fitting_results[region_name]["data"][rec]

                if region_name == "reg_1":  # max
                    max_results = tgui.get_data_from_qtable("max", rec_from, rec_to, "curve_fitting", return_regions=True)[0]
                    assert utils.allclose(np.max(tgui.adata.vm_array[rec]), max_results[rec], 1e-08), "b0"

                if region_name == "reg_2":  # mean
                    mean_results = tgui.get_data_from_qtable("mean", rec_from, rec_to, "curve_fitting", return_regions=True)[0]
                    mean_of_data = np.mean(tgui.mw.loaded_file.data.vm_array[rec][tgui.adata.start_sample:tgui.adata.stop_sample + 1])
                    assert utils.allclose(mean_results[rec], mean_of_data, 1e-08), "mean"

                if region_name == "reg_3":  # monoexp
                    assert utils.allclose(data["tau"], tgui.adata.get_tau(func_type, "tau", rec_from, rec_to, rec), 1e-08), "tau"

                if region_name == "reg_4":  # biexp_ vent
                    event_time = list(data["event_info"].keys())[0]
                    peak_im = data["event_info"][event_time]["peak"]["im"]
                    peak_data = np.max(tgui.adata.vm_array[rec])
                    assert peak_im == peak_data
                    assert (data["r2"] < 1).all()  # cant fit a biexp to a monoexp

                if region_name == "reg_5":  # biexp decay
                    tau = data["tau2"] if data["b2"] > data["b1"] else data["tau1"]
                    assert utils.allclose(tau, tgui.adata.get_tau("monoexp", "tau1", rec_from, rec_to, rec), 1e-02), "tau1"

                if region_name == "reg_6":   # triexp
                    # Here we just checking triexp is analysed correctly in region 6 so this suffices
                    amplitude = np.max(tgui.adata.vm_array[rec]) - tgui.adata.resting_vm
                    b1_sums = np.sum([data["b1"], data["b2"], data["b3"]])
                    assert utils.allclose(amplitude, b1_sums, 1e-05)
