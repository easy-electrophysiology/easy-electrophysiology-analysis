import scipy.stats
from PySide2 import QtWidgets, QtCore, QtGui
from PySide2 import QtTest
from PySide2.QtTest import QTest
import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from ..easy_electrophysiology import easy_electrophysiology
from ephys_data_methods import core_analysis_methods
MainWindow = easy_electrophysiology.MainWindow
from utils import utils
keyClick = QTest.keyClick
from setup_test_suite import GuiTestSetup
os.environ["PYTEST_QT_API"] = "pyside2"

SPEED = "fast"

class TestInputResistanceGui:

    @pytest.fixture(scope="function", params=["normalised", "cumulative"], ids=["normalised", "cumulative"])
    def tgui(test, request):
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = SPEED
        tgui.setup_artificial_data(request.param, analysis_type="Ri")
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def quick_setup_and_run_ir(tgui, analyse_specific_recs):
        test_vm, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs,
                                                                      data=tgui.adata.current_injection_amplitude)
        tgui.switch_to_input_resistance_and_set_im_combobox()
        tgui.run_ri_analysis_bounds()

        return test_vm, rec_from, rec_to

    @staticmethod
    def get_b1_from_ir_plot_title(results_plot):
        b1 = results_plot.plot.titleLabel.text.split("b<sub>1</sub> ")[1]
        b1 = float(b1.split("</b>")[0])
        return b1

    def check_plot_matches_table(self, tgui, test_ir):
        tgui.left_mouse_click(tgui.mw.mw.ri_plot_fit_tablepanel_button)
        results_plot = tgui.mw.loaded_file.ir_results_plot
        b1 = self.get_b1_from_ir_plot_title(results_plot)
        results_plot.close()

        assert b1 == np.round(test_ir, 3)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_input_resistance_vm_with_bounds(self, tgui, analyse_specific_recs):
        """
        Test that Vm is measured / displayed correctly and stored correctly in model and stored tables
        """
        test_vm, rec_from, rec_to = self.quick_setup_and_run_ir(tgui, analyse_specific_recs)

        assert tgui.eq(tgui.mw.loaded_file.ir_data["vm_delta"],
                       test_vm)
        assert tgui.eq(tgui.mw.stored_tabledata.ir_data[0]["vm_delta"],
                       test_vm)
        assert tgui.eq(tgui.get_data_from_qtable("vm_delta", rec_from, rec_to, analysis_type="Ri"),
                       tgui.clean(test_vm))

    def test_input_resistance_vm_one_sample_case(self, tgui):
        """
        """
        __, __, _ = tgui.handle_analyse_specific_recs(True, rec_from=0, rec_to=0)
        tgui.switch_to_input_resistance_and_set_im_combobox()
        tgui.run_ri_analysis_bounds()

        vm_in_mv = tgui.mw.loaded_file.ir_data["vm_delta"][0]
        im_in_pa = tgui.mw.loaded_file.ir_data["im_delta"][0]

        R = vm_in_mv / (im_in_pa / 1000)

        assert R == tgui.mw.loaded_file.ir_data["input_resistance"][0]

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_input_resistance_im_with_bounds(self, tgui, analyse_specific_recs):
        """
        Test that Im is measured / displayed correctly and stored correctly in model and stored tables
        """
        test_vm, rec_from, rec_to = self.quick_setup_and_run_ir(tgui, analyse_specific_recs)

        assert tgui.eq(tgui.mw.loaded_file.ir_data["im_delta"],
                       test_vm)
        assert tgui.eq(tgui.mw.stored_tabledata.ir_data[0]["im_delta"],
                       test_vm)
        assert tgui.eq(tgui.get_data_from_qtable("im_delta", rec_from, rec_to, analysis_type="Ri"),
                       tgui.clean(test_vm))

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_input_resistance_counted_recs_with_bounds(self, tgui, analyse_specific_recs):
        """
        Test that records is measured / displayed correctly and stored correctly in model and stored tables
        """
        test_counted_recs = np.array([rec for rec in range(1, tgui.adata.num_recs + 1)])
        test_counted_recs, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs,
                                                                                data=test_counted_recs)
        tgui.switch_to_input_resistance_and_set_im_combobox()
        tgui.run_ri_analysis_bounds()

        assert tgui.eq(tgui.mw.loaded_file.ir_data["record_num"],
                       test_counted_recs)
        assert tgui.eq(tgui.mw.stored_tabledata.ir_data[0]["record_num"],
                       test_counted_recs)
        assert tgui.eq(tgui.get_data_from_qtable("record_num", rec_from, rec_to, analysis_type="Ri"),
                       tgui.clean(test_counted_recs))

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_input_resistance_with_bounds(self, tgui, analyse_specific_recs):
        """
        Test input resistance is calculated and displayed / stored correctly (calculated with boundaries)
        """
        test_vm, __, __ = self.quick_setup_and_run_ir(tgui, analyse_specific_recs)

        test_input_resistance = tgui.get_test_ir(test_vm)

        assert np.isclose(tgui.mw.loaded_file.ir_data["input_resistance"][0],
                          test_input_resistance, atol=1e-10, rtol=0)
        assert np.isclose(tgui.mw.stored_tabledata.ir_data[0]["input_resistance"][0],
                          test_input_resistance, atol=1e-10, rtol=0)
        assert np.isclose(tgui.get_data_from_qtable("input_resistance", 1, 1, analysis_type="Ri"),
                          test_input_resistance, atol=1e-10, rtol=0)

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_input_resistance_with_im_protocol(self, tgui, analyse_specific_recs):
        """
        Test input resistance is calculated and displayed / stored correctly (calculated with im protocol i.e. Start / Stop)
        """
        test_vm, __, __ = tgui.handle_analyse_specific_recs(analyse_specific_recs,
                                                            data=tgui.adata.current_injection_amplitude)

        tgui.run_input_resistance_im_protocol()

        test_input_resistance = tgui.get_test_ir(test_vm)

        assert np.isclose(tgui.mw.loaded_file.ir_data["input_resistance"][0],  # isclose because of the tiny delta function inserted for sag/hmp detection
                          test_input_resistance, 1e-2)
        assert np.isclose(tgui.mw.stored_tabledata.ir_data[0]["input_resistance"][0],
                          test_input_resistance, 1e-2)
        assert np.isclose(tgui.get_data_from_qtable("input_resistance", 1, 1, analysis_type="Ri"),
                          test_input_resistance, 1e-2)

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_input_resistance_with_user_im(self, tgui, analyse_specific_recs):
        """
        Test input resistance is calculated and displayed / stored correctly (calculated with user-input im)
        """
        test_vm, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs,
                                                                      data=tgui.adata.current_injection_amplitude)

        rows_to_fill_in = tgui.run_ri_analysis_user_input_im(rec_from, rec_to)

        test_im = np.linspace(0, rows_to_fill_in-1, rows_to_fill_in)
        test_input_resistance = tgui.get_test_ir(test_vm, user_test_im=test_im)

        try:
            assert np.isclose(tgui.mw.loaded_file.ir_data["input_resistance"][0],
                              test_input_resistance[0], 1e-2)
            assert np.isclose(tgui.mw.stored_tabledata.ir_data[0]["input_resistance"][0],
                              test_input_resistance[0], 1e-2)
            assert np.isclose(tgui.get_data_from_qtable("input_resistance", 1, 1, analysis_type="Ri"),
                              test_input_resistance[0], 1e-2)
        except:
            breakpoint()
            
    def test_user_im_input_is_valid(self, tgui):
        test_vm, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=False)

        rows_to_fill_in = rec_to - rec_from + 1

        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting="user_input_im")

        tgui.left_mouse_click(tgui.mw.mw.ir_set_im_button)
        tgui.fill_user_im_input_widget(rows_to_fill_in, tgui.mw.mw.ir_set_im_button, all_numbers_the_same=True)

        QtCore.QTimer.singleShot(1000, lambda: self.check_user_input_im_has_variation(tgui))
        tgui.left_mouse_click(tgui.mw.mw.ir_calc_button)

        tgui.left_mouse_click(tgui.mw.mw.ir_set_im_button)

        tgui.mw.dialogs["user_im_entry"].dia.step_table.clear()

        text = QtWidgets.QTableWidgetItem("Text Input")
        tgui.mw.dialogs["user_im_entry"].dia.step_table.setItem(0, 0, text)

        QtWidgets.QApplication.processEvents()

        QtCore.QTimer.singleShot(500, lambda: self.check_user_im_input_is_text(tgui))
        keyClick(tgui.mw.dialogs["user_im_entry"].dia.step_table, QtGui.Qt.Key_Enter)

    @staticmethod
    def check_user_im_input_is_text(tgui):
        assert "Check input is number" in tgui.mw.messagebox.text()
        tgui.mw.messagebox.close()

    @staticmethod
    def check_user_input_im_has_variation(tgui):
        assert "There is no variation in the user-input Im protocol" in tgui.mw.messagebox.text()
        tgui.mw.messagebox.close()

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_sag(self, tgui, analyse_specific_recs):
        """
        Test the sag is displayed / stored correctly (artificial trace contains a small delta function on top of the injected current)
        """
        tgui.switch_to_input_resistance_and_set_im_combobox()
        test_sag_hump, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs, data=tgui.adata.sag_hump_peaks[:, 0])
        tgui.run_ri_analysis_bounds(set_sag_analysis=True)

        assert tgui.eq(tgui.mw.loaded_file.ir_data["sag_hump_peaks"],
                       test_sag_hump)
        assert tgui.eq(tgui.mw.stored_tabledata.ir_data[0]["sag_hump_peaks"],
                       test_sag_hump)
        assert tgui.eq(tgui.get_data_from_qtable("sag_hump_peaks", rec_from, rec_to, analysis_type="Ri"), tgui.clean(test_sag_hump))

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_calculate_sag_ratio(self, tgui, analyse_specific_recs):
        """
        Test the sag ratio is displayed / stored correctly (sag ratio is sag divided by the measured current injection)
        """
        tgui.switch_to_input_resistance_and_set_im_combobox()

        test_sag_hump_ratio = tgui.adata.sag_hump_peaks[:, 0] / tgui.adata.peak_deflections
        test_sag_hump_ratio, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs, data=test_sag_hump_ratio)
        tgui.run_ri_analysis_bounds(set_sag_analysis=True)

        assert tgui.eq(tgui.mw.loaded_file.ir_data["sag_hump_ratio"], test_sag_hump_ratio)
        assert tgui.eq(tgui.mw.stored_tabledata.ir_data[0]["sag_hump_ratio"],
                       test_sag_hump_ratio)
        assert tgui.eq(tgui.get_data_from_qtable("sag_hump_ratio", rec_from, rec_to, analysis_type="Ri"),
                       tgui.clean(test_sag_hump_ratio))

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_plot(self, tgui, analyse_specific_recs):
        """
        Test the sag/hump circle is displayed on the plot correctly. Conduct analysis then scroll through every record checking
        plot display is correct.
        """
        tgui.switch_to_input_resistance_and_set_im_combobox()
        test_sag_hump, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs,
                                                                            data=tgui.adata.sag_hump_peaks[:, 0])
        tgui.run_ri_analysis_bounds(set_sag_analysis=True)

        for rec in range(rec_from, rec_to + 1):
            tgui.mw.update_displayed_rec(rec)
            plot = tgui.mw.loaded_file_plot.ir_plot
            x_data = plot.xData
            y_data = plot.yData
            assert x_data.astype(float) == tgui.adata.time_array[rec][tgui.adata.sag_hump_idx]
            sag_peak = tgui.adata.sag_hump_peaks[rec, 0] + tgui.adata.resting_vm + tgui.adata.current_injection_amplitude[rec]
            assert y_data == sag_peak

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_ir_results_plot(self, tgui, analyse_specific_recs):
        """
        Open up the Ri results plot and check it matches the results table.
        Click to delete a point, check it is deleted and the Ri is calculated properly in the label
        """
        for filenum in range(2):

            # Setup and Run Analysis
            tgui.switch_to_input_resistance_and_set_im_combobox()
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)
            tgui.mw.loaded_file.data.vm_array += np.random.randint(0, 100000, (tgui.adata.num_recs,
                                                                               tgui.adata.num_samples))  # completely randomise Vm so deleting a point makes a difference
            tgui.run_ri_analysis_bounds()

            tgui.left_mouse_click(tgui.mw.mw.ri_plot_fit_tablepanel_button)
            results_plot = tgui.mw.loaded_file.ir_results_plot

            # Check plot matches results
            assert np.array_equal(results_plot.trace1.xData * 1000,  # Im  plot in nA
                                  tgui.get_data_from_qtable("im_delta", rec_from, rec_to, analysis_type="Ri"))
            assert np.array_equal(results_plot.trace1.yData,
                                  tgui.get_data_from_qtable("vm_delta", rec_from, rec_to, analysis_type="Ri"))
            ir_before_deleteion = self.get_b1_from_ir_plot_title(results_plot)

            assert np.isclose(ir_before_deleteion,
                              tgui.get_data_from_qtable("input_resistance", rec_from, rec_to, analysis_type="Ri"), 1e-3)  # label is rounded to 3dp

            # Delete a point and confirm deletion of x and y coords
            point_idx_to_del = np.random.randint(0, len(results_plot.x1) - 1)
            point_to_del = [results_plot.trace1.scatter.points()[point_idx_to_del]]
            results_plot.handle_click_on_plot(point_to_del, background_click=False)
            QtWidgets.QApplication.processEvents()
            results_plot.handle_click_on_plot(point_to_del, background_click=False)

            assert point_to_del[0].pos()[0] not in results_plot.trace1.xData
            assert point_to_del[0].pos()[1] not in results_plot.trace1.yData

            # test new Ri is calculated properly
            test_ir, __, __, __, __ = scipy.stats.linregress(results_plot.trace1.xData, results_plot.trace1.yData)
            ir_after_deletion = self.get_b1_from_ir_plot_title(results_plot)

            assert np.isclose(ir_after_deletion, test_ir, 1e-3)
            assert ir_before_deleteion != ir_after_deletion

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data(tgui.time_type, "spkcnt")
        tgui.shutdown()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Rounding Im
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    def test_multiple_files_with_and_without_rounding(self, tgui, analyse_specific_recs):
        """
        Check that Im rounding works well and Ri analysis is updated (even in batch mode) when the Im is changed
        """
        num_to_round = 15

        # First load a file, calculate and check Ri, round Im and check Ri is updated, and that plot matches table
        test_vm, rec_from, rec_to = self.quick_setup_and_run_ir(tgui, analyse_specific_recs)

        file_1_unrounded_ir = tgui.get_test_ir(test_vm)

        assert utils.allclose(tgui.mw.loaded_file.ir_data["input_resistance"][0],
                              file_1_unrounded_ir, 1e-10)

        self.set_round_im(tgui, "round", num_to_round)

        file_1_rounded_ir = tgui.get_test_ir(test_vm, round_im=num_to_round)

        assert utils.allclose(tgui.mw.loaded_file.ir_data["input_resistance"][0],
                              file_1_rounded_ir, 1e-10)

        self.check_plot_matches_table(tgui, file_1_rounded_ir)

        # Now load a new file, calculate Ri (rounding is already set) and Ri calculation
        # is correct and table matches plot
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        tgui.setup_artificial_data(tgui.time_type, analysis_type="Ri")

        test_vm, rec_from, rec_to = self.quick_setup_and_run_ir(tgui, analyse_specific_recs)

        self.set_round_im(tgui, "round", num_to_round)

        file_2_rounded_ir = tgui.get_test_ir(test_vm, round_im=num_to_round)
        assert utils.allclose(tgui.mw.loaded_file.ir_data["input_resistance"][0],
                              file_2_rounded_ir, 1e-10)

        self.check_plot_matches_table(tgui, file_2_rounded_ir)

        # Now unround, check the calculation is updated for both files
        self.set_round_im(tgui, "unround", num_to_round)
        assert utils.allclose(file_1_unrounded_ir,
                              float(tgui.mw.mw.table_tab_tablewidget.item(2, 7).data(0)), 1e-10)

        file_2_unrounded_ir = tgui.get_test_ir(test_vm)
        assert utils.allclose(file_2_unrounded_ir,
                              float(tgui.mw.mw.table_tab_tablewidget.item(2, 15).data(0)), 1e-10)

        # round again, check the calculation is updated for both files
        self.set_round_im(tgui, "round", num_to_round)

        assert utils.allclose(file_1_rounded_ir,
                              float(tgui.mw.mw.table_tab_tablewidget.item(2, 7).data(0)), 1e-10)

        assert utils.allclose(file_2_rounded_ir,
                              float(tgui.mw.mw.table_tab_tablewidget.item(2, 15).data(0)), 1e-10)

        tgui.shutdown()

    @staticmethod
    def set_round_im(tgui, round_or_unround, num_to_round):

        if round_or_unround == "unround":
            tgui.set_combobox(tgui.mw.mw.ir_im_opts_combobox, idx=0)

        elif round_or_unround == "round":
            tgui.set_combobox(tgui.mw.mw.ir_im_opts_combobox, idx=1)
            QtWidgets.QApplication.processEvents()
            tgui.enter_number_into_spinbox(tgui.mw.dialogs["user_im_round"].dia.im_round_input, num_to_round, setValue=True)
            keyClick(tgui.mw.dialogs["user_im_round"], QtGui.Qt.Key_Enter)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Input Resistance Bounds Across Recs
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("analyse_specific_recs", [True, False])
    @pytest.mark.parametrize("mode", ["dont_align_across_recs"])
    def test_ir_bounds_not_linked_across_recs(self, tgui, analyse_specific_recs, mode):
        """
        Large and quickly written function to test new boundary recs aligned .vs not aligned across recs. TODO: refactor

        first set the bounds option to Im Vm link always off as we want to test separately. Test both when bounds are linked across
        recs and when not. Here we test every single Ri bound (upper and lowerplot, baselines and exp).

        Go through two files with batch mode ON just to check this doesn't mess anything up
        first randomly assign a boundary and move the bound across all recs and get all_start_stop_times which
        holds the position of every bound (upper and lower)

        Next convert these times to indicies and index out the test data. The indexed out data should match the data
        measured between the bounds in ee.

        Test they match for loaded_file, stored_tabledata and the displayed table. s
        """
        tgui.set_analysis_type("Ri")
        tgui.mw.mw.actionLink_im_vm_off.trigger()
        tgui.set_link_across_recs(tgui, mode)

        for filenum in range(2):

            tgui.set_recs_to_analyse_spinboxes_checked(on=False)  # we need to turn this off for assign_random_boundary_position_for_every_rec_and_test() to work

            # Get and set every region randomly across all recs. Get the bounds first otherwise
            # the boundary region will not exist but will try to be moved and create problems in the test environment
            all_start_stop_times = tgui.assign_random_boundary_position_for_every_rec_and_test(tgui,
                                                                                               tgui.mw.ir_bounds,
                                                                                               mode)
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs)
            tgui.left_mouse_click(tgui.mw.mw.ir_calc_button)

            # Convert all boundary times to sample index
            all_start_stop_times = tgui.convert_random_boundary_positions_from_time_to_samples(tgui,
                                                                                               all_start_stop_times)

            # Convert the times to indicies and index out the relevant data. Calculate additional params
            # from this indexed data (deta im / vm and input resistance)
            test_results, test_delta_im_pa, test_delta_vm_mv, test_ir = tgui.calculate_test_measures_from_boundary_start_stop_indicies(all_start_stop_times,
                                                                                                                                       ["upper_bl_lr", "upper_exp_lr", "lower_bl_lr", "lower_exp_lr"],
                                                                                                                                       ["vm_baseline", "vm_steady_state", "im_baseline", "im_steady_state"],
                                                                                                                                       rec_from, rec_to)
            # Test they match for Loaded File, Stored Tabledata and the analysis results table
            for test_dataset in [tgui.mw.loaded_file.ir_data, tgui.mw.stored_tabledata.ir_data[filenum]]:
                assert np.array_equal(test_dataset["vm_baseline"], test_results["vm_baseline"], equal_nan=True)
                assert np.array_equal(test_dataset["vm_steady_state"], test_results["vm_steady_state"], equal_nan=True)
                assert np.array_equal(test_dataset["im_baseline"], test_results["im_baseline"], equal_nan=True)
                assert np.array_equal(test_dataset["im_steady_state"], test_results["im_steady_state"], equal_nan=True)
                assert np.array_equal(test_dataset["vm_delta"], test_delta_vm_mv, equal_nan=True)
                assert np.array_equal(test_dataset["im_delta"], test_delta_im_pa, equal_nan=True)
                try:
                    assert np.isclose(test_dataset["input_resistance"][0], test_ir.slope, atol=1e-10, rtol=0)
                except:
                    breakpoint()
            # this always seems to get the last file on the table so no need to account for anything
            assert np.array_equal(tgui.get_data_from_qtable("vm_baseline", rec_from, rec_to, analysis_type="Ri"),
                                  test_results["vm_baseline"][rec_from:rec_to + 1])
            assert np.array_equal(tgui.get_data_from_qtable("vm_steady_state", rec_from, rec_to, analysis_type="Ri"),
                                  test_results["vm_steady_state"][rec_from:rec_to + 1])
            assert np.array_equal(tgui.get_data_from_qtable("im_baseline", rec_from, rec_to, analysis_type="Ri"),
                                  test_results["im_baseline"][rec_from:rec_to + 1])
            assert np.array_equal(tgui.get_data_from_qtable("im_steady_state", rec_from, rec_to, analysis_type="Ri"),
                                  test_results["im_steady_state"][rec_from:rec_to + 1])
            assert np.array_equal(tgui.get_data_from_qtable("vm_delta", rec_from, rec_to, analysis_type="Ri"),
                                  test_delta_vm_mv[rec_from:rec_to + 1])
            assert np.array_equal(tgui.get_data_from_qtable("im_delta", rec_from, rec_to, analysis_type="Ri"),
                                  test_delta_im_pa[rec_from:rec_to + 1])
            assert np.isclose(tgui.get_data_from_qtable("input_resistance", 1, 1, analysis_type="Ri"),
                              test_ir.slope,
                              atol=1e-10, rtol=0)

            # load a new file
            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.setup_artificial_data(tgui.time_type, "spkcnt")
        tgui.shutdown()
