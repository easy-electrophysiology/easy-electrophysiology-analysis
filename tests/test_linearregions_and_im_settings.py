import copy

from PySide2 import QtWidgets, QtCore, QtGui
from PySide2 import QtTest
from PySide2.QtTest import QTest
import pytest
import sys
import os
import numpy as np
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), 'easy_electrophysiology'))
from ..easy_electrophysiology import easy_electrophysiology
MainWindow = easy_electrophysiology.MainWindow
from setup_test_suite import GuiTestSetup

SPEED = "slow"

class TestLinearRegionsImSettings:
    """
    Test the linearregion boundaries (e.g. 'experimental' and 'baseline') for spikecounts ('spkcnt'), input resistance and spike kinetics (skinetics).
    Because (as far as I know) dialog presence cannot be easily checked with QTest, to check an analysis is not run (e.g. if the settings are not
    correct) an analysis is attempted and if there is no data in the model it is assumed the analysis did not run.

    METHODS: local methods used for testing im boundaries.

    TESTS:
            1) Test Correct LinearRegions are shown: Cycle through spkcnt, Ri and skinetics and set all available Im / bound options. Check that
               the correct boundaries are displayed in the GUI.

            2) Test Boundaries and Recursion Error: If the 'link_im_vm' option is turned on but only one set of boundaries is shown (e.g. input resistance
               upperplot but not input resistance lowerplot, because Im is set to user input) a recursion error will occur, after the displayed_rec
               is update. This checks all boundaries with link_im_on/off to ensure to such error occurs.

            3) Test all Im Widgets: This will run through all Im widgets and fill them out correctly, or incorrectly. It will try to run an analysis
               with correct / incorrect results and check that analysis is / isn't run.

            4) Test Error Caught When Boundary is Zero: In theory the gap between two linearitems of a linearregion can be zero
               (e.g. bounds.getRegion() = [0, 0]. Note this is not the case for baseline linearregions (e.g. upper_bl_lr) as these are forced
               to not be zero in linear_bounds class to avoid any errors (TODO: this isn't the neatest arrangement).
               If the experimental linearregions (e.g. upper_exp_lr) gap is zero and is not caught it will not cause an error,
               but no result will be displayed. These tests set every exp linearregion gap to [0, 0] then try to run an analysis.
               If analysis is not run, test passed.


            5) Test Im baselines are correct: Checks the region averaged within a baseline linearregion is correct. Checks the average and
               position of the baseline axline. Also, this method is then used to check all linearregion items position cfg match the
               positions saved in the config file.

            6) Test Boundaries are Shown Within Specific Records: Boundaries should only be displayed on records to be analysed (specified
               through the 'analyise_specific_recs' widgets/configs. This checks through all analysis boundaries and ensured bounds are only
               displayed on records to be analysed.

    NOTES: Testing the GUI with pytest can cause some strange issues that I have not figured out how to solve yet:
           - In some instances if there is an error in MainWindow the test will hang. This is frustrating to track down, the best solution
             at present is to set a breakpoint in the test that it is hanging on and run line by line within pbd to see the assertion.
           - If the QTimer singleshot times are too long, sometimes this can lead to strange behaviour - keep these short.
    """

    @pytest.fixture(scope="function", params=[True, False], ids=["normalised", "cumulative"])
    def tgui(test, request):
        """
        The tested file must have two channels
        """
        if request.param:
            tgui = GuiTestSetup('wcp')
            tgui.setup_mainwindow(show=True)
            tgui.test_update_fileinfo()
            tgui.test_load_norm_time_file()
        else:
            tgui = GuiTestSetup('cc_two_channel_abf')
            tgui.setup_mainwindow(show=True)
            tgui.test_update_fileinfo(norm=False)
            tgui.test_load_cumu_time_file()
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Methods
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# Check Bounds Methods--------------------------------------------------------------------------------------------------------------------------------

    def check_bounds_link_on_and_off(self, tgui, bounds_obj, region_type, bounds):
        """
        Move bounds all the way to the left / right of the axis and change record - with link im off this will cause no error
        but can cause recursion error if link on and proper conditionals not set in linear_region class.
        """
        steps_num = 20
        tgui.mw.mw.actionLink_im_vm_off.trigger()
        self.check_bounds(tgui, bounds_obj, region_type, bounds)
        tgui.repeat_mouse_click(tgui.mw.mw.current_rec_leftbutton, steps_num, 0.00005)
        tgui.repeat_mouse_click(tgui.mw.mw.current_rec_rightbutton, steps_num, 0.00005)
        tgui.mw.mw.actionLink_im_vm_on.trigger()
        self.check_bounds(tgui, bounds_obj, region_type, bounds)
        tgui.repeat_mouse_click(tgui.mw.mw.current_rec_leftbutton, steps_num, 0.00005)

    def check_bounds(self, tgui, bounds_obj, analysis_type, region_types):
        """
        Move bounds all te way left then move bounds all the way right
        Bounds types e.g. '['upper_exp_lr', 'lower_bl_lr', 'lower_exp_lr']'
        """
        for region_type in region_types:
            x_axis = tgui.mw.loaded_file.data.min_max_time[tgui.mw.cfgs.main['displayed_rec']]

            assert self.check_linearregion_bounds_right(bounds_obj, region_type, x_axis), \
                analysis_type + ' leftbound failed ' + region_type
            assert self.check_linearregion_bounds_left(bounds_obj, region_type, x_axis), \
                analysis_type + ' rightbound failed ' + region_type

    @staticmethod
    def check_linearregion_bounds_right(bounds_obj, region_type, x_axis):
        """
        Set bound to the leftmost extreme value. this should make the bounds to the axis limit.
        The start bound will by 4*ts+eps min difference from stop bound as specified in
        linear_regions.check_and_remendy_zero_difference_in_bounds()
        """
        tmp_bounds = bounds_obj.bounds[region_type].getRegion()  # save default boundary position so boundaries are left in default after test finished
        bounds_obj.bounds[region_type].setRegion((x_axis[1] * 2, x_axis[1] * 2))
        QtWidgets.QApplication.processEvents()

        min_ts = 2  # see linear region _check_and_remedy_zero_difference_in_bounds()
        offset = bounds_obj.loaded_file.data.ts * (min_ts + bounds_obj.loaded_file.data.ts) + np.finfo(float).eps
        min_difference = bounds_obj.loaded_file.data.ts * min_ts + np.finfo(float).eps
        result = bounds_obj.bounds[region_type].getRegion() == (x_axis[1] - min_difference - offset, x_axis[1])
        bounds_obj.bounds[region_type].setRegion(tmp_bounds)
        return result

    @staticmethod
    def check_linearregion_bounds_left(bounds_obj, region_type, x_axis):
        tmp_bounds = bounds_obj.bounds[region_type].getRegion()
        bounds_obj.bounds[region_type].setRegion((x_axis[0] - x_axis[1],  x_axis[0] - x_axis[1]))
        QtWidgets.QApplication.processEvents()

        min_ts = 2  # see linear region _check_and_remedy_zero_difference_in_bounds()
        min_difference = bounds_obj.loaded_file.data.ts * min_ts + np.finfo(float).eps
        offset = bounds_obj.loaded_file.data.ts * (min_ts + bounds_obj.loaded_file.data.ts) + np.finfo(float).eps
        result = bounds_obj.bounds[region_type].getRegion() == (x_axis[0], x_axis[0] + min_difference + offset)

        bounds_obj.bounds[region_type].setRegion(tmp_bounds)
        return result

    @staticmethod
    def check_lr_baseline(tgui, bounds_obj, region_type, vm_or_im_array, analysis_cfg):
        """
        Take the average between a bounds object (designed for a bl_lr but can also test exp_lr and will not return axline).
        Set the boundary to re-assigned start and stop time (in tgui class). Take the true mean within this region and test this against
        the start/stop samples saved in configs (get_test_idx_from_time_bounds) and the average as dispalayed on the axline.
        """
        rec = tgui.mw.cfgs.main['displayed_rec']
        bounds_obj.bounds[region_type].setRegion((tgui.test_bl_lr_start_time,
                                                  tgui.test_bl_lr_stop_time))  # make the assumption any tested rec will have time period at least 50 ms long
        # get bounds idx
        [start_sample, stop_sample] = tgui.get_test_idx_from_time_bounds(analysis_cfg,
                                                                         region_type + '_lowerbound',
                                                                         region_type + '_upperbound',
                                                                         rec)

        average_bounds = np.mean(vm_or_im_array[rec][start_sample:stop_sample + 1])
        average_axline = bounds_obj.axlines[region_type].value() if '_bl_' in region_type else None
        test_average = np.mean(vm_or_im_array[rec][tgui.test_bl_lr_start_idx:
                                                   tgui.test_bl_lr_stop_idx + 1])

        return average_bounds, average_axline, test_average

# Check Widgets --------------------------------------------------------------------------------------------------------------------------------------

    def check_im_protocol(self, tgui, analysis, correctly_filled):
        """
        Load the im_protocol widget in which the user inputs their own im protocol. If correctly_filled=False this is filled out incorrectly
        (only with start time) and cancel (this is kind of pointless) but then try to run analysis. It should not run.
        If correctly_filled=True then the widget is filled out correctly and the analysis will run.
        """
        set_im_button, run_analysis_button = tgui.get_analysis_im_and_run_buttons(analysis)
        QtWidgets.QApplication.processEvents()
        tgui.mw.loaded_file.init_analysis_results_tables()
        if correctly_filled:
            self.check_im_protocol_widgets_no_error(tgui,
                                                    set_im_button,
                                                    run_analysis_button)
            loaded_file_analysis_df = tgui.get_analysis_dataframe(analysis)
            result = ~np.all(np.isnan(loaded_file_analysis_df['record_num']))
        else:
            self.check_im_protocol_widgets_error(tgui,
                                                 set_im_button,
                                                 run_analysis_button)
            loaded_file_analysis_df = tgui.get_analysis_dataframe(analysis)
            result = np.all(np.isnan(loaded_file_analysis_df['record_num']))
        return result

    def check_user_im_input(self, tgui, num_recs, analysis, correctly_filled, within_recs=False):
        """
        Similar to check_im_protocol with the user_im_input table. If correctly_filled, table is filled with random numbers as Im
        for the correct number of recs. If not correctly_filled, the wrong number of recs will be filled.
        """
        set_im_button, run_analysis_button = tgui.get_analysis_im_and_run_buttons(analysis)
        QtWidgets.QApplication.processEvents()
        QTest.qWait(1000)

        tgui.mw.loaded_file.init_analysis_results_tables()
        self.check_user_im_input_widgets(tgui, num_recs,
                                         set_im_button,
                                         run_analysis_button,
                                         correctly_filled,
                                         within_recs)
        if correctly_filled:
            loaded_file_analysis_df = tgui.get_analysis_dataframe(analysis)
            result = ~np.all(np.isnan(loaded_file_analysis_df['record_num']))
        else:
            loaded_file_analysis_df = tgui.get_analysis_dataframe(analysis)
            result = np.all(np.isnan(loaded_file_analysis_df['record_num']))
        return result

    @staticmethod
    def check_im_protocol_widgets_error(tgui, analysis_set_im_button, analysis_calc_button):
        """
        Check im protocol. Because this is an excc need to use keyboard to fill it. Can fill out correctly and run analysis,
        or fill out incorrectly in which case analysis should not be run.
        """
        QtCore.QTimer.singleShot(500, lambda: tgui.mw.dialogs['im_inj_protocol'].dia.im_injprot_start_spinbox.setValue(0.1))
        QtCore.QTimer.singleShot(1000, lambda: tgui.left_mouse_click(tgui.mw.dialogs['im_inj_protocol'].dia.im_injprot_buttonbox.button(QtGui.QDialogButtonBox.Cancel)))  
        tgui.left_mouse_click(analysis_set_im_button)
        QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.left_mouse_click(analysis_calc_button)

    @staticmethod
    def check_im_protocol_widgets_no_error(tgui, analysis_set_im_button, analysis_calc_button):
        tgui.fill_im_injection_protocol_dialog(analysis_set_im_button, '0.1', '0.8')
        tgui.left_mouse_click(analysis_calc_button)

    def check_user_im_input_widgets(self, tgui, num_recs, analysis_set_im_button, analysis_calc_button, correctly_filled, within_recs):
        """
        Fill user im tablewidget with random numbers to test if it works or not. If correctly_filled=False the
        number of records filled will be one less the actual number which should raise an error.
        """
        rows_to_fill_in = num_recs if correctly_filled else num_recs - 1
        if within_recs:
            QtCore.QTimer.singleShot(1500, lambda: tgui.mw.messagebox.close())
        tgui.fill_user_im_input_widget(rows_to_fill_in, analysis_set_im_button)

        if not correctly_filled:
            QtCore.QTimer.singleShot(1500, lambda: self.check_messagebox_num_recs_error(tgui.mw.messagebox))
        tgui.left_mouse_click(analysis_calc_button)

    def check_messagebox_num_recs_error(self, messagebox):
        assert "Length of input Im protocol must be the same as number of records that will be analyzed" in messagebox.text()
        messagebox.close()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Correct LinearRegions are shown
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_spikecount_bounds_none(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=False)
        assert tgui.mw.spkcnt_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items, 'Spikecount no bounds - upper_exp_lr'
        assert tgui.mw.spkcnt_bounds.bounds['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items, 'Spikecount no bounds - upper_bl_lr'
        assert tgui.mw.spkcnt_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items, 'Spikecount no bounds - lower_exp_lr'
        assert tgui.mw.spkcnt_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items, 'Spikecount no bounds - lower_bl_lr'
        assert tgui.mw.spkcnt_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items, 'Spikecount no bounds - axlines lower_bl_lr'

    def test_spikecount_bounds_upper(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=False)
        assert tgui.mw.spkcnt_bounds.bounds['upper_exp_lr'] in tgui.mw.loaded_file_plot.upperplot.items,  'Spikecount bounds upper - upper_exp_lr'
        assert tgui.mw.spkcnt_bounds.bounds['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items,  'Spikecount bounds upper - upper_bl_lr'
        assert tgui.mw.spkcnt_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items,  'Spikecount bounds upper - lower_bl_lr'
        assert tgui.mw.spkcnt_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items,   'Spikecount bounds upper - lower_exp_lr'
        assert tgui.mw.spkcnt_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items, 'Spikecount bounds upper - axline lower_bl_lr'

    def test_spikecount_bounds_all(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=True)
        assert tgui.mw.spkcnt_bounds.bounds['upper_exp_lr'] in tgui.mw.loaded_file_plot.upperplot.items,  'Spikecount bounds all - upper_exp_lr'
        assert tgui.mw.spkcnt_bounds.bounds['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items,  'Spikecount bounds all - upper_bl_lr'
        assert tgui.mw.spkcnt_bounds.bounds['lower_bl_lr'] in tgui.mw.loaded_file_plot.lowerplot.items,  'Spikecount bounds all - lower_bl_lr'
        assert tgui.mw.spkcnt_bounds.bounds['lower_exp_lr'] in tgui.mw.loaded_file_plot.lowerplot.items,  'Spikecount bounds all - lower_exp_lr'
        assert tgui.mw.spkcnt_bounds.axlines['lower_bl_lr'] in tgui.mw.loaded_file_plot.lowerplot.items, 'Spikecount im protocol - lower bl axline'

    def test_spikecount_bounds_im_protocol(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=True,
                                                       im_setting='im_protocol')
        assert tgui.mw.spkcnt_bounds.bounds['upper_exp_lr'] in tgui.mw.loaded_file_plot.upperplot.items,  'Spikecount im protocol - upper_exp_lr'
        assert tgui.mw.spkcnt_bounds.bounds['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items,  'Spikecount im protocol - upper_bl_lr'
        assert tgui.mw.spkcnt_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items,  'Spikecount im protocol - lower_exp_lr'
        assert tgui.mw.spkcnt_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items,  'Spikecount im protocol - lower_bl_lr'
        assert tgui.mw.spkcnt_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items, 'Spikecount im protocol - lower bl axline'

    def test_spikecount_bounds_user_im(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True,
                                                       im_setting='user_input_im')
        assert tgui.mw.spkcnt_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.spkcnt_bounds.bounds['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.spkcnt_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.spkcnt_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.spkcnt_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items

    def test_input_resistance_bounds_bounds(self, tgui):
        tgui.set_analysis_type('Ri')
        assert tgui.mw.ir_bounds.bounds['upper_exp_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['upper_bl_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.axlines['upper_bl_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['lower_bl_lr'] in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.bounds['lower_exp_lr'] in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.axlines['lower_bl_lr'] in tgui.mw.loaded_file_plot.lowerplot.items

    def test_input_resistance_bounds_im_protocol(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting='im_protocol')
        assert tgui.mw.ir_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.axlines['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items

    def test_input_resistance_bounds_user_im(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting='user_input_im')
        assert tgui.mw.ir_bounds.bounds['upper_exp_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['upper_bl_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.axlines['upper_bl_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items

    def test_skinetics_bounds_set(self, tgui):
        tgui.switch_to_skinetics_and_set_bound(skinetics_bounds_on=True)
        assert tgui.mw.skinetics_bounds.bounds['upper_exp_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.skinetics_bounds.bounds['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.skinetics_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.skinetics_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test all Im widgets
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_spikecount_im_protocol_error_catch(self, tgui):
        """
        Check im protocol widgets working - fill incorrectly and check this is caught
        """
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=True,
                                                       im_setting='im_protocol')
        assert self.check_im_protocol(tgui, 'spkcnt', correctly_filled=False), \
            "Test Spikecount im protocol analysis run - filled incorrectly"

    def test_spikecount_im_protocol_no_error(self, tgui):
        """
        Check im protocol widgets working - fill correctly and check analysis proceeds without error
        """
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=True,
                                                       im_setting='im_protocol')
        assert self.check_im_protocol(tgui, 'spkcnt', correctly_filled=True), \
            "Test Spikecount im protocol analysis run - filled correctly"

    def test_spikecount_user_im_widgets_error_catch(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True,
                                                       im_setting='user_input_im')
        assert self.check_user_im_input(tgui, tgui.num_recs,
                                        'spkcnt', correctly_filled=False),\
            "Test Spikecount user input analysis run - filled incorrectly"

    def test_spikecount_user_im_widgets_no_error(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True,
                                                       im_setting='user_input_im')

        assert self.check_user_im_input(tgui, tgui.num_recs,
                                        'spkcnt', correctly_filled=True), \
            "Test Spikecount user input analysis run - filled correctly"

    def test_input_resistance_im_protocol_error_catch(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting='im_protocol')
        assert self.check_im_protocol(tgui, 'Ri', correctly_filled=False), 'test'

    def test_input_resistance_im_protocol_no_error(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting='im_protocol')
        assert self.check_im_protocol(tgui, 'Ri', correctly_filled=True), \
            "Test Input Resistance user input analysis run - filled correctly"

    def test_input_resistance_user_im_error(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting='user_input_im')
        assert self.check_user_im_input(tgui, tgui.num_recs,
                                        'Ri', correctly_filled=False), 'test'

    def test_spkcnt_resistance_user_im_no_error(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting='user_input_im')
        assert self.check_user_im_input(tgui, tgui.num_recs,
                                        'Ri', correctly_filled=True), "TEST "

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Boundaries and Recursion Error
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_spikecount_bound_limits(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=True)
        self.check_bounds_link_on_and_off(tgui, tgui.mw.spkcnt_bounds, ' Spikecount', ['upper_exp_lr', 'lower_bl_lr', 'lower_exp_lr'])

    def test_input_resistance_bound_limits(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox()
        self.check_bounds_link_on_and_off(tgui, tgui.mw.ir_bounds, ' Ri', ['upper_bl_lr', 'upper_exp_lr', 'lower_bl_lr', 'lower_exp_lr'])

    def test_skinetics_bound_limits(self, tgui):
        tgui.switch_to_skinetics_and_set_bound(skinetics_bounds_on=True)
        self.check_bounds_link_on_and_off(tgui, tgui.mw.skinetics_bounds, ' skinetics', ['upper_exp_lr'])

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Boundaries linked across record and combination of link across records and across Im / Vm
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def check_bounds_dict_has_correct_number_of_recs(self, analysis_cfg, num_recs):
        for key in ["upper_bl_lr_lowerbound", "upper_bl_lr_upperbound", "upper_exp_lr_lowerbound", "upper_exp_lr_upperbound",
                    "lower_bl_lr_lowerbound", "lower_bl_lr_upperbound", "lower_exp_lr_lowerbound", "lower_exp_lr_upperbound"]:
            if key in analysis_cfg:
                if analysis_cfg[key] is not None:
                    if len(analysis_cfg[key]) != num_recs:
                        return False
        return True

    def check_boundary_cfgs_length_matches_num_recs(self, tgui, num_recs):
        """
        """
        for analysis_cfg in [tgui.mw.cfgs.spkcnt, tgui.mw.cfgs.skinetics, tgui.mw.cfgs.ir]:
            assert self.check_bounds_dict_has_correct_number_of_recs(analysis_cfg,
                                                                     num_recs), analysis_cfg["name"] + "analysis config boudary recs wrong"

        for region_name, region in tgui.mw.curve_fitting_regions.items():
            analysis_cfg = tgui.mw.cfgs.curve_fitting_region_position_configs[region_name]
            assert self.check_bounds_dict_has_correct_number_of_recs(analysis_cfg,
                                                                     num_recs), analysis_cfg["name"] + "analysis config boudary recs wrong"

        tgui.mw.mw.actionRemove_Baseline.trigger()
        analysis_cfgs = tgui.mw.dialogs["remove_baseline"].bounds_cfgs
        assert self.check_bounds_dict_has_correct_number_of_recs(analysis_cfgs, num_recs), analysis_cfg["name"] + "analysis config boudary recs wrong"
        tgui.mw.dialogs["remove_baseline"].close()

    # Test all boundaries are updated correctly on new file load
    def test_bondary_configs_are_updated_on_new_file_load(self, tgui):
        """
        tgui loads a file that is 15 recs, then current clamp is 7 recs, voltage_clamp is 1 recs.
        This then loads a number of different files smaller and bigger and checks the num recs of all bounds
        dicts are correct.
        """
        num_recs = tgui.mw.loaded_file.data.num_recs
        tgui.mw.set_curve_fitting_linear_regions_on_new_file()
        self.check_boundary_cfgs_length_matches_num_recs(tgui,
                                                         num_recs)

        tgui.load_a_filetype(filetype="current_clamp")
        num_recs = tgui.mw.loaded_file.data.num_recs
        self.check_boundary_cfgs_length_matches_num_recs(tgui,
                                                         num_recs)

        tgui.load_a_filetype(filetype="voltage_clamp_1_record")
        num_recs = tgui.mw.loaded_file.data.num_recs
        self.check_boundary_cfgs_length_matches_num_recs(tgui, num_recs)

        tgui.load_a_filetype(filetype="current_clamp")
        num_recs = tgui.mw.loaded_file.data.num_recs
        self.check_boundary_cfgs_length_matches_num_recs(tgui,
                                                         num_recs)

    def test_all_bounds_are_linked_to_correct_configs(self, tgui):
        """
        Double check the bounds objects are connected to the correct analysis configs.
        """
        for bounds, cfg_name in zip([tgui.mw.spkcnt_bounds, tgui.mw.skinetics_bounds, tgui.mw.ir_bounds],
                                    ["spkcnt", "skinetics", "Ri"]):
            assert bounds.analysis_cfg["name"] == cfg_name

        for region_name, region in tgui.mw.curve_fitting_regions.items():
            mem_address_cfgs = hex(id(tgui.mw.cfgs.curve_fitting_region_position_configs[region_name]))
            mem_address_bounds_cfg = hex(id(region.analysis_cfg))
            assert mem_address_cfgs == mem_address_bounds_cfg

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Boundaries linked across record and combination of link across records and across Im / Vm
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Note this only tests the configs are updated correctly with linearregion positions. For analysis results, see the respective test files

    def handle_linked_im_vm_all_recs_mode(self, tgui, mode):
        """
        """
        if mode == "link_im_vm_ON_and_link_recs_ON":
            tgui.mw.mw.actionLink_im_vm_on.trigger()
            tgui.mw.mw.actionLink_Across_Records_on.trigger()

        elif mode == "link_im_vm_OFF_and_link_recs_ON":
            tgui.mw.mw.actionLink_im_vm_off.trigger()
            tgui.mw.mw.actionLink_Across_Records_on.trigger()

        elif mode == "link_im_vm_ON_and_link_recs_OFF":
            tgui.mw.mw.actionLink_im_vm_on.trigger()
            tgui.mw.mw.actionLink_Across_Records_off.trigger()

        elif mode == "link_im_vm_OFF_and_link_recs_OFF":
            tgui.mw.mw.actionLink_im_vm_off.trigger()
            tgui.mw.mw.actionLink_Across_Records_off.trigger()

    # TODO: need to check what happens FOR SINGLE FILE!?

    @pytest.mark.parametrize("mode", ["dont_align_across_recs", "align_across_recs"])
    def test_spkcnt_ir_skinetics_linked_across_records(self, tgui, mode):
        """
        Don't test Remove baseline here, it will be tested in configs and if everything else is working
        it should be working fine.

        These test will switch to the relevant analysis, open all possible bounds, and iterate through all recs
        changing the position of bounds and checking the relevant record on the config dict is updated.

        If mode is the same across all recs, only the first rec in the analysis dict is changed and then all other
        entries are updated when the bound is finished moving.

        Otherwise, moving the bound on a certain record updates the relevant entry in the config dict.
        """
        for filenum in range(2):
            tgui.set_link_across_recs(tgui, mode)

            tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                           im_groupbox_on=False)
            tgui.assign_random_boundary_position_for_every_rec_and_test(tgui,
                                                                        tgui.mw.spkcnt_bounds,
                                                                        mode)

            tgui.switch_to_input_resistance_and_set_im_combobox()
            tgui.assign_random_boundary_position_for_every_rec_and_test(tgui,
                                                                        tgui.mw.ir_bounds,
                                                                        mode)

            tgui.switch_to_skinetics_and_set_bound(skinetics_bounds_on=True)
            tgui.assign_random_boundary_position_for_every_rec_and_test(tgui,
                                                                        tgui.mw.skinetics_bounds,
                                                                        mode)
            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.load_file("current_clamp_cumu", SPEED)
        tgui.shutdown()

    @pytest.mark.parametrize("mode", ["dont_align_across_recs", "align_across_recs"])
    def test_curve_fitting_linked_across_records(self, tgui, mode):
        """
        Cycle through every curve fitting region, for two different files in batch mode,
        and then move through every rec and move the boundaries + test position with
        assign_random_boundary_position_for_every_rec_and_test()
        """
        for filenum in range(2):
            tgui.set_analysis_type("curve_fitting")
            tgui.set_link_across_recs(tgui, mode)

            for reg in ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]:
                tgui.assign_random_boundary_position_for_every_rec_and_test(tgui,
                                                                            tgui.mw.curve_fitting_regions[reg],
                                                                            mode)

                tgui.left_mouse_click(tgui.mw.mw.curve_fitting_scroll_region_right_button)

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.load_file("current_clamp_cumu", SPEED)
        tgui.shutdown()

    @pytest.mark.parametrize("mode", ["link_im_vm_ON_and_link_recs_ON",  "link_im_vm_OFF_and_link_recs_ON",
                                      "link_im_vm_ON_and_link_recs_OFF",   "link_im_vm_OFF_and_link_recs_OFF"])
    def test_ir_bounds_with_all_bounds(self, tgui, mode):
        """
        Test the interaction between regions for Ri when they are linked / unlinked for both
        across Im / Vm (i.e. upperplot and lowerplot) and across records.

        Here if recs are not linked they should really be checked after every rec but as this is done in
        assign_random_boundary_position_for_every_rec_and_test() it is not done here for clarity.

        Also at the end, load a new cumulative time file and try again with batch mode on just to ensure it is consistent across files.
        The file loaded is 'current_clamp' which also tests cumulative recs
        """
        for filenum in range(2):

            self.handle_linked_im_vm_all_recs_mode(tgui, mode)
            tgui.switch_to_input_resistance_and_set_im_combobox()
            bounds = tgui.mw.ir_bounds

            # Cycle through all pairs of boundaries, move both the lower and upper version
            # and checking the linked one (e.g. if Im and vm are linked, lower and upper regions are linked)
            for bound_pair in [["lower_bl_lr", "upper_bl_lr"],
                               ["lower_exp_lr", "upper_exp_lr"],
                               ["upper_bl_lr", "lower_bl_lr"],
                               ["upper_exp_lr", "lower_exp_lr"]]:

                bounds_to_move, linked_bound = bound_pair

                all_start = []
                all_stop = []
                saved_cfg = copy.deepcopy(bounds.analysis_cfg)
                for rec in range(tgui.mw.loaded_file.data.num_recs):
                    # for each rec, set the region of the 'bounds_to_move' region and check the configs below.
                    QtWidgets.QApplication.processEvents()
                    tgui.mw.update_displayed_rec(rec)

                    start_time, stop_time = tgui.gen_random_times_for_bounds(tgui, rec)
                    bounds.bounds[bounds_to_move].setRegion((start_time, stop_time))

                    all_start.append(start_time - tgui.mw.loaded_file.data.min_max_time[rec][0])  # save the normalised version that matches configs
                    all_stop.append(stop_time - tgui.mw.loaded_file.data.min_max_time[rec][0])

                if mode == "link_im_vm_ON_and_link_recs_ON":
                    # In this case, all record entries should be the same for the moved bounds. The
                    # bounds should also be identical for the linked bounds
                    assert all(all_start[-1] == bounds.analysis_cfg[bounds_to_move + "_lowerbound"])
                    assert all(all_start[-1] == bounds.analysis_cfg[linked_bound + "_lowerbound"])
                    assert all(all_stop[-1] == bounds.analysis_cfg[bounds_to_move + "_upperbound"])
                    assert all(all_stop[-1] == bounds.analysis_cfg[linked_bound + "_upperbound"])

                elif mode == "link_im_vm_OFF_and_link_recs_ON":
                    # Here all record entries for the moved bound should be the same, but the
                    # other boundary should not change from previus as it is unlinked.
                    assert all(all_start[-1] == bounds.analysis_cfg[bounds_to_move + "_lowerbound"])
                    assert bounds.analysis_cfg[linked_bound + "_lowerbound"] == saved_cfg[linked_bound + "_lowerbound"]
                    assert all(all_stop[-1] == bounds.analysis_cfg[bounds_to_move + "_upperbound"])
                    assert bounds.analysis_cfg[linked_bound + "_upperbound"] == saved_cfg[linked_bound + "_upperbound"]

                elif mode == "link_im_vm_ON_and_link_recs_OFF":
                    # bounds should be linked across Vm / Im but not across recs
                    assert all_start == bounds.analysis_cfg[bounds_to_move + "_lowerbound"]
                    assert all_start == bounds.analysis_cfg[linked_bound + "_lowerbound"]
                    assert all_stop == bounds.analysis_cfg[bounds_to_move + "_upperbound"]
                    assert all_stop == bounds.analysis_cfg[linked_bound + "_upperbound"]

                elif mode == "link_im_vm_OFF_and_link_recs_OFF":
                    # no bounds should match
                    assert all_start == bounds.analysis_cfg[bounds_to_move + "_lowerbound"]
                    assert bounds.analysis_cfg[linked_bound + "_lowerbound"] == saved_cfg[linked_bound + "_lowerbound"]
                    assert all_stop == bounds.analysis_cfg[bounds_to_move + "_upperbound"]
                    assert bounds.analysis_cfg[linked_bound + "_upperbound"] == saved_cfg[linked_bound + "_upperbound"]

            tgui.mw.mw.actionBatch_Mode_ON.trigger()
            tgui.load_file("current_clamp_cumu", SPEED)
        tgui.shutdown()

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Im baselines are correct
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_spikecounts_im_baseline_is_correct(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True)
        average_bounds, average_axline, test_average = self.check_lr_baseline(tgui,
                                                                              tgui.mw.spkcnt_bounds,
                                                                              'lower_bl_lr',
                                                                              tgui.mw.loaded_file.data.im_array,
                                                                              tgui.mw.cfgs.spkcnt)
        assert average_bounds == test_average
        assert average_axline == test_average

    def test_input_resistance_vm_baseline_is_correct(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox()
        average_bounds, average_axline, test_average = self.check_lr_baseline(tgui,
                                                                              tgui.mw.ir_bounds,
                                                                              'upper_bl_lr',
                                                                              tgui.mw.loaded_file.data.vm_array,
                                                                              tgui.mw.cfgs.ir)
        assert average_bounds == test_average
        assert average_axline == test_average

    def test_input_resistance_im_baseline_is_correct(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox()
        average_bounds, average_axline, test_average = self.check_lr_baseline(tgui,
                                                                              tgui.mw.ir_bounds,
                                                                              'lower_bl_lr',
                                                                              tgui.mw.loaded_file.data.im_array,
                                                                              tgui.mw.cfgs.ir)
        assert average_bounds == test_average
        assert average_axline == test_average

    # also jsut use this method to check position of linearregions
    def test_input_resistance_im_lower_exp_is_correct(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox()
        average_bounds, __, test_average = self.check_lr_baseline(tgui,
                                                                  tgui.mw.ir_bounds,
                                                                  'lower_exp_lr',
                                                                  tgui.mw.loaded_file.data.im_array,
                                                                  tgui.mw.cfgs.ir)
        assert average_bounds == test_average

    def test_input_resistance_im_upper_exp_is_correct(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox()
        average_bounds, __, test_average = self.check_lr_baseline(tgui,
                                                                  tgui.mw.ir_bounds,
                                                                  'upper_exp_lr',
                                                                  tgui.mw.loaded_file.data.im_array,
                                                                  tgui.mw.cfgs.ir)
        assert average_bounds == test_average

    def test_spikecounts_lower_exp_is_correct(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True)
        average_bounds, __, test_average = self.check_lr_baseline(tgui,
                                                                  tgui.mw.spkcnt_bounds,
                                                                  'lower_exp_lr',
                                                                  tgui.mw.loaded_file.data.im_array,
                                                                  tgui.mw.cfgs.spkcnt)
        assert average_bounds == test_average

    def test_spikecounts_upper_exp_is_correct(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=False)
        average_bounds, __, test_average = self.check_lr_baseline(tgui,
                                                                  tgui.mw.spkcnt_bounds,
                                                                  'upper_exp_lr',
                                                                  tgui.mw.loaded_file.data.vm_array,
                                                                  tgui.mw.cfgs.spkcnt)
        assert average_bounds == test_average

    def test_skinetics_upper_exp_is_correct(self, tgui):
        tgui.switch_to_skinetics_and_set_bound(skinetics_bounds_on=True)
        average_bounds, __, test_average = self.check_lr_baseline(tgui,
                                                                  tgui.mw.skinetics_bounds,
                                                                  'upper_exp_lr',
                                                                  tgui.mw.loaded_file.data.vm_array,
                                                                  tgui.mw.cfgs.skinetics)
        assert average_bounds == test_average

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Boundaries are Shown Within Specific Records
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_spikecount_user_im_widgets_error_catch_within_recs(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True,
                                                       im_setting='user_input_im')
        num_recs = tgui.set_analyse_specific_recs(rec_from=5,
                                                  rec_to=10)
        assert self.check_user_im_input(tgui, num_recs,
                                        'spkcnt', correctly_filled=False, within_recs=True), "Test Spikecount user input analysis run - filled incorrectly"

    def test_spikecount_user_im_widgets_no_error_within_recs(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=False,
                                                       im_groupbox_on=True,
                                                       im_setting='user_input_im')
        num_recs = tgui.set_analyse_specific_recs(rec_from=5,
                                                  rec_to=10)
        assert self.check_user_im_input(tgui, num_recs,
                                        'spkcnt', correctly_filled=True), "Test Spikecount user input analysis run - filled correctly"

    def test_input_resistance_user_im_error_within_recs(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting='user_input_im')
        num_recs = tgui.set_analyse_specific_recs(rec_from=5,
                                                  rec_to=10)
        assert self.check_user_im_input(tgui, num_recs,
                                        'Ri', correctly_filled=False), 'test'

    def test_input_resistance_user_im_no_error_within_recs(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox(im_setting='user_input_im')
        num_recs = tgui.set_analyse_specific_recs(rec_from=5,
                                                  rec_to=10)
        assert self.check_user_im_input(tgui, num_recs,
                                        'Ri', correctly_filled=False), 'test'

# Check Im widgets again but when analysis is restricted to certain recs -----------------------------------------------------------------------------

    def test_spikecount_boundaries_within_analysed_recs(self, tgui):
        tgui.switch_to_spikecounts_and_set_im_combobox(spike_bounds_on=True,
                                                       im_groupbox_on=True)
        _ = tgui.set_analyse_specific_recs(rec_from=5,
                                           rec_to=10)

        # before rec
        tgui.mw.update_displayed_rec(4)
        assert tgui.mw.spkcnt_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.spkcnt_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.spkcnt_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.spkcnt_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items

        # in reocs
        tgui.mw.update_displayed_rec(5)
        assert tgui.mw.spkcnt_bounds.bounds['upper_exp_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.spkcnt_bounds.bounds['lower_exp_lr'] in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.spkcnt_bounds.bounds['lower_bl_lr'] in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.spkcnt_bounds.axlines['lower_bl_lr'] in tgui.mw.loaded_file_plot.lowerplot.items

        # after recs
        tgui.mw.update_displayed_rec(11)
        assert tgui.mw.spkcnt_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.spkcnt_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.spkcnt_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.spkcnt_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items

    def test_input_resistance_boundaries_within_analysed_recs(self, tgui):
        tgui.switch_to_input_resistance_and_set_im_combobox()
        _ = tgui.set_analyse_specific_recs(rec_from=5,
                                           rec_to=10)

        # before rec
        tgui.mw.update_displayed_rec(4)
        assert tgui.mw.ir_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.axlines['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items

        # in reocs
        tgui.mw.update_displayed_rec(5)
        assert tgui.mw.ir_bounds.bounds['upper_exp_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['upper_bl_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.axlines['upper_bl_lr'] in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['lower_exp_lr'] in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.bounds['lower_bl_lr'] in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.axlines['lower_bl_lr'] in tgui.mw.loaded_file_plot.lowerplot.items

        # after recs
        tgui.mw.update_displayed_rec(11)
        assert tgui.mw.ir_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.axlines['upper_bl_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
        assert tgui.mw.ir_bounds.bounds['lower_exp_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.bounds['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items
        assert tgui.mw.ir_bounds.axlines['lower_bl_lr'] not in tgui.mw.loaded_file_plot.lowerplot.items

    def test_skinetics_boundaries_within_analysed_recs(self, tgui):
        tgui.set_analysis_type('skinetics')
        _ = tgui.set_analyse_specific_recs(rec_from=5,
                                           rec_to=10)
        # before rec
        tgui.mw.update_displayed_rec(3)
        assert tgui.mw.skinetics_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items

        # in reocs
        tgui.mw.update_displayed_rec(4)
        assert tgui.mw.skinetics_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items

        # after recs
        tgui.mw.update_displayed_rec(11)
        assert tgui.mw.skinetics_bounds.bounds['upper_exp_lr'] not in tgui.mw.loaded_file_plot.upperplot.items
