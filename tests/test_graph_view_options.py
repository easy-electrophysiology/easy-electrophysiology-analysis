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
from utils import utils

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Graph View Options
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def load_graph_view_dialo(func):
    def wrapper(*args, **kwargs):
        self.mw.mw.actionGraph_Options.trigger()
        self.mw.dialogs['graph_view_options'].hide()
        return func(*args, **kwargs)
    return wrapper

class TestGraphViewOptions:

    @pytest.fixture(scope="function")
    def tgui(test):
        tgui = GuiTestSetup('cc_two_channel_abf')
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo(norm=False)
        tgui.test_load_cumu_time_file()
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    def test_display_rec_toggles(self, tgui):
        """
        Check both display modes - all records or per record.

        First set to all records and cycle through all recs, changing the y axis limits never change (these is set to the
        maximum of all recs).

        Next change mode to display current record only (not this mode has a performance decrease) and switch through the plots,
        checking the view changes each record

        The y axis view is padded around the displayed data,
        so to check we need to use the same function in plotgraphs that generates the padding.
        """
        tgui.mw.mw.actionGraph_Options.trigger()

        # Test y range does not change
        upper_y_range = tgui.mw.loaded_file_plot.upperplot.getAxis("left").range
        lower_y_range = tgui.mw.loaded_file_plot.lowerplot.getAxis("left").range

        for __ in range(tgui.mw.loaded_file.data.num_recs):

            tgui.left_mouse_click(tgui.mw.mw.current_rec_rightbutton)
            assert tgui.mw.loaded_file_plot.upperplot.getAxis("left").range == upper_y_range
            assert tgui.mw.loaded_file_plot.lowerplot.getAxis("left").range == lower_y_range

        # set scale per range and check it matches data
        tgui.mw.update_displayed_rec(0)
        tgui.left_mouse_click(tgui.mw.dialogs['graph_view_options'].dia.upper_scale_display_rec_radiobutton)
        tgui.left_mouse_click(tgui.mw.dialogs['graph_view_options'].dia.lower_scale_display_rec_radiobutton)

        vm_array = tgui.mw.loaded_file.data.vm_array
        im_array = tgui.mw.loaded_file.data.im_array
        for rec in range(tgui.mw.loaded_file.data.num_recs):

            __, __, y_min_w_pad, y_max_w_pad = tgui.mw.loaded_file_plot.get_y_axis_view_limits(vm_array, min_ext=0.05, max_ext=0.1, viewbox_padding=0.008)
            assert tgui.mw.loaded_file_plot.upperplot.vb.state["limits"]["yLimits"] == [y_min_w_pad, y_max_w_pad]

            __, __, y_min_w_pad, y_max_w_pad = tgui.mw.loaded_file_plot.get_y_axis_view_limits(im_array, min_ext=0.2, max_ext=0.15, viewbox_padding=0.008)
            assert tgui.mw.loaded_file_plot.lowerplot.vb.state["limits"]["yLimits"] == [y_min_w_pad, y_max_w_pad]

            tgui.left_mouse_click(tgui.mw.mw.current_rec_rightbutton)

    def test_plot_size_ratio_slider(self, tgui):
        """
        Change the size of the plot (this is achieved by changing the layout row stretch factors)
        and check the plot stretch factors change correctly.
        """
        tgui.mw.mw.actionGraph_Options.trigger()

        slider = tgui.mw.dialogs['graph_view_options'].dia.size_ratio_slider

        slider_range = np.abs(slider.minimum()) + slider.maximum()

        assert tgui.mw.mw.graphics_layout_widget.ci.layout.rowStretchFactor(0) == 18
        assert tgui.mw.mw.graphics_layout_widget.ci.layout.rowStretchFactor(1) == 1

        tgui.repeat_key_click(widget=slider,
                              key=QtGui.Qt.Key_Left,
                              n_clicks=slider_range * 2,
                              delay=0.05)

        assert tgui.mw.mw.graphics_layout_widget.ci.layout.rowStretchFactor(0) == 0
        assert tgui.mw.mw.graphics_layout_widget.ci.layout.rowStretchFactor(1) == 100

        tgui.repeat_key_click(slider,
                              QtGui.Qt.Key_Right,
                              slider_range * 2,
                              0.05)

        assert tgui.mw.mw.graphics_layout_widget.ci.layout.rowStretchFactor(0) == 100
        assert tgui.mw.mw.graphics_layout_widget.ci.layout.rowStretchFactor(1) == 0

    def test_gridline_buttons(self, tgui):
        """
        Turn the gridlines off and on and check the display changes
        """
        tgui.mw.mw.actionGraph_Options.trigger()

        assert tgui.mw.loaded_file_plot.upperplot.getAxis("left").grid == 80
        assert tgui.mw.loaded_file_plot.upperplot.getAxis("left").grid == 80

        tgui.switch_checkbox(tgui.mw.dialogs['graph_view_options'].dia.upper_plot_gridlines, on=False)
        tgui.switch_checkbox(tgui.mw.dialogs['graph_view_options'].dia.lower_plot_gridlines, on=False)

        assert tgui.mw.loaded_file_plot.upperplot.getAxis("left").grid == 0
        assert tgui.mw.loaded_file_plot.upperplot.getAxis("left").grid == 0

        tgui.switch_checkbox(tgui.mw.dialogs['graph_view_options'].dia.upper_plot_gridlines, on=True)
        tgui.switch_checkbox(tgui.mw.dialogs['graph_view_options'].dia.lower_plot_gridlines, on=True)

        assert tgui.mw.loaded_file_plot.upperplot.getAxis("left").grid == 80
        assert tgui.mw.loaded_file_plot.upperplot.getAxis("left").grid == 80

    def test_plot_width_pen(self, tgui):
        """
        Shift the plot-pen width from 1 to 0.1 in 0.1 steps and check the plot changes.
        """
        tgui.mw.mw.actionGraph_Options.trigger()

        plotpen_spinbox = tgui.mw.dialogs['graph_view_options'].dia.plotpen_thickness_spinbox

        for delta_width in range(0, 10):

            new_width = 1 - (delta_width/10)

            tgui.enter_number_into_spinbox(plotpen_spinbox,
                                           round(new_width, 1))

            assert utils.allclose(tgui.mw.loaded_file_plot.curve_upper.opts["pen"].widthF(), new_width, 1e-10)
            assert utils.allclose(tgui.mw.loaded_file_plot.curve_lower.opts["pen"].widthF(), new_width, 1e-10)

    def test_swap_plot_positions(self, tgui):
        """
        Check that the position of the plots is properly swapped when the button is pressed.

        Also run a spike-count analysis and check the results are displayed on the proper plot.
        This is kind of over-kill, the results are plot onto the PlotItem which does not change, only the
        position on the graphics_layout_widget does.
        """
        tgui.mw.mw.actionGraph_Options.trigger()

        swap_button = tgui.mw.dialogs['graph_view_options'].dia.swap_plot_position_button

        assert tgui.mw.mw.graphics_layout_widget.ci.getItem(0, 0).objectName() == "upperplot"
        assert tgui.mw.mw.graphics_layout_widget.ci.getItem(1, 0).objectName() == "lowerplot"

        # swap plot position
        tgui.left_mouse_click(swap_button)

        assert tgui.mw.mw.graphics_layout_widget.ci.getItem(1, 0).objectName() == "upperplot"
        assert tgui.mw.mw.graphics_layout_widget.ci.getItem(0, 0).objectName() == "lowerplot"

        # run analysis and check results are plot correctly
        tgui.run_spikecount_analysis()
        switch_to_rec = 10  # switch to a rec with spikes
        tgui.mw.update_displayed_rec(switch_to_rec)

        assert np.array_equal(tgui.mw.loaded_file_plot.spkcnt_plot.xData,
                              np.array(list(tgui.mw.loaded_file.spkcnt_spike_info[switch_to_rec].keys())).astype(np.float64))

        assert not tgui.mw.loaded_file_plot.spkcnt_plot in tgui.mw.mw.graphics_layout_widget.ci.getItem(0, 0).allChildItems()
        assert tgui.mw.loaded_file_plot.spkcnt_plot in tgui.mw.mw.graphics_layout_widget.ci.getItem(1, 0).allChildItems()

        # swap plot position with results displayed and check they move
        tgui.left_mouse_click(swap_button)

        assert not tgui.mw.loaded_file_plot.spkcnt_plot in tgui.mw.mw.graphics_layout_widget.ci.getItem(1, 0).allChildItems()
        assert tgui.mw.loaded_file_plot.spkcnt_plot in tgui.mw.mw.graphics_layout_widget.ci.getItem(0, 0).allChildItems()

        assert tgui.mw.mw.graphics_layout_widget.ci.getItem(0, 0).objectName() == "upperplot"
        assert tgui.mw.mw.graphics_layout_widget.ci.getItem(1, 0).objectName() == "lowerplot"

    def test_limit_range_performance(self, tgui):
        """
        Check that the plot view for a long voltage clamp file with 1 record is properly cut according to the settings.

        By default, 500k samples only should be shown. Then turn the option off, and the entire file should be shown (pyqtgraph the limit is None).
        Turn it back on and try 10k samples, check this switch works well.
        """
        tgui.load_a_filetype("voltage_clamp_1_record")
        view_range_default_start = 0.0005

        default_num_samples = 500000
        assert tgui.mw.loaded_file_plot.upperplot.vb.state["limits"]["xRange"] == [view_range_default_start, default_num_samples * tgui.mw.loaded_file.data.ts]

        # Turn off
        tgui.mw.mw.actionGraph_Options.trigger()
        tgui.switch_groupbox(tgui.mw.dialogs['graph_view_options'].dia.limit_view_range_for_performance_groupbox,
                             on=False)

        tgui.load_a_filetype("voltage_clamp_1_record")
        assert tgui.mw.loaded_file_plot.upperplot.vb.state["limits"]["xRange"] == [view_range_default_start, None]

        # Turn on
        test_num_samples = 10000
        limit_spinbox_dia = tgui.mw.dialogs['graph_view_options'].dia
        tgui.switch_groupbox(limit_spinbox_dia.limit_view_range_for_performance_groupbox, on=True)
        tgui.enter_number_into_spinbox(limit_spinbox_dia.limit_view_range_for_performance_spinbox,
                                       round(test_num_samples / 1000))  # spinbox is in units of thousand samples

        tgui.load_a_filetype("voltage_clamp_1_record")
        assert tgui.mw.loaded_file_plot.upperplot.vb.state["limits"]["xRange"] == [view_range_default_start, test_num_samples * tgui.mw.loaded_file.data.ts]
