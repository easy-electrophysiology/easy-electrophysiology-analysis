from PySide2 import QtWidgets, QtCore, QtGui
from PySide2 import QtTest
from PySide2.QtTest import QTest
import pytest
import sys
import os
import numpy as np
import time
import keyboard
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))

from ..easy_electrophysiology import easy_electrophysiology
MainWindow = easy_electrophysiology.MainWindow
from ephys_data_methods import current_calc
from utils import utils
from setup_test_suite import GuiTestSetup


class TestLinearRegionsImSettings:

    @pytest.fixture(scope="function", params=["normalised", "cumulative"], ids=["normalised", "cumulative"])
    def tgui(test, request):
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.speed = "fast" # has no effect
        tgui.setup_artificial_data(request.param)
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    def get_limits(self, tgui, rec):
        """
        add padding back on so view range will match raw data
        """
        vb_pad_left = tgui.mw.loaded_file_plot.x_axis_pad_left
        vb_pad_right = tgui.mw.loaded_file_plot.x_axis_pad_right

        pad_left = (tgui.mw.loaded_file.data.min_max_time[rec][1] -
                    tgui.mw.loaded_file.data.min_max_time[rec][0]) * vb_pad_left  # small offset added to end of viewbox set exactly to range cuts off last xticklabel

        pad_right = (tgui.mw.loaded_file.data.min_max_time[rec][1] -
                     tgui.mw.loaded_file.data.min_max_time[rec][0]) * vb_pad_right

        upper_min = tgui.mw.loaded_file_plot.upperplot.vb.state["viewRange"][0][0] + pad_left
        upper_max = tgui.mw.loaded_file_plot.upperplot.vb.state["viewRange"][0][1] - pad_right
        lower_min = tgui.mw.loaded_file_plot.lowerplot.vb.state["viewRange"][0][0] + pad_left
        lower_max = tgui.mw.loaded_file_plot.lowerplot.vb.state["viewRange"][0][1] - pad_right

        return upper_min, upper_max, lower_min, lower_max

    def eq(self, arg1, arg2):
        return np.array_equal(arg1,
                              arg2,
                              equal_nan=True)

    def test_x_axis_on_scroll(self, tgui):

        for rec in range(0, tgui.adata.num_recs):

            upper_min, upper_max, lower_min, lower_max = self.get_limits(tgui, rec)

            assert np.isclose(upper_min,
                              tgui.mw.loaded_file.data.min_max_time[rec][0], atol=1e-08, rtol=0), "error in upper min rec:{0}".format(rec)
            assert np.isclose(upper_max,
                              tgui.mw.loaded_file.data.min_max_time[rec][1], atol=1e-08, rtol=0), "error in upper max rec:{0}".format(rec)
            assert np.isclose(lower_min,
                              tgui.mw.loaded_file.data.min_max_time[rec][0], atol=1e-08, rtol=0), "error in lower min rec:{0}".format(rec)
            assert np.isclose(lower_max,
                              tgui.mw.loaded_file.data.min_max_time[rec][1], atol=1e-08, rtol=0), "error in lower max rec:{0}".format(rec)

            tgui.left_mouse_click(tgui.mw.mw.current_rec_rightbutton)
