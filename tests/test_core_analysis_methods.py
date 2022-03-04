import copy

from PySide2 import QtWidgets, QtCore, QtGui
from PySide2 import QtTest
from PySide2.QtTest import QTest
import pytest
import sys
import os
import pandas as pd
import numpy as np
import time
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), 'easy_electrophysiology'))
from easy_electrophysiology.easy_electrophysiology.easy_electrophysiology import MainWindow
from ephys_data_methods import core_analysis_methods, event_analysis_master, voltage_calc

class TestCoreAnalysisMethods:
    """
    Load artificial data from test_generate_artificial_data that contains pre-generated traces with 'spikes' or current
    injection. Test analysing within specific rec/bounds.
    """
    def test_sort_dict_based_on_keys(self):
        """
        Make a dict of matching randomly generated float key and values, sort them and check
        against known correct order
        """
        keys = np.random.choice(1000001, 1000001, replace=False) / 1000  # use float more similar to times
        test_dict = {str(key): key for key in keys}

        sorted_test_dict = core_analysis_methods.sort_dict_based_on_keys(test_dict)

        assert np.array_equal(np.sort(keys), np.array(list(sorted_test_dict.keys())).astype(float))
        assert np.array_equal(np.sort(keys), np.array(list(sorted_test_dict.values())))

    @staticmethod
    def make_diff(n, test_data, num_recs, ts_in_ms):

        diff = np.diff(np.hstack([test_data.vm_array, np.zeros((num_recs, n))]), n=n) / ts_in_ms

        return diff

    def test_set_data_params(self):
        """
        Make a fake data class, ensure all parameters are calculated correctly
        """
        num_recs = 10
        num_samples = 10000
        rec_start_times = np.atleast_2d(np.arange(num_recs) * 10).T
        rec_len_s = 1

        class TestClass:
            def __init__(self):

                time_array = np.linspace(0, rec_len_s, num_samples)
                time_array = time_array + rec_start_times

                self.time_array = time_array
                self.vm_array = np.random.random((num_recs, num_samples))
                self.im_array = np.random.random((num_recs, num_samples))
                self.ts = np.mean(np.diff(self.time_array[0, :]))

        test_data = TestClass()

        data_object = copy.deepcopy(test_data)
        core_analysis_methods.set_data_params(data_object)

        assert np.array_equal(data_object.min_max_time,
                              np.hstack([rec_start_times, rec_start_times + rec_len_s]))

        assert data_object.t_start == 0
        assert data_object.t_stop == rec_start_times[-1] + rec_len_s
        assert data_object.num_recs == num_recs
        assert data_object.num_samples == num_samples
        assert data_object.rec_time == rec_len_s

        assert data_object.ts == test_data.ts
        assert data_object.fs == 1 / test_data.ts

        ts_in_ms = data_object.ts * 1000
        assert np.array_equal(data_object.norm_first_deriv_vm,
                              self.make_diff(1, test_data, num_recs, ts_in_ms))
        assert np.array_equal(data_object.norm_second_deriv_vm,
                              self.make_diff(2, test_data, num_recs, ts_in_ms))
        assert np.array_equal(data_object.norm_third_deriv_vm,
                              self.make_diff(3, test_data, num_recs, ts_in_ms))

        for item in [data_object.vm_array, data_object.im_array, data_object.time_array, data_object.min_max_time,
                     data_object.norm_first_deriv_vm, data_object.norm_third_deriv_vm, data_object.norm_second_deriv_vm]:

            assert item.flags["WRITEABLE"] is False

    def test_twohundred_kHz_interpolate(self):

        num_samples = 10000
        y = np.random.random(num_samples)
        x = np.linspace(0, 1.5, num_samples)

        interp_y, interp_x = core_analysis_methods.twohundred_kHz_interpolate(y, x)

        ts = interp_x[1] - interp_x[0]

        assert np.isclose(1/ts, 200000, atol=1e-10, rtol=0)

    def test_area_under_curve(self):

        y = np.zeros(10000)
        y[2000:2500] = 1
        ts = 1 / 1000
        area, area_time_ms = core_analysis_methods.area_under_curve_ms(y, ts)

        assert area == 500
        assert area_time_ms == 9999
