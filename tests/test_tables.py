import math
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6 import QtTest
from PySide6.QtTest import QTest
import pytest
import sys
import os
import numpy as np
import copy

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology.mainwindow.mainwindow import (
    MainWindow,
)
from setup_test_suite import GuiTestSetup
from utils import utils
import scipy.stats
import test_curve_fitting
import more_itertools
import keyboard
import time
from slow_vs_fast_settings import get_settings

SPEED = "fast"

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test Table Data Classes
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class SpkcntTGui(GuiTestSetup):
    """
    This class requires multiple Guis with different files loaded, so is not handled with pytests native interface but through these custom classes.
    Shutting down in the __del__ field aims for neat teardown at the end of the test functions.

    The logic of these tests is to run 5 files for each analysis and check that all data is correctly displayed on the table, both in row and column mode.
    In column mode, the summary statistics are calculated in the test function and then summary statistics options checked in EE, and
    it is checked that correct summary statistics are displayed on the table.
    """

    def __init__(self):
        super(SpkcntTGui, self).__init__("artificial")
        # = GuiTestSetup("artificial")  # no setup main window, do in function
        self.setup_mainwindow(show=True)
        self.test_update_fileinfo()
        self.speed = SPEED
        self.setup_artificial_data("cumulative")
        self.raise_mw_and_give_focus()

    def __del__(self):
        self.shutdown()


class EventsGui(GuiTestSetup):
    def __init__(self):
        super(EventsGui, self).__init__("artificial_events_multi_record_cont")

        self.setup_mainwindow(show=True)
        self.test_update_fileinfo(norm=True)
        self.analysis_type = "events_multi_record_table"
        self.speed = SPEED
        self.setup_artificial_data("cumulative", analysis_type="events_multi_record_table")
        self.raise_mw_and_give_focus()

    def __del__(self):
        self.shutdown()


class CurveFittingGui(GuiTestSetup):
    def __init__(self):
        super(CurveFittingGui, self).__init__("artificial")

        self.setup_mainwindow(show=True)
        self.test_update_fileinfo()
        self.setup_artificial_data("cumulative", analysis_type="curve_fitting")
        self.raise_mw_and_give_focus()

    def __del__(self):
        self.shutdown()


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Test Table
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class TestTables:
    def test_curve_fitting_column_wise(self):
        """
        See test_events_per_column_table()
        """
        tgui = CurveFittingGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()

        # File 1, reg_1, reg_3,  triexp, area_under_curve_cf -------------------------------------------------------------------------------------------------------------------------------

        self.run_curve_fitting_analysis(tgui, "1")

        file1_start_col = 0
        file1_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file1_col_keys = self.get_col_keys(tgui, file1_start_col, file1_end_col, "curve_fitting", row=1)

        self.check_curve_fitting_filenames(tgui, [file1_start_col], ["reg_1"], tgui.fake_filename)

        assert file1_col_keys == self.combine([self.cf_keys("triexp"), self.cf_keys("area_under_curve_cf")])

        data_on_table, __ = self.get_and_cut_col_data_on_table(tgui, file1_start_col, file1_end_col)

        stored_tabledata, __ = self.get_curve_fitting_table(
            tgui,
            rec_from=0,
            rec_to=tgui.adata.num_recs - 1,
            regions=["reg_1", "reg_3"],
            region_keys=[self.cf_keys("triexp"), self.cf_keys("area_under_curve_cf")],
            file_idx=0,
        )

        assert np.array_equal(data_on_table, stored_tabledata)

        # File 2, reg_2, reg_3, min, mean ------------------------------------------------------- -------------------------------------------------------------

        rec_from, rec_to = self.run_curve_fitting_analysis(tgui, "2")

        file2_start_col = self.get_col_file_start_indexes(tgui)[2]
        file2_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file2_col_keys = self.get_col_keys(tgui, file2_start_col, file2_end_col, "curve_fitting", row=1)

        self.check_curve_fitting_filenames(
            tgui,
            self.get_col_file_start_indexes(tgui)[2:4],
            ["reg_2", "reg_3"],
            tgui.fake_filename,
        )

        assert file2_col_keys == self.combine([self.cf_keys("min"), self.cf_keys("mean")])

        data_on_table, __ = self.get_and_cut_col_data_on_table(tgui, file2_start_col, file2_end_col)
        stored_tabledata, __ = self.get_curve_fitting_table(
            tgui,
            rec_from=rec_from,
            rec_to=rec_to,
            regions=["reg_2", "reg_3"],
            region_keys=[self.cf_keys("min"), self.cf_keys("mean")],
            file_idx=1,
        )

        assert np.array_equal(data_on_table, stored_tabledata)

        # File 3, reg_1, reg_2, reg_3, reg_4, area_under_curve, monoexp, biexp_event, biexp_decay --------------------------------------------------------------------------------

        self.run_curve_fitting_analysis(tgui, "3")

        file3_start_col = self.get_col_file_start_indexes(tgui)[4]  # all hard coded
        file3_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file3_col_keys = self.get_col_keys(tgui, file3_start_col, file3_end_col, "curve_fitting", row=1)

        self.check_curve_fitting_filenames(
            tgui,
            self.get_col_file_start_indexes(tgui)[4:8],
            ["reg_1", "reg_2", "reg_3", "reg_4"],
            tgui.fake_filename,
        )

        assert file3_col_keys == self.combine(
            [
                self.cf_keys("area_under_curve_cf"),
                self.cf_keys("monoexp"),
                self.cf_keys("biexp_event"),
                self.cf_keys("biexp_decay"),
            ]
        )

        data_on_table, table_text = self.get_and_cut_col_data_on_table(tgui, file3_start_col, file3_end_col)
        stored_tabledata, __ = self.get_curve_fitting_table(
            tgui,
            rec_from=0,
            rec_to=tgui.adata.num_recs - 1,
            regions=["reg_1", "reg_2", "reg_3", "reg_4"],
            region_keys=[
                self.cf_keys("area_under_curve_cf"),
                self.cf_keys("monoexp"),
                self.cf_keys("biexp_event"),
                self.cf_keys("biexp_decay"),
            ],
            file_idx=2,
        )

        assert np.array_equal(data_on_table, stored_tabledata, equal_nan=True)

        # File 4, reg_5, reg_6, median, triexp -------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_curve_fitting_analysis(tgui, "4")

        file4_start_col = self.get_col_file_start_indexes(tgui)[8]  # all hard coded
        file4_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file4_col_keys = self.get_col_keys(tgui, file4_start_col, file4_end_col, "curve_fitting", row=1)

        self.check_curve_fitting_filenames(
            tgui,
            self.get_col_file_start_indexes(tgui)[8:10],
            ["reg_5", "reg_6"],
            tgui.fake_filename,
        )

        assert file4_col_keys == self.combine([self.cf_keys("median"), self.cf_keys("triexp")])

        data_on_table, __ = self.get_and_cut_col_data_on_table(tgui, file4_start_col, file4_end_col)
        stored_tabledata, __ = self.get_curve_fitting_table(
            tgui,
            rec_from=rec_from,
            rec_to=rec_to,
            regions=["reg_5", "reg_6"],
            region_keys=[self.cf_keys("median"), self.cf_keys("triexp")],
            file_idx=3,
        )

        assert np.array_equal(data_on_table, stored_tabledata, equal_nan=True)

        # File 5, reg_1, reg_2, reg_3, reg_4, reg_5, reg_6, min, median, triexp, biexp_event, mean, biexp_decay -----------------------------------------

        rec_from, rec_to = self.run_curve_fitting_analysis(tgui, "5")

        file5_start_col = self.get_col_file_start_indexes(tgui)[10]  # all hard coded
        file5_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file5_col_keys = self.get_col_keys(tgui, file5_start_col, file5_end_col, "curve_fitting", row=1)

        self.check_curve_fitting_filenames(
            tgui,
            self.get_col_file_start_indexes(tgui)[10:],
            ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"],
            tgui.fake_filename,
        )

        assert file5_col_keys == self.combine(
            [
                self.cf_keys("min"),
                self.cf_keys("median"),
                self.cf_keys("triexp"),
                self.cf_keys("biexp_event"),
                self.cf_keys("mean"),
                self.cf_keys("biexp_decay"),
            ]
        )

        data_on_table, __ = self.get_and_cut_col_data_on_table(tgui, file5_start_col, file5_end_col)
        stored_tabledata, __ = self.get_curve_fitting_table(
            tgui,
            rec_from=rec_from,
            rec_to=rec_to,
            regions=["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"],
            region_keys=[
                self.cf_keys("min"),
                self.cf_keys("median"),
                self.cf_keys("triexp"),
                self.cf_keys("biexp_event"),
                self.cf_keys("mean"),
                self.cf_keys("biexp_decay"),
            ],
            file_idx=4,
        )

        assert np.array_equal(data_on_table, stored_tabledata, equal_nan=True)

    def test_curve_fitting_row_wise_different_analyses_across_files(self):
        """
        See test_events_per_row_table()
        """
        tgui = CurveFittingGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        self.set_row_or_column_display(tgui, "row")

        # File 1, reg_1, triexp -------------------------------------------------------------------------------------------------------------------------------

        self.run_curve_fitting_analysis(tgui, "1")

        test_file1_col_keys = self.combine(
            [
                self.cf_keys("triexp", add_filename=True),
                self.cf_keys("area_under_curve_cf", add_filename=True),
            ]
        )
        file1_end_col = len(test_file1_col_keys)

        file1_col_keys, file1_filename = self.get_col_keys(
            tgui, 0, file1_end_col, "curve_fitting", row=0, return_filename=True
        )

        file1_regions = ["reg_1", "reg_3"]
        assert file1_col_keys == test_file1_col_keys

        file1_start_row = 1
        file1_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_curve_fitting_row_table(
            tgui,
            to_check=["1"],
            info={
                "file1_regions": file1_regions,
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_end_col": file1_end_col,
            },
        )

        # File 2, reg_2, reg_3, min, mean ------------------------------------------------------- -------------------------------------------------------------

        rec_from, rec_to = self.run_curve_fitting_analysis(tgui, "2")

        test_file2_col_keys = self.combine(
            [
                self.cf_keys("min", add_filename=True),
                self.cf_keys("mean", add_filename=True),
            ]
        )
        file2_end_col = len(test_file2_col_keys)

        # need to calc num cols or hard code
        file2_col_keys, file2_filename = self.get_col_keys(
            tgui,
            0,
            file2_end_col,
            "curve_fitting",
            row=file1_end_row + 1,
            return_filename=True,
        )  # TODO: nicer way to this

        assert file2_col_keys == test_file2_col_keys

        file2_regions = ["reg_2", "reg_3"]
        file2_start_row = file1_end_row + 2
        file2_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_curve_fitting_row_table(
            tgui,
            to_check=["1", "2"],
            info={
                "file1_regions": file1_regions,
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_end_col": file1_end_col,
                "file2_regions": file2_regions,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file2_end_col": file2_end_col,
            },
            rec_from=rec_from,
            rec_to=rec_to,
        )

        # File 3, reg_1, reg_2, reg_3, reg_4, area_under_curve, monoexp, biexp_event, biexp_decay --------------------------------------------------------------------------------

        self.run_curve_fitting_analysis(tgui, "3")

        test_file3_col_keys = self.combine(
            [
                self.cf_keys("area_under_curve_cf", add_filename=True),
                self.cf_keys("monoexp", add_filename=True),
                self.cf_keys("biexp_event", add_filename=True),
                self.cf_keys("biexp_decay", add_filename=True),
            ]
        )

        file3_end_col = len(test_file3_col_keys)

        # need to calc num cols or hard code
        file3_col_keys, file3_filename = self.get_col_keys(
            tgui,
            0,
            file3_end_col,
            "curve_fitting",
            row=file2_end_row + 1,
            return_filename=True,
        )  # TODO: nicer way to this

        file3_regions = ["reg_1", "reg_2", "reg_3", "reg_4"]
        assert file3_col_keys == test_file3_col_keys

        file3_start_row = file2_end_row + 2
        file3_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_curve_fitting_row_table(
            tgui,
            to_check=["1", "2", "3"],
            info={
                "file1_regions": file1_regions,
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_end_col": file1_end_col,
                "file2_regions": file2_regions,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file2_end_col": file2_end_col,
                "file3_regions": file3_regions,
                "file3_col_keys": file3_col_keys,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file3_end_col": file3_end_col,
            },
            rec_from=rec_from,
            rec_to=rec_to,
        )

        # File 4, reg_5, reg_6, median, triexp -------------------------------------------------------------------------------------------------------------

        self.run_curve_fitting_analysis(tgui, "4")

        test_file4_col_keys = self.combine(
            [
                self.cf_keys("median", add_filename=True),
                self.cf_keys("triexp", add_filename=True),
            ]
        )

        file4_end_col = len(test_file4_col_keys)

        # need to calc num cols or hard code
        file4_col_keys, file4_filename = self.get_col_keys(
            tgui,
            0,
            file4_end_col,
            "curve_fitting",
            row=file3_end_row + 1,
            return_filename=True,
        )  # TODO: nicer way to this

        file4_regions = ["reg_5", "reg_6"]
        assert file4_col_keys == test_file4_col_keys

        file4_start_row = file3_end_row + 2
        file4_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_curve_fitting_row_table(
            tgui,
            to_check=["1", "2", "3", "4"],
            info={
                "file1_regions": file1_regions,
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_end_col": file1_end_col,
                "file2_regions": file2_regions,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file2_end_col": file2_end_col,
                "file3_regions": file3_regions,
                "file3_col_keys": file3_col_keys,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file3_end_col": file3_end_col,
                "file4_regions": file4_regions,
                "file4_col_keys": file4_col_keys,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
                "file4_end_col": file4_end_col,
            },
            rec_from=rec_from,
            rec_to=rec_to,
        )

        # File 5, reg_1, reg_2, reg_3, reg_4, reg_5, reg_6, min, median, triexp, biexp_event, mean, biexp_decay -----------------------------------------

        self.run_curve_fitting_analysis(tgui, "5")

        test_file5_col_keys = self.combine(
            [
                self.cf_keys("min", add_filename=True),
                self.cf_keys("median", add_filename=True),
                self.cf_keys("triexp", add_filename=True),
                self.cf_keys("biexp_event", add_filename=True),
                self.cf_keys("mean", add_filename=True),
                self.cf_keys("biexp_decay", add_filename=True),
            ]
        )

        file5_end_col = len(test_file5_col_keys)

        # need to calc num cols or hard code
        file5_col_keys, file5_filename = self.get_col_keys(
            tgui,
            0,
            file5_end_col,
            "curve_fitting",
            row=file4_end_row + 1,
            return_filename=True,
        )  # TODO: nicer way to this

        file5_regions = ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]
        assert file5_col_keys == test_file5_col_keys

        file5_start_row = file4_end_row + 2
        file5_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_curve_fitting_row_table(
            tgui,
            to_check=["1", "2", "3", "4", "5"],
            info={
                "file1_regions": file1_regions,
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_end_col": file1_end_col,
                "file2_regions": file2_regions,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file2_end_col": file2_end_col,
                "file3_regions": file3_regions,
                "file3_col_keys": file3_col_keys,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file3_end_col": file3_end_col,
                "file4_regions": file4_regions,
                "file4_col_keys": file4_col_keys,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
                "file4_end_col": file4_end_col,
                "file5_regions": file5_regions,
                "file5_col_keys": file5_col_keys,
                "file5_start_row": file5_start_row,
                "file5_end_row": file5_end_row,
                "file5_filename": file5_filename,
                "file5_end_col": file5_end_col,
            },
            rec_from=rec_from,
            rec_to=rec_to,
        )

    def cf_keys(self, analysis_type, add_filename=False):
        keys = {
            "min": ["record_num", "baseline", "min", "amplitude"],
            "max": ["record_num", "baseline", "max", "amplitude"],
            "mean": ["record_num", "baseline", "mean", "amplitude"],
            "area_under_curve_cf": [
                "record_num",
                "baseline",
                "area_under_curve_cf",
                "area_under_curve_cf_ms",
            ],
            "median": ["record_num", "baseline", "median", "amplitude"],
            "monoexp": [
                "record_num",
                "baseline",
                "peak",
                "amplitude",
                "b0",
                "b1",
                "tau",
                "r2",
            ],
            "biexp_decay": [
                "record_num",
                "baseline",
                "peak",
                "amplitude",
                "b0",
                "b1",
                "tau1",
                "b2",
                "tau2",
                "r2",
            ],
            "biexp_event": [
                "record_num",
                "b0",
                "b1",
                "fit_rise",
                "fit_decay",
                "r2",
                "baseline",
                "peak",
                "amplitude",
                "rise",
                "half_width",
                "decay_perc",
                "area_under_curve",
                "event_period",
                "max_rise",
                "max_decay",
                "monoexp_fit_b0",
                "monoexp_fit_b1",
                "monoexp_fit_tau",
                "events_monoexp_r2",
            ],
            "triexp": [
                "record_num",
                "baseline",
                "peak",
                "amplitude",
                "b0",
                "b1",
                "tau1",
                "b2",
                "tau2",
                "b3",
                "tau3",
                "r2",
            ],
        }
        results = keys[analysis_type]

        if add_filename:
            results = ["filename"] + results

        return results

    def combine(self, list_of_regions):
        return utils.flatten_list(list_of_regions)

    def check_curve_fitting_filenames(self, tgui, indexes, regions, filename):
        for idx, reg in zip(indexes, regions):
            assert tgui.mw.mw.table_tab_tablewidget.item(0, idx).data(0) == filename + " - " + "region " + reg[-1]

    def test_curve_fitting_row_wise_same_analyses_across_files_1(self):
        """
        Curve fitting shows only 1 header when all analysis are the same across files.
        """
        tgui = CurveFittingGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        self.set_row_or_column_display(tgui, "row")
        table = tgui.mw.mw.table_tab_tablewidget

        self.run_curve_fitting_analysis(tgui, "5")
        self.run_curve_fitting_analysis(tgui, "5")

        assert "region" in table.item(3, 0).data(0)
        assert table.item(4, 0).data(0) == ""
        assert "region" in table.item(5, 0).data(0)

    def test_curve_fitting_row_wise_same_analyses_across_files_2(self):
        """
        Curve fitting shows a header-per-file when the analysis are different across files.
        """
        tgui = CurveFittingGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        self.set_row_or_column_display(tgui, "row")
        table = tgui.mw.mw.table_tab_tablewidget

        self.run_curve_fitting_analysis(tgui, "3")
        self.run_curve_fitting_analysis(tgui, "3")

        assert "region" in table.item(5, 0).data(0)
        assert table.item(6, 0).data(0) == ""
        assert "region" in table.item(7, 0).data(0)

        self.run_curve_fitting_analysis(tgui, "3")

        assert "region" in table.item(11, 0).data(0)
        assert table.item(12, 0).data(0) == ""
        assert "region" in table.item(13, 0).data(0)

    @pytest.mark.parametrize("region", get_settings(SPEED, "curve_fitting_table"))
    @pytest.mark.parametrize(
        "analysis",
        [
            "min",
            "max",
            "mean",
            "median",
            "area_under_curve_cf",
            "monoexp",
            "biexp_decay",
            "biexp_event",
            "triexp",
        ],
    )
    def test_curve_fitting_summary_statistics(self, region, analysis):
        """
        Check summary statistics are calculated correctly for curve fitting analysis shown on table.
        """
        tgui = CurveFittingGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()

        tgui.switch_mw_tab(1)
        tgui.set_combobox(
            tgui.mw.mw.curve_fitting_summary_statistics_combobox,
            idx=int(region[-1]) - 1,
        )

        summary_statistics = self.run_curve_fitting_for_summary_statistics(
            tgui, region, analysis
        )  # this will run 4 files

        col_keys_union = summary_statistics["file_1"]["col_keys"]
        self.check_summary_statistics(
            tgui,
            col_keys_union,
            summary_statistics,
            ["file_1", "file_2", "file_3", "file_4"],
            "curve_fitting",
        )

    def run_curve_fitting_for_summary_statistics(self, tgui, region, analysis):
        """
        Run 4 curve fitting files and save the summary statistics for testing in summary statistics.
        """
        all_summary_statistics = {
            "file_1": {},
            "file_2": {},
            "file_3": {},
            "file_4": {},
        }
        for file_idx in range(4):
            self.reload_curve_fitting_file(tgui)

            if analysis == "biexp_event":
                tgui.mw.cfgs.events["max_slope"]["on"] = True
            tgui.set_fake_filename()

            tgui.switch_groupbox(tgui.mw.mw.curve_fitting_recs_to_analyse_groupbox, on=False)  # reset....

            if file_idx in [1, 3]:
                __, rec_from, rec_to = tgui.handle_analyse_specific_recs(True)
            else:
                rec_from = 0
                rec_to = tgui.adata.num_recs - 1

            self.run_default_curve_fitting(tgui, analysis, region)

            col_keys = self.cf_keys(analysis)

            results, __ = self.get_curve_fitting_table(
                tgui,
                rec_from=rec_from,
                rec_to=rec_to,
                regions=[region],
                region_keys=[col_keys],
                file_idx=file_idx,
            )
            num_recs = rec_to - rec_from + 1

            summary_statistics = {
                "data": {},
                "col_keys": col_keys,
                "file_filename": tgui.fake_filename + " - " + "region " + region[-1],
                "rec_from": rec_from,
                "rec_to": rec_to,
                "num_recs": num_recs,
            }

            summary_statistics = self.make_summary_statistics_from_column_data(
                results,
                summary_statistics,
                col_keys,
                rec_from,
                rec_to,
                num_recs,
                record_col_idx=0,
            )

            all_summary_statistics["file_" + str(file_idx + 1)] = summary_statistics

        return all_summary_statistics

    def check_row_table_against_curve_fitting(
        self,
        tgui,
        file_regions,
        file_col_keys,
        file_row_start,
        file_row_end,
        file_filename,
        file_end_col,
        rec_from,
        rec_to,
        file_idx,
    ):
        """
        Check the entries for the row-wise table view is correct for curve fitting entry.
        """
        filename_cols = [idx for (idx, key) in enumerate(file_col_keys) if key == "filename"]

        for col, reg in zip(filename_cols, file_regions):
            test_filename = file_filename + " - " + "region " + reg[-1]
            self.check_row_filenames(tgui, file_row_start, file_row_end, test_filename, filename_col=col)

        split_list = list(more_itertools.split_before(file_col_keys, lambda x: x == "filename"))

        stored_tabledata, __ = self.get_curve_fitting_table(
            tgui,
            rec_from=rec_from,
            rec_to=rec_to,
            regions=file_regions,
            region_keys=split_list,
            file_idx=file_idx,
        )

        data_on_table, text = self.get_table(
            tgui,
            row_start=file_row_start,
            row_end=file_row_end,
            col_start=0,
            col_end=file_end_col,
        )

        assert np.array_equal(data_on_table, stored_tabledata, equal_nan=True)

    def check_curve_fitting_row_table(self, tgui, to_check, info, rec_from=None, rec_to=None):
        """
        Convenience function to coordinate curve fitting row-table checks
        """
        if "1" in to_check:
            self.check_row_table_against_curve_fitting(
                tgui,
                info["file1_regions"],
                info["file1_col_keys"],
                info["file1_start_row"],
                info["file1_end_row"],
                info["file1_filename"],
                info["file1_end_col"],
                rec_from=0,
                rec_to=tgui.adata.num_recs - 1,
                file_idx=0,
            )
        if "2" in to_check:
            self.check_row_table_against_curve_fitting(
                tgui,
                info["file2_regions"],
                info["file2_col_keys"],
                info["file2_start_row"],
                info["file2_end_row"],
                info["file2_filename"],
                info["file2_end_col"],
                rec_from=rec_from,
                rec_to=rec_to,
                file_idx=1,
            )
        if "3" in to_check:
            self.check_row_table_against_curve_fitting(
                tgui,
                info["file3_regions"],
                info["file3_col_keys"],
                info["file3_start_row"],
                info["file3_end_row"],
                info["file3_filename"],
                info["file3_end_col"],
                rec_from=0,
                rec_to=tgui.adata.num_recs - 1,
                file_idx=2,
            )
        if "4" in to_check:
            self.check_row_table_against_curve_fitting(
                tgui,
                info["file4_regions"],
                info["file4_col_keys"],
                info["file4_start_row"],
                info["file4_end_row"],
                info["file4_filename"],
                info["file4_end_col"],
                rec_from=rec_from,
                rec_to=rec_to,
                file_idx=3,
            )
        if "5" in to_check:
            self.check_row_table_against_curve_fitting(
                tgui,
                info["file5_regions"],
                info["file5_col_keys"],
                info["file5_start_row"],
                info["file5_end_row"],
                info["file5_filename"],
                info["file5_end_col"],
                rec_from=rec_from,
                rec_to=rec_to,
                file_idx=4,
            )

    def reload_curve_fitting_file(self, tgui):
        tgui.update_curve_fitting_function(
            "across_records", "monoexp", tgui.time_type
        )  # order important this will overwrite analyse specific recs
        tgui.set_fake_filename()

    def run_curve_fitting_analysis(self, tgui, file_num):
        self.reload_curve_fitting_file(tgui)
        rec_from = rec_to = None
        tgui.switch_groupbox(tgui.mw.mw.curve_fitting_recs_to_analyse_groupbox, on=False)  # reset....

        if file_num == "1":
            for region_name, func_type in zip(["reg_1", "reg_3"], ["triexp", "area_under_curve_cf"]):
                self.run_default_curve_fitting(tgui, func_type, region_name)

        if file_num == "2":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(True)
            for region_name, func_type in zip(["reg_2", "reg_3"], ["min", "mean"]):
                self.run_default_curve_fitting(tgui, func_type, region_name)

        if file_num == "3":
            for region_name, func_type in zip(
                ["reg_1", "reg_2", "reg_3", "reg_4"],
                ["area_under_curve_cf", "monoexp", "biexp_event", "biexp_decay"],
            ):
                self.run_default_curve_fitting(tgui, func_type, region_name)

        if file_num == "4":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(True)
            for region_name, func_type in zip(["reg_5", "reg_6"], ["median", "triexp"]):
                self.run_default_curve_fitting(tgui, func_type, region_name)

        if file_num == "5":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(True)
            for region_name, func_type in zip(
                ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"],
                ["min", "median", "triexp", "biexp_event", "mean", "biexp_decay"],
            ):
                self.run_default_curve_fitting(tgui, func_type, region_name)

        if rec_from:
            return rec_from, rec_to

    def run_default_curve_fitting(self, tgui, func_type, region_name):
        test_curve_fitting.setup_and_run_curve_fitting_analysis(
            tgui,
            func_type=func_type,
            region_name=region_name,
            rec_from=0,
            rec_to=tgui.adata.num_recs,  #
            set_options_only=False,
        )

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def handle_grouped_analysis(self, tgui, group, rec_from=None, rec_to=None):
        if group != "all":
            idx_to_click = tgui.set_groups_and_delete_adata_of_all_non_group(
                tgui, "threshold", group, run=False, rec_from=rec_from, rec_to=rec_to
            )[0]
        else:
            idx_to_click = None
        tgui.switch_mw_tab(1)  # switch back to table for next table analysis
        return idx_to_click

    @pytest.mark.parametrize("group", ["all", "1", "3"])
    def test_events_per_column_table(self, group):
        """
        Test the table displayed stored_tabledata correctly in 'Column mode'. Load 5 files with batch mode on
        and check the most recently loaded file.

        Check the headers are expected. For Events, the columns shown are only the analysed parameters per-file. The optional parameters are fit type and
        max slope on / off, of which different combinations are tested here. Note that these tests only check the most recently analysed files
        (unlike row-rise tests which check all plotted files).

        The tests work by calculating the start / finish column per file and comparing each file against the stored tabledata individually.

        NOTE / TODO: the rec to / rec from for analysis without analyse_specific_recs is hard-coded.

        Then switch to summary statistics to check these are calculated / down correctly. The summary statistics are calculated
        at the same time the column data is checked. Many tests are combined into one to avoid repeating the long time it takes
        to re-run analysis for each test, however it does make these tests quite heavy.
        """
        tgui = EventsGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()

        summary_statistics = {
            "file_1": {},
            "file_2": {},
            "file_3": {},
            "file_4": {},
            "file_5": {},
        }  # below, the order is fixed and should never change (it will if there is aproblem in script explain)
        summary_stats_headers = [
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "max_rise",
            "max_decay",
            "monoexp_fit_b0",
            "monoexp_fit_b1",
            "monoexp_fit_tau",
            "events_monoexp_r2",
            "biexp_fit_b0",
            "biexp_fit_b1",
            "biexp_fit_rise",
            "biexp_fit_decay",
            "events_biexp_r2",
        ]

        # File 1, max slope off, monoexp on --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_events_test_file(tgui, "1")
        idx_to_click = self.handle_grouped_analysis(tgui, group)

        file1_start_col = 0
        file1_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file1_col_keys = self.get_col_keys(tgui, file1_start_col, file1_end_col, "events", row=1)

        assert file1_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "monoexp_fit_b0",
            "monoexp_fit_b1",
            "monoexp_fit_tau",
            "events_monoexp_r2",
        ]

        summary_statistics["file_1"] = self.check_column_table_against_events(
            tgui,
            file1_col_keys,
            file1_start_col,
            file1_end_col,
            file_idx=0,
            rec_from=0,
            rec_to=13,
            group=group,
            idx_to_click=idx_to_click,
        )

        self.check_summary_statistics(tgui, file1_col_keys, summary_statistics, ["file_1"], "events", group)

        # File 2, max slope on, biexp on --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_events_test_file(tgui, "2")

        idx_to_click = self.handle_grouped_analysis(tgui, group, rec_from=rec_from, rec_to=rec_to)

        file2_start_col = self.get_col_file_start_indexes(tgui)[1]
        file2_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file2_col_keys = self.get_col_keys(tgui, file2_start_col, file2_end_col, "events", row=1)

        assert file2_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "max_rise",
            "max_decay",
            "biexp_fit_b0",
            "biexp_fit_b1",
            "biexp_fit_rise",
            "biexp_fit_decay",
            "events_biexp_r2",
        ]

        summary_statistics["file_2"] = self.check_column_table_against_events(
            tgui,
            file2_col_keys,
            file2_start_col,
            file2_end_col,
            file_idx=1,
            rec_from=rec_from,
            rec_to=rec_to,
            group=group,
            idx_to_click=idx_to_click,
        )

        self.check_summary_statistics(
            tgui,
            summary_stats_headers,
            summary_statistics,
            ["file_1", "file_2"],
            "events",
            group,
        )

        # File 3, ma slope on, no fit --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_events_test_file(tgui, "3")
        idx_to_click = self.handle_grouped_analysis(tgui, group)

        file3_start_col = self.get_col_file_start_indexes(tgui)[2]
        file3_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file3_col_keys = self.get_col_keys(tgui, file3_start_col, file3_end_col, "events", row=1)

        assert file3_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "max_rise",
            "max_decay",
        ]

        summary_statistics["file_3"] = self.check_column_table_against_events(
            tgui,
            file3_col_keys,
            file3_start_col,
            file3_end_col,
            file_idx=2,
            rec_from=0,
            rec_to=13,
            group=group,
            idx_to_click=idx_to_click,
        )

        self.check_summary_statistics(
            tgui,
            summary_stats_headers,
            summary_statistics,
            ["file_1", "file_2", "file_3"],
            "events",
            group,
        )

        # File 4, max slope off, biexp on --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_events_test_file(tgui, "4")
        idx_to_click = self.handle_grouped_analysis(tgui, group)

        file4_start_col = self.get_col_file_start_indexes(tgui)[3]
        file4_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file4_col_keys = self.get_col_keys(tgui, file4_start_col, file4_end_col, "events", row=1)

        assert file4_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "biexp_fit_b0",
            "biexp_fit_b1",
            "biexp_fit_rise",
            "biexp_fit_decay",
            "events_biexp_r2",
        ]

        summary_statistics["file_4"] = self.check_column_table_against_events(
            tgui,
            file4_col_keys,
            file4_start_col,
            file4_end_col,
            file_idx=3,
            rec_from=0,
            rec_to=0,
            group=group,
            idx_to_click=idx_to_click,
        )

        self.check_summary_statistics(
            tgui,
            summary_stats_headers,
            summary_statistics,
            ["file_1", "file_2", "file_3", "file_4"],
            "events",
            group,
        )

        # File 5, max slope on, monoexp on --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_events_test_file(tgui, "5")
        idx_to_click = self.handle_grouped_analysis(tgui, group, rec_from=rec_from, rec_to=rec_to)

        file5_start_col = self.get_col_file_start_indexes(tgui)[4]
        file5_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file5_col_keys = self.get_col_keys(tgui, file5_start_col, file5_end_col, "events", row=1)

        assert file5_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "max_rise",
            "max_decay",
            "monoexp_fit_b0",
            "monoexp_fit_b1",
            "monoexp_fit_tau",
            "events_monoexp_r2",
        ]

        summary_statistics["file_5"] = self.check_column_table_against_events(
            tgui,
            file5_col_keys,
            file5_start_col,
            file5_end_col,
            file_idx=4,
            rec_from=rec_from,
            rec_to=rec_to,
            group=group,
            idx_to_click=idx_to_click,
        )

        self.check_summary_statistics(
            tgui,
            summary_stats_headers,
            summary_statistics,
            ["file_1", "file_2", "file_3", "file_4", "file_5"],
            "events",
            group,
        )

    @pytest.mark.parametrize("group", ["all", "1", "3"])
    def test_events_per_row_table(self, group):
        """
        Test the events results are displayed correctly in row mode. See test_events_per_column_table() for details,
        very similar implementation except the row start / stop per file is first calculated and
        tested per-file.

        Summary statistics are tested in column mode only as the test values are calculated at the same time the column
        data is checked.

        TODO: This test is slightly better than column test as it checks every file is updated correctly after each loaded file)
        """
        group = "1"

        tgui = EventsGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        self.set_row_or_column_display(tgui, "row")

        # File 1, max slope off, monoexp on --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_events_test_file(tgui, to_run="1")
        file1_idx_to_click = self.handle_grouped_analysis(tgui, group)

        file1_col_keys, file1_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "events",
            row=0,
            return_filename=True,
        )

        assert file1_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "monoexp_fit_b0",
            "monoexp_fit_b1",
            "monoexp_fit_tau",
            "events_monoexp_r2",
        ]

        file1_start_row = 1
        file1_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_events_row_table(
            tgui,
            to_check=["1"],
            group=group,
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_idx_to_click": file1_idx_to_click,
            },
        )

        # File 2, max slope on, biexp on --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_events_test_file(tgui, to_run="2")
        file2_idx_to_click = self.handle_grouped_analysis(tgui, group, rec_from, rec_to)

        file2_col_keys, file2_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "events",
            row=0,
            return_filename=True,
        )

        assert file2_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "max_rise",
            "max_decay",
            "monoexp_fit_b0",
            "monoexp_fit_b1",
            "monoexp_fit_tau",
            "events_monoexp_r2",
            "biexp_fit_b0",
            "biexp_fit_b1",
            "biexp_fit_rise",
            "biexp_fit_decay",
            "events_biexp_r2",
        ]

        file2_start_row = file1_end_row + 1
        file2_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_events_row_table(
            tgui,
            to_check=["1", "2"],
            group=group,
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_idx_to_click": file1_idx_to_click,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file2_idx_to_click": file2_idx_to_click,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
        )

        # File 3, max slope on, no fit --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_events_test_file(tgui, to_run="3")
        file3_idx_to_click = self.handle_grouped_analysis(tgui, group)

        file3_col_keys, file3_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "events",
            row=0,
            return_filename=True,
        )

        assert file3_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "max_rise",
            "max_decay",
            "monoexp_fit_b0",
            "monoexp_fit_b1",
            "monoexp_fit_tau",
            "events_monoexp_r2",
            "biexp_fit_b0",
            "biexp_fit_b1",
            "biexp_fit_rise",
            "biexp_fit_decay",
            "events_biexp_r2",
        ]

        file3_start_row = file2_end_row + 1
        file3_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_events_row_table(
            tgui,
            to_check=["1", "2", "3"],
            group=group,
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_idx_to_click": file1_idx_to_click,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file2_idx_to_click": file2_idx_to_click,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file3_idx_to_click": file3_idx_to_click,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
        )

        # File 4, max slope off, biexp on --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_events_test_file(tgui, to_run="4")
        file4_idx_to_click = self.handle_grouped_analysis(tgui, group)

        file4_col_keys, file4_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "events",
            row=0,
            return_filename=True,
        )

        assert file4_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "max_rise",
            "max_decay",
            "monoexp_fit_b0",
            "monoexp_fit_b1",
            "monoexp_fit_tau",
            "events_monoexp_r2",
            "biexp_fit_b0",
            "biexp_fit_b1",
            "biexp_fit_rise",
            "biexp_fit_decay",
            "events_biexp_r2",
        ]

        file4_start_row = file3_end_row + 1
        file4_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_events_row_table(
            tgui,
            to_check=["1", "2", "3", "4"],
            group=group,
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_idx_to_click": file1_idx_to_click,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file2_idx_to_click": file2_idx_to_click,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file3_idx_to_click": file3_idx_to_click,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
                "file4_idx_to_click": file4_idx_to_click,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
        )

        # File 5, max slope on, monoexp on --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_events_test_file(tgui, to_run="5")
        file5_idx_to_click = self.handle_grouped_analysis(tgui, group, rec_from, rec_to)

        file5_col_keys, file5_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "events",
            row=0,
            return_filename=True,
        )

        assert file5_col_keys == [
            "event_num",
            "template_num",
            "record_num",
            "event_time",
            "baseline",
            "peak",
            "amplitude",
            "rise",
            "half_width",
            "decay_perc",
            "area_under_curve",
            "event_period",
            "max_rise",
            "max_decay",
            "monoexp_fit_b0",
            "monoexp_fit_b1",
            "monoexp_fit_tau",
            "events_monoexp_r2",
            "biexp_fit_b0",
            "biexp_fit_b1",
            "biexp_fit_rise",
            "biexp_fit_decay",
            "events_biexp_r2",
        ]

        file5_start_row = file4_end_row + 1
        file5_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_events_row_table(
            tgui,
            to_check=["1", "2", "3", "4", "5"],
            group=group,
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file1_idx_to_click": file1_idx_to_click,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file2_idx_to_click": file2_idx_to_click,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file3_idx_to_click": file3_idx_to_click,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
                "file4_idx_to_click": file4_idx_to_click,
                "file5_start_row": file5_start_row,
                "file5_end_row": file5_end_row,
                "file5_filename": file5_filename,
                "file5_idx_to_click": file5_idx_to_click,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
        )

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Run Events
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run_events_test_file(self, tgui, to_run):
        """
        Run events analysis for each file. There is a lot of hard-coding here, each file is specifically
        chosen to alternate a range of parameters analysed and analysed speciifc recs. e.g. see check_column_table_against_events()
        If wanting to test new parameter combinations, add new files.

        The files run are:

        File 1, max slope off, monoexp on
        File 2, max slope on, biexp on, analyse specific recs
        File 3, max slope on, no fit
        File 4, max slope off, biexp on
        File 5, max slope on, monoexp on, analyse specific recs
        """
        multi_rec = "events_multi_record_table" if to_run in ["1", "2", "3", "5"] else "events_one_record"
        tgui.setup_artificial_data("cumulative", analysis_type=multi_rec)
        tgui.update_events_to_varying_amplitude_and_tau()
        tgui.set_fake_filename()
        rec_from = rec_to = None

        tgui.mw.mw.actionEvents_Analyis_Options.trigger()
        options_dialog = tgui.mw.dialogs["events_analysis_options"]
        self.reset_events_options(tgui, options_dialog)

        if to_run == "1":
            tgui.run_artificial_events_analysis(tgui, "threshold")  # assumes defaults

        elif to_run == "2":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=True)
            tgui.setup_max_slope_events(tgui)
            tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 1)
            tgui.run_artificial_events_analysis(tgui, "threshold", biexp=True)

        elif to_run == "3":
            tgui.setup_max_slope_events(tgui)
            tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 2)
            tgui.run_artificial_events_analysis(tgui, "threshold")

        elif to_run == "4":
            tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 1)
            tgui.run_artificial_events_analysis(tgui, "threshold", biexp=True)

        elif to_run == "5":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=True)
            tgui.setup_max_slope_events(tgui)
            tgui.set_combobox(options_dialog.dia.event_fit_method_combobox, 0)
            tgui.run_artificial_events_analysis(tgui, "threshold", biexp=False)

        if rec_from:
            return rec_from, rec_to

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Check Events Column and Row
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def check_column_table_against_events(
        self,
        tgui,
        file_col_keys,
        file_start_col,
        file_end_col,
        file_idx,
        rec_to,
        rec_from,
        group="all",
        idx_to_click=None,
    ):
        """
        Check the table data in column mode is correct. Check the filename shown is correct,
        then get data as shown on table and the stored tabledata.

        Stored tabledata contains all parameters, so we need to delete the columns for the non-analysed parameters
        (which are nan). This is another factor that means analysis / files are hard-coded (see run_events_test_file())
        """
        assert self.get_item(tgui, 0, file_start_col) == tgui.fake_filename

        data_on_table, __ = self.get_and_cut_col_data_on_table(tgui, file_start_col, file_end_col)

        stored_tabledata = tgui.mw.stored_tabledata.event_info[file_idx]
        stored_tabledata = tgui.reshape_events_into_table(stored_tabledata, group)

        if file_idx == 0:
            stored_tabledata = np.delete(stored_tabledata, (12, 13, 22, 21, 20, 19, 18), axis=1)
        elif file_idx == 1:
            stored_tabledata = np.delete(stored_tabledata, (14, 15, 16, 17), axis=1)
        elif file_idx == 2:
            stored_tabledata = np.delete(stored_tabledata, (14, 15, 16, 17, 22, 21, 20, 19, 18), axis=1)
        elif file_idx == 3:
            stored_tabledata = np.delete(stored_tabledata, (12, 13, 14, 15, 16, 17), axis=1)
        elif file_idx == 4:
            stored_tabledata = np.delete(stored_tabledata, (22, 21, 20, 19, 18), axis=1)

        if idx_to_click is not None:
            event_nums = np.array(np.sort(idx_to_click)) + 1
            stored_tabledata[:, 0] = event_nums

        assert np.array_equal(data_on_table, stored_tabledata, equal_nan=True), "col events: idx {0}".format(file_idx)

        if group == "all":
            assert np.min(data_on_table[:, 2]) == rec_from + 1, "col rec_from events: idx {0}".format(file_idx)
            assert np.max(data_on_table[:, 2]) == rec_to + 1, "col rec_to events: idx {0}".format(file_idx)

        # Calc. Summary statistics. Due to fencepost the two methods of calculating
        # num recs (one for absolute, one for indexing) don't match so calculate
        # twice.
        num_recs = rec_to - rec_from + 1
        alt_num_recs = len(np.unique(data_on_table[:, 2]))

        summary_statistics = {
            "data": {},
            "col_keys": file_col_keys,
            "file_filename": tgui.fake_filename,
            "rec_from": rec_from,
            "rec_to": rec_to,
            "num_recs": alt_num_recs,
        }  # num_recs

        summary_statistics = self.make_summary_statistics_from_column_data(
            data_on_table,
            summary_statistics,
            file_col_keys,
            rec_from,
            rec_to,
            alt_num_recs,
            record_col_idx=2,
            group=group,
        )

        return summary_statistics

    def check_row_table_against_events(
        self,
        tgui,
        file_row_start,
        file_row_end,
        file_filename,
        rec_from,
        rec_to,
        file_idx=0,
        group="all",
        idx_to_click=None,
    ):
        """
        Checking function for row, see check_column_table_against_events() for details, very similar.
        """
        self.set_row_or_column_display(tgui, "row")

        self.check_row_filenames(tgui, file_row_start, file_row_end, file_filename)
        data_on_table, text = self.get_table(
            tgui,
            row_start=file_row_start,
            row_end=file_row_end,
            col_start=1,
            col_end=tgui.mw.mw.table_tab_tablewidget.columnCount(),
        )

        stored_tabledata = tgui.mw.stored_tabledata.event_info[file_idx]
        events_data = tgui.reshape_events_into_table(stored_tabledata, group)

        if file_idx == 0 and data_on_table.shape[1] != 23:
            events_data = np.delete(events_data, (12, 13, 22, 21, 20, 19, 18), axis=1)  # delete unanalysed cols

        if idx_to_click is not None:
            event_nums = np.array(np.sort(idx_to_click)) + 1
            events_data[:, 0] = event_nums

        assert np.array_equal(data_on_table, events_data, equal_nan=True), "row events idx: {0}".format(file_idx)
        if group == "all":
            assert np.min(data_on_table[:, 2]) == rec_from + 1, "row rec_from events idx: {0}".format(file_idx)
            assert np.max(data_on_table[:, 2]) == rec_to + 1, "row rec_to events idx: {0}".format(file_idx)

    def check_events_row_table(self, tgui, to_check, group, info):
        """
        Convenience function for coordinating checking rows, see run_events_test_file() for hard coding.
        """
        if "1" in to_check:
            self.check_row_table_against_events(
                tgui,
                info["file1_start_row"],
                info["file1_end_row"],
                info["file1_filename"],
                rec_from=0,
                rec_to=13,
                file_idx=0,
                group=group,
                idx_to_click=info["file1_idx_to_click"],
            )
        if "2" in to_check:
            self.check_row_table_against_events(
                tgui,
                info["file2_start_row"],
                info["file2_end_row"],
                info["file2_filename"],
                rec_from=info["rec_from"],
                rec_to=info["rec_to"],
                file_idx=1,
                group=group,
                idx_to_click=info["file2_idx_to_click"],
            )
        if "3" in to_check:
            self.check_row_table_against_events(
                tgui,
                info["file3_start_row"],
                info["file3_end_row"],
                info["file3_filename"],
                rec_from=0,
                rec_to=13,
                file_idx=2,
                group=group,
                idx_to_click=info["file3_idx_to_click"],
            )
        if "4" in to_check:
            self.check_row_table_against_events(
                tgui,
                info["file4_start_row"],
                info["file4_end_row"],
                info["file4_filename"],
                rec_from=0,
                rec_to=0,
                file_idx=3,
                group=group,
                idx_to_click=info["file4_idx_to_click"],
            )
        if "5" in to_check:
            self.check_row_table_against_events(
                tgui,
                info["file5_start_row"],
                info["file5_end_row"],
                info["file5_filename"],
                rec_from=info["rec_from"],
                rec_to=info["rec_to"],
                file_idx=4,
                group=group,
                idx_to_click=info["file5_idx_to_click"],
            )

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Skinetics
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def test_skinetics_per_column_table(self):
        """
        See test_events_per_column_table()

        TODO: this test runs very slowly because there are 75 recs, each with many spikes. Reduce number of recs for this test.
        """
        tgui = SpkcntTGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        summary_statistics = {
            "file_1": {},
            "file_2": {},
            "file_3": {},
            "file_4": {},
            "file_5": {},
        }

        # File 1 --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_skinetics_test_file(tgui, "1")

        file1_start_col = 0
        file1_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file1_col_keys = self.get_col_keys(tgui, file1_start_col, file1_end_col, "skinetics", row=1)

        assert file1_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
        ]

        self.check_skinetics_column_table(
            tgui,
            ["1"],
            file1_col_keys,
            file1_start_col,
            file1_end_col,
            summary_statistics,
            rec_to=rec_to,
            rec_from=rec_from,
        )

        self.check_summary_statistics(tgui, file1_col_keys, summary_statistics, ["file_1"], "skinetics")

        # File 2 --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_skinetics_test_file(tgui, "2")

        file2_start_col = self.get_col_file_start_indexes(tgui)[1]
        file2_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file2_col_keys = self.get_col_keys(tgui, file2_start_col, file2_end_col, "skinetics", row=1)

        assert file2_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
            "max_rise",
            "max_decay",
        ]

        self.check_skinetics_column_table(
            tgui,
            ["2"],
            file2_col_keys,
            file2_start_col,
            file2_end_col,
            summary_statistics,
            rec_to=rec_to,
            rec_from=rec_from,
        )

        self.check_summary_statistics(tgui, file2_col_keys, summary_statistics, ["file_1", "file_2"], "skinetics")

        # File 3 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_skinetics_test_file(tgui, "3")

        file3_start_col = self.get_col_file_start_indexes(tgui)[2]
        file3_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file3_col_keys = self.get_col_keys(tgui, file3_start_col, file3_end_col, "skinetics", row=1)

        assert file3_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
            "max_rise",
            "max_decay",
        ]

        self.check_skinetics_column_table(
            tgui,
            ["3"],
            file3_col_keys,
            file3_start_col,
            file3_end_col,
            summary_statistics,
        )

        self.check_summary_statistics(
            tgui,
            file2_col_keys,
            summary_statistics,
            ["file_1", "file_2", "file_3"],
            "skinetics",
        )

        # File 4 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_skinetics_test_file(tgui, "4")

        file4_start_col = self.get_col_file_start_indexes(tgui)[3]
        file4_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file4_col_keys = self.get_col_keys(tgui, file4_start_col, file4_end_col, "skinetics", row=1)

        assert file4_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
        ]

        self.check_skinetics_column_table(
            tgui,
            ["4"],
            file4_col_keys,
            file4_start_col,
            file4_end_col,
            summary_statistics,
        )

        self.check_summary_statistics(
            tgui,
            file2_col_keys,
            summary_statistics,
            ["file_1", "file_2", "file_3", "file_4"],
            "skinetics",
        )

        # File 5 --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_skinetics_test_file(tgui, "5")

        file5_start_col = self.get_col_file_start_indexes(tgui)[4]
        file5_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file5_col_keys = self.get_col_keys(tgui, file5_start_col, file5_end_col, "skinetics", row=1)

        assert file5_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
            "max_rise",
            "max_decay",
        ]

        self.check_skinetics_column_table(
            tgui,
            ["5"],
            file5_col_keys,
            file5_start_col,
            file5_end_col,
            summary_statistics,
            rec_to=rec_to,
            rec_from=rec_from,
        )

        self.check_summary_statistics(
            tgui,
            file2_col_keys,
            summary_statistics,
            ["file_1", "file_2", "file_3", "file_4", "file_5"],
            "skinetics",
        )

    def test_skinetics_per_row_table(self):
        """
        See test_events_per_row_table()
        """
        tgui = SpkcntTGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        self.set_row_or_column_display(tgui, "row")

        # File 1 --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_skinetics_test_file(tgui, "1")

        file1_col_keys, file1_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "skinetics",
            row=0,
            return_filename=True,
        )

        assert file1_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
        ]

        file1_start_row = 1
        file1_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_skinetics_row_table(
            tgui,
            rec_from,
            rec_to,
            to_check=["1"],
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
            },
        )

        # File 2 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_skinetics_test_file(tgui, "2")

        file2_col_keys, file2_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "skinetics",
            row=0,
            return_filename=True,
        )

        assert file2_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
            "max_rise",
            "max_decay",
        ]

        file2_start_row = file1_end_row + 1
        file2_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_skinetics_row_table(
            tgui,
            rec_from,
            rec_to,
            to_check=["1", "2"],
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
            },
        )

        # File 3 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_skinetics_test_file(tgui, "3")

        file3_col_keys, file3_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "skinetics",
            row=0,
            return_filename=True,
        )

        assert file3_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
            "max_rise",
            "max_decay",
        ]

        file3_start_row = file2_end_row + 1
        file3_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_skinetics_row_table(
            tgui,
            rec_from,
            rec_to,
            to_check=["1", "2", "3"],
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
            },
        )

        # File 4 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_skinetics_test_file(tgui, "4")

        file4_col_keys, file4_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "skinetics",
            row=0,
            return_filename=True,
        )

        assert file4_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
            "max_rise",
            "max_decay",
        ]

        file4_start_row = file3_end_row + 1
        file4_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_skinetics_row_table(
            tgui,
            rec_from,
            rec_to,
            to_check=["1", "2", "3", "4"],
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
            },
        )

        # File 5 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_skinetics_test_file(tgui, "5")

        file5_col_keys, file5_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "skinetics",
            row=0,
            return_filename=True,
        )

        assert file5_col_keys == [
            "record_num",
            "spike_number",
            "spike_time",
            "peak",
            "amplitude",
            "thr",
            "rise_time",
            "decay_time",
            "fwhm",
            "fahp",
            "mahp",
            "max_rise",
            "max_decay",
        ]

        file5_start_row = file4_end_row + 1
        file5_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_skinetics_row_table(
            tgui,
            rec_from,
            rec_to,
            to_check=["1", "2", "3", "4", "5"],
            info={
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
                "file5_start_row": file5_start_row,
                "file5_end_row": file5_end_row,
                "file5_filename": file5_filename,
            },
        )

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Run Skinetics
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run_skinetics_test_file(self, tgui, to_run):
        """
        See run_events_test_file(). Interp is on to allow more variability in
        calculated parameters for summary stats calculation.

        For skinetics, the only option parameter is max_slope on vs. off

        File 1 max slope off, analyse specific recs
        File 2 max sloep on, analyse specific recs
        File 3 max slope on
        File 4 max slope off
        File 5 max sloep on, analyse specific recs
        """
        norm_or_cum = "normalised" if to_run in ["1", "5"] else "cumulative"
        rec_from, rec_to = self.setup_artificial_spkcnt(tgui, norm_or_cum, "skinetics_table")

        if to_run == "1":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=True)
            tgui.run_artificial_skinetics_analysis(interp=True)

        if to_run == "2":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=True)
            tgui.run_artificial_skinetics_analysis(max_slope={"n_samples_rise": 2, "n_samples_decay": 3}, interp=True)

        if to_run == "3":
            tgui.run_artificial_skinetics_analysis(max_slope={"n_samples_rise": 2, "n_samples_decay": 3}, interp=True)

        if to_run == "4":
            tgui.run_artificial_skinetics_analysis(interp=True)

        if to_run == "5":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=True)
            tgui.run_artificial_skinetics_analysis(max_slope={"n_samples_rise": 2, "n_samples_decay": 3}, interp=True)

        if rec_to:
            return rec_from, rec_to

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Check Skinetics Row and Column
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Check Column -----------------------------------------------------------------------------------------------------------------------------------------------

    def check_skinetics_column_table(
        self,
        tgui,
        to_check,
        file_col_keys,
        file_start_col,
        file_end_col,
        summary_statistics,
        rec_from=None,
        rec_to=None,
    ):
        """
        Convenience function for checking skinetics column talbe is correct. Analyse specific options are hard coded based on the number of
        recs in the file.
        """
        assert self.get_item(tgui, 0, file_start_col == tgui.fake_filename)

        if "1" in to_check:
            summary = self.check_column_table_against_skinetics(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                file_idx=0,
                rec_to=rec_to,
                rec_from=rec_from,
                max_slope=False,
            )
            summary_statistics["file_1"] = summary

        elif "2" in to_check:
            summary = self.check_column_table_against_skinetics(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                file_idx=1,
                rec_to=rec_to,
                rec_from=rec_from,
            )
            summary_statistics["file_2"] = summary

        elif "3" in to_check:
            summary = self.check_column_table_against_skinetics(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                file_idx=2,
                rec_from=0,
                rec_to=tgui.adata.num_recs - 1,
            )
            summary_statistics["file_3"] = summary

        elif "4" in to_check:
            summary = self.check_column_table_against_skinetics(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                file_idx=3,
                rec_from=0,
                rec_to=tgui.adata.num_recs - 1,
                max_slope=False,
            )
            summary_statistics["file_4"] = summary

        elif "5" in to_check:
            summary = self.check_column_table_against_skinetics(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                file_idx=4,
                rec_to=rec_to,
                rec_from=rec_from,
            )
            summary_statistics["file_5"] = summary

    def check_column_table_against_skinetics(
        self,
        tgui,
        file_col_keys,
        file_start_col,
        file_end_col,
        file_idx,
        rec_to,
        rec_from,
        max_slope=True,
    ):
        """

        See events version of this function
        """
        data_on_table, text = self.get_and_cut_col_data_on_table(tgui, file_start_col, file_end_col)

        stored_tabledata = tgui.mw.stored_tabledata.skinetics_data[file_idx]
        skinetics_data = tgui.reshape_skinetics_data_into_table(stored_tabledata)

        if not max_slope:
            skinetics_data = skinetics_data[:, :-2]

        assert np.array_equal(data_on_table, skinetics_data, equal_nan=True), "col skinetics idx: "
        assert np.min(data_on_table[:, 0]) == rec_from + 1, "rec_from skinetics idx: "
        assert np.max(data_on_table[:, 0]) == rec_to + 1, "rec_to skinetics idx: "

        # Summary Statistics

        # account for difference when analyse-specific-recs on / off
        num_recs = rec_to - rec_from + 1
        alt_num_recs = len(np.unique(data_on_table[:, 0]))

        summary_statistics = {
            "data": {},
            "col_keys": file_col_keys,
            "file_filename": tgui.fake_filename,
            "rec_from": rec_from,
            "rec_to": rec_to,
            "num_recs": num_recs,
        }

        summary_statistics = self.make_summary_statistics_from_column_data(
            data_on_table,
            summary_statistics,
            file_col_keys,
            rec_from,
            rec_to,
            alt_num_recs,
            record_col_idx=0,
        )

        return summary_statistics

    # Check Row -----------------------------------------------------------------------------------------------------------------------------------------------

    def check_skinetics_row_table(self, tgui, rec_from, rec_to, to_check, info):
        """
        Same idea as events version
        """
        if "1" in to_check:
            max_slope = False if to_check == ["1"] else True
            self.check_row_table_against_stored_tabledata_skinetics(
                tgui,
                info["file1_start_row"],
                info["file1_end_row"],
                info["file1_filename"],
                rec_from,
                rec_to,
                file_idx=0,
                max_slope=max_slope,
            )
        if "2" in to_check:
            self.check_row_table_against_stored_tabledata_skinetics(
                tgui,
                info["file2_start_row"],
                info["file2_end_row"],
                info["file2_filename"],
                rec_from,
                rec_to,
                file_idx=1,
            )
        if "3" in to_check:
            self.check_row_table_against_stored_tabledata_skinetics(
                tgui,
                info["file3_start_row"],
                info["file3_end_row"],
                info["file3_filename"],
                rec_from=0,
                rec_to=tgui.adata.num_recs - 1,
                file_idx=2,
            )
        if "4" in to_check:
            self.check_row_table_against_stored_tabledata_skinetics(
                tgui,
                info["file4_start_row"],
                info["file4_end_row"],
                info["file4_filename"],
                rec_from=0,
                rec_to=tgui.adata.num_recs - 1,
                file_idx=3,
            )
        if "5" in to_check:
            self.check_row_table_against_stored_tabledata_skinetics(
                tgui,
                info["file5_start_row"],
                info["file5_end_row"],
                info["file5_filename"],
                rec_from,
                rec_to,
                file_idx=4,
            )

    def check_row_table_against_stored_tabledata_skinetics(
        self,
        tgui,
        file_row_start,
        file_row_end,
        file_filename,
        rec_from,
        rec_to,
        file_idx,
        max_slope=True,
    ):
        """
        Same idea as events, if max slope is not calculated the columns must be cut from the stored tabledata for comparison (which are NaN).
        """
        self.set_row_or_column_display(tgui, "row")
        self.check_row_filenames(tgui, file_row_start, file_row_end, file_filename)

        data_on_table, text = self.get_table(
            tgui,
            row_start=file_row_start,
            row_end=file_row_end,
            col_start=1,
            col_end=tgui.mw.mw.table_tab_tablewidget.columnCount(),
        )

        stored_tabledata = tgui.mw.stored_tabledata.skinetics_data[file_idx]
        stored_tabledata = tgui.reshape_skinetics_data_into_table(stored_tabledata)

        if not max_slope:
            stored_tabledata = stored_tabledata[:, :-2]

        assert np.array_equal(data_on_table, stored_tabledata, equal_nan=True), "row skinetics idx: {0}".format(
            file_idx
        )
        assert np.min(data_on_table[:, 0]) == rec_from + 1, "rec_from skinetics idx: {0}".format(file_idx)
        assert np.max(data_on_table[:, 0]) == rec_to + 1, "rec_to skinetics idx: {0}".format(file_idx)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Ri
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def test_input_resistance_per_column_table(self):
        """
        Input resistance checks on column. see test_events_per_column_table().
        """
        tgui = SpkcntTGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        summary_statistics = {
            "file_1": {},
            "file_2": {},
            "file_3": {},
            "file_4": {},
            "file_5": {},
        }

        # File 1 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_input_resistance_test_file(tgui, to_run="1")

        file1_start_col = 0
        file1_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file1_col_keys = self.get_col_keys(tgui, file1_start_col, file1_end_col, "spkcnt_and_input_resistance", row=1)

        assert file1_col_keys == [
            "record_num",
            "user_input_im",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "input_resistance",
        ]

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["1"],
            file1_col_keys,
            file1_start_col,
            file1_end_col,
            summary_statistics,
            analysis_type="Ri",
        )

        self.check_summary_statistics(tgui, file1_col_keys, summary_statistics, ["file_1"], "Ri")

        # File 2 --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_input_resistance_test_file(tgui, to_run="2")

        file2_start_col = self.get_col_file_start_indexes(tgui)[1]
        file2_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file2_col_keys = self.get_col_keys(tgui, file2_start_col, file2_end_col, "spkcnt_and_input_resistance", row=1)

        summary_stats_col_keys = self.check_input_resistance_stats_col_keys(tgui, file1_start_col, file2_end_col)
        assert file2_col_keys == [
            "record_num",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "input_resistance",
            "sag_hump_peaks",
            "sag_hump_ratio",
        ]

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["2"],
            file2_col_keys,
            file2_start_col,
            file2_end_col,
            summary_statistics,
            analysis_type="Ri",
            rec_to=rec_to,
            rec_from=rec_from,
        )

        self.check_summary_statistics(tgui, summary_stats_col_keys, summary_statistics, ["file_1", "file_2"], "Ri")

        # File 3 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_input_resistance_test_file(tgui, to_run="3")

        file3_start_col = self.get_col_file_start_indexes(tgui)[2]
        file3_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file3_col_keys = self.get_col_keys(tgui, file3_start_col, file3_end_col, "spkcnt_and_input_resistance", row=1)

        summary_stats_col_keys = self.check_input_resistance_stats_col_keys(tgui, file1_start_col, file2_end_col)
        assert file3_col_keys == [
            "record_num",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "input_resistance",
            "sag_hump_peaks",
            "sag_hump_ratio",
        ]

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["3"],
            file3_col_keys,
            file3_start_col,
            file3_end_col,
            summary_statistics,
            analysis_type="Ri",
        )
        self.check_summary_statistics(
            tgui,
            summary_stats_col_keys,
            summary_statistics,
            ["file_1", "file_2", "file_3"],
            "Ri",
        )

        # File 4 --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_input_resistance_test_file(tgui, to_run="4")

        file4_start_col = self.get_col_file_start_indexes(tgui)[3]
        file4_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file4_col_keys = self.get_col_keys(tgui, file4_start_col, file4_end_col, "spkcnt_and_input_resistance", row=1)

        summary_stats_col_keys = self.check_input_resistance_stats_col_keys(tgui, file1_start_col, file4_end_col)
        assert file4_col_keys == [
            "record_num",
            "user_input_im",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "input_resistance",
            "sag_hump_peaks",
            "sag_hump_ratio",
        ]

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["4"],
            file4_col_keys,
            file4_start_col,
            file4_end_col,
            summary_statistics,
            analysis_type="Ri",
            rec_to=rec_to,
            rec_from=rec_from,
        )
        self.check_summary_statistics(
            tgui,
            summary_stats_col_keys,
            summary_statistics,
            ["file_1", "file_2", "file_3", "file_4"],
            "Ri",
        )

        # File 5 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_input_resistance_test_file(tgui, to_run="5")

        file5_start_col = self.get_col_file_start_indexes(tgui)[4]
        file5_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file5_col_keys = self.get_col_keys(tgui, file5_start_col, file5_end_col, "spkcnt_and_input_resistance", row=1)

        summary_stats_col_keys = self.check_input_resistance_stats_col_keys(tgui, file1_start_col, file5_end_col)
        assert file5_col_keys == [
            "record_num",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "input_resistance",
            "sag_hump_peaks",
            "sag_hump_ratio",
        ]

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["5"],
            file5_col_keys,
            file5_start_col,
            file5_end_col,
            summary_statistics,
            analysis_type="Ri",
        )
        self.check_summary_statistics(
            tgui,
            summary_stats_col_keys,
            summary_statistics,
            ["file_1", "file_2", "file_3", "file_4", "file_5"],
            "Ri",
        )

    def test_input_resistance_per_row_table(self):
        """
        see test_events_per_row_table()
        """
        tgui = SpkcntTGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        self.set_row_or_column_display(tgui, "row")

        if tgui.speed == "fast":
            rec_from, rec_to = [1, 3]
        else:
            rec_from, rec_to = [4, 50]  # TODO: these are hard coded from tgui.rec_from

        # File 1 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_input_resistance_test_file(tgui, to_run="1")

        file1_col_keys, file1_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )

        assert file1_col_keys == [
            "record_num",
            "user_input_im",
            "input_resistance",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
        ]

        file1_start_row = 1
        file1_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
            },
            analysis_type="Ri",
        )

        # File 2 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_input_resistance_test_file(tgui, to_run="2")

        file2_col_keys, file2_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )

        assert file2_col_keys == [
            "record_num",
            "user_input_im",
            "input_resistance",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "sag_hump_peaks",
            "sag_hump_ratio",
        ]

        file2_start_row = file1_end_row + 1
        file2_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1", "2"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
            analysis_type="Ri",
        )

        # File 3 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_input_resistance_test_file(tgui, to_run="3")

        file3_col_keys, file3_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )

        assert file3_col_keys == [
            "record_num",
            "user_input_im",
            "input_resistance",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "sag_hump_peaks",
            "sag_hump_ratio",
        ]

        file3_start_row = file2_end_row + 1
        file3_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1", "2", "3"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file3_col_keys": file3_col_keys,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
            analysis_type="Ri",
        )

        # File 4 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_input_resistance_test_file(tgui, to_run="4")

        file4_col_keys, file4_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )

        assert file4_col_keys == [
            "record_num",
            "user_input_im",
            "input_resistance",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "sag_hump_peaks",
            "sag_hump_ratio",
        ]

        file4_start_row = file3_end_row + 1
        file4_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1", "2", "3", "4"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file3_col_keys": file3_col_keys,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file4_col_keys": file4_col_keys,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
            analysis_type="Ri",
        )

        # File 5 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_input_resistance_test_file(tgui, to_run="5")

        file5_col_keys, file5_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )

        assert file5_col_keys == [
            "record_num",
            "user_input_im",
            "input_resistance",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "sag_hump_peaks",
            "sag_hump_ratio",
        ]

        file5_start_row = file4_end_row + 1
        file5_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1", "2", "3", "4", "5"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file3_col_keys": file3_col_keys,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file4_col_keys": file4_col_keys,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
                "file5_col_keys": file4_col_keys,
                "file5_start_row": file5_start_row,
                "file5_end_row": file5_end_row,
                "file5_filename": file5_filename,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
            analysis_type="Ri",
        )

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Run Input Resistance
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def run_input_resistance_test_file(self, tgui, to_run=None):
        """
        For input resistance, the different parameters are Im calc
        method used and sag / hump on or off.

        File 1 user input im, sag off
        File 2 bounds im, sag on, analyse specific recs
        File 3 im protocol, sag off
        File 4 user imput im, sag on, analyse specific recs
        File 5, bounds, sag off
        """
        norm_or_cum = "normalised" if to_run in ["1", "5"] else "cumulative"
        rec_from, rec_to = self.setup_artificial_spkcnt(tgui, norm_or_cum, "Ri")

        if to_run == "1":
            tgui.run_ri_analysis_user_input_im(0, tgui.adata.num_recs - 1)

        elif to_run == "2":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=True)
            tgui.run_ri_analysis_bounds(set_sag_analysis=True)

        elif to_run == "3":
            tgui.run_input_resistance_im_protocol()

        elif to_run == "4":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=True)
            tgui.run_ri_analysis_user_input_im(rec_from, rec_to, set_sag_analysis=True)

        elif to_run == "5":
            tgui.run_ri_analysis_bounds(set_sag_analysis=False)

        if rec_to:
            return rec_from, rec_to

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Spkcnt
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def test_spkcnt_per_column_table(self):
        """
        see test_events_per_column_table()
        """
        tgui = SpkcntTGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        summary_statistics = {
            "file_1": {},
            "file_2": {},
            "file_3": {},
            "file_4": {},
            "file_5": {},
        }

        # File 1 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_spkcnt_test_file(tgui, to_run="1")

        file1_start_col = 0
        file1_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file1_col_keys = self.get_col_keys(tgui, file1_start_col, file1_end_col, "spkcnt_and_input_resistance", row=1)

        assert file1_col_keys == [
            "record_num",
            "num_spikes",
            "user_input_im",
            "rheobase",
            "mean_isi_ms",
        ]
        assert self.get_item(tgui, 0, 0) == tgui.fake_filename

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["1"],
            file1_col_keys,
            file1_start_col,
            file1_end_col,
            summary_statistics,
        )

        self.check_summary_statistics(tgui, file1_col_keys, summary_statistics, ["file_1"], "spkcnt")

        # File 2 --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_spkcnt_test_file(tgui, to_run="2")

        file2_start_col = self.get_col_file_start_indexes(tgui)[1]
        file2_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file2_col_keys = self.get_col_keys(tgui, file2_start_col, file2_end_col, "spkcnt_and_input_resistance", row=1)

        summary_stats_col_keys = self.check_spkcnt_summary_stats_col_keys(tgui, file1_start_col, file2_end_col)
        assert file2_col_keys == [
            "record_num",
            "num_spikes",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "rheobase",
            "fs_latency_ms",
            "mean_isi_ms",
            "spike_fa_divisor",
            "spike_fa_local_variance",
        ]  # these are not hidden in spkcnt mode

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["2"],
            file2_col_keys,
            file2_start_col,
            file2_end_col,
            summary_statistics,
            rec_to=rec_to,
            rec_from=rec_from,
        )

        self.check_summary_statistics(
            tgui,
            summary_stats_col_keys,
            summary_statistics,
            ["file_1", "file_2"],
            "spkcnt",
        )

        # File 3 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_spkcnt_test_file(tgui, to_run="3")

        file3_start_col = self.get_col_file_start_indexes(tgui)[2]
        file3_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file3_col_keys = self.get_col_keys(tgui, file3_start_col, file3_end_col, "spkcnt_and_input_resistance", row=1)

        summary_stats_col_keys = self.check_spkcnt_summary_stats_col_keys(tgui, file1_start_col, file3_end_col)
        assert file3_col_keys == [
            "record_num",
            "num_spikes",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "rheobase",
            "fs_latency_ms",
            "mean_isi_ms",
            "spike_fa_divisor",
            "spike_fa_local_variance",
        ]

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["3"],
            file3_col_keys,
            file3_start_col,
            file3_end_col,
            summary_statistics,
        )
        self.check_summary_statistics(
            tgui,
            summary_stats_col_keys,
            summary_statistics,
            ["file_1", "file_2", "file_3"],
            "spkcnt",
        )

        # File 4 --------------------------------------------------------------------------------------------------------------------------------------------------

        rec_from, rec_to = self.run_spkcnt_test_file(tgui, to_run="4")

        file4_start_col = self.get_col_file_start_indexes(tgui)[3]
        file4_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file4_col_keys = self.get_col_keys(tgui, file4_start_col, file4_end_col, "spkcnt_and_input_resistance", row=1)

        summary_stats_col_keys = self.check_spkcnt_summary_stats_col_keys(tgui, file1_start_col, file4_end_col)
        assert file4_col_keys == [
            "record_num",
            "num_spikes",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "rheobase",
            "fs_latency_ms",
            "mean_isi_ms",
            "spike_fa_divisor",
            "spike_fa_local_variance",
        ]

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["4"],
            file4_col_keys,
            file4_start_col,
            file4_end_col,
            summary_statistics,
            rec_to=rec_to,
            rec_from=rec_from,
        )
        self.check_summary_statistics(
            tgui,
            summary_stats_col_keys,
            summary_statistics,
            ["file_1", "file_2", "file_3", "file_4"],
            "spkcnt",
        )

        # File 5 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_spkcnt_test_file(tgui, to_run="5")

        file5_start_col = self.get_col_file_start_indexes(tgui)[4]
        file5_end_col = tgui.mw.mw.table_tab_tablewidget.columnCount()
        file5_col_keys = self.get_col_keys(tgui, file5_start_col, file5_end_col, "spkcnt_and_input_resistance", row=1)

        summary_stats_col_keys = self.check_spkcnt_summary_stats_col_keys(tgui, file1_start_col, file5_end_col)
        assert file5_col_keys == [
            "record_num",
            "num_spikes",
            "user_input_im",
            "rheobase",
            "fs_latency_ms",
            "mean_isi_ms",
            "spike_fa_divisor",
            "spike_fa_local_variance",
        ]

        self.check_spkcnt_or_input_resistance_column_table(
            tgui,
            ["5"],
            file5_col_keys,
            file5_start_col,
            file5_end_col,
            summary_statistics,
        )
        self.check_summary_statistics(
            tgui,
            summary_stats_col_keys,
            summary_statistics,
            ["file_1", "file_2", "file_3", "file_4", "file_5"],
            "spkcnt",
        )

    def test_spkcnt_per_row_table(self):
        """
        see test_events_per_row_table()
        """
        tgui = SpkcntTGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        self.set_row_or_column_display(tgui, "row")

        if SPEED == "fast":
            rec_from, rec_to = [1, 3]
        else:
            rec_from, rec_to = [4, 50]

        # File 1 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_spkcnt_test_file(tgui, to_run="1")

        file1_col_keys, file1_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )

        assert file1_col_keys == [
            "record_num",
            "num_spikes",
            "user_input_im",
            "mean_isi_ms",
            "rheobase",
        ]  # TODO: note order is different to column

        file1_start_row = 1
        file1_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
        )

        # File 2 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_spkcnt_test_file(tgui, to_run="2")

        file2_col_keys, file2_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )
        assert file2_col_keys == [
            "record_num",
            "num_spikes",
            "user_input_im",
            "mean_isi_ms",
            "rheobase",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "fs_latency_ms",
            "spike_fa_divisor",
            "spike_fa_local_variance",
        ]

        file2_start_row = file1_end_row + 1  # really?!
        file2_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1", "2"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
        )

        # File 3 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_spkcnt_test_file(tgui, to_run="3")

        file3_col_keys, file3_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )
        assert file3_col_keys == [
            "record_num",
            "num_spikes",
            "user_input_im",
            "mean_isi_ms",
            "rheobase",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "fs_latency_ms",
            "spike_fa_divisor",
            "spike_fa_local_variance",
        ]

        file3_start_row = file2_end_row + 1  # really?!
        file3_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1", "2", "3"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file3_col_keys": file3_col_keys,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
        )

        # File 4 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_spkcnt_test_file(tgui, to_run="4")

        file4_col_keys, file4_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )
        assert file4_col_keys == [
            "record_num",
            "num_spikes",
            "user_input_im",
            "mean_isi_ms",
            "rheobase",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "fs_latency_ms",
            "spike_fa_divisor",
            "spike_fa_local_variance",
        ]

        file4_start_row = file3_end_row + 1  # really?!
        file4_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1", "2", "3", "4"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file3_col_keys": file3_col_keys,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file4_col_keys": file4_col_keys,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
        )

        # File 5 --------------------------------------------------------------------------------------------------------------------------------------------------

        self.run_spkcnt_test_file(tgui, to_run="5")

        file5_col_keys, file5_filename = self.get_col_keys(
            tgui,
            1,
            tgui.mw.mw.table_tab_tablewidget.columnCount(),
            "spkcnt_and_input_resistance",
            row=0,
            return_filename=True,
        )
        assert file5_col_keys == [
            "record_num",
            "num_spikes",
            "user_input_im",
            "mean_isi_ms",
            "rheobase",
            "im_baseline",
            "im_steady_state",
            "im_delta",
            "fs_latency_ms",
            "spike_fa_divisor",
            "spike_fa_local_variance",
        ]

        file5_start_row = file4_end_row + 1
        file5_end_row = tgui.mw.mw.table_tab_tablewidget.rowCount()

        self.check_spkcnt_and_input_resistance_row_table(
            tgui,
            to_check=["1", "2", "3", "4", "5"],
            info={
                "file1_col_keys": file1_col_keys,
                "file1_start_row": file1_start_row,
                "file1_end_row": file1_end_row,
                "file1_filename": file1_filename,
                "file2_col_keys": file2_col_keys,
                "file2_start_row": file2_start_row,
                "file2_end_row": file2_end_row,
                "file2_filename": file2_filename,
                "file3_col_keys": file3_col_keys,
                "file3_start_row": file3_start_row,
                "file3_end_row": file3_end_row,
                "file3_filename": file3_filename,
                "file4_col_keys": file4_col_keys,
                "file4_start_row": file4_start_row,
                "file4_end_row": file4_end_row,
                "file4_filename": file4_filename,
                "file5_col_keys": file4_col_keys,
                "file5_start_row": file5_start_row,
                "file5_end_row": file5_end_row,
                "file5_filename": file5_filename,
                "rec_from": rec_from,
                "rec_to": rec_to,
            },
        )

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Running Spkcnt
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def fast_run_spkcnt(self, tgui, im_setting=False, analysis_to_run=False):
        if im_setting == "bounds":
            bounds_im = {
                "bl": [[0] * tgui.adata.num_recs, [0.1] * tgui.adata.num_recs],
                "exp": [[0.4] * tgui.adata.num_recs, [1.4] * tgui.adata.num_recs],
            }
        elif im_setting == "im_protocol":
            bounds_im = {"start": 0.1, "stop": 1.1}

        elif im_setting == "user_input_im":
            tgui.switch_groupbox(tgui.mw.mw.spkcnt_im_groupbox, on=True)
            tgui.set_combobox(tgui.mw.mw.spkcnt_im_combobox, 2)
            tgui.left_mouse_click(tgui.mw.mw.spkcnt_set_im_button)
            tgui.fill_user_im_input_widget(tgui.adata.num_recs, tgui.mw.mw.spkcnt_set_im_button)
            bounds_im = None
        else:
            bounds_im = False

        tgui.run_spikecount_analysis(
            analysis_to_run=analysis_to_run,
            im_setting=im_setting,
            run_=True,
            bounds_im=bounds_im,
        )

    def run_spkcnt_test_file(self, tgui, to_run=None):
        """
        Convenience function to run spkcnt analysis. This is hard coded for tests and chosen to cover
        all analysis types in a range of orders.

        For spikecounts, there are a lot of parameter options. The Im can be user input im,
        bounds or im protocol. first-spike latency, mean inter-spike interval, rheobase (record vs. exact)
        and spike frequency accomodation an all be on / off

        File 1 user input im, mean isi, rheobase exact
        File 2 bounds im, fs latency, rheobase record, analyse specific recs
        File 3 im protoco, mrheobase record
        File 4 everything off, analyse_specific recs
        File 5 user im, other parameters off
        """
        norm_or_cum = "normalised" if to_run in ["2", "5"] else "cumulative"
        rec_from, rec_to = self.setup_artificial_spkcnt(tgui, norm_or_cum, "spkcnt")

        if to_run == "1":
            self.fast_run_spkcnt(tgui, "user_input_im", ["mean_isi_ms", "rheobase_record"])

        elif to_run == "2":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=True)
            self.fast_run_spkcnt(tgui, "bounds", ["fs_latency_ms", "rheobase_exact", "spike_fa"])
        elif to_run == "3":
            self.fast_run_spkcnt(tgui, "im_protocol", ["rheobase_record"])
        elif to_run == "4":
            __, rec_from, rec_to = tgui.handle_analyse_specific_recs(analyse_specific_recs=True)
            self.fast_run_spkcnt(tgui)

        elif to_run == "5":
            self.fast_run_spkcnt(tgui, "user_input_im")

        if rec_to:
            return rec_from, rec_to

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Checking Spkcnt Row and Column
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def check_spkcnt_and_input_resistance_row_table(self, tgui, to_check, info, analysis_type="spkcnt"):
        """
        Convenience function for running row checks.
        """
        if "1" in to_check:
            self.check_row_table_against_stored_tabledata(
                tgui,
                info["file1_col_keys"],
                info["file1_start_row"],
                info["file1_end_row"],
                info["file1_filename"],
                "record",
                file_idx=0,
                analysis_type=analysis_type,
            )
        if "2" in to_check:
            self.check_row_table_against_stored_tabledata(
                tgui,
                info["file2_col_keys"],
                info["file2_start_row"],
                info["file2_end_row"],
                info["file2_filename"],
                "exact",
                file_idx=1,
                rec_from=info["rec_from"],
                rec_to=info["rec_to"],
                analysis_type=analysis_type,
            )
        if "3" in to_check:
            self.check_row_table_against_stored_tabledata(
                tgui,
                info["file3_col_keys"],
                info["file3_start_row"],
                info["file3_end_row"],
                info["file3_filename"],
                "record",
                file_idx=2,
                analysis_type=analysis_type,
            )
        if "4" in to_check:
            self.check_row_table_against_stored_tabledata(
                tgui,
                info["file4_col_keys"],
                info["file4_start_row"],
                info["file4_end_row"],
                info["file4_filename"],
                None,
                file_idx=3,
                rec_from=info["rec_from"],
                rec_to=info["rec_to"],
                analysis_type=analysis_type,
            )
        if "5" in to_check:
            self.check_row_table_against_stored_tabledata(
                tgui,
                info["file5_col_keys"],
                info["file5_start_row"],
                info["file5_end_row"],
                info["file5_filename"],
                None,
                file_idx=4,
                analysis_type=analysis_type,
            )

    def check_spkcnt_or_input_resistance_column_table(
        self,
        tgui,
        to_check,
        file_col_keys,
        file_start_col,
        file_end_col,
        summary_statistics,
        analysis_type="spkcnt",
        rec_from=0,
        rec_to=None,
    ):
        """
        Convenience function for checking table results when in column mode and updating the summary statistics
        """
        assert self.get_item(tgui, 0, file_start_col) == tgui.fake_filename

        if "1" in to_check:
            summary = self.check_column_table_against_stored_tabledata(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                "record",
                file_idx=0,
                analysis_type=analysis_type,
            )
            summary_statistics["file_1"] = summary

        elif "2" in to_check:
            summary = self.check_column_table_against_stored_tabledata(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                "exact",
                file_idx=1,
                analysis_type=analysis_type,
                rec_from=rec_from,
                rec_to=rec_to,
            )
            summary_statistics["file_2"] = summary

        elif "3" in to_check:
            summary = self.check_column_table_against_stored_tabledata(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                "record",
                file_idx=2,
                analysis_type=analysis_type,
            )
            summary_statistics["file_3"] = summary

        elif "4" in to_check:
            summary = self.check_column_table_against_stored_tabledata(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                None,
                file_idx=3,
                analysis_type=analysis_type,
                rec_to=rec_to,
                rec_from=rec_from,
            )
            summary_statistics["file_4"] = summary

        elif "5" in to_check:
            summary = self.check_column_table_against_stored_tabledata(
                tgui,
                file_col_keys,
                file_start_col,
                file_end_col,
                None,
                file_idx=4,
                analysis_type=analysis_type,
                rec_to=rec_to,
                rec_from=rec_from,
            )
            summary_statistics["file_5"] = summary

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Check Column and Row For Spikecounting and Input resistance
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def check_column_table_against_stored_tabledata(
        self,
        tgui,
        file_col_keys,
        file_start_col,
        file_end_col,
        rheobase_rec_or_exact,
        file_idx,
        rec_from=0,
        rec_to=None,
        analysis_type="Ri",
    ):  # TODO: use for all analyses?
        """
        This function will iterate through all spkcnt results on the table (shown in Column mode) and check they match the underlying
        stored tabledata. This tabledata is already checked in other functions.

        First get all data in the table between the input file_start_col and file_end_col. Iterate through the file col keys
        which are taken from the table previously (and so order will match table data).

        Each column is iterated through and checked the data on table matches that in the stored tabledata. Also, all
        information is saved for testing the summary statistics.
        """
        data_on_table, text = self.get_table(tgui, row_start=2, col_start=file_start_col, col_end=file_end_col)

        stored_tabledata = self.get_stored_tabledata(tgui, analysis_type, file_idx)

        if rec_to is None:
            rec_to = tgui.mw.mw.table_tab_tablewidget.rowCount() - 2

        num_recs = rec_to - rec_from if rec_from == 0 else rec_to - rec_from + 1

        summary_statistics = {
            "data": {},
            "col_keys": file_col_keys,
            "file_filename": tgui.fake_filename,
            "rec_from": rec_from,
            "rec_to": rec_to,
            "num_recs": num_recs,
        }

        summary_statistics["data"]["record_count"] = {"record_count": stored_tabledata.iloc[:, 0].dropna().size}
        summary_statistics["data"]["record_count"]["all_rec_count"] = np.ones(stored_tabledata.iloc[:, 0].dropna().size)

        for col, col_key in enumerate(file_col_keys):
            if col_key == "rheobase":
                # Check the rheobase shown and method type on the first two records are correct
                assert np.array_equal(
                    stored_tabledata.loc[0, col_key],
                    data_on_table[0, col],
                    equal_nan=True,
                )

                summary_statistics["data"][col_key] = {
                    "M": data_on_table[0, col],
                    "SD": None,
                    "SE": None,
                }
                if text:
                    assert rheobase_rec_or_exact in text[0][2]

            elif col_key == "input_resistance":
                assert np.array_equal(
                    stored_tabledata.loc[0, col_key],
                    data_on_table[0, col],
                    equal_nan=True,
                )
                summary_statistics["data"][col_key] = {
                    "M": data_on_table[0, col],
                    "SD": None,
                    "SE": None,
                }

            else:
                # Check all other data and store details for summary statistics
                col_data_on_table = data_on_table[:num_recs, col]

                assert np.array_equal(
                    stored_tabledata.loc[rec_from:rec_to, col_key],
                    col_data_on_table,
                    equal_nan=True,
                )

                summary_statistics["data"][col_key] = {
                    "M": np.mean(col_data_on_table),
                    "SD": np.std(col_data_on_table, ddof=1),
                    "SE": scipy.stats.sem(col_data_on_table),
                    "all_rec_data": {
                        "M": stored_tabledata.loc[rec_from:rec_to, col_key],
                        "SD": None,
                        "SE": None,
                    },
                }

        return summary_statistics

    def check_row_table_against_stored_tabledata(
        self,
        tgui,
        file_col_keys,
        file_row_start,
        file_row_end,
        file_filename,
        rheobase_rec_or_exact,
        file_idx,
        rec_from=0,
        rec_to=None,
        analysis_type="Ri",
    ):
        """ """
        self.set_row_or_column_display(tgui, "row")
        self.check_row_filenames(tgui, file_row_start, file_row_end, file_filename)

        data_on_table, text = self.get_table(
            tgui,
            row_start=file_row_start,
            row_end=file_row_end,
            col_start=1,
            col_end=tgui.mw.mw.table_tab_tablewidget.columnCount(),
        )

        stored_tabledata = self.get_stored_tabledata(tgui, analysis_type, file_idx)

        for col, col_key in enumerate(file_col_keys):
            if col_key == "rheobase":
                assert np.array_equal(
                    stored_tabledata.loc[0, col_key],
                    data_on_table[0, col],
                    equal_nan=True,
                )  # ERROR

                if text:
                    assert rheobase_rec_or_exact in text[0][2]  # will have to fix
            else:
                assert np.array_equal(
                    stored_tabledata.loc[rec_from:rec_to, col_key],
                    data_on_table[:, col],
                    equal_nan=True,
                )  # TODO: will have to fix recs

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Curve Fitting Error-Prone Tests
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("region", ["reg_2", "reg_3", "reg_4", "reg_5", "reg_6"])
    def test_curve_fitting_summary_statistics_different_analysis(self, region):
        """
        When the analysis is different for a region, summary statistics cannot be calculated
        and a window is shown - check this occurs here.
        """
        tgui = CurveFittingGui()
        tgui.mw.mw.actionBatch_Mode_ON.trigger()
        self.set_row_or_column_display(tgui, "row")
        tgui.switch_mw_tab(1)

        self.reload_curve_fitting_file(tgui)
        self.run_default_curve_fitting(tgui, "min", region)
        self.reload_curve_fitting_file(tgui)
        self.run_default_curve_fitting(tgui, "max", region)

        QtCore.QTimer.singleShot(1000, lambda: self.check_sumstat_messagebox_error(tgui))
        if region == "reg_1":
            tgui.mw.mw.curve_fitting_summary_statistics_combobox.activated.emit(
                0
            )  # currentIndexChanged() wont activate as idx is already 0
        else:
            tgui.mw.mw.curve_fitting_summary_statistics_combobox.setCurrentIndex(int(region[-1]) - 1)

        tgui.shutdown()

    def check_sumstat_messagebox_error(self, tgui):
        assert (
            "All region analysis type (e.g. Minimum) must match across files to calculate summary statistics"
            in tgui.mw.messagebox.text()
        )
        tgui.mw.messagebox.accept()
        del tgui.mw.messagebox

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Summary Statistics Methods
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Not these are shared across all analysis types

    def check_summary_statistics(
        self,
        tgui,
        col_keys_union,
        summary_statistics,
        file_nums,
        analysis_type,
        group="all",
    ):
        """
        Check summary statistics shown on the results table. Do this for average across and within all recs,
        and check every combination of M, SD, SE.
        """
        check_function = self.get_compare_function_and_switch_checkbox(tgui, analysis_type, on=True)

        for file_num in file_nums:
            # Switch to table tab, and open analysis table options
            # dialog, update col keys to remove non-averaged data
            file_idx = int(file_num[-1]) - 1
            tgui.switch_mw_tab(1)
            tgui.mw.mw.actionTable_Options.trigger()
            dialog = tgui.mw.dialogs["analysis_statistics_options_dialog"]

            # Hacky, all true parameters are analysed separately so removed
            # at this stage. Now, we want to check "record_count" so pull
            # it out here to test against separately.
            file_record_count = summary_statistics[file_num]["data"]["record_count"]

            col_keys_union = self.remove_bad_col_keys(tgui, summary_statistics, file_num, col_keys_union)

            # Set to average across all records, and cycle through each row checking the
            # info shown on table matches the true M, SE and SD results. Now we know the tabledata
            # matches stored tabledata, and these summarys are correct based on the tabledata.
            dialog.dia.calculate_across_records_combobox.setCurrentIndex(0)

            if dialog.dia.calculate_across_records_combobox.currentIndex() == 1:
                # some Qt bug the above can fail
                tgui.mw.cfgs.analysis["summary_stats_within_rec"] = False

            row = int(file_num[-1]) + 1
            self.check_across_rec_mode_record_count(tgui, file_record_count, row)

            for parameters in [
                ["M", "SD", "SE"],
                ["M", "SE"],
                ["M", "SD"],
                ["SD", "SE"],
                ["M", "SE"],
                ["SD"],
                ["SE"],
            ]:
                self.set_summary_statistics_parameter_checkboxes(tgui, dialog, parameters)

                check_function(
                    tgui,
                    summary_statistics,
                    col_keys_union,
                    row=row,
                    parameters=parameters,
                    file_num=file_num,
                )

            # Now set to average within records, and cycle through all rows (average of record per row)
            # and again check this matches the summary_statistics saved data. Calculate the next
            # file start row using the num_recs per file.
            tgui.set_combobox(dialog.dia.calculate_across_records_combobox, 1)

            start_row = (
                np.cumsum([summary_statistics["file_" + str(filenum + 1)]["num_recs"] for filenum in range(file_idx)])[
                    -1
                ]
                if file_idx > 0
                else 0
            )

            file_num_recs = summary_statistics[file_num]["num_recs"]

            start_row += 2 + file_idx  # each file as one row of space underneath

            end_row = start_row + file_num_recs if file_num_recs == 1 else start_row + file_num_recs - 1

            self.check_row_filenames(tgui, start_row, end_row, summary_statistics[file_num]["file_filename"])

            if group == "all":
                self.check_row_records(
                    tgui,
                    start_row,
                    start_row + file_num_recs - 1,
                    summary_statistics[file_num],
                )

            self.check_within_rec_mode_record_count(tgui, file_record_count, start_row, file_num_recs)

            for parameters in [
                ["M", "SD", "SE"],
                ["M", "SE"],
                ["M", "SD"],
                ["SD", "SE"],
                ["M", "SE"],
                ["SD"],
                ["SE"],
            ]:
                self.set_summary_statistics_parameter_checkboxes(tgui, dialog, parameters)

                for rec_idx, row in enumerate(range(start_row, start_row + file_num_recs - 1)):
                    check_function(
                        tgui,
                        summary_statistics,
                        col_keys_union,
                        row,
                        parameters,
                        file_num,
                        rec_mode_idx=rec_idx,
                    )

        self.get_compare_function_and_switch_checkbox(tgui, analysis_type, on=False)

    def check_across_rec_mode_record_count(self, tgui, file_record_count, row):
        """ """
        count = tgui.mw.mw.table_tab_tablewidget.item(row, 2).data(0)
        assert int(count) == file_record_count["record_count"]

    def check_within_rec_mode_record_count(self, tgui, file_record_count, start_row, file_num_recs):
        """ """
        counts = []
        for row in range(start_row, start_row + file_num_recs - 1 + 1):
            count = tgui.mw.mw.table_tab_tablewidget.item(row, 2).data(0)
            counts.append(int(count))
        assert np.array_equal(file_record_count["all_rec_count"], counts)

    def check_skinetics_events__cf_summary_stats_row(
        self,
        tgui,
        summary_statistics,
        col_keys_union,
        row,
        parameters,
        file_num,
        rec_mode_idx=None,
    ):
        """ """
        table = tgui.mw.mw.table_tab_tablewidget
        QtWidgets.QApplication.processEvents()

        col_start = 3
        if rec_mode_idx is None:
            assert table.item(row, 1).data(0) == "All"

            for col, col_key in enumerate(col_keys_union):
                for parameter in parameters:
                    if col_key not in summary_statistics[file_num]["data"]:
                        assert table.item(row, col_start) is None, "across recs null: {0} {1} {2}".format(
                            file_num, col_key, parameter
                        )
                    else:
                        assert (
                            float(self.get_item(tgui, row, col_start))
                            == summary_statistics[file_num]["data"][col_key][parameter]
                        ), "across recs: {0} {1} {2}".format(file_num, col_key, parameter)
                    col_start += 1

        else:
            for col, col_key in enumerate(col_keys_union):
                for parameter in parameters:
                    if col_key not in summary_statistics[file_num]["data"]:
                        assert table.item(row, col_start) is None, "within recs null: {0} {1} {2}".format(
                            file_num, col_key, parameter
                        )

                    else:
                        sum_stat_data = summary_statistics[file_num]["data"][col_key]["all_rec_data"][parameter][
                            rec_mode_idx
                        ]

                        if table.item(row, col_start).data(0) == "":
                            assert sum_stat_data is None or np.isnan(sum_stat_data), "empty fail {0} {1}".format(
                                col_key, parameter
                            )
                        else:
                            assert (
                                float(self.get_item(tgui, row, col_start)) == sum_stat_data
                            ), "within recs: {0} {1} {2}".format(file_num, col_key, parameter)
                    col_start += 1

    def check_spkcnt_ri_summary_stats_row(
        self,
        tgui,
        summary_statistics,
        col_keys_union,
        row,
        parameters,
        file_num,
        rec_mode_idx=None,
    ):
        """
        Check summary statistics
        """
        table = tgui.mw.mw.table_tab_tablewidget
        QtWidgets.QApplication.processEvents()

        start_col = 3
        for col, col_key in enumerate(col_keys_union):
            # Check the header is correct
            header = tgui.mw.cfgs.get_table_col_headers("spkcnt_and_input_resistance", col_key)[1]
            assert table.item(0, start_col).data(0).strip() == header, "header failed {0}".format(col_key)

            # Skips the rows in which no data of this type is found for this filetype.
            # This is blank on the table, and is saved as None in the summary statistics
            if col_key not in summary_statistics[file_num]["col_keys"]:
                assert (
                    table.item(row, start_col) is None or table.item(row, start_col).data(0) == ""
                ), "empty failed {0}".format(col_key)
                start_col += len(parameters)
                continue

            sum_stat_data = summary_statistics[file_num]["data"][col_key]

            for parameter in parameters:
                assert table.item(1, start_col).data(0).strip() == parameter, "param failed {0} {1}".format(
                    col_key, parameter
                )

                if rec_mode_idx is not None:
                    self.check_spkcnt_summary_statistics_average_within_recs(
                        table,
                        col_key,
                        row,
                        start_col,
                        sum_stat_data,
                        parameter,
                        rec_mode_idx,
                    )
                else:
                    self.check_spkcnt_summary_statistics_average_across_recs(
                        table, col_key, row, start_col, sum_stat_data, parameter
                    )

                start_col += 1

    def check_spkcnt_summary_statistics_average_within_recs(
        self, table, col_key, row, start_col, sum_stat_data, parameter, rec_mode_idx
    ):
        """ """
        if col_key in ["rheobase", "input_resistance"]:
            if table.item(row, start_col).data(0) != "":
                assert float(table.item(row, start_col).data(0)) == sum_stat_data["M"], "rheobase sumstat fail"

        elif table.item(row, start_col).data(0) == "":
            assert sum_stat_data["all_rec_data"][parameter] is None or np.isnan(
                sum_stat_data["all_rec_data"][parameter].iloc[rec_mode_idx]
            ), "empty fail {0}".format((col_key))

        else:
            on_table_data = sum_stat_data["all_rec_data"][parameter]
            if on_table_data is None:
                assert table.item(row, start_col).data(0) == "", "empty fail {0} {1}".format(col_key, parameter)
            else:
                assert (
                    float(table.item(row, start_col).data(0)) == on_table_data.iloc[rec_mode_idx]
                ), "fail {0} {1}".format(col_key, parameter)

    def check_spkcnt_summary_statistics_average_across_recs(
        self, table, col_key, row, start_col, sum_stat_data, parameter
    ):
        """ """
        assert table.item(row, 1).data(0) == "All", "rec fail {0} {1}".format(col_keys, parameter)

        if table.item(row, start_col).data(0) == "":
            assert sum_stat_data[parameter] is None or np.isnan(sum_stat_data[parameter]), "empty fail {0} {1}".format(
                col_key, parameter
            )

        elif col_key == "rheobase":
            assert float(table.item(row, start_col).data(0)) == sum_stat_data["M"], "rheobase fail {0} {1}".format(
                col_key, parameter
            )
        else:
            assert float(table.item(row, start_col).data(0)) == sum_stat_data[parameter], "fail {0} {1}".format(
                col_key, parameter
            )

    # Calc Summary Statistics
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def make_summary_statistics_from_column_data(
        self,
        data_on_table,
        summary_statistics,
        file_col_keys,
        rec_from,
        rec_to,
        num_recs,
        record_col_idx,
        group="all",
    ):
        """
        Function to calculate the M, SD, SE for testing against from column data in the column data checks functions
        """
        for col, key in enumerate(file_col_keys):
            col_data_on_table = data_on_table[:, col]

            if col == 0:  # only need to do this once
                summary_statistics["data"]["record_count"] = {"record_count": col_data_on_table.size}

            summary_statistics["data"][key] = {
                "M": np.mean(col_data_on_table),
                "SD": np.std(col_data_on_table, ddof=1),
                "SE": scipy.stats.sem(col_data_on_table),
            }

            all_event_records = data_on_table[:, record_col_idx]

            if group == "all":
                rec_range = (
                    np.arange(rec_from, rec_to + 1) + 1
                )  # use test data, unless we are using grouped analysis in which case
            else:  # default to data based on the table.
                unique_records = np.unique(all_event_records)
                rec_range = unique_records
                num_recs = len(unique_records)

            if col == 0:  # Only need to do this once TODO: Major DRY
                all_rec_record_count = utils.np_empty_nan((num_recs))
                for idx, rec in enumerate(rec_range):
                    this_rec_indicies = np.where(all_event_records == rec)
                    all_rec_record_count[idx] = col_data_on_table[this_rec_indicies].size

                summary_statistics["data"]["record_count"]["all_rec_count"] = all_rec_record_count  # TODO: rename

            all_rec_data_mean = utils.np_empty_nan((num_recs))
            all_rec_data_std = utils.np_empty_nan((num_recs))
            all_rec_data_ste = utils.np_empty_nan((num_recs))

            for idx, rec in enumerate(rec_range):
                this_rec_indicies = np.where(all_event_records == rec)

                all_rec_data_mean[idx] = np.mean(col_data_on_table[this_rec_indicies])
                all_rec_data_std[idx] = np.std(col_data_on_table[this_rec_indicies], ddof=1)
                all_rec_data_ste[idx] = scipy.stats.sem(col_data_on_table[this_rec_indicies])

            summary_statistics["data"][key]["all_rec_data"] = {
                "M": all_rec_data_mean,
                "SD": all_rec_data_std,
                "SE": all_rec_data_ste,
            }

        return summary_statistics

    # Check filenames and records for row-mode
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def check_row_filenames(self, tgui, file_row_start, file_row_end, file_filename, filename_col=0):
        """
        For row mode and summary statistics, check the filenames are correct in the filename column
        for both options show all, show only first row
        """
        filenames = self.get_filenames_col(tgui, file_row_start, file_row_end, filename_col)

        assert all([file_filename == filename for filename in filenames])

        tgui.mw.mw.actionTable_Options.trigger()
        dialog = tgui.mw.dialogs["analysis_statistics_options_dialog"]

        tgui.switch_checkbox(dialog.dia.show_filename_once_checkbox, on=True)
        filenames = self.get_filenames_col(tgui, file_row_start, file_row_end, filename_col)

        assert all(["" == filename for filename in filenames[1:]])
        assert filenames[0] == file_filename

        tgui.switch_checkbox(dialog.dia.show_filename_once_checkbox, on=False)

    def check_row_records(self, tgui, file_row_start, file_row_end, file_summary_statistics):
        """
        Check the row records are correct for summary stats data
        """
        rec_nums = []
        for row in range(file_row_start, file_row_end + 1):
            rec_num = tgui.mw.mw.table_tab_tablewidget.item(row, 1).data(0)
            rec_nums.append(int(rec_num))

        rec_from = file_summary_statistics["rec_from"]
        test_file_recs = np.arange(rec_from, rec_from + file_summary_statistics["num_recs"]) + 1

        assert np.array_equal(rec_nums, test_file_recs)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def check_input_resistance_stats_col_keys(self, tgui, file_start_col, file_end_col):
        """
        For summary statistics, a union of all files headers is used so that all data can be plot on a single table.
        """
        summary_stats_col_keys = self.get_col_keys(
            tgui, file_start_col, file_end_col, "spkcnt_and_input_resistance", row=1
        )  # just get all keys because first keys change as second keys written
        summary_stats_col_keys = self.get_summary_statistics_col_keys([summary_stats_col_keys])

        assert summary_stats_col_keys == [
            "record_num",
            "user_input_im",
            "vm_baseline",
            "vm_steady_state",
            "vm_delta",
            "input_resistance",
            "sag_hump_peaks",
            "sag_hump_ratio",
            "im_baseline",
            "im_steady_state",
            "im_delta",
        ]
        return summary_stats_col_keys

    def check_spkcnt_summary_stats_col_keys(self, tgui, file_start_col, file_end_col):
        """
        For summary statistics, a union of all files headers is used so that all data can be plot on a single table.
        """
        summary_stats_col_keys = self.get_col_keys(
            tgui, file_start_col, file_end_col, "spkcnt_and_input_resistance", row=1
        )  # just get all keys because first keys change as second keys written
        summary_stats_col_keys = self.get_summary_statistics_col_keys([summary_stats_col_keys])

        assert summary_stats_col_keys == [
            "record_num",
            "num_spikes",
            "user_input_im",
            "rheobase",
            "fs_latency_ms",
            "mean_isi_ms",
            "spike_fa_divisor",
            "spike_fa_local_variance",
            "im_baseline",
            "im_steady_state",
            "im_delta",
        ]
        return summary_stats_col_keys

    def reset_spkcnt_options(self, tgui):
        """
        easier to turn everything off and re-set during analysis
        """
        tgui.switch_groupbox(tgui.mw.mw.spkcnt_recs_to_analyse_groupbox, on=False)
        tgui.set_combobox(tgui.mw.mw.spkcnt_im_combobox, 0)
        tgui.switch_groupbox(tgui.mw.mw.spkcnt_im_groupbox, on=False)
        tgui.switch_checkbox(tgui.mw.spkcnt_popup.dia.fs_latency_checkbox, on=False)
        tgui.switch_checkbox(tgui.mw.spkcnt_popup.dia.mean_isi_checkbox, on=False)
        tgui.switch_checkbox(tgui.mw.spkcnt_popup.dia.spike_freq_accommodation_checkbox, on=False)
        tgui.switch_checkbox(tgui.mw.spkcnt_popup.dia.spkcnt_rheobase_record_radiobutton, on=True)
        tgui.switch_groupbox(tgui.mw.spkcnt_popup.dia.spkcnt_rheobase_groupbox, on=False)

        tgui.set_combobox(tgui.mw.mw.ir_im_combobox, idx=0)
        tgui.switch_groupbox(tgui.mw.mw.ir_recs_to_analyse_groupbox, on=False)
        tgui.switch_groupbox(tgui.mw.mw.ir_calculate_sag_groupbox, on=False)

        tgui.switch_groupbox(tgui.mw.mw.skinetics_recs_to_analyse_groupbox, on=False)
        tgui.mw.mw.actionSpike_Kinetics_Options_2.trigger()
        tgui.switch_groupbox(tgui.mw.dialogs["skinetics_options"].dia.max_slope_groupbox, on=False)

    def set_row_or_column_display(self, tgui, row_or_column):
        """ """
        tgui.mw.mw.actionTable_Options.trigger()

        if row_or_column == "row":
            tgui.switch_checkbox(
                tgui.mw.dialogs["analysis_statistics_options_dialog"].dia.file_per_row_radiobutton,
                on=True,
            )

        elif row_or_column == "column":
            tgui.switch_checkbox(
                tgui.mw.dialogs["analysis_statistics_options_dialog"].dia.file_per_column_radiobutton,
                on=True,
            )

    def get_col_file_start_indexes(self, tgui):
        """ """
        table = tgui.mw.mw.table_tab_tablewidget

        indexes = []
        for col in range(table.columnCount()):
            try:
                data = table.item(0, col).data(0)
                if type(data) == str:
                    indexes.append(col)
            except:
                pass
        return indexes

    def get_table(self, tgui, row_start=0, row_end=None, col_start=0, col_end=None):
        """ """
        table = tgui.mw.mw.table_tab_tablewidget

        if not row_end:
            row_end = table.rowCount()

        if not col_end:
            col_end = table.columnCount()

        num_rows = row_end - row_start
        num_cols = col_end - col_start

        results = utils.np_empty_nan((num_rows, num_cols))
        text = []
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                if table.item(row, col) is None or table.item(row, col).data(0) == "":
                    continue

                item = self.get_item(tgui, row, col)

                if "method" in item or item in ["Thr.", "off"] or "region" in item:
                    text.append([row, col, item])
                else:
                    results[row - row_start, col - col_start] = float(item)

        return results, text

    def get_col_keys(self, tgui, col_start, col_end, analysis_type, row=1, return_filename=False):
        """ """
        file_col_headers = [self.get_item(tgui, row, col).strip() for col in range(col_start, col_end)]
        file_header_dict = tgui.mw.cfgs.get_table_col_headers(analysis_type)[0]
        file_col_keys = [
            utils.get_dict_key_from_value(file_header_dict, header) if header != "Filename" else "filename"
            for header in file_col_headers
        ]

        if return_filename:
            return file_col_keys, tgui.fake_filename
        else:
            return file_col_keys

    def get_item(self, tgui, row, col):
        """ """
        return tgui.mw.mw.table_tab_tablewidget.item(row, col).data(0)

    def get_summary_statistics_col_keys(self, file_col_keys):
        """ """
        col_keys_union = []
        for col_keys in file_col_keys:
            for key in col_keys:
                if key not in col_keys_union:
                    col_keys_union.append(key)
        return col_keys_union

    def set_summary_statistics_parameter_checkboxes(self, tgui, dialog, parameters, set_all_on=True):
        """ """
        if set_all_on:
            self.set_summary_statistics_parameter_checkboxes(
                tgui, dialog, ["M", "SD", "SE"], set_all_on=False
            )  # first turn all on as checkbox will show if all are attemtped turned off

        bool_ = True if "M" in parameters else False
        tgui.switch_checkbox(dialog.dia.mean_checkbox, on=bool_)

        bool_ = True if "SD" in parameters else False
        tgui.switch_checkbox(dialog.dia.standard_deviation_checkbox, on=bool_)

        bool_ = True if "SE" in parameters else False
        tgui.switch_checkbox(dialog.dia.standard_error_checkbox, on=bool_)

    def remove_bad_col_keys(self, tgui, summary_statistics, file_num, col_keys_union):
        """ """
        excluded_keys = tgui.mw.cfgs.get_summary_statistics_excluded_headers("col_keys")

        col_keys = [True if key not in excluded_keys else False for key in summary_statistics[file_num]["col_keys"]]

        for key in excluded_keys:
            if key in summary_statistics[file_num]["data"].keys():
                summary_statistics[file_num]["data"].pop(key)

        summary_statistics[file_num]["col_keys"] = [
            summary_statistics[file_num]["col_keys"][i] for i in range(len(col_keys)) if col_keys[i]
        ]
        col_keys_union = [key for key in col_keys_union if key not in excluded_keys]
        return col_keys_union

    def get_stored_tabledata(self, tgui, analysis_type, file_idx):
        if analysis_type == "spkcnt":  # TODO: OWN FUNCTION
            stored_tabledata = tgui.mw.stored_tabledata.spkcnt_data[
                file_idx
            ]  # TODO: very similar to abouve but some awk differences so just live with some DRY
        elif analysis_type == "Ri":
            stored_tabledata = tgui.mw.stored_tabledata.ir_data[
                file_idx
            ]  # TODO: very similar to abouve but some awk differences so just live with some DRY
        return stored_tabledata

    def reset_events_options(self, tgui, options_dialog):
        tgui.switch_groupbox(options_dialog.dia.max_slope_groupbox, on=False)
        tgui.switch_groupbox(tgui.mw.mw.events_threshold_recs_to_analyse_groupbox, on=False)

    def get_and_cut_col_data_on_table(self, tgui, file_start_col, file_end_col):
        """
        cut the rows that are not have dat
        """
        data_on_table, text = self.get_table(tgui, row_start=2, col_start=file_start_col, col_end=file_end_col)
        data_on_table = data_on_table[~np.isnan(data_on_table)[:, 0], :]
        return data_on_table, text

    def get_compare_function_and_switch_checkbox(self, tgui, analysis_type, on=True):
        """ """
        checkboxes = {
            "spkcnt": tgui.mw.mw.spkcnt_summary_statistics_table_checkbox,
            "Ri": tgui.mw.mw.input_resistance_summary_statistics_table_checkbox,
            "skinetics": tgui.mw.mw.skinetics_summary_statistics_table_checkbox,
            "events": tgui.mw.mw.events_summary_statistics_table_checkbox,
            "curve_fitting": tgui.mw.mw.curve_fitting_summary_statistics_table_checkbox,
        }

        tgui.switch_checkbox(checkboxes[analysis_type], on=on)

        if on:
            if analysis_type in ["spkcnt", "Ri"]:
                return self.check_spkcnt_ri_summary_stats_row  # RENAME
            else:
                return self.check_skinetics_events__cf_summary_stats_row

    def get_filenames_col(self, tgui, file_row_start, file_row_end, filename_col=0):
        """ """
        filenames = []
        for rec in range(file_row_start, file_row_end):
            cell = tgui.mw.mw.table_tab_tablewidget.item(rec, filename_col)
            filename = cell.data(0).strip() if cell else ""
            filenames.append(filename)
        return filenames

    def setup_artificial_spkcnt(self, tgui, norm_or_cum, data_type):
        tgui.setup_artificial_data(norm_or_cum, data_type)
        tgui.set_fake_filename()
        self.reset_spkcnt_options(tgui)
        rec_from = rec_to = None
        return rec_from, rec_to

    def get_curve_fitting_table(
        self, tgui, rec_from, rec_to, regions, region_keys, file_idx
    ):  # TODO: merge with test_backup_options
        num_recs = rec_to - rec_from + 1

        results = utils.np_empty_nan((num_recs, sum([len(list_) for list_ in region_keys])))

        # dont forget different rec numbers per region
        col = 0
        text = []
        for reg, region_col_keys in zip(regions, region_keys):
            for key in region_col_keys:
                if key == "record_num":
                    results[:, col] = np.arange(rec_from, rec_to + 1) + 1

                elif key == "filename":
                    col += 1
                    continue
                else:
                    analysed_recs = tgui.mw.stored_tabledata.curve_fitting[file_idx][reg]["data"][rec_from : rec_to + 1]

                    if "event_period" in region_col_keys:  # i.e. is this events
                        data = self.get_events_data(analysed_recs, key)
                    else:
                        data = np.array([entry[key] for entry in analysed_recs])

                    if "off" in data:  # max rise / slope
                        text.append(["off"] * num_recs)
                    else:
                        results[:, col] = data

                col += 1

        return results, text

    def get_events_data(self, analysed_recs, key):
        if key in ["record_num", "b0", "b1", "fit_rise", "fit_decay", "r2"]:
            data = np.array([entry[key] for entry in analysed_recs])
        else:
            first_key, second_key = self.get_events_data_keys(key)
            data = np.array(
                [
                    entry["event_info"][list(entry["event_info"].keys())[0]][first_key][second_key]
                    for entry in analysed_recs
                ]
            )

        return data

    def get_events_data_keys(self, key):
        if key == "monoexp_fit_b0":
            return "monoexp_fit", "b0"
        elif key == "monoexp_fit_b1":
            return "monoexp_fit", "b1"
        elif key == "monoexp_fit_tau":
            return "monoexp_fit", "tau_ms"
        elif key == "events_monoexp_r2":
            return "monoexp_fit", "r2"
        else:
            second_keys = {
                "baseline": "im",
                "peak": "im",
                "amplitude": "im",
                "rise": "rise_time_ms",
                "half_width": "fwhm_ms",
                "decay_perc": "decay_time_ms",
                "area_under_curve": "im",
                "event_period": "time_ms",
                "max_rise": "max_slope_ms",
                "max_decay": "max_slope_ms",
            }
            return key, second_keys[key]
