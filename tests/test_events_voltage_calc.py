from PySide6 import QtWidgets, QtCore, QtGui
from PySide6 import QtTest
from PySide6.QtTest import QTest
import pytest
import sys
import os
import numpy as np

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from easy_electrophysiology.mainwindow.mainwindow import (
    MainWindow,
)
from ephys_data_methods import core_analysis_methods, voltage_calc

os.environ["PYTEST_QT_API"] = "PySide6"
from setup_test_suite import GuiTestSetup
from utils import utils


class TestEvents:
    """
    Unit test the voltage_calc.py functions (not the GUI)
    """

    @pytest.fixture(scope="function")
    def tgui(test):
        tgui = GuiTestSetup("artificial")
        tgui.setup_mainwindow(show=True)
        tgui.load_a_filetype("voltage_clamp_1_record")
        tgui.raise_mw_and_give_focus()
        yield tgui
        tgui.shutdown()

    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # All voltage_calc unit tests
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("direction", [-1, 1])
    @pytest.mark.parametrize("num_rec_multiplier", [1, 5])
    @pytest.mark.parametrize("array_samples", [1, 1000, 10000])
    def test_check_peak_against_threshold_lower(self, direction, num_rec_multiplier, array_samples):
        """
        Test every input (linear, curved, rms) to this function that determines whether data is within threshold-lower
        """
        # Test linear threshold
        peak_im = 10.00001 * direction
        peak_idx = 50 if array_samples > 1 else 0
        rec_to_test = 5
        array_rec_size = (rec_to_test * num_rec_multiplier) + 1

        run_settings = {
            "direction": direction,
            "threshold_type": "manual",
            "threshold_lower": [10 * direction],
            "rec": rec_to_test,
        }
        assert voltage_calc.check_peak_against_threshold_lower(peak_im, peak_idx, run_settings)

        run_settings["threshold_lower"] = [10.00002 * direction]
        assert not voltage_calc.check_peak_against_threshold_lower(peak_im, peak_idx, run_settings)

        # test curved threshold
        run_settings["threshold_type"] = "curved"
        run_settings["threshold_lower"] = np.random.random((array_rec_size, array_samples))
        run_settings["threshold_lower"][rec_to_test][peak_idx] = 10 * direction
        assert voltage_calc.check_peak_against_threshold_lower(peak_im, peak_idx, run_settings)

        run_settings["threshold_lower"][rec_to_test][peak_idx] = 10.00002 * direction
        assert not voltage_calc.check_peak_against_threshold_lower(peak_im, peak_idx, run_settings)

        # test rms
        run_settings["threshold_type"] = "rms"
        run_settings["threshold_lower"] = {"baseline": np.random.random((array_rec_size, array_samples))}
        run_settings["threshold_lower"]["n_times_rms"] = np.random.random(array_rec_size)

        add_func = np.add if direction == -1 else np.subtract
        run_settings["threshold_lower"]["n_times_rms"][rec_to_test] = add_func(
            10, run_settings["threshold_lower"]["baseline"][rec_to_test][peak_idx]
        )
        assert voltage_calc.check_peak_against_threshold_lower(peak_im, peak_idx, run_settings)

        run_settings["threshold_lower"]["n_times_rms"][rec_to_test] = add_func(
            10.0002, run_settings["threshold_lower"]["baseline"][rec_to_test][peak_idx]
        )
        assert not voltage_calc.check_peak_against_threshold_lower(peak_im, peak_idx, run_settings)

    @pytest.mark.parametrize("direction", [-1, 1])
    def test_check_peak_height_threshold(self, direction):
        """
        Test the threshold upper i.e. that the peak of the event (peak im) is less than the cutoff threshold.
        If the direction is negative, the event should be less negative than the threshold to be within threshold.
        If the direction is positive, the event should be less positive than the threshold to be within threshold.
        """
        peak_im = -10.0001 * direction
        peak_thr = -10 * direction
        assert voltage_calc.check_peak_height_threshold(peak_im, peak_thr, direction)

        peak_im = -10.0001 * direction
        peak_thr = -10.0002 * direction
        assert not voltage_calc.check_peak_height_threshold(peak_im, peak_thr, direction)

        peak_im = 10.0001 * direction
        peak_thr = -10.0002 * direction
        assert not voltage_calc.check_peak_height_threshold(peak_im, peak_thr, direction)

        peak_im = -10.0001 * direction
        peak_thr = 10.0002 * direction
        assert voltage_calc.check_peak_height_threshold(peak_im, peak_thr, direction)

    @pytest.mark.parametrize("window", [8, 10000, 10, 100000, 1])
    @pytest.mark.parametrize("direction", [-1, 1])
    def test_find_event_peak_after_smoothing(self, direction, window):
        """
        Make an event where the largest value (30) is a noise peak and the true peak of the event
        is earlier. Check that the true event peak is found after smoothing and that the result is stable
        to large window sizes and direction.
        """
        n_samples = 10000
        data = np.zeros(n_samples)
        time_ = np.arange(0, n_samples)
        # fmt: off
        event = np.array([0, 10, 20, 22, 22, 22, 24, 24, 24, 26, 27, 26, 24, 22, 20, 10, 30, 10, 0,]) * direction
        # fmt: onn
        data[100 : 100 + len(event)] = event
        start_peak_idx = 100 + 16
        true_peak_idx = 100 + 10
        samples_to_smooth = 3

        if (
            window == 1
        ):  # if just searching around the search region. In this case it will smooth across peak +/- 1 then take the first min value. Minor bias in this case then for the earlier sample when all else equal.
            peak_time, peak_idx, peak_im = voltage_calc.find_event_peak_after_smoothing(
                time_, data, start_peak_idx, window, samples_to_smooth, direction
            )
            assert peak_time == start_peak_idx - 1
            assert peak_time == start_peak_idx - 1
            assert np.isclose(
                peak_im, np.mean([10, 30, 10]) * direction, atol=1e-08, rtol=0
            )

        else:
            peak_time, peak_idx, peak_im = voltage_calc.find_event_peak_after_smoothing(
                time_, data, start_peak_idx, window, samples_to_smooth, direction
            )
            assert peak_time == true_peak_idx
            assert peak_time == true_peak_idx
            assert np.isclose(
                peak_im, np.mean([26, 27, 26]) * direction, atol=1e-08, rtol=0
            )

    @pytest.mark.parametrize("slope", np.arange(-100000, 100000, 9999))  # avoid 0
    @pytest.mark.parametrize("baseline", np.arange(-100, 100, 10))
    def test_enhanced_baseline_calculation(self, slope, baseline):
        """
        Ensure that the implementation of Jonas et al (1993) method to find the point where
        the baseline Im meets a straight line drawn through the 20-80 rise time is stable to slope and baseline.
        """
        direction = -1 if slope < 0 else 1

        time_ = np.arange(0, 200)
        sloping_line = np.arange(0, 100) * slope + baseline
        test_data = np.concatenate([np.zeros(100) + baseline, sloping_line])

        peak_im = np.max(test_data) if direction == 1 else np.min(test_data)
        event_info = {"peak": {"idx": 200, "im": peak_im}}
        results = voltage_calc.enhanced_baseline_calculation(
            test_data,
            time_,
            bl_idx=100,
            bl_im=baseline,
            event_info=event_info,
            run_settings={"direction": direction},
        )

        assert results["idx"] == 100
        assert results["time"] == 100
        assert results["im"] == baseline

    @pytest.mark.parametrize("direction", [-1, 1])
    @pytest.mark.parametrize("scaling", ["over_zero", "positive", "negative", "natural"])
    def test_calculate_event_baseline_from_thr(self, direction, scaling):
        """
        This tests the method which finds the baseline as the first sample prior to crossing a pre-set threshold.
        Model the event as a parabola (no real reason except for fun) and take the baseline as intersecting the parabola or
        beign just below / above. In this case the function will take the nearest sample to the baseline.

        Scale the y axis so that the event goes from neg to pos, is all pos or all neg just to model every eventuality.
        Test positive and negative events.

        Find the true baseline by finding the first sample the parabola crosses the intersecting baseline and
        minus 1, this is easier than finding the first sample prior to the crossing.

        Also test a massive window to ensure the idx cutoff is working well.
        """
        n_samples = 10000
        time_ = np.linspace(-50, 50, 10000)

        change_direction = -1 if direction == 1 else 1
        parabola = (1 * (time_ - 1) ** 2 + 1) * change_direction

        if scaling == "over_zero":
            parabola -= np.mean(parabola)
        elif scaling == "positive":
            parabola += 1000000
        elif scaling == "negative":
            parabola -= 1000000

        # With a baseline that intersects the event
        peak_idx = np.argmax(parabola) if direction == 1 else np.argmin(parabola)
        baseline = np.min(parabola) + (np.max(parabola) - np.min(parabola)) / 2

        bl_idx, bl_time, bl_im = voltage_calc.calculate_event_baseline_from_thr(
            time_,
            data_array=parabola,
            thr_im=baseline,
            peak_idx=peak_idx,
            window=n_samples,
            direction=direction,
        )

        compare_func = (
            np.greater if direction == 1 else np.less
        )  # it is much easier to find the first sample over the threshold then subtract 1 to get the first under it
        true_bl_idx = np.min(np.where(compare_func(parabola, baseline))) - 1
        assert bl_idx == true_bl_idx
        assert bl_time == time_[true_bl_idx]
        assert bl_im == parabola[true_bl_idx]

        # Try a baseline that does not intersect the event
        baseline = np.min(parabola) - 1 if direction == -1 else np.max(parabola) + 1
        bl_idx, bl_time, bl_im = voltage_calc.calculate_event_baseline_from_thr(
            time_,
            data_array=parabola,
            thr_im=baseline,
            peak_idx=peak_idx,
            window=n_samples,
            direction=direction,
        )

        true_bl_idx = np.argmin(parabola) if direction == -1 else np.argmax(parabola)
        assert bl_idx == true_bl_idx
        assert bl_time == time_[true_bl_idx]
        assert bl_im == parabola[true_bl_idx]

    @pytest.mark.parametrize("execution_number", range(100))
    def test_average_baseline_period(self, execution_number):
        """
        Test baseline averaging. First test where the window goes over the first sample to check the indexing window down works okay.
        Then check and random position of the baseline and random window size.
        Finally, check a really small, intuitive example to be sure.
        """
        n_samples = 1000000
        data = np.random.random(n_samples) * np.random.randint(-500, 500)

        baseline_idx = 5
        samples_to_average = 10
        smoothed_bl = voltage_calc.average_baseline_period(data, baseline_idx, samples_to_average)
        assert smoothed_bl == np.mean(data[0 : baseline_idx + 1])

        baseline_idx = np.random.randint(6, n_samples)
        samples_to_average = np.random.randint(
            0, baseline_idx
        )  # make sure sample doesn't cross zero that is tested above
        smoothed_bl = voltage_calc.average_baseline_period(data, baseline_idx, samples_to_average)
        assert smoothed_bl == np.mean(data[baseline_idx - samples_to_average : baseline_idx + 1])

        data = np.array([1, 2, 3, 4, 5])
        smoothed_bl = voltage_calc.average_baseline_period(data, bl_idx=4, samples_to_average=5)
        assert smoothed_bl == np.mean(data)

    @pytest.mark.parametrize("direction", [-1, 1])
    def test_update_baseline_that_is_before_previous_event_peak(self, direction):  # test direction
        """
        Quick toy example to check that if the detected baseline is before the previous event peak, move it to the min / max
        sample between the two peaks (for pos and neg direction respectively)
        """
        time_ = np.linspace(0, 1, 11)
        data = np.array([1, 1, 10, 5, 3, 1, 3, 5, 10, 1, 1]) * direction
        peak_idx = 8
        run_settings = {"previous_event_idx": 2, "direction": direction}
        (
            bl_idx,
            bl_time,
            bl_im,
        ) = voltage_calc.update_baseline_that_is_before_previous_event_peak(data, time_, peak_idx, run_settings)

        assert bl_idx == 5
        assert bl_time == time_[5]
        assert bl_im == data[5]

    @pytest.mark.parametrize("direction", [-1, 1])
    @pytest.mark.parametrize(
        "next_event_position",
        [
            "None",
            "None_window_extend_past_end",
            "before_peak",
            "after_peak_within_window",
            "after_peak",
        ],
    )
    def test_calculate_event_decay_point_entire_search_region(self, direction, next_event_position):
        """
        Test every possible option for where the next event index could be, including check for when search window runs off
        end of the data
        """
        n_samples = 100000
        time_ = np.linspace(0, 100, n_samples)
        data = np.random.random(n_samples)
        peak_idx = len(data) - 10 if next_event_position == "None_window_extend_past_end" else 5000
        window = 1000
        bl_im = np.mean(data)

        baseline_idx = {
            "before_peak": 2500,
            "after_peak": 7500,
            "after_peak_within_window": int(peak_idx + window / 2),
            "None": None,
            "None_window_extend_past_end": None,
        }
        next_event_baseline_idx = baseline_idx[next_event_position]

        run_settings = {
            "direction": direction,
            "next_event_baseline_idx": next_event_baseline_idx,
        }

        (
            decay_idx,
            decay_time,
            decay_im,
        ) = voltage_calc.calculate_event_decay_point_entire_search_region(
            time_, data, peak_idx, window, run_settings, bl_im
        )
        if next_event_position == "before_peak":
            assert peak_idx < decay_idx < peak_idx + window
        elif next_event_position == "after_peak":
            assert decay_idx == peak_idx + window
        elif next_event_position == "after_peak_within_window":
            assert decay_idx == next_event_baseline_idx - 1
        elif next_event_position == "None_window_extend_past_end":
            assert decay_idx == len(data) - 1
        elif next_event_position == "None":
            assert decay_idx == peak_idx + window

        assert decay_time == time_[decay_idx]
        assert decay_im == data[decay_idx]

    @pytest.mark.parametrize("direction", [-1, 1])
    @pytest.mark.parametrize("offset", [10000, -10000, 0])
    @pytest.mark.parametrize(
        "mode",
        [
            "sample_cross_baseline",
            "no_sample_cross_baseline_above",
            "no_sample_cross_baseline_below",
        ],
    )
    @pytest.mark.parametrize("shift_decay_1_idx_with_next_event", [True, False])
    def test_decay_endpoint_improved_method____calculate_event_decay_point_crossover_methods___decay_point_first_crossover_method(
        self, direction, offset, mode, shift_decay_1_idx_with_next_event
    ):
        """
        Test the voltage calc functions:
            decay_endpoint_improved_method()
            calculate_event_decay_point_crossover_methods()
            decay_point_first_crossover_method()

        These are nested, and are tested in reverse-nested order i.e. lowest level first. If this passes, the next method up is tested,
        and finally the top-level method.  These functions determine the event endpoint as the first crossing of the baseline Im, or nearest
        value if the baseline Im does not intersect the event.

        This tests that the bl_im which is set to either intersect at Im = 20, (decay end = 19) or when the baseline is entirely
        above or below the data (in which case it is min / max value of the event depending on the direction).
        """
        # Test decay_endpoint_improved_method() ------------------------------------------------------------------------------------------------------

        event = np.array([100, 80, 60, 40, 20, 19, 0]) * direction

        bl_im_opts = {
            "sample_cross_baseline": 20 * direction,
            "no_sample_cross_baseline_above": 120 * direction,
            "no_sample_cross_baseline_below": -20 * direction,
        }
        bl_im = bl_im_opts[mode]

        event += offset
        bl_im += offset

        decay_idx = voltage_calc.decay_endpoint_improved_method(
            event, peak_idx=0, bl_im=bl_im, direction=direction
        )  # add peak index will just give offset

        correct_indexes = {
            "sample_cross_baseline": 5,
            "no_sample_cross_baseline_above": 0,
            "no_sample_cross_baseline_below": 6,
        }

        assert correct_indexes[mode] == decay_idx

        # Test calculate_event_decay_point_crossover_methods() ---------------------------------------------------------------------------------------
        # This function acts on the entire Im trace (not just the event)

        time_ = np.linspace(0, 100, 1000)
        data = np.ones(1000) * offset
        data[100 : 100 + len(event)] = event

        (
            decay_idx,
            decay_time,
            decay_im,
        ) = voltage_calc.calculate_event_decay_point_crossover_methods(
            time_,
            data,
            peak_idx=100,
            bl_im=bl_im,
            direction=direction,
            window=1000000,  # use a window that goes off the end to test
            use_legacy=False,
        )
        data_decay_idx = correct_indexes[mode] + 100
        assert decay_idx == data_decay_idx
        assert decay_time == time_[data_decay_idx]
        assert decay_im == data[data_decay_idx]

        # Test decay_point_first_crossover_method() --------------------------------------------------------------------------------------------------
        # This function additionally checks the position of the next event

        if shift_decay_1_idx_with_next_event:
            run_settings = {"next_event_baseline_idx": 105, "direction": direction}
        else:
            run_settings = {"next_event_baseline_idx": 108, "direction": direction}

        (
            decay_idx,
            decay_time,
            decay_im,
        ) = voltage_calc.decay_point_first_crossover_method(
            time_,
            data,
            peak_idx=100,
            window=1000000,
            run_settings=run_settings,
            bl_im=bl_im,
        )
        if shift_decay_1_idx_with_next_event and mode != "no_sample_cross_baseline_above":
            data_decay_idx = run_settings["next_event_baseline_idx"] - 1
        else:
            data_decay_idx = correct_indexes[mode] + 100

        assert decay_idx == data_decay_idx
        assert decay_time == time_[data_decay_idx]
        assert decay_im == data[data_decay_idx]

    @pytest.mark.parametrize("direction", [-1, 1])
    @pytest.mark.parametrize("offset", [10000, -10000, None])
    @pytest.mark.parametrize(
        "mode",
        [
            "sample_cross_baseline",
            "no_sample_cross_baseline_above",
            "no_sample_cross_baseline_below",
        ],
    )
    def test_find_first_baseline_crossing(self, direction, offset, mode):
        """
        Find the first datapoint at which the baseline is crossed (for calculation of event endpoint). Should return None
        if no point crosses the baseline. The baseline could intersect the data, or be entirely above / below the data.
        If all samples are above the baseline, should return the first.
        """
        data = np.array([0, 0, -10, 0, 10, 0])

        if offset:
            data += offset

        bl_im_opts = {
            "sample_cross_baseline": np.mean(data),
            "no_sample_cross_baseline_above": -20 * direction,
            "no_sample_cross_baseline_below": 20 * direction,
        }
        bl_im = bl_im_opts[mode]

        first_baseline_idx = voltage_calc.find_first_baseline_crossing(data, bl_im, direction)

        correct_indexes = {
            "sample_cross_baseline": 4 if direction == -1 else 2,
            "no_sample_cross_baseline_above": False if direction == -1 else 0,
            "no_sample_cross_baseline_below": 0 if direction == -1 else False,
        }

        assert correct_indexes[mode] == first_baseline_idx

    @pytest.mark.parametrize("percent", np.arange(1, 100, 5))
    @pytest.mark.parametrize("direction", [-1, 1])
    @pytest.mark.parametrize("smooth", [True, False])
    def test_calclate_decay_percentage_peak_from_smoothed_decay(self, smooth, direction, percent):
        """
        Check the % peak from the decay is calculated correctly. Model the decay as a straight line idx 1-0. Then the desired
        decay % should be the index of the decay %. Tested with a exponential function also below.
        """
        data = np.arange(1, 101).astype(float) * direction
        time_ = np.linspace(5, 6, 100)

        if smooth:
            data += np.random.randn(len(data))
            smooth_window_samples = 3
        else:
            smooth_window_samples = 1

        (
            decay_percent_time,
            decay_percent_im,
            decay_time_ms,
            raw_smoothed_decay_time,
            raw_smoothed_decay_im,
        ) = voltage_calc.calclate_decay_percentage_peak_from_smoothed_decay(
            time_,
            data,
            peak_idx=0,
            decay_idx=len(data) - 1,
            bl_im=0,
            smooth_window_samples=smooth_window_samples,
            ev_amplitude=100,
            amplitude_percent_cutoff=percent,
            interp=False,
        )
        smoothed_data = voltage_calc.quick_moving_average(data, smooth_window_samples)
        test_data = data if not smooth else smoothed_data
        decay_idx = np.argmin(abs(percent - test_data))

        assert decay_percent_time == time_[decay_idx]
        assert decay_percent_im == test_data[decay_idx]
        assert decay_time_ms == (decay_percent_time - time_[0]) * 1000
        assert np.array_equal(raw_smoothed_decay_time, time_)
        assert np.array_equal(raw_smoothed_decay_im, smoothed_data)

    @pytest.mark.parametrize("percent", np.arange(1, 100, 5))
    @pytest.mark.parametrize("direction", [-1, 1])
    @pytest.mark.parametrize("offset", [-100000, 0, 100000])
    def test_find_nearest_decay_sample_to_amplitude(self, percent, direction, offset):
        """
        Very similar to test_calculate_decay_percentage_peak_from_smoothed_decay(), do with exp just for change of pace
        """
        time_ = np.linspace(0, 100, 100)  # the data is scaled between 0 - 100
        data = offset + 100 * np.exp(-time_ / 10) * direction

        (
            decay_percent_time,
            decay_percent_im,
            decay_time_ms,
        ) = voltage_calc.find_nearest_decay_sample_to_amplitude(
            time_,
            data,
            peak_time=0,
            bl_im=offset,
            ev_amplitude=np.min(data) - offset if direction == -1 else np.max(data) - offset,
            amplitude_percent_cutoff=percent,
        )
        if direction == -1:
            percent_point = (percent - offset) * -1
        else:
            percent_point = offset + percent

        data_idx = np.argmin(np.abs(percent_point - data))
        assert decay_percent_time == time_[data_idx]
        assert decay_percent_im == data[data_idx]
        assert decay_time_ms == decay_percent_time * 1000

    def test_quick_moving_average(self):
        """
        See method doc for details. Test against expected output so the method of smoothing is extremely clear.
        """

        def m(nums):
            return np.mean(nums)

        data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Odd Windows
        window = 3
        correct_output = np.array(
            [
                m([0, 0, 1]),
                m([0, 1, 2]),
                m([1, 2, 3]),
                m([2, 3, 4]),
                m([3, 4, 5]),
                m([4, 5, 6]),
                m([5, 6, 7]),
                m([6, 7, 8]),
                m([7, 8, 9]),
                m([8, 9, 10]),
                m([9, 10, 10]),
            ]
        )
        test_smoothed_data = voltage_calc.quick_moving_average(data, window)
        assert utils.allclose(correct_output, test_smoothed_data, 1e-08)

        window = 5
        correct_output = np.array(
            [
                m([0, 0, 0, 1, 2]),
                m([0, 0, 1, 2, 3]),
                m([0, 1, 2, 3, 4]),
                m([1, 2, 3, 4, 5]),
                m([2, 3, 4, 5, 6]),
                m([3, 4, 5, 6, 7]),
                m([4, 5, 6, 7, 8]),
                m([5, 6, 7, 8, 9]),
                m([6, 7, 8, 9, 10]),
                m([7, 8, 9, 10, 10]),
                m([8, 9, 10, 10, 10]),
            ]
        )
        test_smoothed_data = voltage_calc.quick_moving_average(data, window)
        assert utils.allclose(correct_output, test_smoothed_data, 1e-08)

        # Even Windows
        window = 4
        correct_output = np.array(
            [
                m([0, 0, 0, 1]),
                m([0, 0, 1, 2]),
                m([0, 1, 2, 3]),
                m([1, 2, 3, 4]),
                m([2, 3, 4, 5]),
                m([3, 4, 5, 6]),
                m([4, 5, 6, 7]),
                m([5, 6, 7, 8]),
                m([6, 7, 8, 9]),
                m([7, 8, 9, 10]),
                m([8, 9, 10, 10]),
            ]
        )
        test_smoothed_data = voltage_calc.quick_moving_average(data, window)
        assert utils.allclose(correct_output, test_smoothed_data, 1e-08)

        window = 6
        correct_output = np.array(
            [
                m([0, 0, 0, 0, 1, 2]),
                m([0, 0, 0, 1, 2, 3]),
                m([0, 0, 1, 2, 3, 4]),
                m([0, 1, 2, 3, 4, 5]),
                m([1, 2, 3, 4, 5, 6]),
                m([2, 3, 4, 5, 6, 7]),
                m([3, 4, 5, 6, 7, 8]),
                m([4, 5, 6, 7, 8, 9]),
                m([5, 6, 7, 8, 9, 10]),
                m([6, 7, 8, 9, 10, 10]),
                m([7, 8, 9, 10, 10, 10]),
            ]
        )
        test_smoothed_data = voltage_calc.quick_moving_average(data, window)
        assert utils.allclose(correct_output, test_smoothed_data, 1e-08)

    @pytest.mark.parametrize("demean", [True, False])
    @pytest.mark.parametrize("offset", [-10000, 0, 10000])
    @pytest.mark.parametrize("direction", [-1, 1])
    def test_normalise_amplitude_1d(self, direction, offset, demean):
        """
        Test normalise amplitude with a slightly different formula
        """
        x = np.linspace(0, 30, 500)
        y = core_analysis_methods.biexp_event_function(x, (offset, direction, 0.5, 5))

        mean_y = np.mean(y)
        offset = mean_y if demean else 0

        norm_y = voltage_calc.normalise_amplitude(y, demean)
        test_norm_y = (y - offset) / (np.max(y) - np.min(y))

        assert abs(np.max(norm_y) - np.min(norm_y)) == 1
        assert utils.allclose(norm_y, test_norm_y, 1e-10)

        if demean:
            assert utils.allclose(np.mean(norm_y), 0)

    @pytest.mark.parametrize("demean", [True, False])
    @pytest.mark.parametrize("offset", [-10000, 0, 10000])
    @pytest.mark.parametrize("direction", [-1, 1])
    def test_normalise_amplitude_2d(self, direction, offset, demean):
        """
        Test normalise amplitude with a slightly different formula
        """
        x = np.linspace(0, 30, 500)
        y = core_analysis_methods.biexp_event_function(x, (offset, direction, 0.5, 5))

        y = np.vstack([y,
                       y + 10])

        mean_y = np.mean(y, axis=1, keepdims=True)
        offset = mean_y if demean else np.vstack([0, 0])

        norm_y = voltage_calc.normalise_amplitude(y, demean)
        test_norm_y = (y - offset) / (np.max(y, axis=1, keepdims=True) - np.min(y, axis=1, keepdims=True))

        assert abs(np.max(norm_y[0, :]) - np.min(norm_y[0, :])) == 1
        assert abs(np.max(norm_y[1, :]) - np.min(norm_y[1, :])) == 1
        assert utils.allclose(norm_y, test_norm_y, 1e-10)

        if demean:
            assert utils.allclose(np.mean(norm_y[0, :]), 0)
            assert utils.allclose(np.mean(norm_y[1, :]), 0)

    @pytest.mark.parametrize(
        "param",
        [
            "bl_percentile",
            "weight_distance_from_peak",
            "decay_period_to_smooth",
            "event_peak_smoothing",
            "foot_detection_low_rise_percent",
            "foot_detection_high_rise_percent",
            "deconv_peak_search_region_multiplier",
        ],
    )
    def test_consts(self, param):
        lookup_table = {
            "bl_percentile": 60,
            "weight_distance_from_peak": 5,
            "decay_period_to_smooth": 3,
            "event_peak_smoothing": 3,
            "foot_detection_low_rise_percent": 20,
            "foot_detection_high_rise_percent": 80,
            "deconv_peak_search_region_multiplier": 3,
        }

        assert lookup_table[param] == voltage_calc.consts(param)
