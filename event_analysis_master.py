"""
Copyright © Joseph John Ziminski 2020-2021.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy.signal
from ephys_data_methods import ap_detection_methods, core_analysis_methods, voltage_calc
from utils import utils

if TYPE_CHECKING:
    from custom_types import Direction, FalseOrDict, FalseOrFloat, Int, NpArray64
    from numpy.typing import NDArray

# --------------------------------------------------------------------------------------
# Fit Sliding Window
# --------------------------------------------------------------------------------------


def fit_sliding_window(
    data_array: NpArray64, run_settings: Dict, progress_bar_callback: Callable
) -> Tuple[NpArray64, NpArray64]:
    """
    Calculate the sliding window correlation or detection criterion using the
    algorithm of Clements and Betters (1997).

    Due to the implementation of the sliding window method the template b1 must be
    forced positive. Don't normalise the template, does not affect the cutoff vs.
    data detection criterion scaling but makes the show-fit plot on the data
    incorrect.
    """
    template = make_template_from_run_settings(run_settings, override_b1=1, normalise=False)

    detection_criterion, betas, r = voltage_calc.clements_bekkers_sliding_window(
        data_array, template, progress_bar_callback
    )

    detection_coefs = r if run_settings["detection_threshold_type"] == "correlation" else abs(detection_criterion)

    return betas, detection_coefs


def deconvolution_template_detection(data: NpArray64, run_settings: Dict, u: Callable) -> NpArray64:
    """
    Run deconvolution event detection method. See
    get_filtered_template_data_deconvolution() for details

    The template is (b1 = -1 or 1) and first scaled to the
    peak amplitude of the data

    INPUTS:
           data: 1 x N data
           run_settings: dictionary of options
           u: progress bar callback function
    """
    data_range = (np.min(data) - np.max(data)) if run_settings["direction"] == -1 else (np.max(data) - np.min(data))

    template = make_template_from_run_settings(run_settings, normalise=True)
    template *= abs(data_range)

    u()

    deconv = voltage_calc.get_filtered_template_data_deconvolution(
        data,
        template,
        run_settings["fs"],
        run_settings["deconv_options"]["filt_low_hz"],
        run_settings["deconv_options"]["filt_high_hz"],
    )
    return deconv


def make_template_from_run_settings(
    run_settings: Dict,
    override_b1: Union[Literal[False], Int, np.float64] = False,
    normalise: bool = False,
) -> NpArray64:
    """
    Generate the biexponential event template based on user-input options.

    override_b1: can override the b1 for the template, otherwise the direction (-1 or
                1) will be used

    normalise: if true, template will be normalised 0 = 1 or -1 (depending on b1)
    """
    W = int(run_settings["window_len_s"] / run_settings["ts"])
    x = core_analysis_methods.generate_time_array(0.0, run_settings["window_len_s"], W, run_settings["ts"])

    b0 = 0
    b1 = run_settings["direction"] if override_b1 is False else override_b1
    coefs = (b0, b1, run_settings["rise_s"], run_settings["decay_s"])
    template = core_analysis_methods.biexp_event_function(x, coefs)

    if normalise:
        template = normalise_template(template, run_settings["direction"])

    return template


def normalise_template(data: NpArray64, direction: Direction) -> NpArray64:
    """
    Normalise the template to the range 0 - 1, or 0 - -1 depending on direction
    """
    min_ = np.min(data)
    max_ = np.max(data)

    if direction == 1:
        norm_data = data - np.min(data)
        norm_data /= abs(max_ - min_)

    elif direction == -1:
        norm_data = data - np.max(data)
        norm_data /= abs(min_ - max_)

    return norm_data


# --------------------------------------------------------------------------------------
# Events Threshold
# --------------------------------------------------------------------------------------


def events_thresholding_get_putative_peaks(
    data_array: NpArray64, min_distance_between_maxima_samples: Int, direction: Direction
) -> Union[Literal[False], NDArray[np.integer]]:
    """
    Find peaks using the thresholding method.

    Make all events positive before passing to scipy find_peaks.
    """
    putative_peak_idxs = scipy.signal.find_peaks(data_array * direction, distance=min_distance_between_maxima_samples)[
        0
    ]

    if not np.any(putative_peak_idxs):
        return False

    return putative_peak_idxs


# --------------------------------------------------------------------------------------
# Peak Processing
# --------------------------------------------------------------------------------------


def calculate_event_peaks(
    detection_coefs: NpArray64,
    betas: Optional[NpArray64],
    time_array: NpArray64,
    data_array: NpArray64,
    run_settings: Dict,
) -> Dict:
    """
    Threshold events from their peak and correlation values and save them to
    self.event_info dict.

    After sliding window fit, we have a correlation value at each timepoint for the
    correlation of the biexp fit to the data. Next we want to calculate the contiguous
    events and their peak.

    First, only beta values in the direction of the expected events are kept.
    Next, only Im points that after over the correlation threshold are kept.

    We now have a situation where consecutive events are retained but sometimes padded
    with nans depending on the odd correlation value that may not have been over
    threshold e.g. [nan nan -1 -20 -30 -40 -30 -20 nan -1 nan nan]. We need to smooth
    over these gaps and then find the peak of all contiguous events.  A smoothing window
    of half the sliding window works well across event types.

    Now we are left with chunked contiguous events stored in the above_threshold_ev.
    We finally cycle through these, Find their peak and other information and check
    they are above our linear or dynamic Im threshold. If so save to self.event_info.
    """
    thr = get_detection_threshold(run_settings)

    detection_coefs = null_wrong_direction_peaks(detection_coefs, betas, run_settings)

    if run_settings["detection_threshold_type"] == "deconvolution":
        peak_idxs = ap_detection_methods.index_peaks_above_threshold(detection_coefs, detection_coefs, thr, "positive")
        peak_idxs = get_peaks_for_deconvolution(peak_idxs, data_array, run_settings)
    else:
        smoothing_window = int(run_settings["window_len_samples"] * 0.5)
        event_dir = "positive" if run_settings["direction"] == 1 else "negative"

        peak_idxs = ap_detection_methods.index_peaks_above_threshold(
            detection_coefs, data_array, thr, event_dir, smoothing_window
        )

    # If direction is negative, flip the Im so max peaks are detected
    peak_idxs = core_analysis_methods.exclude_peak_idxs_based_on_local_maximum_period(
        peak_idxs,
        data_array[peak_idxs] * run_settings["direction"],
        run_settings["min_distance_between_maxima_samples"],
    )

    event_info = make_peak_event_info_from_peak_idxs(time_array, data_array, peak_idxs, run_settings)

    return event_info


def get_peaks_for_deconvolution(
    deconv_peak_idxs: NDArray[np.integer],
    data_array: NpArray64,
    run_settings: Dict,
) -> NDArray[np.integer]:
    """
    Find all the peaks above threshold in the deconvolution trace.

    Then find the nearest peak in the data following the deconv peak within
    3x ( "deconv_peak_search_region_multiplier") the number of samples from
    baseline to peak.
    """
    peak_find_func = np.argmin if run_settings["direction"] == -1 else np.argmax
    bl_to_peak = estimate_bl_to_peak_from_template(run_settings)
    search_region = int(bl_to_peak) * voltage_calc.consts("deconv_peak_search_region_multiplier")

    # cut idx that are out of data range,
    # reshape into peak_num x peak: peak + search region indices
    deconv_peak_idxs = np.delete(deconv_peak_idxs, np.where(deconv_peak_idxs + search_region > len(data_array)))
    deconv_peak_idxs = np.atleast_2d(deconv_peak_idxs).T
    peak_search_region = deconv_peak_idxs + np.arange(search_region)

    ev_peak_idxs = peak_find_func(data_array[peak_search_region], axis=1)

    peak_idxs = deconv_peak_idxs.squeeze() + ev_peak_idxs

    return peak_idxs


def make_peak_event_info_from_peak_idxs(
    time_: NpArray64,
    data: NpArray64,
    peak_idxs: Union[List[Int], NDArray[np.integer]],
    run_settings: Dict,
) -> Dict:
    """
    From an array of peak indices perform checks for threshold_lower, threshold_upper
    and omit times.

    See check_putatitve_event() for details on checks. The extended event_info that
    is used for full analysis can be found at cfgs.make_event_info_dict()

    Ued both for events analysis and curve_fitting analysis fit biexpoential_event
    For curve fitting and average event, omitting times / baseline settings for whole
    record are not required.

    TODO
    ----
    `direction` and `records_are_contiguous` should be moved to analysis-wide property
    rather than an individual event property, as is the same for all events.
    Once one more cases like this occurs, perform a big refactor of
    `event_info` to contain a analysis-wide property field.
    """
    event_info = {}
    for peak_idx in peak_idxs:
        peak_time = time_[peak_idx]
        peak_im = data[peak_idx]
        template_num = run_settings["template_num"] if "template_num" in run_settings else None

        if run_settings["average_peak_points"]["on"]:
            peak_idx, peak_time, peak_im = smooth_peak(time_, data, peak_idx, run_settings)

        if run_settings["name"] == "event_kinetics":
            if "manual_select" not in run_settings or run_settings["manual_select"]["use_thresholding"]:
                success = check_putative_event(peak_idx, peak_time, peak_im, time_, run_settings)

                if not success:
                    continue

        event_info[str(peak_time)] = {
            "peak": {
                "time": peak_time,
                "im": peak_im,
                "idx": peak_idx,
                "template_num": template_num,
                "direction": run_settings["direction"],
                "records_are_contiguous": run_settings["records_are_contiguous"],
            }
        }

    return event_info


def null_wrong_direction_peaks(detection_coefs: NpArray64, betas: Optional[NpArray64], run_settings: Dict) -> NpArray64:
    """
    Cut anywhere the detection coefficients are high in a region where
    the beta are very negative, we only want positive correlation.
    """
    if run_settings["detection_threshold_type"] == "deconvolution":
        pass
    else:
        assert betas is not None, "Type Narrowing check betas"

        if run_settings["direction"] == 1:
            incorrect_direction = betas[1] < 0
        elif run_settings["direction"] == -1:
            incorrect_direction = betas[1] > 0

        detection_coefs[incorrect_direction] = 0

    return detection_coefs


def get_detection_threshold(run_settings: Dict) -> NpArray64:
    if run_settings["detection_threshold_type"] == "correlation":
        thr = run_settings["corr_cutoff"]
    elif run_settings["detection_threshold_type"] == "detection_criterion":
        thr = run_settings["detection_criterion"]
    elif run_settings["detection_threshold_type"] == "deconvolution":
        thr = run_settings["deconv_options"]["detection_threshold"]

    return thr


def estimate_bl_to_peak_from_template(run_settings: Dict) -> Union[Int, NpArray64]:
    peak_find_func = np.argmin if run_settings["direction"] == -1 else np.argmax
    bl_to_peak = peak_find_func(make_template_from_run_settings(run_settings))
    return bl_to_peak


# Peak smoothing and checking
# --------------------------------------------------------------------------------------


def smooth_peak(
    time_: NpArray64, data: NpArray64, peak_idx: Int, run_settings: Dict
) -> Tuple[Int, NpArray64, NpArray64]:
    """
    Smooth the peak region and then re-find peak. See
    voltage_calc.find_event_peak_after_smoothing()
    """
    window = np.floor(
        core_analysis_methods.quick_get_time_in_samples(run_settings["ts"], run_settings["decay_search_period_s"]) / 4
    ).astype(int)
    window = 1 if window == 0 else window
    samples_to_smooth = core_analysis_methods.quick_get_time_in_samples(
        run_settings["ts"], run_settings["average_peak_points"]["value_s"]
    )

    peak_idx, peak_time, peak_im = voltage_calc.find_event_peak_after_smoothing(
        time_, data, peak_idx, window, samples_to_smooth, run_settings["direction"]
    )
    return peak_idx, peak_time, peak_im


def check_putative_event(
    peak_idx: Int, peak_time: NpArray64, peak_im: NpArray64, time_: NpArray64, run_settings: Dict
) -> bool:
    """
    Check if an event peak is within user-specified thresholds
    (e.g. threshold lower, omit times). Return False if event
    should be excluded. See voltage_calc methods for details.

    Within thresholds:
        threshold_lower - check the event is above the lower threshold
                          (this threshold might be linear, curve or drawn).
        threshold higher - check the event is under the higher
                           threshold (this is input via spinbox)
    """
    if np.any(run_settings["omit_start_stop_times"]):

        if run_settings["omit_start_stop_include_exclude"] == "exclude":
            for start, stop in run_settings["omit_start_stop_times"]:

                start_time, stop_time = get_events_omit_start_stop_time(start, stop, run_settings, time_)
                if start_time < peak_time < stop_time:
                    return False
        else:
            include_periods = []
            for start, stop in run_settings["omit_start_stop_times"]:

                start_time, stop_time = get_events_omit_start_stop_time(start, stop, run_settings, time_)
                if start_time < peak_time < stop_time:
                    include_periods.append(True)
                else:
                    include_periods.append(False)

            if not any(include_periods):
                return False

    within_threshold = voltage_calc.check_peak_against_threshold_lower(peak_im, peak_idx, run_settings)
    if not within_threshold:
        return False

    if run_settings["threshold_upper_limit_on"]:
        within_threshold = voltage_calc.check_peak_height_threshold(
            peak_im,
            run_settings["threshold_upper_limit_value"],
            run_settings["direction"],
        )
    if not within_threshold:
        return False

    return True


def get_events_omit_start_stop_time(start, stop, run_settings, time_):
    start_time = start if run_settings["omit_start_stop_mode"] == "absolute" else time_[0] + start
    stop_time = stop if run_settings["omit_start_stop_mode"] == "absolute" else time_[0] + stop
    return start_time, stop_time


# ---------------------------------------------------------------------------------------
# RMS
# --------------------------------------------------------------------------------------


def calculate_rms(data: NpArray64, baseline: NpArray64) -> Tuple[NpArray64, NpArray64]:
    """ """
    assert isinstance(baseline, float) or baseline.size == 1 or data.size == baseline.size, (
        "`data` and `baseline` must be of same size for RMS calculation, " "or `baseline` must be scalar."
    )
    mse = np.sum((baseline - data) ** 2) / data.size
    rms = np.sqrt(mse)
    return rms


# --------------------------------------------------------------------------------------
# Event Kinetics
# --------------------------------------------------------------------------------------


def calculate_event_kinetics(
    time_: NpArray64,
    data: NpArray64,
    peak_idx: Int,
    peak_time: NpArray64,
    peak_im: NpArray64,
    template_num: Int,
    records_are_contiguous: bool,
    run_settings: Dict,
    info: Optional[Dict] = None,
) -> FalseOrDict:
    """
    Calculate all event kinetics the peak info.
    """
    event_info = make_event_info_dict()
    event_info["record_num"]["rec_idx"] = run_settings["rec"]

    if info:
        event_info["info"] = info  # TODO: this is so fucking jenky...

    # Peak
    event_info["peak"] = {
        "idx": peak_idx,
        "time": peak_time,
        "im": peak_im,
        "template_num": template_num,
        "records_are_contiguous": records_are_contiguous,
    }

    # Baseline
    event_info["baseline"] = calculate_event_baseline(time_, data, peak_idx, run_settings, event_info)
    if not event_info["baseline"]:
        return False

    # Amplitude
    event_info["amplitude"] = calculate_and_threshold_amplitude(
        run_settings, event_info["peak"]["im"], event_info["baseline"]["im"]
    )

    if not event_info["amplitude"]:
        return False

    # Event Fitting
    success = caculate_decay_and_fit_monoexp_or_biexp(time_, data, event_info, run_settings)
    if not success:
        return False

    # Rise Time
    event_info["rise"] = calculate_event_rise_time(time_, data, event_info, run_settings)
    if not event_info["rise"]:
        return False

    # Half-width
    event_info["half_width"] = calculate_half_width(time_, data, event_info, run_settings)
    if not event_info["half_width"]:
        return False

    # Max Slope
    calculate_max_slope(time_, data, event_info, run_settings)

    # Area Under Curve (AUC)
    success = calculate_area_under_curve_and_threshold(data, time_, event_info, run_settings)

    if not success:
        return False

    return event_info


# Baseline
# --------------------------------------------------------------------------------------


def calculate_event_baseline(
    time_: NpArray64,
    data: NpArray64,
    peak_idx: Int,
    run_settings: Dict,
    event_info: Dict,
) -> FalseOrDict:
    """
    Calculate an event baseline point from data and current settings.

    There are two main settings - detect baseline per-event or use a pre-defined
    threshold

    Per-event: find_foot method: events may be detected by the "find_foot" method
              which find the baseline by calculating the foot from the peak
               algorithmically. This can then be smoothed by the
               "average_baseline_points" option.

    pre-defined threshold:   This can be either a straight line (linear, basically a
                             single baseline value), a curve or drawn. Whatever the
                             threshold method, the first sample that crosses the
                             threshold in the opposite direction of the peak is
                             used as baseline.

    Smoothing: the baseline_im (bl_im) is adjusted by looking backwards from the
               identified bl idx and smoothing

    New feature, check baseline is not before previous event peak - if it is,
    adjust. Do not do this if legacy baseline detection is switched on.

    See voltage_calc methods for details.
    """
    window = core_analysis_methods.quick_get_time_in_samples(
        run_settings["ts"], run_settings["baseline_search_period_s"]
    )
    window = 1 if window == 0 else window

    if run_settings["edit_kinetics_mode"]:
        baseline_edit_idx = run_settings["edit_kinetics_mode"]["kinetic_clicked"]["baseline"]["idx"]
        if baseline_edit_idx:
            bl_idx = baseline_edit_idx
            bl_time = time_[bl_idx]
            bl_im = data[bl_idx]
            return {"idx": bl_idx, "time": bl_time, "im": bl_im}

    if "from_fit_not_data" in run_settings and run_settings["from_fit_not_data"]:
        bl_idx = 0
        bl_time = time_[bl_idx]
        bl_im = data[bl_idx]

    elif run_settings["baseline_type"] == "per_event":
        bl_idx, bl_time, bl_im = voltage_calc.calculate_event_baseline(
            time_, data, peak_idx, run_settings["direction"], window
        )

    elif run_settings["baseline_type"] in ["manual", "curved", "drawn"]:
        bl_idx, bl_time, bl_im = calculate_event_baseline_from_data_baseline(
            time_, data, peak_idx, window, run_settings
        )
    # Check, handle smoothing, return data
    if bl_idx is False or bl_time is False or bl_im is False or peak_idx - bl_idx < 1:  # type narrowing syntax
        return False

    if bl_idx < 0:  # edge case for events on edge of recording time
        bl_idx = 0

    if not run_settings["legacy_options"]["baseline_method"]:
        if event_baseline_is_before_previous_event_peak(bl_idx, run_settings):
            (
                bl_idx,
                bl_time,
                bl_im,
            ) = voltage_calc.update_baseline_that_is_before_previous_event_peak(data, time_, peak_idx, run_settings)
    if run_settings["average_baseline_points"]["on"]:
        bl_im = average_baseline_points(data, bl_idx, run_settings)

    bl_results = {"idx": bl_idx, "time": bl_time, "im": bl_im}

    if not run_settings["legacy_options"]["baseline_enhanced_position"]:
        bl_results = enhance_baseline_position_and_resmooth_if_required(
            data, time_, bl_idx, bl_im, event_info, run_settings, bl_results
        )
    return bl_results


def enhance_baseline_position_and_resmooth_if_required(
    data: NpArray64,
    time_: NpArray64,
    bl_idx: Int,
    bl_im: NpArray64,
    event_info: Dict,
    run_settings: Dict,
    bl_results: FalseOrDict,
) -> FalseOrDict:
    """
    Improve the foot position of the event. If smoothing is on,
    re-smooth after the baseline position has been adjusted
    """
    bl_results_ = voltage_calc.enhanced_baseline_calculation(data, time_, bl_idx, bl_im, event_info, run_settings)
    if bl_results_ is not False:
        bl_results = bl_results_

        if run_settings["average_baseline_points"]["on"]:
            bl_results_["im"] = average_baseline_points(data, bl_results_["idx"], run_settings)
    return bl_results


def calculate_event_baseline_from_data_baseline(
    time_: NpArray64, data: NpArray64, peak_idx: Int, window: Int, run_settings: Dict
) -> Tuple[Int, NpArray64, NpArray64]:
    """
    Calculate event baseline as the first time the data crosses a data baseline
    ( either linear or dynamic). If it does not cross within the region
    (default half window samples) then use closest within region.

    This is in contrast to voltage_calc.calculate_event_baseline() which calculates
    bl per-event.
    """
    if run_settings["baseline_type"] == "manual":
        thr_im = run_settings["baseline"]

    elif run_settings["baseline_type"] in ["curved", "drawn"]:
        threshold = run_settings["baseline"][run_settings["rec"]]
        if window > peak_idx:  # make sure index not outside available data
            window = peak_idx
        thr_im = threshold[peak_idx - window : peak_idx + 1]

    bl_idx, bl_time, bl_im = voltage_calc.calculate_event_baseline_from_thr(
        time_, data, thr_im, peak_idx, window, run_settings["direction"]
    )
    return bl_idx, bl_time, bl_im


def average_baseline_points(data: NpArray64, bl_idx: Int, run_settings: Dict) -> NpArray64:
    """
    Look back from the baseline idx and average as many samples as
    specified by the user. Set the new bl_im as this (note time will not change).
    """
    samples_to_average = core_analysis_methods.quick_get_time_in_samples(
        run_settings["ts"], run_settings["average_baseline_points"]["value_s"]
    )
    bl_im = voltage_calc.average_baseline_period(data, bl_idx, samples_to_average)
    return bl_im


def event_baseline_is_before_previous_event_peak(bl_idx: Int, run_settings: Dict) -> bool:
    """
    Convenience function to check that the baseline detected is not before the
    previous event peak.
    """
    return run_settings["previous_event_idx"] is not None and bl_idx < run_settings["previous_event_idx"]


# Amplitude
# --------------------------------------------------------------------------------------


def calculate_and_threshold_amplitude(run_settings: Dict, peak_im: NpArray64, bl_im: NpArray64) -> FalseOrDict:
    """
    Threshold event - reject if amplitude too small.
    """
    amplitude_results = {"im": peak_im - bl_im}

    if "manual_select" not in run_settings or run_settings["manual_select"]["use_thresholding"]:
        if np.abs(amplitude_results["im"]) < run_settings["amplitude_threshold"]:
            return False

    return amplitude_results


# --------------------------------------------------------------------------------------
# Decay and Event Fitting
# --------------------------------------------------------------------------------------


def caculate_decay_and_fit_monoexp_or_biexp(
    time_: NpArray64, data: NpArray64, event_info: Dict, run_settings: Dict
) -> Union[bool, None]:
    """
    First calculate the event endpoint, to which the function will be fit / decay %
    parameter searched for.

    Then fit the function based on user settings and update event_info.
    """
    event_info["decay_point"] = calculate_event_endpoint(time_, data, event_info, run_settings)
    if event_info["decay_point"] is False:
        return False

    # Fit function (unless "do not fit")
    fit_method = run_settings["decay_or_biexp_fit_method"]
    if fit_method == "monoexp":
        event_info["monoexp_fit"] = fit_monoexp_function_to_decay(
            time_,
            data,
            event_info["peak"]["idx"],
            event_info["decay_point"]["idx"],
            run_settings,
        )
        event_info_key = "monoexp_fit"

    elif fit_method == "biexp":
        event_info["biexp_fit"] = calculate_biexp_fit_to_event(
            time_,
            data,
            event_info["baseline"]["idx"],
            event_info["decay_point"]["idx"],
            run_settings,
        )
        event_info_key = "biexp_fit"

    if fit_method in ["monoexp", "biexp"]:
        if event_info[event_info_key] is False:
            return False

        # overwrite the end Im point Im to the fit not data, this is cosmetic
        event_info["decay_point"]["im"] = event_info[event_info_key]["fit_im"][-1]

    # Decay Time
    event_info["decay_one_point"], event_info["decay_between_points"] = calculate_decay_time(
        time_, data, event_info, run_settings
    )
    # One is always False / None, if both are False / None then it failed.
    if not event_info["decay_one_point"] and not event_info["decay_between_points"]:
        return

    return True


# Decay Point Time
# --------------------------------------------------------------------------------------


def calculate_event_endpoint(time_: NpArray64, data: NpArray64, event_info: Dict, run_settings: Dict) -> FalseOrDict:
    """
    Find the end of the event. If edit kinetics mode is on and baseline is moved,
    the idx of decay is pre-stored in run_settings so that
    decay endpoint is not re-calculated in case user previously changed it manually.

    Otherwise, calculate the event endpoint based on the users settings.
    """

    if run_settings["edit_kinetics_mode"]:
        decay_edit_idx = run_settings["edit_kinetics_mode"]["kinetic_clicked"]["decay_point"]["idx"]
        if decay_edit_idx:
            decay_idx = decay_edit_idx
            decay_time = time_[decay_idx]
            decay_im = data[decay_idx]
            return {"time": decay_time, "im": decay_im, "idx": decay_idx}

    window = core_analysis_methods.quick_get_time_in_samples(run_settings["ts"], run_settings["decay_search_period_s"])
    if window < 3:
        # Don't let search period be too small or causes errors downstream (e.g.
        # individual event presentation)
        return False

    bl_im = event_info["baseline"]["im"]
    peak_idx = event_info["peak"]["idx"]

    if run_settings["decay_period_type"] == "auto_search_data":
        # must come first as overrides over options
        if run_settings["legacy_options"]["decay_detection_method"]:
            (decay_idx, decay_time, decay_im,) = voltage_calc.calculate_event_decay_point_crossover_methods(
                time_,
                data,
                peak_idx,
                bl_im,
                run_settings["direction"],
                window,
                use_legacy=True,
            )
        else:
            if run_settings["endpoint_search_method"] == "entire_search_region":
                (decay_idx, decay_time, decay_im,) = voltage_calc.calculate_event_decay_point_entire_search_region(
                    time_, data, peak_idx, window, run_settings, bl_im
                )
            elif run_settings["endpoint_search_method"] == "first_baseline_cross":
                (
                    decay_idx,
                    decay_time,
                    decay_im,
                ) = voltage_calc.decay_point_first_crossover_method(time_, data, peak_idx, window, run_settings, bl_im)
        if decay_idx is False:
            return False

    elif run_settings["decay_period_type"] == "use_end_of_region":
        decay_idx = peak_idx + window
        decay_time = time_[decay_idx]
        decay_im = data[decay_idx]

    if decay_idx - peak_idx < 3:
        return False

    decay_point_results = {"time": decay_time, "im": decay_im, "idx": decay_idx}

    return decay_point_results


# Decay fit
# ----------------------------------------------------------------------------------------------------------------------------------------------------


def fit_monoexp_function_to_decay(
    time_: NpArray64, data: NpArray64, peak_idx: int, decay_idx: int, run_settings: Dict
) -> FalseOrDict:
    """
    Fit a monoexponential function to the decay period (data between peak and
    decay endpoint). Note that if fitting fails the event is excluded from
    analysis.

    First, fit a curve between peak and decay point. Then, if user has
    specified to adjust startpoint either using r2 or bounds as a cutoff,
    update the fit by adjusting the startpoint with adjust_fit_start_point().

    Next, perform final checks on r2 and bounds (if user has selected these
    options. For R2, it is excluded if setting "exclude_from_r2_on" is on and
    the R2 of fit is under "exclude_from_r2_value".

    For bounds, the final check on whether the fit parameters are in these
    limits is performed if "exclude_if_params_not_in_bounds" or
    "adjust_startpoint_bounds_on" is on. If the latter, this is in case
    no adjusting was able to bring the fit into bounds.
    """
    opts = run_settings["monoexp_fit"]
    decay_period_time = time_[peak_idx : decay_idx + 1]
    decay_period_data = data[peak_idx : decay_idx + 1]

    coefs, fit, r2 = core_analysis_methods.fit_curve(
        "monoexp", decay_period_time, decay_period_data, run_settings["direction"]
    )
    if coefs is False or fit is False or r2 is False:  # type narrowing syntax
        return False

    monoexp_fit_results = update_fit_results(coefs, r2, decay_period_time, fit, "monoexp")

    if opts["adjust_startpoint_r2_on"] or (
        opts["adjust_startpoint_bounds_on"] and not is_tau_in_limits(run_settings, monoexp_fit_results["tau_ms"])
    ):
        updated_monoexp_fit_results = adjust_fit_start_point(
            time_, data, peak_idx, decay_idx, r2, run_settings, "monoexp"
        )
        if updated_monoexp_fit_results is not None:
            monoexp_fit_results = updated_monoexp_fit_results

    if not is_tau_in_limits(run_settings, monoexp_fit_results["tau_ms"]):
        return False

    if not check_fit_within_r2_bounds(monoexp_fit_results["r2"], "monoexp", run_settings):
        return False

    return monoexp_fit_results


def calculate_biexp_fit_to_event(
    time_: NpArray64, data: NpArray64, bl_idx: int, decay_idx: int, run_settings: Dict
) -> FalseOrDict:
    """
    Fit a biexponential function to the event.

    First, if analysis is Events Template, use the initial estimate for the fit from
    the biexpoential template.

    Otherwise, set to None to use canonical defaults (rise: 0.5 ms, decay: 5ms).
    """
    opts = run_settings["biexp_fit"]
    x_to_fit = time_[bl_idx : decay_idx + 1]
    y_to_fit = data[bl_idx : decay_idx + 1]

    initial_est = get_initial_est_for_biexp_fit(x_to_fit, y_to_fit, run_settings)

    coefs, fit, r2 = core_analysis_methods.fit_curve(
        "biexp_event",
        x_to_fit,
        y_to_fit,
        run_settings["direction"],
        initial_est=initial_est,
    )
    if coefs is False or fit is False or r2 is False:  # type narrowing syntax
        return False

    biexp_fit_results = update_fit_results(coefs, r2, x_to_fit, fit, "biexp")

    if opts["adjust_startpoint_r2_on"] or (
        opts["adjust_startpoint_bounds_on"] and not check_rise_and_decay_in_limits(biexp_fit_results, run_settings)
    ):
        updated_biexp_fit_results = adjust_fit_start_point(
            time_, data, bl_idx, decay_idx, r2, run_settings, "biexp", initial_est
        )
        if updated_biexp_fit_results is not None:
            biexp_fit_results = updated_biexp_fit_results

    if not check_rise_and_decay_in_limits(biexp_fit_results, run_settings):
        return False

    if not check_fit_within_r2_bounds(biexp_fit_results["r2"], "biexp", run_settings):
        return False

    return biexp_fit_results


# Adjusting start point
# --------------------------------------------------------------------------------------


def adjust_fit_start_point(
    time_: NpArray64,
    data: NpArray64,
    start_idx: int,
    decay_idx: int,
    best_r2: NpArray64,
    run_settings: Dict,
    fit_type: str,
    initial_est: Optional[Tuple] = None,
) -> Optional[Dict]:
    """
    Adjust the start-point of the fit to either improve the R2 or
    ensure the fit parameters are within user-specified bounds.

    If option "adjust_startpoint_r2_on" is on, all start points will
    be tried and the fit with the best r2 returned (as event_info dict).
    """
    best_fit_results = None
    config_key = fit_type + "_fit"

    # get startpoint and details from fit type
    if fit_type == "monoexp":
        startpoint_samples = get_adjust_startpoint_based_on_r2_or_bounds(fit_type, run_settings)
        adjust_values = np.array(range(1, startpoint_samples + 1))
        points_to_try = np.concatenate([adjust_values, adjust_values * -1])
        fit_curve_key = "monoexp"

    elif fit_type == "biexp":
        startpoint_samples = get_adjust_startpoint_based_on_r2_or_bounds(fit_type, run_settings)
        points_to_try = range(1, startpoint_samples)
        fit_curve_key = "biexp_event"

    # Index out portions of the data and fit
    for i in points_to_try:  # type: ignore
        try:
            x_to_fit = time_[start_idx + i : decay_idx + 1]
            y_to_fit = data[start_idx + i : decay_idx + 1]
        except IndexError:
            continue

        if len(x_to_fit) < 3:
            continue

        coefs, fit, r2 = core_analysis_methods.fit_curve(
            fit_curve_key,
            x_to_fit,
            y_to_fit,
            run_settings["direction"],
            initial_est=initial_est,
        )
        if coefs is False or fit is False or r2 is False:  # type narrowing syntax
            continue

        # If r2 is set, update with any better r2.
        # If bounds, return if bounds in limits
        if run_settings[config_key]["adjust_startpoint_r2_on"]:
            if r2 > best_r2:
                best_r2 = r2
                best_fit_results = update_fit_results(coefs, r2, x_to_fit, fit, fit_type)

        elif run_settings[config_key]["adjust_startpoint_bounds_on"]:
            best_fit_results = update_fit_results(coefs, r2, x_to_fit, fit, fit_type)

            if fit_type == "monoexp":
                fit_within_bounds = is_tau_in_limits(run_settings, best_fit_results["tau_ms"])
            elif fit_type == "biexp":
                fit_within_bounds = check_rise_and_decay_in_limits(best_fit_results, run_settings)

            if fit_within_bounds:
                return best_fit_results

    return best_fit_results


# Fitting helpers
# --------------------------------------------------------------------------------------


def update_fit_results(
    coefs: Tuple,
    r2: FalseOrFloat,
    x_to_fit: NpArray64,
    fit: FalseOrFloat,
    fit_type: str,
) -> Dict:
    """
    Convenience function for saving output of a fit to configs.

    fit_type: "monoexp" or, "biexp"
    """
    if fit_type == "monoexp":
        b0, b1, tau = coefs
        best_fit_results = {
            "fit_time": x_to_fit,
            "fit_im": fit,
            "b0": b0,
            "b1": b1,
            "tau_ms": tau * 1000,
            "r2": r2,
        }

    elif fit_type == "biexp":
        b0, b1, rise, decay = coefs
        best_fit_results = {
            "fit_time": x_to_fit,
            "fit_im": fit,
            "b0": b0,
            "b1": b1,
            "rise_ms": rise * 1000,
            "decay_ms": decay * 1000,
            "r2": r2,
        }
    return best_fit_results


def get_adjust_startpoint_based_on_r2_or_bounds(fit_type: str, run_settings: Dict) -> Int:
    """
    Convenience function to get the r2 or bounds num samples to try
    """
    config = run_settings[fit_type + "_fit"]
    if config["adjust_startpoint_r2_on"]:
        return config["adjust_startpoint_r2_value"]
    elif config["adjust_startpoint_bounds_on"]:
        return config["adjust_startpoint_bounds_value"]
    else:
        raise TypeError("r2 or bounds must be on!")


def get_initial_est_for_biexp_fit(x_to_fit: NpArray64, y_to_fit: NpArray64, run_settings: Dict) -> Tuple:
    """
    Get the b0 and b1 from the core_analysis_methods method. Then update the rise and
    decay coefficients either with the rise / decay used to detect the events,
    or the user-specified default coefficients.
    """
    initial_est = core_analysis_methods.get_biexp_event_initial_est(x_to_fit, y_to_fit, run_settings["direction"])
    initial_est = list(initial_est)

    if "template" in run_settings["analysis_type"]:
        initial_est[2] = run_settings["rise_s"]
        initial_est[3] = run_settings["decay_s"]

    elif run_settings["analysis_type"] in ["curve_fitting", "threshold"]:
        initial_est[2] = run_settings["cannonical_initial_biexp_coefficients"]["rise"] / 1000
        initial_est[3] = run_settings["cannonical_initial_biexp_coefficients"]["decay"] / 1000

    else:
        raise BaseException("analysis type not recognised")

    return tuple(initial_est)


# Decay Percent Time
# --------------------------------------------------------------------------------------


def calculate_decay_time(
    time_: NpArray64, data: NpArray64, event_info: Dict, run_settings: Dict
) -> Tuple[Optional[FalseOrDict], Optional[FalseOrDict]]:
    """
    Find the decay point that is at the user-specified percent of the event amplitude.
    If a monoexponential or biexpoential is fit to the event, use this to find the %
    point.
    """
    peak_idx = event_info["peak"]["idx"]

    if run_settings["decay_times_baseline"] == "baseline":
        base_im = event_info["baseline"]["im"]
    else:
        base_im = event_info["decay_point"]["im"]

    if run_settings["decay_or_biexp_fit_method"] == "do_not_fit":
        smooth_window_samples = core_analysis_methods.quick_get_time_in_samples(
            run_settings["ts"], run_settings["decay_period_smooth_s"]
        )

        if run_settings["decay_mode"] == "decay_one_point":
            (
                decay_timepoint,
                decay_value,
                decay_time_ms,
                smoothed_decay_time,
                smoothed_decay_im,
            ) = voltage_calc.calculate_decay_to_point_from_smoothed_decay(
                time_,
                data,
                peak_idx,
                event_info["decay_point"]["idx"],
                base_im,
                smooth_window_samples,
                event_info["amplitude"]["im"],
                run_settings["decay_amplitude_percent"],
                run_settings["interp_200khz"],
            )

            decay_perc_results = {
                "time": decay_timepoint,
                "im": decay_value,
                "decay_time_ms": decay_time_ms,
                "smoothed_decay_time": smoothed_decay_time,
                "smoothed_decay_im": smoothed_decay_im,
            }
            decay_cutoff_results = None

        elif run_settings["decay_mode"] == "decay_between_points":

            (
                min_time,
                min_data,
                max_time,
                max_data,
                decay_time,
                smoothed_decay_time,
                smoothed_decay_im,
            ) = voltage_calc.calculate_decay_cutoff_for_smoothed_decay(
                time_,
                data,
                peak_idx,
                event_info["decay_point"]["idx"],
                base_im,
                event_info["amplitude"]["im"],
                smooth_window_samples,
                run_settings["decay_cutoff_low"],
                run_settings["decay_cutoff_high"],
                run_settings["interp_200khz"],
            )

            decay_cutoff_results = {
                "min_time": min_time,
                "min_im": min_data,
                "max_time": max_time,
                "max_im": max_data,
                "decay_time_ms": decay_time * 1000,
                "smoothed_decay_time": smoothed_decay_time,
                "smoothed_decay_im": smoothed_decay_im,
            }
            decay_perc_results = None

    else:
        if "from_fit_not_data" in run_settings and run_settings["from_fit_not_data"]:
            decay_period_time = time_[peak_idx:]
            decay_period_data = data[peak_idx:]

        elif run_settings["decay_or_biexp_fit_method"] == "monoexp":
            decay_period_time = event_info["monoexp_fit"]["fit_time"]
            decay_period_data = event_info["monoexp_fit"]["fit_im"]

        elif run_settings["decay_or_biexp_fit_method"] == "biexp":
            (
                decay_period_time,
                decay_period_data,
            ) = get_biexp_function_rise_and_decay_period(event_info["biexp_fit"], run_settings["direction"], "decay")
            if len(decay_period_time) < 3:
                return False, False

        smoothed_decay_time = smoothed_decay_im = None

        if run_settings["decay_mode"] == "decay_one_point":

            try:  # TODO: in the rarest of instances no conditional is met and decay percent time is not defined
                (decay_timepoint, decay_value, decay_time_ms,) = voltage_calc.calclate_decay_to_point_from_exp_fit(
                    decay_period_time,
                    decay_period_data,
                    event_info["peak"]["time"],
                    base_im,
                    event_info["amplitude"]["im"],
                    run_settings["decay_amplitude_percent"],
                    run_settings["interp_200khz"],
                )
            except:
                return False, False

            decay_perc_results = {
                "time": decay_timepoint,
                "im": decay_value,
                "decay_time_ms": decay_time_ms,
                "smoothed_decay_time": smoothed_decay_time,
                "smoothed_decay_im": smoothed_decay_im,
            }
            decay_cutoff_results = None

        elif run_settings["decay_mode"] == "decay_between_points":

            (min_time, min_data, max_time, max_data, decay_time,) = voltage_calc.calculate_decay_cutoff_for_exp_fit(
                decay_period_time,
                decay_period_data,
                base_im,
                event_info["amplitude"]["im"],
                run_settings["decay_cutoff_low"],
                run_settings["decay_cutoff_high"],
                run_settings["interp_200khz"],
            )

            decay_cutoff_results = {
                "min_time": min_time,
                "min_im": min_data,
                "max_time": max_time,
                "max_im": max_data,
                "decay_time_ms": decay_time * 1000,
                "smoothed_decay_time": smoothed_decay_time,
                "smoothed_decay_im": smoothed_decay_im,
            }
            decay_perc_results = None

    return decay_perc_results, decay_cutoff_results


# Checking R2 and Bounds
# --------------------------------------------------------------------------------------


def check_fit_within_r2_bounds(r2: NpArray64, fit_type: str, run_settings: Dict) -> bool:
    """
    If thresholding for r2 is on and r2 is under threshold, return False
    else True.
    """
    config_key = fit_type + "_fit"

    r2_threshold_on = run_settings[config_key]["exclude_from_r2_on"]
    r2_threshold = run_settings[config_key]["exclude_from_r2_value"]

    if r2_threshold_on and r2 < r2_threshold:
        return False

    return True


def check_rise_and_decay_in_limits(biexp_dict: Dict, run_settings: Dict) -> bool:
    """ """
    opts = run_settings["biexp_fit"]
    if opts["exclude_if_params_not_in_bounds"] or opts["adjust_startpoint_bounds_on"]:
        if not (opts["rise_cutoff_min"] < biexp_dict["rise_ms"] < opts["rise_cutoff_max"]):
            return False

        if not (opts["decay_cutoff_min"] < biexp_dict["decay_ms"] < opts["decay_cutoff_max"]):
            return False

    return True


def is_tau_in_limits(run_settings: Dict, tau_ms: NpArray64) -> bool:
    """
    Check if the fit tau is within the user-specified limits
    """
    opts = run_settings["monoexp_fit"]
    if opts["exclude_if_params_not_in_bounds"] or opts["adjust_startpoint_bounds_on"]:
        return opts["tau_cutoff_min"] <= tau_ms <= opts["tau_cutoff_max"]
    return True


# --------------------------------------------------------------------------------------
# Rise Time
# --------------------------------------------------------------------------------------


def calculate_event_rise_time(time_: NpArray64, data: NpArray64, event_info: Dict, run_settings: Dict) -> FalseOrDict:
    """
    Calculate the event rise time, wrapper around
    voltage_calc.calculate_event_rise_time()
    """
    if event_info["peak"]["idx"] - event_info["baseline"]["idx"] < 2:
        return False

    (rise_min_time, rise_min_im, rise_max_time, rise_max_im, rise_time,) = voltage_calc.calculate_event_rise_time(
        time_,
        data,
        event_info["baseline"]["idx"],
        event_info["peak"]["idx"],
        event_info["baseline"]["im"],
        event_info["peak"]["im"],
        min_cutoff_perc=run_settings["rise_cutoff_low"],
        max_cutoff_perc=run_settings["rise_cutoff_high"],
        interp=run_settings["interp_200khz"],
    )

    if rise_time < 0:
        return False

    rise_time_results = {
        "min_time": rise_min_time,
        "min_im": rise_min_im,
        "max_time": rise_max_time,
        "max_im": rise_max_im,
        "rise_time_ms": rise_time * 1000,
    }

    return rise_time_results


# --------------------------------------------------------------------------------------
# Half Width
# --------------------------------------------------------------------------------------


def calculate_half_width(time_: NpArray64, data: NpArray64, event_info: Dict, run_settings: Dict) -> FalseOrDict:
    """
    Find the half-width of the event.

    If a biexponential is fit, calculate half-width on this.
    If decay monoexponetial or decay is smoothed, calculate
    decay-side midpoint from this.
    """
    bl_idx, peak_idx, bl_im, amplitude, decay_perc_info, decay_cutoff = [
        event_info["baseline"]["idx"],
        event_info["peak"]["idx"],
        event_info["baseline"]["im"],
        event_info["amplitude"]["im"],
        event_info["decay_one_point"],
        event_info["decay_between_points"],
    ]
    half_amp = bl_im + amplitude / 2
    bl_to_peak_time = time_[bl_idx : peak_idx + 1]
    bl_to_peak_im = data[bl_idx : peak_idx + 1]

    if "from_fit_not_data" in run_settings and run_settings["from_fit_not_data"]:  # this must come first
        decay_exp_fit_time = time_[peak_idx:]
        decay_exp_fit_im = data[peak_idx:]

    elif run_settings["decay_or_biexp_fit_method"] == "do_not_fit":
        # TODO: rename decay_perc_info and decay_cutoff to new
        # decay time one point and between points (respectively).
        if decay_perc_info is not None:
            decay_exp_fit_time = decay_perc_info["smoothed_decay_time"]
            decay_exp_fit_im = decay_perc_info["smoothed_decay_im"]
        else:
            decay_exp_fit_time = decay_cutoff["smoothed_decay_time"]
            decay_exp_fit_im = decay_cutoff["smoothed_decay_im"]

    elif run_settings["decay_or_biexp_fit_method"] == "monoexp":
        decay_exp_fit_time = event_info["monoexp_fit"]["fit_time"]
        decay_exp_fit_im = event_info["monoexp_fit"]["fit_im"]

    elif run_settings["decay_or_biexp_fit_method"] == "biexp":
        bl_to_peak_time, bl_to_peak_im = get_biexp_function_rise_and_decay_period(
            event_info["biexp_fit"], run_settings["direction"], "rise"
        )
        decay_exp_fit_time, decay_exp_fit_im = get_biexp_function_rise_and_decay_period(
            event_info["biexp_fit"], run_settings["direction"], "decay"
        )

    if len(bl_to_peak_time) < 2 or len(decay_exp_fit_time) < 2:
        return False

    (
        rise_midtime,
        rise_mid_im,
        decay_midtime,
        decay_mid_im,
        half_width,
        rise_mid_idx,
        decay_mid_idx,
    ) = core_analysis_methods.calculate_fwhm(
        bl_to_peak_time,
        bl_to_peak_im,
        decay_exp_fit_time,
        decay_exp_fit_im,
        half_amp,
        interp=run_settings["interp_200khz"],
    )

    half_width_results = {
        "rise_midtime": rise_midtime,
        "rise_mid_im": rise_mid_im,
        "decay_midtime": decay_midtime,
        "decay_mid_im": decay_mid_im,
        "fwhm_ms": half_width * 1000,
        "rise_mid_idx": bl_idx + rise_mid_idx,
        "decay_mid_idx": peak_idx + decay_mid_idx,
    }

    return half_width_results


def get_biexp_function_rise_and_decay_period(
    biexp_fit_dict: Dict, direction: Direction, rise_or_decay: str
) -> Tuple[NpArray64, NpArray64]:
    """
    Convenience function to get the rise (start:peak) or decay
    (peak:end) portion of a biexponential fit.
    """
    peak_func = np.argmin if direction == -1 else np.argmax
    biexp_peak_idx = peak_func(biexp_fit_dict["fit_im"])

    if rise_or_decay == "rise":
        period_time = biexp_fit_dict["fit_time"][0 : biexp_peak_idx + 1]
        period_data = biexp_fit_dict["fit_im"][0 : biexp_peak_idx + 1]

    elif rise_or_decay == "decay":
        period_time = biexp_fit_dict["fit_time"][biexp_peak_idx:]
        period_data = biexp_fit_dict["fit_im"][biexp_peak_idx:]

    return period_time, period_data


# Area Under the Curve
# --------------------------------------------------------------------------------------


def calculate_max_slope(time_: NpArray64, data: NpArray64, event_info: Dict, run_settings: Dict) -> None:
    if run_settings["max_slope"]["on"]:
        (
            event_info["max_rise"]["max_slope_ms"],
            event_info["max_rise"]["fit_time"],
            event_info["max_rise"]["fit_data"],
        ) = calculate_max_slope_rise_or_decay(time_, data, "rise", event_info, run_settings)
        (
            event_info["max_decay"]["max_slope_ms"],
            event_info["max_decay"]["fit_time"],
            event_info["max_decay"]["fit_data"],
        ) = calculate_max_slope_rise_or_decay(time_, data, "decay", event_info, run_settings)

    else:
        event_info["max_rise"]["max_slope_ms"] = event_info["max_decay"]["max_slope_ms"] = "off"


def calculate_max_slope_rise_or_decay(
    time_: NpArray64,
    data: NpArray64,
    rise_or_decay: Literal["rise", "decay"],
    event_info: Dict,
    run_settings: Dict,
) -> Tuple[NpArray64, NpArray64, NpArray64]:
    """
    Calculate the max rise and decay of the event.

    For the decay, the period searched can be to the event endpoint
    calculated by whatever method the user has chosen, or always be
    the first crossover method. The first crossover method is more
    accurate and faster than the entire search region method. If
    the user has already chosen the crossover method, there is no
    need to calculate. However, if the user is using entire search
    region to calculate the event endpoint, it will need to be
    calculated here.
    """
    if rise_or_decay == "rise":
        start_idx, stop_idx = [event_info["baseline"]["idx"], event_info["peak"]["idx"]]
        argmax_func = np.argmin if run_settings["direction"] == -1 else np.argmax
        window_samples = run_settings["max_slope"]["rise_num_samples"]

    elif rise_or_decay == "decay":
        start_idx = event_info["peak"]["idx"]

        if need_to_recalculate_event_endpoint(run_settings):
            window = core_analysis_methods.quick_get_time_in_samples(
                run_settings["ts"], run_settings["decay_search_period_s"]
            )

            stop_idx, __, __ = voltage_calc.decay_point_first_crossover_method(
                time_,
                data,
                event_info["peak"]["idx"],
                window,
                run_settings,
                event_info["baseline"]["im"],
            )
        else:
            stop_idx = event_info["decay_point"]["idx"]

        argmax_func = np.argmax if run_settings["direction"] == -1 else np.argmin
        window_samples = run_settings["max_slope"]["decay_num_samples"]

    (max_slope_ms, fit_time, fit_data,) = core_analysis_methods.calculate_max_slope_rise_or_decay(
        time_,
        data,
        start_idx,
        stop_idx,
        window_samples,
        ts=run_settings["ts"],
        smooth_settings=run_settings["max_slope"]["smooth"],
        argmax_func=argmax_func,
    )
    return max_slope_ms, fit_time, fit_data


def calculate_area_under_curve_and_threshold(
    data: NpArray64, time_: NpArray64, event_info: Dict, run_settings: Dict
) -> bool:
    """
    Calculate the area under the event (from baseline to end of decay period).

    Set the first datapoint to the bl im value in case the baseline is smoothed.
    This will have a very small effect, but image a large positive noise spike
    at 1 pA at the baseline in a negative-going event with peak -5 pA. The
    baseline smoothed is say 0 pA. If the first idx is left free, it will
    decrease the area slightly due to this noise. However we are interested
    in the AUC from the baseline onwards, so set this first datapoint to the
    baseline.

    For curve fitting and average event analysis, the data to plot for AUC
    curves is saved to event_info for reconstruction in the plots. This is
    not the case for standard analysis because it is easy to reconstruct
    this from mainwindow data, and will not waste memory as curve
    fitting / average event number is low (e.g. 1 event).
    """
    bl_idx, decay_idx = [
        event_info["baseline"]["idx"],
        event_info["decay_point"]["idx"],
    ]

    y = copy.deepcopy(data[bl_idx : decay_idx + 1])
    y[0] = event_info["baseline"]["im"]
    norm_y = y - y[0]

    (
        area_under_curve,
        area_under_curve_time_ms,
    ) = core_analysis_methods.area_under_curve_ms(norm_y, run_settings["ts"])

    # dont threshold c.f. or average event
    if run_settings["area_under_curve"]["on"] and "curve_fitting_or_average_event_flag" not in run_settings:
        if np.abs(area_under_curve) < run_settings["area_under_curve"]["value_pa_ms"]:
            return False

    save_auc_results_to_event_info(
        event_info,
        area_under_curve,
        area_under_curve_time_ms,
        run_settings,
        y,
        time_,
        bl_idx,
        decay_idx,
    )

    return True


def save_auc_results_to_event_info(
    event_info: Dict,
    area_under_curve: NpArray64,
    area_under_curve_time_ms: NpArray64,
    run_settings: Dict,
    y: NpArray64,
    time_: NpArray64,
    bl_idx: Int,
    decay_idx: int,
) -> None:
    """
    see calculate_area_under_curve_and_threshold()
    """
    event_info["area_under_curve"]["im"] = area_under_curve
    event_info["event_period"]["time_ms"] = area_under_curve_time_ms

    if "curve_fitting_or_average_event_flag" in run_settings and run_settings["curve_fitting_or_average_event_flag"]:
        event_info["area_under_curve"]["event_data"] = y
        event_info["area_under_curve"]["time_array"] = time_[bl_idx : decay_idx + 1]
        event_info["area_under_curve"]["baseline_period"] = np.repeat(event_info["baseline"]["im"], len(y))


def need_to_recalculate_event_endpoint(run_settings: Dict) -> bool:
    """
    Max slope can have own endpoint calculation as first_baseline_cross works
    best for this.

    If use first baseline cross already used, return True (we don't need to
    recalculate it). Otherwise, if it is off, we will need to recalculate it.
    Also, if legacy decay method is selected, it overrides the
    "endpoint_search_method", so we will also need to recalculate it.
    """
    if run_settings["max_slope"]["use_baseline_crossing_endpoint"]:
        if (
            run_settings["endpoint_search_method"] == "first_baseline_cross"
            and not run_settings["legacy_options"]["decay_detection_method"]
        ):
            return False
        else:
            return True
    else:
        return False


# --------------------------------------------------------------------------------------
# Overlay and Average Events
# --------------------------------------------------------------------------------------


def index_out_individual_events(
    data_array: NpArray64,
    window_length_samples: List[Int],
    alignment_method: Literal["peak", "rise_half_width", "baseline"],
    event_info: List[Dict],
    average: bool = True,
) -> NpArray64:
    """
    Index into the Im array to create an average event
    based on event_info events.

    Due to significant overlap in function, this method can either return
    a num_events by timepoints array, showing all events un-averaged.
    Alternatively, it can average the events
    """
    total_num_events = core_analysis_methods.total_num_events(event_info)

    window_samples_left, window_samples_right = window_length_samples

    traces = utils.np_empty_nan((total_num_events, window_samples_left + window_samples_right))

    ev_idx = -1
    for rec in range(len(event_info)):
        if any(event_info[rec]):
            for ev in event_info[rec].values():

                ev_idx += 1

                rec_data_array = data_array[rec]

                start_idx, end_idx = get_window_indexes_false_if_not_full_window(
                    ev,
                    alignment_method,
                    rec_data_array,
                    window_samples_left,
                    window_samples_right,
                )

                if start_idx is False or end_idx is False:
                    continue

                event_to_add = rec_data_array[start_idx:end_idx]

                traces[ev_idx, :] = event_to_add

    if average:
        traces = average_events_overlay(traces)

    return traces


def average_events_overlay(events_overlay: NpArray64) -> NpArray64:
    """
    Average `events_overlay`, a num_events x num_samples
    array holding multiple events to be plot on EventsOverlay.

    Average column-wise ignoring any NaNs.
    """
    return np.nanmean(events_overlay, axis=0)


def shift_and_scale_array(
    events_overlay: NpArray64,
    shift: bool,
    scale: bool,
    shift_method: Optional[Literal["demean", "aligned_param", "start_idx"]],
    scale_method: Optional[Literal["amplitude_to_1", "std_to_1"]],
    param_idx: Optional[Int] = None,
) -> NpArray64:
    """
    Given an array of events (num_events x num_samples), shift and scale
    each event.

    INPUTS
    ------

    events_overlay :(num_events x num_samples) array of indexed events.
    shift : if  `True`, events will be shifted according to `shift_method`.
    scale : if `True`, events will be scaled according to `scale_method`.
    shift_method : The operations are applied per event:
        "demean" - subtract the mean
        "aligned_param" - subtract the value of the parameter aligned to, passed as `param_idx`
        "start_idx" - subtract the first value of the event
    scale_method : The operations are applied per event:
        "amplitude_to_1" - scale the event so it's amplitude is 1
        "std_to_1" - scale the event of the std is 1
    param_idx :
        If `shift_method` is set to "aligned_param", this index is the
        index of the aligned param, that will be shifted to 0.
    """
    validate_shift_and_scale_inputs(events_overlay, shift, scale, shift_method, param_idx, scale_method)

    if scale:
        if scale_method == "amplitude_to_1":
            events_overlay = voltage_calc.normalise_amplitude(events_overlay)

        elif scale_method == "std_to_1":
            events_overlay = events_overlay / np.std(events_overlay, axis=-1, keepdims=True)

    if shift:
        if shift_method == "demean":
            events_overlay = events_overlay - np.mean(events_overlay, axis=-1, keepdims=True)

        elif shift_method == "aligned_param":
            if events_overlay.ndim == 1:
                events_overlay = events_overlay - events_overlay[param_idx]
            else:
                events_overlay = events_overlay - events_overlay[:, param_idx][:, np.newaxis]

        elif shift_method == "start_idx":
            if events_overlay.ndim == 1:
                events_overlay = events_overlay - events_overlay[0]
            else:
                events_overlay = events_overlay - events_overlay[:, 0][:, np.newaxis]

    return events_overlay


def validate_shift_and_scale_inputs(
    events_overlay: NpArray64,
    shift: bool,
    scale: bool,
    shift_method: Optional[Literal["demean", "aligned_param", "start_idx"]],
    param_idx: Optional[Int],
    scale_method: Optional[Literal["amplitude_to_1", "std_to_1"]],
) -> None:
    """
    Assert the inputs to `shift_and_scale_array()` are valid.
    """
    assert events_overlay.ndim in [1, 2], "`events_overlay` cannot have more than 2 dimensions."
    if shift_method == "aligned_param":
        assert param_idx is not None, "`param_idx` must be passed if `aligned_param` is used."

    if shift:
        assert shift_method in ["demean", "aligned_param", "start_idx"]
    if scale:
        assert scale_method in ["amplitude_to_1", "std_to_1"]


def get_window_indexes_false_if_not_full_window(
    ev: Dict,
    alignment_method: Literal["peak", "rise_half_width", "baseline"],
    rec_data_array: NpArray64,
    num_left_edge_samples: Int,
    window_length_samples: Int,
) -> Tuple[Union[Literal[False], Int], ...]:
    """
    Index out an event with the window centre on the
    feature to align to (peak, rise_half_width or baseline).

    Note the window is not symmetric, but the left edge is
    fixed set by `get_num_left_edge_samples` while
    the right-edge is adjustable.
    """
    if alignment_method == "peak":
        parameter_to_align_idx = ev["peak"]["idx"]

    elif alignment_method == "rise_half_width":
        parameter_to_align_idx = ev["half_width"]["rise_mid_idx"]

    elif alignment_method == "baseline":
        parameter_to_align_idx = ev["baseline"]["idx"]

    start_idx, end_idx = index_event_and_check_window_size(
        rec_data_array,
        parameter_to_align_idx,
        num_left_edge_samples,
        window_length_samples,
    )
    return start_idx, end_idx


def index_event_and_check_window_size(
    rec_data_array: NpArray64,
    parameter_to_align_idx: Int,
    num_left_edge_samples: Int,
    window_length_samples: Int,
) -> Tuple[Union[Literal[False], int], ...]:
    """
    Exclude any event that is not a full window length.
    """
    start_idx = parameter_to_align_idx - num_left_edge_samples
    end_idx = parameter_to_align_idx + window_length_samples
    if start_idx < 0 or end_idx >= len(rec_data_array):
        return False, False

    return start_idx, end_idx


def get_num_left_edge_samples(event_info: List[Dict]) -> Int:
    """
    Get the left-edge on the window size for (visualisation purposes only),
    based on the maximum distance between baseline and peak of any event detected,
    and a constant determined subjectively based on tested
    datasets.
    """
    scale_left_edge_const = 1.33
    num_left_edge_samples = int(get_max_bl_to_peak_samples(event_info) * scale_left_edge_const)
    return num_left_edge_samples


def get_max_bl_to_peak_samples(event_info: List[Dict]) -> Union[Int, NpArray64]:
    """
    Find the maximum number of samples between an events baseline and it's peak,
    across every detected event. This is used to set the left-side
    window on average and overlay plots.
    """
    max_bl_to_peak = 0
    for rec in range(len(event_info)):
        if any(event_info[rec]):
            for ev in event_info[rec].values():
                bl_to_peak = ev["peak"]["idx"] - ev["baseline"]["idx"]
                if bl_to_peak > max_bl_to_peak:
                    max_bl_to_peak = bl_to_peak
    return max_bl_to_peak


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def make_event_info_dict() -> Dict:
    """
    Holds all event kinetics info, it is 2 layers deeps.

    CombinedAnalysisModel.
          This is particularly important if adding a third dict. layer.

    Notes
    -----
    If editing this, ensure convert_event_info_serializable()
    is updated on
    """
    event_info = {
        "peak": {"time": None, "im": None, "idx": None},
        "baseline": {"time": None, "im": None, "idx": None},
        "amplitude": {"im": None},
        "rise": {
            "min_time": None,
            "min_im": None,
            "max_time": None,
            "max_im": None,
            "rise_time_ms": None,
        },
        "decay_point": {"time": None, "im": None, "idx": None},
        "monoexp_fit": {
            "fit_im": None,
            "fit_time": None,
            "b0": None,
            "b1": None,
            "tau_ms": None,
            "r2": None,
        },
        "biexp_fit": {
            "b0": None,
            "b1": None,
            "rise_ms": None,
            "decay_ms": None,
            "fit_time": None,
            "fit_im": None,
            "r2": None,
        },
        "decay_one_point": {
            "time": None,
            "im": None,
            "decay_time_ms": None,
            "smoothed_decay_time": None,
            "smoothed_decay_im": None,
        },
        "decay_between_points": {
            "min_time": None,
            "min_im": None,
            "max_time": None,
            "max_im": None,
            "decay_time_ms": None,
        },
        "half_width": {
            "rise_midtime": None,
            "rise_mid_im": None,
            "decay_midtime": None,
            "decay_mid_im": None,
            "fwhm_ms": None,
            "rise_mid_idx": None,
            "decay_mid_idx": None,
        },
        "record_num": {"rec_idx": None},
        # note baseline_period and time_array are only used for curve
        # fitting and event average kinetics plotting
        "area_under_curve": {
            "im": None,
            "event_data": None,
            "baseline_period": None,
            "time_array": None,
        },
        "event_period": {"time_ms": None},
        "max_rise": {
            "max_slope_ms": [np.nan],
            "fit_time": [np.nan],
            "fit_data": [np.nan],
        },
        "max_decay": {
            "max_slope_ms": [np.nan],
            "fit_time": [np.nan],
            "fit_data": [np.nan],
        },
        "info": {"ev_num": np.nan, "group_num": np.nan},
    }

    return event_info
