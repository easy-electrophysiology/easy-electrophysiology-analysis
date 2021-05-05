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
import numpy as np
from ephys_data_methods import voltage_calc, core_analysis_methods, current_calc
from utils import utils
import scipy.signal

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Fit Sliding Window
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def fit_sliding_window(im_array, minimum_chunk_factor, run_settings, progress_bar_callback):
    """
    Coordinate event fitting with sliding window method. See voltage_calc.sliding_window() for details.

    Conduct sliding window analysis. If downsampled, interpolate the coefficients back up
    to the data sample size.
    """
    window_width = run_settings["window_len_s"]

    corr_coefs, betas = voltage_calc.sliding_window(im_array,
                                                    window_width,
                                                    run_settings["rise_s"],
                                                    run_settings["decay_s"],
                                                    run_settings["ts"],
                                                    run_settings["downsample_options"],
                                                    minimum_chunk_factor,
                                                    progress_bar_callback)

    if run_settings["downsample_options"]["on"]:
        # Interpolate the betas"s and correlation values back up
        # to N samples if the window was downsampled.
        time_ = np.arange(1, len(corr_coefs) + 1)
        corr_coefs = core_analysis_methods.interpolate_data(corr_coefs,
                                                            time_,
                                                            "cubic",
                                                            run_settings["downsample_options"]["downsample_factor"],
                                                            axis=0)
        betas = core_analysis_methods.interpolate_data(betas,
                                                       time_,
                                                       "cubic",
                                                       run_settings["downsample_options"]["downsample_factor"],
                                                       axis=1)

        # After interpolation there is infrequently a smaller difference in sample number between coefs and raw data (~10 samples)
        # due to the downsampling method; fix this here.
        sample_diff = (len(corr_coefs) - (len(im_array[0])))

        if sample_diff > 0:
            corr_coefs = corr_coefs[0:-sample_diff]
            betas = betas[:, 0:-sample_diff]

        elif sample_diff < 0:
            sample_diff = np.abs(sample_diff)

            fill_in_corr_coefs = np.tile(corr_coefs[0], sample_diff)
            corr_coefs = np.hstack([fill_in_corr_coefs, corr_coefs])

            fill_in_betas = np.tile(betas[0:2, 0].reshape((2, 1)), sample_diff)
            betas = np.hstack([fill_in_betas, betas])

    return betas, corr_coefs

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Events Threshold
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def events_thresholding_get_putative_peaks(im_array,
                                           min_distance_between_maxima_samples,
                                           direction):
    """
    Find peaks using the thresholding method.

    Make all events positive before passing to scipy find_peaks.
    """
    putative_peaks_idx = scipy.signal.find_peaks(im_array[0] * direction,
                                                 distance=min_distance_between_maxima_samples)[0]

    if not np.any(putative_peaks_idx):
        return False

    return putative_peaks_idx

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Peak Processing
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_event_peaks(corr_coefs,
                          betas,
                          time_array,
                          im_array,
                          run_settings):
    """
    Threshold events from their peak and correlation values and save them to self.event_info dict.

    After sliding window fit, we have a correlation value at each timepoint for the correlation of the biexp fit
    to the data. Next we want to calculate the contiguous events and their peak.

    First, only beta values in the direction of the expected events are kept.
    Next, only Im points that after over the correlation threshold are kept.

    We now have a situation where consecutive events are retained but sometimes padded with nans depending
    on the odd correlation value that may not have been over threshold e.g.
    [nan nan -1 -20 -30 -40 -30 -20 nan -1 nan nan]. We need to smooth over these gaps and then
    find the peak of all contiguous events. This is done in current_calc.index_out_continuos_above_threshold_samples().
    A smoothing window of half the sliding window works well across event types.

    Now we are left with chunked contiguous events stored in the above_threshold_ev. We finally cycle through these,
    Find their peak and other information and check they are above our linear or dynamic Im threshold. If so save to
    self.event_info.
    """
    if run_settings["direction"] == 1:
        idx_b1_pos = betas[1] < 0
    elif run_settings["direction"] == -1:
        idx_b1_pos = betas[1] > 0
    corr_coefs[idx_b1_pos] = 0

    thr = run_settings["corr_cutoff"]
    peaks = utils.np_empty_nan(len(corr_coefs))
    peaks[np.where(corr_coefs > thr)] = im_array[0][np.where(corr_coefs > thr)]

    smoothing_window = int(run_settings["window_len_samples"] * 0.5)
    cum_event_index = current_calc.index_out_continuos_above_threshold_samples(peaks,
                                                                               smooth=smoothing_window)

    event_dir = "positive" if run_settings["direction"] == 1 else "negative"
    peaks_idx = current_calc.get_peaks_idx_from_cum_idx(cum_event_index,
                                                        im_array[0],
                                                        event_dir)

    event_info = make_peak_event_info_from_peaks_idx(time_array[0],
                                                     im_array[0],
                                                     peaks_idx,
                                                     run_settings)
    return event_info

def make_peak_event_info_from_peaks_idx(time_,
                                        data,
                                        peaks_idx,
                                        run_settings):
    """
    From an array of peak indicies perform checks for threshold_lower, threshold_upper and omit times.

    See check_putatitve_event() for details on checks.
    The extended event_info that is used for full analysis can be found at cfgs.make_event_info_dict()

    Ued both for events analysis and curve_fitting analysis fit biexpoential_event
    For curve fitting and average event, omitting times / baseline settings for whole record are not required.
    """
    event_info = {}
    for peak_idx in peaks_idx:
        peak_time = time_[peak_idx]
        peak_im = data[peak_idx]

        # Smooth peak around 1/4 of the window size. # ARBITARY constant - add to configs?

        if run_settings["average_peak_points"]["on"]:
            peak_idx, peak_time, peak_im = smooth_peak(time_,
                                                       data,
                                                       peak_idx,
                                                       run_settings)
        if run_settings["name"] == "event_kinetics":
            sucess = check_putative_event(peak_idx,
                                          peak_time,
                                          peak_im,
                                          run_settings)
            if not sucess:
                continue

        event_info[str(peak_time)] = {"peak": {"time": peak_time, "im": peak_im, "idx": peak_idx}}

    return event_info

def smooth_peak(time_,
                data,
                peak_idx,
                run_settings):
    """
    Smooth the peak region and then re-find peak. See voltage_calc.find_event_peak_after_smoothing()
    """
    window = np.floor(core_analysis_methods.quick_get_time_in_samples(run_settings["ts"],
                                                                      run_settings["decay_search_period_s"]) / 4).astype(int)
    window = 1 if window == 0 else window
    samples_to_smooth = core_analysis_methods.quick_get_time_in_samples(run_settings["ts"],
                                                                        run_settings["average_peak_points"]["value_s"])

    peak_idx, peak_time, peak_im = voltage_calc.find_event_peak_after_smoothing(time_,
                                                                                data,
                                                                                peak_idx,
                                                                                window,
                                                                                samples_to_smooth,
                                                                                run_settings["direction"])
    return peak_idx, peak_time, peak_im


def check_putative_event(peak_idx,
                         peak_time,
                         peak_im,
                         run_settings):
    """
    Check if an event peak is within user-specified thresholds (e.g. threshold lower, omit times). Return False if
    event should be excluded. See voltage_calc methods for details.

    Within thresholds: threshold_lower - check the event is above the lower threshold (this threshold might be linear, curve or drawn).
                       threshold higher - check the event is under the higher threshold (this is input via spinbox)
    """
    if np.any(run_settings["omit_start_stop_times"]):
        for start, stop in run_settings["omit_start_stop_times"]:
            if start < peak_time < stop:
                return False

    within_threshold = voltage_calc.check_peak_against_threshold_lower(peak_im,
                                                                       peak_idx,
                                                                       run_settings["direction"],
                                                                       run_settings["threshold_type"],
                                                                       run_settings["threshold_lower"])
    if not within_threshold:
        return False

    if run_settings["threshold_upper_limit_on"]:
        within_threshold = voltage_calc.check_peak_height_threshold(peak_im,
                                                                    run_settings["threshold_upper_limit_value"],
                                                                    run_settings["direction"])
    if not within_threshold:
        return False

    return True

def calculate_rms(im_array, n_times, baseline):

    num_samples = len(im_array)
    mse = np.sum((baseline - im_array)**2) / num_samples
    rms = np.sqrt(mse)
    n_times_rms = rms * n_times
    return rms, n_times_rms

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Event Kinetics
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_event_kinetics(time_,
                             data,
                             peak_idx,
                             peak_time,
                             peak_im,
                             run_settings):
    """
    Calculate all event kinetics from the peak information.

    Quick test with ~1500 events and optimisation on. Clearly, the majority of time is spent fitting decay.
    As such, the option was to skip decay fitting has been added.

    {"peak": 0.004000186920166016, "baseline": 0.6544697284698486, "amplitude": 0.007997751235961914, "decay": 43.11507487297058, "rise_time": 0.8327746391296387, "half_width": 1.6689302921295166}
    """
    event_info = make_event_info_dict()

    # Peak
    event_info["peak"] = {"idx": peak_idx,
                          "time": peak_time,
                          "im": peak_im}

    # Baseline
    event_info["baseline"] = calculate_event_baseline(time_,
                                                      data,
                                                      peak_idx,
                                                      run_settings)
    if not event_info["baseline"]:
        return False

    # Amplitude
    event_info["amplitude"] = calculate_and_threshold_amplitude(run_settings,
                                                                event_info["peak"]["im"],
                                                                event_info["baseline"]["im"])

    if not event_info["amplitude"]:
        return False

    # Decay
    event_info["decay"], event_info["decay_perc"] = calcalte_decay_point_fit_and_percent(time_,
                                                                                         data,
                                                                                         event_info,
                                                                                         run_settings)
    if not event_info["decay"]:
        return False

    # Rise Time
    event_info["rise"] = calculate_event_rise_time(time_,
                                                   data,
                                                   event_info,
                                                   run_settings)

    # Half-width
    event_info["half_width"] = calculate_half_width(time_,
                                                    data,
                                                    event_info,
                                                    run_settings)

    return event_info

# Baseline
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_event_baseline(time_,
                             data,
                             peak_idx,
                             run_settings):
    """
    Calculate an event baseline point from data and current settings.

    There are two main settings - detect baseline per-event or use a pre-defined threshold

    Per-event: find_foot method: events may be detected by the "find_foot" method which find the baseline by calculating a the
                                 foot from the peak algorithmically. This can then be smoothed by the "average_baseline_points" option.

    pre-defined threshold:   This can be either a straight line (linear, basically a single baseline value), a curve or drawn.
        Whatever the threshold method, the first sample that crosses the threshold in the opposite direction of the peak is used as baseline.

    Smoothing: the baseline_im (bl_im) is adjusted by looking backwards from the identified bl idx and smoothing

    See voltage_calc methods for details.
    """
    window = core_analysis_methods.quick_get_time_in_samples(run_settings["ts"],
                                                             run_settings["baseline_search_period_s"])
    window = 1 if window == 0 else window

    if "from_fit_not_data" in run_settings and run_settings["from_fit_not_data"]:
        bl_idx = 0
        bl_time = time_[bl_idx]
        bl_im = data[bl_idx]

    elif run_settings["baseline_type"] == "per_event":
        bl_idx, bl_time, bl_im = voltage_calc.calculate_event_baseline(time_,
                                                                       data,
                                                                       peak_idx,
                                                                       run_settings["direction"],
                                                                       window)

    elif run_settings["baseline_type"] in ["manual", "curved", "drawn"]:
        bl_idx, bl_time, bl_im = calculate_event_baseline_from_data_baseline(time_,
                                                                             data,
                                                                             peak_idx,
                                                                             window,
                                                                             run_settings)
    # Check, handle smoothing, return data
    if bl_idx is False or peak_idx - bl_idx < 1:
        return False
    if bl_idx < 0:  # edge case for events on edge of recording time
        bl_idx = 0

    if run_settings["average_baseline_points"]["on"]:
        bl_im = average_baseline_points(data,
                                        bl_idx,
                                        run_settings)

    bl_results = {"idx": bl_idx, "time": bl_time, "im": bl_im}

    return bl_results

def calculate_event_baseline_from_data_baseline(time_,
                                                data,
                                                peak_idx,
                                                window,
                                                run_settings):
    """
    Calculate event baseline as the first time the data crosses a data baseline (either linear or dynamic).
    If it does not cross within the region (default half window samples) then use closest within region.

    This is in contrast to voltage_calc.calculate_event_baseline() which calculates bl per-event.

    See voltage_calc methods for details.
    """
    if run_settings["baseline_type"] == "manual":
        thr_im = run_settings["baseline"]

    elif run_settings["baseline_type"] in ["curved", "drawn"]:
        threshold = run_settings["baseline"]
        if window > peak_idx:  # make sure index not outside availble data
            window = peak_idx
        thr_im = threshold[peak_idx - window: peak_idx + 1]

    bl_idx, bl_time, bl_im = voltage_calc.calculate_event_baseline_from_thr(time_,
                                                                            data,
                                                                            thr_im,
                                                                            peak_idx,
                                                                            window,
                                                                            run_settings["direction"])
    return bl_idx, bl_time, bl_im

def average_baseline_points(data,
                            bl_idx,
                            run_settings):
    """
    Look back from the baseline idx and average as many samples as
    specified by the user. Set the new bl_im as this (note time will not change).
    """
    samples_to_average = core_analysis_methods.quick_get_time_in_samples(run_settings["ts"],
                                                                         run_settings["average_baseline_points"]["value_s"])
    bl_im = voltage_calc.average_baseline_period(data, bl_idx, samples_to_average)
    return bl_im


# Amplitude
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_and_threshold_amplitude(run_settings, peak_im, bl_im):
    """
    Threshold event - reject if amplitude too small.
    """
    amplitude_results = {"im": peak_im - bl_im}

    if np.abs(amplitude_results["im"]) < run_settings["amplitude_threshold"]:
        return False

    return amplitude_results

# Decay
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calcalte_decay_point_fit_and_percent(time_,
                                         data,
                                         event_info,
                                         run_settings):
    """
    Find a decay point, fit an exponential and find the % of peak amplitude on the decay specified by user.
    If fit exp is not chosen, the bl period is smoothed (user can set in misc options.

    See voltage_calc methods for details.
    """
    window = core_analysis_methods.quick_get_time_in_samples(run_settings["ts"],
                                                             run_settings["decay_search_period_s"])

    if window < 3:
        return False, False  # dont let search period be too small or causes errors downstream (e.g. individual event presentation)

    bl_im = event_info["baseline"]["im"]
    peak_idx = event_info["peak"]["idx"]

    if run_settings["decay_period_type"] == "auto_search_data":
        decay_idx, decay_time = voltage_calc.calculate_event_decay_point(time_,
                                                                         data,
                                                                         peak_idx,
                                                                         bl_im,
                                                                         run_settings["direction"],
                                                                         window)

    elif run_settings["decay_period_type"] == "use_end_of_region":
        decay_idx = peak_idx + window
        decay_time = time_[decay_idx]

    if decay_idx - peak_idx < 3:
        return False, False

    # Fit an exponential between peak and this point. Return False (drop event) if iterations reached without fit succeeding.
    # overwrite the im of the end point to the final point of the curve, it is the time that is important

    if run_settings["dont_fit_exp_per_event"]:
        decay_exp_fit_time = decay_exp_fit_im = [np.nan, np.nan]
        exp_fit_tau_ms = decay_im = np.nan
    else:
        decay_exp_fit_time, decay_exp_fit_im, exp_fit_tau_ms = fit_monoexp_function_to_decay(time_, data, peak_idx, decay_idx, run_settings)

        if not np.any(decay_exp_fit_time):
            return False, False  # critical check for VoltageClampModel.was_event_decay_fit()

        decay_im = decay_exp_fit_im[-1]

    decay_results = {"time": decay_time, "im": decay_im, "idx": decay_idx,
                     "exp_fit_time": decay_exp_fit_time, "exp_fit_im": decay_exp_fit_im, "exp_fit_tau_ms": exp_fit_tau_ms}

    # Calculate Decay Percentage
    decay_perc_results = calculate_decay_percent(time_,
                                                 data,
                                                 decay_exp_fit_time,
                                                 decay_exp_fit_im,
                                                 bl_im,
                                                 decay_idx,
                                                 peak_idx,
                                                 run_settings,
                                                 event_info)

    return decay_results, decay_perc_results

def calculate_decay_percent(time_,
                            data,
                            decay_exp_fit_time,
                            decay_exp_fit_im,
                            bl_im,
                            decay_idx,
                            peak_idx,
                            run_settings,
                            event_info):
    """
    Find the decay point that is at the user-specifed percent of the event amplitude.
    If a monoexponential is fit to the decay, use this to find the % point.

    See voltage_calc methods for details.
    """
    peak_time = event_info["peak"]["time"]
    amplitude = event_info["amplitude"]["im"]

    if run_settings["dont_fit_exp_per_event"]:
        smooth_window_samples = core_analysis_methods.quick_get_time_in_samples(run_settings["ts"],
                                                                                run_settings["decay_period_smooth_s"])
        decay_percent_time, decay_percent_im, decay_time_ms, \
            smoothed_decay_time, smoothed_decay_im = voltage_calc.calclate_decay_percentage_peak_from_smoothed_decay(time_,
                                                                                                                     data,
                                                                                                                     peak_idx,
                                                                                                                     decay_idx,
                                                                                                                     bl_im,
                                                                                                                     smooth_window_samples,
                                                                                                                     amplitude,
                                                                                                                     run_settings["decay_amplitude_percent"],
                                                                                                                     run_settings["interp_200khz"])
    else:
        if "from_fit_not_data" in run_settings and run_settings["from_fit_not_data"]:
            decay_exp_fit_time = time_[peak_idx:]
            decay_exp_fit_im = data[peak_idx:]

        decay_percent_time, decay_percent_im, decay_time_ms = voltage_calc.calclate_decay_percentage_peak_from_exp_fit(decay_exp_fit_time,
                                                                                                                       decay_exp_fit_im,
                                                                                                                       peak_time,
                                                                                                                       bl_im,
                                                                                                                       amplitude,
                                                                                                                       run_settings["decay_amplitude_percent"],
                                                                                                                       run_settings["interp_200khz"])
        smoothed_decay_time = smoothed_decay_im = None

    decay_perc_results = {"time": decay_percent_time, "im": decay_percent_im, "decay_time_ms": decay_time_ms,
                          "smoothed_decay_time": smoothed_decay_time, "smoothed_decay_im": smoothed_decay_im}

    return decay_perc_results

def fit_monoexp_function_to_decay(time_,
                                  data,
                                  peak_idx,
                                  decay_idx,
                                  run_settings):
    """
    Fit expoenential decay to the decay period

    Any events less than 2 samples or where coefs could not be first within 800 iterations are automatically discarded.

    Depending on the user setting, any event outside the specified tau range is either

    a) discarded outright

    b) undergoes an optimisation procedure where a few different peaks are used. Sometimes moving one sample to the left/right
       can drastically improve the fit. If this does not get the tau within range, discard.

    See voltage_calc methods for details.
    """
    decay_exp_fit_time = time_[peak_idx:decay_idx + 1]
    ev_data = data[peak_idx:decay_idx + 1]

    coefs, decay_exp_fit_im = core_analysis_methods.fit_curve("monoexp",
                                                              decay_exp_fit_time,
                                                              ev_data,
                                                              run_settings["direction"])
    if not np.any(coefs):
        return False, False, False

    # Try to fix fits / exclude events based on user-provided tau.
    exp_fit_tau_ms = coefs[2] * 1000
    if run_settings["decay_tau_options"]["adjust_peak_to_get_in_limits"] or \
            run_settings["decay_tau_options"]["exclude_event_based_on_limits"]:

        if not is_tau_in_limits(run_settings,
                                exp_fit_tau_ms):
            if run_settings["decay_tau_options"]["adjust_peak_to_get_in_limits"]:

                decay_exp_fit_time, decay_exp_fit_im, exp_fit_tau_ms = voltage_calc.adjust_peak_to_optimise_tau(time_,
                                                                                                                data,
                                                                                                                peak_idx,
                                                                                                                decay_idx,
                                                                                                                run_settings)
                if not np.any(decay_exp_fit_time):
                    return False, False, False

            elif run_settings["decay_tau_options"]["exclude_event_based_on_limits"]:
                return False, False, False

    return decay_exp_fit_time, decay_exp_fit_im, exp_fit_tau_ms

# Rise Time
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_event_rise_time(time_,
                              data,
                              event_info,
                              run_settings):
    """
    Calculate the event rise time, wrapper around voltage_calc.calculate_event_rise_time()
    """
    rise_min_time, rise_min_im, rise_max_time, rise_max_im, rise_time = voltage_calc.calculate_event_rise_time(time_,
                                                                                                               data,
                                                                                                               event_info["baseline"]["idx"],
                                                                                                               event_info["peak"]["idx"],
                                                                                                               event_info["baseline"]["im"],
                                                                                                               event_info["peak"]["im"],
                                                                                                               run_settings["direction"],
                                                                                                               min_cutoff_perc=run_settings["rise_cutoff_low"],
                                                                                                               max_cutoff_perc=run_settings["rise_cutoff_high"],
                                                                                                               interp=run_settings["interp_200khz"])
    rise_time_results = {"min_time": rise_min_time, "min_im": rise_min_im, "max_time": rise_max_time,
                         "max_im": rise_max_im, "rise_time_ms": rise_time * 1000}

    return rise_time_results

# Half Width
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_half_width(time_,
                         data,
                         event_info,
                         run_settings):
    """
    Find the half-width of the event.

    For the rise, take the baseline - peak section of the data.

    For the decay, it depends on the previous settings.
    If fitting to a biexp fit, use the entire decay region to search for the half-width decay point. It is important htis comes first so that
        if the fit is to the biexponential the fit is definitely to the biexp and the monoexp settings are not taken into account.
        else, if fit biexp and dont fit exp per event were both on smoothed decay will be used rather than the fit

    Else if a monoexponential has been fit to the raw data, use a point along the fit rather than the raw data.
    Else if no monoexpoential has been fit, use the smoothed data.
    """
    bl_idx, peak_idx, bl_im, amplitude, decay_info, decay_perc_info = [event_info["baseline"]["idx"],
                                                                       event_info["peak"]["idx"],
                                                                       event_info["baseline"]["im"],
                                                                       event_info["amplitude"]["im"],
                                                                       event_info["decay"],
                                                                       event_info["decay_perc"]
                                                                       ]

    half_amp = (bl_im + amplitude / 2)
    bl_to_peak_time = time_[bl_idx: peak_idx + 1]
    bl_to_peak_im = data[bl_idx: peak_idx + 1]

    if "from_fit_not_data" in run_settings and run_settings["from_fit_not_data"]:  # this must come first
        decay_exp_fit_time = time_[peak_idx:]
        decay_exp_fit_im = data[peak_idx:]

    elif run_settings["dont_fit_exp_per_event"]:
        decay_exp_fit_time = decay_perc_info["smoothed_decay_time"]
        decay_exp_fit_im = decay_perc_info["smoothed_decay_im"]

    else:
        decay_exp_fit_time = decay_info["exp_fit_time"]
        decay_exp_fit_im = decay_info["exp_fit_im"]

    rise_midtime, rise_mid_im, decay_midtime, decay_mid_im, half_width = core_analysis_methods.calculate_fwhm(bl_to_peak_time,
                                                                                                              bl_to_peak_im,
                                                                                                              decay_exp_fit_time,
                                                                                                              decay_exp_fit_im,
                                                                                                              half_amp,
                                                                                                              interp=run_settings["interp_200khz"])
    half_width_results = {"rise_midtime": rise_midtime, "rise_mid_im": rise_mid_im,
                          "decay_midtime": decay_midtime, "decay_mid_im": decay_mid_im, "fwhm_ms": half_width * 1000}

    return half_width_results

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Average Event
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def make_average_event(im_array,
                       ts,
                       window_length_samples,
                       alignment_method,
                       event_info):
    """
    Index into the Im array to create an average event
    based on event_info events.
    """
    avg_trace = np.zeros(window_length_samples * 2)
    num_events = 0
    for ev in event_info.values():

        if alignment_method == "peak":
            parameter_to_align_idx = ev["peak"]["idx"]

        elif alignment_method == "rise_half_width":
            rise_half_width_time = ev["half_width"]["rise_midtime"]
            parameter_to_align_idx = core_analysis_methods.quick_get_time_in_samples(ts,
                                                                                     rise_half_width_time)

        elif alignment_method == "baseline":
            parameter_to_align_idx = ev["baseline"]["idx"]

        event_to_add = index_event_and_check_window_size(im_array[0],
                                                         parameter_to_align_idx,
                                                         window_length_samples)
        if event_to_add is False:
            continue

        avg_trace += event_to_add
        num_events += 1

    avg_trace /= num_events

    return avg_trace

def index_event_and_check_window_size(im_array, peak_idx, window_length_samples):
    """
    Exclude any event that is not a full window length.
    """
    start_idx = peak_idx - window_length_samples
    end_idx = peak_idx + window_length_samples
    if start_idx < 0 or end_idx >= len(im_array):
        return False

    event_to_add = im_array[start_idx: end_idx]

    return event_to_add

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def is_tau_in_limits(run_settings, tau_ms):
    """
    Check if the fit tau is within the user-specified limits
    """
    return run_settings["decay_tau_options"]["tau_cutoff_min"] <= tau_ms <= run_settings["decay_tau_options"]["tau_cutoff_max"]

def make_event_info_dict():
    """
    """
    event_info = {"peak": {"time": None, "im": None, "idx": None},
                  "baseline": {"time": None, "im": None, "idx": None},
                  "amplitude": {"im": None},
                  "rise": {"min_time": None, "min_im": None, "max_time": None,
                           "max_im": None, "rise_time_ms": None},
                  "decay": {"time": None, "im": None, "idx": None,
                            "exp_fit_im": None, "exp_fit_time": None, "exp_fit_tau_ms": None},
                  "decay_perc": {"time": None, "im": None, "idx": None, "decay_time_ms": None},
                  "half_width": {"rise_midtime": None, "rise_mid_im": None, "decay_midtime": None,
                                 "decay_mid_im": None, "fwhm_ms": None}
                  }
    return event_info







