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
from utils import utils
from ephys_data_methods import core_analysis_methods, event_analysis_master
import copy

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Event Detection - Fit Biexp Function b0, b1 to data with Sliding Window
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def sliding_window(data,
                   width_s,
                   rise,
                   decay,
                   ts,
                   downsample,
                   min_chunk_factor,
                   progress_bar_callback):
    """
    Vectorised sliding window that passes data (typically sEPSC trace) and:

    1) fits the free b0 and b1 parameter on the biexpotential function (rise and decay know) to each window
    2) correltates this best fit with the data.

    INPUTS: width_s, rise, decay - template coefficients in s
            ts - sampling step, scalar
            downsample - dict with downsample settings:
                            "on" - bool, downsample or not
                            "chunk_factor" - chunk to divide the data into
                            "downsample_factor" - factor to downsamlpe the sliding window by.
            min_chunk_factor - chunk factor is downsampling not applied

    As some data may be very large e.g. 30 million samples, x event width in samples (200-700) it is necessary to
    chunk the data with a pre-determined chunk factor. To increase speed, the sliding window can be downsampled so that
    the coefficients are calculated every N samples and interpolated later.

    Tests show that:
        1) numpy float64 data type is fatest under these conditions
       2) The chunk_factor size has no effect on time (unless is is very large), a chunk_factor size of around 20 is optimal.

    Downsample tests:
        None - 13.72 s,
        downsample factor 80 - 1.45 s (doesnt work)
                          40 - 1.83 s
                          20 - 1.446
                          12 - 2.3
                          10 - 1.973
                          2  - 5.73
    """
    all_y = data[0]
    W = int(width_s/ts)

    # Split the data indicies up based on
    # the chunk_factor
    samples_to_run = len(all_y)
    all_idx = np.arange(0, samples_to_run)

    # Set the chunk_factor to split data, check users memory
    # and increase chunk_factor if necessary
    if downsample["on"]:
        chunk_factor = downsample["chunk_factor"]
        downsample_factor = downsample["downsample_factor"]
    else:
        chunk_factor = min_chunk_factor

    idx_split = np.array_split(all_idx, chunk_factor)

    # Initialise variables to save correlation and
    # beta values in and start looping through
    # data chunks
    all_betas = utils.np_empty_nan((2, samples_to_run))
    all_corr = utils.np_empty_nan(samples_to_run)

    num_chunks = len(idx_split)
    progress_bar_callback(0, init_maximum=num_chunks)
    for i in range(num_chunks):

        progress_bar_callback(i)

        idx = idx_split[i]
        y = all_y[idx]
        samples_in_chunk = len(idx)

        # Index the sliding Window. A Window X Sample
        # Matrix is added to a row of cumulativeiny increasing samples
        # to create a cumlatively increasing with each row e.g.
        # [0 1 2 3; 0 1 2 3] + [0; 1] = [0 1 2 3; 1 2 3 4]
        idx_raw = np.tile(np.arange(0, W), (samples_in_chunk, 1)).T
        window_idx = (idx_raw + idx.reshape(1, len(idx))).astype(int)

        if downsample["on"]:
            # Downsample the window indicies to keep window size but reduce spaccing between
            # windows. See adjust_chunk_size_for_downsampling() for the importance of chunk size here.
            #
            # It is also important to re-index based on the new downsamlped indicies. Later the holding variables
            # all_betas, all_corr are filled using these indicies, then excess nan values cut.
            window_idx = core_analysis_methods.downsample_data(window_idx, downsample_factor, filter_opts=None)
            new_num_samples = window_idx.shape[1]

            if i == 0:
                current_non_nan_entries = 0
                new_chunk_start_idx = 0
            else:
                current_non_nan_entries = current_non_nan_entries + new_num_samples
                new_chunk_start_idx = current_non_nan_entries

            idx = np.arange(new_chunk_start_idx, new_chunk_start_idx + new_num_samples)

        # Before indexing, the size of the window needs to be taken into account.
        # The number of window indicies will be + W indicies larger than y because
        # the window is indexed from the first sample (so at the last sample the window will extend
        # into nothing. Here, if the loop is not the last, add W number of samples from the next chunk.
        # Otherwise, use the last sample of the array. These final W values will be cut at the end once
        # all_corr / all_betas are filled.
        if i == len(idx_split) - 1:
            pad = np.tile(y[-1], W)
        else:
            pad = all_y[window_idx[:, -1] + 1]

        pad_y = np.concatenate((y, pad))
        y = pad_y[window_idx - np.min(window_idx)]  # normalise idx for all chunks

        # Create a template of the biexpotential function
        # with the known rise / decay parameters
        x = core_analysis_methods.generate_time_array(0, width_s, W, ts)
        template = np.array([(1 - np.exp(-(x - x[0]) / rise)) * np.exp(-(x - x[0]) / decay)]).T

        # Fit the free b0 and b1 parameters of this biexpotential function
        # to every window on the dataset. Assign the outputs to the chunks idx.
        X = np.hstack([np.ones((W, 1)),
                       template])

        all_betas[:, idx] = np.linalg.inv(X.T @ X) @ X.T @ y
        yhat = all_betas[0, idx] + all_betas[1, idx] * template

        # Pearson correlate every fit biexpotential curve to the
        # data at every window. Assign the outputs to the chunks idx.
        all_corr[idx] = vectorised_pearsons(y, yhat)

    if downsample["on"]:  # remove all nans from over-initialised array, instead cut window off after interp in calling function
        all_corr = all_corr[~np.isnan(all_corr)]
        all_betas = all_betas[:, ~np.isnan(all_betas)[0]]
        return all_corr, all_betas

    return all_corr, all_betas


def vectorised_pearsons(y, yhat):
    """
    Perform row-wise pearsons correlation between two N x M matrices.
    This is useful for correlating multiple timeseries with eachother at once
    (e.g. during sliding window). Use machine epsiolon to avoid division by zero.

    See sliding_window() for details on maximising speed when vectorising
    """
    y_bar = np.mean(y, axis=0)
    yhat_bar = np.mean(yhat, axis=0)
    numerator = np.sum((y - y_bar) * (yhat - yhat_bar), axis=0) / len(y)
    demoninator = (np.std(y, axis=0) * np.std(yhat, axis=0))
    demoninator[demoninator == 0] = np.finfo(float).eps
    r = numerator / demoninator

    return r

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Event Detection - Thresholding / Smoothing Event Peak / Amplitude
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def check_peak_against_threshold_lower(peak_im,
                                       peak_idx,
                                       direction,
                                       threshold_type,
                                       threshold_lower):
    """
    Check whether the peak of an event is within predetermined threshold. This could be
    a single value (based on linear), an array of values indexed by the peak index (e.g. curve or drawn curve).
    For events analysis, it is required the event is larger (or smaller for negative events) the lower value cutoff.
    Thus within_threshold means it exceeds this lower value.

    If the event is positive, check if the peak is higher than the threshold. Otherwise check if the peak is lower.

    INPUTS:
        if curved or drawn, threshold lower is a 1 x num_samples array
        if linear, threshold lower is a [scalar]
        if rms, threshold_lower is a dict with fields ["baseline"] containing the baseline (e.g. curved, 1 x num_samples)
                                                  and ["n_times_rms"] which is a scalar value for user-specified n times the rms
                                                  calculated between the data and baseline.

    This function is called a lot - minimze coppying or addition and operate on one index only.
    Coded explicitly here as very confusing otherwise with all the diferent possibilities

    OUTPUT:
        within_threshold: bool indicating whether the peak is within the threshold (True) or not
    """
    if threshold_type == "rms":
        threshold_lower, n_times_rms = [threshold_lower["baseline"],
                                        threshold_lower["n_times_rms"]
                                        ]
        idx = 0 if len(threshold_lower) == 1 else peak_idx
        indexed_threshold_lower = threshold_lower[idx] + n_times_rms if direction == 1 else threshold_lower[idx] - n_times_rms

    elif threshold_type == "manual":
        indexed_threshold_lower = threshold_lower[0]

    elif threshold_type in ["curved", "drawn"]:
        indexed_threshold_lower = threshold_lower[peak_idx]

    compare_func = np.greater if direction == 1 else np.less
    within_threshold = compare_func(peak_im,
                                    indexed_threshold_lower)
    return within_threshold

def check_peak_height_threshold(peak_im,
                                peak_height_limit,
                                direction):
    """
    Convenience function to check  if an event peak exceeds a pre-determined
    threshold dependent on direction.
    """
    within_threshold = False
    if direction == 1:
        within_threshold = peak_im < peak_height_limit
    elif direction == -1:
        within_threshold = peak_im > peak_height_limit

    return within_threshold

def find_event_peak_after_smoothing(time_array,
                                    data,
                                    peak_idx,
                                    window,
                                    samples_to_smooth,
                                    direction):
    """
    Smooth the event region and find a new peak around the existing peak.

    A number of methods were tested for peak smoothing. The most intuitive is to find a peak on unsmoothed data,
    then set a new value for the data at that point which is an average around the region. The problem with this
    is that if a noise spike is chosen as peak, the data value will be adjusted but its position will not be at
    the natural peak of the event.

    The solution to this is to first smooth the entire event, then find the peak. The second version of this function
    smoothed the event and then searched the entire event window. However, because for threshold analysis the
    event window is defined by the decay search period, when the user set to large decay search period
    very strange behaviour occured. If an event was selected manually, the event detected could be very
    far away, because the entire "event" region was searched.

    The final, best solution is to smooth the entire event but only search a small region around the original peak
    (detected without smoothing for the new peak.

    INPUTS:
    time and data: rec x num_samples array of timepoints / data
    peak_idx: index of peak detected on unsmoothed data (indexed to full data i.e. data)
    window: window defined as the "event". This is the window for template matching or decay search region for thresholding (one value).
            If curve fitting biexponential event, this input is a list with [start stop] that define the curve fitting region (presumably
            the user has set these around the event).
    samples to smooth: number of samples to smooth (set by the user in Avearge Peak (ms)
    direction: -1 or 1, event direction

    """
    if samples_to_smooth == 0:
        samples_to_smooth = 1

    # get start of event
    if np.size(window) > 1:
        window_start, window_end = window
    else:
        window_start = window_end = window

    start_idx = peak_idx - window_start
    start_idx = is_at_least_zero(start_idx)

    # find end of event
    end_idx = peak_idx + window_end
    if end_idx + 1 >= len(time_array):
        end_idx = len(time_array) - 1

    # Index out event, smooth it and find the new peak within a window +/- x 3 the smoothing window
    ev_im = data[start_idx: end_idx + 1]
    smoothed_ev = quick_moving_average(x=ev_im,  n=samples_to_smooth)

    # Things get a little hairy with indexing here. Be careful if refactoring.
    # We want to index around the peak in terms of the smoothed event only. The we want to convert this back to full data indicies.
    # But also need to make sure the start index is never less than zero or more than the length of the event.
    ev_peak_idx = peak_idx - start_idx
    smooth_search_period = samples_to_smooth * consts("event_peak_smoothing")

    peak_search_region_start = ev_peak_idx - smooth_search_period
    peak_search_region_start = is_at_least_zero(peak_search_region_start)

    peak_search_region_end = ev_peak_idx + smooth_search_period
    if peak_search_region_end + 1 >= len(smoothed_ev):
        peak_search_region_end = len(smoothed_ev) - 2

    smoothed_ev_peak_search_region = smoothed_ev[peak_search_region_start:peak_search_region_end + 1]

    # Find the peak of the smoothed event and convert indicies back to full data
    if direction == 1:
        smoothed_peak_im = np.max(smoothed_ev_peak_search_region)
        smoothed_peak_idx = start_idx + peak_search_region_start + np.argmax(smoothed_ev_peak_search_region)
    elif direction == -1:
        smoothed_peak_im = np.min(smoothed_ev_peak_search_region)
        smoothed_peak_idx = start_idx + peak_search_region_start + np.argmin(smoothed_ev_peak_search_region)

    smoothed_peak_time = time_array[smoothed_peak_idx]

    return smoothed_peak_idx, smoothed_peak_time, smoothed_peak_im

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate Event Parameters - Baselines
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_event_baseline(time_array,
                             data,
                             peak_idx,
                             direction,
                             window):
    """
    Find the baseline of an event from the peak autoamtically ("per_event" in configs).

    First determine the region to search for the baseline. This is taken as half the search window, which
    works well when tested across events with many different types of kinetics.

    Within this region, draw a straight line from each sample to the peak. Of these lines, take the steepest (i.e. closest to the
    peak) but within the top 40% of lengths. This is to protect against noise on the peak which can cause 1-2 steep, short lines.s

    see find_event_peak_after_smoothing() for inputs
    """
    start_idx = peak_idx - window
    start_idx = is_at_least_zero(start_idx)
    idx = np.arange(start_idx, peak_idx)

    sample_times = time_array[idx]
    sample_ims = data[idx]
    peak_time = time_array[peak_idx]
    peak_im = data[peak_idx]

    slopes = ((peak_im - sample_ims) / (peak_time - sample_times))
    norms = np.sqrt((peak_im - sample_ims)**2 + (peak_time - sample_times)**2)

    perc = np.percentile(norms,
                         consts("bl_percentile"))
    slopes[norms < perc] = np.nan

    if direction == 1:
        min_slope = np.nanargmax(slopes)
    elif direction == -1:
        min_slope = np.nanargmin(slopes)

    bl_idx = start_idx + min_slope
    bl_time = time_array[bl_idx]
    bl_im = data[bl_idx]

    return bl_idx, bl_time, bl_im

def calculate_event_baseline_from_thr(time_array,
                                      data_array,
                                      thr_im,
                                      peak_idx,
                                      window,
                                      direction):
    """
    Calculate the baseline using a pre-defined threshold.

    The first data sample (prior to the event peak) to cross this
    threshold is determined as the baseline. If none cross, the nearest sample is taken.

    see find_event_peak_after_smoothing() for inputs
    """
    start_idx = peak_idx - window
    start_idx = is_at_least_zero(start_idx)

    ev_im = data_array[start_idx: peak_idx + 1]

    if direction == 1:
        under_threshold = ev_im < thr_im
    elif direction == -1:
        under_threshold = ev_im > thr_im

    try:  # take the closeest if none cross
        first_idx_under_baseline = np.max(np.where(under_threshold))
    except:
        first_idx_under_baseline = np.argmin(np.abs(ev_im - thr_im))

    bl_idx = peak_idx - window + first_idx_under_baseline
    bl_time = time_array[bl_idx]
    bl_im = data_array[bl_idx]

    return bl_idx, bl_time, bl_im

def average_baseline_period(data, bl_idx, samples_to_average):
    """
    "Look back" from the baseline index and smooth num samples to average
    """
    start_idx = bl_idx - samples_to_average
    start_idx = is_at_least_zero(start_idx)
    bl_data = np.mean(data[start_idx:bl_idx + 1])
    return bl_data

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Decay
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_event_decay_point(time_array,
                                data,
                                peak_idx,
                                bl_im,
                                direction,
                                window):
    """
    Find the event endpoint to take the decay monoexp fit to, while accounting for doublet events
    This is typically the max / min value in the window length(depending on the direction).

    INPUTS: bl_im - data value (typically Im for events) of the baseline

    see find_event_peak_after_smoothing() for other inputs

    The decay is slightly smoothed (3 samples) and the first decay point that crosses baseline Im is used.
    The smoothing is to get rid of large 1-sample spikes that may cross the baseline but be halfway down the decay.

    Just in case a doublet is included in the window length (and if wanting to increase window length somewhat) the
    event amplitudes are first weighted as the inverse of the time point. This means second spikes are exagerated.
    Then the miniumm point before the maximum is taken.
    """
    decay_period_data = data[peak_idx:peak_idx + window + 1]
    time_idx = np.arange(1, len(decay_period_data) + 1)
    offset = 100000 if direction == 1 else -100000  # ensures data wholly positive or negative (cannot use abs * direction)

    decay_period_data = decay_period_data + offset
    weight_distance_from_peak = consts("weight_distance_from_peak")
    wpeak = decay_period_data / ((1 / time_idx**weight_distance_from_peak) + np.finfo(float).eps)

    smoothed_data = quick_moving_average(decay_period_data,
                                         consts("decay_period_to_smooth"))

    if direction == 1:
        second_peak_idx = np.argmax(wpeak)

        first_decay_idx = np.argmin(np.abs(smoothed_data[0:second_peak_idx] < bl_im))
        if not np.any(first_decay_idx):
            first_decay_idx = np.argmin(decay_period_data[0:second_peak_idx])

    elif direction == -1:
        second_peak_idx = np.argmin(wpeak)
        first_decay_idx = np.argmin(np.abs(smoothed_data[0:second_peak_idx] > bl_im))
        if not np.any(first_decay_idx):
            first_decay_idx = np.argmax(decay_period_data[0:second_peak_idx])

    decay_idx = peak_idx + first_decay_idx
    decay_time = time_array[decay_idx]

    return decay_idx, decay_time

def calclate_decay_percentage_peak_from_smoothed_decay(time_array,
                                                       data,
                                                       peak_idx,
                                                       decay_idx,
                                                       bl_im,
                                                       smooth_window_samples,
                                                       ev_amplitude,
                                                       amplitude_percent_cutoff,
                                                       interp):
    """
    To increase speed there is the option to not fit a exp to decay.
    In this instance we need to calculate the decay percentage point from the data. However when there
    is noise this works poorly. Thus smooth before.

    INPUTS:
        decay_exp_fit_time - 1 x time array of timepoints between peak and decay end point
        decay_exp_fit_im - 1 x time array of datapoints (typically Im for events) between peak and decay end point
        ev_amplitude - Amplitude of the event (peak - baseline)
        amplitude_percent_cutoff - percentage of amplitude that the decay has returned to, 37% default
        peak_idx, decay_idx - sample of event peak and decay
        bm_im - value of baseline
        smooth_window_samples - width of smoothing window in samples
        interp - bool, to 200 kHz interp or not

    OUTPUTS:
        decay_percent_time: timepoint of the decay %
        decay_percent_im: data point of the decay %
        decay_time_ms: time in ms of decay_percent point - peak
        raw_smoothed_decay_time: uninterpolated decay period time, used for calculating half-width
        raw_smoothed_decay_im: uninterpolated decay period data, used for calcualting half-width
    """
    peak_time = time_array[peak_idx]
    decay_im = data[peak_idx: decay_idx + 1]

    if smooth_window_samples > len(decay_im):
        smooth_window_samples = len(decay_im)
    elif smooth_window_samples == 0:
        smooth_window_samples = 1

    smoothed_decay_im = quick_moving_average(decay_im, smooth_window_samples)
    smoothed_decay_time = time_array[peak_idx: decay_idx + 1]

    if interp:
        smoothed_decay_im, smoothed_decay_time = core_analysis_methods.twohundred_kHz_interpolate(smoothed_decay_im, smoothed_decay_time)

    decay_percent_time, decay_percent_im, decay_time_ms = find_nearest_decay_sample_to_amplitude(smoothed_decay_time,
                                                                                                 smoothed_decay_im,
                                                                                                 peak_time, bl_im,
                                                                                                 ev_amplitude,
                                                                                                 amplitude_percent_cutoff)

    raw_smoothed_decay_time = time_array[peak_idx: decay_idx + 1]  # have to re-init in case was interped
    raw_smoothed_decay_im = quick_moving_average(decay_im, smooth_window_samples)

    return decay_percent_time, decay_percent_im, decay_time_ms, \
           raw_smoothed_decay_time, raw_smoothed_decay_im

def calclate_decay_percentage_peak_from_exp_fit(decay_exp_fit_time,
                                                decay_exp_fit_im,
                                                peak_time,
                                                bl_im,
                                                ev_amplitude,
                                                amplitude_percent_cutoff,
                                                interp):
    """
    Find the nearest sample on the decay to the specified amplitude_percent_cutoff.
    This uses the decay monoexp fit if available for increased temporal resolution

    INPUTS:
        decay_exp_fit_time - 1 x time array of timepoints between peak and decay end point
        decay_exp_fit_im - 1 x time array of datapoints (typically Im for events) between peak and decay end point
        ev_amplitude - Amplitude of the event (peak - baseline)
        amplitude_percent_cutoff - percentage of amplitude that the decay has returned to, 37% default
        interp - bool, to 200 kHz interp or not

        See find_event_peak_after_smoothing() for other inputs and calclate_decay_percentage_peak_from_smoothed_decay()
        for outputs
    """
    if interp:
        decay_exp_fit_im, decay_exp_fit_time = core_analysis_methods.twohundred_kHz_interpolate(decay_exp_fit_im, decay_exp_fit_time)

    decay_percent_time, decay_percent_im, decay_time_ms = find_nearest_decay_sample_to_amplitude(decay_exp_fit_time,
                                                                                                 decay_exp_fit_im,
                                                                                                 peak_time, bl_im,
                                                                                                 ev_amplitude,
                                                                                                 amplitude_percent_cutoff)

    return decay_percent_time, decay_percent_im, decay_time_ms

def find_nearest_decay_sample_to_amplitude(decay_time,
                                           decay_im,
                                           peak_time,
                                           bl_im,
                                           ev_amplitude,
                                           amplitude_percent_cutoff):
    """
    For calculating the decay %, find the nearest datapoint to the decay %.

    e.g. if user has set decay % to 37%, we want to find the datapoint that is 37% of the decay
    amplitude. This might not be an exact sample so find the nearest.
    See calclate_decay_percentage_peak_from_smoothed_decay for input / output.
    """
    amplitude_fraction = amplitude_percent_cutoff / 100
    amplitude_fraction = bl_im + ev_amplitude * amplitude_fraction

    nearest_im_to_amp_idx = np.argmin(np.abs(decay_im - amplitude_fraction))
    decay_percent_im = decay_im[nearest_im_to_amp_idx]
    decay_percent_time = decay_time[nearest_im_to_amp_idx]
    decay_time_ms = (decay_percent_time - peak_time) * 1000

    return decay_percent_time, decay_percent_im, decay_time_ms

def adjust_peak_to_optimise_tau(time_,
                                data,
                                peak_idx,
                                decay_idx,
                                run_settings):
    """
    Sometimes the tau will be very high. This is usually due to noise on the peak. Sometimes moving a few samples
    left or right (usually right towards the decay end) can improve the fit. First just towards the direction fo decay.

    Here we loop through sample offset from peak until we get a fit where the tau is in range. If cannot fix, return FAlse.

    See fit_monoexp_function_to_decay().
    """
    decay_period_samples = decay_idx - peak_idx
    forward_samples = np.arange(1, np.floor(decay_period_samples * 0.1))
    backward_samples = np.arange(-np.floor(decay_period_samples * 0.5), -1)
    offsets_to_try = np.concatenate([forward_samples, backward_samples]).astype(int)

    for offset in offsets_to_try:
        new_peak_to_try = peak_idx + offset
        try_decay_time = time_[new_peak_to_try: decay_idx + 1]
        try_decay_im = data[new_peak_to_try: decay_idx + 1]

        if len(try_decay_time) < 2:
            continue

        coefs, try_decay_exp_fit_im = core_analysis_methods.fit_curve("monoexp",
                                                                      try_decay_time,
                                                                      try_decay_im,
                                                                      run_settings["direction"])

        if np.any(coefs):
            try_exp_fit_tau_ms = coefs[2] * 1000
            if event_analysis_master.is_tau_in_limits(run_settings,
                                                      try_exp_fit_tau_ms):
                return try_decay_time, try_decay_exp_fit_im, try_exp_fit_tau_ms

    return False, False, False

def calculate_event_rise_time(time_array,
                              data,
                              bl_idx,
                              peak_idx,
                              bl_im,
                              peak_im,
                              direction,
                              min_cutoff_perc,
                              max_cutoff_perc,
                              interp=False):
    """
    Calculate the rise time of the event using core_analysis_methods (see these methods for input / outputs)
    """
    ev_data = data[bl_idx: peak_idx + 1]
    ev_time = time_array[bl_idx: peak_idx + 1]

    calculate_slope_func = core_analysis_methods.calc_rising_slope_time if direction == 1 else core_analysis_methods.calc_falling_slope_time

    max_time, max_data, min_time, min_data, rise_time = calculate_slope_func(ev_data,
                                                                             ev_time,
                                                                             bl_im,
                                                                             peak_im,
                                                                             min_cutoff_perc,
                                                                             max_cutoff_perc,
                                                                             interp)
    return max_time, max_data, min_time, min_data, rise_time

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Moving Average
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def quick_moving_average(x, n):
    """
    Combine np.convolve for speed but then smooth gracefully
    at the edges by iteratively decreasing the smoothing window.
    """
    out = np.convolve(x, np.ones(n) / n, mode="same")

    num_samples = len(x)
    window = np.floor(n / 2).astype(int)
    for i in range(window):
        if i == 0:
            out[i] = np.mean(x[0:n])

        elif i < window:
            out[i] = np.mean(x[0: i + window])

    for i in range(num_samples - window, num_samples):
        if i == num_samples:
            out[i] = np.mean(x[num_samples - n:num_samples])

        elif i > num_samples - window - 1:
            out[i] = np.mean(x[i - window: num_samples])

    return out

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Convenience functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def is_at_least_zero(start_idx):
    if start_idx < 0:
        start_idx = 0
    return start_idx

def normalise_amplitude(data,
                        remove_baseline=False):
    """
    Keep the amplitude of the signal to 1. If remove basleine, centre to the first sample.
    Don"t demean as the first-sample is more interesting for cutting the left edge
    during generating the template.
    """
    if remove_baseline:
        data = data - data[0]

    amplitude = find_amplitude_min_or_max(data,
                                          use_first_sample_as_baseline=remove_baseline)
    norm_curve = data * (1 / np.abs(amplitude))

    return norm_curve


def find_amplitude_min_or_max(data, use_first_sample_as_baseline):
    """
    Only works if data normanlised to zero!!

    Find the minimum or maximum peak of a trace (normalised to zero) depending on which is larger.
    Useful for single events where it is not known if they are positive or negative.
    """
    abs_min = np.abs(np.min(data))
    abs_max = np.abs(np.max(data))

    if abs_max > abs_min:
        baseline = np.min(data)
        peak = np.max(data)
    else:
        peak = np.min(data)
        baseline = np.max(data)

    if use_first_sample_as_baseline:
        baseline = data[0]

    amplitude = peak - baseline

    return amplitude

def consts(constant_name):
    """
    Constants for various kinetics calculation derived from testing across many event types

    constant_name -

    "bl_pecentile" - percentile for slope length used in auto. baseline detection
    "weight_distance_from_peak" - weight on the automatic decay endpoint finder to increase noise by distance from peak.
                                  Rarely, the exp grow too large and undefined, throwing a numpy RunTimeWarning (rare)
    "decay_period_to_smooth" - smooth the decay period when auto-detected decay period end to avoid noise spikes biasing the result.

    """
    if constant_name == "bl_percentile":
        const = 60

    elif constant_name == "weight_distance_from_peak":
        const = 5

    elif constant_name == "decay_period_to_smooth":
        const = 3

    elif constant_name == "event_peak_smoothing":
        const = 3

    return const

