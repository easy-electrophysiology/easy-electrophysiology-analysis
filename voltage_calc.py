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

from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Dict, Literal, Tuple, Union

import bottleneck as bn
import numpy as np
import scipy.fftpack as fftpack
from ephys_data_methods import core_analysis_methods
from scipy import signal

if TYPE_CHECKING:
    from custom_types import Direction, FalseOrFloat, FalseOrIdxFloatFloat, Int, NpArray64
    from numpy.typing import NDArray

# --------------------------------------------------------------------------------------
# Event Detection - Sliding Window
# --------------------------------------------------------------------------------------


def clements_bekkers_sliding_window(
    data: NpArray64, template: NDArray, u: Callable
) -> Tuple[NpArray64, NpArray64, NpArray64]:
    """
    Sliding window template matching with implementation details from Clements and
    Bekkers (1997). This is a highly optimised method to perform sliding window
    template matching. See the original paper for derivation.

    Pad the end of the data so the last n samples are not ignored. The correlation
    method is also calculated using this algorithm for speed. This necessitates
    that the correlation coefficient does not account for negative correlations
    between template and data. However, as the template is first scaled to match
    the data, negative correlates will not be possible.

    Testing shows this implementation exactly matches a version
    in which the r can be negative or positive.

    data: 1 x t array of data
    template: 1 x n array of the template biexponential function
    u: the callback function for the progress bar.

    Clements, J. D., & Bekkers, J. M. (1997). Detection of spontaneous synaptic
    events with an optimally selected template. Biophysical Journal, 73(1), 220-229.

    """
    n = template.size

    data = np.hstack([data, np.tile(data[-1], n - 1)])
    u()

    params = SimpleNamespace(
        n=n,
        sum_template_data=signal.correlate(data, template, mode="valid"),
        sum_data=bn.move_sum(data, n)[n - 1 :],
        sum_data2=bn.move_sum(data**2, n)[n - 1 :],
        sum_template=np.sum(template),
        sum_template2=np.sum(template**2),
    )
    u()

    scale = fit_scale(params)
    offset = fit_offset(scale, params)
    std_error, sse = calc_std_error(offset, scale, params)
    detection_criterion = scale / std_error
    betas = np.vstack([offset, scale])
    sst = calc_sst(params)
    r, __ = calculate_r_and_r_squared(sse, sst)
    u()

    return detection_criterion, betas, r


def fit_scale(p: SimpleNamespace) -> NpArray64:
    """
    An efficient algorithm for fitting the scale parameter (i.e. b1) of a
    template to data.

    Intuitively it can be interpreted as the scaled covariance (minus some offset)
    normalised to the scaled variance (minus some offset). When covariance is
    maximal and data is higher than template, the scale will represent the y-axis
    scaling necessary to match the amplitude of the template to the data.

    However, if the data is higher than template but the covariance is zero,
    the scale will be zero i.e. it is not possible to scale the template to
    the data. As such this parameter represents the y-axis scaling of the
    template to the data while taking into account the covariance of
    template and data
    """
    return (p.sum_template_data - p.sum_template * p.sum_data / p.n) / (
        p.sum_template2 - p.sum_template * p.sum_template / p.n
    )


def fit_offset(scale: NpArray64, p: SimpleNamespace) -> NpArray64:
    """
    Fit the offset based on data, template and template scaling.
    """
    return (p.sum_data - scale * p.sum_template) / p.n


def calc_std_error(offset: NpArray64, scale: NpArray64, p: SimpleNamespace) -> Tuple[NpArray64, NpArray64]:
    """
    Calculate the standard error between template and data
    with offset, scale and original template factored out.
    """
    sse = (
        p.sum_data2
        + scale**2 * p.sum_template2
        + p.n * offset**2
        - 2 * (scale * p.sum_template_data + offset * p.sum_data - scale * offset * p.sum_template)
    )
    error = sse / (p.n - 1)

    # fix floating point rounding error, this value
    # is later a divisor so add eps.
    error[error < 0] = 0 + np.finfo(float).eps
    std_error = np.sqrt(error)

    return std_error, sse


def calc_sst(p: SimpleNamespace) -> NpArray64:
    return p.sum_data2 - p.sum_data**2 / p.n


def calculate_r_and_r_squared(sse: NpArray64, sst: NpArray64) -> Tuple[NpArray64, NpArray64]:
    """
    Fast calculation of r from sse, sst array. Note ignores sign.
    """
    r_squared = 1 - sse / sst

    # floating point round error from sse / sst or 0 sst
    r_squared[np.logical_or(r_squared < 0, np.isnan(r_squared))] = 0

    r = np.sqrt(r_squared)
    assert not np.any(np.isnan(r))  # TODOu this can be removed after testing beta

    return r, r_squared


# --------------------------------------------------------------------------------------
# Event Detection - Deconvolution
# --------------------------------------------------------------------------------------


def get_filtered_template_data_deconvolution(
    data: NpArray64,
    template: NpArray64,
    fs: NpArray64,
    low_hz: NpArray64,
    high_hz: NpArray64,
) -> NpArray64:
    """
    Return filtered deconvolution of template and data.

    Data and template are transformed to the Fourier domain (where deconvolution is
    point-wise division) and filter with a Gaussian window before inverse FFT.

    INPUTS:
        data: 1 x N signal
        template: 1 x T template signal
        fs: sampling frequency of the original signal
        low_hz, high_hz: frequency cutoffs in Hz

    Pernia-Andrade, A., Goswami, S. P., Stickler, Y., Frobe, U. Schlogl, A., & Jonas,
    P. (2012). A Deconvolution-Based Method with High Sensitivity and Temporal
    Resolution for Detection of Spontaneous Synaptic Currents In Vitro and In Vivo.
    Biophysical Journal. 103(7), 1429-1439.
    """
    num_samples = data.shape[0]
    pad_template = np.zeros(num_samples, dtype=np.float64)
    pad_template[0 : template.shape[0]] = template

    fft_template = fftpack.fft(pad_template)
    fft_data = fftpack.fft(data)
    fft_deconv = fft_data / fft_template
    filt_fft_deconv = fft_filter_gaussian_window(fft_deconv, low_hz, high_hz, num_samples, fs)
    filt_deconv = np.real(fftpack.ifft(filt_fft_deconv))

    return filt_deconv


def fft_filter_gaussian_window(
    data: NpArray64,
    low_hz: NpArray64,
    high_hz: NpArray64,
    num_samples: Int,
    fs: NpArray64,
) -> NpArray64:
    """
    Apply a band-pass filter in the frequency domain, low-pass Gaussian
    window and sharp high-pass cutoff.

    INPUTS:
        data: Fourier-transformed signal 1 ... N
        see get_filtered_template_data_deconvolution() for other inputs
    """
    freqs = fftpack.fftfreq(num_samples, 1 / fs)
    gauss_window = 1 / np.sqrt(2 * np.pi * high_hz / fs) * np.exp(-0.5 * (freqs / high_hz) ** 2)
    gauss_window[np.where(np.abs(freqs) < low_hz)] = 0
    data = gauss_window * data * fs
    return data


def calculate_deconv_detection_threshold(detection_coefs: NpArray64, n_times_sigma: NpArray64) -> SimpleNamespace:
    """
    Calculate sigma, the detection threshold for deconvolution event detection.
    Sigma is the standard deviation from the all-points histogram of the
    deconvolution. For multi-record files the deconvolution is collapsed
    across all records.

    The histogram is 10x upsampled by linear interpolation before Gaussian fitting.

    INPUTS:
        detection_coefs: the Record x N deconvolution of data with event template
        n_times_sigma: the user-specified sigma cutoff
    """
    bin_edges, frequencies = calculate_histogram_bins_and_freq(detection_coefs)
    detection_threshold, mu, sigma, gaussian_fit = calculate_theta_from_histogram(bin_edges, frequencies, n_times_sigma)

    detection_threshold_info = SimpleNamespace(
        gaussian_fit=gaussian_fit,
        bin_edges=bin_edges,
        frequencies=frequencies,
        mu=mu,
        sigma=sigma,
        detection_threshold=detection_threshold,
    )

    return detection_threshold_info


def calculate_theta_from_histogram(
    bin_edges: NpArray64, frequencies: NpArray64, n_times_std: NpArray64
) -> Tuple[FalseOrFloat, ...]:
    """
    Fit a Gaussian function to the all-points histogram from deconvolution.
    see calculate_histogram_bins_and_freq() for INPUTS

    """
    coefs, gaussian_fit, __ = core_analysis_methods.fit_curve(
        "gaussian", bin_edges, frequencies, normalise_time=False, direction=1
    )

    if coefs is False:
        return False, False, False, False

    __, mu, sigma = coefs
    theta = n_times_std * sigma

    return theta, mu, sigma, gaussian_fit


def calculate_histogram_bins_and_freq(
    detection_coefs: NpArray64,
) -> Tuple[NpArray64, NpArray64]:
    """
    Calculate the all-points histogram for deconvolution event detection.

    Histogram is 10x linear interpolated.

    INPUT: detection_coefs: Record x N data - template deconvolution, some records
    (not analysed) may be filled with Nan and are removed in the first line while
    the record structure is collapsed.

    OUTPUT:
         interp_bin_edges: upsampled left-bin edges of the all-points histogram
         interp_hist_y: upsampled frequencies of the all-points histogram

    """
    all_detection_coefs = detection_coefs[~np.isnan(detection_coefs)]
    detection_coefs_min, detection_coefs_max = [
        np.min(all_detection_coefs),
        np.max(all_detection_coefs),
    ]
    num_bins = int(np.sqrt(len(all_detection_coefs)))

    hist_ = np.histogram(
        all_detection_coefs,
        bins=num_bins,
        range=(detection_coefs_min, detection_coefs_max),  # type: ignore
    )

    hist_y = hist_[0]
    bin_edges = hist_[1][1:]
    interp_hist_y = core_analysis_methods.interpolate_data(hist_y, bin_edges, "linear", 10, 0)
    interp_bin_edges = core_analysis_methods.interpolate_data(bin_edges, bin_edges, "linear", 10, 0)

    return interp_bin_edges, interp_hist_y


# --------------------------------------------------------------------------------------
# Event Detection - Thresholding / Smoothing Event Peak / Amplitude
# --------------------------------------------------------------------------------------


def check_peak_against_threshold_lower(peak_im: NpArray64, peak_idx: Int, run_settings: Dict) -> NDArray[np.bool_]:
    """
    Check whether the peak of an event is within predetermined threshold. This
    could be a single value (based on linear), an array of values indexed by
    the peak index (e.g. curve or drawn curve).

    For events analysis, it is required the event is larger (or smaller for
    negative events) the lower value cutoff. Thus within_threshold means it
    exceeds this lower value.

    If the event is positive, check if the peak is higher than the threshold.
    Otherwise check if the peak is lower.

    INPUTS:
        if curved or drawn, threshold lower is a 1 x num_samples array
        if linear, threshold lower is a [scalar]
        if rms, threshold_lower is a dict with fields ["baseline"] containing the
        baseline (e.g. curved, 1 x num_samples) and ["n_times_rms"] which is a
        scalar value for user-specified n=times the rms calculated between the
        data and baseline.

    OUTPUT:
        within_threshold: bool indicating whether the peak is within the
        threshold (True) or not
    """
    direction, threshold_type, threshold_lower, rec = (
        run_settings["direction"],
        run_settings["threshold_type"],
        run_settings["threshold_lower"],
        run_settings["rec"],
    )
    if threshold_type == "rms":
        baseline_info, n_times_rms = [
            threshold_lower["baseline_info"],
            threshold_lower["n_times_rms"],
        ]
        if baseline_info["type"] == "per_event":
            baseline = baseline_info["baseline"][rec]
        elif baseline_info["type"] == "manual":
            baseline = baseline_info["baseline"]
        elif baseline_info["type"] in ["drawn", "curved"]:
            baseline = baseline_info["baseline"][rec][peak_idx]

        indexed_threshold_lower = baseline + n_times_rms[rec] if direction == 1 else baseline - n_times_rms[rec]

    elif threshold_type == "manual":
        # these are the same for every record (single linear cutoff)
        # but organise per-rec for consistency
        indexed_threshold_lower = threshold_lower[0]

    elif threshold_type in ["curved", "drawn"]:
        indexed_threshold_lower = threshold_lower[rec][peak_idx]

    compare_func = np.greater if direction == 1 else np.less
    within_threshold = compare_func(peak_im, indexed_threshold_lower)
    return within_threshold


def check_peak_height_threshold(
    peak_im: NpArray64, peak_height_limit: NpArray64, direction: Direction
) -> NDArray[np.bool_]:
    """
    Convenience function to check  if an event peak exceeds a
    pre-determined threshold dependent on direction.
    """
    within_threshold = False
    if direction == 1:
        within_threshold = peak_im < peak_height_limit
    elif direction == -1:
        within_threshold = peak_im > peak_height_limit

    return within_threshold


def find_event_peak_after_smoothing(
    time_array: NpArray64,
    data: NpArray64,
    peak_idx: Int,
    window: Int,
    samples_to_smooth: Int,
    direction: Direction,
) -> Tuple[Int, NpArray64, NpArray64]:
    """
    Smooth the event region and find a new peak around the existing peak.

    A number of methods were tested for peak smoothing. The most intuitive
    is to find a peak on unsmoothed data, then set a new value for the data
    at that point which is an average around the region. The problem with this
    is that if a noise spike is chosen as peak, the data value will be adjusted
    but its position will not be at the natural peak of the event.

    The solution to this is to first smooth the entire event, then find the peak.
    The second version of this function smoothed the event and then searched the
    entire event window. However, because for threshold analysis the event window
    is defined by the decay search period, when the user set to large decay search
    period very strange behaviour occurred. If an event was selected manually, the
    event detected could be very far away, because the entire "event" region was
    searched.

    The final, best solution is to smooth the entire event but only search a small
    region around the original peak (detected without smoothing for the new peak.

    INPUTS:
    time and data: rec x num_samples array of timepoints / data
    peak_idx: index of peak detected on unsmoothed data (indexed to full data i.e. data)
    window: window defined as the "event". This is the window for template matching
            or decay search region for thresholding (one value).
            If curve fitting biexponential event, this input is a list with [start
            stop] that define the curve fitting region (presumably
            the user has set these around the event).
    samples to smooth: number of samples to smooth (set by the user in Avearge Peak (ms)
    direction: -1 or 1, event direction

    """
    if samples_to_smooth == 0:
        samples_to_smooth = 1

    window_start = window_end = window

    start_idx = peak_idx - window_start
    start_idx = is_at_least_zero(start_idx)

    # find end of event
    end_idx = peak_idx + window_end
    if end_idx + 1 >= len(time_array):
        end_idx = len(time_array) - 1

    # Index out event, smooth it and find the new peak within a window +/- x 3 the
    # smoothing window
    ev_im = data[start_idx : end_idx + 1]
    smoothed_ev = quick_moving_average(x=ev_im, n=samples_to_smooth)

    # Things get a little hairy with indexing here. Be careful if refactoring.
    # We want to index around the peak in terms of the smoothed event only. Then we
    # want to convert this back to full data indices.
    # But also need to make sure the start index is never less than zero or more than
    # the length of the event.
    ev_peak_idx = peak_idx - start_idx
    smooth_search_period = samples_to_smooth * consts("event_peak_smoothing")

    peak_search_region_start = ev_peak_idx - smooth_search_period
    peak_search_region_start = is_at_least_zero(peak_search_region_start)

    peak_search_region_end = ev_peak_idx + smooth_search_period
    if peak_search_region_end + 1 >= len(smoothed_ev):
        peak_search_region_end = len(smoothed_ev) - 2

    smoothed_ev_peak_search_region = smoothed_ev[peak_search_region_start : peak_search_region_end + 1]

    # Find the peak of the smoothed event and convert indices back to full data
    if direction == 1:
        smoothed_peak_im = np.max(smoothed_ev_peak_search_region)
        smoothed_peak_idx = start_idx + peak_search_region_start + np.argmax(smoothed_ev_peak_search_region)
    elif direction == -1:
        smoothed_peak_im = np.min(smoothed_ev_peak_search_region)
        smoothed_peak_idx = start_idx + peak_search_region_start + np.argmin(smoothed_ev_peak_search_region)

    smoothed_peak_time = time_array[smoothed_peak_idx]

    return smoothed_peak_idx, smoothed_peak_time, smoothed_peak_im  # type: ignore


# --------------------------------------------------------------------------------------
# Calculate Event Parameters - Baselines
# ---------------------------------------------------------------------------------------


def calculate_event_baseline(
    time_array: NpArray64,
    data: NpArray64,
    peak_idx: Int,
    direction: Direction,
    window: Int,
) -> FalseOrIdxFloatFloat:
    """
    Find the baseline of an event from the peak automatically ("per_event" in configs).

    First determine the region to search for the baseline. This is taken as half the
    search window, which works well when tested across events with different types
    of kinetics.

    Within this region, draw a straight line from each sample to the peak. Of these
    lines, take the steepest (i.e. closest to the peak) but within the top 40% of
    lengths. This is to protect against noise on the peak which can cause 1-2 steep,
    short lines.s

    see find_event_peak_after_smoothing() for inputs
    """
    start_idx = peak_idx - window
    start_idx = is_at_least_zero(start_idx)
    idx = np.arange(start_idx, peak_idx)

    sample_times = time_array[idx]
    sample_ims = data[idx]
    peak_time = time_array[peak_idx]
    peak_im = data[peak_idx]

    slopes = (peak_im - sample_ims) / (peak_time - sample_times)
    norms = np.sqrt((peak_im - sample_ims) ** 2 + (peak_time - sample_times) ** 2)

    if not np.any(norms):  # added for SUPPORT_CODE: 101
        return False, False, False

    perc = np.percentile(norms, consts("bl_percentile"))
    slopes[norms < perc] = np.nan

    if direction == 1:
        min_slope = np.nanargmax(slopes)
    elif direction == -1:
        min_slope = np.nanargmin(slopes)

    bl_idx = start_idx + min_slope
    bl_time = time_array[bl_idx]
    bl_im = data[bl_idx]

    return bl_idx, bl_time, bl_im


def enhanced_baseline_calculation(
    data: NpArray64,
    time_: NpArray64,
    bl_idx: Int,
    bl_im: NpArray64,
    event_info: Dict,
    run_settings: Dict,
) -> Union[Literal[False], Dict]:
    """
    Enhanced baseline detection based on moving the baseline close to the event foot.
    A straight line is drawn between the 20-80 rise time points and the intersection
    with the detected baseline is taken as the event foot.

    This will only change the position of the baseline, not its Im value which is
    previously calculated.

    Jonas P, Major G, Sakman B. (1993) Quantal components of unitary EPSCs at the
    mossy fibre synapse on CA3 pyramidal cells of rat hippocampus.
    The Journal of Physiology; 472: 615-663.
    """
    rise_min_time, rise_min_im, rise_max_time, rise_max_im, rise_time = calculate_event_rise_time(
        time_,
        data,
        bl_idx,
        event_info["peak"]["idx"],  # type: ignore
        bl_im,
        event_info["peak"]["im"],  # type: ignore
        min_cutoff_perc=consts("foot_detection_low_rise_percent"),
        max_cutoff_perc=consts("foot_detection_high_rise_percent"),
        interp=False,
    )

    rise_time = rise_max_time - rise_min_time
    if rise_time == 0:
        return False

    slope = (rise_max_im - rise_min_im) / rise_time
    delta_t = abs(rise_max_im - bl_im) / abs(slope)
    foot_time = rise_max_time - delta_t

    timepoints = time_[bl_idx : event_info["peak"]["idx"] + 1]  # type: ignore
    datapoints = data[bl_idx : event_info["peak"]["idx"] + 1]  # type: ignore

    if len(timepoints) < 2:
        return False

    # Take the min euclidean distance of the slope / baseline
    # intersection to the data as the new baseline
    euc_distance = core_analysis_methods.nearest_point_euclidean_distance(foot_time, timepoints, bl_im, datapoints)

    new_bl_idx_offset = np.argmin(euc_distance)

    new_bl_idx = bl_idx + new_bl_idx_offset
    new_bl_time = time_[new_bl_idx]

    return {"idx": new_bl_idx, "time": new_bl_time, "im": bl_im}


def calculate_event_baseline_from_thr(
    time_array: NpArray64,
    data_array: NpArray64,
    thr_im: NpArray64,
    peak_idx: Int,
    window: Int,
    direction: Direction,
) -> Tuple[Int, NpArray64, NpArray64]:
    """
    Calculate the baseline using a pre-defined threshold.

    The first data sample (prior to the event peak) to cross this
    threshold is determined as the baseline.
    If none cross, the nearest sample is taken.

    see find_event_peak_after_smoothing() for inputs
    """
    start_idx = peak_idx - window
    start_idx = is_at_least_zero(start_idx)

    ev_im = data_array[start_idx : peak_idx + 1]

    if direction == 1:
        under_threshold = ev_im < thr_im
    elif direction == -1:
        under_threshold = ev_im > thr_im

    try:  # take the closest if none cross
        first_idx_under_baseline = np.max(np.where(under_threshold))
    except:
        first_idx_under_baseline = np.argmin(np.abs(ev_im - thr_im))

    bl_idx = start_idx + first_idx_under_baseline
    bl_time = time_array[bl_idx]
    bl_im = data_array[bl_idx]

    return bl_idx, bl_time, bl_im


def average_baseline_period(data: NpArray64, bl_idx: Int, samples_to_average: Int) -> NpArray64:
    """
    "Look back" from the baseline index and smooth num samples to average
    """
    start_idx = bl_idx - samples_to_average
    start_idx = is_at_least_zero(start_idx)
    bl_data = np.mean(data[start_idx : bl_idx + 1])
    bl_data = np.array(bl_data)  # for typing
    return bl_data


def update_baseline_that_is_before_previous_event_peak(
    data: NpArray64,
    time_: NpArray64,
    peak_idx: Int,
    run_settings: Dict,
) -> Tuple[Int, NpArray64, NpArray64]:
    """
    If event baseline was before the peak of the last baseline, it must
    have been selected in error and means the previous event is very close
    in time and almost certainly a doublet. As such, use the max/min of the
     data values in-between the two close peaks as the second event's baseline.
    """
    bl_func = np.argmin if run_settings["direction"] == 1 else np.argmax
    bl_idx = bl_func(data[run_settings["previous_event_idx"] : peak_idx + 1])
    bl_idx += run_settings["previous_event_idx"]
    bl_time = time_[bl_idx]
    bl_im = data[bl_idx]

    return bl_idx, bl_time, bl_im


# --------------------------------------------------------------------------------------
# Decay
# --------------------------------------------------------------------------------------

# Decay / Event Endpoint Search Methods
# --------------------------------------------------------------------------------------


def calculate_event_decay_point_entire_search_region(
    time_array: NpArray64,
    data: NpArray64,
    peak_idx: Int,
    window: Int,
    run_settings: Dict,
    bl_im: NpArray64,
) -> FalseOrIdxFloatFloat:
    """
    Decay index is taken as either the full length of the decay_search_period
    option or up until the next detected baseline.

    The run_settings["next_event_baseline_idx"] < peak_idx conditional should only
    occur when legacy baseline detection options are selected.
    """
    if run_settings["next_event_baseline_idx"] is None:
        decay_idx = peak_idx + window if peak_idx + window < len(data) else len(data) - 1

    elif run_settings["next_event_baseline_idx"] < peak_idx:
        decay_idx, decay_point_time, decay_point_data = calculate_event_decay_point_crossover_methods(
            time_array,
            data,
            peak_idx,
            bl_im,
            run_settings["direction"],
            window,
            use_legacy=True,
        )
        return decay_idx, decay_point_time, decay_point_data

    elif peak_idx + window > run_settings["next_event_baseline_idx"]:
        decay_idx = run_settings["next_event_baseline_idx"] - 1

    else:
        decay_idx = peak_idx + window

    decay_point_time = time_array[decay_idx]
    decay_point_data = data[decay_idx]

    return decay_idx, decay_point_time, decay_point_data


def decay_point_first_crossover_method(
    time_array: NpArray64,
    data: NpArray64,
    peak_idx: Int,
    window: Int,
    run_settings: Dict,
    bl_im: NpArray64,
) -> FalseOrIdxFloatFloat:
    """
    Coordinate the event endpoint calculation while adjusting for other events.

    First calculate the decay endpoint using the smoothed first crossover method. Then
    check if it is beyond the net baseline idx - if so set to 1 sample before the
    next baseline idx.
    """
    decay_idx, __, __ = calculate_event_decay_point_crossover_methods(
        time_array,
        data,
        peak_idx,
        bl_im,
        run_settings["direction"],
        window,
        use_legacy=False,
    )

    if decay_idx is False:
        return False, False, False

    if run_settings["next_event_baseline_idx"] is not None and decay_idx >= run_settings["next_event_baseline_idx"]:
        decay_idx = run_settings["next_event_baseline_idx"] - 1

    decay_point_time = time_array[decay_idx]
    decay_point_data = data[decay_idx]

    return decay_idx, decay_point_time, decay_point_data


def calculate_event_decay_point_crossover_methods(
    time_array: NpArray64,
    data: NpArray64,
    peak_idx: Int,
    bl_im: NpArray64,
    direction: Direction,
    window: Int,
    use_legacy: bool,
) -> FalseOrIdxFloatFloat:
    """
    Coordinate two different methods of decay endpoint calculation

    Find the event endpoint to take the decay monoexp fit to, while accounting for
    doublet events.  This is typically the max / min value in the window length
    (depending on the direction).
    """
    decay_period_data = data[peak_idx : peak_idx + window + 1]

    if len(decay_period_data) < 2:
        return False, False, False

    offset = (
        1000000 if direction == 1 else -1000000
    )  # ensures data wholly positive or negative (cannot use abs * direction)
    decay_period_data = decay_period_data + offset
    offset_bl_im = bl_im + offset

    if use_legacy:
        decay_idx = decay_endpoint_legacy_method(decay_period_data, peak_idx, direction)
    else:
        decay_idx = decay_endpoint_improved_method(decay_period_data, peak_idx, offset_bl_im, direction)

    decay_point_time = time_array[decay_idx]
    decay_point_data = data[decay_idx]

    return decay_idx, decay_point_time, decay_point_data


def decay_endpoint_improved_method(
    decay_period_data: NpArray64, peak_idx: Int, bl_im: NpArray64, direction: Direction
) -> Int:
    """
    Improved decay endpoint in which data is smoothed to avoid noise peaks,
    and the first sample to cross the baseline is taken as the end point.
    Quite destrictive - usually underestimates slightly the end of the event.

    If no sample crosses the baseline, take the nearest sample to the baseline.
    """
    smoothed_data = quick_moving_average(decay_period_data, consts("decay_period_to_smooth"))

    if direction == 1:
        first_decay_idx = find_first_baseline_crossing(smoothed_data, bl_im, direction)
        if first_decay_idx is False:
            first_decay_idx = np.argmin(decay_period_data)

    elif direction == -1:
        first_decay_idx = find_first_baseline_crossing(smoothed_data, bl_im, direction)
        if first_decay_idx is False:
            first_decay_idx = np.argmax(decay_period_data)

    decay_idx = peak_idx + first_decay_idx

    return decay_idx


def find_first_baseline_crossing(data: NpArray64, bl_im: NpArray64, direction: Direction) -> Union[Literal[False], Int]:
    """
    Find the idx of the first sample that crosses the baseline idx.
    Account for event direction (e.g. if event is negative we are
    looking for the first datapoint that crosses above the baseline idx).

    return False if no sample crosses the baseline.
    """
    try:
        if direction == 1:
            first_decay_idx = np.min(np.where(data < bl_im))

        elif direction == -1:
            first_decay_idx = np.min(np.where(data > bl_im))

        return first_decay_idx

    except ValueError:
        return False


def decay_endpoint_legacy_method(decay_period_data: NpArray64, peak_idx: Int, direction: Direction) -> Int:
    """
    Old method of finding end of event, simply taking the min / max value
    of the search region.

    Just in case a doublet is included in the window length (and if wanting
    to increase window length somewhat) the event amplitudes are first
    weighted as the inverse of the time point. This means second spikes
    are exaggerated. Then the minimum point before the maximum is taken.

    This method is depreciated and only kept for past users who may need
    it and is earmarked for full depreciation in the future. It is
    superseded by methods that take into account the next event, and so
    do not need to find the next peak.
    """
    time_idx = np.arange(1, len(decay_period_data) + 1)
    weight_distance_from_peak = consts("weight_distance_from_peak")
    wpeak = decay_period_data / ((1 / time_idx**weight_distance_from_peak) + np.finfo(float).eps)

    if direction == 1:
        second_peak_idx = np.argmax(wpeak)
        first_decay_idx = np.argmin(decay_period_data[0:second_peak_idx])

    elif direction == -1:
        second_peak_idx = np.argmin(wpeak)
        first_decay_idx = np.argmax(decay_period_data[0:second_peak_idx])

    decay_idx = peak_idx + first_decay_idx

    return decay_idx


# Decay Percent Calculation
# --------------------------------------------------------------------------------------


def calculate_decay_to_point_from_smoothed_decay(
    time_array: NpArray64,
    data: NpArray64,
    peak_idx: Int,
    decay_idx: Int,
    bl_im: NpArray64,
    smooth_window_samples: Int,
    ev_amplitude: NpArray64,
    amplitude_percent_cutoff: NpArray64,
    interp: bool,
) -> Tuple[NpArray64, NpArray64, NpArray64, NpArray64, NpArray64]:
    """
    To increase speed there is the option to not fit an exponential to decay.
    In this instance we need to calculate the decay percentage point from the data.
    However, when there is noise this works poorly. Thus smooth before.

    INPUTS:
        decay_exp_fit_time - 1 x time array of timepoints between peak and decay end
        point
        decay_exp_fit_im - 1 x time array of data-points (typically Im for events)
        between peak and decay end point
        ev_amplitude - Amplitude of the event (peak - baseline)
        amplitude_percent_cutoff - percentage of amplitude that the decay has
        returned to, 37% default
        peak_idx, decay_idx - sample of event peak and decay
        bm_im - value of baseline
        smooth_window_samples - width of smoothing window in samples
        interp - bool, to 200 kHz interp or not

    OUTPUTS:
        decay_timepoint: timepoint of the decay %
        decay_value: data point of the decay %
        decay_time_ms: decay time in ms
        raw_smoothed_decay_time: uninterpolated decay period time, used for
        calculating half-width
        raw_smoothed_decay_im: uninterpolated decay period data, used for calculating
        half-width
    """
    (
        peak_time,
        smoothed_decay_time,
        smoothed_decay_im,
        raw_smoothed_decay_time,
        raw_smoothed_decay_im,
    ) = get_decay_data_for_smoothed_decay(
        time_array,
        data,
        peak_idx,
        decay_idx,
        smooth_window_samples,
        interp,
    )

    (decay_timepoint, decay_value, decay_time_ms,) = find_nearest_decay_sample_to_amplitude(
        smoothed_decay_time,
        smoothed_decay_im,
        peak_time,
        bl_im,
        ev_amplitude,
        amplitude_percent_cutoff,
    )

    return (
        decay_timepoint,
        decay_value,
        decay_time_ms,
        raw_smoothed_decay_time,
        raw_smoothed_decay_im,
    )


def calculate_decay_cutoff_for_smoothed_decay(
    time_array: NpArray64,
    data_array: NpArray64,
    peak_idx: Int,
    decay_idx: Int,
    bl_im: NpArray64,
    ev_amplitude: NpArray64,
    smooth_window_samples: Int,
    min_cutoff_perc: NpArray64,
    max_cutoff_perc: NpArray64,
    interp: bool,
):
    """
    Calculate the decay percent time (e.g. 10-90) on
    raw data (i.e. not exponential) with optional smoothing
    and interpolation of decay period before calculation.

    INPUTS:
        time_array : full record time array
        data_array : full record data array
        peak_idx : index of event peak
        decay_idx : index of event decay point
        bl_im : data value at the baseline
        ev_amplitude : event amplitude
        smooth_window_samples : number of samples to smooth over the decay period
        min_cutoff_perc : minimum decay percent cutoff
        max_cutoff_perc : maximum decay percent cutoff
        interp : interpolate the decay period prior to decay time calculation

    RETURNS:
        see find_multiple_decay_sample_percentages()
    """
    (
        _,
        smoothed_decay_time,
        smoothed_decay_im,
        raw_smoothed_decay_time,
        raw_smoothed_decay_im,
    ) = get_decay_data_for_smoothed_decay(
        time_array,
        data_array,
        peak_idx,
        decay_idx,
        smooth_window_samples,
        interp,
    )

    return find_multiple_decay_sample_percentages(
        smoothed_decay_time, smoothed_decay_im, bl_im, ev_amplitude, 100 - min_cutoff_perc, 100 - max_cutoff_perc
    ) + (smoothed_decay_time, smoothed_decay_im)


def get_decay_data_for_smoothed_decay(
    time_array,
    data_array,
    peak_idx,
    decay_idx,
    smooth_window_samples,
    interp,
):
    """
    Process the raw decay period (i.e. not an exponential fit) by
    (optionally) smoothing and/or interpolating.

    See calculate_decay_cutoff_for_smoothed_decay for arguments.
    """
    peak_time = time_array[peak_idx]
    decay_data = data_array[peak_idx : decay_idx + 1]

    if smooth_window_samples > len(decay_data):
        smooth_window_samples = len(decay_data)
    elif smooth_window_samples == 0:
        smooth_window_samples = 1

    smoothed_decay_time = time_array[peak_idx : decay_idx + 1]
    smoothed_decay_im = quick_moving_average(decay_data, smooth_window_samples)

    raw_smoothed_decay_time = time_array[peak_idx : decay_idx + 1]  # have to re-init in case was interpolated
    raw_smoothed_decay_im = quick_moving_average(decay_data, smooth_window_samples)

    if interp:
        (
            smoothed_decay_im,
            smoothed_decay_time,
        ) = core_analysis_methods.twohundred_kHz_interpolate(smoothed_decay_im, smoothed_decay_time)

    return peak_time, smoothed_decay_time, smoothed_decay_im, raw_smoothed_decay_time, raw_smoothed_decay_im


def calclate_decay_to_point_from_exp_fit(
    decay_exp_fit_time: NpArray64,
    decay_exp_fit_im: NpArray64,
    peak_time: NpArray64,
    bl_im: NpArray64,
    ev_amplitude: NpArray64,
    amplitude_percent_cutoff: NpArray64,
    interp: bool,
) -> Tuple[NpArray64, ...]:
    """
    Find the nearest sample on the decay to the specified amplitude_percent_cutoff
    for the calculation of decay time. This uses the decay monoexp fit
    for increased temporal resolution

    INPUTS:
        decay_exp_fit_time - 1 x time array of timepoints between peak and decay end
        point
        decay_exp_fit_im - 1 x time array of datapoints (typically Im for events)
        between peak and decay end point
        ev_amplitude - Amplitude of the event (peak - baseline)
        amplitude_percent_cutoff - percentage of amplitude that the decay has
        returned to, 37% default
        interp - bool, to 200 kHz interp or not

        See find_event_peak_after_smoothing() for other inputs and
        calculate_decay_to_point_from_smoothed_decay()
        for outputs
    """
    if interp:
        (
            decay_exp_fit_im,
            decay_exp_fit_time,
        ) = core_analysis_methods.twohundred_kHz_interpolate(decay_exp_fit_im, decay_exp_fit_time)

    (decay_timepoint, decay_value, decay_time_ms,) = find_nearest_decay_sample_to_amplitude(
        decay_exp_fit_time,
        decay_exp_fit_im,
        peak_time,
        bl_im,
        ev_amplitude,
        amplitude_percent_cutoff,
    )

    return decay_timepoint, decay_value, decay_time_ms


def calculate_decay_cutoff_for_exp_fit(
    decay_exp_fit_time: NpArray64,
    decay_exp_fit_im: NpArray64,
    bl_im: NpArray64,
    ev_amplitude: NpArray64,
    min_cutoff_perc: NpArray64,
    max_cutoff_perc: NpArray64,
    interp: bool,
):
    """
    Calculate the single decay cutoff point on an exponential
    fit of the decay.

    INPUTS:
        decay_exp_fit_time : 1D numpy array of decay period time
        decay_exp_fit_im : 1D numpy array of decay period data
        bl_im : baseline value for the associated event
        ev_amplitude : event ampltiude
        min_cutoff_perc : minimum amplitude cutoff (%)
        max_cutoff_perc : maximum amplitude cutoff (%)
        interp : interpolate the fit before calculating decay time
    """
    if interp:
        (
            decay_exp_fit_im,
            decay_exp_fit_time,
        ) = core_analysis_methods.twohundred_kHz_interpolate(decay_exp_fit_im, decay_exp_fit_time)

    return find_multiple_decay_sample_percentages(
        decay_exp_fit_time, decay_exp_fit_im, bl_im, ev_amplitude, 100 - min_cutoff_perc, 100 - max_cutoff_perc
    )


def find_multiple_decay_sample_percentages(
    time_array: NpArray64,
    data_array: NpArray64,
    bl_im: NpArray64,
    ev_amplitude: NpArray64,
    amplitude_min: Union[Int, NpArray64],
    amplitude_max: Union[Int, NpArray64],
) -> Tuple[NpArray64, ...]:
    """
    Find the decay time by taking the difference in time between two points
    at a percentage of the amplitude decay, typically 10-90.

    INPUTS:
        time_array : 1D numpy array of decay period time
        data_array : 1D numpy array of decay period data
        bl_im : baseline value for the associated event
        ev_amplitude : event ampltiude
        amplitude_min : minimum amplitude cutoff (%)
        amplitude_max : maximum amplitude cutoff (%)

    RETURNS:
        min_time : time at the minimum cutoff
        min_data : data value at the minimum cutoff
        max_time : time at the max cutoff
        max_data : data value at the max cutoff
        decay_time : measured decay time, max_time - min_time
    """
    nearest_im_to_amp_idx_1 = get_closest_sample_to_percent_decay(data_array, amplitude_min, bl_im, ev_amplitude)
    nearest_im_to_amp_idx_2 = get_closest_sample_to_percent_decay(data_array, amplitude_max, bl_im, ev_amplitude)

    min_time = time_array[nearest_im_to_amp_idx_1]
    min_data = data_array[nearest_im_to_amp_idx_1]

    max_time = time_array[nearest_im_to_amp_idx_2]
    max_data = data_array[nearest_im_to_amp_idx_2]

    time_diff = max_time - min_time

    return min_time, min_data, max_time, max_data, time_diff


def find_nearest_decay_sample_to_amplitude(
    time_array: NpArray64,
    data_array: NpArray64,
    peak_time: NpArray64,
    bl_im: NpArray64,
    ev_amplitude: NpArray64,
    amplitude_percent_cutoff: NpArray64,
) -> Tuple[NpArray64, ...]:
    """
    For calculating the decay time, find the nearest datapoint to the decay %.

    e.g. if user has set decay % to 37%, we want to find the datapoint that is 37% of
    the decay amplitude. This might not be an exact sample so find the nearest.
    See calculate_decay_to_point_from_smoothed_decay for input / output.
    """
    nearest_im_to_amp_idx = get_closest_sample_to_percent_decay(
        data_array, amplitude_percent_cutoff, bl_im, ev_amplitude
    )
    decay_value = data_array[nearest_im_to_amp_idx]
    decay_timepoint = time_array[nearest_im_to_amp_idx]
    decay_time_ms = (decay_timepoint - peak_time) * 1000

    return decay_timepoint, decay_value, decay_time_ms


def get_closest_sample_to_percent_decay(
    data_array: NpArray64,
    amplitude_percent_cutoff: Union[Int, NpArray64],
    bl_im: NpArray64,
    ev_amplitude: NpArray64,
):
    """
    Given data (a 1d numy array) typically the rise or falling phase
    of an event, find the datapoint that is at the amplitude_percent_cutoff.
    """
    amplitude_fraction = amplitude_percent_cutoff / 100
    amplitude_fraction = bl_im + ev_amplitude * amplitude_fraction
    nearest_im_to_amp_idx = np.argmin(np.abs(data_array - amplitude_fraction))
    return nearest_im_to_amp_idx


def calculate_event_rise_time(
    time_array: NpArray64,
    data: NpArray64,
    bl_idx: Int,
    peak_idx: Int,
    bl_im: NpArray64,
    peak_im: NpArray64,
    min_cutoff_perc: Union[Int, NpArray64],
    max_cutoff_perc: Union[Int, NpArray64],
    interp: bool = False,
) -> Tuple[NpArray64, ...]:
    """
    Calculate the rise time of the event using core_analysis_methods
    (see these methods for input / outputs)
    """
    ev_data = data[bl_idx : peak_idx + 1]
    ev_time = time_array[bl_idx : peak_idx + 1]

    if interp:
        (
            ev_data,
            ev_time,
        ) = core_analysis_methods.twohundred_kHz_interpolate(ev_data, ev_time)

    ev_amplitude = peak_im - bl_im

    min_time, min_data, max_time, max_data, rise_time = find_multiple_decay_sample_percentages(  # TOOD: rename
        ev_time, ev_data, bl_im, ev_amplitude, min_cutoff_perc, max_cutoff_perc
    )

    return (min_time, min_data, max_time, max_data, rise_time)


# --------------------------------------------------------------------------------------
# Moving Average
# --------------------------------------------------------------------------------------


def quick_moving_average(x: NpArray64, n: Int) -> NpArray64:
    """
    Smooth using convolution with a box function, padding the start
    and end of array to handle edge effects.

    x: array to smooth
    n: samples to smooth
    """
    data = np.hstack([np.tile(x[0], n), x, np.tile(x[-1], n)])  # type: ignore
    out = np.convolve(data, np.ones(n) / n, mode="same")
    out = out[n:-n]
    return out


# --------------------------------------------------------------------------------------
# Convenience functions
# --------------------------------------------------------------------------------------


def is_at_least_zero(start_idx: Int) -> Int:
    if start_idx < 0:
        start_idx = 0
    return start_idx


def normalise_amplitude(data: NpArray64, demean: bool = False) -> NpArray64:
    """
    Scale to unity amplitude with optional demean.

    data : a 1 x num_samples or num_events by num_samples
          array
    demean : if `True`, data is demeaned row-wise.
    """
    assert data.ndim in [1, 2], "`data` must be 1 or 2 dimensional."

    amplitudes = np.abs(np.max(data, axis=-1) - np.min(data, axis=-1))
    if data.ndim == 2:
        amplitudes = amplitudes[:, np.newaxis]

    norm_curves = data / amplitudes

    if demean:
        means = np.mean(norm_curves, axis=-1)
        if data.ndim == 2:
            means = means[:, np.newaxis]
        norm_curves = norm_curves - means

    return norm_curves


def consts(constant_name: str) -> Int:
    """
    Constants for various kinetics calculation derived from testing
    across many event types

    constant_name -

        "bl_percentile" - percentile for slope length used in auto. baseline detection
        "weight_distance_from_peak" - weight on the automatic decay endpoint finder
                                      to increase noise by distance from peak.
                                      Rarely, the exp grow too large and undefined,
                                      throwing a numpy RunTimeWarning (rare)
        "decay_period_to_smooth" - smooth the decay period when auto-detected decay
                                   period end to avoid noise spikes biasing the result.
    """
    if constant_name == "bl_percentile":
        const = 60

    elif constant_name == "weight_distance_from_peak":
        const = 5

    elif constant_name == "decay_period_to_smooth":
        const = 3

    elif constant_name == "event_peak_smoothing":
        const = 3

    elif constant_name == "foot_detection_low_rise_percent":
        const = 20

    elif constant_name == "foot_detection_high_rise_percent":
        const = 80

    elif constant_name == "deconv_peak_search_region_multiplier":
        const = 3

    return const
