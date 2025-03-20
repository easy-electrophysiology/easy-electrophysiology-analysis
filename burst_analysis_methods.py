from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import numpy as np
import scipy.signal
import statsmodels.api as statsmodels
from ephys_data_methods import core_analysis_methods

if TYPE_CHECKING:
    from custom_types import FalseOrInt, Int, NpArray64
    from numpy.typing import NDArray

# --------------------------------------------------------------------------------------
# Detect Bursts
# --------------------------------------------------------------------------------------


def detect_burst_start_stop_index(peak_times_ms: NpArray64, burst_cfgs: Dict) -> Tuple[FalseOrInt, FalseOrInt]:
    """
    Entry function for burst detection. Returns the burst start / stop indexes of the
    peak_times_ms array. Uses the logISI or interval method

    Cotterril (2016). A comparison of computational methods for detecting bursts in
    neuronal spike trains and their application to human stem cell-derived neuronal
    networks. J Neurophysiol. 116(2), 306-321.

    INPUTS:
        peak_times_ms - 1 x N numpy array of peak times in ms

            burst_cfgs - dictionary of options.
            "detection_method" - "interval" or "log_isi" for detection method used.
            "min_spikes_per_burst" - for "log_isi" - the minimum number of spikes in
                                     a burst
            "max_short_isi_ms" - for "log_isi" - smaller ISI threshold value (maxISI1)
            "max_long_isi_ms" - for "log_isi" - larger ISI threshold maxlue (maxISI2)
            "interval_params_ms" - dictionary of options for interval method. see
                                   calculate_max_interval_method() for details

    OUTPUTS:
        burst_start_idx, burst_stop_idx - start and stop indexes of spikes in
        peak_times_ms. e.g. if spike times are
        1, 2, 3, 5, 10, 15, 20, 21, 22 and spikes 1,2,3 are in burst 1
        and 20, 21, 22 in burst 2, results will be
        burst_start_idx = [0, 6]
        burst_stop_idx = [2, 8]
    """
    isi_ms = np.diff(peak_times_ms)

    if burst_cfgs["detection_method"] == "log_isi":
        if burst_cfgs["max_long_isi_ms"] > burst_cfgs["max_short_isi_ms"]:
            burst_start_idx, burst_stop_idx = log_isi_burst_detection_long(
                peak_times_ms,
                isi_ms,
                burst_cfgs["min_spikes_per_burst"],
                burst_cfgs["max_short_isi_ms"],
                burst_cfgs["max_long_isi_ms"],
            )

        else:
            burst_start_idx, burst_stop_idx, __ = log_isi_burst_detection_short(
                isi_ms,
                burst_cfgs["min_spikes_per_burst"],
                burst_cfgs["max_long_isi_ms"],
            )

    elif burst_cfgs["detection_method"] == "interval":
        burst_start_idx, burst_stop_idx = calculate_max_interval_method(
            peak_times_ms, isi_ms, burst_cfgs["interval_params_ms"]
        )

    return burst_start_idx, burst_stop_idx


# --------------------------------------------------------------------------------------
# LogISI Threshold Method
# --------------------------------------------------------------------------------------


def log_isi_burst_detection_long(
    peak_times: NpArray64,
    isi: NpArray64,
    min_spikes_per_burst: Int,
    max_short_isi: np.float64,
    max_long_isi: np.float64,
) -> Tuple[FalseOrInt, FalseOrInt]:
    """
    Detect bursts using the logISI method as described in:

    Pasquale V, Martinoia S, Chiappalone M. A self-adapting approach for the
    detection of bursts and network bursts in neuronal cultures.
    J Comput Neurosci 29: 213–229, 2010

    INPUTS and OUTPUTS:
        see detect_burst_start_stop_index()
    """
    short_rise_idx, short_fall_idx, __ = log_isi_burst_detection_short(isi, min_spikes_per_burst, max_short_isi)

    if short_rise_idx is False or short_fall_idx is False:
        return False, False

    # join bursts smaller than max_long_isi (here we are indexing peak_times)
    idx_to_join = np.where(peak_times[short_rise_idx[1:]] - peak_times[short_fall_idx[0:-1]] <= max_long_isi)[0]
    short_rise_idx = np.delete(short_rise_idx, idx_to_join + 1)
    short_fall_idx = np.delete(short_fall_idx, idx_to_join)

    # find bursts with the longer ISI
    long_rise_idx, long_fall_idx, __ = log_isi_burst_detection_short(isi, min_spikes_per_burst, max_long_isi)

    if long_rise_idx is False or long_fall_idx is False:
        return False, False

    # Create and sort the all edge matrix
    # fmt: off
    all_edge_sort = np.vstack([

        np.hstack([peak_times[short_rise_idx],             peak_times[short_fall_idx],              peak_times[long_rise_idx],          peak_times[long_fall_idx]]),

        np.hstack([np.ones(short_rise_idx.size) * 2,       np.ones(short_fall_idx.size) * 3,        np.ones(long_rise_idx.size),       np.ones(long_fall_idx.size) * 4]),

        np.hstack([short_rise_idx,                         short_fall_idx,                          long_rise_idx,                      long_fall_idx])


    ])
    # fmt: on

    # first sort by the times, then when for times that are identical sort by row idx
    # 1 (i.e. keys above)
    sort_idx = np.lexsort((all_edge_sort[1, :], all_edge_sort[0, :]))
    all_edge_sort = all_edge_sort[:, sort_idx]

    # Find outer burst start and begin cycling through them
    all_outer_burst_begin_idx = np.where(all_edge_sort[1, :] == 1)[0]

    burst_train = []

    # now we are indexing all_edge_sort (which itself contains peak times / indexes)
    for outer_begin_idx in all_outer_burst_begin_idx:
        if all_edge_sort[1, outer_begin_idx + 1] == 4:
            continue

        outer_end_idx = np.min(np.where(all_edge_sort[1, outer_begin_idx:] == 4)) + outer_begin_idx

        # for each burst, look at the sub-bursts and split if more than one sub-burst
        # found.
        # Save the start and end peak idx for each burst. If only one sub-burst,
        # the burst is just the entire burst (outer burst and sub burst)
        all_sub_burst_idx = np.arange(outer_begin_idx + 1, outer_end_idx)
        all_sub_burst_start_idx = np.where(all_edge_sort[1, all_sub_burst_idx] == 2)[0] + all_sub_burst_idx[0]

        if all_sub_burst_start_idx.size > 1:
            sub_burst_begin = np.hstack([outer_begin_idx, all_sub_burst_start_idx[1:]])

            sub_burst_end = np.hstack([all_sub_burst_start_idx[1:], outer_end_idx])

            for i in range(sub_burst_begin.size):
                if i == sub_burst_begin.size - 1:
                    burst_start_end = [
                        all_edge_sort[2, sub_burst_begin[i]],
                        all_edge_sort[2, sub_burst_end[i]],
                    ]
                else:
                    # For sub bursts, finish each sub-burst 1 spike before the start
                    # of the next sub-burst. This has to do be done here due to the
                    # nature of indexing AllEdgeSort
                    burst_start_end = [
                        all_edge_sort[2, sub_burst_begin[i]],
                        all_edge_sort[2, sub_burst_end[i]] - 1,
                    ]

                burst_train.append(burst_start_end)

        else:
            burst_start_end = [
                all_edge_sort[2, outer_begin_idx],
                all_edge_sort[2, outer_end_idx],
            ]
            burst_train.append(burst_start_end)

    burst_train = np.array(burst_train, dtype="int")

    return burst_train[:, 0], burst_train[:, 1]


def log_isi_burst_detection_short(
    isi: NpArray64, min_spikes_per_burst: Int, max_isi: np.float64
) -> Tuple[FalseOrInt, FalseOrInt, FalseOrInt]:
    """
    Find the idx of rising (burst start) and falling (burst finish)
    edges for log ISI burst detection.
    """
    rising_idx, falling_idx = find_burst_edges(isi, max_isi)

    num_spikes = falling_idx - rising_idx + 1

    bursts_with_enough_spikes = num_spikes >= min_spikes_per_burst

    num_spikes = num_spikes[bursts_with_enough_spikes]
    rising_idx = rising_idx[bursts_with_enough_spikes]
    falling_idx = falling_idx[bursts_with_enough_spikes]

    if rising_idx.size == 0:
        return False, False, False

    return rising_idx, falling_idx, num_spikes


def find_burst_edges(isi: NpArray64, max_short_isi: np.float64) -> Tuple[NDArray[np.integer], NDArray[np.integer]]:
    """
    Find rising and falling burst edges (i.e. burst start / stop times)
    for bursts containing ISI shorter than max_short_isi.
    """
    bin_less_than_isi = np.hstack([0, isi <= max_short_isi, 0])
    rising_and_falling = np.diff(bin_less_than_isi)

    rising_idx = np.where(rising_and_falling == 1)[0]
    falling_idx = np.where(rising_and_falling == -1)[0]

    return rising_idx, falling_idx


# --------------------------------------------------------------------------------------
# Calculate LogISI Threshold from Log ISI Histogram
# --------------------------------------------------------------------------------------


def calculate_log_threshold(isi: NpArray64, burst_cfgs: Dict) -> Tuple[Union[str, np.float64], Dict]:
    """
    Calculate the threshold ISI which separates intra-burst ISI.

    In a timeseries with bursting, the inter-spike interval (ISI) can be
    separated into shorter intra-burst ISI and the ISI expected under normal
    conditions (e.g. Poisson distributed spikes). Taking the histogram of the
    logarithm of the ISI (note ISI for poisson distributed spikes is exponential)
    gives peaks at the intra-burst interval, and the expected larger peak.

    Using the algorithm described in Pasquale et al (2009) these peaks can be
    optimally separated and the ISI which best separates them is used as the
    thershold for burst detection.

    INPUTS:
    isi: 1 x N array of inter-spike (or inter-event for events analysis) intervals.

    burst_cfgs: dictionary containing options -
                min_void_parameter - minimum void parameter accepted for
                                     distinguishing two peaks
                bins_per_decade  - bins per decade for the logISI histogram bins
                histogram_smoothing - use lowess smoothing on the histogram (bool)
                lowess_fraction - fraction of total number of samples used for
                                  the local regression
                intraburst_peak_cutoff_ms - threshold used to determine
                                            region of histogram that contains
                                            intra-burst peak
                min_samples_between_peaks  - min samples between peaks used
                                             for peak detection
    OUTPUTS:
        max_long_isi - max ISI threshold used as maxISI2, or ISIth
                       (see Pasqule et al., 2009).

        if error will return a str in place of max_long_isi -
            "no_valid_minimum_error"
            "no_other_peak_error"

    Cotterril (2016). A comparison of computational methods for detecting bursts in
    neuronal spike trains and their application to human stem cell-derived neuronal
    networks. J Neurophysiol. 116(2), 306-321.

    Notes
    -----
    The variable info_for_plotting must be updated after each function as if an
    error occurs the plot is update with parameters calculated thus far.
    """
    info_for_plotting = {
        "bins": np.array([]),
        "count_prob": np.array([]),
        "min_minimum_index": None,
        "intra_burst_peak_idx": None,
        "larger_isi_peak_indexes": None,
        "void_parameters": np.array([]),
    }

    bins, count_prob = calculate_logisi_histogram(isi, burst_cfgs)

    info_for_plotting["bins"] = bins
    info_for_plotting["count_prob"] = count_prob

    # find intra-burst peak and other peaks
    intra_burst_peak_idx, larger_isi_peak_indexes = get_log_isi_hist_peaks(burst_cfgs, bins, count_prob)

    info_for_plotting["intra_burst_peak_idx"] = intra_burst_peak_idx

    if not np.any(larger_isi_peak_indexes):
        return "no_other_peak_error", info_for_plotting

    info_for_plotting["larger_isi_peak_indexes"] = larger_isi_peak_indexes

    # calculate max_long_isi using the void parameter
    (minimum_indexes, info_for_plotting["void_parameters"],) = get_minimums_with_valid_void_parameter(
        count_prob,
        intra_burst_peak_idx,
        larger_isi_peak_indexes,
        burst_cfgs["min_void_parameter"],
    )

    if not np.any(minimum_indexes):
        return "no_valid_minimum_error", info_for_plotting

    min_minimum_index = min(minimum_indexes)

    # take the lowest idx void param > min void param
    max_long_isi_log = bins[min_minimum_index]
    max_long_isi = 10**max_long_isi_log

    info_for_plotting["min_minimum_index"] = min_minimum_index

    return max_long_isi, info_for_plotting  # type: ignore


def calculate_logisi_histogram(isi: NpArray64, burst_cfgs: Dict) -> Tuple[NpArray64, NpArray64]:
    """
    Calculate the histogram of log10 ISI with centered bins.
    Histogram frequencies are converted to probabilities.
    Optional smoothing of the histogram with LOWESS smoothing.

    By default, the histograms will be calculated between the minimum and maximum isi.

    see calculate_log_threshold() for inputs
    """
    log_isi = np.log10(isi)

    if burst_cfgs["bin_override"] is False:
        bins = np.round((np.max(log_isi) - np.min(log_isi)) * burst_cfgs["bins_per_decade"]).astype(int)
    else:
        bins = burst_cfgs["bin_override"]

    count, bin_edges = np.histogram(log_isi, bins=bins)

    format_bins = core_analysis_methods.format_bin_edges(bin_edges, burst_cfgs["bin_edge_method"])

    count_prob = count / np.sum(count)

    if burst_cfgs["histogram_smoothing"]:
        # default to span = 5 points similar to matlab smooth function
        lowess_frac = burst_cfgs["lowess_fraction"]
        frac = lowess_frac if len(count_prob) * lowess_frac > 5 else 5 / len(count_prob)
        count_prob = statsmodels.nonparametric.lowess(
            exog=format_bins, endog=count_prob, return_sorted=False, frac=frac
        )

    return format_bins, count_prob


def get_log_isi_hist_peaks(burst_cfgs: Dict, bins: NpArray64, count_prob: NpArray64) -> Tuple[Int, NDArray[np.integer]]:
    """
    Find intra-burst peak and other peaks.

    see calculate_log_threshold() for inputs
    """
    log_intraburst_cutoff = np.log10(burst_cfgs["intraburst_peak_cutoff_ms"])
    intra_burst_peak_idx = np.argmax(count_prob * (bins <= log_intraburst_cutoff))

    other_peak_indexes = scipy.signal.find_peaks(count_prob, distance=burst_cfgs["min_samples_between_peaks"])[0]
    larger_isi_peak_indexes = other_peak_indexes[other_peak_indexes > intra_burst_peak_idx]

    return intra_burst_peak_idx, larger_isi_peak_indexes


def get_minimums_with_valid_void_parameter(
    g: NpArray64,
    intra_burst_peak_idx: Int,
    larger_isi_peak_indexes: NDArray[np.integer],
    min_void_parameter: np.float64,
) -> Tuple[List, List]:
    """
    g is the distribution g(x) of x = log(ISI) (i.e. the count_prob from
    calculate_logisi_histogram(), g(x) chosen to match paper).

    The void parameter is the ratio of the height of the minimum between
    two peaks to the geometric average of the two peaks. If the minimum
    is zero, the peaks are perfectly separated (void = 0). If void = 0.5,
    the minimum between the two peaks is half the geometric average of the
    height of the two peaks.

    The void parameter is calculated between the intra-burst peak and every other
    peak in the histogram.

    see calculate_log_threshold() for inputs
    """
    void_parameters = []
    minimum_indexes = []

    for peak_idx in larger_isi_peak_indexes:
        min_idx = np.argmin(g[intra_burst_peak_idx:peak_idx]) + intra_burst_peak_idx

        void_parameter = 1 - g[min_idx] / (np.sqrt(g[intra_burst_peak_idx] * g[peak_idx]))

        if void_parameter <= min_void_parameter:
            continue

        void_parameters.append(void_parameter)
        minimum_indexes.append(min_idx)

    return minimum_indexes, void_parameters


# --------------------------------------------------------------------------------------
# Max Interval Method
# --------------------------------------------------------------------------------------


def calculate_max_interval_method(peak_times: NpArray64, isi: NpArray64, params: Dict) -> Tuple[FalseOrInt, FalseOrInt]:
    """
    Max interval method, uses the below parameters to detect bursts:

    max_interval            - maximum ISI interval for spikes in a burst
    max_end_interval        - maximum ISI interval for the last spike of a burst
    min_spikes_per_burst    - minimum number of spikes in a burst
    min_burst_duration
    min_burst_interval

    As described in: Cotterril (2016). A comparison of computational methods for
    detecting bursts in neuronal spike trains and their  application to human
    stem cell-derived neuronal networks. J Neurophysiol. 116(2), 306-321.
    """
    all_burst_peak_idx = []
    spike_i = 0

    while True:
        if spike_i >= len(isi):
            break

        cur_isi = isi[spike_i]

        if cur_isi <= params["max_interval"]:
            # if the isi is <= than max interval, take it as the first
            # spike of a burst. Below we continue to cycle
            # through isi, if any > max end interval end the burst.

            burst_peak_idx = [spike_i]

            while True:  # cycle through isi looking for end of burst
                spike_i += 1

                if spike_i == len(isi):
                    break

                if isi[spike_i] >= params["max_end_interval"]:
                    break

            burst_peak_idx.append(spike_i)

            (
                burst_start_time,
                burst_end_time,
                num_spikes_in_burst,
            ) = get_burst_times_and_peak_num(peak_times, burst_peak_idx)

            # If burst is longer than one peak, and min burst duration,
            # add to all_burst_peak_idx by updating or merging
            if (
                num_spikes_in_burst >= params["min_spikes_per_burst"]
                and burst_end_time - burst_start_time >= params["min_burst_duration"]
            ):
                # Check previous burst,
                if (
                    len(all_burst_peak_idx) >= 1
                    and burst_start_time - peak_times[all_burst_peak_idx[-1][-1]] <= params["min_burst_interval"]
                ):
                    all_burst_peak_idx[-1][-1] = burst_peak_idx[-1]
                else:
                    all_burst_peak_idx.append(burst_peak_idx)

        spike_i += 1

    if len(all_burst_peak_idx) == 0:
        return False, False

    all_idx = np.array(all_burst_peak_idx)
    start_idx = all_idx[:, 0]
    stop_idx = all_idx[:, 1]

    return start_idx, stop_idx


# --------------------------------------------------------------------------------------
# Calculate Burst Parameters
# --------------------------------------------------------------------------------------


def get_burst_times_and_peak_num(
    peak_times: NpArray64, burst_peak_idx: List[int]
) -> Tuple[np.float64, np.float64, Int]:
    """
    Get the start / stop times, and num spikes in burst, from peak times
    and a list of indexes that index a burst.
    """
    burst_start_time = peak_times[burst_peak_idx[0]]
    burst_end_time = peak_times[burst_peak_idx[-1]]
    num_spikes_in_burst = burst_peak_idx[-1] - burst_peak_idx[0] + 1

    return burst_start_time, burst_end_time, num_spikes_in_burst


def calculate_burst_parameters(
    peak_times: NpArray64,
    burst_start_idx: NDArray[np.integer],
    burst_stop_idx: NDArray[np.integer],
) -> Tuple[NpArray64, NDArray[np.integer], NpArray64, np.float64, List, List, List]:
    """
    From the peak times (s), and burst start / stop indexes calculate burst parameters:

    burst_lengths_ms                 - length of each burst in ms
    num_spikes_in_burst              - number of spikes in each burst
    inter_burst_intervals            - interval between bursts (i.e. difference
                                       between burst end and next burst start) in ms
    rec_fraction_of_spikes_in_burst  - percent of all spikes that are in a burst
                                       (these parameters are calculated per record)

    all_burst_idx                    - index of every spike in burst (indexes peak_
                                       times) (list of lists, sublist contains
                                       burst indexes)
    all_burst_peak_times             - time of every peak for every spike in each
                                       burst (list of lists, sublist contains
                                       burst peak times)
    all_intra_burst_ISI              - average ISI of spikes within each burst (list
                                       of scalar)
    """
    peak_times_ms = peak_times * 1000

    burst_lengths_ms = peak_times_ms[burst_stop_idx] - peak_times_ms[burst_start_idx]
    num_spikes_in_burst = burst_stop_idx - burst_start_idx + 1
    inter_burst_intervals = peak_times_ms[burst_start_idx[1:]] - peak_times_ms[burst_stop_idx[:-1]]
    rec_fraction_of_spikes_in_burst = np.sum(num_spikes_in_burst) / peak_times_ms.size

    all_burst_idx = [np.arange(start_idx, stop_idx + 1) for start_idx, stop_idx in zip(burst_start_idx, burst_stop_idx)]
    all_burst_peak_times = [peak_times[indexes] for indexes in all_burst_idx]

    # burst peak times must be in s later, so this awkward conversion needed
    all_intra_burst_ISI = [np.mean(np.diff(bust_peak_times * 1000)) for bust_peak_times in all_burst_peak_times]

    return (
        burst_lengths_ms,
        num_spikes_in_burst,
        inter_burst_intervals,
        rec_fraction_of_spikes_in_burst,
        all_burst_idx,
        all_burst_peak_times,
        all_intra_burst_ISI,
    )
