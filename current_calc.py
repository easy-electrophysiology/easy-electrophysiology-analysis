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

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import scipy
from ephys_data_methods import core_analysis_methods
from utils import utils

if TYPE_CHECKING:
    import pandas as pd
    from configs.configs import ConfigsClass
    from custom_types import FalseOrFloat, Info, Int, NpArray64
    from importdata import RawData
    from numpy.typing import NDArray

# --------------------------------------------------------------------------------------
# Spike Counting Methods - Automatically Detect Spikes
# --------------------------------------------------------------------------------------


def auto_find_spikes(
    data: RawData,
    thr: Dict,
    rec_from: Int,
    rec_to: Int,
    time_bounds: Union[Literal[False], List[List]],
    bound_start_or_stop: Union[Literal[False], List[str]],
) -> Info:
    """
    Calculate spike threshold per record based on options specified in
    configurations. Returns list of dicts (see find_spikes_above_record_threshold()).

    There are two methods to find spikes. One is a per-spike method based on
    time bounds and derivative thresholds. The second is to take any part of
    the time series that rises and falls over a pre-defined threshold and call
    it a spike.

    Here, spikes_from_auto_threshold_per_record will first use the per-spike
    method to find all possible spikes based on the thresholds. Then it will
    average the half-amplitudes and use the vm as a pre-defined threshold,
    anything over which is called a spike. This is less refined than the
    per-spike method alone but will not throw up any random spikes.

    Alternatively, the spikes_from_auto_threshold_per_spike method just returns the
    results of the per-spike method.


    INPUT: data:               object containing vm to analyse
           thr:                a dictionary of various thresholding values for
                               automatic spike count detection (see configs)
               thr["auto_thr_amp"] - minimum amplitude of spikes to be used in
                                     threshold calculation
               thr["auto_thr_rise"] - threshold of the first derivative in the
                                      positive direction for spikes to be used in
                                      threshold calculation in ms
               thr["auto_thr_fall"] - threshold of the first derivative for spikes
                                      the negative direction spikes to be used in
                                      threshold calculation in ms
               thr["auto_thr_width"] - region to search for first deriv pass negative
                                       threshold after passing positive threshold in ms.
           rec_from:     rec to analyse from
           rec_to:       rec to analyse to
           time_bounds:  boundary to analyse within, specified in seconds
           bounds_start_or_stop: whether "time_bounds" bounds are start or stop (e.g.
           ["start", "stop"])
    """
    upper_inclusive_rec_bound = rec_to + 1
    search_region_in_s = thr["auto_thr_width"] / 1000
    thr["N"] = core_analysis_methods.quick_get_time_in_samples(data.ts, search_region_in_s)

    # Find Spike Threshold per Record
    if thr["threshold_type"] == "auto_record":
        spike_info = spikes_from_auto_threshold_per_record(
            data, rec_from, rec_to, thr, time_bounds, bound_start_or_stop
        )
    if thr["threshold_type"] == "auto_spike":
        spike_info = spikes_from_auto_threshold_per_spike(
            data,
            rec_from,
            upper_inclusive_rec_bound,
            thr,
            time_bounds,
            bound_start_or_stop,
        )

    return spike_info


# Auto - thresholding
# --------------------------------------------------------------------------------------


def spikes_from_auto_threshold_per_record(
    data: RawData,
    rec_from: Int,
    rec_to: Int,
    thr: Dict,
    time_bounds: Union[Literal[False], List[List]],
    bound_start_or_stop: Union[Literal[False], List[str]],
) -> Info:
    """
    For every spike in the record find Vm that is 50% spike peak. Average
    across these to create a "threshold" for this record, only use spikes
    that are above the user-specified spike amplitude threshold.

    thr: dictionary containing information about AP threshold
         "N": thr["auto_thr_width"] in samples

    See auto_find_spikes() for other inputs
    """
    upper_inclusive_rec_bound = rec_to + 1

    thresholds = utils.np_empty_nan(data.num_recs)
    for rec in range(rec_from, upper_inclusive_rec_bound):
        vm = data.vm_array[rec]
        vm_diff = data.norm_first_deriv_vm[rec]

        rec_time_bounds = [time_bounds[0][rec], time_bounds[1][rec]] if time_bounds is not False else False
        (
            time_bound_start_sample,
            time_bound_stop_sample,
        ) = get_bound_times_in_sample_units(rec_time_bounds, bound_start_or_stop, data, rec)
        candidate_spikes_idx = find_candidate_spikes(vm_diff, thr, time_bound_start_sample, time_bound_stop_sample)

        if candidate_spikes_idx.any():
            (peak_vms, thr_vms, __, __,) = clean_and_amplitude_thr_candidate_spikes_and_extract_parameters(
                vm, candidate_spikes_idx, thr, rec_time_array=data.time_array[rec]
            )
            thresholds[rec] = (np.median(peak_vms) + np.median(thr_vms)) / 2
        else:
            thresholds[rec] = np.nan

    # If no spike in rec, uses the average of all thresholds as threshold
    if not np.all(np.isnan(thresholds)):
        mean_thr = np.nanmean(thresholds[rec_from:upper_inclusive_rec_bound])
        thresholds[np.isnan(thresholds)] = mean_thr

    spike_info = find_spikes_above_record_threshold(
        data, thresholds, rec_from, rec_to, time_bounds, bound_start_or_stop
    )

    return spike_info


def spikes_from_auto_threshold_per_spike(
    data: RawData,
    rec_from: Int,
    upper_inclusive_rec_bound: Int,
    thr: Dict,
    time_bounds: Union[Literal[False], List[List]],
    bound_start_or_stop: Union[Literal[False], List[str]],
) -> Info:
    """
    Find APs from auto-thresholding without further robust thresholding based on
    detected APs.

    see auto_find_spikes() for inputs
    """
    spike_info: Info
    spike_info = [[] for __ in range(data.num_recs)]

    for rec in range(rec_from, upper_inclusive_rec_bound):
        vm = data.vm_array[rec]
        vm_diff = data.norm_first_deriv_vm[rec]

        rec_time_bounds = [time_bounds[0][rec], time_bounds[1][rec]] if time_bounds is not False else False
        (
            time_bound_start_sample,
            time_bound_stop_sample,
        ) = get_bound_times_in_sample_units(rec_time_bounds, bound_start_or_stop, data, rec)
        candidate_spikes_idx = find_candidate_spikes(vm_diff, thr, time_bound_start_sample, time_bound_stop_sample)

        if candidate_spikes_idx.any():
            (peak_vms, __, peak_idxs, peak_times,) = clean_and_amplitude_thr_candidate_spikes_and_extract_parameters(
                vm, candidate_spikes_idx, thr, rec_time_array=data.time_array[rec]
            )

            if peak_vms.any():
                spike_info[rec] = {}
                for peak_vm, peak_time, peak_sample in zip(peak_vms, peak_times, peak_idxs):
                    dict_key = str(peak_time)
                    spike_info[rec][dict_key] = [peak_vm, peak_sample]  # type: ignore
            else:
                spike_info[rec] = 0
        else:
            spike_info[rec] = 0

    return spike_info


# Search for within-threshold spikes
# --------------------------------------------------------------------------------------


def find_candidate_spikes(
    vm_diff: NpArray64,
    thr: Dict,
    time_bound_start_sample: Int,
    time_bound_stop_sample: Int,
) -> NpArray64:
    """
    Find candidate spikes based on user-specified thresholds, fully vectorised for
    speed.

    INPUTS:
        vm_diff: first derivative per unit time,
                 see spikes_from_auto_threshold_per_record() for thr

    For each record take the 1st derivative of the trace and find a timeperiod length
    N which contains an increase in diff above the positive threshold followed by a
    decreased below the negative threshold

    Where ix is a sample over the differential threshold, finds all instances where
    within the range x : x + N there is  also a point under diff threshold.

    Deletes instances where x increases by 1 i.e.
    x:x+N, x+1:x+1, x+2:x+N because these
    are repeat instances of the same spike.
    """
    samples_above = np.where(vm_diff > thr["auto_thr_rise"])[0].squeeze()
    samples_above = samples_above[
        samples_above >= int(time_bound_start_sample + thr["N"])
    ]  # only consider points within boundary
    samples_above = samples_above[samples_above < int(time_bound_stop_sample - thr["N"])]

    samples_below = np.where(vm_diff < thr["auto_thr_fall"])[0].squeeze()
    samples_below = samples_below[samples_below >= int(time_bound_start_sample + thr["N"])]
    samples_below = samples_below[samples_below < int(time_bound_stop_sample - thr["N"])]

    # make a sample x N array for every sample above thr
    n_samples_above = utils.np_empty_nan((len(samples_above), thr["N"]))
    n_samples_above[:, :] = np.linspace(samples_above, samples_above + thr["N"] - 1, thr["N"]).transpose()
    n_samples_above.astype(int)

    # for the indicies sample_above + N, check if any contains a below thr instance
    idx = np.isin(n_samples_above, samples_below).any(axis=1)
    candidate_spikes_idx = n_samples_above[idx]

    return candidate_spikes_idx


def clean_and_amplitude_thr_candidate_spikes_and_extract_parameters(
    vm: NpArray64,
    candidate_spikes_idx: NpArray64,
    thr: Dict,
    rec_time_array: NpArray64,
) -> Tuple[NpArray64, ...]:
    """
    Based on matrix (spike x N) of indices, index out vm leaving a
    matrix of spike values in vm units, fully vectorised.

    A problem with find_candidate_spikes() method is that because is it
    single-index based it will return multiple candidate spikes from the
    same spike. e.g. threshold - end, threshold + 1 - end.

    Here we must delete all of these repeats from the same spike by
    deleting candidate spikes with cumulatively increasing first
    indices.

    Amplitude thresholding is also conducted here.
    """
    # clean repeats which are from the same spike
    clean_repeats = np.diff(candidate_spikes_idx[:, 0], prepend=0)
    del_ = np.where(clean_repeats == 1)
    candidate_spikes_idx = np.delete(candidate_spikes_idx, del_, axis=0).astype(int)

    # find spikes above the minimum spike amplitude threshold
    spike_vm_values = vm[candidate_spikes_idx]
    peak_idx_all = np.argmax(spike_vm_values, axis=1).squeeze()
    peak_vm_all = np.max(spike_vm_values, axis=1).squeeze()

    # idx in units of raw data,
    peak_idxs_all = candidate_spikes_idx[np.arange(np.size(peak_idx_all)), peak_idx_all].squeeze()
    thr_idx_all = candidate_spikes_idx[:, 0]
    thr_vm_all = vm[thr_idx_all].squeeze()

    spike_amplitudes_all = (peak_vm_all - thr_vm_all).squeeze()
    spikes_above_thr = spike_amplitudes_all > np.array(thr["auto_thr_amp"])

    peak_vms = peak_vm_all[spikes_above_thr]
    thr_vms = thr_vm_all[spikes_above_thr]
    peak_idxs = peak_idxs_all[spikes_above_thr]
    peak_times = rec_time_array[peak_idxs]

    return peak_vms, thr_vms, peak_idxs, peak_times


# Find Spikes Above Set Threshold
# --------------------------------------------------------------------------------------


def find_spikes_above_record_threshold(
    data: RawData,
    thresholds: NpArray64,
    rec_from: Int,
    rec_to: Int,
    time_bounds: Union[Literal[False], List[List]],
    bound_start_or_stop: Union[Literal[False], List[str]],
) -> Info:
    """
    Finds the time, peak and idx of all spikes above a thresholds for each record in
    the file. Returns as a list of dicts "spike_info" containing
    information about each spike.

    Cuts the trace down in-between bounds to optimise speed.

    NOTE: one current issue that a spike may be counted if the decay is sliced in two.

    INPUT: data:         data object class
           thresholds:    thresholds over which to measure spikes. Can be single
                         thresholds (applied to all records), or a
                         array of thresholds (per record)
           rec_from:     rec to analyse from
           rec_to:       rec to analyse to
           time_bounds:  boundary to analyse within, specified in seconds

    OUTPUT: spike_info:  A list of records with entry [] for rec not analysed,
                         0 for analysed and no spikes, or a dictionary with
                         the spike time (S) as key, containing a list [spike_peak_vm,
                         spike_idx].
    """
    upper_inclusive_rec_bound = rec_to + 1

    spike_info: Info
    spike_info = [[] for rec in range(data.num_recs)]

    # cut vm and time down to time period to be analysed

    all_time_bound_start_sample = utils.np_empty_nan(data.num_recs)

    bound_vm_array = [np.array([]) for __ in range(data.num_recs)]
    bound_time_array = [np.array([]) for __ in range(data.num_recs)]
    for rec in range(data.num_recs):
        rec_time_bounds = [time_bounds[0][rec], time_bounds[1][rec]] if time_bounds is not False else False
        (
            time_bound_start_sample,
            time_bound_stop_sample,
        ) = get_bound_times_in_sample_units(rec_time_bounds, bound_start_or_stop, data, rec)
        all_time_bound_start_sample[rec] = time_bound_start_sample

        bound_vm_array[rec] = data.vm_array[rec, time_bound_start_sample : time_bound_stop_sample + 1]
        bound_time_array[rec] = data.time_array[rec, time_bound_start_sample : time_bound_stop_sample + 1]

    # Count Spikes
    # get vector of vm indices above thresholds (with length either 1 when thresholds
    # manual set or len(recs) when thresholds autoset), otherwise nan
    if np.size(thresholds) == 1:
        thresholds = np.tile(thresholds, (data.num_recs, 1))

    above_threshold_vm_matrix = [[] for __ in range(data.num_recs)]
    for rec in range(rec_from, upper_inclusive_rec_bound):
        above_threshold_vm_matrix[rec] = np.ma.masked_array(
            bound_vm_array[rec],
            np.invert(bound_vm_array[rec] > thresholds[rec]),
            fill_value=np.nan,
        ).filled()

    # Find contiguous above-threshold regions and get peak information of these regions
    for rec in range(rec_from, upper_inclusive_rec_bound):
        cum_ap_index = index_out_continuous_above_threshold_samples(above_threshold_vm_matrix[rec])

        # get indexed vm and time of > thr vm
        peaks_idx = get_peaks_idx_from_cum_idx(cum_ap_index, bound_vm_array[rec], event_dir="positive")

        # Using peak indices, save the vm peak, its time and idx
        # into a dict where key is the time of spike
        if np.any(peaks_idx):
            spike_info[rec] = {}
            for peak_idx, peak_time, peak_vm in zip(
                peaks_idx,
                bound_time_array[rec][peaks_idx],
                bound_vm_array[rec][peaks_idx],
            ):
                spike_info[rec][str(peak_time)] = [  # type: ignore
                    peak_vm,
                    peak_idx + all_time_bound_start_sample[rec].astype(int),
                ]  # add back the first bound
        else:
            spike_info[rec] = 0

    return spike_info


def index_out_continuous_above_threshold_samples(
    binary_ts: Union[List[NpArray64], NpArray64], smooth: Optional[Int] = None
) -> NpArray64:
    """
    Create an vector of length = number of samples where each spike is batched to a
    cumulative index i.e [00011100022200033],
    0 when data < thr or cumsum > 1...N spikes are above threshold

    INPUTS:
    binary_ts (binary timeseries) : 1 x N samples of zero or 1, where period
                                    of 1s represent a continuous event of
                                    which we are interested in the peak
                                    (e.g. a spike, event)

    smooth : smooth across the binary vector with window length ( FALSE OR INT)
             useful for events rather than spikes, where the
             odd sample may not be above threshold e.g.
             [0 0 0 (event starts) 1 1 1 1 1 1 1 0 0 1 1 1 1 1 (
             event ends) 0 0 0 0 0 ]
    """
    ii = np.zeros((len(binary_ts), 1))
    jj = np.invert(np.isnan(binary_ts))
    jj = jj.astype(int)

    # smooth any ones by 1 windows width in case of gaps
    if isinstance(smooth, int):
        W = smooth
        box_function = np.concatenate([[0, 0], np.ones(W), [0, 0]])
        jj = np.convolve(jj, box_function)
        jj[jj > 0] = 1
        jj = jj[0 : len(binary_ts)]

    jj = np.hstack((0, jj))  # match diff and ii length
    diffs = np.ediff1d(jj)
    jj = np.delete(jj, 0)
    start_indx = np.where(diffs == 1)
    ii[start_indx] = 1
    cum_index = np.cumsum(ii) * jj

    return cum_index


def get_peaks_idx_from_cum_idx(
    cum_event_index: NpArray64,
    data_vector: NpArray64,
    event_dir: Literal["positive", "negative"],
) -> NDArray[np.integer]:
    """
    from a 1 x n array of indices corresponding to contiguous events
    (e.g. [ 0 0 1 1 1 0 0 2 2 2 ] index out the idx, time and vm/vm
    for each event into a list. Use for spikes and events.

    cum_event_index: 1D array of cumulatively increasing indices separated by zeros
                     where each contiguous set of indices indicate an above threshold
                     event
    data_vector: List (per rec) of 1D array of Im or Vm data (for event or spike count respectively)
    event_dir: "positive" or "negative" (i.e. AP, GABA events and glu events
    respectively)

    use sparse matrix for speed increase
    """
    cum_event_index_sparse = fast_indexer(cum_event_index.astype(int))

    idx_ = [row.data for row in cum_event_index_sparse]
    idx_.pop(0)
    if event_dir == "positive":
        peaks_idx = np.array([idx[0] + np.argmax(data_vector[idx]) for idx in idx_])
    elif event_dir == "negative":
        peaks_idx = np.array([idx[0] + np.argmin(data_vector[idx]) for idx in idx_])
    peaks_idx = np.array(peaks_idx).astype(int)

    return peaks_idx


def fast_indexer(array: NpArray64) -> NpArray64:
    """
    Use spare matrix for indexing an array of mostly zeros for speed.
    """
    col_num = np.arange(array.size)
    return scipy.sparse.csr_matrix((col_num, (array.ravel(), col_num)), shape=(array.max() + 1, array.size))


# --------------------------------------------------------------------------------------
# Spike Parameter Methods
# --------------------------------------------------------------------------------------


def spkcount_and_recnums_from_spikeinfo(
    spike_info: Info,
) -> Tuple[NpArray64, NpArray64]:
    """
    Calculate number of spikes and records analysed from self.spkcnt_spike_info.
    spike_info is a rec x 1 list of dicts with key = peak time containing spike
    information.
    """
    spike_count = utils.np_empty_nan((len(spike_info)))
    counted_rec_nums = utils.np_empty_nan((len(spike_info)))

    for rec_idx, rec_spk_info in enumerate(spike_info):
        if isinstance(rec_spk_info, Dict) and rec_spk_info:
            spike_count[rec_idx] = len(rec_spk_info)
            counted_rec_nums[rec_idx] = rec_idx + 1

        elif rec_spk_info == 0:
            spike_count[rec_idx] = 0
            counted_rec_nums[rec_idx] = rec_idx + 1

    return spike_count, counted_rec_nums


def get_first_spike_latency(spike_info: Info, min_max_time: NpArray64, im_injection_start: NpArray64) -> NpArray64:
    """
    Calculate first spike latency per record from spike_info.

    INPUTS
    spike_info: list where entry = {} is record not analysed, entry = 0 is
                analysed with no spikes, and entry is dict of
                spikes with time of spike as key

    min_max_time: array of min/max time (S) per record

    im_injection_start: time (s) of current injection to subtract from first
                        spike time

    OUTPUT
    numpy array with nan = not analysed,  0 = spike or first_spike latency.
    """
    fs_latency = utils.np_empty_nan(len(spike_info))
    for rec_idx, rec_spikes in enumerate(spike_info):
        if isinstance(rec_spikes, Dict) and rec_spikes:
            sorted_rec_spikes = core_analysis_methods.sort_dict_based_on_keys(rec_spikes)
            first_spike_time = list(sorted_rec_spikes.keys())[0]

            fs_latency[rec_idx] = float(first_spike_time) - im_injection_start - min_max_time[rec_idx][0]
        elif rec_spikes == 0:
            fs_latency[rec_idx] = 0

    return fs_latency


def calculate_isi_measures(
    spike_info: Info,
    analysis_type: Literal["mean_isi_ms", "sfa_divisor_method", "sfa_local_variance_method"],
) -> NpArray64:
    """
    Calculate the mean isi or spike frequency accommodation (SFA)
    with divisor or local variance methods.

    INPUTS:
        spike_info: a list dictionaries, one list entry per record in the file.
                    Dictionary keys are spiketimes, values [spike peak Vm, spike idx]
                    analysis_type: isi method to run

    OUTPUT:
        analysed_data: list of calculated ISI measure, entries per record.
    """
    analysed_data = utils.np_empty_nan(len(spike_info))

    for rec_idx, rec_spikes in enumerate(spike_info):
        # rec analysed, not enough spikes
        if rec_spikes == 0 or len(rec_spikes) == 1 or (len(rec_spikes) == 2 and "sfa_" in analysis_type):
            analysed_data[rec_idx] = 0

        elif len(rec_spikes) == 0:
            # rec not analysed, leave as NaN
            continue

        else:
            assert isinstance(rec_spikes, Dict), "Type Error: spike_info not narrowed"
            sorted_rec_spikes = core_analysis_methods.sort_dict_based_on_keys(rec_spikes)
            spike_times = np.array(list(sorted_rec_spikes.keys())).astype(np.float64)

            if analysis_type == "mean_isi_ms":
                analysed_data[rec_idx] = np.mean(np.diff(spike_times))

            elif analysis_type == "sfa_divisor_method":
                isis = np.diff(spike_times)
                analysed_data[rec_idx] = isis[0] / isis[-1]

            elif analysis_type == "sfa_local_variance_method":
                analysed_data[rec_idx] = calculate_sfa_local_variance_method(spike_times)

    return analysed_data


def calculate_sfa_local_variance_method(
    spike_times: NpArray64,
) -> Union[Literal[0], NpArray64]:
    """
    Local variance method from:

    Shinomoto, Shima and Tanji. (2003). Differences in Spiking Patterns Among
    Cortical Neurons.  Neural Computation, 15, 2823-2842.

    INPUT: 1 x t array of spike times.
    """
    isi = np.diff(spike_times)

    if len(isi) < 2:
        return 0

    isi_shift_1_idx = isi[1:]
    isi_cut = isi[0:-1]
    n_minus_1 = len(isi_cut)

    local_variance = np.sum((3 * (isi_cut - isi_shift_1_idx) ** 2) / (isi_cut + isi_shift_1_idx) ** 2) / n_minus_1

    local_variance = np.array(local_variance)  # for typing

    return local_variance


def calculate_rheobase(
    spike_info: Info,
    single_im_per_rec_values: NpArray64,
    im_array: NpArray64,
    rec_or_exact: Literal["record", "exact"],
    baselines: Optional[NpArray64],
    rec_from: Int,
    rec_to: Int,
) -> Tuple[Union[Literal[False], Int], FalseOrFloat]:
    """
    Coordinate calculation of rheobase by record (average Im across record) or exact
    (exact Im at peak time)

    INPUTS
    single_im_per_rec_values:  If "record" method, a rec x 1 array of average Im
    im_array: if "exact", the full rec x n_samples Im array
    rec_or_exac:  rheobase method, "record" or "exact"
    baselines: Im baseline to subtract the absolute Im at AP peak from to
               get delta Im
    """
    upper_inclusive_rec_bound = rec_to + 1
    for rec in range(rec_from, upper_inclusive_rec_bound):
        spikes = spike_info[rec]
        if spikes != 0 and any(spikes):
            first_spike_rec = rec
            break
    else:
        return False, False

    rheobase_rec = rec

    if rec_or_exact == "record":
        rheobase = single_im_per_rec_values[rec]

    elif rec_or_exact == "exact":
        first_spike_key = list(spike_info[first_spike_rec].keys())[0]  # type: ignore
        first_spike_idx = spike_info[first_spike_rec][first_spike_key][1]  # type: ignore

        assert baselines is not None, "Type Narrow baselines"
        rheobase = im_array[first_spike_rec][first_spike_idx] - baselines[rec]

    return rheobase_rec, rheobase


def update_rheobase_after_im_round(
    round_or_not_round: Literal["round", "not_round"], analysis_df: pd.DataFrame
) -> pd.DataFrame:
    """ """
    if (
        np.any(analysis_df.loc[0, "rheobase"])
        and "record" == analysis_df.loc[0, "rheobase_method"]
        and not np.any(analysis_df.loc[:, "user_input_im"])
    ):
        rheobase_rec = analysis_df.loc[0, "rheobase_rec"]

        if round_or_not_round == "round":
            analysis_df.loc[0, "rheobase"] = analysis_df.loc[rheobase_rec, "im_delta_round"]

        elif round_or_not_round == "not_round":
            analysis_df.loc[0, "rheobase"] = analysis_df.loc[rheobase_rec, "im_delta"]

    return analysis_df


def round_im_injection_to_user_stepsize(
    input_im: Union[pd.Series, NpArray64],
    step_size: Int,
    im_injection_step_direction: Literal["repeat", "increasing", "decreasing"],
) -> NpArray64:
    """
    Round raw im injection values to step size as input by user.

    INPUT
    input_im: array containing im injection value per
              record formatted as a pandas series

    step_size: step size provided by user (e.g. 4pA)

    OUTPUT
    numpy array of rounded im values
    """
    if not isinstance(input_im, np.ndarray):
        input_im = np.asarray(input_im)

    rounded_im_inj_np = utils.np_empty_nan((len(input_im)))

    pos_idx = 0
    for rec_idx, rec_im in enumerate(input_im):
        if np.isnan(rec_im):
            continue
        pos_idx += 1

        if step_size == 0:
            rounded_im_inj_np[rec_idx] = rec_im
        else:
            rounded_im_inj_np[rec_idx] = step_size * (round(rec_im / step_size))

        this_rec_rounded_im = rounded_im_inj_np[rec_idx]
        last_rec_rounded_im = rounded_im_inj_np[rec_idx - 1]

        if im_injection_step_direction == "repeat":
            continue

        elif im_injection_step_direction == "decreasing":
            if (
                pos_idx > 2
                and last_rec_rounded_im * 0.85 < this_rec_rounded_im < last_rec_rounded_im * 1.15
                and last_rec_rounded_im != this_rec_rounded_im + step_size
            ):
                rounded_im_inj_np[rec_idx] = last_rec_rounded_im - step_size

        elif im_injection_step_direction == "increasing":
            if (
                pos_idx > 2
                and last_rec_rounded_im * 0.85 < this_rec_rounded_im < last_rec_rounded_im * 1.15
                and last_rec_rounded_im != this_rec_rounded_im - step_size
            ):
                rounded_im_inj_np[rec_idx] = last_rec_rounded_im + step_size

    return rounded_im_inj_np


# --------------------------------------------------------------------------------------
# Input Resistance / Im Calculation Methods
# --------------------------------------------------------------------------------------


def calculate_input_resistance(im_in_pa: pd.DataFrame, vm_in_mv: pd.DataFrame) -> Tuple[NpArray64, Optional[NpArray64]]:
    """
    Calculate the input resistance as an OLS linear fit to delta-I, delta-V values.

    If only 1 pair of Im, Vm is input, calculate Ri by R = V / I

    INPUTS:
        im_in_pa: rec x 1 pandas series of current steps in pA
        vm_im_pA: rec x 1 pandas series of the change in voltage from baseline

    """
    im_in_pa = utils.convert_pd_to_np(im_in_pa)
    vm_in_mv = utils.convert_pd_to_np(vm_in_mv)

    im_in_na = im_in_pa / 1000

    if im_in_na.size == 1:
        input_resistance = vm_in_mv / im_in_na
        intercept = None

    else:
        im_in_na = np.atleast_1d(im_in_na)

        input_resistance, intercept, __, __, __ = scipy.stats.linregress(im_in_na, vm_in_mv)

    return input_resistance, intercept


def calculate_sag_ratio(sag_hump: NpArray64, peak_deflections: NpArray64) -> NpArray64:
    """
    Calculate the sag / hump ratio the sag / peak deflection.
    This is equivalent to the % of the total voltage deflection
    that is accounted for by the sag/hump.

    INPUT
        sag_hump: sag (or hump) rec x 1 of sag (peak - steady state) values
        peak_deflection: rec x 1 array of peak - baseline values

    OUTPUT
    rec x 1 array of sag ratios
    """
    sag_ratio = sag_hump / peak_deflections
    return sag_ratio


def calculate_baseline_minus_inj(
    data: NpArray64,
    time_array: NpArray64,
    bounds: List,
    bounds_start_or_stop: List,
    rec_from: Int,
    rec_to: Int,
    min_max_time: NpArray64,
) -> Tuple[NpArray64, ...]:
    """
    Finds difference between Im or Vm baseline (pre Im injection) and during
    injection, within time bounds.

    INPUT:
    data:         rec X sample array of Im or Vm
    time_array:   rec X sample array of time (S)
    bounds:       baseline start/stop and within-injection start/stop times (S)
    rec_from:     record to analyse from (zero-indexed)
    rec_to:       record to analyse to (zero-indexed)
    min_max_time: numpy array of min/max times per record
                  (used to account for cumulatively increasing time across
                  records)

    OUTPUT
    counted_recs:     array of analysed record numbers
    avg_over_period:  rec X value array of the difference between data at
                      baseline and during Im injection (averaged over bounds)
    """
    upper_inclusive_rec_bound = rec_to + 1

    num_recs = len(data)
    counted_recs = utils.np_empty_nan(num_recs)
    avg_over_period = utils.np_empty_nan(num_recs)
    baselines = utils.np_empty_nan(num_recs)
    steady_states = utils.np_empty_nan(num_recs)

    for rec in range(rec_from, upper_inclusive_rec_bound):
        bounds_sample = []
        for bound, start_or_stop in zip(bounds, bounds_start_or_stop):
            processed_bound = check_bounds_are_in_rec_or_single_form(bound, rec)

            bounds_sample.append(
                convert_time_to_samples(
                    processed_bound,
                    start_or_stop,
                    time_array,
                    min_max_time,
                    rec,
                    add_offset_back=True,
                )
            )

        baselines[rec] = np.mean(data[rec][bounds_sample[0] : bounds_sample[1] + 1])
        steady_states[rec] = np.mean(data[rec][bounds_sample[2] : bounds_sample[3] + 1])
        avg_over_period[rec] = steady_states[rec] - baselines[rec]
        counted_recs[rec] = rec + 1

    return counted_recs, avg_over_period, baselines, steady_states


def check_bounds_are_in_rec_or_single_form(bound: Union[List, NpArray64], rec: Int) -> NpArray64:
    """
    If boundaries are generated by linear region, they are 1 x rec lists.
    If they are generated by Input Im protocol, they are scalar and the same
    for all recs.
    """
    if isinstance(bound, list):
        return bound[rec]
    else:
        return bound


def find_negative_peak(
    vm: NpArray64,
    time_array: NpArray64,
    start_time: NpArray64,
    stop_time: NpArray64,
    min_max_time: NpArray64,
    rec_from: Int,
    rec_to: Int,
    avg_over_vm: NpArray64,
    vm_baselines: NpArray64,
    vm_steady_state: NpArray64,
    peak_direction: Literal["follow_im", "min", "max"],
) -> Tuple[List, NpArray64, NpArray64]:
    """
    Find the minimum, maximum or min/max in same direction as Im injection for a
    bounded Im period. Used for Sag / Hump Analysis

    INPUTS
    start_time, stop_time: time in s of the period to mind the min/max within
                            avg_over_vm - a rec x 1 array of the average change in
                            Vm used for following Im injection (Im not used in
                            case of 1-channel data)
    vm_steady_state: rec x 1 array of vm_steady_state Vm (as calculated by
                     the 'vm_steady_state' regions during input resistance analysis)
                     peak_direction - user specified direction of peak to find,
                     either "follow_im", "min" or "max". If "follow_im", min will be
                     used if Im injection was negative and max will be used
                     if positive

    see calculate_baseline_minus_inj() and calculate_rheobase() for other inputs.

    OUTPUTS:
        peaks: dict of length num_recs with peak information used for plotting
        sag_hump: the sag (or hump) (rec x 1 vector) for sag as defined peak -
                  vm_steady_state
        peak_deflection: rec x 1 vector of peak deflection as calculated peak -
        vm_baseline
    """
    upper_inclusive_rec_bound = rec_to + 1

    start_sample = convert_time_to_samples(
        start_time, "start", time_array, min_max_time, rec_from, add_offset_back=True
    )
    stop_sample = convert_time_to_samples(stop_time, "stop", time_array, min_max_time, rec_from, add_offset_back=True)

    peaks = [{} for rec in range(0, len(vm))]
    sag_humps = utils.np_empty_nan((len(vm)))
    peak_deflections = utils.np_empty_nan((len(vm)))
    for rec in range(rec_from, upper_inclusive_rec_bound):
        if peak_direction == "follow_im":
            if avg_over_vm[rec] > 0:
                peak_idx = np.argmax(vm[rec][start_sample : stop_sample + 1])
            else:
                peak_idx = np.argmin(vm[rec][start_sample : stop_sample + 1])

        elif peak_direction == "min":
            peak_idx = np.argmin(vm[rec][start_sample : stop_sample + 1] + 1)

        elif peak_direction == "max":
            peak_idx = np.argmax(vm[rec][start_sample : stop_sample + 1])

        peak_idx = start_sample + peak_idx
        peak_time = time_array[rec][peak_idx]
        peak_vm = vm[rec][peak_idx]
        sag_hump_vm = peak_vm - vm_steady_state[rec]

        peaks[rec][str(peak_time)] = [peak_vm, peak_idx, sag_hump_vm]
        sag_humps[rec] = sag_hump_vm
        peak_deflections[rec] = peak_vm - vm_baselines[rec]

    return peaks, sag_humps, peak_deflections


# --------------------------------------------------------------------------------------
# Spike Kinetic Methods
# --------------------------------------------------------------------------------------


def analyse_spike_kinetics(
    cfgs: ConfigsClass,
    data: RawData,
    start_time: NpArray64,
    rec_idx: Int,
    peak_idx: Int,
) -> Union[Literal[False], str, Dict]:
    """
    Calculate spike kinetic parameters from a records vm data given the
    index of the spike peak. Options to filter trace prior to threshold
    calculation, and 200 KhZ interpolation of spike prior to calculation
    of rise, decay and fwhm.

    INPUTS:  cfgs:       config class containing relevant analysis parameters
             data:       data object containing vm, and time data as well as key
             parameters
             start_time: time in ms preceding spike peak from which to analyse (e.g.
             for threshold detection)
             rec_idx:    rec_idx (zero-index) to analyse
             peak_idx:   sample idx of spike peak.

    OUTPUTS: dictionary containing time, vm and parameters of
             spike threshold,
             spike peak,
             fAHP & mAHP,
             full-width at half maximum (fwhm or half-width),
             rise time,
             decay time
             spike amplitude (peak - threshold).

             all times returned in S (for plotting), except for fwhm, rise_time and
             decay_time which are in ms (for display on table)

     norm derivatives are calculated as:
            data.norm_first_deriv_vm = np.diff(rec_vm, append=0) / sample_spacing_in_ms
            data.norm_second_deriv_vm = np.diff(rec_vm, n=2, append=[0,
            0]) / sample_spacing_in_ms
            data.norm_third_deriv_vm = np.diff(rec_vm, n=3, append=[0, 0,
            0]) / sample_spacing_in_ms

    """
    start_idx = convert_time_to_samples(
        start_time,
        "start",
        data.time_array,
        data.min_max_time,
        rec_idx,
        add_offset_back=False,
    )
    time_ = data.time_array[rec_idx]
    vm = data.vm_array[rec_idx]
    rec_first_deriv = data.norm_first_deriv_vm[rec_idx]
    rec_second_deriv = data.norm_second_deriv_vm[rec_idx]
    rec_third_deriv = data.norm_third_deriv_vm[rec_idx]

    if not vm.any():
        return False

    # Peak
    peak_vm = vm[peak_idx]
    peak_time = time_[peak_idx]

    # check spike-analysis region does not go over edge of rec
    fahp_stop_time, mahp_stop_time = get_ahp_times(cfgs, peak_time, time_)

    if start_time <= time_[0] or not all([fahp_stop_time, mahp_stop_time]):
        return False

    # Calculate threshold and amplitude
    thr_idx = core_analysis_methods.calculate_threshold(
        rec_first_deriv, rec_second_deriv, rec_third_deriv, start_idx, peak_idx, cfgs
    )
    if thr_idx is False:
        return "minimum_thr_to_peak_size_error"

    thr_idx = start_idx + thr_idx
    thr_vm = vm[thr_idx]
    thr_time = time_[thr_idx]
    amplitude = peak_vm - thr_vm

    # AHP
    fahp_time, fahp_vm, fahp_idx, fahp = core_analysis_methods.calculate_ahp(
        data,
        cfgs.skinetics["fahp_start"] * 1000,
        fahp_stop_time * 1000,
        vm,
        time_,
        peak_idx,
        thr_vm,
    )

    mahp_time, mahp_vm, mahp_idx, mahp = core_analysis_methods.calculate_ahp(
        data,
        cfgs.skinetics["mahp_start"] * 1000,
        mahp_stop_time * 1000,
        vm,
        time_,
        peak_idx,
        thr_vm,
    )

    # Rise, decay and fwhm
    thr_to_peak_vm = vm[thr_idx : peak_idx + 1]
    thr_to_peak_time = time_[thr_idx : peak_idx + 1]

    # calculate fall to where vm thr crosses or fahp
    peak_to_end_vm, peak_to_end_time = calculate_peak_to_end(vm, time_, peak_idx, fahp_idx, thr_vm, cfgs)
    if len(peak_to_end_vm) < 3:
        return "minimum_peak_to_decay_size_error"

    (rise_min_time, rise_min_vm, rise_max_time, rise_max_vm, rise_time,) = core_analysis_methods.calc_rising_slope_time(
        thr_to_peak_vm,
        thr_to_peak_time,
        thr_vm,
        peak_vm,
        cfgs.skinetics["rise_cutoff_low"],
        cfgs.skinetics["rise_cutoff_high"],
        cfgs.skinetics["interp_200khz"],
    )

    min_bound = thr_vm if cfgs.skinetics["decay_to_thr_not_fahp"] else fahp_vm
    (
        decay_min_time,
        decay_min_vm,
        decay_max_time,
        decay_max_vm,
        decay_time,
    ) = core_analysis_methods.calc_falling_slope_time(
        peak_to_end_vm,
        peak_to_end_time,
        peak_vm,
        min_bound,
        cfgs.skinetics["decay_cutoff_low"],
        cfgs.skinetics["decay_cutoff_high"],
        cfgs.skinetics["interp_200khz"],
    )

    half_amp = thr_vm + (amplitude / 2)
    (rise_mid_time, rise_mid_vm, decay_mid_time, decay_mid_vm, fwhm, __, __,) = core_analysis_methods.calculate_fwhm(
        thr_to_peak_time,
        thr_to_peak_vm,
        peak_to_end_time,
        peak_to_end_vm,
        half_amp,
        interp=cfgs.skinetics["interp_200khz"],
    )

    (
        rise_max_slope_ms,
        rise_max_slope_fit_time,
        rise_max_slope_fit_data,
        decay_max_slope_ms,
        decay_max_slope_fit_time,
        decay_max_slope_fit_data,
    ) = run_skinetics_max_slope(thr_to_peak_time, thr_to_peak_vm, peak_to_end_time, peak_to_end_vm, data, cfgs)
    # Save Outputs
    output = cfgs.skinetics_params()
    output["thr"] = {
        "time": thr_time,
        "vm": thr_vm,
        "thr_idx": thr_idx,
    }  # TODO: add to params!!
    output["peak"] = {"time": peak_time, "vm": peak_vm, "peak_idx": peak_idx}
    output["fahp"] = {
        "time": fahp_time,
        "vm": fahp_vm,
        "value": fahp,
        "fahp_idx": fahp_idx,
    }
    output["mahp"] = {"time": mahp_time, "vm": mahp_vm, "value": mahp}
    output["fwhm"] = {
        "rise_mid_time": rise_mid_time,
        "rise_mid_vm": rise_mid_vm,
        "decay_mid_time": decay_mid_time,
        "decay_mid_vm": decay_mid_vm,
        "fwhm_ms": fwhm * 1000,
    }
    output["rise_time"] = {
        "rise_min_time": rise_min_time,
        "rise_min_vm": rise_min_vm,
        "rise_max_time": rise_max_time,
        "rise_max_vm": rise_max_vm,
        "rise_time_ms": rise_time * 1000,
    }
    output["decay_time"] = {
        "decay_min_time": decay_min_time,
        "decay_min_vm": decay_min_vm,
        "decay_max_time": decay_max_time,
        "decay_max_vm": decay_max_vm,
        "decay_time_ms": decay_time * 1000,
    }
    output["amplitude"] = {"vm": amplitude}

    output["max_rise"] = {
        "max_slope_ms": rise_max_slope_ms,
        "fit_time": rise_max_slope_fit_time,
        "fit_data": rise_max_slope_fit_data,
    }
    output["max_decay"] = {
        "max_slope_ms": decay_max_slope_ms,
        "fit_time": decay_max_slope_fit_time,
        "fit_data": decay_max_slope_fit_data,
    }

    return output


def get_ahp_times(cfgs: ConfigsClass, peak_time: NpArray64, time_: NpArray64) -> Tuple[FalseOrFloat, FalseOrFloat]:
    """
    If any of the user-set time paranters extend past the trace
    end, set them to the trace endpoints
    """
    fahp_stop_time = adjust_ahp_time_for_end_of_trace(cfgs, peak_time, time_, "fahp")
    mahp_stop_time = adjust_ahp_time_for_end_of_trace(cfgs, peak_time, time_, "mahp")

    return fahp_stop_time, mahp_stop_time


def adjust_ahp_time_for_end_of_trace(
    cfgs: ConfigsClass, peak_time: NpArray64, time_: NpArray64, ahp_key: str
) -> FalseOrFloat:
    """
    Determine whether the user-set fAHP or mAHP search period extends
    over the end of the trace. If so, set it to the last timepoint in
    the trace. If the endpoint is now before the start point,
    return False.
    """
    ahp_stop_time = cfgs.skinetics[ahp_key + "_stop"]

    if peak_time + cfgs.skinetics[ahp_key + "_stop"] >= time_[-1]:
        ahp_stop_time = time_[-1] - peak_time

        if ahp_stop_time <= cfgs.skinetics[ahp_key + "_start"]:
            ahp_stop_time = False

    return ahp_stop_time


def run_skinetics_max_slope(
    thr_to_peak_time: NpArray64,
    thr_to_peak_vm: NpArray64,
    peak_to_end_time: NpArray64,
    peak_to_end_vm: NpArray64,
    data: RawData,
    cfgs: ConfigsClass,
) -> Tuple[
    Union[NpArray64, Literal["off"]],
    Union[NpArray64, List[float]],
    Union[NpArray64, List[float]],
    Union[NpArray64, Literal["off"]],
    Union[NpArray64, List[float]],
    Union[NpArray64, List[float]],
]:
    """ """
    if cfgs.skinetics["max_slope"]["on"]:
        (
            rise_max_slope_ms,
            rise_max_slope_fit_time,
            rise_max_slope_fit_data,
        ) = core_analysis_methods.calculate_max_slope_rise_or_decay(
            thr_to_peak_time,
            thr_to_peak_vm,
            start_idx=0,
            stop_idx=len(thr_to_peak_time) - 1,
            window_samples=cfgs.skinetics["max_slope"]["rise_num_samples"],
            ts=data.ts,
            smooth_settings={"on": False, "num_samples": 1},
            argmax_func=np.argmax,
        )

        (
            decay_max_slope_ms,
            decay_max_slope_fit_time,
            decay_max_slope_fit_data,
        ) = core_analysis_methods.calculate_max_slope_rise_or_decay(
            peak_to_end_time,
            peak_to_end_vm,
            start_idx=0,
            stop_idx=len(peak_to_end_time) - 1,
            window_samples=cfgs.skinetics["max_slope"]["decay_num_samples"],
            ts=data.ts,
            smooth_settings={"on": False, "num_samples": 1},
            argmax_func=np.argmin,
        )
    else:
        rise_max_slope_ms = decay_max_slope_ms = "off"
        rise_max_slope_fit_time = [np.nan]
        rise_max_slope_fit_data = [np.nan]
        decay_max_slope_fit_time = [np.nan]
        decay_max_slope_fit_data = [np.nan]

    return (
        rise_max_slope_ms,
        rise_max_slope_fit_time,
        rise_max_slope_fit_data,
        decay_max_slope_ms,
        decay_max_slope_fit_time,
        decay_max_slope_fit_data,
    )


def calculate_peak_to_end(
    vm: NpArray64,
    time_: NpArray64,
    peak_idx: Int,
    fahp_idx: Int,
    thr_vm: NpArray64,
    cfgs: ConfigsClass,
) -> Tuple[NpArray64, NpArray64]:
    """
    Get the time and datapoints from the AP peak to end, with end
    determined as fAHP or thr Vm based on user settings.
    """
    peak_to_fahp_vm = vm[peak_idx : fahp_idx + 1]
    peak_to_fahp_time = time_[peak_idx : fahp_idx + 1]
    if cfgs.skinetics["decay_to_thr_not_fahp"]:
        fall_thr_idx = peak_idx + np.abs(peak_to_fahp_vm - thr_vm).argmin()
        peak_to_end_vm = vm[peak_idx : fall_thr_idx + 1]
        peak_to_end_time = time_[peak_idx : fall_thr_idx + 1]
    else:
        peak_to_end_vm = peak_to_fahp_vm
        peak_to_end_time = peak_to_fahp_time

    return peak_to_end_vm, peak_to_end_time


# --------------------------------------------------------------------------------------
# Phase Plot Analysis
# --------------------------------------------------------------------------------------


def calculate_phase_plot(data: NpArray64, ts: NpArray64, interpolate: bool) -> Tuple[NpArray64, NpArray64]:
    """
    Calculate the phase plot of the data (usually Vm values of action potential).
    The phase plot is Vm diff (delta Vm / ms) plot against Vm.

    Options for cublic spline interpolation also provided. Interpolation factor of 100
    chosen during testing, providing high degree of smoothing without major impact on
    performance.

    INPUTS:
        data - 1D vector of data values (typically Vm values of an action potential)
        ts - time step used to calculate dVm / ms
        interpolate - bool for cubic spline interpolation of the data
    """
    vm_diff = np.diff(data) / (ts * 1000)
    vm = data[0:-1]

    if interpolate:
        interp_factor = 100
        time_ = np.arange(0, len(vm))
        vm_diff = core_analysis_methods.interpolate_data(vm_diff, time_, "cubic", interp_factor, 0)
        vm = core_analysis_methods.interpolate_data(vm, time_, "cubic", interp_factor, 0)

    return vm, vm_diff


def calculate_threshold(vm: NpArray64, vm_diff: NpArray64, threshold_cutoff: NpArray64) -> Tuple[NpArray64, NpArray64]:
    """
    Find the first dVm/ms datapoint over the given threshold (default 10).
    Gives the value in Vm and Vm derivative for plotting on phase-space plots.

    see calculate_phase_plot() for inputs.
    """
    above_threshold = np.where(vm_diff > threshold_cutoff)[0]
    if above_threshold.size == 0:
        return np.nan, np.nan  # type: ignore

    idx = np.min(above_threshold)
    return vm[idx], vm_diff[idx]


def calculate_vmax(vm: NpArray64, vm_diff: NpArray64) -> Tuple[NpArray64, NpArray64]:
    """
    Find maximum Vm (see calculate_threshold() for details)
    """
    idx = np.argmax(vm)
    return vm[idx], vm_diff[idx]


def calculate_vm_diff_max(vm: NpArray64, vm_diff: NpArray64) -> Tuple[NpArray64, NpArray64]:
    """
    Find maximum first derivative (see calculate_threshold() for details)
    """
    idx = np.argmax(vm_diff)
    return vm[idx], vm_diff[idx]


def calculate_vm_diff_min(vm: NpArray64, vm_diff: NpArray64) -> Tuple[NpArray64, NpArray64]:
    """
    Find minimum first derivative (see calculate_threshold() for details)
    """
    idx = np.argmin(vm_diff)
    return vm[idx], vm_diff[idx]


# --------------------------------------------------------------------------------------
# Time Indexing Methods and Utils
# --------------------------------------------------------------------------------------


def check_num_rec_samples(data: NpArray64, rec_from: Int, rec_to: Int) -> bool:
    """
    Check all records in the data have equal number of samples.
    """
    upper_inclusive_rec_bound = rec_to + 1
    for rec in range(rec_from, upper_inclusive_rec_bound):
        if rec == 0:
            pass
        else:
            if len(data[rec]) != len(data[rec - 1]):
                return False
    return True


def get_bound_times_in_sample_units(
    time_bounds: Union[Literal[False], List],
    start_or_stop: Union[Literal[False], List],
    data: RawData,
    rec_from: Int,
) -> Tuple[Int, Int]:
    """
    Converts boundary times from seconds to samples.
    Used for constructing period to search for spikes
    during spike counting.
    """
    if time_bounds is not False and start_or_stop is not False:
        time_bound_start_sample = convert_time_to_samples(
            time_bounds[0],
            start_or_stop[0],
            data.time_array,
            data.min_max_time,
            rec_from,
            add_offset_back=True,
        )
        time_bound_stop_sample = convert_time_to_samples(
            time_bounds[1],
            start_or_stop[1],
            data.time_array,
            data.min_max_time,
            rec_from,
            add_offset_back=True,
        )
    else:
        time_bound_start_sample = 0
        time_bound_stop_sample = len(data.vm_array[0])

    return time_bound_start_sample, time_bound_stop_sample


def convert_time_to_samples(
    timepoint: Union[float, np.float64, NpArray64],
    start_or_stop: Literal["start", "stop"],
    time_array: NpArray64,
    min_max_time: Optional[NpArray64],
    base_rec: Int,
    add_offset_back: bool,
) -> Int:
    """
    Convert a timepoint (s) to it's nearest sample, while accounting for cumulative
    changes in time across records

    If the timepoint supplied is a boundary, the nearest sample may be the wrong side
    e.g a start boundary of 0.1 S might have it's closest sample as 0.0999999 S.
    If a boundary type ("start" or "stop") is supplied and the found timepoint
    is the wrong side of the bound, the idx is +/- 1 to bring it to the correct
    side of the bound.
    """
    if add_offset_back:
        assert min_max_time is not None, "Type Narrow min_max_time"
        timepoint = timepoint + min_max_time[base_rec][0]

    idx = (np.abs(time_array[base_rec] - timepoint)).argmin()

    if start_or_stop == "start":
        if time_array[base_rec][idx] < timepoint:
            idx += 1
    elif start_or_stop == "stop":
        if time_array[base_rec][idx] > timepoint:
            idx -= 1

    return idx
