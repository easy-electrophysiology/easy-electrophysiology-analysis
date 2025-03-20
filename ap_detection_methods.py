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
from ephys_data_methods import current_calc
from ephys_data_methods.core_analysis_methods import (
    exclude_peak_idxs_based_on_local_maximum_period,
    quick_get_time_in_samples,
)
from utils import utils

if TYPE_CHECKING:
    from custom_types import Info, Int, NpArray64
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

    Alternatively, the spikes_from_auto_threshold_per_spike_legacy method just returns the
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
               thr["auto_thr_width_v2"] - region to search for first deriv pass negative
                                       threshold after passing positive threshold in ms.
           rec_from:     rec to analyse from
           rec_to:       rec to analyse to
           time_bounds:  boundary to analyse within, specified in seconds
           bounds_start_or_stop: whether "time_bounds" bounds are start or stop (e.g.
           ["start", "stop"])
    """
    search_region_in_s = thr["auto_thr_width_v2"] / 1000
    thr["N"] = quick_get_time_in_samples(data.ts, search_region_in_s)

    # Find Spike Threshold per Record
    if thr["threshold_type"] == "auto_record":
        spike_info = spikes_from_auto_threshold_per_record(
            data,
            rec_from,
            rec_to,
            thr,
            time_bounds,
            bound_start_or_stop,
        )
    elif thr["threshold_type"] == "auto_spike":
        spike_info = spikes_from_auto_threshold_per_spike(
            data,
            rec_from,
            rec_to,
            thr,
            time_bounds,
            bound_start_or_stop,
        )

    return spike_info


# --------------------------------------------------------------------------------------
# Original (slower, Slightly more accurate) implementation
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
         "N": thr["auto_thr_width_v2"] in samples
        "min_distance_between_peaks":  minimum distance between ap peaks in ms

    See auto_find_spikes() for other inputs
    """
    upper_inclusive_rec_bound = rec_to + 1

    thresholds = utils.np_empty_nan(data.num_recs)
    for rec in range(rec_from, upper_inclusive_rec_bound):
        data_array = data.get_primary_data()[rec]
        data_diff = data.get_norm_first_deriv_data()[rec]

        rec_time_bounds = [time_bounds[0][rec], time_bounds[1][rec]] if time_bounds is not False else False
        (
            time_bound_start_sample,
            time_bound_stop_sample,
        ) = current_calc.get_bound_times_in_sample_units(rec_time_bounds, bound_start_or_stop, data, rec)

        candidate_spikes_idx = find_candidate_spikes(data_diff, thr, time_bound_start_sample, time_bound_stop_sample)

        if candidate_spikes_idx.any():
            (peak_vms, thr_vms, __, __,) = clean_and_amplitude_thr_candidate_spikes_and_extract_parameters(
                data_array, candidate_spikes_idx, thr, rec_time_array=data.time_array[rec]
            )
            thresholds[rec] = (np.median(peak_vms) + np.median(thr_vms)) / 2
        else:
            thresholds[rec] = np.nan

    # If no spike in rec, uses the average of all thresholds as threshold
    if not np.all(np.isnan(thresholds)):
        mean_thr = np.nanmean(thresholds[rec_from:upper_inclusive_rec_bound])
        thresholds[np.isnan(thresholds)] = mean_thr

    spike_info = find_spikes_above_record_threshold(
        data, thresholds, rec_from, rec_to, time_bounds, bound_start_or_stop, thr["min_distance_between_aps_ms"]
    )

    return spike_info


def spikes_from_auto_threshold_per_spike(
    data: RawData,
    rec_from: Int,
    rec_to: Int,
    thr: Dict,
    time_bounds: Union[Literal[False], List[List]],
    bound_start_or_stop: Union[Literal[False], List[str]],
) -> Info:
    """
    Find APs from auto-thresholding without further robust thresholding based on
    detected APs.

    see auto_find_spikes() for inputs
    """
    upper_inclusive_rec_bound = rec_to + 1

    spike_info: Info
    spike_info = [[] for __ in range(data.num_recs)]

    for rec in range(rec_from, upper_inclusive_rec_bound):
        data_array = data.get_primary_data()[rec]
        data_diff = data.get_norm_first_deriv_data()[rec]

        rec_time_bounds = [time_bounds[0][rec], time_bounds[1][rec]] if time_bounds is not False else False
        (
            time_bound_start_sample,
            time_bound_stop_sample,
        ) = current_calc.get_bound_times_in_sample_units(rec_time_bounds, bound_start_or_stop, data, rec)

        candidate_spikes_idx = find_candidate_spikes(data_diff, thr, time_bound_start_sample, time_bound_stop_sample)

        if candidate_spikes_idx.any():
            (_, _, peak_idxs, _,) = clean_and_amplitude_thr_candidate_spikes_and_extract_parameters(
                data_array, candidate_spikes_idx, thr, rec_time_array=data.time_array[rec]
            )

            peak_idxs = process_min_distance_between_aps(
                peak_idxs, data_array[peak_idxs], thr["min_distance_between_aps_ms"], data.ts
            )

            if peak_idxs.any():
                spike_info[rec] = {}
                for peak_vm, peak_time, peak_sample in zip(
                    data_array[peak_idxs], data.time_array[rec][peak_idxs], peak_idxs
                ):
                    dict_key = str(peak_time)
                    spike_info[rec][dict_key] = [peak_vm, peak_sample]  # type: ignore
            else:
                spike_info[rec] = 0
        else:
            spike_info[rec] = 0

    return spike_info


def find_spikes_above_record_threshold(
    data: RawData,
    thresholds: NpArray64,
    rec_from: Int,
    rec_to: Int,
    time_bounds: Union[Literal[False], List[List]],
    bound_start_or_stop: Union[Literal[False], List[str]],
    min_distance_between_peaks: Optional[float] = None,
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
           min_distance_between_peaks:  minimum distance between ap peaks in ms

    OUTPUT: spike_info:  A list of records with entry [] for rec not analysed,
                         0 for analysed and no spikes, or a dictionary with
                         the spike time (S) as key, containing a list [spike_peak_vm,
                         spike_idx].
    """
    upper_inclusive_rec_bound = rec_to + 1

    spike_info: Info
    spike_info = [[] for rec in range(data.num_recs)]

    # Count Spikes
    # get vector of vm indices above thresholds (with length either 1 when thresholds
    # manual set or len(recs) when thresholds autoset), otherwise nan
    if np.size(thresholds) == 1:
        thresholds = np.tile(thresholds, (data.num_recs, 1))

    # cut vm and time down to time period to be analysed
    all_time_bound_start_sample = utils.np_empty_nan(data.num_recs)

    for rec in range(rec_from, upper_inclusive_rec_bound):

        rec_time_bounds = [time_bounds[0][rec], time_bounds[1][rec]] if time_bounds is not False else False
        (
            time_bound_start_sample,
            time_bound_stop_sample,
        ) = current_calc.get_bound_times_in_sample_units(rec_time_bounds, bound_start_or_stop, data, rec)

        all_time_bound_start_sample[rec] = time_bound_start_sample

        bound_data_array = data.get_primary_data()[rec, time_bound_start_sample : time_bound_stop_sample + 1]
        bound_time_array = data.time_array[rec, time_bound_start_sample : time_bound_stop_sample + 1]

        peak_idxs = index_peaks_above_threshold(
            bound_data_array, bound_data_array, thresholds[rec], direction="positive"
        )

        # Using peak indices, save the vm peak, its time and idx
        # into a dict where key is the time of spike
        if np.any(peak_idxs):

            if min_distance_between_peaks is not None:
                peak_idxs = process_min_distance_between_aps(
                    peak_idxs, bound_data_array[peak_idxs], min_distance_between_peaks, data.ts
                )

            spike_info[rec] = {}
            for peak_idx, peak_time, peak_vm in zip(
                peak_idxs,
                bound_time_array[peak_idxs],
                bound_data_array[peak_idxs],
            ):
                spike_info[rec][str(peak_time)] = [  # type: ignore
                    peak_vm,
                    peak_idx + all_time_bound_start_sample[rec].astype(int),
                ]  # add back the first bound
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
) -> NDArray[np.integer]:
    """
    Find candidate spikes based on user-specified thresholds, fully vectorised for
    speed.

    INPUTS:
        vm_diff: first derivative per unit time,
                 see spikes_from_auto_threshold_per_record_legacy() for thr

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
    # only consider points within boundary
    samples_above = samples_above[samples_above >= int(time_bound_start_sample + thr["N"])]
    samples_above = samples_above[samples_above < int(time_bound_stop_sample - thr["N"])]

    samples_below = np.where(vm_diff < thr["auto_thr_fall"])[0].squeeze()
    samples_below = samples_below[samples_below >= int(time_bound_start_sample + thr["N"])]
    samples_below = samples_below[samples_below < int(time_bound_stop_sample - thr["N"])]

    # make a sample x N array for every sample above thr
    n_samples_above = np.linspace(samples_above, samples_above + thr["N"] - 1, thr["N"], dtype=int).T

    # for the indices sample_above + N, check if any contains a below thr instance
    idx = np.isin(n_samples_above, samples_below).any(axis=1)
    candidate_spikes_idx = n_samples_above[idx]

    return candidate_spikes_idx


def clean_and_amplitude_thr_candidate_spikes_and_extract_parameters(
    vm: NpArray64,
    candidate_spikes_idx: NDArray[np.integer],
    thr: Dict,
    rec_time_array: NpArray64,
) -> Tuple[NpArray64, NpArray64, NDArray[np.integer], NpArray64]:
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


def process_min_distance_between_aps(
    peak_idxs: NDArray[np.integer], peaks_vm: NpArray64, min_distance_between_peaks_ms: float, ts: NpArray64
) -> NDArray[np.integer]:
    """"""
    min_distance = quick_get_time_in_samples(ts, min_distance_between_peaks_ms / 1000)
    if min_distance == 0:
        min_distance = 1

    peak_idxs = exclude_peak_idxs_based_on_local_maximum_period(peak_idxs, peaks_vm, min_distance)
    return peak_idxs


def index_peaks_above_threshold(
    detection_coefs: NpArray64,
    data_array: NpArray64,
    thr: NpArray64,
    direction: Literal["positive", "negative"],
    smooth=None,
) -> NDArray[np.integer]:
    """"""
    jj = (detection_coefs > thr).astype(int)

    # smooth any ones by 1 windows width in case of gaps
    if isinstance(smooth, int):
        W = smooth
        box_function = np.concatenate([[0, 0], np.ones(W), [0, 0]])
        jj = np.convolve(jj, box_function)
        jj[jj > 0] = 1
        jj = jj[0 : len(detection_coefs)]

    diffs = np.diff(np.r_[0, jj, 0]).astype(int)
    ups = np.where(diffs == 1)[0]
    downs = np.where(diffs == -1)[0] - 1

    peak_idxs = np.zeros(ups.size, dtype=int)

    for i, (up, down) in enumerate(zip(ups, downs)):

        if direction == "positive":
            peak_idxs[i] = up + np.argmax(data_array[np.arange(up, down + 1)])
        elif direction == "negative":
            peak_idxs[i] = up + np.argmin(data_array[np.arange(up, down + 1)])

    return peak_idxs
