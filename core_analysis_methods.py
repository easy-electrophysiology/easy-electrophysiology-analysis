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
import scipy
from easye_cython_code._vendored.scipy import _peak_finding_utils
from ephys_data_methods import voltage_calc
from statsmodels.stats import diagnostic
from utils import utils

if TYPE_CHECKING:
    from configs.configs import ConfigsClass
    from custom_types import Bool, Direction, FalseOrFloat, Info, Int, NpArray64
    from importdata import RawData
    from numpy.typing import NDArray

# --------------------------------------------------------------------------------------
# Curve Fitting
# --------------------------------------------------------------------------------------


def monoexp_function(x: NpArray64, coefs: Tuple) -> NpArray64:
    """
    Return monoexponential function of data x.

    INPUTS:
        x: time - time[0]
        coefs:  list / tuple of coefficients to use in order b0, b1, tau
    """
    b0, b1, tau = coefs

    return b0 + b1 * np.exp(-x / tau)


def biexp_decay_function(x: NpArray64, coefs: Tuple) -> NpArray64:
    """
    Return a biexpoential decay function of data x. See
    biexp_event_function for alternative formation of biexp
    function used for event fitting.

    INPUTS:
        x: time - time[0]
        coefs:  list / tuple of coefficients to use in order b0, b1, tau1, b2, tau2
    """
    b0, b1, tau1, b2, tau2 = coefs

    return b0 + b1 * np.exp(-x / tau1) + b2 * np.exp(-x / tau2)


def biexp_event_function(x: NpArray64, coefs: Tuple) -> NpArray64:
    """
    Return biexponential function of data x. Biexponential function
    is in the form described in:

    Jonas, P. Major, G. Sakmann, B. (1993). Quantal components of unitary EPSCs at
    the mossy fibre synapose on CA3 pyramidal cells of rat hippocampus.
    J Physio. 472, 615-663.

    Default coefficients 0.5 ms, 5 ms for rise and decay respectively work well.

    INPUTS:
        x: time - time[0]
        coefs: list / tuple of coefficients to use in order b0, b1, rise, decay

    """
    b0, b1, rise, decay = coefs

    return b0 + b1 * (1 - np.exp(-x / rise)) * np.exp(-x / decay)


def triexp_decay_function(x: NpArray64, coefs: Tuple) -> NpArray64:
    """
    Triexponential decay function of data x.

    INPUTS:
        x: time - time[0]
        coefs: list / tuple of coefficients to use in order b0, b1,
               tau1, b2, tau2, b3, tau3
    """
    b0, b1, tau1, b2, tau2, b3, tau3 = coefs

    return b0 + b1 * np.exp(-x / tau1) + b2 * np.exp(-x / tau2) + b3 * np.exp(-x / tau3)


def gaussian_function(x: NpArray64, coefs: Tuple) -> NpArray64:
    a, mu, sigma = coefs
    gauss = a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gauss


# Least Squares Cost Functions
# --------------------------------------------------------------------------------------
# Note that scipy.optimize.least_squares expects the cost-function to be residuals
# not the sum of squared error (it handles this under the hood)
# --------------------------------------------------------------------------------------


def monoexp_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    yhat = monoexp_function(x, coefs)
    return yhat - y


def biexp_event_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    yhat = biexp_event_function(x, coefs)
    return yhat - y


def biexp_decay_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    yhat = biexp_decay_function(x, coefs)
    return yhat - y


def triexp_decay_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    yhat = triexp_decay_function(x, coefs)
    return yhat - y


def gaussian_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    y_hat = gaussian_function(x, coefs)
    return y_hat - y


# Fit Curve ----------------------------------------------------------------------------


def fit_curve(
    analysis_name: str,
    x_to_fit: NpArray64,
    y_to_fit: NpArray64,
    direction: Direction,
    initial_est: Optional[Tuple] = None,
    bounds: Optional[Tuple] = None,
    normalise_time: Bool = True,
) -> Tuple[Union[Literal[False], Tuple], FalseOrFloat, FalseOrFloat]:
    """
    Fit curve of type monoexponential ("monoexp"), biexponential decay ("biexp_decay"),
    biexponential event ("biexp_event") and triponential ("triexp"), or "gaussian"

    INPUTS:
        analysis name: name of curve to fit (above; str)
        x_to_fit, y_to_fit: 1 x n vector of times (x) and data (y) to fit.
        direction: -1 for a negative "event" and 1 for a positive "event"
        initial_est: tuple in the form (est c1, est c2, est c3, ... cn) for starting
                     estimates of the coefficients (see get_initial_est())
        bounds: tuple in the form ((min c1, min c2, min c3, ... min cn),
                (max c1, max c2, max c3, ... max cn))
        normalise_time: subtract the first timepoint from x_to_fit so that
                        the time starts at zero.
                        Setting this to false can severely compromise fitting.

    OUTPUT:
        coefs: coefficients for the least-squares fit
        fit: the function generated with the coefs

    """
    if normalise_time:
        x_to_fit = x_to_fit - x_to_fit[0]  # x_to_fit is read only (no -=)

    if initial_est is None:
        initial_est = get_initial_est(analysis_name, direction, x_to_fit, y_to_fit)

    if bounds is None:
        bounds = get_curve_fit_bounds(analysis_name)

    try:
        coefs = scipy.optimize.least_squares(
            get_least_squares_fit_function(analysis_name),
            x0=initial_est,
            args=(x_to_fit, y_to_fit),
            bounds=bounds,
        )
        coefs = coefs.x
    except:
        return False, False, False

    func = get_fit_functions(analysis_name)
    fit = func(x_to_fit, coefs)
    r2 = calc_r2(y_to_fit, fit)

    return coefs, fit, r2


def calc_r2(y: NpArray64, y_hat: NpArray64) -> NpArray64:
    """
    Calculate the r-squared for a fit to data
    """
    y_bar = np.mean(y)
    SST = np.sum((y - y_bar) ** 2)
    RSS = np.sum((y - y_hat) ** 2)
    r2 = 1 - RSS / SST
    return utils.fix_numpy_typing(r2)


# Bounds and Initial Estimates
# --------------------------------------------------------------------------------------


def get_curve_fit_bounds(analysis_name: str) -> Tuple:
    """
    Set default bounds on curve fitting. For any Tau coefficients,
    set the start boundary to zero (avoid strange occurrences of negative
    tau when fit is poor).
    """
    bounds = {
        "monoexp": ((-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)),
        "biexp_decay": (
            (-np.inf, -np.inf, 0, -np.inf, 0),
            (np.inf, np.inf, np.inf, np.inf, np.inf),
        ),
        "biexp_event": ((-np.inf, -np.inf, 0, 0), (np.inf, np.inf, np.inf, np.inf)),
        "triexp": (
            (-np.inf, -np.inf, 0, -np.inf, 0, -np.inf, 0),
            (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
        ),
        "gaussian": ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)),
    }

    return bounds[analysis_name]


def get_initial_est(analysis_name: str, direction: Direction, x_to_fit: NpArray64, y_to_fit: NpArray64) -> Tuple:
    """
    Get the default initial estimates for Curve Fitting functions.

    see fit_curve() for input and output. Exponential functions ("monoexp",
    "biexp_decay", "triexp") share the same estimate generation with the tau
    is tai weighted across coefficients. "biexp_event" uses rise and deay
    from Jonas et al., 1993.
    """
    if analysis_name == "monoexp":
        est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit, y_to_fit)
        initial_est = (est_offset, est_slope, est_tau)

    elif analysis_name == "biexp_decay":
        est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit, y_to_fit)
        initial_est = (est_offset, est_slope, est_tau * 0.1, est_slope, est_tau * 0.9)

    elif analysis_name == "biexp_event":
        initial_est = get_biexp_event_initial_est(x_to_fit, y_to_fit, direction)

    elif analysis_name == "triexp":
        est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit, y_to_fit)
        initial_est = (
            est_offset,
            est_slope,
            est_tau / 3,
            est_slope,
            est_tau / 3,
            est_slope,
            est_tau / 3,
        )

    elif analysis_name == "gaussian":
        initial_est = get_gaussian_initial_est(x_to_fit, y_to_fit)

    return initial_est


def get_exponential_function_initial_est(
    x_to_fit: NpArray64, y_to_fit: NpArray64
) -> Tuple[NpArray64, NpArray64, NpArray64]:
    """
    Get starting estimates for exponential functions.
    See fit_curve() for inputs.

    Calculate offset as mean, slope as amplitude and tau as
    time to half-amplitude / log(2).
    """
    start = y_to_fit[0]
    end = y_to_fit[-1]

    est_offset = np.mean(y_to_fit)
    est_slope = start - end

    half_amp = start + ((end - start) / 2)
    rough_midpoint = np.argmin(np.abs(half_amp - y_to_fit))
    est_tau = x_to_fit[rough_midpoint] / np.log(2)

    return (
        utils.fix_numpy_typing(est_offset),
        utils.fix_numpy_typing(est_slope),
        utils.fix_numpy_typing(est_tau),
    )


def get_biexp_event_initial_est(x_to_fit: NpArray64, y_to_fit: NpArray64, direction: Direction) -> Tuple:
    """
    Get the Curve Fitting starting estimates for the biexponential event function.

    Since v2.3.0 rise and decay initial estimates are always taken from user configs,
    and filled in to the returned initial_est.
    """
    default_rise = default_decay = None

    est_offset, est_slope, __ = get_exponential_function_initial_est(x_to_fit, y_to_fit)

    if direction == -1:
        est_slope = np.min(y_to_fit) - y_to_fit[0]
    elif direction == 1:
        est_slope = np.max(y_to_fit) - y_to_fit[0]
    initial_est = (est_offset, est_slope, default_rise, default_decay)

    return initial_est


def get_gaussian_initial_est(x: NpArray64, y: NpArray64) -> Tuple[NpArray64, NpArray64, NpArray64]:
    """
    Return the initial estimate of parameters for fitting gaussian.
    """
    a = np.max(y)
    mu = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mu) ** 2) / np.sum(y))
    return (
        utils.fix_numpy_typing(a),
        utils.fix_numpy_typing(mu),
        utils.fix_numpy_typing(sigma),
    )


# Convenience Functions
# --------------------------------------------------------------------------------------


def get_least_squares_fit_function(analysis_name: str) -> Callable:
    """
    Convenient access for residual fitting functions that return y - yhat used in
    scipy curve fit. Note that scipy.optimize.least_squares expects the cost-function
    to be residuals (not sum of squared error, it handles this under the hood).
    """
    least_squares_fit_functions = {
        "monoexp": monoexp_least_squares_cost_function,
        "biexp_decay": biexp_decay_least_squares_cost_function,
        "biexp_event": biexp_event_least_squares_cost_function,
        "triexp": triexp_decay_least_squares_cost_function,
        "gaussian": gaussian_least_squares_cost_function,
    }
    return least_squares_fit_functions[analysis_name]


def get_fit_functions(analysis_name: str) -> Callable:
    """
    Convenient access for fitting functions.
    """
    fit_functions = {
        "monoexp": monoexp_function,
        "biexp_decay": biexp_decay_function,
        "biexp_event": biexp_event_function,
        "triexp": triexp_decay_function,
        "gaussian": gaussian_function,
    }

    return fit_functions[analysis_name]


# --------------------------------------------------------------------------------------
# Spike Detection - local maximum period thresholding
# --------------------------------------------------------------------------------------


def exclude_peak_idxs_based_on_local_maximum_period(
    peak_idxs: NDArray[np.integer], peaks_im: NpArray64, min_distance: Int
) -> NDArray[np.integer]:
    """
    Exclude peaks using SciPy's _select_peak_by_distance algorithm that underpins
    Scipy's find_peaks `distance` argument. This is used for Template analysis,
    ensuring distance exclusions are performed similarly for Template and Threshold
    analysis. It will find the most positive peak, so need to convert if negative
    events.

    INPUTS
    ------
    peak_idxs : Numpy array of event peak indices
    peaks_im : Numpy array of event peak data values
    min_distance : minimum distance in samples that must separate peaks
    """
    keep = _peak_finding_utils._select_by_peak_distance(peak_idxs.astype(np.intp), peaks_im, float(min_distance))
    return peak_idxs[keep]  # type: ignore


# --------------------------------------------------------------------------------------
# Spike Kinetics - Thresholding
# --------------------------------------------------------------------------------------


def calculate_threshold(
    rec_first_deriv: NpArray64,
    start_idx: Int,
    peak_idx: Int,
    cfgs: ConfigsClass,
) -> Int:
    """
    Calculate AP threshold using first and third derivative methods,
    or the phase-space methods described in:

    Sekerli, M. Del Negro, C. A. Lee, R. H and Butera, R. J. (2004).
    Estimating action potential thresholds from neuronal time-series:
    new metrics and evaluation of methodologies.
    IEEE Trans Biomed Eng. 51(9): 1665-1672.

    In practice, the forward difference works best across different
    sampling rates. The higher order derivatives is only calculated when needed,
    leading to this nested conditional structure that is suboptimal from
    a readability perspective. The first derivative is needed for spike
    detected and is cached on the DataModel.
    """
    calc_first_deriv = rec_first_deriv[start_idx : peak_idx + 2]
    first_deriv = calc_first_deriv[:-2]

    if first_deriv.size == 0:
        return False

    if cfgs.skinetics["thr_method"] == "first_deriv":
        idx = first_derivative_threshold(cfgs, first_deriv)

    elif cfgs.skinetics["thr_method"] == "leading_inflection":
        idx = leading_inflection_threshold(first_deriv)

    else:
        calc_second_deriv = np.diff(calc_first_deriv)
        second_deriv = calc_second_deriv[:-1]

        if second_deriv.size == 0:
            return False

        if cfgs.skinetics["thr_method"] == "method_I":
            idx = method_i_threshold(cfgs, first_deriv, second_deriv)

        elif cfgs.skinetics["thr_method"] == "max_curvature":
            idx = max_curvature_threshold(first_deriv, second_deriv)

        else:
            third_deriv = np.diff(calc_second_deriv)

            if third_deriv.size == 0:
                return False

            if cfgs.skinetics["thr_method"] == "third_deriv":
                idx = third_derviative_threshold(cfgs, third_deriv)

            elif cfgs.skinetics["thr_method"] == "method_II":
                idx = method_ii_threshold(cfgs, first_deriv, second_deriv, third_deriv)

            else:
                raise ValueError("`thr_method` not recognised.")

    return idx


def first_derivative_threshold(cfgs: ConfigsClass, first_deriv: NpArray64) -> Int:
    if cfgs.skinetics["first_deriv_max_or_cutoff"] == "max":
        idx = np.nanargmax(first_deriv)
    else:
        cutoff = cfgs.skinetics["first_deriv_cutoff"]
        idx = (first_deriv > cutoff).argmax()
    return idx


def third_derviative_threshold(cfgs: ConfigsClass, third_deriv: NpArray64) -> Int:
    if cfgs.skinetics["third_deriv_max_or_cutoff"] == "max":
        idx = np.nanargmax(third_deriv)
    else:
        cutoff = cfgs.skinetics["third_deriv_cutoff"]
        idx = (third_deriv > cutoff).argmax()
    return idx


def method_i_threshold(cfgs: ConfigsClass, first_deriv: NpArray64, second_deriv: NpArray64) -> Int:
    g = second_deriv / first_deriv
    lower_bound = cfgs.skinetics["method_I_lower_bound"]
    g[first_deriv <= lower_bound] = np.nan

    if np.all(np.isnan(g)):
        return False

    idx = np.nanargmax(g)
    return idx


def method_ii_threshold(
    cfgs: ConfigsClass,
    first_deriv: NpArray64,
    second_deriv: NpArray64,
    third_deriv: NpArray64,
) -> Int:

    num = third_deriv * first_deriv - second_deriv**2
    denom = first_deriv**3

    h = np.divide(num, denom, out=np.zeros_like(num, dtype=np.float64), where=denom != 0)

    lower_bound = cfgs.skinetics["method_II_lower_bound"]

    h[first_deriv <= lower_bound] = 0

    if not np.any(h):
        return False

    idx = np.argmax(h)

    return idx


def leading_inflection_threshold(first_deriv: NpArray64) -> Int:
    idx = np.nanargmin(first_deriv)
    return idx


def max_curvature_threshold(first_deriv: NpArray64, second_deriv: NpArray64) -> Int:
    k = second_deriv * (1 + first_deriv**2) ** (-3 / 2)
    idx = np.nanargmax(k)
    return idx


# --------------------------------------------------------------------------------------
# Spike and Event Kinetics
# --------------------------------------------------------------------------------------


def calculate_ahp(
    data: RawData,
    start_time: Union[Int, NpArray64],
    stop_time: Union[Int, NpArray64],
    vm: NpArray64,
    time_: NpArray64,
    peak_idx: Int,
    thr_vm: NpArray64,
) -> Tuple[NpArray64, NpArray64, Int, NpArray64]:
    """
    Find the action potential after-hyperpolarisation (AHP) from Vm and time arrays

    INPUTS:
        data: data class with .fs attribute as sampling frequency
        start_time: time (in ms) after peak to begin search for AHP
        stop_time: time (in ms) after peak to stop search for AHP
        vm: 1 x t array containing Vm values
        time: 1 x t array containing time values
        peak_idx: idx of the AP peak
        thr_vm: Vm value of the AP threshold.

    Calculate ahp, used to calculate both fAHP and mAHP based on bounds.
    See analyse_spike_kinetics() for inputs.
    """
    samples_per_ms = data.fs / 1000
    ahp_search_start_idx = peak_idx + np.round(start_time * samples_per_ms).astype(np.int32)
    ahp_search_stop_idx = peak_idx + np.round(stop_time * samples_per_ms).astype(np.int32)
    ahp_idx = np.argmin(vm[ahp_search_start_idx : ahp_search_stop_idx + 1])
    ahp_idx = ahp_search_start_idx + ahp_idx
    ahp = vm[ahp_idx] - thr_vm

    return time_[ahp_idx], vm[ahp_idx], ahp_idx, ahp


def calculate_fwhm(
    start_to_peak_time: NpArray64,
    start_to_peak_data: NpArray64,
    peak_to_end_time: NpArray64,
    peak_to_end_data: NpArray64,
    half_amp: NpArray64,
    interp: bool,
) -> Tuple[NpArray64, NpArray64, NpArray64, NpArray64, NpArray64, np.integer, np.integer]:
    """
    Calculate the full-width at half maximum (half width) or an AP or PSP
    (hereby referred to as "event").

    First find the "true" half width as amplitude / 2.

    Then find the nearest real sample to this point for both the AP rise and
    decay and subtract to find the time between them.

    Interpolation significantly increases the accuracy.

    start_to_peak_time: a 1 x t array containing timepoints for an event from the
                        threshold to peak
    start_to_peak_data: a 1 x t array containing datapoints as above
    peak_to_end_time: a 1 x t array containing timepoints for an even from the
                      peak to the end point (e.g. fAHP)
    peak_to_end_data: a 1 x t array containing datapoints as above
    half_amp: scalar value, the amplitude / 2 representing the true half-maximum data
    interp: bool, interpolate data to 200 kHz before calculating fwhm.
    """
    # save the un-interpolated index to calculate average event from
    orig_rise_mid_idx = np.abs(half_amp - start_to_peak_data).argmin()
    orig_decay_mid_idx = np.abs(half_amp - peak_to_end_data).argmin()

    if interp:
        start_to_peak_data, start_to_peak_time = twohundred_kHz_interpolate(start_to_peak_data, start_to_peak_time)
        peak_to_end_data, peak_to_end_time = twohundred_kHz_interpolate(peak_to_end_data, peak_to_end_time)

    rise_mid_idx = np.abs(half_amp - start_to_peak_data).argmin()
    decay_mid_idx = np.abs(half_amp - peak_to_end_data).argmin()

    rise_midpoint = start_to_peak_data[rise_mid_idx]
    rise_midtime = start_to_peak_time[rise_mid_idx]
    fall_midpoint = peak_to_end_data[decay_mid_idx]
    fall_midtime = peak_to_end_time[decay_mid_idx]
    fwhm = fall_midtime - rise_midtime

    return (
        rise_midtime,
        rise_midpoint,
        fall_midtime,
        fall_midpoint,
        fwhm,
        orig_rise_mid_idx,
        orig_decay_mid_idx,
    )


def calc_rising_slope_time(
    slope_data: NpArray64,
    slope_time: NpArray64,
    min_: NpArray64,
    max_: NpArray64,
    min_cutoff_perc: Union[Int, NpArray64],
    max_cutoff_perc: Union[Int, NpArray64],
    interp: bool,
) -> Tuple[NpArray64, ...]:
    """
    Calculate the rise-time of an AP or PSP  (hereby referred to as "event").
    Find the time from the threshold of an event to the peak, or some percentage
    of the amplitude specified the user e.g. the 10-90 rise time.

    INPUT:
        slope_data: 1 x t array of datapoints (e.g. Vm) from threshold to peak
        slope_time: 1 x t array of timepoints as above
        min_: the minimum (i.e. threshold) value
        max_: the maximum (i.e. peak) value
        min_cutoff_perc: the scalar cutoff value (minimum) to measure the rise_time
                         from (i.e. 10 for the 10-90 rise time)
        max_cutoff_perc: the scalar cutoff value (maximum) to measure the rise
                         time to (i.e. 90 in the 10-90 rise time)
        interp: bool, whether to interpolate prior to calculation

    Notes
    -----
    The reason min_ and max_ are not taken as the min or max values of  the slope
    array is in case these values do not correspond to the raw data e.g. if the
    baseline data value has been averaged over prior points.
    """
    if interp:
        slope_data, slope_time = twohundred_kHz_interpolate(slope_data, slope_time)

    amplitude = max_ - min_
    min_cutoff = min_ + amplitude * (min_cutoff_perc / 100)
    max_cutoff = min_ + amplitude * (max_cutoff_perc / 100) if max_cutoff_perc != 100 else max_
    min_cutoff_idx = np.abs(slope_data - min_cutoff).argmin()
    max_cutoff_idx = np.abs(slope_data - max_cutoff).argmin()

    min_time = slope_time[min_cutoff_idx]
    min_data = slope_data[min_cutoff_idx]
    max_time = slope_time[max_cutoff_idx]
    max_data = slope_data[max_cutoff_idx]
    rise_time = max_time - min_time

    return min_time, min_data, max_time, max_data, rise_time


def calc_falling_slope_time(
    slope_data: NpArray64,
    slope_time: NpArray64,
    max_: NpArray64,
    min_: NpArray64,
    max_cutoff_perc: Union[Int, NpArray64],
    min_cutoff_perc: Union[Int, NpArray64],
    interp: bool,
) -> Tuple[NpArray64, ...]:
    """
    Calculate the time constant of a falling slope (this is rise time for events
    and decay time for APs. See calc_rising_slope_time or details.

    INPUT
        slope_data:  timeseries from max to min of the Im or Vm (e.g. AP peak to fAHP)
        slope_time: time units of the period (s)
        max_: max value of the slope of interest (pA for Im and mV for Vm).
              Almost always but not necessarily
              the max value of the timeseries. e.g. AP peak or negative event baseline
        min_: min value of the slope of interest (e.g. AP fAHP or negative event peak)
        max_cutoff_perc: cutoff percentage for the minimum value
        min_cutoff_perc: cutoff percentage for the maximum value
        interp: bool, to interp 200 kHz
    """
    if interp:
        slope_data, slope_time = twohundred_kHz_interpolate(slope_data, slope_time)

    fall_amplitude = max_ - min_
    max_cutoff = max_ - fall_amplitude * (max_cutoff_perc / 100)
    min_cutoff = max_ - fall_amplitude * (min_cutoff_perc / 100) if min_cutoff_perc != 100 else min_
    max_cutoff_idx = np.abs(slope_data - max_cutoff).argmin()
    min_cutoff_idx = np.abs(slope_data - min_cutoff).argmin()

    max_time = slope_time[max_cutoff_idx]
    max_data = slope_data[max_cutoff_idx]
    min_time = slope_time[min_cutoff_idx]
    min_data = slope_data[min_cutoff_idx]
    decay_time = min_time - max_time

    return max_time, max_data, min_time, min_data, decay_time


def calculate_max_slope_rise_or_decay(
    time_: NpArray64,
    data: NpArray64,
    start_idx: Int,
    stop_idx: Int,
    window_samples: Int,
    ts: NpArray64,
    smooth_settings: Dict,
    argmax_func: Callable,
) -> Tuple[NpArray64, ...]:
    """
    Calculate the maximum slope over a period of data, size 1 ... N.
    The regression over M points is calculated from
        1:M ... N-M:M. The regression is not calculated across points N-M:N.
    If M is larger than N, no max slope is calculated.

    For N = 2, slope is calculated as rise over run for performance increase.

    INPUTS:
        time_: all timepoints in a record, 1...Z
        data:  all datapoints in a record, 1...Z
        start_idx: start index for the period to search the max slope
        stop_idx: stop index for the period to search the max slope (1...N)
        window_samples: number of samples to calculate the regression over, 1...M
        ts: sample spacing in seconds
        smooth_settings: dict with settings on the smoothing:
                         {"on": bool_, "num_samples": num samples to smooth}
        argmax_func: np.max() or np.min() depending on if the slope is positive
                     or negative.

    Notes
    -----
    It was attempted to improve speed using statsmodel rolling_ols but
    it made little difference.
    """
    stop_idx -= window_samples
    n_samples = stop_idx - start_idx + 1

    if n_samples < 1:
        return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

    if smooth_settings["on"]:
        data = voltage_calc.quick_moving_average(data, smooth_settings["num_samples"])
    if window_samples == 2:
        idx = np.arange(start_idx, stop_idx + 1)
        time_step_ms = ts * 1000

        windowed_deriv = (data[idx + 1] - data[idx]) / time_step_ms
        max_slope_idx = argmax_func(windowed_deriv)
        max_slope_ms = windowed_deriv[max_slope_idx]
        max_slope_idx += start_idx

        fit_time = np.array([time_[max_slope_idx], time_[max_slope_idx + 1]])
        fit_data = np.array([data[max_slope_idx], data[max_slope_idx + 1]])
    else:
        idx = (
            np.tile(np.arange(start_idx, start_idx + window_samples), (n_samples, 1))
            + np.atleast_2d(np.arange(n_samples)).T
        )
        all_x = time_[idx]
        all_y = data[idx]

        params = utils.np_empty_nan((n_samples, 2))
        for i in range(n_samples):
            x = all_x[i, :]
            y = all_y[i, :]
            params[i, 0], params[i, 1], __, __, __ = scipy.stats.linregress(x, y)

        max_slope_idx = argmax_func(params[:, 0])
        slope, intercept = params[max_slope_idx, :]

        fit_time = all_x[max_slope_idx, :]
        fit_data = fit_time * slope + intercept
        max_slope_ms = slope / 1000

    return max_slope_ms, fit_time, fit_data


def twohundred_kHz_interpolate(
    vm: NpArray64,
    time_: NpArray64,
    khz_interpolate: float = 200000.0,
    interp_method: str = "linear",
) -> Tuple[NpArray64, NpArray64]:
    """
    Interpolate to 200 kHz with linear interpolation.
    """
    fs, _ = calc_fs_and_ts(time_)
    interp_factor = khz_interpolate / fs
    vm = interpolate_data(vm, time_, interp_method, interp_factor, 0)
    time_array = interpolate_data(time_, time_, "linear", interp_factor, 0)

    return vm, time_array


def area_under_curve_ms(y: NpArray64, ts: NpArray64) -> Tuple[NpArray64, NpArray64]:
    """
    Calculate the area under the curve of data. Note that it is assumed data is
    already baseline subtracted if required.

    y = 1 x N data array (baseline subtracted if required)
    ts - sampling step in seconds

    OUTPUT
        area_under_curve - area under the curve in units Data units x ms (e.g. pA ms)
        area_under_curve_time_ms - time period over which the AUC was calculated
    """
    ts_ms = ts * 1000

    area_under_curve_time_ms = ts_ms * (y.size - 1)

    area_under_curve = np.trapz(y, dx=ts_ms)  # type: ignore

    return area_under_curve, area_under_curve_time_ms


# --------------------------------------------------------------------------------------
# Data Tools
# --------------------------------------------------------------------------------------

# Filter
# --------------------------------------------------------------------------------------


def filter_data(
    data: NpArray64,
    fs: Union[float, NpArray64],
    filter_: Literal["bessel", "butter"],
    order: Int,
    cutoff_hz: Union[float, NpArray64, Tuple],
    btype: str,
    axis: Int,
) -> NpArray64:
    """
    Filter data with bessel or butterworth digital filter (wrapper
    around Scipy functions)

    INPUT: data: data to filter
           fs:                sampling frequency (Hz)
           filter_:           "bessel" or "butter"
           order:             order of filter. Note that because filtfilt() is
                              zero-phase data is filtered twice is forward and reverse
                              direction, this effective order is doubled.
                              e.g. if order = 1, true order of the filter will be 2.
           cutoff_hz:         the frequecy cutoff in Hz
           btype:  "lowpass" or "highpass"
           axis:              axis of array to filter along
    """
    if filter_ == "bessel":
        sos = get_bessel(fs, order, cutoff_hz, btype)
    elif filter_ == "butter":
        sos = get_butterworth(fs, order, cutoff_hz, btype)

    filtered_data = scipy.signal.sosfiltfilt(sos, data, axis=axis)

    return filtered_data


def get_bessel(
    fs: Union[float, NpArray64],
    order: Int,
    cutoff_hz: Union[float, NpArray64, Tuple],
    btype: str,
) -> NpArray64:
    sos = scipy.signal.bessel(N=order, Wn=cutoff_hz, btype=btype, output="sos", fs=fs)
    return sos


def get_butterworth(
    fs: Union[float, NpArray64],
    order: Int,
    cutoff_hz: Union[float, NpArray64, Tuple],
    btype: str,
) -> NpArray64:
    sos = scipy.signal.butter(N=order, Wn=cutoff_hz, btype=btype, output="sos", fs=fs)
    return sos


def get_fft(y: NpArray64, detrend: bool, fs: NpArray64, cut_down: bool) -> Dict:
    """
    Returns the fast-Fourier transformation of the signal y, wrapper for scipy function.

    INPUT: y:       data to fft
           detrend: bool to detrend signal prior to transformation
           fs:      if sampling frequency in Hz provided, returns the frequencies
                    corresponding to power at Y.
           cut_down: if true, negative frequencies will be cut out and the
                     magnitude doubled.
                     An offset (10 default) is cut from the start of the spectrum
                     so low frequency noise doesn't dominate the plot.

    OUTPUTS: a dictionary containing:   out["Y"]: fft of y
                                        out["freqs"]: sample frequencies
                                        out["cutx"]: if cut_down is selected, this is
                                                     the offset to cut low frequency
                                                     noise
                                        out["N"]: number of samples in the
                                                  cut-down spectra
    """
    out = {"Y": None, "freqs": None, "cutx": None, "N": None}

    n_samples = len(y)
    if detrend:
        y = scipy.signal.detrend(y)

    Y = scipy.fft.fft(y) / n_samples
    Y = np.abs(Y)
    freqs = scipy.fft.fftfreq(n_samples, (1 / fs))

    if cut_down:
        offset = 10
        N = int(len(y) / 2)
        Y = Y[offset:N] * 2
        freqs = freqs[offset:N]
        out["cutx"], out["N"] = offset, N  # type: ignore

    out["Y"], out["freqs"] = Y, freqs
    return out


# Interpolate and Downsample
# --------------------------------------------------------------------------------------


def interpolate_data(
    data_to_interp: NpArray64,
    time_array: Union[NDArray[np.integer], NpArray64],
    interp_method: str,
    interp_factor: Union[Int, NpArray64],
    axis: Int,
) -> NpArray64:
    """
    Interpolate data, wrapper for Scipy function.

    INPUT: data_to_interp: array to interpolate
           time_array:     corresponding time array of data to interp
           interp_method:  ‘linear’, ‘nearest’ or ‘cubic used here.
           interp_factor   integer factor to interpolate by
           axis:           axis to interpolate
    """

    interp_f = scipy.interpolate.interp1d(time_array, data_to_interp, axis=axis, kind=interp_method)

    n = len(time_array)
    new_sample_num = interp_factor * (n - 1) + 1

    time_interp = np.linspace(time_array[0], time_array[-1], int(new_sample_num))

    interp_data = interp_f(time_interp)

    return interp_data


def downsample_data(data: NpArray64, downsample_factor: Int, filter_opts: Optional[Dict] = None) -> NpArray64:
    """
    Downsample data by lower-pass filtered followed by linear interpolation.
    Slices along axis=1.

    INPUTS:
         data: data to downsample (matrix record X sample)
         downsample_factor: integer factor to downsample by
         filter_opts: optional dictionary - if provided data is low-pass filtered
                      below the nyquist frequency of the downsampled data before
                      downsampling

                      keys: data_fs: sampling frequency of the data
                            filter:  type of filter (see filter_data) (e.g. "bessel",
                                     "butter")
                            filter_order: order of the filter (see filter_data)

     OUTPUT:
        downsampled_data
    """
    if filter_opts:
        downsampled_fs = filter_opts["data_fs"] / downsample_factor
        lowpass_cutoff = np.floor(downsampled_fs / 2)

        data = filter_data(
            data,
            filter_opts["data_fs"],
            filter_opts["filter"],
            filter_opts["filter_order"],
            lowpass_cutoff,
            "lowpass",
            axis=1,
        )

    downsample_slice = [slice(None, None, None), slice(None, None, downsample_factor)]
    downsampled_data = data[tuple(downsample_slice)]

    return downsampled_data


# Detrend ------------------------------------------------------------------------------


def detrend_data(x: NpArray64, y: NpArray64, poly_order: Int) -> Tuple[NpArray64, NpArray64]:
    """
    Detrend signal with polynomial order N. Keep the mean.

    x: time data (1 x time (1st record))
    y: data (rec x time)
    poly_order: order of polynomial to detrend (1-20 for detrend data manipulation)
    """
    fit = fit_polynomial(x, y, poly_order)
    y_mean = np.mean(y) if y.ndim <= 1 else np.mean(y, axis=1)
    y_mean = np.atleast_2d(y_mean)
    detrended_y = y - fit + y_mean.T
    return detrended_y, fit


def fit_polynomial(x: NDArray, y: NpArray64, poly_order: Int, progress_bar_callback=None) -> NpArray64:
    """
    Convenience function for fitting polynomial to multiple rows at once,
    or a single 1D array.
    x: a N x 1 1D array
    y: a N x 1 1D array or N x M matrix

    where N is timepoint and M is record number. Matrix data is transposed
    for np.polyfit which only works with data column-wise
    """
    if y.ndim == 1:
        coefs = np.polyfit(x, y, deg=poly_order)
        fit = np.polyval(coefs, x)
        fit = np.reshape(fit, (1, len(fit)))
    else:
        coefs = np.polyfit(x, y.T, deg=poly_order).T

        fit = np.zeros(y.shape, dtype=np.float64)
        for irow in range(y.shape[0]):
            fit[irow, :] = np.polyval(coefs[irow, :], x)

            if progress_bar_callback is not None:
                progress_bar_callback()

    return fit


# Cut trace length and normalise time
# --------------------------------------------------------------------------------------


def cut_trace_length_time(
    time_method: str,
    time_array: NpArray64,
    new_start_sample: Union[NDArray[np.integer], Int],
    new_stop_sample: Union[NDArray[np.integer], Int],
) -> NpArray64:
    """
    Cut down the trace length between user-specified times. For mutli-record files,
    this provides  different methods the new time array can be handled
    (see time_method). Note the cut is not upper  bound inclusive.

    INPUTS:
        time_method -  "raw_times": keep the raw times e.g. if a record is cut from
                                    [0-1, 1-2] to 0.2 - 0.8 the times will
                                    be [0.2-0.8, 1.2-1.8]

                       "cumulative": the new time any start offset is removed and the
                                     time increases cumulatively. e.g. [0-1, 1-2] to
                                     0.2 - 0.8 will be [0-(0.6-ts), 0.6-(1.2-ts)]

                       "normalised": the times will be the same across all records
                                     with any start offset removed e.g. [0-1, 1-2] to
                                     0.2 - 0.8 will be [0-(0.6-ts), 0-(0.6-ts)]

        time_array - rec x num_samples matrix of time points

        new_start_sample, new_stop_sample - sample idx to cut the trace from / to

    """
    if time_method == "cumulative":
        norm_time_array = cut_trace_and_normalise_time(time_array, new_start_sample, new_stop_sample)

        rec_len = norm_time_array[0][-1]
        num_recs = norm_time_array.shape[0]
        ts = norm_time_array[0][1] - norm_time_array[0][0]

        rec_nums = np.atleast_2d(np.arange(0, num_recs)).T
        cut_time_array = norm_time_array + rec_nums * rec_len + rec_nums * ts

    elif time_method == "normalised":
        cut_time_array = cut_trace_and_normalise_time(time_array, new_start_sample, new_stop_sample)

    return cut_time_array


def cut_trace_and_normalise_time(
    time_array: NpArray64,
    new_start_sample: Union[NDArray[np.integer], Int],
    new_stop_sample: Union[NDArray[np.integer], Int],
) -> NpArray64:
    """
    see cut_trace_length_time()
    NOTE: this slicing is upper bound exclusive
    """
    norm_time_array = copy_and_normalise_time_array(time_array)
    norm_time_array = norm_time_array[:, new_start_sample:new_stop_sample]
    norm_time_array = norm_time_array - norm_time_array[0][0]

    return norm_time_array


def copy_and_normalise_time_array(time_array: NpArray64) -> NpArray64:
    """
    Normalise the time across records so it is the same for each record
    (use the first row and repeat across all records).
    """
    time_array = copy.deepcopy(time_array)
    time_array[1:None, :] = time_array[0, :]
    return time_array


# Reshape Data
# --------------------------------------------------------------------------------------


def reshape(data: NpArray64, num_samples: Int, num_records: Int) -> NpArray64:
    reshaped_data = np.reshape(data, [num_records, int(num_samples / num_records)])

    return reshaped_data


# --------------------------------------------------------------------------------------
# Cumulative Probability
# --------------------------------------------------------------------------------------


def process_frequency_data_for_cum_prob(
    event_times: Union[List[NpArray64], Tuple[NpArray64]],
    event_nums: Tuple[Int],
) -> Tuple[NpArray64, List[Int]]:
    """
    Process event times for cumulative probability graphs.

    event_times = list of event times (float)

    Start from zero to find the event-interval for the first event, all
    inter-event intervals. Also, return labels of the event numbers.
    """
    assert np.array_equal(event_times, np.sort(event_times)), "process_frequency_data_for_cum_prob times not sorted"

    data = np.diff(event_times)

    new_event_nums = []
    for idx in range(len(event_times) - 1):
        new_event_nums.append(str(event_nums[idx]) + "-" + str(event_nums[idx + 1]))

    return data, new_event_nums


def process_amplitude_for_frequency_table(
    amplitudes: Tuple[NpArray64], sort_method: str, event_nums: Tuple[Int]
) -> Tuple[NpArray64, NDArray[np.integer]]:
    """
    Return sorted, absolute amplitudes for cum prob analysis,
    and event numbers or labelling.
    """
    data = np.sort(np.abs(amplitudes))
    sort_idx = np.argsort(np.abs(amplitudes))

    if sort_method == "event_num":
        event_num_sort_idx = np.argsort(sort_idx)
        data = data[event_num_sort_idx]
        sort_idx = sort_idx[event_num_sort_idx]

    new_event_nums = np.array(event_nums)[sort_idx]

    return data, new_event_nums


def process_non_negative_param_for_frequency_table(
    non_neg_param: Tuple[NpArray64], sort_method: str, event_nums: Tuple[Int]
) -> Tuple[NpArray64, NDArray[np.integer]]:
    """
    Return sorted non-negative parameter (decay data)
    for cum prob binning, and array of event labels.
    """
    data = np.sort(non_neg_param)
    sort_idx = np.argsort(non_neg_param)

    if sort_method == "event_num":
        event_num_sort_idx = np.argsort(sort_idx)
        data = data[event_num_sort_idx]
        sort_idx = sort_idx[event_num_sort_idx]

    new_event_nums = np.array(event_nums)[sort_idx]

    return data, new_event_nums


def calc_cumulative_probability_or_histogram(
    data: NpArray64, settings: Dict, parameter: str, legacy_bin_sizes: bool
) -> Tuple[FalseOrFloat, FalseOrFloat, FalseOrFloat, Union[Literal[False], Int]]:
    """
    Return the histogram or cumulative probabilities of 'data'.

    INPUTS:
        data: n x 1 array of values (here Event parameter
              values e.g inter-event intervals)
        settings: dict of settings with fields:
            'plot_type': 'cum_prob' or 'hist'
            'binning method': 'auto' (numpy implementation),'custom_binnum',
                              'custom_binsize', 'num_events_divided_by'
            'custom_binnum': number of bins to divide the data into
            'custom_binsize': data range of bins, same as max(data) / binnum
            'divide_by_number': divisor if 'num_events_divided_by' option is chosen
            'x_axis_display': 'bin_centers', 'left_edge', 'right_edge'

        parameter: event parameter being analysed (frequency, amplitude, decay_tau, ...
                  (see configs, frequency_data_options["custom_binsize"]))
        legacy_bin_sizes: bool
    """
    cum_prob_or_hist = settings["plot_type"]

    if len(data) < 1:
        return False, False, False, False

    # Calculate number of bins based on user settings

    # Use bin range from 0 - max of data, previously a little extra padding
    # was added similar to scipy implementation, but using the data is more
    # interpretable for end user.
    if legacy_bin_sizes:
        raise NotImplementedError("Legacy bin sizes is deprecated")

    if settings["binning_method"] == "custom_binsize":
        limits, bins_or_num_bins = get_bins_and_limits_for_custom_binsize(data, settings, parameter)
    else:
        limits = (np.min(data), np.max(data))
        bins_or_num_bins = get_num_bins_from_settings(data, settings)

    # Calculate the y values and bin edges for the histogram / cumulative probability
    if cum_prob_or_hist == "cum_prob":
        y_values, bin_edges, binsize = calc_cumulative_probability(data, bins_or_num_bins, limits)

    elif cum_prob_or_hist == "hist":
        y_values, bin_edges, binsize = calc_histogram(data, bins_or_num_bins, limits)

    # format the x values based on user settings
    x_values = format_bin_edges(bin_edges, settings["x_axis_display"])

    return y_values, x_values, binsize, bins_or_num_bins


def get_bins_and_limits_for_custom_binsize(data, settings, parameter):
    """
    Custom binsize returns the full array of bins, giving more control
    that passing the number of bins to the underlying numpy functions.
    It also exposes the ability to adjust the starting bin value.

    see calc_cumulative_probability_or_histogram() for inputs

    limits: tuple
        (min, max) value of the bins. Only used for custom_binsize.
    """
    max_ = np.max(data)

    if settings["fix_start_bin"][parameter]["on"]:
        min_ = settings["fix_start_bin"][parameter]["value"]
        if min_ >= max_:
            min_ = np.min(data)
        limits = (min_, max_)
    else:
        limits = (np.min(data), max_)

    if settings["custom_binsize"][parameter] == 0:
        bins_or_num_bins = int(get_auto_num_bins(data))
    else:
        bins_or_num_bins = np.arange(
            limits[0], limits[1] + settings["custom_binsize"][parameter], settings["custom_binsize"][parameter]
        )

    return limits, bins_or_num_bins


def get_num_bins_from_settings(data: NpArray64, settings: Dict) -> Int:
    """
    Calculate the bin number based on user settings.
    see calc_cumulative_probability_or_histogram() for inputs
    """
    binning_method = settings["binning_method"]
    n_samples = len(data)

    if binning_method == "auto":
        num_bins = int(get_auto_num_bins(data))

    elif binning_method == "custom_binnum":
        num_bins = int(settings["custom_binnum"]) if settings["custom_binnum"] <= n_samples else n_samples

    elif binning_method == "num_events_divided_by":
        num_bins = int(np.round(len(data) / settings["divide_by_number"]))

    if num_bins > len(data) or num_bins == 0:
        num_bins = len(data)
    elif num_bins < 2:
        num_bins = 2

    return num_bins


def get_auto_num_bins(data: NpArray64) -> NpArray64:
    """
    Get bin numbers for cumulative frequency / histogram
    plots automatically.
    """
    try:
        num_bins = len(np.histogram_bin_edges(data, bins="auto"))
    except MemoryError:
        # can occur with this function when bin sizes are very small.
        num_bins = len(np.histogram_bin_edges(data, bins="sqrt"))
    return num_bins


def calc_histogram(data: NpArray64, bins_or_num_bins: Int, limits: Tuple) -> Tuple[NpArray64, NpArray64, NpArray64]:
    """ """
    if isinstance(bins_or_num_bins, int):
        info = np.histogram(data, bins=bins_or_num_bins, range=limits)
    else:
        info = np.histogram(data, bins=bins_or_num_bins)

    y_values = info[0]
    bin_edges = info[1]
    binsize = bin_edges[1] - bin_edges[0]

    return y_values, bin_edges, binsize


def calc_cumulative_probability(
    data: NpArray64, bins_or_num_bins: Int | NpArray64, limits: Tuple
) -> Tuple[NpArray64, NpArray64, NpArray64]:
    """
    Calculate the binned cumulative probability from data processed
    for cum prob analysis.

    INPUT:
        data: data processed for cum prob analysis (see above functions)
        bins_or_num_bins: number of bins to divide the data into, or an array of bin edges
        limits: start / end limits for the bins

    OUTPUT:
        cdf: 1 x bin cumulative probabilities
        bin_edges, binsize - generated bin edges and binsizes
    """
    counts, bin_edges, binsize = calc_histogram(data, bins_or_num_bins, limits)

    pdf = counts / np.sum(counts)
    cdf = np.cumsum(pdf)

    return cdf, bin_edges, binsize


def format_bin_edges(bin_edges: NpArray64, x_axis_display: str) -> NpArray64:
    """
    Format n + 1 array of bin edges for a histogram or cumulative probability plot.

    INPUTS:
        'bin edges': n + 1 list of bins, n x 1 array

        Format the bin edges as:
        x_axis_display (str):
            'bin_centre': take the center of each bin
                          (e.g bin 1 = 0 - 2, bin center = 1)
            'left_edge': take the left edge of the bins (i.e. the minimum
                         value, bin 1 = 0 - 2, left edge = 0)
            'right_edge': take the right edge of the bins (i.e. the
                          maximum value, bin 1 = 0 - 2, right edge = 2)

    """
    if x_axis_display == "bin_centre":
        x_values = get_bin_centers(bin_edges)
    elif x_axis_display == "left_edge":
        x_values = bin_edges[0:-1]
    elif x_axis_display == "right_edge":
        x_values = bin_edges[1:]

    return x_values


def get_bin_centers(bin_edges: NpArray64) -> NpArray64:
    return (bin_edges[:-1] + bin_edges[1:]) / 2


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------


def run_ks_test(
    data1: NpArray64,
    method: str,
    pop_mean: Optional[float] = None,
    pop_stdev: Optional[float] = None,
    format_p_for_gui: bool = True,
) -> Union[Literal[False], Dict]:
    """
    Run a one- or two-sample KS test. It is assumed data1 and data2 are already
    checked n matches. It would be neater to call scipy function once and pass
    empty args for args, n. But do not want to pass unused args to scipy function.

    data1 - 1 x N array of raw data values (not a CDF)

    data2 - 1 x N array of raw data values for a 2-sample KS test, or None for
            one sample (compared to gaussian M, SD of data1)

    alternative_hypothesis - tails on the test, see scipy.stats.kstest()

    format_p_for_gui - format p value for display on GUI
    """
    n = len(data1)

    if method == "lilliefors":
        results_tuple = diagnostic.lilliefors(data1)

    elif method == "user_input_population":
        results_tuple = scipy.stats.kstest(
            rvs=data1,
            cdf="norm",
            args=(pop_mean, pop_stdev),
            N=n,
            alternative="two-sided",
            mode="auto",
        )
    if np.isnan(results_tuple).any():
        return False

    results = handle_ks_test_results_tuple(results_tuple, n, format_p_for_gui)

    return results


def run_two_sample_ks_test(
    data1: NpArray64,
    data2: NpArray64,
    alternative_hypothesis: str,
    format_p_for_gui: bool = True,
) -> Dict:
    """
    Run a one- or two-sample KS test. It is assumed data1 and data2 are already
    checked n matches. It would be neater to call scipy function once and pass
    empty args for args, n. But do not want to pass unused args to scipy function.

    data1 - 1 x N array of raw data values (not a CDF)

    data2 - 1 x N array of raw data values for a 2-sample KS test, or None for one
            sample (compared to gaussian M, SD of data1)

    alternative_hypothesis - tails on the test, see scipy.stats.kstest()

    format_p_for_gui - format p value for display on GUI
    """
    n = len(data1)

    results_tuple = scipy.stats.kstest(rvs=data1, cdf=data2, alternative=alternative_hypothesis, mode="auto")
    results = handle_ks_test_results_tuple(results_tuple, n, format_p_for_gui)

    return results


def handle_ks_test_results_tuple(results_tuple: Tuple, n: Int, format_p_for_gui: bool) -> Dict:
    results = {
        "statistic": results_tuple[0],
        "pvalue": results_tuple[1],
        "n": n,
    }

    if format_p_for_gui:
        results["pvalue"] = format_p_value(results["pvalue"])

    return results


def format_p_value(p: NpArray64) -> str:
    """
    Format p value for display in GUI.

    Sometimes p values returned from scipy can be very small e.g. <1e-297.
    For convenient display cutoff at <1e-08.
    """
    if p >= 1e-08:
        format_p = "{0:.8f}".format(p).rstrip("0")
    elif p < 1e-08:
        format_p = "<1e-08"
    return format_p


def calc_empirical_cdf(data: NpArray64) -> Tuple[NpArray64, NpArray64]:
    """
    Calculate the empirical cumulative distribution function from a 1 x N sorted array.

    Note that np.unique outputs sorted data (smallest to largest)
    """
    x_values, counts = np.unique(data, return_counts=True)
    cumprob = np.cumsum(counts) / len(data)

    return x_values, cumprob


# --------------------------------------------------------------------------------------
# Stimulus Artefact Removal
# --------------------------------------------------------------------------------------


def get_stimulus_artefact_removed_data(
    data: NpArray64,
    ts: NpArray64,
    peak_detection_settings: Dict,
    forward_search_samples: Int,
    back_search_samples: Int,
    back_search_threshold: NpArray64,
    end_of_stimulus_perc_median: NpArray64,
    window_pad: Int,
) -> NpArray64:
    """
    Remove stimulus artefacts from all records in a dataset.

    Stimulus artefacts are characterised by two phases, a rapid rise
    caused by the stimulus injection, and slower decay that reflects
    membrane kinetics. Stimulus artefact detection works by first
    detecting the peak of the artefact, then separately handles
    removal of the forward (i.e. after the peak, slow) and 'backward'
    (i.e. before the peak, fast) states of the stimulus.

    For forward removal, a search period is used. The first
    sample within this search period that is different to
    the local median by less than some % of the local median is
    considered the end of the forward section.

    The backward section is assumed to be short and fast, and so a
    absolute derivative threshold is used. The first sample
    with a derivative higher than the derivative threshold is taken,
    and all samples between this sample and the peak is considered
    the rising phase.

    The rising (backward) and decaying (forward) sections of the
    stimulus are filled in with random samples from a Gaussian
    distribution with mean and std as estimated from data
    local to the stimulus.

    INPUTS:

        data : num records x num samples array of data
        ts : sampling interval

        peak_detection_settings : dictionary containing the following options
            "method" :  "derivative" or "abs_threshold". The method used to detect
                        the peaks of the stimulus artefact. This must be faster (if
                        using derivative threshold) or larger / smaller (if using
                        absolute threshold) that the maximum of the signals
                        (e.g. events) you are interested in.

            "derivative_threshold": derivatuve threshold used (delta per millisecond),
                                    used if peak detection method is "derivative".
                                    Note this is the absolute derivative.

            "abs_threshold": threshold used if peak detection method
                             is  "abs_threshold"0,

            "abs_threshold_direction" : "positive" or "negative", direction of
                                        peak if absolute threshold is used.

        forward_search_samples : number of samples to search for the end of
                                 the stimulus decay
        back_search_samples : number of searches to search for the start of
                              the stimulus rise
        back_search_threshold : absolute derivative threshold for the rising phase
        end_of_stimulus_perc_median : % of the median
        window_pad : optional pad to extend the forward and backward search periods.

    """
    data_stimulus_artefact_removed = utils.np_empty_nan(data.shape)

    for rec in range(data.shape[0]):
        data_stimulus_artefact_removed[rec, :] = remove_stimulus_artefact_from_record(
            data[rec, :].copy(),
            ts,
            peak_detection_settings,
            forward_search_samples,
            back_search_samples,
            back_search_threshold,
            end_of_stimulus_perc_median,
            window_pad,
        )
    return data_stimulus_artefact_removed


def remove_stimulus_artefact_from_record(
    data: NpArray64,
    ts: NpArray64,
    peak_detection_settings: Dict,
    forward_search_samples: Int,
    back_search_samples: Int,
    back_search_threshold: NpArray64,
    end_of_stimulus_perc_median: NpArray64,
    window_pad: Int = 1,
) -> NpArray64:
    """
    See get_stimulus_artefact_removed_data() for arguments.
    """
    main_peaks = calculate_stimulus_artefact_peaks(data, peak_detection_settings, ts)

    # For each peak, determine the local mean and median, and
    # forward and backward ranges to fill in.
    all_idx = []
    for peak_idx in main_peaks:
        left_edge, right_edge = get_artefact_peak_window_edges(data, peak_idx, forward_search_samples, window_pad)

        local_dat = data[np.arange(left_edge, right_edge)]

        diff_median_, median_ = calculate_local_statistics(local_dat)

        # Backward range
        back_search_start_idx = peak_idx - back_search_samples
        if back_search_start_idx < 0:
            back_search_start_idx = 0

        back_search_range = np.arange(back_search_start_idx, peak_idx + 1)
        back_search_range_diff_ms = np.abs(np.diff(data[back_search_range])) / (ts * 1000)

        over_thr_idx = back_search_start_idx + np.where(back_search_range_diff_ms > back_search_threshold)[0]
        if np.any(over_thr_idx):
            bad_back_idx = np.arange(np.min(over_thr_idx), peak_idx + 1)
        else:
            bad_back_idx = np.array([])

        # Forward Range
        perc_median = np.abs(diff_median_ * (end_of_stimulus_perc_median / 100))

        forward_search_range = data[peak_idx : peak_idx + forward_search_samples + 1]
        below_threshold = np.where(np.abs(forward_search_range - median_) < perc_median)[0]

        if np.any(below_threshold):
            forward_end_idx = peak_idx + np.min(np.where(np.abs(forward_search_range - median_) < perc_median)[0])
        else:
            forward_end_idx = data.size - 1

        bad_forward_idx = np.arange(peak_idx, forward_end_idx + 1)
        all_idx.append([bad_back_idx, bad_forward_idx, median_, diff_median_])

    # For all sections of data, fill in with Gaussian
    # random samples
    for idx_pair in all_idx:
        bad_back_idx, bad_forward_idx, median_, diff_median_ = idx_pair

        if np.any(bad_forward_idx):
            data[bad_forward_idx] = np.random.normal(median_, diff_median_, bad_forward_idx.size)

        if np.any(bad_back_idx):
            data[bad_back_idx] = np.random.normal(median_, diff_median_, bad_back_idx.size)

    return data


def calculate_stimulus_artefact_peaks(
    data: NpArray64, peak_detection_settings: Dict, ts: NpArray64
) -> NDArray[np.integer]:
    """
    Calculate the stimulus peaks for stimulus artefact removal.
    See get_stimulus_artefact_removed_data() for arguments.
    """
    if peak_detection_settings["method"] == "derivative":
        forward_search_range_diff_ms = np.abs(np.diff(data)) / (ts * 1000)

        # shift diff
        main_peaks = np.where(forward_search_range_diff_ms > peak_detection_settings["derivative_threshold"])[0] + 1

    elif peak_detection_settings["method"] == "abs_threshold":
        if peak_detection_settings["abs_threshold_direction"] == "positive":
            main_peaks = np.where(data > peak_detection_settings["abs_threshold"])[0]

        elif peak_detection_settings["abs_threshold_direction"] == "negative":
            main_peaks = np.where(data < peak_detection_settings["abs_threshold"])[0]

    if len(main_peaks) > 1:
        main_peaks = strip_repeats(data, main_peaks)

    return main_peaks


def calculate_local_statistics(local_dat: NpArray64) -> Tuple[NpArray64, NpArray64]:
    """
    Calculate the median and noise of the data for stimulus artefact removal.
    The noise is calculated as the median difference between consecutive samples,
    which is conservative and robust, performing well in the context of
    stimulus artefact removal
    """
    diffs = np.abs(np.diff(local_dat))
    diff_median_ = np.median(diffs)
    median_ = np.median(local_dat)
    return utils.fix_numpy_typing(diff_median_), utils.fix_numpy_typing(median_)


def strip_repeats(data: NpArray64, main_peaks: NDArray[np.integer]) -> NDArray[np.integer]:
    """
    Strip contiguous above threshold peaks from results.
    This is faster that scipy find peaks with prominence / min distances

    Find the +1 (rise) and -1 (fall) of the contiguous array.
    If the last samples are contiguous, there will be rising 1
    but no falling -1, so add last sample if so.

    Make idx from all contiguous samples and index out the max of the data
    then fill back in the single peaks.
    """
    bools = np.diff(np.r_[np.inf, main_peaks]) == 1
    tags = np.diff(bools.astype("int"))
    ups = np.where(tags == 1)[0]
    downs = np.where(tags == -1)[0]

    if not any(ups):
        return main_peaks

    if ups.size > downs.size:
        downs = np.r_[downs, tags.size]

    idx = [np.arange(up, down + 1) for up, down in zip(ups, downs)]

    main_peaks_repeated = np.array([idxs[np.argmax(data[main_peaks[idxs]])] for idxs in idx]).ravel()

    single_idx = np.ones(main_peaks.size, dtype=bool)
    single_idx[np.hstack(idx)] = False

    main_peaks = np.r_[main_peaks[main_peaks_repeated], main_peaks[single_idx]]

    return main_peaks


def get_artefact_peak_window_edges(
    data: NpArray64, peak_idx: Int, forward_search_samples: Int, pad: Int
) -> Tuple[Int, Int]:
    """
    Get the regions for local_dat for stimulus artefact removal. This
    includes the stimulus itself, which is handled by using robust
    statistics in calculate_local_statistics().
    """
    left_edge = peak_idx - pad
    right_edge = peak_idx + forward_search_samples + pad

    if left_edge < 0:
        left_edge = 0
        right_edge = pad * 2

    if right_edge > data.size:
        right_edge = data.size
        left_edge = peak_idx - (pad * 2)

    if left_edge < 0:
        left_edge = 0

    return left_edge, right_edge


# --------------------------------------------------------------------------------------
# Calculate Data Params
# --------------------------------------------------------------------------------------


def set_data_params(data_object: RawData) -> None:
    """TODO: this is kind of confusing"""
    data_object.min_max_time = np.vstack([data_object.time_array[:, 0], data_object.time_array[:, -1]]).T
    data_object.t_start = data_object.time_array[0][0]
    data_object.t_stop = data_object.time_array[-1][-1]
    data_object.num_recs = data_object.channel_1_data.shape[0]  # both im and vm array are same size
    data_object.num_samples = data_object.channel_1_data.shape[1]
    data_object.rec_time = data_object.min_max_time[0][1] - data_object.min_max_time[0][0]
    fs, ts = calc_fs_and_ts(data_object.time_array[0])
    data_object.fs = fs

    # overwrite these from neo for when data is manipulated (e.g. interp)
    data_object.ts = ts

    data_object.norm_first_deriv_data = None

    data_object.channel_1_data.setflags(write=False)
    data_object.channel_2_data.setflags(write=False)
    data_object.time_array.setflags(write=False)
    data_object.min_max_time.setflags(write=False)

    data_object.records_are_contiguous = check_if_record_time_is_contiguous(
        data_object.min_max_time, data_object.num_recs, data_object.ts
    )


def check_if_record_time_is_contiguous(min_max_time: NpArray64, num_recs: int, ts: NpArray64) -> bool:
    """
    Determine if recordings in a multi-record file are contiguous.
    This finds the difference in time between consecutive records,
    and checks it is equal to the time step (ts) (to 10 dp tolerance).
    """
    if num_recs == 1:
        return True

    if np.unique(min_max_time[:, 0]).size == 1:
        # a 'normalised' time recording
        return False

    diffs = min_max_time[1:, 0] - min_max_time[:-1, 1]

    is_contiguous = np.allclose(diffs, ts, rtol=0, atol=1e-10)

    return is_contiguous


# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------


def calc_fs_and_ts(time_: NpArray64) -> Tuple[NpArray64, ...]:
    ts = time_[1] - time_[0]
    fs = 1 / ts
    return fs, ts


def nearest_point_euclidean_distance(
    x1: NpArray64, timepoints: NpArray64, y1: NpArray64, datapoints: NpArray64
) -> NpArray64:
    """
    Return the euclidean distance between a (x1, y1) datapoint and the nearest
    true sample  with all data standardized to the variance.

    INPUTS:

        x1, y1: coordinates of a datapoint
        timepoints (x1...xn),
        datapoints (y1...yn): coordinates of datapoints to find the closest
    """
    time_std = np.std(timepoints)
    data_std = np.std(datapoints)
    distance = np.sqrt(((y1 - datapoints) / data_std) ** 2 + ((x1 - timepoints) / time_std) ** 2)

    return distance


def quick_get_time_in_samples(ts: NpArray64, timepoint: Union[float, NpArray64]) -> Int:
    return int(np.round(timepoint / ts))


def quick_get_samples_in_time(ts: NpArray64, num_samples: Int) -> NpArray64:
    return num_samples * ts


def generate_time_array(
    start: float,
    stop: float,
    num_samples: Int,
    known_ts: NpArray64,
    start_stop_time_in_ms: bool = False,
) -> NpArray64:
    """
    Generate time array based on start, stop time, N.

    INPUTS:

    start, stop : the start and stop times, in s or ms (if ms,
                  start_stop_time_in_ms must = True)

    num_samples : number of samples required in timeseries

    known_ts : the pre-calculated ts of the timeseries

    start_stop_time_in_ms : set True if passed start, stop values are in ms
    """
    if start_stop_time_in_ms:
        start /= 1000
        stop /= 1000

    time_array = np.linspace(start, (stop - known_ts), num_samples)

    if start_stop_time_in_ms:
        time_array *= 1000

    return time_array


def sort_dict_based_on_keys(dict_to_sort: Dict) -> Dict:
    """
    Sort a dict by the key, assuming the key is a str time
    at which a spike / event occurred.
    """
    sorted_dict = dict(sorted(dict_to_sort.items(), key=lambda item: float(item[0])))

    return sorted_dict


def get_conversion_to_pa_table() -> Dict:
    """
    Reference table for converting units to pA on file load (see importdata.py)
    """
    conversion_to_pa_table = {
        "fA": 1e-3,
        "nA": 1e3,
        "uA": 1e6,
        "mA": 1e9,
        "A": 1e12,
    }
    return conversion_to_pa_table


def get_conversion_to_mv_table() -> Dict:
    """
    Reference table for converting units to mV on file load (see importdata.py)
    """
    conversion_to_mv_table = {
        "pV": 1e-9,
        "nV": 1e-6,
        "uV": 1e-3,
        "V": 1e3,
    }
    return conversion_to_mv_table


def total_num_events(event_info: Union[List[Dict], Info], return_per_rec: bool = False) -> NDArray[np.integer]:
    """
    Calculate the total number of events in the event info
    (rec of dicts) or spkcnt / skinetics Info type
    either summed or per-rec
    """
    num_recs = len(event_info)
    per_rec = np.zeros(num_recs, dtype=int)
    for rec in range(num_recs):
        rec_info = event_info[rec]
        if rec_info != 0 and any(rec_info):
            per_rec[rec] = len(rec_info)

    events_to_return = np.sum(per_rec).astype(int) if not return_per_rec else per_rec

    return events_to_return


def calculate_summary_statistics(
    data: NpArray64,
    remove_nan: bool = False,
) -> Tuple[np.float64, np.float64, np.float64]:  # Tuple[np.float64, np.float64, np.float64]:
    """
    Calculate the mean, standard deviation and standard error for 1D data.
    """
    if remove_nan:
        data = data[~np.isnan(data)]

    mean = np.mean(data)
    stdev = np.std(data, ddof=1) if data.size > 1 else np.float64(np.nan)
    sterr = np.std(data, ddof=1) / np.sqrt(len(data)) if data.size > 1 else np.float64(np.nan)

    return mean, stdev, sterr


def update_event_numbers(event_info: List[Dict]) -> None:
    """
    Update the event number field on all events
    in event info, so they are correctly ordered.
    This needs to be run when event detection is first run,
    and any time event_info is changed,
    for example after event deletion / selection.
    """
    ev_num = 1
    for rec in range(len(event_info)):
        for event in event_info[rec].values():
            event["info"]["ev_num"] = ev_num
            ev_num += 1
