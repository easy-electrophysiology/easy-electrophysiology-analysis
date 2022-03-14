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
import copy

import numpy as np
from utils import utils
import scipy
from ephys_data_methods import voltage_calc
from statsmodels.stats import diagnostic

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Curve Fitting
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def monoexp_function(x, coefs):
    """
    Return monoexponential function of data x.

    INPUTS:
        x: time - time[0]
        coefs:  list / tuple of coefficients to use in order b0, b1, tau
    """
    b0, b1, tau = coefs

    return b0 + b1 * np.exp(-x / tau)

def biexp_decay_function(x, coefs):
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

def biexp_event_function(x, coefs):
    """
    Return biexponential function of data x. Biexponential function
    is in the form described in:

    Jonas, P. Major, G. Sakmann, B. (1993). Quantal components of unitary EPSCs at the mossy
    fibre synapose on CA3 pyramidal cells of rat hippocampus. J Physio. 472, 615-663.

    Default coefficients 0.5 ms, 5 ms for rise and decay respectively work well.

    INPUTS:
        x: time - time[0]
        coefs: list / tuple of coefficients to use in order b0, b1, rise, decay

    """
    b0, b1, rise, decay = coefs

    return b0 + b1 * (1 - np.exp(-x / rise)) * np.exp(-x / decay)

def triexp_decay_function(x, coefs):
    """
    Triexponential decay function of data x.

    INPUTS:
        x: time - time[0]
        coefs: list / tuple of coefficients to use in order b0, b1, tau1, b2, tau2, b3, tau3
    """
    b0, b1, tau1, b2, tau2, b3, tau3 = coefs

    return b0 + b1 * np.exp(-x / tau1) + b2 * np.exp(-x / tau2) + b3 * np.exp(-x / tau3)

def gaussian_function(x, coefs):
    a, mu, sigma = coefs
    gauss = a * np.exp(-0.5 * ((x - mu) / sigma)**2)
    return gauss

# Least Squares Cost Functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Note that scipy.optimize.least_squares expects the cost-function to be residuals not the sum of squared error (it handles this under the hood)

def monoexp_least_squares_cost_function(coefs, x, y):
    yhat = monoexp_function(x,
                            coefs)
    return yhat - y

def biexp_event_least_squares_cost_function(coefs, x, y):
    yhat = biexp_event_function(x,
                                coefs)
    return yhat - y

def biexp_decay_least_squares_cost_function(coefs, x, y):
    yhat = biexp_decay_function(x,
                                coefs)
    return yhat - y

def triexp_decay_least_squares_cost_function(coefs, x, y):
    yhat = triexp_decay_function(x,
                                 coefs)
    return yhat - y

def gaussian_least_squares_cost_function(coefs, x, y):
    y_hat = gaussian_function(x, coefs)
    return y_hat - y

# Fit Curve ------------------------------------------------------------------------------------------------------------------------------------------

def fit_curve(analysis_name,
              x_to_fit,
              y_to_fit,
              direction,
              initial_est=None,
              bounds=None,
              normalise_time=True):
    """
    Fit curve of type monoexponential ("monoexp"), biexponential decay ("biexp_decay"),
    biexponential event ("biexp_event") and triponential ("triexp"), or "gaussian"

    INPUTS:
        analysis name: name of curve to fit (above; str)
        x_to_fit, y_to_fit: 1 x n vector of times (x) and data (y) to fit.
        direction: -1 for a negative "event" and 1 for a positive "event"
        initial_est: tuple in the form (est c1, est c2, est c3, ... cn) for starting estimates of the coefficients (see get_initial_est())
        bounds: tuple in the form ((min c1, min c2, min c3, ... min cn), (max c1, max c2, max c3, ... max cn))
        normalise_time: subtract the first timepoint from x_to_fit so that the time starts at zero.
                        Setting this to false can severely comprimise fitting.

    OUTPUT:
        coefs: coefficients for the least-squares fit
        fit: the function generated wth the coefs

    """
    if normalise_time:
        x_to_fit = x_to_fit - x_to_fit[0]  # x_to_fit is read only (no -=)

    if initial_est is None:
        initial_est = get_initial_est(analysis_name, direction, x_to_fit, y_to_fit)

    if bounds is None:
        bounds = get_curve_fit_bounds(analysis_name)

    try:
        coefs = scipy.optimize.least_squares(get_least_squares_fit_function(analysis_name), x0=initial_est, args=(x_to_fit, y_to_fit), bounds=bounds)
        coefs = coefs.x
    except:
        return False, False, False

    func = get_fit_functions(analysis_name)
    fit = func(x_to_fit, coefs)
    r2 = calc_r2(y_to_fit, fit)

    return coefs, fit, r2

def calc_r2(y, y_hat):
    """
    Calculate the r-squared for a fit to data
    """
    y_bar = np.mean(y)
    SST = np.sum((y - y_bar)**2)
    RSS = np.sum((y - y_hat)**2)
    r2 = 1 - RSS / SST
    return r2

# Bounds and Initial Estimates -----------------------------------------------------------------------------------------------------------------------

def get_curve_fit_bounds(analysis_name):
    """
    Set default bounds on curve fitting. For any Tau coefficients, set the start boundary
    to zero (avoid strange occurrences of negative tau when fit is poor).
    """
    bounds = {"monoexp": ((-np.inf, -np.inf, 0),
                          (np.inf, np.inf, np.inf)),
              "biexp_decay": ((-np.inf, -np.inf, 0, -np.inf, 0),
                              (np.inf, np.inf, np.inf, np.inf, np.inf)),
              "biexp_event": ((-np.inf, -np.inf, 0, 0),
                              (np.inf, np.inf, np.inf, np.inf)),
              "triexp":  ((-np.inf, -np.inf, 0, -np.inf, 0, -np.inf, 0),
                          (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)),
              "gaussian": ((-np.inf, -np.inf, -np.inf),
                           (np.inf, np.inf, np.inf)),
              }

    return bounds[analysis_name]

def get_initial_est(analysis_name,
                    direction,
                    x_to_fit,
                    y_to_fit):
    """
    Get the default initial estimates for Curve Fitting functions.

    see fit_curve() for input and output. Exponential functions ("monoexp", "biexp_decay", "triexp")
    share the same estimate generation with the tau is tai weighted across coefficients.
    "biexp_event" uses rise and deay from Jonas et al., 1993.
    """
    if analysis_name == "monoexp":
        est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit,
                                                                              y_to_fit)
        initial_est = (est_offset, est_slope, est_tau)

    elif analysis_name == "biexp_decay":
        est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit,
                                                                              y_to_fit)
        initial_est = (est_offset, est_slope, est_tau * 0.1, est_slope, est_tau * 0.9)

    elif analysis_name == "biexp_event":
        initial_est = get_biexp_event_initial_est(x_to_fit, y_to_fit, direction)

    elif analysis_name == "triexp":
        est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit,
                                                                              y_to_fit)
        initial_est = (est_offset, est_slope, est_tau / 3, est_slope, est_tau / 3, est_slope, est_tau / 3)

    elif analysis_name == "gaussian":
        initial_est = get_gaussian_initial_est(x_to_fit, y_to_fit)

    return initial_est

def get_exponential_function_initial_est(x_to_fit,
                                         y_to_fit):
    """
    Get starting estimates for exponential functions.
    See fit_curve() for inputs.

    Calculate offset as mean, slope as amplitude and tau as
    time to half-amplitude / log(2).
    """
    start = y_to_fit[0]
    end = y_to_fit[-1]

    est_offset = np.mean(y_to_fit)
    est_slope = (start - end)

    half_amp = start + ((end - start) / 2)
    rough_midpoint = np.argmin(np.abs(half_amp - y_to_fit))
    est_tau = (x_to_fit[rough_midpoint] / np.log(2))

    return est_offset, est_slope, est_tau

def get_biexp_event_initial_est(x_to_fit,
                                y_to_fit,
                                direction):
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

def get_gaussian_initial_est(x, y):
    """
    Return the initial estimate of parameters for fitting gaussian.
    """
    a = np.max(y)
    mu = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mu)**2) / np.sum(y))
    return a, mu, sigma

# Convenience Functions ------------------------------------------------------------------------------------------------------------------------------

def get_least_squares_fit_function(analysis_name):
    """
    Convenient access for residual fitting functions that return y - yhat used in scipy curve fit. Note that scipy.optimize.least_squares
    expects the cost-function to be residuals (not sum of squared error, it handles this under the hood).
    """
    least_squares_fit_functions = {"monoexp": monoexp_least_squares_cost_function,
                                   "biexp_decay": biexp_decay_least_squares_cost_function,
                                   "biexp_event": biexp_event_least_squares_cost_function,
                                   "triexp": triexp_decay_least_squares_cost_function,
                                   "gaussian": gaussian_least_squares_cost_function,
                                   }
    return least_squares_fit_functions[analysis_name]

def get_fit_functions(analysis_name):
    """
    Convenient access for fitting functions.
    """
    fit_functions = {"monoexp": monoexp_function,
                     "biexp_decay": biexp_decay_function,
                     "biexp_event": biexp_event_function,
                     "triexp": triexp_decay_function,
                     "gaussian": gaussian_function}

    return fit_functions[analysis_name]

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Spike Kinetics - Thresholding
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_threshold(rec_first_deriv,
                        rec_second_deriv,
                        rec_third_deriv,
                        start_idx,
                        peak_idx,
                        cfgs):
    """
    Calculate AP threshold using first and third derivative methods,
    or the phase-space methods described in:

    Sekerli, M. Del Negro, C. A. Lee, R. H and Butera, R. J. (2004). Estimating action potential thresholds from
    neuronal time-series: new metrics and evaluation of methodologies. IEEE Trans Biomed Eng. 51(9): 1665-1672.
    """
    first_deriv = rec_first_deriv[start_idx:peak_idx]
    second_deriv = rec_second_deriv[start_idx:peak_idx]
    third_deriv = rec_third_deriv[start_idx:peak_idx]

    if (np.array([first_deriv.size, second_deriv.size, third_deriv.size]) == 0).any():
        return False

    if cfgs.skinetics["thr_method"] == "first_deriv":
        idx = first_derivative_threshold(cfgs,
                                         first_deriv)

    if cfgs.skinetics["thr_method"] == "third_deriv":
        idx = third_derviative_threshold(cfgs,
                                         third_deriv)

    if cfgs.skinetics["thr_method"] == "method_I":
        idx = method_i_threshold(cfgs,
                                 first_deriv,
                                 second_deriv)

    if cfgs.skinetics["thr_method"] == "method_II":
        idx = method_ii_threshold(cfgs,
                                  first_deriv,
                                  second_deriv,
                                  third_deriv)

    if cfgs.skinetics["thr_method"] == "leading_inflection":
        idx = leading_inflection_threshold(first_deriv)

    if cfgs.skinetics["thr_method"] == "max_curvature":
        idx = max_curvature_threshold(first_deriv,
                                      second_deriv)
    return idx

def first_derivative_threshold(cfgs,
                               first_deriv):

    if cfgs.skinetics["first_deriv_max_or_cutoff"] == "max":
        idx = np.nanargmax(first_deriv)
    else:
        cutoff = cfgs.skinetics["first_deriv_cutoff"]
        idx = (first_deriv > cutoff).argmax()
    return idx

def third_derviative_threshold(cfgs,
                               third_deriv):

    if cfgs.skinetics["third_deriv_max_or_cutoff"] == "max":
        idx = np.nanargmax(third_deriv)
    else:
        cutoff = cfgs.skinetics["third_deriv_cutoff"]
        idx = (third_deriv > cutoff).argmax()
    return idx

def method_i_threshold(cfgs,
                       first_deriv,
                       second_deriv):

    g = second_deriv / first_deriv
    lower_bound = cfgs.skinetics["method_I_lower_bound"]
    g[first_deriv <= lower_bound] = np.nan

    if np.all(np.isnan(g)):
        return False

    idx = np.nanargmax(g)
    return idx

def method_ii_threshold(cfgs,
                        first_deriv,
                        second_deriv,
                        third_deriv):

    h = ((third_deriv * first_deriv) - (second_deriv ** 2)) / (first_deriv ** 3)
    lower_bound = cfgs.skinetics["method_II_lower_bound"]
    h[first_deriv <= lower_bound] = np.nan

    if np.all(np.isnan(h)):
        return False

    idx = np.nanargmax(h)
    return idx

def leading_inflection_threshold(first_deriv):
    idx = np.nanargmin(first_deriv)
    return idx

def max_curvature_threshold(first_deriv,
                            second_deriv):

    k = second_deriv * (1 + first_deriv ** 2) ** (-3 / 2)
    idx = np.nanargmax(k)
    return idx


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Spike and Event Kinetics
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_ahp(data,
                  start_time,
                  stop_time,
                  vm,
                  time_,
                  peak_idx,
                  thr_vm):
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
    ahp_idx = np.argmin(vm[ahp_search_start_idx:
                           ahp_search_stop_idx + 1])
    ahp_idx = ahp_search_start_idx + ahp_idx
    ahp = vm[ahp_idx] - thr_vm

    return time_[ahp_idx], vm[ahp_idx], ahp_idx, ahp

def calculate_fwhm(start_to_peak_time,
                   start_to_peak_data,
                   peak_to_end_time,
                   peak_to_end_data,
                   half_amp,
                   interp):
    """
    Calculate the full-width at half maximum (half width) or an AP or PSP (hereby referred to as "event").
    First find the "true" half width as amplitude / 2. Then find the nearest real sample to this point
    for both the AP rise and decay and subtract to find the time between them.
    Interpolation significantly increases the accuracy.

    start_to_peak_time: a 1 x t array containing timepoints for an event from the threshold to peak
    start_to_peak_data: a 1 x t array containing datapoints as above
    peak_to_end_time: a 1 x t array containing timepoints for an even from the peak to the end point (e.g. fAHP)
    peak_to_end_data: a 1 x t array containing datapoints as above
    half_amp: scalar value, the amplitude / 2 representing the true half-maximum data
    interp: bool, interpolate data to 200 kHz before calculating fwhm.
    """
    orig_rise_mid_idx = np.abs(half_amp - start_to_peak_data).argmin()  # save the un-interpred index to calculate average event from
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

    return rise_midtime, rise_midpoint, fall_midtime, fall_midpoint, fwhm, orig_rise_mid_idx, orig_decay_mid_idx

def calc_rising_slope_time(slope_data,
                           slope_time,
                           min_,
                           max_,
                           min_cutoff_perc,
                           max_cutoff_perc,
                           interp):
    """
    Calculate the rise-time of an AP or PSP  (hereby referred to as "event").
    Find the time from the threshold of an event to the peak, or some percentage of the amplitude
    specified the user e.g. the 10-90 rise time.

    INPUT:
        slope_data: 1 x t array of datapoints (e.g. Vm) from threshold to peak
        slope_time: 1 x t array of timepoints as above
        min_: the minimum (i.e. threshold) value
        max_: the maximum (i.e. peak) value
        min_cutoff_perc: the scalar cutoff value (minimum) to measure the rise_time from (i.e. 10 for the 10-90 rise time)
        max_cutoff_perc: the scalar cutoff value (maximum) to measure the rise time to (i.e. 90 in the 10-90 rise time)
        interp: bool, whether to interpolate prior to calculation

    NOTE: the reason min_ and max_ are not taken as the min or max values of the sloep_data array is in case
          these values do not correspond to the raw data e.g. if the baseline data value has been averaged over prior points.

    TODO: This function is similar to calc_falling_slope_time, though with some key differences.
          Revisit to see if it is worthwhile to factor out differences and combine into one function.
    """
    if interp:
        slope_data, slope_time = twohundred_kHz_interpolate(slope_data, slope_time)

    amplitude = max_ - min_
    min_cutoff = min_ + amplitude * (min_cutoff_perc/100)
    max_cutoff = min_ + amplitude * (max_cutoff_perc / 100) if max_cutoff_perc != 100 else max_
    min_cutoff_idx = np.abs(slope_data - min_cutoff).argmin()
    max_cutoff_idx = np.abs(slope_data - max_cutoff).argmin()

    min_time = slope_time[min_cutoff_idx]
    min_data = slope_data[min_cutoff_idx]
    max_time = slope_time[max_cutoff_idx]
    max_data = slope_data[max_cutoff_idx]
    rise_time = max_time - min_time

    return min_time, min_data, max_time, max_data, rise_time

def calc_falling_slope_time(slope_data,
                            slope_time,
                            max_,
                            min_,
                            max_cutoff_perc,
                            min_cutoff_perc,
                            interp):
    """
    Calculate the time constant of a falling slope (this is rise time for events and decay time
    for APs. See calc_rising_slope_time or details.

    INPUT
        slope_data:  timeseries from max to min of the Im or Vm (e.g. AP peak to fAHP)
        slope_time: time units of the period (s)
        max_: max value of the slope of interest (pA for Im and mV for Vm). Almost always but not necessarily
              the max value of the timeseries. e.g. AP peak or negative event baseline
        min_: min value of the slope of interst (e.g. AP fAHP or negative event peak)
        max_cutoff_perc: cutoff percentage for the minimum value
        min_cutoff_perc: cutoff percentage for the maximum value
        interp: bool, to interp 200 kHz
    """
    if interp:
        slope_data, slope_time = twohundred_kHz_interpolate(slope_data, slope_time)

    fall_amplitude = max_ - min_
    max_cutoff = max_ - fall_amplitude * (max_cutoff_perc/100)
    min_cutoff = max_ - fall_amplitude * (min_cutoff_perc/100) if min_cutoff_perc != 100 else min_
    max_cutoff_idx = np.abs(slope_data - max_cutoff).argmin()
    min_cutoff_idx = np.abs(slope_data - min_cutoff).argmin()

    max_time = slope_time[max_cutoff_idx]
    max_data = slope_data[max_cutoff_idx]
    min_time = slope_time[min_cutoff_idx]
    min_data = slope_data[min_cutoff_idx]
    decay_time = min_time - max_time

    return max_time, max_data, min_time, min_data, decay_time

def calculate_max_slope_rise_or_decay(time_,
                                      data,
                                      start_idx,
                                      stop_idx,
                                      window_samples,
                                      ts,
                                      smooth_settings,
                                      argmax_func):
    """
    Calculate the maximum slope over a period of data, size 1 ... N. The regression over M points is calculated from
    1:M ... N-M:M. The regression is not calculated across points N-M:N. If M is larger than N, no max slope is calculated.

    For N = 2, slope is calculated as rise over run for performance increase.

    INPUTS:
        time_: all timepoints in a record, 1...Z
        data:  all datapoints in a record, 1...Z
        start_idx: start index for the period to search the max slope
        stop_idx: stop index for the period to search the max slope (1...N)
        window_samples: number of samples to calculate the regression over, 1...M
        ts: sample spacing in seconds
        smooth_settings: dict with settings on the smoothing: {"on": bool_, "num_samples": num samples to smooth}
        argmax_func: np.max() or np.min() depending on if the slope is positive or negative.

    NOTES:
        It was attempted to improve speed using statsmodel rolling_ols but it made little difference.
    """
    stop_idx -= window_samples
    n_samples = stop_idx - start_idx + 1

    if n_samples < 1:
        return [np.nan], [np.nan], [np.nan]

    if smooth_settings["on"]:
        data = voltage_calc.quick_moving_average(data,
                                                 smooth_settings["num_samples"])
    if window_samples == 2:

        idx = np.arange(start_idx,
                        stop_idx + 1)
        time_step_ms = ts * 1000

        windowed_deriv = (data[idx + 1] - data[idx]) / time_step_ms
        max_slope_idx = argmax_func(windowed_deriv)
        max_slope_ms = windowed_deriv[max_slope_idx]
        max_slope_idx += start_idx

        fit_time = np.array([time_[max_slope_idx],
                             time_[max_slope_idx + 1]])
        fit_data = np.array([data[max_slope_idx],
                             data[max_slope_idx + 1]])
    else:

        idx = np.tile(np.arange(start_idx, start_idx + window_samples),
                      (n_samples, 1)) + np.atleast_2d(np.arange(n_samples)).T
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

def twohundred_kHz_interpolate(vm,
                               time_):
    """
    Interpolate to 200 kHz with linear interpolation.
    """
    fs = calc_fs(time_)
    interp_factor = 200000 / fs
    vm = interpolate_data(vm, time_, "linear", interp_factor, 0)
    time_array = interpolate_data(time_, time_, "linear", interp_factor, 0)

    return vm, time_array

def area_under_curve_ms(y, ts):
    """
    Calculate the area under the curve of data. Note that it is assumed data is already baseline subtracted if required.

    y = 1 x N data array (baseline subtracted if required)
    ts - sampling step in seconds

    OUTPUT
        area_under_curve - area under the curve in units Data units x ms (e.g. pA ms)
        area_under_curve_time_ms - time period over which the AUC was calculated
    """
    ts_ms = ts * 1000

    area_under_curve_time_ms = ts_ms * (y.size - 1)

    area_under_curve = np.trapz(y, dx=ts_ms)

    return area_under_curve, area_under_curve_time_ms

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Data Tools
# ----------------------------------------------------------------------------------------------------------------------------------------------------

# Filter
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def filter_data(data,
                fs,
                filter_,
                order,
                cutoff_hz,
                low_or_highpass,
                axis):
    """
    Filter data with bessel or butterworth digital filter (wrapper around Scipy functions)

    INPUT: data: data to filter
           fs:                sampling frequency (Hz)
           filter_:           "bessel" or "butter"
           order:             order of filter. Note that because filtfilt() is zero-phase data is filtered twice is forward and reverse direction, this effective
                              order is doubled. e.g. if order = 1, true order of the filter will be 2.
           cutoff_hz:         the frequecy cutoff in Hz
           low_or_highpass:  "lowpass" or "highpass"
           axis:              axis of array to filter along
    """
    if filter_ == "bessel":
        b, a = get_bessel(fs, order, cutoff_hz, low_or_highpass)
    elif filter_ == "butter":
        b, a = get_butterworth(fs, order, cutoff_hz, low_or_highpass)

    filtered_data = scipy.signal.filtfilt(b, a, data, axis=axis)
    return filtered_data

def get_bessel(fs, order, cutoff_hz, low_or_highpass):
    nyquist = fs / 2
    b, a = scipy.signal.bessel(N=order, Wn=cutoff_hz/nyquist, btype=low_or_highpass)
    return b, a

def get_butterworth(fs, order, cutoff_hz, low_or_highpass):
    nyquist = fs / 2
    b, a = scipy.signal.butter(N=order, Wn=cutoff_hz/nyquist, btype=low_or_highpass)
    return b, a

def get_fft(y,
            detrend,
            fs,
            cut_down):
    """
    Returns the fast-Fourier transformation of the signal y, wrapper for scipy function.

    INPUT: y:       data to fft
           detrend: optional bool to detrend signal prior to transformation
           fs:      if sampling frequency in Hz provided, returns the frequencies corresponding to
                    power at Y.
           cut_down: if true, negative frequencies will be cut out and the magnitude doubled.
                     An offset (10 default) is cut from the start of the spectrum so low frequency noise
                     doesn't dominate the plot.

    OUTPUTS: a dictionary containing:   out["Y"]: fft of y
                                        out["freqs"]: sample frequencies
                                        out["cutx"]: if cut_down is selected, this is the offset to cut low frequency noise
                                        out["N"]: number of samples in the cut-down spectra
    """
    out = {"Y": None, "freqs": None,
           "cutx": None, "N": None}

    n_samples = len(y)
    if detrend:
        y = scipy.signal.detrend(y)

    Y = scipy.fft.fft(y) / n_samples
    Y = np.abs(Y)
    freqs = scipy.fft.fftfreq(n_samples,  (1 / fs))

    if cut_down:
        offset = 10
        N = int(len(y) / 2)
        Y = Y[offset: N] * 2
        freqs = freqs[offset: N]
        out["cutx"], out["N"] = offset, N

    out["Y"], out["freqs"] = Y, freqs
    return out

# Interpolate and Downsample
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def interpolate_data(data_to_interp,
                     time_array,
                     interp_method,
                     interp_factor,
                     axis):
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

    time_interp = np.linspace(time_array[0],
                              time_array[-1],
                              int(new_sample_num))

    interp_data = interp_f(time_interp)

    return interp_data

def downsample_data(data,
                    downsample_factor,
                    filter_opts=None):
    """
    Downsample data by lower-pass filtered followed by linear interpolation.
    Slices along axis=1.

    INPUTS:
         data: data to downsample (matrix record X sample)
         downsample_factor: intger factor to downsample by
         filter_opts: optional dictionary - if provided data is low-pass filtered under the nyquist frequency
                                            of the downsampled data before downsampling

                     keys: data_fs: sampling frequency of the data
                           filter:  type of filter (see filter_data) (e.g. "bessel", "butter")
                           filter_order: order of the filter (see filter_data)

     OUTPUT:
        downsampled_data
    """
    if filter_opts:
        downsampled_fs = filter_opts["data_fs"] / downsample_factor
        lowpass_cutoff = np.floor(downsampled_fs / 2)

        data = filter_data(data,
                           filter_opts["data_fs"],
                           filter_opts["filter"],
                           filter_opts["filter_order"],
                           lowpass_cutoff,
                           "lowpass", axis=1)

    downsample_slice = [slice(None, None, None),
                        slice(None, None, downsample_factor)]
    downsampled_data = data[tuple(downsample_slice)]

    return downsampled_data

# Detrend --------------------------------------------------------------------------------------------------------------------------------------------

def detrend_data(x,
                 y,
                 poly_order):
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


def fit_polynomial(x,
                   y,
                   poly_order):
    """
    Convenience function for fitting polynomial to multiple rows at once, or a single 1D array.
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

        fit = utils.np_empty_nan((np.shape(y)))
        for irow in range(y.shape[0]):
            fit[irow, :] = np.polyval(coefs[irow, :], x)

    return fit

# Cut trace length and normalise time
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def cut_trace_length_time(time_method, time_array, new_start_sample, new_stop_sample):
    """
    Cut down the trace length between user-specified times. For mutli-record files, this provides
    different methods the new time array can be handled (see time_method). Note the cut is not upper
    bound inclusive.

    INPUTS:
        time_method -  "raw_times": keep the raw times e.g. if a record is cut from [0-1, 1-2] to 0.2 - 0.8 the times will be [0.2-0.8, 1.2-1.8]

                       "cumulative": the new time any start offset is removed and the time increases cumulatively. e.g. [0-1, 1-2] to 0.2 - 0.8 will be [0-(0.6-ts), 0.6-(1.2-ts)]

                       "normalised": the times will be the same across all records with any start offset removed e.g. [0-1, 1-2] to 0.2 - 0.8 will be [0-(0.6-ts), 0-(0.6-ts)]

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

def cut_trace_and_normalise_time(time_array, new_start_sample, new_stop_sample):
    """
    see cut_trace_length_time()
    """
    norm_time_array = copy_and_normalise_time_array(time_array)
    norm_time_array = norm_time_array[:, new_start_sample:new_stop_sample]
    norm_time_array = norm_time_array - norm_time_array[0][0]

    return norm_time_array

def copy_and_normalise_time_array(time_array):
    """
    Normalise the time across records so it is the same for each record (use the first row and
    repeat across all records).
    """
    time_array = copy.deepcopy(time_array)
    time_array[1:None, :] = time_array[0, :]
    return time_array

# Reshape Data
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def reshape(data, num_samples, num_records):
    reshaped_data = np.reshape(data,
                              [num_records, int(num_samples / num_records)])

    return reshaped_data

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Cumulative Probability
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def process_frequency_data_for_cum_prob(event_times, rec_starting_ev_idx):
    """
    Process event times for cumulative probability graphs.

    event_times = list of event times (float)

    rec_starting_ev_idx = idx of event times for the analysed record

    Start from zero to find the event-interval for the first event, all inter-event intervals.
    Also, return labels of the event numbers.
    """
    assert np.array_equal(event_times, np.sort(event_times)), "process_frequency_data_for_cum_prob times not sorted"

    data = np.diff(event_times)

    sort_idx = []
    for ev_num in range(1, len(event_times)):
        sort_idx.append(str(ev_num + rec_starting_ev_idx) + "-" + str(ev_num + rec_starting_ev_idx + 1))

    return data, sort_idx


def process_amplitude_for_frequency_table(amplitudes, sort_method):
    """
    Return sorted, absolute amplitudes for cum prob analysis,
    and event numbers or labelling.
    """
    data = np.sort(np.abs(amplitudes))
    sort_idx = np.argsort(np.abs(amplitudes)) + 1

    if sort_method == "event_num":
        event_num_sort_idx = np.argsort(sort_idx)
        data = data[event_num_sort_idx]
        sort_idx = sort_idx[event_num_sort_idx]

    return data, sort_idx

def process_non_negative_param_for_frequency_table(non_neg_param, sort_method):
    """
    Return sorted non-negative parameter (decay data)
    for cum prob binning, and array of event labels.
    """
    data = np.sort(non_neg_param)
    sort_idx = np.argsort(non_neg_param) + 1

    if sort_method == "event_num":
        event_num_sort_idx = np.argsort(sort_idx)  # TODO: OWN METHOD
        data = data[event_num_sort_idx]
        sort_idx = sort_idx[event_num_sort_idx]

    return data, sort_idx

def get_num_bins_from_settings(data, settings, parameter):
    """
    Calculate the bin number based on user settings.
    see calc_cumulative_probability_or_histogram() for inputs
    """
    binning_method = settings["binning_method"]
    n_samples = len(data)

    if binning_method == "auto":
        try:
            num_bins = len(np.histogram_bin_edges(data, bins="auto"))
        except MemoryError:  # can occur with this function when bin sizes are very small.
            num_bins = len(np.histogram_bin_edges(data, bins="sqrt"))

    elif binning_method == "custom_binnum":
        num_bins = int(settings["custom_binnum"]) if settings["custom_binnum"] <= n_samples else n_samples

    elif binning_method == "custom_binsize":
        num_bins = np.round(np.ptp(data) / settings["custom_binsize"][parameter]).astype(int)

    elif binning_method == "num_events_divided_by":
        num_bins = np.round(len(data) / settings["divide_by_number"]).astype(int)

    if num_bins > len(data) or num_bins == 0:
        num_bins = len(data)
    elif num_bins < 2:
        num_bins = 2

    return num_bins

def calc_cumulative_probability_or_histogram(data, settings, parameter, legacy_bin_sizes):
    """
    Return the histogram or cumulative probabilities of 'data'.

    INPUTS:
        data: n x 1 array of values (here Event parameter values e..g inter-event intervals)
        settings: dict of settings with fields:
            'plot_type': 'cum_prob' or 'hist'
            'binning method': 'auto' (numpy implementation), 'custom_binnum', 'custom_binsize', 'num_events_divided_by'
            'custom_binnum': number of bins to divide the data into
            'custom_binsize': data range of bins, same as max(data) / binnum
            'divide_by_number': divisor if 'num_events_divided_by' option is chosen
            'x_axis_display': 'bin_centers', 'left_edge', 'right_edge'

        paramter: event paramter beign analysed (frequency, amplitude, decay_tay, decay_percent..
                                                 (see configs, frequency_data_options["custom_binsize"]))
        legacy_bin_sizes: bool
    """
    cum_prob_or_hist = settings["plot_type"]

    if len(data) < 1:
        return False, False, False, False

    # Calculate number of bins based on user settings
    num_bins = get_num_bins_from_settings(data, settings, parameter)

    # Use bin range from 0 - max of data, previously a little extra padding was added
    # similar to scipy implementation, but using the data is more interpretable for end user.
    if legacy_bin_sizes:
        s = (np.max(data) - np.min(data)) / (2. * (num_bins - 1.))
    else:
        s = 0
    max_ = np.max(data) + s
    limits = (np.min(data),
              max_)

    # Calculate the y values and bin edges for the histogram / cumulative probability
    if cum_prob_or_hist == "cum_prob":
        y_values, bin_edges, binsize = calc_cumulative_probability(data, num_bins, limits)

    elif cum_prob_or_hist == "hist":
        y_values, bin_edges, binsize = calc_histogram(data, num_bins, limits)

    # format the x values based on user settings
    x_values = format_bin_edges(bin_edges, settings["x_axis_display"])

    return y_values, x_values, binsize, num_bins

def calc_histogram(data, num_bins, limits):
    """
    """
    info = np.histogram(data,
                        bins=num_bins,
                        range=limits)
    y_values = info[0]
    bin_edges = info[1]
    binsize = bin_edges[1] - bin_edges[0]

    return y_values, bin_edges, binsize

def calc_cumulative_probability(data, num_bins, limits):
    """
    Calculate the binned cumulative probability from data processed
    for cum prob analysis.

    INPUT:
        data: data processed for cum prob analysis (see above functions)
        num_bins: number of bins to divide the data into
        limits: start / end limits for the bins

    OUTPUT:
        cdf: 1 x bin cumulative probabilities
        bin_edges, binsize - generated bin edges and binsizes

    NOTES: see https://stackoverflow.com/questions/10640759/how-to-get-the-cumulative-distribution-function-with-numpy
           for an alternative 'continuous' method of calculation. Binning is preferred here as it is more flexible.
    """
    counts, bin_edges, binsize = calc_histogram(data, num_bins, limits)

    pdf = counts / np.sum(counts)
    cdf = np.cumsum(pdf)

    return cdf, bin_edges, binsize

def format_bin_edges(bin_edges, x_axis_display):
    """
    Format n + 1 array of bin edges for a histogram or cumulative probability plot.

    INPUTS:
        'bin edges': n + 1 list of bins, n x 1 array

        Format the bin edges as:
        x_axis_display (str):
            'bin_centre': take the center of each bin (e.g bin 1 = 0 - 2, bin center = 1)
            'left_edge': take the left edge of the bins (i.e. the mimum value, bin 1 = 0 - 2, left edge = 0)
            'right_edge': take the right edge of the bins (i.e. the maximum value, bin 1 = 0 - 2, right edge = 2)

    """
    if x_axis_display == "bin_centre":
        x_values = get_bin_centers(bin_edges)
    elif x_axis_display == "left_edge":
        x_values = bin_edges[0:-1]
    elif x_axis_display == "right_edge":
        x_values = bin_edges[1:]

    return x_values

def get_bin_centers(bin_edges):
    return bin_edges[0:-1] + (np.diff(bin_edges) / 2)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def run_ks_test(data1, method, pop_mean=None, pop_stdev=None, format_p_for_gui=True):
    """
    Run a one- or two-sample KS test. It is assumed data1 and data2 are already
    checked n matches. It would be neater to call scipy function once and pass
    empty args for args, n. But do not want to pass unused args to scipy function.

    data1 - 1 x N array of raw data values (not a CDF)

    data2 - 1 x N array of raw data values for a 2-sample KS test, or None for one sample (compared to gaussian M, SD of data1)

    alternative_hypothesis - tails on the test, see scipy.stats.kstest()

    format_p_for_gui - format p value for display on GUI
    """
    n = len(data1)

    if method == "lilliefors":

        results_tuple = diagnostic.lilliefors(data1)

    elif method == "user_input_population":

        results_tuple = scipy.stats.kstest(rvs=data1,
                                           cdf="norm", args=(pop_mean, pop_stdev), N=n,
                                           alternative="two-sided",  mode="auto")
    if np.isnan(results_tuple).any():
        return False

    results = handle_ks_test_results_tuple(results_tuple, n, format_p_for_gui)

    return results

def run_two_sample_ks_test(data1, data2, alternative_hypothesis, format_p_for_gui=True):
    """
    Run a one- or two-sample KS test. It is assumed data1 and data2 are already
    checked n matches. It would be neater to call scipy function once and pass
    empty args for args, n. But do not want to pass unused args to scipy function.

    data1 - 1 x N array of raw data values (not a CDF)

    data2 - 1 x N array of raw data values for a 2-sample KS test, or None for one sample (compared to gaussian M, SD of data1)

    alternative_hypothesis - tails on the test, see scipy.stats.kstest()

    format_p_for_gui - format p value for display on GUI
    """
    n = len(data1)

    results_tuple = scipy.stats.kstest(rvs=data1,
                                       cdf=data2,
                                       alternative=alternative_hypothesis,
                                       mode="auto")
    results = handle_ks_test_results_tuple(results_tuple, n, format_p_for_gui)

    return results

def handle_ks_test_results_tuple(results_tuple, n, format_p_for_gui):

    results = {
        "statistic": results_tuple[0],
        "pvalue": results_tuple[1],
        "n": n,
    }

    if format_p_for_gui:
        results["pvalue"] = format_p_value(results["pvalue"])

    return results

def format_p_value(p):
    """
    Format p value for display in GUI.

    Sometimes p values returned from scipy can be very small e.g. <1e-297. For convenient display
    cutoff at <1e-08.
    """
    if p >= 1e-08:
        format_p = "{0:.8f}".format(p).rstrip("0")
    elif p < 1e-08:
        format_p = "<1e-08"
    return format_p


def calc_empirical_cdf(data):
    """
    Calculate the empirical cumulative distribution function from a 1 x N sorted array.

    Note that np.unique outputs sorted data (smallest to largest)
    """
    x_values, counts = np.unique(data,
                                 return_counts=True)
    cumprob = np.cumsum(counts) / len(data)

    return x_values, cumprob

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Calculate Data Params
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def set_data_params(data_object):
    data_object.min_max_time = np.asarray([[np.min(__), np.max(__)] for __ in data_object.time_array])
    data_object.t_start = data_object.time_array[0][0]
    data_object.t_stop = data_object.time_array[-1][-1]
    data_object.num_recs = len(data_object.vm_array)
    data_object.num_samples = len(data_object.vm_array[0])
    data_object.rec_time = data_object.min_max_time[0][1] - data_object.min_max_time[0][0]
    data_object.fs = calc_fs(data_object.time_array[0])
    data_object.ts = 1 / data_object.fs  # overwrite these from neo for when data is manipulated (e.g. interp)
    sample_spacing_in_ms = data_object.ts * 1000

    data_object.norm_first_deriv_vm = np.diff(data_object.vm_array,
                                              append=0) / sample_spacing_in_ms
    data_object.norm_second_deriv_vm = np.diff(data_object.vm_array, n=2,
                                               append=np.zeros((data_object.num_recs, 2))) / sample_spacing_in_ms
    data_object.norm_third_deriv_vm = np.diff(data_object.vm_array, n=3,
                                              append=np.zeros((data_object.num_recs, 3))) / sample_spacing_in_ms

    data_object.vm_array.setflags(write=False)
    data_object.im_array.setflags(write=False)
    data_object.time_array.setflags(write=False)
    data_object.min_max_time.setflags(write=False)
    data_object.norm_first_deriv_vm.setflags(write=False)
    data_object.norm_third_deriv_vm.setflags(write=False)
    data_object.norm_second_deriv_vm.setflags(write=False)

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Utils
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def calc_fs(time_):
    num_samples = len(time_)
    time_period = np.max(time_) - np.min(time_)
    fs = (num_samples - 1) / time_period
    return fs

def nearest_point_euclidean_distance(x1, timepoints, y1, datapoints):
    """
    Return the euclidean distance between a (x1, y1) datapoint and the nearest true sample
    with all data standardized to the variance.

    INPUTS:

        x1, y1: coordinates of a datapoint
        timepoints (x1...xn), datapoints (y1...yn): coordinates of datapoints to find the closest
    """
    time_std = np.std(timepoints)
    data_std = np.std(datapoints)
    distance = np.sqrt(((y1 - datapoints) / data_std)**2 + ((x1 - timepoints) / time_std)**2)

    return distance

def quick_get_time_in_samples(ts, timepoint):
    return np.round(timepoint / ts).astype(int)

def quick_get_samples_in_time(ts,
                              num_samples):
    return num_samples * ts

def generate_time_array(start, stop, num_samples, known_ts, start_stop_time_in_ms=False):
    """
    """
    if start_stop_time_in_ms:
        start /= 1000
        stop /= 1000

    time_array = np.linspace(start, (stop - known_ts), num_samples)

    if start_stop_time_in_ms:
        time_array *= 1000

    return time_array

def sort_dict_based_on_keys(dict_to_sort):
    """
    Sort a dict by the key, assuming the key is a str time
    at which a spike / event occured.
    """
    sorted_dict = dict(sorted(dict_to_sort.items(), key=lambda item: float(item[0])))

    return sorted_dict

def get_conversion_to_pa_table():
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

def get_conversion_to_mv_table():
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

def total_num_events(event_info, return_per_rec=False):  # TODO: this method is called a lot - would be better to call once and save results.
    """
    calculate the total number of events in the event info (rec of dicts) either summed or per-rec
    """
    num_recs = len(event_info)
    per_rec = np.zeros(num_recs)
    for rec in range(num_recs):
        if np.any(event_info[rec]):
            per_rec[rec] = len(event_info[rec])

    events_to_return = np.sum(per_rec) if not return_per_rec else per_rec

    return events_to_return
