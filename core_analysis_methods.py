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
import scipy

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

# Least Squares Cost Functions
# ----------------------------------------------------------------------------------------------------------------------------------------------------

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
    return y - yhat

def triexp_decay_least_squares_cost_function(coefs, x, y):
    yhat = triexp_decay_function(x,
                                 coefs)
    return y - yhat

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
    biexponential event ("biexp_event") and triponential ("triexp")

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
        return False, False

    func = get_fit_functions(analysis_name)
    fit = func(x_to_fit, coefs)

    return coefs, fit

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
                          (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf))
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
    Get the Curve Fitting starting estimates for the biexpoentnial event function.
    """
    default_rise = 0.0005
    default_decay = 0.005

    est_offset, est_slope, __ = get_exponential_function_initial_est(x_to_fit, y_to_fit)

    if direction == -1:
        est_slope = np.min(y_to_fit) - y_to_fit[0]
    elif direction == 1:
        est_slope = np.max(y_to_fit) - y_to_fit[0]
    initial_est = (est_offset, est_slope, default_rise, default_decay)

    return initial_est

# Convenience Functions ------------------------------------------------------------------------------------------------------------------------------

def get_least_squares_fit_function(analysis_name):
    """
    Convenient access for residual fitting functions that return y - yhat used in scipy curve fit.
    """
    least_squares_fit_functions = {"monoexp": monoexp_least_squares_cost_function,
                                   "biexp_decay": biexp_decay_least_squares_cost_function,
                                   "biexp_event": biexp_event_least_squares_cost_function,
                                   "triexp": triexp_decay_least_squares_cost_function}
    return least_squares_fit_functions[analysis_name]

def get_fit_functions(analysis_name):
    """
    Convenient access for fitting functions.
    """
    fit_functions = {"monoexp": monoexp_function,
                     "biexp_decay": biexp_decay_function,
                     "biexp_event": biexp_event_function,
                     "triexp": triexp_decay_function}

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
    Calculate AP threshold using fisrt and third derivative methods,
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
    g[first_deriv <= lower_bound] = np.nan  # 0
    idx = np.nanargmax(g)
    return idx

def method_ii_threshold(cfgs,
                        first_deriv,
                        second_deriv,
                        third_deriv):

    h = ((third_deriv * first_deriv) - (second_deriv ** 2)) / (first_deriv ** 3)
    lower_bound = cfgs.skinetics["method_II_lower_bound"]
    h[first_deriv <= lower_bound] = np.nan  # 0
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
    ahp_search_start_idx = peak_idx + int(start_time * samples_per_ms)
    ahp_search_stop_idx = peak_idx + int(stop_time * samples_per_ms)
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
    start_to_peak_data: a 1 x t array containng datapoints as above
    peak_to_end_time: a 1 x t array containg timepoints for an even from the peak to the end point (e.g. fAHP)
    peak_to_end_data: a 1 x t array containng datapoints as above
    half_amp: scalar value, the amplitude / 2 representing the true half-maximum data
    interp: bool, interpolate data to 200 kHz before calculting fwhm.
    """
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

    return rise_midtime, rise_midpoint, fall_midtime, fall_midpoint, fwhm

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

    TODO: This function is similar to to calc_falling_slope_time, though with some key differences.
          Revisit to see if it is worthfile to factor out differences and combine into one function.
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
              the max value of the timseries. e.g. AP peak or negative event baseline
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

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Data Manipulation
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
    poly_order: order of polynomial to detrend (1-20 for Detrend data manipulation)
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

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Cumulative Probability
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def process_frequency_data_for_cum_prob(event_times):
    """
    Process event times for cumulative probability graphs.
    Check events are sorted (there has been a problem somewhere if events are not in order)
    Start from zero to find the event-interval for the first event, all inter-event intervals.
    Also, return labels of the event numbers.
    """
    assert np.array_equal(event_times, np.sort(event_times)), "process_frequency_data_for_cum_prob times not sorted"

    data = np.concatenate([[0],
                           event_times])
    data = np.diff(data)

    sort_idx = [str(i) + "-" + str(i + 1) for i in range(len(event_times))]

    return data, sort_idx


def process_decay_data_for_cum_prob(decay_taus):
    """
    Return sorted decay data for cum prob binning,
    and array of event labels.
    """
    data = np.sort(decay_taus)
    sort_idx = np.argsort(decay_taus) + 1
    return data, sort_idx


def process_amplitude_data_for_cum_prob(amplitudes):
    """
    Return sorted, absolute amplitudes for cum prob analysis,
    and event numbers or labelling.
    """
    data = np.sort(np.abs(amplitudes))
    sort_idx = np.argsort(np.abs(amplitudes)) + 1

    return data, sort_idx


def calc_cumulative_frequency(data, bin_divisor):
    """
    Calculate the binned cumulative probability from data processed
    for cum prob analysis.

    INPUT:
        data: data processed for cum prob analysis (see above functions)
        bin_divisor: scalar to divide the max number of bins. Use can set in
                     Event Analysis - Misc. Options.

    OUTPUT:
        cum_prob: 1 x bin cumulative probability (in the interval 0 1)
        x_values: cumulative bins
    """
    if len(data) < 4:
        return False, False

    # default max but lower limit = 0
    num_bins = np.floor(len(data) / bin_divisor).astype(int)
    s = (np.max(data) - np.min(data)) / (2. * (num_bins - 1.))
    max_ = np.max(data) + s

    info = scipy.stats.cumfreq(data,
                               numbins=num_bins,
                               defaultreallimits=(0, max_))

    cum_prob = info.cumcount / len(data)
    x_values = info.lowerlimit + np.linspace(0,
                                             info.binsize * len(info.cumcount),
                                             info.cumcount.size)
    return cum_prob, x_values

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
    at which a spike / event occured
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

