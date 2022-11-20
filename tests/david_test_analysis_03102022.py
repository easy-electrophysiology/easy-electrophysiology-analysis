from pathlib import Path
import pandas as pd
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------
# Fitting functions from Easy Electrophysiology (Can Ignore)
# -----------------------------------------------------------------------------------

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

def biexp_decay_least_squares_cost_function(coefs, x, y):
    yhat = biexp_decay_function(x,
                                coefs)
    return yhat - y

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

def get_initial_est(x_to_fit,
                    y_to_fit):
    est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit,
                                                                          y_to_fit)
    initial_est = (est_offset, est_slope, est_tau * 0.1, est_slope, est_tau * 0.9)

    return initial_est


# -----------------------------------------------------------------------------------
# Run Analysis
# -----------------------------------------------------------------------------------

# Load Raw Data

base_path = Path(r"C:\fMRIData\git-repo\easy_electrophysiology\tests\test_data")

raw_event = pd.read_csv(base_path / "test_david_analysis_average_event_2.csv").to_numpy()
event_kinetics = pd.read_csv(base_path / "test_david_analysis_average_event_kinetics.csv")

# Extract the baseline and peak im from the Easy Electrophysiology results
# (Average Event > Display Results Table, copy bottom 2 rows to a .csv).  Get the
# raw event data

baseline_im = float(event_kinetics["Baseline (pA)   "])  # the header names have strange space formatting for display reasons.
peak_im = float(event_kinetics["Peak (pA)   "])

event_time = raw_event[:, 0]
event_im = raw_event[:, 1]

peak_sample = np.argmin(event_im)
end_sample = event_time.size

# Get the time and Im to fit (the decay period i.e. peak to end of average event.
# The time must be normalised or the fit will be poor.
time_to_fit = event_time[peak_sample:end_sample + 1]
time_to_fit = time_to_fit - time_to_fit[0]
im_to_fit = event_im[peak_sample:end_sample + 1]

# Perform curve fitting to get coefficients of the biexponential fit to the decay,
# note the initial estimate is important for good fits.

initial_est = get_initial_est(time_to_fit, im_to_fit)
coefs = scipy.optimize.least_squares(biexp_decay_least_squares_cost_function, x0=initial_est, args=(time_to_fit, im_to_fit))
coefs = coefs.x

# Use the coefficients to generate the fit to check fitting, and print
# the tau1 and tau2 parameters (b0, b1 and b2 are ignored).

exp_decay_fit = biexp_decay_function(time_to_fit, coefs)
b0, b1, tau1, b2, tau2 = coefs
print(f"Time constant 1: {tau1:.4f} ms\nTime constant 2: {tau2:.4f} ms")

# Plot the fit over the original data

peak_time = event_time[peak_sample]
plt.plot(event_time, event_im)
plt.plot(peak_time + time_to_fit, exp_decay_fit)  # add back the time that was removed during normalisation
plt.show()

# Calculate the area (i.e. integral) of the data between
# the peak and end of recording. Note the data is baseline subtracted
# to ensure calculation of the area between the baseline and data.

ts_ms = time_to_fit[1] - time_to_fit[0]
decay_auc = np.trapz(im_to_fit - baseline_im, dx=ts_ms)

# Calculate the weighted decay, as the integral from end of event
# to peak divided by the peak im.

weighted_decay = decay_auc / peak_im
print(f"Weighted decay: {weighted_decay:.4f}")

