from PySide2 import QtWidgets, QtCore, QtGui
import numpy as np
from utils import utils
from ephys_data_methods import core_analysis_methods, event_analysis_master, voltage_calc
import numpy as np
from numpy import *

def sliding_window(data,
                   width_s,
                   rise,
                   decay,
                   ts,
                   downsample,
                   min_chunk_factor,
                   ):
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
    for i in range(num_chunks):

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

def clements_bekkers(data, template):
    """
    Luke Campagnola's implementation:
    https://github.com/campagnola/neuroanalysis/blob/master/neuroanalysis/event_detection.py
    """
    # Strip out meta-data for faster computation
    D = data.view(ndarray)
    T = template.view(ndarray)

    # Prepare a bunch of arrays we'll need later
    N = len(T)
    sumT = T.sum()
    sumT2 = (T**2).sum()
    sumD = rolling_sum(D, N)
    sumD2 = rolling_sum(D**2, N)
    sumTD = np.correlate(D, T, mode='valid')

    # compute scale factor, offset at each location:
    scale = (sumTD - sumT * sumD / N) / (sumT2 - sumT**2 / N)
    offset = (sumD - scale * sumT) / N

    # compute SSE at every location
    SSE = sumD2 + scale**2 * sumT2 + N * offset**2 - 2 * (scale*sumTD + offset*sumD - scale*offset*sumT)

    # finally, compute error and detection criterion
    error = sqrt(SSE / (N-1))
    DC = scale / error
    return DC, scale, offset

def rolling_sum(data, n):
    """
    Luke Campagnola's implementation:
        https://github.com/campagnola/neuroanalysis/blob/master/neuroanalysis/event_detection.py
    """
    d1 = np.cumsum(data)
    d2 = np.empty(len(d1) - n + 1, dtype=data.dtype)
    d2[0] = d1[n-1]  # copy first point
    d2[1:] = d1[n:] - d1[:-n]  # subtract
    return d2

