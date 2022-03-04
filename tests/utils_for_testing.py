import numpy as np
from utils import utils

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Tests move - was moved from test_spikecalc
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def random_int_with_minimum_distance(min_val, max_val, n, min_distance, avoid_range=False):
    """
    Create an array of random integers of length n separated by a minimum distance.
    Will throw a warning if the minimum distance is with 25% of the maximum distance.
    """
    number_range = max_val - min_val
    assert number_range / min_distance > n, "Number of samples must be big enough to fit n samples of min distance"
    if not number_range / min_distance > n * 1.25:
        warnings.warn("max_val / min_distance is close to n_freqs, while loop may take some time.")

    if avoid_range:
        min_bound, max_bound, dist = avoid_range

    while True:
        rand_nums = np.random.randint(min_val, max_val, n)
        rand_nums = np.sort(rand_nums)

        if avoid_range:
            # use for generating test spikes, dont let them be too close to boudns
            if ((rand_nums > min_bound - dist) & (rand_nums < min_bound + dist)).any() or \
                    ((rand_nums > max_bound - dist) & (rand_nums < max_bound + dist)).any():
                continue

        max_difference = np.abs(np.min(np.diff(rand_nums)))
        if max_difference > min_distance:
            break

    return rand_nums

def vals_within_bounds(array, lower_bound, upper_bound, fill=np.nan):
    """
    Return the value between two bounds in a numpy array. Bounds may be either a single integer or (currently tested up to)
    a 1D or N x 1 array of bounds. If bounds are a 1D array they will be converted to a N x 1 array otherwise it cannot be used
    to correctly slice numpy array.
    """
    if type(lower_bound) == list:
        lower_bound = np.array(lower_bound)

    if type(upper_bound) == list:
        upper_bound = np.array(upper_bound)

    bounds = {"lower": lower_bound, "upper": upper_bound}
    for key, bound in bounds.items():
        if isinstance(bound, np.ndarray) and len(np.shape(bound)) == 1:
            bounds[key] = np.reshape(bound, (len(bound), 1))

    logical_mask = np.logical_and(bounds["lower"] <= array, array <= bounds["upper"])  # [inclusive, exclusive] to match current_calcs
    masked_array = np.ma.masked_array(array, ~logical_mask, fill_value=fill).filled()
    return masked_array

def generate_test_frequency_spectra(fs=16384, dist=1000):
    """
    Generate noisy data made up of added together sine waves at various frequencies. The Hz of sine waves to insert is determined
    with a minimum distance so that when used to generate data for testing filtering, the separation between frequencies is sufficient to
    avoid filter roll-off effects.

    TODO: very similar to generate_artificial_data.generate_test_frequency_spectra()
    """
    freq_near_nyquist = (fs / 2) * 0.95  # dont insert freqs too close to nyquist
    n_freqs = int((freq_near_nyquist/dist)-1)
    hz_to_add = random_int_with_minimum_distance(min_val=10,  # too low gets strange resutls when testing filter
                                                 max_val=freq_near_nyquist,
                                                 n=n_freqs,
                                                 min_distance=dist)
    x = np.linspace(0, 2 * np.pi, fs)
    y = np.zeros(fs)
    for hz in hz_to_add:
        y = y + np.sin(x * hz)

    return y, hz_to_add, fs, dist, n_freqs

def get_spike_times_from_spike_info(test_spkcnt, spike_info, param_type="times"):
    """
    get time vs. amplitude vs. idx from spike_info.
    """
    spike_param = utils.np_empty_nan((test_spkcnt.num_recs, test_spkcnt.max_num_spikes))
    for idx, rec_spikes in enumerate(spike_info):
        if rec_spikes:
            if param_type == "times":
                param = [float(time) for time in rec_spikes.keys()]
            elif param_type == "amplitude":
                param = [float(spike[1][0]) for spike in rec_spikes.items() if rec_spikes]
            elif param_type == "idx":
                param = [int(spike[1][1]) for spike in rec_spikes.items() if rec_spikes]
            spike_param[idx, 0:len(param)] = param
    return spike_param
