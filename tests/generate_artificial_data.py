import numpy as np
import sys, os
import copy

sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
from utils import utils
import utils_for_testing as test_utils
from ephys_data_methods import core_analysis_methods
import scipy

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Artificial Data Base Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class GenerateArtificialData:
    """
    Setup a base artificial dataset with a set number of runs and samples, resting vm and random im injections between -500 and 500.
    Artificial spike and input resistance data will inherit from this class.
    """

    def __init__(self, num_recs=75, num_samples=16324, time_stop=2, inter_rec_time_gap=False):
        self.inter_rec_time_gap = inter_rec_time_gap
        self.num_recs = num_recs
        self.num_samples = num_samples
        self.time_start = 0
        self.time_stop = time_stop
        self.im_offset = -20
        self.resting_vm = -60  #
        self.time_type = None  # filled in setup_test_suite TODO remove from all function calls
        self.rec_time = time_stop
        self.fs = (self.num_samples - 1) / self.rec_time
        self.ts = 1 / self.fs

        # make an array of baseline vm (-61) and im (-20) values, time series (norm or cumulative) and min/max times across the records
        self.vm_array = self.gen_vm_array()
        self.im_array = self.gen_im_array()
        self.norm_time_array = self.gen_norm_time_array()
        self.cum_time_array = self.gen_cum_time_array(self.norm_time_array, inter_rec_time_gap)
        self.norm_min_max_time = self.gen_min_max_times(self.norm_time_array)
        self.cum_min_max_time = self.gen_min_max_times(self.cum_time_array)
        self.sin_time_array = self.gen_sin_time_array()
        self.sin_min_max_time = self.gen_min_max_times(self.sin_time_array)
        self.time_array = None  # these will be filled with the currently tested time (e.g. norm vs cum)
        self.min_max_time = None

        # setup a random current injecion between -500 and 500 for each record, with a slight ramp to the stop amplitude. Decide the idx to start and
        # stop the injection and store the injection samples in self.injection
        self.current_injection_amplitude = np.random.randint(-500, 500, self.num_recs)
        self.start_idx = np.floor(self.num_samples * 0.1).astype(int)
        self.stop_idx = np.floor(self.num_samples * 0.9).astype(int)
        self.start_time = self.norm_time_array[0][self.start_idx]
        self.stop_time = self.norm_time_array[0][self.stop_idx]
        self.injection_ = np.linspace(
            self.current_injection_amplitude,
            self.current_injection_amplitude,
            self.stop_idx - self.start_idx,
        ).transpose()
        self.inject_im()

    def inject_im(self):
        for rec in range(0, self.num_recs):
            self.im_array[rec][self.start_idx : self.stop_idx] += self.injection_[rec]

    def gen_vm_array(self):
        vm_array = np.empty((self.num_recs, self.num_samples))
        vm_array.fill(self.resting_vm)
        return vm_array

    def gen_im_array(self):
        im_array = np.empty((self.num_recs, self.num_samples))
        im_array.fill(self.im_offset)  # CONFUSING NAME
        return im_array

    def gen_norm_time_array(self):
        norm_time_array = np.linspace(self.time_start, self.time_stop, self.num_samples)
        norm_time_array = np.tile(norm_time_array, (self.num_recs, 1))
        return norm_time_array

    def gen_sin_time_array(self):
        sin_time_array = np.linspace(0, np.pi * 2, self.num_samples)
        sin_time_array = np.tile(sin_time_array, (self.num_recs, 1))
        return sin_time_array

    def gen_cum_time_array(self, norm_time_array, inter_rec_time_gap):
        """
        generate a time array that increases cumulatively across records (e.g rec1: 0-2, rec2: 2-4).
        """
        cum_time_array = utils.np_empty_nan(np.shape(norm_time_array))
        cum_time_array[0, :] = norm_time_array[0, :]

        # cum time array increases per row by time stop
        if inter_rec_time_gap:
            gaps = np.arange(norm_time_array.shape[0])
        else:
            gaps = np.zeros(norm_time_array.shape[0])

        for rec in range(1, len(norm_time_array)):
            cum_time_array[rec, :] = norm_time_array[rec, :] + (self.time_stop * rec) + (self.ts * rec) + gaps[rec]

        return cum_time_array

    def gen_min_max_times(self, test_time_array):
        test_min_max_time = np.asarray([[np.min(__), np.max(__)] for __ in test_time_array])
        return test_min_max_time

    def get_perfect_tau(self, num_samples):
        """
        Rearrange the monoexponential function, also works for biexponential function
        """
        perfect_tau = np.abs(num_samples / np.log(0.00000000999))
        return perfect_tau


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Artificial Curve Fitting Data Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class ArtificialCurveFitting(GenerateArtificialData):
    def __init__(self, num_recs=5):
        """
        Create Curve Fitting artificial data for testing.

        This works by inserting functions of known coefficients into the data at specified timepoints (always the same). These are
        then measured through the GUI in the test_curve_fitting.py tests, with the appropriate initial estimates, and the
        coefficients are compared.

        To vary the coefficients, a set of canonical coefficients (self.cannonical_coefs) are multiplied by random integers
        between 1 and 10. In a number of getter functions (e.g. get_b_coef) the relevant coefficient is returned
        by multiplying the canonical coeff with its stored offset.

        For more detail, see the test_curve_fitting.py.

        """
        super().__init__(num_recs=num_recs, num_samples=2048, time_stop=1.2)  # long to run, low num recs

        self.function_samples = 500
        self.start_sample = 650
        self.stop_sample = self.start_sample + self.function_samples

        self.store_biexp_event_func_data = [[] for __ in range(num_recs)]
        for rec in range(num_recs):
            self.store_biexp_event_func_data[rec] = []

        self.start_times = None  # These are overwritten in insert_function_to_data() to handle dynamic norm vs. cumulative time settings
        self.stop_times = None
        self.cannonical_coefs = None

        self.current_inections_vm = None  # used for slope tests for moving regions across recs
        self.current_inections_im = None

        self.im_offset = self.resting_vm = self.offset = -60

        self.tau = self.get_perfect_tau(self.function_samples)
        self.tau_ms = self.tau * self.ts * 1000
        self.b1 = 10
        self.coef_offsets = []

        self.update_cannonical_coefs()

    def update_cannonical_coefs(self):
        self.cannonical_coefs = {
            "monoexp": np.array([self.offset, self.b1, self.tau]),
            "biexp_decay": np.array([self.offset, self.b1, self.tau, self.b1, self.tau]),
            "biexp_event": np.array(
                [self.offset, self.b1, self.tau, self.tau]
            ),  # tau is correct for this too, see function
            "triexp": np.array([self.offset, self.b1, self.tau, self.b1, self.tau, self.b1, self.tau]),
        }

    # Varying Coefs and Inserting Functions
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def insert_function_to_data(self, vary_coefs, func_type, pos_or_neg="pos"):
        """
        Insert the function into the data.

        We can cary the coefficients either within a function (e.g. for biexp_decay, b1 != b2) or across records e.g. (monoexp, b1 on
        rec1 != b1 on rec 2. This is implemented here, see main function on test_curve_fitting.py for details.

        We need to update start / stop times based on whether the data used in tests is "normalised" or "cumulative".
        This is set in the gui method update_curve_fitting_function() and the self.min_max_time attribute of this
        class is updated in that method. Due to the positioning of function calls on the test method all works well,
        although it is not the nicest way to do this.
        """
        self.start_times = (self.start_sample * self.ts) + self.min_max_time[:, 0]  # handle cum vs. norm
        self.stop_times = (self.stop_sample * self.ts) + self.min_max_time[:, 0]
        self.b1 = self.b1 * 1 if pos_or_neg == "pos" else self.b1 * -1
        self.update_cannonical_coefs()

        self.set_coef_offsets(vary_coefs)

        if vary_coefs == "within_function":
            self.overwrite_coef_offsets_to_repeat_first_rec()

        self.insert_monoexp_function(func_type)

    def overwrite_coef_offsets_to_repeat_first_rec(self):
        """
        To keep the coefficients the same across records, repeat the first
        record offsets across all records
        """
        num_offsets = len(self.coef_offsets)
        first_offset = self.coef_offsets[0]
        self.coef_offsets = []
        for __ in range(num_offsets):
            self.coef_offsets.append(first_offset)

    def rand(self, b1_or_tau):
        """
        Tau is better to decrease only as if you increase it starts to go above the BL. TBH there is no reason
        why this is bad, its just for aesthetics. Probably a bad idea.
        """
        if b1_or_tau == "b1":
            bl = np.random.randint(1, 10)
            if self.b1 < 0:
                bl *= -1
            return np.random.randint(1, 10)
        if b1_or_tau == "tau":
            return 1 / np.random.randint(1, 10)  # Explain

    def set_coef_offsets(self, vary_coefs):
        """
        These coefficient offsets are multiplied by the canonical coefficients to set coefficients on the inserted
        functions. These are all 1 if vary_coef is false, otherwise random integers as specified in self.rand()
        """
        self.coef_offsets = []

        for rec in range(self.num_recs):
            if vary_coefs in ["within_function", "across_records"]:
                self.coef_offsets.append(
                    {
                        "monoexp": np.array([1, self.rand("b1"), self.rand("tau")]),
                        "biexp_decay": np.array(
                            [
                                1,
                                self.rand("b1"),
                                self.rand("tau"),
                                self.rand("b1"),
                                self.rand("tau"),
                            ]
                        ),
                        "biexp_event": np.array([1, self.rand("b1"), self.rand("tau"), self.rand("tau")]),
                        "triexp": np.array(
                            [
                                1,
                                self.rand("b1"),
                                self.rand("tau"),
                                self.rand("b1"),
                                self.rand("tau"),
                                self.rand("b1"),
                                self.rand("tau"),
                            ]
                        ),
                    }
                )
            else:
                self.coef_offsets.append(
                    {
                        "monoexp": np.array([1, 1, 1]),
                        "biexp_decay": np.array([1, 1, 1, 1, 1]),
                        "biexp_event": np.array([1, 1, 1, 1]),
                        "triexp": np.array([1, 1, 1, 1, 1, 1, 1]),
                    }
                )

    def insert_monoexp_function(
        self, orig_func_type
    ):  # TODO: im not tested! FUNCTION NAME CONFUSING INSERT ANY FUNCTIN!
        """
        Insert a function of func_type into the data.

        Clear the im / vm array.The function is generated with coefficients that are either varied or canonical coefs.

        if min is tested, flip function so it is negative. If mean / median is tested, do a large offset as makes it
        easier to see the true mean / median (if it is a monoexp function the mean is close to zero)
        """
        if orig_func_type in ["min", "max", "mean", "median", "area_under_curve_cf"]:
            func_type = "biexp_event"
        else:
            func_type = orig_func_type

        self.im_array = self.gen_im_array()
        self.vm_array = self.gen_vm_array()

        for rec in range(self.num_recs):
            func = core_analysis_methods.get_fit_functions(func_type)
            coefs = self.cannonical_coefs[func_type]
            coef_offsets = self.coef_offsets[rec][func_type]

            x_to_fit = np.arange(self.function_samples)

            func_data = func(x_to_fit, coefs * coef_offsets)

            if orig_func_type == "min":
                func_data -= self.offset
                func_data *= -1
                func_data += self.offset

            if orig_func_type in ["mean", "median"]:
                func_data += 50

            if func_type == "biexp_event":
                self.store_biexp_event_func_data[rec].append(func_data)

            self.im_array[rec, self.start_sample : self.stop_sample] = func_data
            self.vm_array[rec, self.start_sample : self.stop_sample] = func_data

    # Getters and Setters.
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Used for getting the true coefficient values, taking into account the coefficient offsets. returns for all records for qtable
    # tests or per-record for model tests.
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def get_b0(self, rec_from=None, rec_to=None, rec=None):
        if rec is not None:
            b0 = self.offset
        else:
            num_recs = rec_to - rec_from + 1
            b0 = np.repeat(self.offset, num_recs)
        return b0

    def get_offsets_within_recs(self, rec_from, rec_to, func_type, idx):
        all_data = []
        for rec, data in enumerate(self.coef_offsets):
            if rec_from <= rec <= rec_to:
                all_data.append(data[func_type][idx])
        return np.array(all_data)

    def get_b_coef(self, func_type, b_name, rec_from=None, rec_to=None, rec=None):
        b_idx = {"b1": 1, "b2": 3, "b3": 5}
        idx = b_idx[b_name]

        if rec is not None:
            bx = self.b1 * self.coef_offsets[rec][func_type][idx]
        else:
            all_b1_offsets = self.get_offsets_within_recs(rec_from, rec_to, func_type, idx)
            bx = self.b1 * all_b1_offsets

        return bx

    def get_tau(self, func_type, tau_name, rec_from=None, rec_to=None, rec=None):
        tau_idx = {
            "tau": 2,
            "tau1": 2,
            "tau2": 4,
            "tau3": 6,
            "fit_rise": 2,
            "fit_decay": 3,
        }
        idx = tau_idx[tau_name]

        if rec is not None:
            taux = self.tau_ms * self.coef_offsets[rec][func_type][idx]
        else:
            all_tau_offsets = self.get_offsets_within_recs(rec_from, rec_to, func_type, idx)
            taux = self.tau_ms * all_tau_offsets

        return taux

    # Sloping Im / Vm injections
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def update_vm_im_with_slope_injection(self):
        """
        Fill Im and Vm with (different) sloping injections from 0 to a randint stored in the
        variables:
        self.current_inections_vm
        self.current_inections_im
        """
        self.current_inections_vm = np.random.randint(1, 1000, self.num_recs)
        self.current_inections_im = np.random.randint(1, 1000, self.num_recs)

        self.im_array = self.gen_slope_array(self.current_inections_vm)
        self.vm_array = self.gen_slope_array(self.current_inections_im)

        self.gen_start_stop_samples_and_times_for_different_regions_across_recs()

    def gen_start_stop_samples_and_times_for_different_regions_across_recs(self):
        """
        for testing bounds across recs, we need a different start / stop time per rec.
        The start/stop times are not functional here (e.g. they don't define the start/stop
        of a function like in insert_monoexp_function() but they define the cannonical start/stop
        times over which the test for that rec will be measured.
        """
        self.stop_sample = utils.np_empty_nan(self.num_recs).astype(int)
        self.start_sample = utils.np_empty_nan(self.num_recs).astype(int)
        self.start_times = utils.np_empty_nan(self.num_recs)
        self.stop_times = utils.np_empty_nan(self.num_recs)

        for rec in range(self.num_recs):
            start_sample, stop_sample = test_utils.random_int_with_minimum_distance(
                min_val=10, max_val=self.num_samples - 10, n=2, min_distance=100
            )
            self.start_sample[rec] = start_sample
            self.stop_sample[rec] = stop_sample

            self.start_times[rec] = self.time_array[rec][start_sample]
            self.stop_times[rec] = self.time_array[rec][stop_sample]

    def gen_slope_array(self, current_inections):
        slope_array = utils.np_empty_nan((self.num_recs, self.num_samples))

        for rec in range(self.num_recs):
            slope_array[rec, :] = np.linspace(0, current_inections[rec], self.num_samples)

        return slope_array


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Master Class for Spike Count and Events Artificial Data
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class EventAndSpikesMaster(GenerateArtificialData):
    """
    Class to generate articifial data for testing count spikes. Creates a vm array of -61 mV and randomly inserts "spikes"
    across each record. Saves the number and times of spikes per record (both in a normalised time series and cumulatively
    increasing time series). This can be used to compare the known number of spikes / time times with those calculated from
    the current_calc.count_spikes function
    """

    def __init__(self, num_recs=75, num_samples=16324, time_stop=2, inter_rec_time_gap=False):
        super().__init__(num_recs, num_samples, time_stop, inter_rec_time_gap)
        self.peak_times = dict()

        # make a cannonical spike and array to fill with spikes
        self.spikes_per_rec = np.random.randint(
            self.min_spikes, self.max_num_spikes, (self.num_recs, 1)
        ).squeeze()  # need at least 5 evs for some tests

        self.spikes_per_rec = np.atleast_1d(self.spikes_per_rec)

        self.spike_sample_idx = utils.np_empty_nan(
            (self.num_recs, self.max_num_spikes)
        )  # holds the idx of all spikes (start), add samples_from_start_to_peak to get peak value

        self.peak_times["normalised"] = utils.np_empty_nan((self.num_recs, self.max_num_spikes))

        self.peak_times["cumulative"] = utils.np_empty_nan((self.num_recs, self.max_num_spikes))
        self.peak_times_ = None  # used for artificial data in gui tests
        self.spike_peak_idx = None
        self.bounds_start_idx = 5200
        self.bounds_stop_idx = 13000

    def create_array_of_indicies_to_insert_spikes(self):
        """
        Create a random array of indicies in which to insert the spikes. Test if these are within 6 samples of
        any other spike - if so try again until all spikes are at least 6 samples apart.
        """
        for rec in range(0, self.num_recs):
            spikes_in_rec = self.spikes_per_rec[rec]
            if spikes_in_rec < 1:  # TODO: now it is not possible to have less than 5 spikes
                potential_indicies = np.array([], "int32")
            elif spikes_in_rec == 1:
                potential_indicies = np.random.randint(0, self.num_samples - self.spike_width + 1, spikes_in_rec)
            else:
                potential_indicies = test_utils.random_int_with_minimum_distance(
                    min_val=self.rec_start_stop_spike_pad,
                    max_val=self.num_samples - self.rec_start_stop_spike_pad,
                    n=spikes_in_rec,
                    min_distance=self.min_samples_between_spikes,
                    avoid_range=[self.bounds_start_idx, self.bounds_stop_idx, 100],
                )

            self.spike_sample_idx[rec, 0:spikes_in_rec] = potential_indicies
            self.spike_peak_idx = self.spike_sample_idx + self.samples_from_start_to_peak
            self.peak_times["normalised"][rec, 0:spikes_in_rec] = self.norm_time_array[rec][
                potential_indicies + self.samples_from_start_to_peak
            ]
            self.peak_times["cumulative"][rec, 0:spikes_in_rec] = self.cum_time_array[rec][
                potential_indicies + self.samples_from_start_to_peak
            ]

    def fill_vm_array_with_spikes(self):
        """
        use self.spike_sample_idx to index out from vm_array and insert spike_
        """

        for rec in range(0, self.num_recs):
            for col in range(self.max_num_spikes):
                idx = self.spike_sample_idx[rec, col]
                if ~np.isnan(idx):
                    idx = idx.astype(int)
                    self.vm_array[rec, idx : idx + self.spike_width] = self.cannonical_spike

    def generate_test_rheobase_data_from_spikeinfo(self, rec_from, rec_to, spike_info, change_spikeinfo=True):
        """
        Take the spike_info of spikecount data and create a random record as the first rec by zero filling up to it.
        Return the zero-filled spikecount and the correct rheobase.
        If there are no spikes at all, will return False.
        Dont go too near rec_to as it will create problems for some tests (manual tests in spike count)
        """
        # replace spikes with empty until randomly generated first rheobase record
        if rec_from == rec_to:
            test_rheobase_rec = rec_from
        else:
            test_rheobase_rec = int(np.random.randint(rec_from, rec_to - 1, 1))

        if change_spikeinfo:
            for rec, element in enumerate(spike_info):
                spike_info[rec] = element if rec >= test_rheobase_rec else dict()

        # calculate and compare rheobase
        test_rheobase = True if any(spike_info) else False
        if test_rheobase:
            while not spike_info[
                test_rheobase_rec
            ]:  # if test rheobase rec happends to have zero spikes, find the first rec that doesn't.
                if not spike_info[test_rheobase_rec]:
                    test_rheobase_rec += 1

        return (
            spike_info,
            test_rheobase_rec,
            test_rheobase,
        )  # ARNT THESE ALWAYS THE SAME?

    def generate_bounds(self):
        bounds = [
            [self.norm_time_array[0, self.bounds_start_idx]] * self.num_recs,
            [self.norm_time_array[0, self.bounds_stop_idx]] * self.num_recs,
        ]
        return bounds

    def generate_baseline_bounds(self):
        bounds = [
            [self.norm_time_array[0, 0]] * self.num_recs,
            [self.norm_time_array[0, self.start_idx - 10]] * self.num_recs,
        ]
        return bounds

    def subtract_results_from_data(
        self,
        test_spkcnt,
        ee_spike_info,
        ee_spike_count,
        rec_from,
        rec_to,
        time_type,
        bounds,
    ):
        """
        Compare test data with data calculated by easy electrophysiology (ee_*)
        """
        upper_bound = rec_to + 1
        bound_low, bound_high = bounds

        bound_low = np.array(bound_low)[rec_from:upper_bound] + self.min_max_time[rec_from:upper_bound, 0]
        bound_high = np.array(bound_high)[rec_from:upper_bound] + self.min_max_time[rec_from:upper_bound, 0]

        # Get test spike times within bounds and compare to ee_spike times
        ee_spike_times = test_utils.get_spike_times_from_spike_info(test_spkcnt, ee_spike_info)
        test_spike_times = test_spkcnt.peak_times[time_type][rec_from:upper_bound]
        test_spike_times = test_utils.vals_within_bounds(test_spike_times, bound_low, bound_high)

        spike_times_equal = np.array_equal(
            test_spike_times[~np.isnan(test_spike_times)],
            ee_spike_times[~np.isnan(ee_spike_times)],
        )

        # check spike counts
        true_spike_count = [len(spikes[~np.isnan(spikes)]) for spikes in test_spike_times]
        spike_count_equal = np.array_equal(true_spike_count, ee_spike_count[rec_from:upper_bound])

        return spike_times_equal, spike_count_equal

    def overwrite_spike_times(self):
        """
        Due to different spike with we nee to overwrite cannonical peak times
        with those dynamically generated depending on the width of the individual spike.
        """
        self.peak_times["normalised"] = utils.np_empty_nan((self.num_recs, self.max_num_spikes))
        self.peak_times["cumulative"] = utils.np_empty_nan((self.num_recs, self.max_num_spikes))

        for rec in range(self.num_recs):
            spikes_in_rec = self.spikes_per_rec[rec]
            for spk in range(spikes_in_rec):
                spk_insert_idx = self.spike_sample_idx[rec, spk].astype(int)

                if type(self.samples_from_start_to_peak) == int:
                    samples_from_start_to_peak = self.samples_from_start_to_peak
                else:
                    samples_from_start_to_peak = self.samples_from_start_to_peak[rec][spk]

                self.peak_times["normalised"][rec, spk] = self.norm_time_array[rec][
                    spk_insert_idx + samples_from_start_to_peak
                ]
                self.peak_times["cumulative"][rec, spk] = self.cum_time_array[rec][
                    spk_insert_idx + samples_from_start_to_peak
                ]

    def num_spikes(self):
        spike_times = self.peak_times_[~np.isnan(self.peak_times_)]  # TODO: make remove nans
        return spike_times.size


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Artificial Spike Count Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestArtificialSkCntData(EventAndSpikesMaster):
    def __init__(self, max_num_spikes, min_spikes, num_recs=75):
        self.max_num_spikes = max_num_spikes
        self.min_spikes = min_spikes
        super().__init__(num_recs)

        self.spike_width = 50
        self.cannonical_spike = self.generate_cannonical_spike()
        self.samples_from_start_to_peak = np.argmax(
            self.cannonical_spike
        )  # number of samples start of spike (canonical spike) to baseline and peak

        self.min_samples_between_spikes = 16
        self.rec_start_stop_spike_pad = (
            50  # padding in samples at start / end of record where no spikes will be inserted
        )

        # create vm_array filled with spike counts, and the norm and cumulative times of those spike peaks
        self.create_array_of_indicies_to_insert_spikes()
        self.fill_vm_array_with_spikes()

    def generate_cannonical_spike(self):
        cannonical_spike = 100 * np.sin(np.linspace(0, np.pi * 2, self.spike_width))
        cannonical_spike += self.resting_vm
        return cannonical_spike


class TestArtificialsKinetics(EventAndSpikesMaster):
    """
    For peaks in skinetics, use all_true_peaks_idx
    """

    def __init__(self, num_recs, max_num_spikes, min_spikes, num_samples, time_stop):
        self.max_num_spikes = max_num_spikes
        self.min_spikes = min_spikes  # 15 if num_recs == 1 else 5
        super().__init__(num_recs=num_recs, num_samples=num_samples, time_stop=time_stop)

        self.spike_width = 100  # max spike width with freq = 1
        self.cannonical_spike_params = self.generate_cannonical_spike_params()
        self.samples_from_start_to_peak = 5  # number of samples start of spike (cannonical spike) to baseline and peak

        self.min_samples_between_spikes = self.spike_width * 1.5
        self.rec_start_stop_spike_pad = (
            self.spike_width * 10
        )  # padding in samples at start / end of record where no spikes will be inserted

        # create vm_array filled with spike counts, and the norm and cumulative times of those spike peaks
        self.create_array_of_indicies_to_insert_spikes()
        self.fill_vm_array_with_spikes()
        self.overwrite_spike_times()  # must come last

        self.recording_type = "current_clamp"  # needed for test loading
        self.num_data_channels = 1
        self.vm_units = "mV"
        self.im_units = "pA"

    def generate_cannonical_spike_params(self):
        cannonical_spike_params = []
        for rec in range(self.num_recs):
            spikes_this_rec = self.spikes_per_rec[rec]

            params = {
                "amplitudes": np.random.randint(70, 120, spikes_this_rec),
                "freqs": np.random.randint(1, 6, spikes_this_rec),
            }
            cannonical_spike_params.append(params)

        return cannonical_spike_params

    def generate_spike(self, amplitude, freq):
        x = np.linspace(0, np.pi * 2, self.spike_width + 1) * freq
        spike = self.resting_vm + (amplitude * np.sin(x))
        spike_num_samples = int(self.spike_width / freq)
        spike = spike[0:spike_num_samples]
        return spike, spike_num_samples

    def fill_vm_array_with_spikes(self):
        """
        use self.spike_sample_idx to index out from vm_array and insert spike_
        """
        self.samples_from_start_to_peak = []
        self.all_true_peaks = []
        self.all_true_peaks_idx = []
        self.all_true_mins = []
        for rec in range(0, self.num_recs):
            rec_samples_to_peaks = np.array([])
            true_peaks = np.array([])
            true_mins = np.array([])
            for col in range(self.max_num_spikes):
                idx = self.spike_sample_idx[rec, col]
                if ~np.isnan(idx):
                    idx = idx.astype(int)
                    spike, spike_num_samples = self.generate_spike(
                        self.cannonical_spike_params[rec]["amplitudes"][col],
                        self.cannonical_spike_params[rec]["freqs"][col],
                    )
                    spike_peak_idx = np.argmax(spike)
                    rec_samples_to_peaks = np.hstack([rec_samples_to_peaks, spike_peak_idx]).astype(int)
                    self.vm_array[rec, idx : idx + spike_num_samples] = spike
                    true_peaks = np.hstack([true_peaks, np.max(spike)])
                    true_mins = np.hstack([true_mins, np.min(spike)])

            self.samples_from_start_to_peak.append(rec_samples_to_peaks)
            isnan_rec_spike_idx = self.spike_sample_idx[rec][~np.isnan(self.spike_sample_idx[rec])]
            self.all_true_peaks_idx.append(isnan_rec_spike_idx + rec_samples_to_peaks)
            self.all_true_peaks.append(
                true_peaks
            )  # lower resolution means true peaks is not always expected by sine function
            self.all_true_mins.append(true_mins)

    def convert_sine_to_time(self, sine_time):
        """
        Convert units from the sine calculation of half-width etc.
        to time units used in the actual test data
        """
        test_x = np.linspace(0, np.pi * 2, self.spike_width + 1)
        sine_ts = test_x[1] - test_x[0]
        converted_time = (sine_time / sine_ts) * self.ts
        return converted_time

    def get_spike_max_slope(self, data, n_samples):
        max_slope = []
        slope_data = []
        for i in range(len(data) - n_samples):
            y = data[i : i + n_samples]
            x = np.arange(len(y)) * self.ts

            regress = scipy.stats.linregress(x, y)
            max_slope.append(regress.slope / 1000)  # convert slope from s to ms

            slope = regress.slope * x + regress.intercept
            slope_data.append(slope)

        return max_slope, slope_data

    def get_max_slope(self, n_samples_rise, n_samples_decay):
        """ """
        rec_max_rise = []
        rec_max_decay = []
        rec_rise_slopes = []
        rec_decay_slopes = []
        for rec in range(self.num_recs):
            spike_max_rise = []
            spike_max_decay = []
            spike_rise_slopes = []
            spike_decay_slopes = []
            for amplitude, freq in zip(
                self.cannonical_spike_params[rec]["amplitudes"],
                self.cannonical_spike_params[rec]["freqs"],
            ):
                spike, __ = self.generate_spike(amplitude, freq)
                peak_idx = np.argmax(spike)
                through_idx = np.argmin(spike)

                rise_data = spike[0 : peak_idx + 1]
                decay_data = spike[peak_idx : through_idx + 1]

                max_rise_slope, rise_slopes = self.get_spike_max_slope(rise_data, n_samples_rise)
                max_decay_slope, decay_slopes = self.get_spike_max_slope(decay_data, n_samples_decay)

                max_idx = np.argmax(
                    max_rise_slope
                )  # get the maximum of the set of slopes (max for rise, min for decay)
                min_idx = np.argmin(max_decay_slope)
                spike_max_rise.append(
                    max_rise_slope[max_idx]
                )  # save the max rise / decay and the slope data per spike, later saved per_rec
                spike_max_decay.append(max_decay_slope[min_idx])

                spike_rise_slopes.append(np.hstack(rise_slopes[max_idx]))
                spike_decay_slopes.append(np.hstack(decay_slopes[min_idx]))

            rec_max_rise.append(np.array(spike_max_rise))
            rec_max_decay.append(np.array(spike_max_decay))

            rec_rise_slopes.append(np.hstack(spike_rise_slopes))
            rec_decay_slopes.append(np.hstack(spike_decay_slopes))

        return rec_max_rise, rec_max_decay, rec_rise_slopes, rec_decay_slopes

    def get_times_paramteres(
        self,
        parameter,
        min_cutoff=None,
        max_cutoff=None,
        decay_bound="fAHP",
        bounds_vm=False,
        time_type=None,
    ):
        all_parameter = []
        for rec in range(self.num_recs):
            freqs = self.cannonical_spike_params[rec]["freqs"]

            if bounds_vm:
                within_bounds = self.get_within_bounds_spiketimes(rec, bounds_vm, time_type, return_as_bool=True)[
                    : len(freqs)
                ]
                freqs = freqs[within_bounds]

            rec_freqs = np.array([])
            for freq in freqs:
                if parameter == "rise_time":
                    sine_time = (
                        np.arcsin(min_cutoff / 100) - np.arcsin(max_cutoff / 100)
                    ) / freq  # TODO: DRY from test_skinetics unit tests
                elif parameter == "half_width":
                    sine_time = 2 * (np.arcsin(1) - np.arcsin(0.5)) / freq
                elif parameter == "decay_time":
                    sin_max_cutoff = (
                        1 - ((1 - 0) * max_cutoff / 100) if decay_bound == "thr" else 1 - (2 * (max_cutoff / 100))
                    )
                    expected_min_time = np.arcsin(1) + (np.arcsin(1) - np.arcsin(sin_max_cutoff))
                    sin_min_cutoff = (
                        1 - ((1 - 0) * min_cutoff / 100) if decay_bound == "thr" else (2 * (min_cutoff / 100) - 1)
                    )
                    expected_max_time = (
                        np.arcsin(1) + (np.arcsin(1) - np.arcsin(sin_min_cutoff))
                        if decay_bound == "thr"
                        else np.arcsin(1) * 3 - (np.arcsin(1) - np.arcsin(sin_min_cutoff))
                    )
                    sine_time = expected_min_time / freq - expected_max_time / freq

                expected_param = self.convert_sine_to_time(sine_time)
                rec_freqs = np.hstack([rec_freqs, expected_param])

            all_parameter.append(rec_freqs)
        return all_parameter

    def get_within_bounds_spiketimes(self, rec, bounds_vm, time_type, return_as_bool=False, cut_nan=False):
        """ """
        rec_peak_times = self.peak_times[time_type][rec]

        if cut_nan:
            rec_peak_times = rec_peak_times[~np.isnan(rec_peak_times)]

        within_bound = test_utils.vals_within_bounds(rec_peak_times, bounds_vm["exp"][0][rec], bounds_vm["exp"][1][rec])

        if return_as_bool:
            within_bound = ~np.isnan(within_bound)
        else:
            within_bound = within_bound[~np.isnan(within_bound)]
        return within_bound


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Artificial Events Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestArtificialEventData(EventAndSpikesMaster):
    def __init__(
        self,
        num_recs,
        num_samples,
        time_stop,
        min_num_spikes,
        max_num_spikes,
        event_type="monoexp",
        event_samples=None,
        negative_events=True,
        inter_rec_time_gap=False,
    ):
        """
        Generate events on a dataset. Similar to curve fitting, canonical coefficients for the events are initiated so that
        all events have the same b1 (modelled as a straight line) and decay(modelled as exponential event) This form was chosen
        to properly text the monoexp decay on the events. These canonical coefficients are then multiple my randomly generated
        integers to vary for each event. These integers are saved and getters allow retrieval of the coefficient offset
        for comparison with the measured offset in text functions.
        """
        self.min_spikes = min_num_spikes
        self.max_num_spikes = max_num_spikes  # increase num samples if want to increase further
        self.rise_samples = 100
        self.samples_from_start_to_peak = 100 - 1

        super().__init__(num_recs, num_samples, time_stop, inter_rec_time_gap)

        self.event_type = event_type
        self.im_array = copy.deepcopy(self.vm_array)
        self.resting_im = self.resting_vm
        self.decay_samples = 300

        if event_samples is None:
            self.event_samples = (
                2500 if self.event_type == "biexp" else self.rise_samples + self.decay_samples - 1
            )  # account for overlapping peak - 1 see generate function
        else:
            self.event_samples = event_samples

        self.rise_div = 10  # factor to divide the rise time by for artificial biexp events (rise and decay figure is generated together then rise is divided)

        if (
            self.event_type == "biexp"
        ):  # for biexp we need large B1 to give broad range of event sizes for checking different templates simultaneously.
            self.b1 = (
                -75
            )  # for monoexp that we test rise time half width etc. on we want lower B1 to the rise and decay is not undersampled (I think?)
        else:
            self.b1 = -10 if negative_events else 10

        self.peak_im = self.resting_im + self.b1

        self.spike_width = self.event_samples
        self.event_width_ms = self.event_samples * self.ts * 1000
        self.rec_start_stop_spike_pad = self.event_samples
        self.min_samples_between_spikes = int(
            0.020 * self.fs * 4
        )  # self.event_samples * 3 # 1.5 # int(0.020 * self.fs * 1.5)  # 0.020 is the default 20 ms template size

        self.rec_start_stop_spike_pad = (
            self.event_samples * 2
        )  # padding in samples at start / end of record where no spikes will be inserted
        self.tau = self.get_perfect_tau(self.event_samples)
        self.rise = self.get_perfect_tau(self.event_samples)
        self.decay = self.get_perfect_tau(self.event_samples)

        self.coefs = ()
        self.b1_offsets = np.nan
        self.tau_offsets = np.nan
        self.rise_offsets = np.nan
        self.decay_offsets = np.nan
        self.area_under_curves = np.nan

        self.vary_amplitudes_and_tau = False
        self.create_array_of_indicies_to_insert_spikes()
        self.fill_im_array_with_events()

    def update_with_varying_amplitudes_and_tau(self):
        """
        refresh Im to zero and fill again, this time with varying amplitudes
        """
        self.im_array = copy.deepcopy(self.vm_array)
        self.vary_amplitudes_and_tau = True
        self.fill_im_array_with_events()

    def generate_artificial_spike(self, b1_offset, tau_offset):
        rise = self.get_rise(b1_offset)
        decay_x = np.arange(0, self.decay_samples)
        coefs = (0, self.b1 * b1_offset, self.tau * tau_offset)
        decay = core_analysis_methods.monoexp_function(decay_x, coefs)
        event = np.hstack([rise[:-1], decay])  # dont concatenate full or will duplicate peak sample
        event += self.resting_vm
        area_under_curve = np.trapz(event - event[0], dx=self.ts)
        return event, area_under_curve

    def generate_artificial_biexp_spike(self, b1_offset, rise_offset, decay_offset):
        coefs = (
            0,
            self.b1 * b1_offset,
            self.rise * rise_offset,
            self.decay * decay_offset,
        )
        event_x = np.arange(0, self.event_samples)
        event = core_analysis_methods.biexp_event_function(event_x, coefs)
        event += self.resting_vm
        return event

    def update_events_time_to_irregularly_spaced(self):
        """
        Events only tests cumulative time
        """
        offsets = np.arange(0, 10 * self.time_array.shape[0], 10)
        offsets = np.atleast_2d(offsets).T
        self.time_array = self.time_array + offsets
        self.cum_time_array = self.time_array
        self.min_max_time = self.gen_min_max_times(self.cum_time_array)
        self.overwrite_spike_times()

    # Getters
    # ----------------------------------------------------------------------------------------------------------------------------------------------------
    # Methods for getting coefficients and parameters calculated from the events / the coefficient offets

    def get_rise(self, b1_offset):
        return np.linspace(0, self.b1 * b1_offset, self.rise_samples)

    def get_decay_percent_num_samples(self, percentage, tau_offset=1):
        """
        Find the nearest sample on the decay that is to  decay using the monoexp function
        percentage = 0.37 fo rtau
        0.5 for half-width
        """
        perc_37 = self.b1 * (percentage / 100)
        nearest_x = (self.tau * tau_offset) * np.log(perc_37 / self.b1)  # rearrange the exp function
        nearest_x = abs(round(nearest_x))

        return nearest_x

    def calc_event_half_width(self, tau_offset=1):
        decay_samples = self.get_decay_percent_num_samples(50, tau_offset)
        half_decay_time = (decay_samples * self.ts) * 1000
        half_rise_sample = (
            self.rise_samples * 0.5
        )  # only works for even data but measuring from data directly led to 49 (49 and 50 were equivalent). This is better.
        half_rise_time = (half_rise_sample * self.ts) * 1000
        half_width_ms = (
            half_decay_time + half_rise_time
        )  # opposite way to EE, add the half-to-peak on the rise and peak-to-half on the decay

        return half_width_ms

    def get_all_half_widths(self):
        half_widths = []
        for ev_num in range(self.num_events()):
            ev_hw = self.calc_event_half_width(self.tau_offsets[ev_num])
            half_widths.append(ev_hw)
        return np.array(half_widths)

    def get_all_decay_times(self, decay_percent=37):
        decay_samples = []
        for ev_num in range(self.num_events()):
            ev_decay = self.get_decay_percent_num_samples(decay_percent, self.tau_offsets[ev_num])
            decay_samples.append(ev_decay)

        decay_times = (np.array(decay_samples) * self.ts) * 1000

        return decay_times

    def get_overlay_of_all_events(self, biexp=False):
        events_overlay = np.zeros((self.num_events(), self.event_samples))
        for idx, ev_num in enumerate(range(self.num_events())):
            if not biexp:
                ev, __ = self.generate_artificial_spike(self.b1_offsets[ev_num], self.tau_offsets[ev_num])
            else:
                ev = self.generate_artificial_biexp_spike(
                    self.b1_offsets[ev_num],
                    self.rise_offsets[ev_num],
                    self.decay_offsets[ev_num],
                )
            events_overlay[idx, :] = ev
        return events_overlay

    def get_average_of_all_events(self, biexp=False):
        events_overlay = self.get_overlay_of_all_events(biexp)
        average_event = np.mean(events_overlay, axis=0)
        return average_event

    def num_events(self, rec=None):
        if rec is None:
            event_times = self.peak_times[self.time_type][~np.isnan(self.peak_times[self.time_type])]
        else:
            event_times = self.peak_times[self.time_type][rec][~np.isnan(self.peak_times[self.time_type][rec])]

        return event_times.size

    # Insert events into data
    # ----------------------------------------------------------------------------------------------------------------------------------------------------

    def fill_im_array_with_events(self):
        """
        use self.spike_sample_idx to index out from vm_array and update_with_varying_amplitudesinsert spike_

        updated to fill with biexp function if selected
        """
        b1_offsets = []
        self.all_rec_b1_offsets = []  # TODO: added later, was easier to split this per rec and keep two
        tau_offsets = []
        rise_offsets = []
        decay_offsets = []
        area_under_curves = []
        for rec in range(self.num_recs):
            rec_b1_offsets = []

            for col in range(self.max_num_spikes):
                idx = self.spike_sample_idx[rec, col]
                if ~np.isnan(idx):
                    idx = idx.astype(int)

                    if self.event_type == "monoexp":
                        b1_offset = np.random.randint(1, 10) if self.vary_amplitudes_and_tau else 1
                        tau_offset = (1 / np.random.randint(1, 10)) if self.vary_amplitudes_and_tau else 1

                        event, area_under_curve = self.generate_artificial_spike(b1_offset, tau_offset)
                        self.im_array[rec, idx : idx + self.event_samples] = event

                        tau_offsets.append(tau_offset)
                        area_under_curves.append(area_under_curve)

                    elif self.event_type == "biexp":
                        b1_offset = 10  #
                        rise_and_decay_offset = (
                            (np.random.choice([0.05, 0.3, 0.9], 1)) if self.vary_amplitudes_and_tau else 1
                        )  # restrict to 3 options for 3 template tests

                        event = self.generate_artificial_biexp_spike(
                            b1_offset,
                            rise_and_decay_offset / self.rise_div,
                            rise_and_decay_offset,
                        )
                        self.im_array[rec, idx : idx + self.event_samples] = event
                        self.im_array[rec][idx] = (
                            self.resting_im + 0.001
                        )  # so baseline position is detected correctly for event fit start

                        rise_offsets.append(rise_and_decay_offset / self.rise_div)
                        decay_offsets.append(rise_and_decay_offset)

                    b1_offsets.append(b1_offset)
                    rec_b1_offsets.append(b1_offset)

            self.all_rec_b1_offsets.append(rec_b1_offsets)

        self.area_under_curves = np.array(area_under_curves)
        self.b1_offsets = np.array(b1_offsets)

        if self.event_type == "monoexp":
            self.tau_offsets = np.array(tau_offsets)

        elif self.event_type == "biexp":
            self.rise_offsets = np.array(rise_offsets)
            self.decay_offsets = np.array(decay_offsets)


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Artificial Ri Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestArtificialRiData(GenerateArtificialData):
    """ """

    def __init__(self, num_recs=75):
        super().__init__(num_recs)

        self.sag_hump_amplitude = 10
        self.sag_hump_idx = int(
            np.floor(
                (
                    ((self.num_samples / self.time_stop) * self.stop_time)
                    - (self.num_samples / self.time_stop) * self.start_time
                )
            )
            / 2
        )
        self.sag_hump_peaks = utils.np_empty_nan((self.num_recs, 1))
        self.peak_deflections = utils.np_empty_nan(self.num_recs)

        self.insert_injection_to_vm_arrays()
        self.norm_start_stop_times = self.gen_norm_inj_start_stoptime()
        self.cum_start_stop_times = self.gen_cum_inj_start_stop_time()

        self.add_in_sag_hump()

    def insert_injection_to_vm_arrays(self):
        for rec in range(0, self.num_recs):
            self.vm_array[rec][self.start_idx : self.stop_idx] += self.injection_[rec]

    def add_in_sag_hump(self):
        for rec in range(0, self.num_recs):
            if self.current_injection_amplitude[rec] > 0:
                self.vm_array[rec][self.sag_hump_idx] = self.vm_array[rec][self.sag_hump_idx] + self.sag_hump_amplitude
                self.sag_hump_peaks[rec] = self.sag_hump_amplitude
                self.peak_deflections[rec] = self.current_injection_amplitude[rec] + self.sag_hump_amplitude
            else:
                self.vm_array[rec][self.sag_hump_idx] = self.vm_array[rec][self.sag_hump_idx] - self.sag_hump_amplitude
                self.sag_hump_peaks[rec] = -self.sag_hump_amplitude
                self.peak_deflections[rec] = self.current_injection_amplitude[rec] - self.sag_hump_amplitude

    def gen_norm_inj_start_stoptime(self):
        norm_injection_start_stoptimes = np.tile([self.start_time, self.stop_time], self.num_recs)
        return norm_injection_start_stoptimes

    def gen_cum_inj_start_stop_time(self):
        cumu_injection_start_stoptimes = utils.np_empty_nan((self.num_recs, 2))
        for rec in range(0, self.num_recs):
            cumu_injection_start_stoptimes[rec] = [
                self.start_time + self.cum_min_max_time[0][0],
                self.stop_time + self.cum_min_max_time[0][0],
            ]
        return cumu_injection_start_stoptimes


# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Artificial Data Manipulation Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------


class TestDataTools(GenerateArtificialData):
    def __init__(self, num_recs):
        super().__init__(num_recs)

        self.n_freqs = utils.np_empty_nan((self.num_recs, 1))
        self.hz_to_add = []
        self.vm_array = utils.np_empty_nan((self.num_recs, self.num_samples))

        self.generate_test_frequency_spectra()

    def generate_test_frequency_spectra(self, dist=1000):
        """
        Generate noisey data made up of added together sine waves at various frequencies. The Hz of sine waves to insert is determined
        with a minimum distance so that when used to generate data for testing filtering, the seperation between frequencies is sufficient to
        avoid filter roll-off effects.

        NOTE: this is very similar to test_utils.generate_test_frequency_spectra() but not similar enough to join, but quite DRY
        """
        freq_near_nyquist = (self.fs / 2) * 0.95  # dont insert freqs too close to nyquist

        for rec in range(self.num_recs):
            x = np.linspace(0, 2 * np.pi, self.num_samples)
            rec_n_freqs = int((freq_near_nyquist / dist) - 1)
            rec_hz_to_add = test_utils.random_int_with_minimum_distance(
                min_val=0, max_val=freq_near_nyquist, n=rec_n_freqs, min_distance=dist
            )
            y = np.zeros(self.num_samples)
            for hz in rec_hz_to_add:
                y = y + np.sin(x * hz)

            self.vm_array[rec] = y
            self.n_freqs[rec] = rec_n_freqs
            self.hz_to_add.append(rec_hz_to_add)


class TestSingleSineWave:
    def __init__(self):
        self.num_samples = 8000

    def generate_sine(self, freq, amplitude):
        """ """
        offset = 0
        sin_x = np.linspace(0, 2 * np.pi / freq, self.num_samples + 1)
        sin_y = np.sin(sin_x * freq) * amplitude
        fs = core_analysis_methods.calc_fs(sin_x)
        ts = 1 / fs

        idxs = {
            "max": int(self.num_samples / 4),
            "mid": int(self.num_samples / 2),
            "min": int(self.num_samples / 0.75),
        }
        params = {
            "min": np.min(sin_y),
            "max": np.max(sin_y),
            "offset": 0,
            "num_samples": self.num_samples,
        }
        return sin_x, sin_y, fs, ts, params, idxs


def get_simple_known_event_data():
    # improve docs here and on the
    # num_left_edge_samples = 0, window_length_samples = 5
    # here the left edge is 0 and the right edge is num samples ( 5)
    #                    0  1  2  3  4  5  6  7  8  9   10  11  12  13  14  15  16   17   18   19   20   21 22 23 24
    im_array = np.array([0, 0, 5, 4, 3, 2, 1, 0, 0, 50, 40, 30, 20, 10, 0,  0,  500, 400, 300, 200, 100, 0, 0, 0, 0])
    im_array = np.vstack([im_array, im_array * 1000])

    event_info = [
        {
            "2": {
                "peak": {"idx": 2},
                "baseline": {"idx": 3},
                "half_width": {"rise_mid_idx": 3},
            },
            "9": {
                "peak": {"idx": 9},
                "baseline": {"idx": 11},
                "half_width": {"rise_mid_idx": 12},
            },
            "16": {
                "peak": {"idx": 16},
                "baseline": {"idx": 19},
                "half_width": {"rise_mid_idx": 20},
            },
        },
        {
            "2": {
                "peak": {"idx": 2},
                "baseline": {"idx": 3},
                "half_width": {"rise_mid_idx": 3},
            },
            "9": {
                "peak": {"idx": 9},
                "baseline": {"idx": 11},
                "half_width": {"rise_mid_idx": 12},
            },
            "16": {
                "peak": {"idx": 16},
                "baseline": {"idx": 19},
                "half_width": {"rise_mid_idx": 20},
            },
        },
    ]
    return im_array, event_info
