class TestCfgs:
    def __init__(self):
        self.main = {
            'rec_from': None,
            'rec_to': None,
        }

        self.spkcnt = {
            'man_thr_value': None,
            'auto_thr_amp': 10,
            'auto_thr_rise': 0.01,
            'auto_thr_fall': -0.01,
            'auto_thr_width': 5,
            'upperplot_bl_lowerbound': 0,
            'upperplot_bl_upperbound': 0.1,
            'upperplot_exp_lowerbound': 0.5,
            'upperplot_exp_upperbound': 0.8,
            'lowerplot_bl_lowerbound': 0,
            'lowerplot_bl_upperbound': 0.1,
            'lowerplot_exp_lowerbound': 0.5,
            'lowerplot_exp_upperbound': 0.8,
        }

        self.bounds = {
            'upperplot_bl_lowerbound': 0,
            'upperplot_bl_upperbound': 0.1,
            'upperplot_exp_lowerbound': 0.5,
            'upperplot_exp_upperbound': 0.8,
            'lowerplot_bl_lowerbound': 0,
            'lowerplot_bl_upperbound': 0.1,
            'lowerplot_exp_lowerbound': 0.5,
            'lowerplot_exp_upperbound': 0.8,
        }

        self.skinetics = {
            'analyse_within_bounds': False,
            'man_thr_value': None,
            'threshold_type': 'auto_record',
            'fresh_file_load': False,  # used to update skinetics tabledata when a new file is loaded so a new tabledata can be appended
            'upper_bl_lr_lowerbound': None,
            'upper_bl_lr_upperbound': None,
            'upper_exp_lr_lowerbound': None,
            'upper_exp_lr_upperbound': None,
            'lower_bl_lr_lowerbound': None,
            'lower_bl_lr_upperbound': None,
            'lower_exp_lr_lowerbound': None,
            'lower_exp_lr_upperbound': None,

            'thr_method': 'first_deriv',
            'first_deriv_max_or_cutoff': 'max',
            'third_deriv_max_or_cutoff': 'max',
            'first_deriv_cutoff': 0,
            'third_deriv_cutoff': 0,
            'method_I_lower_bound': 1.00,
            'method_II_lower_bound': 1.00,
            'interp_200khz': False,
            'decay_to_thr_not_fahp': False,
            'rise_cutoff_low': 10,
            'rise_cutoff_high': 90,
            'decay_cutoff_low': 10,
            'decay_cutoff_high': 90,
            'fahp_start': 0 / 1000,
            'fahp_stop': 3 / 1000,
            'mahp_start': 10 / 1000,
            'mahp_stop': 50 / 1000,
            'low_spinbox_larger_than_high': False,
            'search_region_min': 2 / 1000,
        }

    def skinetics_params(self):
            output = {  # all returned in s for plotting, except for fwhm, rise_time and decay_time which are in ms for display on table
                'thr': None,
                'peak': None,
                'fahp': None,
                'mahp': None,
                'fwhm': None,
                'rise_time': None,
                'decay_time': None,
                'amplitude': None,
                }
            return output