

def get_settings(slow_or_fast, data_type, num_recs=1):
        """
        """
        assert slow_or_fast is not None, "Set slow_or_fast!"


        if data_type == "events_one_record":  # TODO: rec is here!

            if slow_or_fast == "slow":
                settings = {
                    "num_recs": 1,
                    "num_samples": 1200000,
                    "time_stop": 120,
                    "min_num_spikes": 15,
                    "max_num_spikes": 50,
                    "start_stop_times": [[1, 5], [20, 30], [54.2, 78.2], [95.123, 95.32]]  # for omit times tests
                }

            elif slow_or_fast == "fast":
                settings = {
                    "num_recs": 1,
                    "num_samples": 25000,
                    "time_stop": 2.5,
                    "min_num_spikes": 5,
                    "max_num_spikes": 15,
                    "start_stop_times": [[1, 3], [5.4, 7.5], [10, 11], [13.2, 15]] if num_recs > 1 else [[0.1, 0.3], [0.54, 0.75], [1.0, 1.1], [2.2, 2.45]]  # TODO: lower num recs
                }

        elif "events_multi_record_biexp" in data_type:

            if slow_or_fast == "slow":
                settings = {
                    "num_recs": 14,
                    "num_samples": 1200000,
                    "time_stop": 15,
                   "min_num_spikes": 5,
                   "max_num_spikes": 15,
                }
            elif slow_or_fast == "fast":
                settings = {
                   "num_recs": 14,
                   "num_samples": 200000,
                   "time_stop": 2.5,
                   "min_num_spikes":  1,
                   "max_num_spikes":  5,
                }


        elif data_type in ["events_multi_record", "events_multi_record_norm", "events_multi_record_table"]:  # lot of re-used code

            if slow_or_fast == "slow":
                settings = {
                    "num_recs": 14,
                    "num_samples": 800000,
                    "time_stop": 15,
                   "min_num_spikes": 5,
                   "max_num_spikes": 15,
                    "start_stop_times": [[1, 3], [5.4, 7.5], [10, 11], [13.2, 15]]
                }
            elif slow_or_fast == "fast":
                settings = {
                   "num_recs": 14,  # do not change these
                   "num_samples": 15000,
                   "time_stop": 1.5,
                   "min_num_spikes":  3,
                   "max_num_spikes":  5,
                   "start_stop_times": [[0.1, 0.3], [0.54, 0.75], [1.0, 1.1]]
                }

        if data_type == "events_multi_record_table":
            if slow_or_fast == "fast":
                settings["min_num_spikes"] = 1
                settings["max_num_spikes"] = 2

        if data_type == "data_tools":
            if slow_or_fast == "slow":
                settings = {"num_recs": 75,
                            "recs_to_split": 8}
            elif slow_or_fast == "fast":
                settings = {"num_recs": 5,
                            "recs_to_split": 3}

        if data_type in ["Ri", "spkcnt", "skinetics", "skinetics_table"]:
            rec_from = num_recs
            if slow_or_fast == "slow":
                settings = {"num_recs": 75,
                            "rec_from": 4,
                            "rec_to": 50,
                            "max_num_spikes": 50,
                            "min_num_spikes": 5,
                            "manually_del": [[rec_from, 2, "m_one"], [rec_from, 0, "m_two"],  # higher first or plot peak idx becomes mismatched
                                             [rec_from + 3, 0, "m_three"], [rec_from + 8, 4, "m_four"]]
                            }
            elif slow_or_fast == "fast":
                settings = {"num_recs": 5,
                            "rec_from": 1,
                            "rec_to": 3,
                            "max_num_spikes": 8,
                            "min_num_spikes": 3,
                            "manually_del": [[rec_from, 2, "m_one"], [rec_from + 1, 1, "m_two"],  # higher first or plot peak idx becomes mismatched
                                             [rec_from + 1, 0, "m_three"]]
                            }

        if data_type == "skinetics_table":
            settings["min_num_spikes"] = 2
            settings["max_num_spikes"] = 5

        if data_type == "skinetics":
            if slow_or_fast == "slow":
                settings = {"num_recs": 10,
                            "num_samples": 64000,
                            "time_stop": 2,
                            "rec_from": 2,
                            "max_num_spikes": 50,
                            "min_spikes": 15,
                            "rec_to": 7,
                            "manually_del": [[0, 1, 1], [1, 0, 3], [6, 1, 5]],
                            "manually_sel": [[0, 0], [0, 1], [0, 2], [3, 0],
                                              [6, 0], [6, 1], [8, 0], [9, 0]],
                            }
            elif slow_or_fast == "fast":
                settings = {"num_recs": 5,
                            "num_samples": 16000,
                            "time_stop": 0.5,
                            "max_num_spikes": 15,
                            "min_spikes": 5,
                            "rec_from": 1,
                            "rec_to": 3,
                            "manually_del": [[2, 0, 2], [1, 0, 1]],  # so hacky, final idx 1 deletes the correct one
                            "manually_sel": [[1, 0], [1, 1], [2, 0]],
                            }

        if data_type == "curve_fitting_table":
                if slow_or_fast == "slow":
                    return ["reg_1", "reg_2", "reg_3", "reg_4", "reg_5", "reg_6"]
                elif slow_or_fast == "fast":
                    return ["reg_3"]

        if data_type == "curve_fitting":
            if slow_or_fast == "slow":
                settings = {
                    "decay_search_period": 1000,
                }
            elif slow_or_fast == "fast":
                settings = {
                    "decay_search_period": 250,
                }

        return settings

        # biexp -

        # test auc against axograph - no need to analyze everything.
        # max_slope events

