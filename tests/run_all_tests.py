import subprocess

print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Don't forget to run GUI tests (see README.md in tests_that_require_gui <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n")

main_tests = [

    "python -m pytest -v -p no:warnings ./test_burst_analysis.py >./test_burst_analysis_log.log",  # useful to be first as still contains some save data code
    "python -m pytest -v -p no:warnings ./test_curve_fitting.py >./test_curve_fitting_log.log",
    "python -m pytest -v -p no:warnings ./test_tables.py >./test_tables_log.log",
    "python -m pytest -v -p no:warnings ./test_data_tools.py >./test_data_tools_log.log",
    "python -m pytest -v -p no:warnings ./test_input_resistance.py >./test_input_res_log.log",
    "python -m pytest -v -p no:warnings ./test_spikecount_gui_analysis.py >./test_spkcnt_gui_log.log",
    "python -m pytest -v -p no:warnings ./test_skinetics_gui.py >./test_skinetics_gui_log.log",
    "python -m pytest -v -p no:warnings ./test_events_and_analysis_frequency_data.py >./test_events_and_analysis_frequency_data.log",
    "python -m pytest -v -p no:warnings ./test_events.py >./test_events_log_.log",
    "python -m pytest -v -p no:warnings ./test_configs.py >./test_configs_log.log",
    "python -m pytest -v -p no:warnings ./test_backup_options.py >./test_backup_options_log.log",
    "python -m pytest -v -p no:warnings ./test_event_detection.py >./test_event_detection_log.log",
    "python -m pytest -v -p no:warnings ./test_events_voltage_calc.py >./test_events_voltage_calc.log",
    "python -m pytest -v -p no:warnings ./test_core_analysis_methods.py >./test_core_analysis_methods_log.log",
    "python -m pytest -v -p no:warnings ./test_main_plot.py >./test_main_plot_log.log",
    "python -m pytest -v -p no:warnings ./test_graph_view_options.py >./test_graph_view_options_log.log",
    "python -m pytest -v -p no:warnings ./test_gui.py >./test_gui_log.log",
    "python -m pytest -v -p no:warnings ./test_importdata.py >./test_importdata_log.log",
    "python -m pytest -v -p no:warnings ./test_linearregions_and_im_settings.py >./test_linearregions_log.log",
    "python -m pytest -v -p no:warnings ./test_skinetics.py >./test_skinetics_log.log",
    "python -m pytest -v -p no:warnings ./test_spikecalc.py >./test_spikecalc_log.log",
    "python -m pytest -v -p no:warnings ./test_startup_widget_disabled.py >./test_startup_widget_log.log",
    "python -m pytest -v -p no:warnings ./test_table_im_checkboxes.py >./test_checkboxes_log.log",
]

for test in main_tests:
    subprocess.call(test, shell=True)