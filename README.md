Copyright (C) 2020-2025, Joseph J Ziminski.

Analysis code for Easy Electrophysiology software.
See www.easyelectrophysiology.com for all news and updates


Modules contain code for all current clamp, voltage clamp and curve fitting analysis for the
Easy Electrophysiology software package. 

importdata.py - methods for loading electrophysiological data and saving into Data class with
                calculation of data parameters
				
current_calc.py - methods for all current clamp analysis 
                  (e.g. action potential counting, thresholding, kinetics)
				  
event_analysis_master.py - methods for coordinating events analysis based on user options.
                           The actual data analysis for most of these methids is in voltage_calc.py
						   
voltage_calc.py - methods for events detection and kinetics analysis 

core_analysis_methods.py - code for curve fitting analysis and other shared analysis methods, 
                           also including AP thresholding.
