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
import neo
import numpy as np
from PySide2.QtWidgets import QInputDialog, QMessageBox
from utils import fonts_and_brushes
from utils import utils
from ephys_data_methods import core_analysis_methods

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Load Data With Neo
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Methods for loading in data. Generates a "Data" class that contains Vm, Im, and Time data.
# Uses the neo electrophysiology module. Handles 1, 2 or more channels (but forces user to input 2 channels
# when there are > 2. The main method "import_data" will return None if import fails.
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def import_data(full_filepath, file_ext, cfgs):
    """
    Import data using Neo and output a structure containing channel information and data.
    Currently only 2 channels are supported, if more are in the file let the user select.
    Converts units to mV, ms and pA.

    INPUTS:
        full_filepath: The full filepath to the data to load (including the ext on end)
        file_ext: The extension of the file (seperate for convenience)
        cfgs: config file

    OUTPUT:
        DataModel file including data arrays and file / data information.
    """

    # Load with neo and get primary and secondary channel index
    channels, reader, block_neo_file = load_file_with_neo(full_filepath, file_ext)
    if channels is None:
        return None

    channel_1, channel_1_idx, channel_2, channel_2_idx = get_channel_data_info(cfgs, channels, reader)
    if channel_1 is None:
        return None

    # Get channel type (Im or Vm), units and load data structure
    channel_1_type, channel_1_units = get_channel_type(channel_1)
    if not channel_1_type:
        return False

    if channel_2:
        channel_2_type, channel_2_units = get_channel_type(channel_2)
        if not channel_2_exists_and_is_different_type_to_channel_1(channel_1_type,
                                                                   channel_2_type):
            return False

        Data = ImportData(block_neo_file, 2, cfgs,
                          channel_1_type, channel_1_units, channel_1_idx,
                          channel_2_type, channel_2_units, channel_2_idx)
    else:
        Data = ImportData(block_neo_file, 1, cfgs,
                          channel_1_type, channel_1_units, channel_1_idx,
                          None, None, None)
    return Data


def channel_2_exists_and_is_different_type_to_channel_1(channel_1_type, channel_2_type):
    if not channel_2_type:
        return False

    if channel_1_type == channel_2_type:
        utils.show_messagebox("Load File Error",
                              "Cannot have two channels of the same data type (e.g. voltage, voltage). "
                              "Please load only one of the channels (File > File Loading Options > Select Channels to Load).")
        return False
    return True

def load_file_with_neo(full_filepath, file_ext):
    """
    Load the file with Neo. Return None if file cannot be loaded.

    OUTPUTS:
        channels: channels header from Neo raw output class
        reader: neo raw output class
        block_neo_file: block from the neo file
    """
    try:
        if file_ext.upper() == ".WCP":
            reader = neo.WinWcpIO(full_filepath)
        elif file_ext.upper() == ".ABF":
            reader = neo.AxonIO(full_filepath)
        elif file_ext.upper() == ".EDR":
            reader = neo.WinEdrIO(full_filepath)
        else:
            utils.show_messagebox("Cannot Determine filetype",
                                  "Cannot determine filetype. Currently supported filetyes\n"
                                  "are .abf and .wcp")
            return None, None, None
    except:
        utils.show_messagebox("Neo Load Error", "Could not load file. Check that the\n"
                                                "sampling rate is identical for all records \n"
                                                "and that read permission is granted.")
        return None, None, None

    block_neo_file = reader.read_block(signal_group_mode="split-all")  # if grouped, grouped signals go to first index destroying the order specified in reader.headers["signal_channels"]
    channels = reader.header["signal_channels"]

    return channels, reader, block_neo_file

def get_channel_data_info(cfgs, channels, reader):
    """
    Load channels from the raw neo file.
    Currently EE can handle only two channel types. If more are present, user is given option to select which channels to load.
    """
    if cfgs.file_load_options["select_channels_to_load"]["on"]:
        return get_users_default_channels(cfgs, reader)
    else:
        return get_default_channels(channels, reader)

def get_users_default_channels(cfgs, reader):
    """
    Be careful about None vs. 0 here, check explicitly for None
    """
    channel_1_idx = cfgs.file_load_options["select_channels_to_load"]["channel_1_idx"]
    channel_2_idx = cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"]

    try:
        channel_1 = reader.header["signal_channels"][channel_1_idx]
        channel_2 = None if channel_2_idx is None else reader.header["signal_channels"][channel_2_idx]

    except:
        utils.show_messagebox("Input Error",
                              "When selecting channels to load in 'File' > 'Load Options', "
                              "ensure the channels exist in the file.")
        return None, None, None, None

    return channel_1, channel_1_idx, channel_2, channel_2_idx

def get_default_channels(channels, reader):
    """
    """
    channel_2 = channel_2_idx = None

    if len(channels) > 2:
        channel_1_idx, channel_2_idx, channel_1, channel_2 = utils.get_channels_from_user_input(reader)

    elif len(channels) == 2:
        channel_1_idx = 0
        channel_2_idx = 1
        channel_1 = reader.header["signal_channels"][channel_1_idx]
        channel_2 = reader.header["signal_channels"][channel_2_idx]

    elif len(channels) == 1:
        channel_1_idx = 0
        channel_1 = reader.header["signal_channels"][channel_1_idx]

    return channel_1, channel_1_idx, channel_2, channel_2_idx


def get_channel_type(channel):
    """
    Find the channel type based on it's saved name or units.

    Often the channel name reflects the input number of the amplifier e.g. IN0 and is no use.
    In this case the units is used.
    """
    channel_type = channel[0].strip()
    channel_units = channel[4].strip()

    if "Vm" in channel_type or channel_units in ["mV", "V"]:
        channel_type = "Vm"
    elif "Im" in channel_type or channel_units in ["pA", "nA"]:
        channel_type = "Im"
    else:
        utils.show_messagebox("Type Error", "Cannot determine recording type. Please contact Easy Electrophysiology")
        return None, None
    return channel_type, channel_units

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Data Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class ImportData:
    """
    Extract Vm, Im and time arrays from Neo object.
    Note that depending on the software which recorded the data, time may increase cumulatively across records.
    Data is kept as 64-bit as is the usually fastest datatype to compute with numpy.

    Time offset:  For some file formats a slight offset is added to the time, meaning it will start at a non-zero value.
                  if this is present in the data, self.time_offset will be set to the offset and data changed at the model level.
                  cannot wrap dialog to this class or will throw pickle error on deepcopy at model level.

    """
    def __init__(self, block_neo_file, num_chans, cfgs,
                 channel_1_type, channel_1_units, channel_1_idx,
                 channel_2_type, channel_2_units, channel_2_idx):

        self.load_setting = cfgs.file_load_options["force_load_options"]
        self.num_recs = len(block_neo_file.segments)
        self.num_samples = len(block_neo_file.segments[0].analogsignals[0])
        self.fs = block_neo_file.segments[0].analogsignals[0].sampling_rate.magnitude  # strip quantities
        self.ts = block_neo_file.segments[0].analogsignals[0].sampling_period.magnitude
        self.time_units = str(block_neo_file.segments[0].t_start).split(" ")[1]
        self.vm_array = np.zeros((self.num_recs,
                                  self.num_samples))
        self.im_array = np.zeros((self.num_recs,
                                  self.num_samples))  # empty must be zero, if empty or nan cannot be properly plot by pyqtgraphs in the one-channel case
        self.time_array = np.zeros((self.num_recs,
                                    self.num_samples))
        self.num_data_channels = num_chans
        self.time_offset = False                      # in rare cases, some setups will add a tiny offset to time so it does not start at zero
        self.vm_units = None
        self.im_units = None
        self.t_start = None                           # start time (first record)
        self.t_stop = None                            # end time (final record)
        self.channel_1_type = channel_1_type
        self.channel_2_type = channel_2_type
        self.recording_type = None

        if self.load_setting is None and channel_1_type == "Vm" or \
                self.load_setting == "current_clamp":
            self.vm_array = self.extract_data_from_array(channel_1_idx, block_neo_file)
            self.vm_units = channel_1_units
            self.recording_type = "current_clamp"

        elif self.load_setting is None and channel_1_type == "Im" or \
                self.load_setting == "voltage_clamp":
            self.im_array = self.extract_data_from_array(channel_1_idx, block_neo_file)
            self.im_units = channel_1_units
            if self.num_recs == 1:
                self.recording_type = "voltage_clamp_1_record"
            else:
                self.recording_type = "voltage_clamp_multi_record"

        if channel_2_idx is not None:
            if self.load_setting is None and channel_2_type == "Vm" or \
                    self.load_setting == "voltage_clamp":
                self.vm_array = self.extract_data_from_array(channel_2_idx, block_neo_file)
                self.vm_units = channel_2_units

            elif self.load_setting is None and channel_2_type == "Im" or \
                    self.load_setting == "current_clamp":
                self.im_array = self.extract_data_from_array(channel_2_idx, block_neo_file)
                self.im_units = channel_2_units

        self.time_array, self.t_start, self.t_stop = self.extract_time_array(block_neo_file)
        self.check_and_clean_data()
        return

    def check_and_clean_data(self):
        """
        Convert V to mV,
        nA to pA,
        ms to s
        """
        if self.vm_units == "V":
            self.vm_array *= 1000

        if self.im_units == "nA":
            self.im_array *= 1000
            self.im_units = "pA"

        if self.time_units == "ms":
            self.t_start /= 1000
            self.t_stop /= 1000
            self.time_array /= 1000

        if self.t_start != 0:
            self.time_offset = np.array(self.t_start)

    def extract_time_array(self, block_neo_file):
        """
        """
        time_array = utils.np_empty_nan((self.num_recs,
                                         self.num_samples))
        for rec in range(self.num_recs):
            time_ = core_analysis_methods.generate_time_array(block_neo_file.segments[rec].t_start.magnitude,
                                                              block_neo_file.segments[rec].t_stop.magnitude,
                                                              self.num_samples,
                                                              self.ts)

            time_array[rec, :] = time_.squeeze()

        t_start = float(block_neo_file.segments[0].t_start)                   # get first and very last time point in case data needs cutting into records (cut_up_data() in data model)
        t_stop = float(block_neo_file.segments[self.num_recs - 1].t_stop)     # convert to float to remove units formatting, zero idx

        return time_array, t_start, t_stop

    def extract_data_from_array(self, data_idx, block_neo_file):
        """
        extract Im or Vm from neo model (depending on analogsignal idx)
        """
        array = utils.np_empty_nan((self.num_recs,
                                    self.num_samples))
        for rec in range(self.num_recs):
            data = block_neo_file.segments[rec].analogsignals[data_idx].magnitude
            array[rec, :] = data.squeeze()

        return array



