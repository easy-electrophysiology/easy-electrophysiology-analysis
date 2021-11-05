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
import copy

import neo
import numpy as np
from PySide2.QtWidgets import QInputDialog, QMessageBox
from utils import fonts_and_brushes
from utils import utils
from ephys_data_methods import core_analysis_methods
from dialog_menus.importdata_show_channels import ShowChannelsWarning, ViewChannels

# TODO: this module is getting big with quite a lot of args passed around, but is not not seem right to make it a class
#       as it is not persistant and only used once to load a file.

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Load Data With Neo
# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Methods for loading in data. Generates a "Data" class that contains Vm, Im, and Time data.
# Uses the neo electrophysiology module. Handles 1, 2 or more channels (but forces user to input 2 channels
# when there are > 2. The main method "import_data" will return None if import fails.
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def import_data(full_filepath, file_ext, cfgs, mw):
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

    TODO: this module has gotten quite big, looked into making a class but it is a little pointless
    as it is not persistent, just called once when loading a file. Use of SimpleNameSpace to simplify
    args sufficient for now.

    """
    # Load with neo and get primary and secondary channel index
    channels, reader, neo_block = load_file_with_neo(full_filepath, file_ext)
    if channels is None:
        return None

    channel_1, channel_1_idx, channel_2, channel_2_idx = get_channel_data_info(cfgs, channels, reader, mw)
    if channel_1 is None or channel_is_proxy(channel_1_idx, neo_block,
                                             reader, cfgs, mw, channel_1=True):
        return None

    # Get channel type (Im or Vm), units and load data structure
    channel_1_units, channel_1_type = get_channel_type(channel_1,
                                                       cfgs,
                                                       neo_block,
                                                       channel_1_idx,
                                                       reader, mw)
    if not channel_1_type:
        return False

    if channel_2 and not channel_is_proxy(channel_2_idx,
                                          neo_block,
                                          reader, cfgs, mw):

        channel_2_units, channel_2_type = get_channel_type(channel_2,
                                                           cfgs,
                                                           neo_block,
                                                           channel_2_idx,
                                                           reader, mw)

        if not channel_2_exists_and_is_different_type_to_channel_1(channel_1_type,
                                                                   channel_2_type,
                                                                   cfgs, reader, mw):
            return False

        Data = ImportData(neo_block, 2, cfgs,
                          channel_1_type, channel_1_units, channel_1_idx,
                          channel_2_type, channel_2_units, channel_2_idx,
                          reader, file_ext)
    else:
        Data = ImportData(neo_block, 1, cfgs,
                          channel_1_type, channel_1_units, channel_1_idx,
                          None, None, None,
                          reader, file_ext)
    return Data

def load_file_with_neo(full_filepath, file_ext):
    """
    Load the file with Neo. Return None if file cannot be loaded.

    OUTPUTS:
        channels: channels header from Neo raw output class
        reader: neo raw output class
        neo_block: block from the neo file
    """
    try:
        if file_ext.upper() == ".ABF":
            reader = neo.AxonIO(full_filepath)
        elif file_ext.upper() in [".AXGX", ".AXGD"]:
            reader = neo.AxographIO(full_filepath)
        elif file_ext.upper() == ".WCP":
            reader = neo.WinWcpIO(full_filepath)
        elif file_ext.upper() == ".EDR":
            reader = neo.WinEdrIO(full_filepath)
        else:
            utils.show_messagebox("Cannot Determine filetype",
                                  "Cannot determine filetype. Currently supported filetypes are:\n"
                                  ".abf, .axgx, .axgd, .wcp, .edr")
            return None, None, None
    except:
        utils.show_messagebox("Neo Load Error", "Could not load file. Check that the "
                                                "sampling rate is identical for all records "
                                                "and that read permission is granted.")
        return None, None, None

    neo_block = reader.read_block(signal_group_mode="split-all")  # if grouped, grouped signals go to first index destroying the order specified in reader.headers["signal_channels"]
    channels = reader.header["signal_channels"]

    return channels, reader, neo_block

# Processing Channels --------------------------------------------------------------------------------------------------------------------------------

def channel_is_proxy(channel_idx, neo_block, reader, cfgs, mw, channel_1=False):
    """
    For some filetypes (possibly just .abf generated by IGOR) self._raw_signals is shorter ('igor_abf_too_many_channels.abf in test_data')
    than the nubmer of channels neo can find. In this instance neo will fill with a proxy analogsignals
    but this will crash data loading class. Here check if any are proxy (abence of data i.e. 'manitude attribute)
    and inform the user that this channel does not exist.

    Show error message if channel 1 index is tested (if another channel fails, jsut channel 1 willl be loaded)
    TODO: confirm filetypes this works on, contact neo
    """
    channel_is_proxy = not hasattr(neo_block.segments[0].analogsignals[channel_idx], 'magnitude')

    if channel_1 and channel_is_proxy:
        show_warning(mw, cfgs, reader,
                     "Load File Error",
                     "Please ensure the data channel exists in the file. "
                     "Please contact support@easyelectrophysiology.com if this channel definitely exists.")
    return channel_is_proxy

def channel_2_exists_and_is_different_type_to_channel_1(channel_1_type, channel_2_type, cfgs, reader, mw):
    if not channel_2_type:
        return False

    if channel_1_type == channel_2_type:
        show_warning(mw, cfgs, reader,
                     "Load File Error",
                     "Cannot have two channels of the same data type (e.g. voltage, voltage). "
                     "Please load only one of the channels (File > File Loading Options > Select Channels to Load).")
        return False
    return True

def get_channel_data_info(cfgs, channels, reader, mw):
    """
    Load channels from the raw neo file.
    Currently EE can handle only two channel types. If more are present, user is given option to select which channels to load.
    """
    if cfgs.file_load_options["select_channels_to_load"]["on"]:
        return get_users_default_channels(cfgs, reader, mw)
    else:
        return get_default_channels(cfgs, channels, reader)

def get_users_default_channels(cfgs, reader, mw):
    """
    Be careful about None vs. 0 here, check explicitly for None
    """
    channel_1_idx = cfgs.file_load_options["select_channels_to_load"]["channel_1_idx"]
    channel_2_idx = cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"]

    try:
        channel_1 = reader.header["signal_channels"][channel_1_idx]
        channel_2 = None if channel_2_idx is None else reader.header["signal_channels"][channel_2_idx]
    except:
        show_warning(mw, cfgs, reader,
                     "Input Error",
                     "When selecting channels to load in 'File' > 'Load Options', "
                     "ensure the channels exist in the file.")
        return None, None, None, None

    return channel_1, channel_1_idx, channel_2, channel_2_idx

def get_default_channels(cfgs, channels, reader):
    """
    """
    channel_2 = channel_2_idx = None
    num_channels = len(channels)

    if num_channels > 2:
        channel_1_idx, channel_2_idx, channel_1, channel_2 = get_channels_from_user_input(cfgs, reader)

    elif num_channels in [1, 2]:
        channel_1_idx = 0
        channel_1 = reader.header["signal_channels"][channel_1_idx]

        if num_channels == 2:
            channel_2_idx = 1
            channel_2 = reader.header["signal_channels"][channel_2_idx]

    return channel_1, channel_1_idx, channel_2, channel_2_idx


def get_channel_type(channel, cfgs, neo_block, channel_idx, reader, mw):
    """
    Find the channel type based on it's saved name or units.

    Often the channel name reflects the input number of the amplifier e.g. IN0 and is no use.
    In this case the units is used.

    TODO: depreciated using channel type to identify channel as too vague. Now specify from
    units type to clearly force units to pA or mV.
    """
    channel_units = channel[4].strip()

    if channel_units == "\x02mV":  # TODO: not the best way to handle edge cases. As soon as a second one arises, refactor
        channel_units = "mV"

    channel_units, channel_type = get_channel_type_from_channel_units(channel_units, cfgs,  neo_block, channel_idx)

    if channel_units == "not_recognised":
        channel_units, channel_type = get_user_input_channel_units(channel[4].strip(), cfgs, neo_block, channel_idx)

    return channel_units, channel_type

def get_channel_type_from_channel_units(channel_units, cfgs,  neo_block, channel_idx):
    """
    """
    if channel_units == "mV":
        channel_type = "Vm"
    elif channel_units == "pA":
        channel_type = "Im"
    elif channel_units in ["fA", "nA", "uA", "mA", "A", "pV", "nV", "uV", "V"]:  # TODO: add to global configs as copied in importdata_show_channels
        channel_units, channel_type = process_non_default_channel_units(channel_units, cfgs,  neo_block, channel_idx)
        if channel_units is None:
            return None, None
    else:
        return "not_recognised", False

    return channel_units, channel_type

def get_user_input_channel_units(channel_units, cfgs, neo_block, channel_idx):

    orig_channel_units = channel_units
    while True:
        channel_units = utils.get_user_input("Channel Not Found",
                                             "Channel {0} units ('{1}') were not recognised. Please enter the correct units below.\n"
                                             "Units must be one of: fA, pA, nA, uA, mA, A, pV, nV, uV, mV, V".format(channel_idx + 1,
                                                                                                                     orig_channel_units),
                                             align_center=True)

        channel_units, channel_type = get_channel_type_from_channel_units(channel_units, cfgs,  neo_block, channel_idx)

        if channel_units == "not_recognised":
            utils.show_messagebox("Units Error",
                                  "Input units were not recognised. Please try again or press cancel.")
            continue

        elif channel_units is None:
            return None, None
        else:
            break

    return channel_units, channel_type

def process_non_default_channel_units(channel_units, cfgs, neo_block, channel_idx):
    """
    Handle channels not in pA or mV

    For some files the channel is specifier at the amplifier as 'A' generically. This is taken as the units for
    the data even though typically they are always recorded in pA. If such a file is encountered and default behaviour
    is not specified, prompt the user to set defaults, with an additional check that the header makes sense (in some cases e.g. Axograph hiles
    the header is in A but data is scaled to nA; see show_bad_header_units_warning())
    Otherwise, load the file as per the defaults.

    Data are converted at the Data class level depending on the channel_units type, so changing units to pA here means they
    will not be converted later.
    """
    if channel_units in ["fA", "nA", "uA", "mA", "A"]:  # TODO: bit verbose but explicit and removes dependency on calling function
        channel_type = "Im"
    elif channel_units in ["pV", "nV", "uV", "V"]:
        channel_type = "Vm"

    if channel_type == "Im" and cfgs.file_load_options["default_im_units"]["on"]:
        channel_units = "pA" if cfgs.file_load_options["default_im_units"]["assume_pa"] else cfgs.file_load_options["default_im_units"]["pa_unit_to_convert"]

    elif channel_type == "Vm" and cfgs.file_load_options["default_vm_units"]["on"]:
        channel_units = "mV" if cfgs.file_load_options["default_vm_units"]["assume_mv"] else cfgs.file_load_options["default_vm_units"]["mv_unit_to_convert"]

    else:
        show_bad_header_units_warning(channel_type, channel_units, neo_block, channel_idx)
        channel_units = channel_type = None

    return channel_units, channel_type

def get_channels_from_user_input(cfgs, reader):
    """
    """
    while True:

        user_input_channels = utils.get_user_input("Channel Warning",
                                                   "Easy Electrophysiology does not currently support recordings of more than two channels.\n"
                                                   "Type 'view' to see a full list of channels.\n\n"
                                                   "Please input the channels you would like to open, separated by a comma.\n"
                                                   "This can be automated at 'File' > 'File Loading Options' > 'Select Channels to Load'",
                                                   align_center=True)

        if user_input_channels.lower() in ["view", "'view'"]:
            ViewChannels(None, cfgs, reader.header["signal_channels"], exec=True)
            continue

        if user_input_channels == "":  # catch user cancel
            return None, None, None, None

        user_input_channels = utils.check_comma_seperated_user_input_and_extract_ints(user_input_channels,
                                                                                      allowed_chars=[","])
        if user_input_channels is None:
            continue

        # Check number of channels
        if len(user_input_channels) == 1:
            if user_input_channels[0] == ",":
                utils.show_messagebox("Input error",
                                      "Please input a number.")
                continue
            else:
                channel_2_input = False

        elif len(user_input_channels) == 2:
            channel_2_input = True

        if len(user_input_channels) not in [1, 2] or \
                0 in user_input_channels or \
                (channel_2_input and user_input_channels[0] == user_input_channels[1]):
            utils.show_messagebox("Input Error",
                                  "Please input up to 2 numbers in the range 1 to num channels.")
            continue

        try:
            channel_1_idx = int(user_input_channels[0]) - 1
            channel_1 = reader.header["signal_channels"][channel_1_idx]
            if channel_2_input:
                channel_2_idx = int(user_input_channels[1]) - 1  # assume user input is not zero indexed
                channel_2 = reader.header["signal_channels"][channel_2_idx]
            else:
                channel_2_idx = None
                channel_2 = None
            break

        except:
            utils.show_messagebox("Channel Error", "Cannot use input. "
                                  "Please ensure input is integer within the range "
                                  "of available data channels.")

    return channel_1_idx, channel_2_idx, channel_1, channel_2

# Handle non-pA or mV channel types ------------------------------------------------------------------------------------------------------------------

def show_bad_header_units_warning(channel_type, channel_units, neo_block, channel_idx):
    """
    Prompt the user to set the correct settings for re-scaling data that is not in pA or mV.

    In some cases (typically Axograph files) the header reads A but the data is scaled to nA. Here guess the most likely
    scaling based on the range of the data and prompt the user if the guessed unit type does not match the header unit type.
    """
    expected_units = "pA" if channel_type == "Im" else "mV"
    message_string = "The units in the file header are {0} but are expected to be {1}. Please specify default " \
                     "file loading options in 'File' > 'File Loading Options' > Specify Default Unit Handling'.".format(channel_units,
                                                                                                                        expected_units)

    guessed_units = check_if_header_units_are_correct(neo_block, channel_idx, channel_type)
    if guessed_units != channel_units:
        extra_warning = "\nWARNING! Header units were read as {0} but are most likely {1}. Try {1} first in File Load Options".format(channel_units,
                                                                                                                                      guessed_units)
        message_string += extra_warning

    utils.show_messagebox("Channel Units Error",
                          message_string)

def check_if_header_units_are_correct(neo_block, channel_idx, im_or_vm):
    """
    Typically data properly scaled to Im / mV will be in the 0-2000 range. The only instance
    this could cause problems is for very large stimulus registering > 2000, but these are rarely seen pA / mV.
    """
    channel_data = neo_block.segments[0].analogsignals[channel_idx].magnitude
    channel_range = max(channel_data) - min(channel_data)

    if im_or_vm == "Im":
        unit_ranges = {"fA": (2000, 9999),
                       "pA": (1, 1999),
                       "nA": (0.001, 0.999),
                       "uA": (0.000001, 0.000999),
                       "mA": (0.000000001, 0.000000999),
                       "A":  (0.000000000001, 0.000000000999),
                       }
    elif im_or_vm == "Vm":
        unit_ranges = {"pV": (100000, 999999),
                       "nV": (10000, 99999),
                       "uV": (2000, 9999),
                       "mV": (1, 1999),
                       "V": (0.001, 0.999),
                       }

    for unit, range_ in unit_ranges.items():
        if range_[0] < abs(channel_range) < range_[1]:
            return unit
    return "undefined: contact Easy Electrophysiology."

def show_warning(mw, cfgs, reader, title, text):
    """
    Show failed load warning with button to show all channels
    """
    ShowChannelsWarning(mw, cfgs, reader.header["signal_channels"], title, text)

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
    def __init__(self, neo_block, num_chans, cfgs,
                 channel_1_type, channel_1_units, channel_1_idx,
                 channel_2_type, channel_2_units, channel_2_idx,
                 reader, file_ext):

        self.load_setting = cfgs.file_load_options["force_load_options"]
        self.num_recs = len(neo_block.segments)
        self.num_samples = len(neo_block.segments[0].analogsignals[0])
        self.fs = neo_block.segments[0].analogsignals[0].sampling_rate.magnitude  # strip quantities
        self.ts = neo_block.segments[0].analogsignals[0].sampling_period.magnitude
        self.time_units = str(neo_block.segments[0].t_start).split(" ")[1]
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
        self.tags = ""
        self.all_channels = reader.header["signal_channels"]

        if self.load_setting is None and channel_1_type == "Vm" or \
                self.load_setting == "current_clamp":
            self.vm_array = self.extract_data_from_array(channel_1_idx, neo_block)
            self.vm_units = channel_1_units
            self.recording_type = "current_clamp"

        elif self.load_setting is None and channel_1_type == "Im" or \
                self.load_setting == "voltage_clamp":
            self.im_array = self.extract_data_from_array(channel_1_idx, neo_block)
            self.im_units = channel_1_units
            if self.num_recs == 1:
                self.recording_type = "voltage_clamp_1_record"
            else:
                self.recording_type = "voltage_clamp_multi_record"

        if channel_2_idx is not None:
            if self.load_setting is None and channel_2_type == "Vm" or \
                    self.load_setting == "voltage_clamp":
                self.vm_array = self.extract_data_from_array(channel_2_idx, neo_block)
                self.vm_units = channel_2_units

            elif self.load_setting is None and channel_2_type == "Im" or \
                    self.load_setting == "current_clamp":
                self.im_array = self.extract_data_from_array(channel_2_idx, neo_block)
                self.im_units = channel_2_units

        else:
            self.handle_generate_axon_protocol(reader, cfgs)

        self.time_array, self.t_start, self.t_stop = self.extract_time_array(neo_block)
        self.check_and_clean_data()
        self.update_tags(reader, file_ext)
        return

    def check_and_clean_data(self):
        """
        Convert V to mV,
        nA to pA,
        ms to s

        If force analysis type the units may be swapped, in which case swap the conversion tables
        """
        if self.time_units == "ms":
            self.t_start /= 1000
            self.t_stop /= 1000
            self.time_array /= 1000
            self.time_units = "s"

        if self.t_start != 0:
            self.time_offset = np.array(self.t_start)

        if self.im_units == "mV" or self.vm_units == "pA":
            return

        conversion_to_pa_table = core_analysis_methods.get_conversion_to_pa_table()
        conversion_to_mv_table = core_analysis_methods.get_conversion_to_mv_table()

        if self.im_units and self.im_units != "pA":
            self.im_array *= conversion_to_pa_table[self.im_units]
            self.im_units = "pA"

        if self.vm_units and self.vm_units != "mV":
            self.vm_array *= conversion_to_mv_table[self.vm_units]
            self.vm_units = "mV"

    def extract_time_array(self, neo_block):
        """
        """
        channel_time_check = True
        time_array = utils.np_empty_nan((self.num_recs,
                                         self.num_samples))
        for rec in range(self.num_recs):
            time_ = core_analysis_methods.generate_time_array(neo_block.segments[rec].analogsignals[0].t_start.magnitude,
                                                              neo_block.segments[rec].analogsignals[0].t_stop.magnitude,
                                                              self.num_samples,
                                                              self.ts)
            time_array[rec, :] = time_.squeeze()

            if not self.channel_times_are_equal(neo_block, rec):
                channel_time_check = False

        t_start = float(neo_block.segments[0].analogsignals[0].t_start)                   # get first and very last time point in case data needs cutting into records (cut_up_data() in data model)
        t_stop = float(neo_block.segments[self.num_recs - 1].analogsignals[0].t_stop)     # convert to float to remove units formatting, zero idx

        if not channel_time_check:
            utils.show_messagebox("Time Error",
                                  "The timing of channel 1 and channel 2 are not the same. The time units for channel 2 will be incorrect. Please contact support@easyelectrophysiology.com")

        return time_array, t_start, t_stop

    def channel_times_are_equal(self, neo_block, rec):
        """
        It is assumed all analogsignals have the same start / stop time. These should always be Im / Vm traces recorded at the same time
        so it would be very unexpected to find a case in which the timings are different.
        """
        if len(neo_block.segments[rec].analogsignals) == 2:
            if (neo_block.segments[rec].analogsignals[0].t_start.magnitude !=
                    neo_block.segments[rec].analogsignals[1].t_start.magnitude or
                    neo_block.segments[rec].analogsignals[0].t_stop.magnitude !=
                    neo_block.segments[rec].analogsignals[1].t_stop.magnitude):
                return False
        return True

    def extract_data_from_array(self, data_idx, neo_block):
        """
        extract Im or Vm from neo model (depending on analogsignal idx)
        """
        array = utils.np_empty_nan((self.num_recs,
                                    self.num_samples))
        for rec in range(self.num_recs):
            data = neo_block.segments[rec].analogsignals[data_idx].magnitude
            array[rec, :] = data.squeeze()

        return array

    def extract_axon_protocol(self, reader):
        """
        https://neo.readthedocs.io/en/stable/io.html#neo.io.AxonIO
        """
        array = utils.np_empty_nan((self.num_recs,
                                    self.num_samples))
        protocol = reader.read_protocol()
        units = protocol[0].analogsignals[0].units.dimensionality.string
        for rec in range(self.num_recs):
            array[rec] = np.squeeze(protocol[rec].analogsignals[0].magnitude)

        return array, units

    def handle_generate_axon_protocol(self, reader, cfgs):
        """
        Coordinate the generation of axon protocol. This must be done in current clamp mode. If settings are correct,
        the Im protocol will generated and loaded to the second channel, handling any errors.
        """
        if cfgs.file_load_options["select_channels_to_load"]["on"] and \
            cfgs.file_load_options["generate_axon_protocol"]:

            if self.recording_type != "current_clamp":
                utils.show_messagebox("Axon Protocol Error",
                                      "Must be in current clamp mode to load Axon Im protocol from header")
                return

            try:
                im_array, im_units = self.extract_axon_protocol(reader)
            except:
                utils.show_messagebox("Axon Protocol Error",
                                      "Could not generate Axon protocol. Please contact support@easyelectrophysiology.com")
                return

            self.im_units = im_units
            self.im_array = im_array
            self.channel_2_type = "Im"
            self.num_data_channels = 2

    def update_tags(self, reader, file_ext):
        """
        Update self.tags with tags (Axon Instruments feature only)
        """
        if file_ext.upper() == ".ABF":
            all_tags = ""
            for i in range(len(reader._axon_info["listTag"])):
                tag = reader._axon_info["listTag"][i]["sComment"].decode("utf-8").strip() + " "
                if i > 0:
                    tag = "Tag {0}: ".format(str(i + 1)) + tag
                all_tags += tag
            self.tags = all_tags
