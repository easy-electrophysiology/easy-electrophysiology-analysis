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
from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, Union

import neo
import numpy as np
from dialog_menus import load_csv_dialog
from dialog_menus.importdata_show_channels import ShowChannelsWarning, ViewChannels
from ephys_data_methods import core_analysis_methods, heka_loader
from neo.rawio.baserawio import _signal_channel_dtype
from utils import utils

if TYPE_CHECKING:
    from configs.configs import ConfigsClass
    from custom_types import ImOrVm, Int, NpArray64
    from mainwindow.mainwindow import MainWindow
    from neo.core.block import Block
    from neo.io.axonio import AxonIO
    from neo.io.baseio import BaseIO
    from numpy.typing import NDArray


class ImportData:
    """
    Import data using Neo and output a structure containing channel information
    and data. Currently, only 2 channels are supported, if more are in the file
    let the user select. Converts units to mV, ms and pA.

    INPUTS:
        full_filepath: The full filepath to the data to load (including the ext on end)
        file_ext: The extension of the file (separate for convenience)
        cfgs: config file

    OUTPUT:
        DataModel file including data arrays and file / data information.
    """

    def __init__(self, full_filepath: str, ext: str, cfgs: ConfigsClass, mw: MainWindow):
        self.full_filepath = full_filepath
        self.ext = ext.upper()
        self.cfgs = cfgs
        self.mw = mw

        self.channels, self.reader, self.neo_block = self.load_file_with_neo()

    def load_data(self) -> Optional[RawData]:
        if self.channels is None:
            return

        (
            channel_1,
            channel_1_idx,
            channel_2,
            channel_2_idx,
        ) = self.get_channel_data_info()

        if (
            channel_1 is None
            or channel_1_idx is None  # type narrow
            or self.channel_is_proxy(channel_1_idx, channel_1=True)
        ):
            return

        # Get channel type (Im or Vm), units and load data structure
        channel_1_units, channel_1_type = self.get_channel_type(channel_1, channel_1_idx)
        if not channel_1_type:
            return

        if channel_2 and channel_2_idx is not None and not self.channel_is_proxy(channel_2_idx):
            channel_2_units, channel_2_type = self.get_channel_type(channel_2, channel_2_idx)

            if not channel_2_type:
                return

            if self.ext == ".SMR":
                # SMR files can have channels with different properties in the same record.
                self.mw.show_messagebox(
                    "SMR Error",
                    "Only able to load one channel at a time for SMR files.\n"
                    "Select the channel to load at 'File' > 'File Loading Options'.",
                )
                return

            Data = RawData(
                self.neo_block,
                2,
                self.cfgs,
                channel_1_type,
                channel_1_units,
                channel_1_idx,
                channel_2_type,
                channel_2_units,
                channel_2_idx,
                self.reader,
                self.channels,
                self.ext,
            )
        else:
            Data = RawData(
                self.neo_block,
                1,
                self.cfgs,
                channel_1_type,
                channel_1_units,
                channel_1_idx,
                None,
                None,
                None,
                self.reader,
                self.channels,
                self.ext,
            )
        return Data

    # --------------------------------------------------------------------------------------
    # Load Data With Neo
    # --------------------------------------------------------------------------------------
    # Methods for loading in data. Generates a "Data" class that contains Vm, Im,
    # and Time data. Uses the neo electrophysiology module. Handles 1, 2 or more
    # channels (but forces user to input 2 channels when there are > 2. The main
    # method "import_data" will return None if import fails.
    # --------------------------------------------------------------------------------------

    def load_file_with_neo(self) -> Tuple[Optional[NDArray], Optional[BaseIO], Optional[Block]]:
        """"""
        if self.ext == ".DAT":
            channels, reader, neo_block = self.load_heka()

        elif self.ext == ".CSV":
            channels, reader, neo_block = self.load_csv()

        else:
            channels, reader, neo_block = self.load_common_neo_api_files()

        if channels is None:
            return None, None, None

        return channels, reader, neo_block

    def load_heka(self) -> Tuple[Optional[NDArray], Optional[BaseIO], Optional[Block]]:
        """"""
        # this will exec to get user selected Tree index
        heka = heka_loader.OpenHeka(self.mw, self.full_filepath)

        if not heka:
            return None, None, None

        reader, neo_block = heka.get_reader_and_neo_block(
            self.cfgs.file_load_options["heka_apply_zero_offset"],
            self.cfgs.file_load_options["heka_stimulus_reconstruction"],
        )

        if not reader:
            return None, None, None

        channels = reader.header["signal_channels"]

        return channels, reader, neo_block

    def load_csv(self) -> Tuple[Optional[NDArray], Optional[BaseIO], Optional[Block]]:
        """"""
        if self.cfgs.load_csv["show_on_file_load"]:
            dialog = load_csv_dialog.LoadCSV(self.mw, self.mw)
            dialog.exec()

            if not dialog.ok_pressed:
                return None, None, None
        try:
            reader = neo.CSVIO(
                self.full_filepath,
                ordered_datatype=self.cfgs.load_csv["ordered_datatype"],
                ordered_units=self.cfgs.get_load_csv_ordered_units(),
                interleaved_or_consecutive=self.cfgs.load_csv["interleaved_or_consecutive"],
                row_or_column=self.cfgs.load_csv["row_or_column"],
                has_header=self.cfgs.load_csv["has_header"],
                has_index=self.cfgs.load_csv["has_index"],
                time_from_sampling_rate=self.cfgs.load_csv["time_from_fs"],
            )

            neo_block = reader.read_block()
        except BaseException as e:
            self.mw.show_messagebox("CSV Error", f"{e}")
            return None, None, None

        channels = reader.header["signal_channels"]

        return channels, reader, neo_block

    def load_common_neo_api_files(
        self,
    ) -> Tuple[Optional[NDArray], Optional[BaseIO], Optional[Block]]:
        """
        Load the file with Neo. Return None if file cannot be loaded.

        OUTPUTS:
            channels: channels header from Neo raw output class
            reader: neo raw output class
            neo_block: block from the neo file
        """
        # Load the reader
        # try:
        if self.ext == ".ABF":
            reader = neo.AxonIO(self.full_filepath)
        elif self.ext in [".AXGX", ".AXGD"]:
            reader = neo.AxographIO(self.full_filepath)
        elif self.ext == ".WCP":
            reader = neo.WinWcpIO(self.full_filepath)
        elif self.ext == ".EDR":
            reader = neo.WinEdrIO(self.full_filepath)
        elif self.ext == ".H5":
            reader = neo.WaveSurferIO(self.full_filepath)
        elif self.ext == ".SMR":
            reader = neo.Spike2IO(self.full_filepath)
        elif self.ext == ".IBW":
            reader = neo.IgorIO(self.full_filepath)
        else:
            utils.show_messagebox(
                "Cannot Determine filetype",
                "Cannot determine filetype. Currently supported filetypes are:\n"
                ".abf, .axgx, .axgd, .csv, .h5 (Wavesurfer), "
                ".dat (HEKA), .smr, .wcp, .edr",
            )
            return None, None, None
        # except:
        #    utils.show_messagebox(
        #        "Neo Load Error",
        #        "Could not load file. Check that the "
        #        "sampling rate is identical for all records "
        #        "and that read permission is granted.",
        #    )
        #    return None, None, None

        if self.ext in [".H5", ".IBW"]:
            neo_block = reader.read_block()
        else:
            # if grouped, grouped signals go to first index destroying the order
            # specified in reader.headers["signal_channels"]
            neo_block = reader.read_block(signal_group_mode="split-all")

        if self.ext == ".IBW":
            channels = self.make_channels_from_analogsignals(neo_block)
            if channels is False:
                return None, None, None
        else:
            channels = reader.header["signal_channels"]

        return channels, reader, neo_block

    # Processing Channels
    # ----------------------------------------------------------------------------------

    def get_channel_data_info(
        self,
    ) -> Tuple[Optional[NDArray], Optional[Int], Optional[NDArray], Optional[Int]]:
        """
        Load channels from the raw neo file.
        Currently, EE can handle only two channel types.
        If more are present, user is given option to select which channels to load.
        """
        if self.cfgs.file_load_options["select_channels_to_load"]["on"]:
            return self.get_users_default_channels()
        else:
            return self.get_default_channels()

    def get_users_default_channels(
        self,
    ) -> Tuple[Optional[NDArray], Optional[Int], Optional[NDArray], Optional[Int]]:
        """
        Be careful about None vs. 0 here, check explicitly for None
        """
        channel_1_idx = self.cfgs.file_load_options["select_channels_to_load"]["channel_1_idx"]
        channel_2_idx = self.cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"]

        # For typing, it cannot tell logic flow from above function
        if self.channels is None:
            return None, None, None, None

        try:
            channel_1 = self.channels[channel_1_idx]
            channel_2 = None if channel_2_idx is None else self.channels[channel_2_idx]
        except:
            self.show_warning(
                "Input Error",
                "When selecting channels to load in 'File' > 'Load Options', " "ensure the channels exist in the file.",
            )
            return None, None, None, None

        return channel_1, channel_1_idx, channel_2, channel_2_idx

    def get_default_channels(
        self,
    ) -> Tuple[NDArray, Int, Optional[NDArray], Optional[Int]]:
        """ """
        channel_2 = channel_2_idx = None

        assert self.channels is not None, "Type Checking, self.channels cannot be None"
        num_channels = len(self.channels)

        if num_channels > 2:
            (
                channel_1_idx,
                channel_2_idx,
                channel_1,
                channel_2,
            ) = self.get_channels_from_user_input()

        elif num_channels in [1, 2]:
            channel_1_idx = 0
            channel_1 = self.channels[channel_1_idx]

            if num_channels == 2:
                channel_2_idx = 1
                channel_2 = self.channels[channel_2_idx]
        else:
            raise TypeError("num channels must be > 0")

        return channel_1, channel_1_idx, channel_2, channel_2_idx  # type: ignore

    def make_channels_from_analogsignals(self, neo_block) -> Union[Literal[False], NDArray]:
        """ """
        num_chan = len(neo_block.segments[0].analogsignals)

        all_channels = []

        for chan_idx in range(num_chan):
            assert neo_block.segments[0].analogsignals[0].units.magnitude == 1
            units = neo_block.segments[0].analogsignals[0].units.dimensionality.unicode
            fs = neo_block.segments[0].analogsignals[0].sampling_rate.magnitude

            all_channels.append(
                (str(chan_idx), str(chan_idx), fs, None, units, None, None, None),
            )

            # Check that all records have the same units / sampling rate
            if not len(neo_block.segments) == 1:
                self.mw.show_messagebox(
                    "Load File Error",
                    "Multi-segment records not currently supported. Please contact" "support@easyelectrophysiology.com",
                )
                return False

            for seg in neo_block.segments:  # For future...
                units_same = units == seg.analogsignals[chan_idx].units.dimensionality.unicode
                fs_same = fs == neo_block.segments[0].analogsignals[0].sampling_rate.magnitude

                if not (units_same and fs_same):
                    self.mw.show_messagebox(
                        "Load File Error",
                        "The file contains different units or sampling rates across "
                        "records. This type of recording is not supported.",
                    )
                    return False

        return np.array(all_channels, dtype=_signal_channel_dtype)

    def channel_is_proxy(self, channel_idx: Int, channel_1: bool = False) -> bool:
        """
        For some filetypes (possibly just .abf generated by IGOR)
        self._raw_signals is shorter ('igor_abf_too_many_channels.abf
        in test_data') than the number of channels neo can find. In
        this instance neo will fill with a proxy analogsignals but
        this will crash data loading class. Here check if any are proxy
        (absence of data i.e. 'magnitude attribute) and inform the user
        that this channel does not exist.

        Show error message if channel 1 index is tested (if another channel
        fails, just channel 1 will be loaded)
        """
        assert self.neo_block is not None, "Type Check neo block"
        segment_analogsignals = self.neo_block.segments[0].analogsignals

        if channel_idx + 1 > len(segment_analogsignals):
            channel_is_proxy = True
        else:
            channel_is_proxy = not hasattr(segment_analogsignals[channel_idx], "magnitude")

        if channel_1 and channel_is_proxy:
            self.show_warning(
                "Load File Error",
                "Please ensure the data channel exists in the file. "
                "Please contact support@easyelectrophysiology.com if "
                "this channel definitely exists.",
            )
        return channel_is_proxy

    def channel_2_exists_and_is_different_type_to_channel_1(
        self, channel_1_type: str, channel_2_type: Optional[str]
    ) -> bool:
        if not channel_2_type:
            return False

        if channel_1_type == channel_2_type:
            self.show_warning(
                "Load File Error",
                "Cannot have two channels of the same data type "
                "(e.g. voltage, voltage). "
                "Please load only one of the channels (File > File Loading Options > "
                "Select Channels to Load).",
            )
            return False
        return True

    def get_channel_type(self, channel: NDArray, channel_idx: Int) -> Tuple[Optional[str], Optional[ImOrVm]]:
        """
        Find the channel type based on it's saved name or units.

        Often the channel name reflects the input number of the amplifier
        e.g. IN0 and is no use. In this case the units is used.
        """
        channel_units = channel[4].strip()

        if channel_units == "\x02mV":
            channel_units = "mV"

        channel_units, channel_type = self.get_channel_type_from_channel_units(channel_units, channel_idx)

        if channel_units == "not_recognised":
            channel_units, channel_type = self.get_user_input_channel_units(channel[4].strip(), channel_idx)

        return channel_units, channel_type  # type: ignore

    def get_channel_type_from_channel_units(
        self, channel_units: str, channel_idx: Int
    ) -> Tuple[Optional[str], Optional[ImOrVm]]:
        """
        TODO
        ----
        add channel_units to configs as copied in importdata_show_channels
        """
        if channel_units == "mV":
            channel_type = "Vm"
        elif channel_units == "pA":
            channel_type = "Im"
        elif channel_units in ["fA", "nA", "uA", "mA", "A", "pV", "nV", "uV", "V"]:
            channel_units, channel_type = self.process_non_default_channel_units(
                channel_units, channel_idx  # type: ignore
            )
            if channel_units is None:
                return None, None
        else:
            return "not_recognised", None

        return channel_units, channel_type

    def get_user_input_channel_units(
        self, channel_units: str, channel_idx: Int
    ) -> Tuple[Optional[str], Optional[ImOrVm]]:
        orig_channel_units = channel_units
        while True:
            channel_units = utils.get_user_input(
                "Channel Not Found",
                "Channel {0} units ('{1}') were not recognised. "
                "Please enter the correct units below.\n"
                "Units must be one of: fA, pA, nA, uA, mA, "
                "A, pV, nV, uV, mV, V".format(channel_idx + 1, orig_channel_units),
                align_center=True,
            )

            if channel_units == "":  # catch cancel
                return None, None

            channel_units, channel_type = self.get_channel_type_from_channel_units(
                channel_units, channel_idx  # type: ignore
            )

            if channel_units == "not_recognised":
                utils.show_messagebox(
                    "Units Error",
                    "Input units were not recognised. Please try again or press cancel.",
                )
                continue

            elif channel_units is None or channel_type is None:  # type narrow
                return None, None

            else:
                break

        return channel_units, channel_type

    def process_non_default_channel_units(
        self, channel_units: str, channel_idx: Int
    ) -> Tuple[Optional[str], Optional[ImOrVm]]:
        """
        Handle channels not in pA or mV

        For some files the channel is specifier at the amplifier as 'A' generically.
        This is taken as the units for the data even though typically they are always
        recorded in pA. If such a file is encountered and default behaviour is not
        specified, prompt the user to set defaults, with an additional check that the
        header makes sense (in some cases e.g. Axograph files the header is in A but
        data is scaled to nA; see show_bad_header_units_warning())

        Otherwise, load the file as per the defaults. HEKA files are automatically
        converted from A / V.

        Data are converted at the Data class level depending on the channel_units type,
        so changing units to pA here means they
        will not be converted later.

        TODO
        ----
        move units to configs
        """
        if channel_units in ["fA", "nA", "uA", "mA", "A"]:
            channel_type = "Im"
        elif channel_units in ["pV", "nV", "uV", "V"]:
            channel_type = "Vm"

        if self.ext == ".DAT":
            channel_units_return = "A" if channel_type == "Im" else "V"

        elif channel_type == "Im" and self.cfgs.file_load_options["default_im_units"]["on"]:
            channel_units_return = (
                "pA"
                if self.cfgs.file_load_options["default_im_units"]["assume_pa"]
                else self.cfgs.file_load_options["default_im_units"]["pa_unit_to_convert"]
            )

        elif channel_type == "Vm" and self.cfgs.file_load_options["default_vm_units"]["on"]:
            channel_units_return = (
                "mV"
                if self.cfgs.file_load_options["default_vm_units"]["assume_mv"]
                else self.cfgs.file_load_options["default_vm_units"]["mv_unit_to_convert"]
            )

        else:
            self.show_bad_header_units_warning(channel_type, channel_units, channel_idx)
            channel_units_return = channel_type = None

        return channel_units_return, channel_type

    def get_channels_from_user_input(
        self,
    ) -> Tuple[Optional[Int], Optional[Int], Optional[NDArray], Optional[NDArray]]:
        """ """
        while True:
            user_input_channels = utils.get_user_input(
                "Channel Warning",
                "Easy Electrophysiology does not currently support recordings "
                "of more than two channels.\n"
                "Type 'view' to see a full list of channels.\n\n"
                "Please input the channels you would like to open, separated"
                "by a comma.\n"
                "This can be automated at "
                "'File' > 'File Loading Options' > 'Select Channels to Load'",
                align_center=True,
            )

            if user_input_channels.lower() in ["view", "'view'"]:
                assert self.channels is not None
                ViewChannels(None, self.cfgs, self.channels, exec=True)
                continue

            if user_input_channels == "":  # catch user cancel
                return None, None, None, None

            user_input_channels = utils.check_comma_separated_user_input_and_extract_ints(
                user_input_channels, allowed_chars=[","]
            )
            if user_input_channels is None:
                continue

            # Check number of channels
            if len(user_input_channels) == 1:
                if user_input_channels[0] == ",":
                    utils.show_messagebox("Input error", "Please input a number.")
                    continue
                else:
                    channel_2_input = False

            elif len(user_input_channels) == 2:
                channel_2_input = True

            if (
                len(user_input_channels) not in [1, 2]
                or 0 in user_input_channels
                or (channel_2_input and user_input_channels[0] == user_input_channels[1])
            ):
                utils.show_messagebox(
                    "Input Error",
                    "Please input up to 2 numbers " "in the range 1 to num channels.",
                )
                continue

            assert self.channels is not None, "Type Checking self.channels"
            try:
                channel_1_idx = int(user_input_channels[0]) - 1
                channel_1 = self.channels[channel_1_idx]

                if channel_2_input:
                    # assume user input is not zero indexed
                    channel_2_idx = int(user_input_channels[1]) - 1
                    channel_2 = self.channels[channel_2_idx]
                else:
                    channel_2_idx = None
                    channel_2 = None
                break

            except:
                utils.show_messagebox(
                    "Channel Error",
                    "Cannot use input. "
                    "Please ensure input is integer within the range "
                    "of available data channels.",
                )

        return channel_1_idx, channel_2_idx, channel_1, channel_2

    # Handle non-pA or mV channel types
    # ----------------------------------------------------------------------------------

    def show_bad_header_units_warning(self, channel_type: ImOrVm, channel_units: str, channel_idx: Int) -> None:
        """
        Prompt the user to set the correct settings for re-scaling data
        that is not in pA or mV.

        In some cases (typically Axograph files) the header reads A but the
        data is scaled to nA. Here guess the most likely scaling based on the
        range of the data and prompt the user if the guessed unit type does
        not match the header unit type.
        """
        expected_units = "pA" if channel_type == "Im" else "mV"
        message_string = (
            "The units in the file header are {0} "
            "but are expected to be {1}. Please specify default "
            "file loading options in 'File > File Loading Options > "
            "Specify Default Unit Handling'.".format(channel_units, expected_units)
        )

        guessed_units = self.check_if_header_units_are_correct(channel_idx, channel_type)
        if guessed_units != channel_units:
            extra_warning = (
                "\n\nWARNING! The units in the file header are {0} but "
                "these data are most likely in {1}. Try {1} first in "
                "'File Load Options'.".format(channel_units, guessed_units)
            )
            message_string += extra_warning

        self.mw.show_messagebox("Channel Units Error", message_string)

    def check_if_header_units_are_correct(self, channel_idx: Int, im_or_vm: ImOrVm) -> str:
        """
        Typically data properly scaled to Im / mV will be in the 0-2000 range.
        The only instance this could cause problems is for very large
        stimulus registering > 2000, but these are rarely seen pA / mV.
        """
        assert self.neo_block is not None, "Type Check neo block"
        channel_data = self.neo_block.segments[0].analogsignals[channel_idx].magnitude

        channel_range = max(channel_data) - min(channel_data)

        if im_or_vm == "Im":
            unit_ranges = {
                "fA": (2000, 9999),
                "pA": (1, 1999),
                "nA": (0.001, 0.999),
                "uA": (0.000001, 0.000999),
                "mA": (0.000000001, 0.000000999),
                "A": (0.000000000001, 0.000000000999),
            }
        elif im_or_vm == "Vm":
            unit_ranges = {
                "pV": (100000, 999999),
                "nV": (10000, 99999),
                "uV": (2000, 9999),
                "mV": (1, 1999),
                "V": (0.001, 0.999),
            }

        for unit, range_ in unit_ranges.items():
            if range_[0] < abs(channel_range) < range_[1]:
                return unit
        return "undefined: contact Easy Electrophysiology."

    def show_warning(self, title: str, text: str) -> None:
        """
        Show failed load warning with button to show all channels
        """
        assert self.channels is not None
        ShowChannelsWarning(self.mw, self.cfgs, self.channels, title, text)  # type: ignore


# --------------------------------------------------------------------------------------
# Data Class
# --------------------------------------------------------------------------------------


class RawData:
    """
    Extract Vm, Im and time arrays from Neo object. Note that depending on the
    software which recorded the data, time may increase cumulatively across records.
     Data is kept as 64-bit as is the usually fastest datatype to compute with numpy.

    Note these attributes are compared against each other when loading multiple files
    to ensure they are of the same type.

    Time offset:  For some file formats a slight offset is added to the time,
                  meaning it will start at a non-zero value. if this is present
                  in the data, self.time_offset will be set to the offset and
                  data changed at the model level. cannot wrap dialog to this
                  class or will throw pickle error on deepcopy at model level.

    """

    def __init__(
        self,
        neo_block: Block,
        num_chans: Int,
        cfgs: ConfigsClass,
        channel_1_type: str,
        channel_1_units: Optional[str],
        channel_1_idx: Int,
        channel_2_type: Optional[str],
        channel_2_units: Optional[str],
        channel_2_idx: Optional[Int],
        reader: BaseIO,
        channels: NDArray,
        file_ext: str,
    ):
        self.load_setting = cfgs.file_load_options["force_load_options"]
        self.num_recs = len(neo_block.segments)
        self.num_samples = len(neo_block.segments[0].analogsignals[channel_1_idx])

        # strip quantities
        self.fs = neo_block.segments[0].analogsignals[channel_1_idx].sampling_rate.magnitude
        self.ts = neo_block.segments[0].analogsignals[channel_1_idx].sampling_period.magnitude
        self.time_units = str(neo_block.segments[0].t_start).split(" ")[1]

        # empty must be zero, if empty or nan cannot be properly plot by pyqtgraph in
        # the one-channel case
        self.channel_1_data = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)
        self.channel_2_data = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)
        self.time_array = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)
        self.num_data_channels = num_chans

        # in rare cases, some setups will add a tiny offset to time so it does not
        # start at zero
        self.time_offset = False
        self.channel_1_type = channel_1_type
        self.channel_2_type = channel_2_type
        self.channel_1_idx = channel_1_idx
        self.channel_2_idx = channel_2_idx
        self.channel_1_units = channel_1_units
        self.channel_2_units = channel_2_units
        self.tags = ""
        self.all_channels = channels  # reader.header["signal_channels"]

        self.norm_first_deriv_data: Optional[NpArray64] = None
        self.recording_type: Literal["current_clamp", "voltage_clamp", "unknown"]
        self.t_start: float  # start time (first record)
        self.t_stop: float  # end time (final record)
        self.records_are_contiguous: bool  # whether time is contiguous between records
        self.min_max_time: NpArray64
        self.rec_time: NpArray64

        if self.load_setting is not None:
            self.fix_channels_to_current_or_voltage_clamp()

        if self.channel_1_type == "Vm" and (self.channel_2_idx is None or self.channel_2_type == "Im"):
            self.recording_type = "current_clamp"
        elif self.channel_1_type == "Im" and (self.channel_2_idx is None or self.channel_2_type == "Vm"):
            self.recording_type = "voltage_clamp"
        else:
            self.recording_type = "unknown"

        if file_ext.upper() == ".DAT":
            # This will scale HEKA's zero offsets, so must occur
            # before channel units are overwritten below.
            self.heka_metadata = self.load_heka_metadata(self.channel_1_type, self.channel_1_units, reader)
        else:
            self.heka_metadata = None

        if self.channel_1_type == "Im":
            self.channel_1_data, final_units = self.extract_data_from_array(
                self.channel_1_idx, neo_block, self.channel_1_type, self.channel_1_units
            )
            self.channel_1_units = final_units

            if self.channel_2_idx is not None:
                self.channel_2_data, final_units = self.extract_data_from_array(
                    self.channel_2_idx, neo_block, self.channel_2_type, self.channel_2_units
                )
                self.channel_2_units = final_units

        elif self.channel_1_type == "Vm":
            self.channel_1_data, final_units = self.extract_data_from_array(
                self.channel_1_idx, neo_block, self.channel_1_type, self.channel_1_units
            )
            self.channel_1_units = final_units

            if self.channel_2_idx is not None:
                self.channel_2_data, final_units = self.extract_data_from_array(
                    self.channel_2_idx, neo_block, self.channel_2_type, self.channel_2_units
                )
                self.channel_2_units = final_units
            elif file_ext.upper() == ".ABF":
                self.handle_generate_axon_protocol(reader, cfgs)

        self.time_array, self.t_start, self.t_stop = self.extract_time_array(neo_block)
        self.check_and_clean_data()

        # file format specified fields.
        if file_ext.upper() == ".ABF":
            self.update_tags(reader)

        return

    def fix_channels_to_current_or_voltage_clamp(self):
        if self.load_setting == "current_clamp":
            self.channel_1_type = "Vm"
            self.channel_1_units = "mV"
            if self.channel_2_type is not None:
                self.channel_2_type = "Im"
                self.channel_2_units = "pA"

        elif self.load_setting == "voltage_clamp":
            self.channel_1_type = "Im"
            self.channel_1_units = "pA"
            if self.channel_2_type is not None:
                self.channel_2_type = "Vm"
                self.channel_2_units = "mV"
        else:
            assert self.load_setting is None

    def get_primary_data(self) -> NpArray64:
        """
        Get data from the primary channel (e.g. the voltage trace from
        voltage clamp).
        """
        return self.channel_1_data

    def get_secondary_data(self) -> NpArray64:
        """
        Get data from the primary channel (e.g. the voltage trace from
        voltage clamp).
        """
        return self.channel_2_data

    def set_data_by_recording_type(
        self,
        primary_channel_array: Optional[NpArray64] = None,
        secondary_channel_array: Optional[NpArray64] = None,
    ) -> None:
        """
        Sets data based on analysis type (see get_primary_data()).
        The passed data should match the shape of the existing data
        Must call `set_data_params` and update the GUI after this function.
        """
        assert self.recording_type in ["current_clamp", "voltage_clamp", "unknown"]
        assert self.channel_1_type in ["Im", "Vm"]
        if self.channel_2_type is not None:
            assert self.channel_2_type in ["Im", "Vm"]

        if primary_channel_array is not None:
            self.channel_1_data = primary_channel_array

        if secondary_channel_array is not None:
            self.channel_2_data = secondary_channel_array

    def get_norm_first_deriv_data(self) -> NpArray64:
        """ """
        sample_spacing_in_ms = self.ts * 1000

        if self.norm_first_deriv_data is None:
            self.norm_first_deriv_data = np.diff(self.get_primary_data(), append=0) / sample_spacing_in_ms
            self.norm_first_deriv_data.setflags(write=False)

        return self.norm_first_deriv_data

    def check_and_clean_data(self) -> None:
        """
        Convert V to mV,
        nA to pA,
        ms to s

        If force analysis type the units may be swapped,
        in which case swap the conversion tables
        """
        assert self.t_start is not None and self.t_stop is not None, "Type Check t_start and t_stop"

        if self.time_units == "ms":
            self.t_start /= 1000
            self.t_stop /= 1000
            self.time_array /= 1000
            self.time_units = "s"

        if self.t_start != 0:
            self.time_offset = np.array(self.t_start)

    def extract_time_array(self, neo_block: Block) -> Tuple[NpArray64, float, float]:
        """ """
        channel_time_check = True

        if self.num_recs == 1:
            time_array = core_analysis_methods.generate_time_array(
                neo_block.segments[0].analogsignals[self.channel_1_idx].t_start.magnitude,
                neo_block.segments[0].analogsignals[self.channel_1_idx].t_stop.magnitude,
                self.num_samples,
                self.ts,
            )[np.newaxis, :]

        else:

            time_array = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)

            for rec in range(self.num_recs):
                time_ = core_analysis_methods.generate_time_array(
                    neo_block.segments[rec].analogsignals[self.channel_1_idx].t_start.magnitude,
                    neo_block.segments[rec].analogsignals[self.channel_1_idx].t_stop.magnitude,
                    self.num_samples,
                    self.ts,
                )
                time_array[rec, :] = time_.squeeze()

                if not self.channel_times_are_equal(neo_block, rec):
                    channel_time_check = False

        # get first and very last time point in case data needs cutting into records
        # (cut_up_data() in data model)
        t_start = float(neo_block.segments[0].analogsignals[self.channel_1_idx].t_start)

        # convert to float to remove units formatting, zero idx
        t_stop = float(neo_block.segments[self.num_recs - 1].analogsignals[self.channel_1_idx].t_stop)

        if not channel_time_check:
            utils.show_messagebox(
                "Time Error",
                "The timing of channel 1 and channel 2 are not "
                "the same. The time units for channel 2 will "
                "be incorrect. "
                "Please contact support@easyelectrophysiology.com",
            )

        return time_array, t_start, t_stop

    def channel_times_are_equal(self, neo_block: Block, rec: Int) -> bool:
        """
        It is assumed all analogsignals have the same start / stop time.
        These should always be Im / Vm traces recorded at the same time
        so it would be very unexpected to find a case in which the timings
        are different.
        """
        if self.channel_2_idx is not None and len(neo_block.segments[rec].analogsignals) == 2:
            if (
                neo_block.segments[rec].analogsignals[self.channel_1_idx].t_start.magnitude
                != neo_block.segments[rec].analogsignals[self.channel_2_idx].t_start.magnitude
                or neo_block.segments[rec].analogsignals[self.channel_1_idx].t_stop.magnitude
                != neo_block.segments[rec].analogsignals[self.channel_2_idx].t_stop.magnitude
            ):
                return False
        return True

    def extract_data_from_array(
        self, data_idx: Int, neo_block: Block, channel_type, channel_units
    ) -> Tuple[NpArray64, str]:
        """
        extract Im or Vm from neo model (depending on analogsignal idx)
        """
        if self.num_recs == 1:
            # avoid pointless data copy
            array = neo_block.segments[0].analogsignals[data_idx].magnitude.T.astype(np.float64)
            assert array.shape[0] == 1
        else:
            array = np.zeros((self.num_recs, self.num_samples), dtype=np.float64)

            for rec in range(self.num_recs):
                data = neo_block.segments[rec].analogsignals[data_idx].magnitude
                array[rec, :] = data.squeeze()

        final_units = self.scale_array_in_place_if_required(channel_type, channel_units, array)

        return array, final_units

    def load_heka_metadata(self, channel_1_type, channel_1_units, reader):
        """ """
        first_channel_offsets = np.array(reader.header["heka_metadata"]["zero_offsets"][self.channel_1_idx])

        if first_channel_offsets is not None:
            self.scale_array_in_place_if_required(channel_1_type, channel_1_units, array=first_channel_offsets)

        heka_metadata = {
            "primary_channel_zero_offsets": first_channel_offsets,
            "add_zero_offset": reader.header["heka_metadata"]["add_zero_offset"],
        }
        return heka_metadata

    def scale_array_in_place_if_required(self, channel_type, channel_units, array):
        """ """
        assert channel_type in ["Im", "Vm"], "`channel_type` must be `Im` or `Vm`."

        if channel_type == "Im":
            if channel_units != "pA":
                conversion_to_pa_table = core_analysis_methods.get_conversion_to_pa_table()
                array *= conversion_to_pa_table[channel_units]
            final_units = "pA"

        elif channel_type == "Vm":
            if channel_units != "mV":
                conversion_to_mv_table = core_analysis_methods.get_conversion_to_mv_table()
                array *= conversion_to_mv_table[channel_units]
            final_units = "mV"

        return final_units

    def extract_axon_protocol(self, reader: AxonIO) -> Tuple[NpArray64, str]:
        """
        https://neo.readthedocs.io/en/stable/io.html#neo.io.AxonIO
        """
        array = utils.np_empty_nan((self.num_recs, self.num_samples))
        protocol = reader.read_protocol()
        units = protocol[0].analogsignals[0].units.dimensionality.string
        for rec in range(self.num_recs):
            array[rec] = np.squeeze(protocol[rec].analogsignals[0].magnitude)

        return array, units

    def handle_generate_axon_protocol(self, reader: AxonIO, cfgs: ConfigsClass) -> None:
        """
        Coordinate the generation of axon protocol. This must be done in
        current clamp mode. If settings are correct, the Im protocol
        will be generated and loaded to the second channel, handling any errors.
        """
        if cfgs.file_load_options["select_channels_to_load"]["on"] and cfgs.file_load_options["generate_axon_protocol"]:
            if self.recording_type != "current_clamp":
                utils.show_messagebox(
                    "Axon Protocol Error",
                    "Must be in current clamp mode to " "load Axon Im protocol from header",
                )
                return

            try:
                im_array, im_units = self.extract_axon_protocol(reader)
            except:
                utils.show_messagebox(
                    "Axon Protocol Error",
                    "Could not generate Axon protocol. Please contact support@easyelectrophysiology.com",
                )
                return

            self.channel_2_units = im_units
            self.channel_2_data = im_array
            self.channel_2_type = "Im"
            self.num_data_channels = 2

    def update_tags(self, reader: AxonIO) -> None:
        """
        Update self.tags with tags (Axon Instruments feature only)
        """
        all_tags = ""
        for i in range(len(reader._axon_info["listTag"])):
            try:
                tag = reader._axon_info["listTag"][i]["sComment"].decode("utf-8").strip() + " "
            except UnicodeDecodeError:
                tag = "Could not load tags.\n" "Please contact support@easyelectrophysiology.com " "for more details."
            if i > 0:
                tag = "Tag {0}: ".format(str(i + 1)) + tag
            all_tags += tag
        self.tags = all_tags


# --------------------------------------------------------------------------------------
# Multi-file importdata checks
# --------------------------------------------------------------------------------------


def check_loaded_files_match(
    main_data: RawData, new_data: RawData, check_num_samples: bool
) -> Union[Literal[True], List]:
    """
    Check that that file parameters match (these are files in a
    list of files to be loaded then concatenated together).
    """
    match = True

    if new_data.channel_1_type != main_data.channel_1_type:
        match = ["channel_1_type", new_data.channel_1_type, main_data.channel_1_type]

    elif new_data.channel_2_type != main_data.channel_2_type:
        match = ["channel_2_type", new_data.channel_2_type, main_data.channel_2_type]

    elif new_data.channel_1_units != main_data.channel_1_units:
        match = ["channel_1_units", new_data.channel_1_units, main_data.channel_1_units]

    elif new_data.channel_2_units != main_data.channel_2_units:
        match = ["channel_2_units", new_data.channel_2_units, main_data.channel_2_units]

    elif new_data.time_offset != main_data.time_offset:
        match = ["time_offset", new_data.time_offset, main_data.time_offset]

    elif new_data.num_data_channels != main_data.num_data_channels:
        match = [
            "num_data_channels",
            new_data.num_data_channels,
            main_data.num_data_channels,
        ]

    elif new_data.fs != main_data.fs:
        match = ["fs", new_data.fs, main_data.fs]

    elif new_data.ts != main_data.ts:
        match = ["ts", new_data.ts, main_data.ts]

    elif new_data.time_units != main_data.time_units:
        match = ["time_units", new_data.time_units, main_data.time_units]

    elif check_num_samples and new_data.num_samples != main_data.num_samples:
        match = ["num_samples", new_data.num_samples, main_data.num_samples]

    return match
