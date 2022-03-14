from PySide2 import QtWidgets, QtCore, QtGui
from PySide2 import QtTest
from PySide2.QtTest import QTest
import pytest
import sys
import os
import pandas as pd
import numpy as np
import time
import keyboard
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/.."))
sys.path.append(os.path.realpath(os.path.dirname(__file__) + "/."))
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__) + "/.."), "easy_electrophysiology"))
from ..easy_electrophysiology import easy_electrophysiology
MainWindow = easy_electrophysiology.MainWindow
from ephys_data_methods import importdata, core_analysis_methods
from utils import utils
import neo
from setup_test_suite import GuiTestSetup
import quantities
from types import SimpleNamespace

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Mock Neo Object
# ----------------------------------------------------------------------------------------------------------------------------------------------------

class TestConfigs:
    def __init__(self):

        self.file_load_options = {"select_channels_to_load": {"on": False},
                                  "force_load_options": None,
                                  "default_im_units": {"on": True,
                                                       "assume_pa": False,
                                                       "pa_unit_to_convert": "fA",
                                                     },
                                  "default_vm_units": {"on": True,
                                                       "assume_mv": False,
                                                       "mv_unit_to_convert": "pV"},
                                  "generate_axon_protocol": False}

class Segments:
    def __init__(self, vm_data, im_data, t_start, t_stop, fs):

        self.t_stop = t_stop * quantities.s
        self.t_start = t_start * quantities.s

        setattr(vm_data, 'sampling_rate', fs * quantities.Hz)
        setattr(im_data, 'sampling_rate', fs * quantities.Hz)
        setattr(vm_data, 'sampling_period', 1 / fs * quantities.Hz)
        setattr(im_data, 'sampling_period', 1 / fs * quantities.Hz)

        setattr(vm_data, 't_start', self.t_start)
        setattr(im_data, 't_start', self.t_start)

        setattr(vm_data, 't_stop', self.t_stop)
        setattr(im_data, 't_stop', self.t_stop)

        self.analogsignals = [vm_data, im_data]


class ArtificialNeoBlock:
    def __init__(self):

        self.num_recs = 75
        self.num_samples = 8192
        self.rec_time_length = 2
        self.segments = []
        self.vm_idx = 0
        self.im_idx = 1

        self.fs = (self.num_samples - 1) / self.rec_time_length
        self.ts = 1 / self.fs

        # make a cul time rec and random vm / im arrays
        self.test_time_array = utils.np_empty_nan((self.num_recs, self.num_samples)) * quantities.s
        self.test_time_array[0] = np.linspace(0, self.rec_time_length - self.ts, self.num_samples) * quantities.s

        for rec in range(1, self.num_recs):
            self.test_time_array[rec] = self.test_time_array[rec - 1] + 2 * quantities.s

        self.test_vm_array = np.random.random((self.num_recs, self.num_samples)) * quantities.V
        self.test_im_array = np.random.random((self.num_recs, self.num_samples)) * quantities.A

        for rec in range(self.num_recs):
            self.segments.append(Segments(self.test_vm_array[rec],
                                          self.test_im_array[rec],
                                          0 + rec * self.rec_time_length,
                                          self.rec_time_length + rec * self.rec_time_length,
                                          self.fs))

    @staticmethod
    def channel_times_are_equal(dummy1, dummy2):
        return True

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Importdata Test Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------

def make_path(file_ext, filename):
    base_dir = os.path.join(os.path.dirname(__file__), "test_data", "importdata_tests")
    full_filename = filename + file_ext
    full_filepath = os.path.join(base_dir, full_filename)
    return full_filepath

class TestImportData:

    @pytest.fixture(scope="function")
    def cfgs(self):
        """
        """
        cfgs = TestConfigs()
        return cfgs

    @pytest.fixture(scope="function", params=[".abf", ".wcp"], ids=["abf", "wcp"])
    def cc_two_channels_path(test, request):
        """
        """
        file_ext = request.param
        if file_ext == ".wcp":
            full_filepath = make_path(file_ext, "cc_two_channels")
        elif file_ext == ".abf":
            full_filepath = make_path(file_ext, "cc_two_channels_cumu")
        return full_filepath, file_ext

    @pytest.fixture(scope="function", params=[".abf", ".wcp"], ids=["abf", "wcp"])
    def cc_one_channel_path(test, request):
        file_ext = request.param
        full_filepath = make_path(file_ext, "cc_one_channel")
        return full_filepath, file_ext

    @pytest.fixture(scope="function", params=[".wcp"], ids=["wcp"])
    def vc_two_channels_path(self, request):
        file_ext = request.param
        full_filepath = make_path(file_ext, "vc_two_channels")
        return full_filepath, file_ext

    @pytest.fixture(scope="function", params=[".edr"], ids=["edr"])
    def vc_one_channel_edr(self, request):
        file_ext = request.param
        full_filepath = make_path(file_ext, "vc_one_channel_edr")
        return full_filepath, file_ext

    @pytest.fixture(scope="function", params=[".axgd"], ids=["axgd"])
    def cc_one_channel_axgd(self, request):
        file_ext = request.param
        file_fullpath = make_path(file_ext, "cc_one_channel_axograph")
        return file_fullpath, file_ext

    @pytest.fixture(scope="function", params=[".axgx"], ids=["axgx"])
    def vc_multi_channel_axgx(self, request):
        file_ext = request.param
        file_fullpath = make_path(file_ext, "vc_multi_channel_axograph")
        return file_fullpath, file_ext

    @pytest.fixture(scope="function")
    def cc_one_channel_abf_info_dict(self):
        cc_one_channel_abf_info_dict = dict(num_recs=7, num_samples=20000, num_data_channels=1, analysis_type="current_clamp",
                                            t_start=0, t_stop=182, vm_units="mV", im_units=None, time_units="s",
                                            time_offset=0, channel_1_type="Vm", channel_2_type=None)
        return cc_one_channel_abf_info_dict

    @pytest.fixture(scope="function")
    def cc_two_channels_abf_info_dict(self):
        cc_two_channels_abf_info_dict = dict(num_recs=12, num_samples=140000, num_data_channels=2, analysis_type="current_clamp",
                                             t_start=0, t_stop=111.4, vm_units="mV", im_units="pA", time_units="s",
                                             time_offset=0, channel_1_type="Vm", channel_2_type="Im")
        return cc_two_channels_abf_info_dict

    @pytest.fixture(scope="function")
    def cc_one_channel_wcp_info_dict(self):
        cc_one_channel_wcp_info_dict = dict(num_recs=7, num_samples=20224, num_data_channels=1, analysis_type="current_clamp",  # weird, WinWCP seems to be resampling the ABF
                                            t_start=0, t_stop=2.0223999489098787, vm_units="mV", im_units=None, time_units="s",
                                            time_offset=0, channel_1_type="Vm", channel_2_type=None)
        return cc_one_channel_wcp_info_dict

    @pytest.fixture(scope="function")
    def cc_two_channels_wcp_info_dict(self):
        cc_two_channels_wcp_info_dict = dict(num_recs=15, num_samples=65536, num_data_channels=2, analysis_type="current_clamp",
                                             t_start=0, t_stop=2.998271942138672, vm_units="mV", im_units="pA", time_units="s",
                                             time_offset=0, channel_1_type="Vm", channel_2_type="Im")
        return cc_two_channels_wcp_info_dict

    @pytest.fixture(scope="function")
    def vc_one_channel_edr_info_dict(self):
        cc_two_channels_wcp_info_dict = dict(num_recs=1, num_samples=120000, num_data_channels=1, analysis_type="voltage_clamp_1_record",
                                             t_start=0, t_stop=120, vm_units=None, im_units="pA", time_units="s",
                                             time_offset=0, channel_1_type="Im", channel_2_type=None)
        return cc_two_channels_wcp_info_dict

    @pytest.fixture(scope="function")
    def vc_two_channels_info_dict(self):
        vc_two_channels_info_dict = dict(num_recs=40, num_samples=2048, num_data_channels=2, analysis_type="voltage_clamp_multi_record",
                                         t_start=0, t_stop=0.9999359846115111, vm_units="mV", im_units="pA", time_units="s",
                                         time_offset=0, channel_1_type="Im", channel_2_type="Vm")
        return vc_two_channels_info_dict

    @pytest.fixture(scope="function")
    def cc_one_channel_axgd_info_dict(self):
        cc_one_channel_axgd_info_dict = dict(num_recs=39, num_samples=40000, num_data_channels=1, analysis_type="current_clamp",
                                             t_start=0.00002, t_stop=0.8000200000000002, vm_units="mV", im_units=None, time_units="s",
                                             time_offset=0.00002, channel_1_type="Vm", channel_2_type=None)
        return cc_one_channel_axgd_info_dict

    @pytest.fixture(scope="function")
    def vc_multi_channel_axograph_info_dict(self):
        vc_multi_channel_axograph_info_dict = dict(num_recs=1, num_samples=1000, num_data_channels=1, analysis_type="voltage_clamp_1_record",
                                                   t_start=0.00005, t_stop=0.050050000000000004, vm_units=None, im_units="pA", time_units="s",
                                                   time_offset=0.00005, channel_1_type="Im", channel_2_type=None)
        return vc_multi_channel_axograph_info_dict

    @pytest.fixture(scope="function")
    def artificial_neo_block(self):
        artificial_neo_block = ArtificialNeoBlock()
        return artificial_neo_block

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def load_file_with_neo(fixture_path_lists, cfgs, load_data=False, mw=None, return_class=False):
        """
        """
        full_filepath = fixture_path_lists[0]
        file_ext = fixture_path_lists[1]

        import_data = importdata.ImportData(full_filepath, file_ext, cfgs, mw)
        if load_data:
            __ = import_data.load_data()

        if return_class:
            return import_data.channels, import_data.reader, import_data.neo_block, file_ext, import_data
        else:
            return import_data.channels, import_data.reader, import_data.neo_block, file_ext

    @staticmethod
    def get_info_dict_from_fileinfo(filename, file_ext,
                                    cc_one_channel_abf_info_dict, cc_two_channels_abf_info_dict,
                                    cc_one_channel_wcp_info_dict, cc_two_channels_wcp_info_dict,
                                    vc_one_channel_edr_info_dict, vc_two_channels_info_dict,
                                    cc_one_channel_axgd_info_dict, vc_multi_channel_axgx_info_dict):

        if filename + file_ext == "cc_one_channel.abf":
            info_dict = cc_one_channel_abf_info_dict
        elif filename + file_ext == "cc_two_channels_cumu.abf":
            info_dict = cc_two_channels_abf_info_dict
        elif filename + file_ext == "cc_one_channel.wcp":
            info_dict = cc_one_channel_wcp_info_dict
        elif filename + file_ext == "cc_two_channels.wcp":
            info_dict = cc_two_channels_wcp_info_dict
        elif filename + file_ext == "vc_one_channel_edr.edr":
            info_dict = vc_one_channel_edr_info_dict
        elif filename + file_ext == "vc_two_channels.wcp":
            info_dict = vc_two_channels_info_dict
        elif filename + file_ext == "cc_one_channel_axograph.axgd":
            info_dict = cc_one_channel_axgd_info_dict
        elif filename + file_ext == "vc_multi_channel_axograph.axgx":
            info_dict = vc_multi_channel_axgx_info_dict

        return info_dict


# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Test Load File with Neo Method
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_cc_two_channels_load(self, cc_two_channels_path, cfgs):

        channels, reader, neo_block, file_ext = self.load_file_with_neo(cc_two_channels_path, cfgs)

        assert len(channels) == 2, "wrong num channels, " + file_ext
        assert channels[0][4].strip() == "mV", "wrong channel 1 type, " + file_ext
        assert channels[1][4].strip() == "pA", "wrong channel 2 type, " + file_ext
        assert type(neo_block) == neo.core.block.Block, "block neo file load failed, : " + file_ext

    def test_cc_one_channel_load(self, cc_one_channel_path, cfgs):

        channels, reader, neo_block, file_ext = self.load_file_with_neo(cc_one_channel_path, cfgs)

        assert len(channels) == 1, "wrong num channels, " + file_ext
        assert channels[0][4].strip() == "mV", "wrong channel 1 type, " + file_ext
        assert type(neo_block) == neo.core.block.Block, "block neo file load failed, : " + file_ext

    def test_two_channels_vc_load(self, vc_two_channels_path, cfgs):

        channels, reader, neo_block, file_ext = self.load_file_with_neo(vc_two_channels_path, cfgs)

        assert len(channels) == 2, "wrong num channels, " + file_ext
        assert channels[0][4].strip() == "pA", "wrong channel 1 type, " + file_ext
        assert channels[1][4].strip() == "mV", "wrong channel 2 type, " + file_ext
        assert type(neo_block) == neo.core.block.Block, "block neo file load failed, : " + file_ext

    def test_vc_one_channel_edr_load(self, vc_one_channel_edr, cfgs):
        channels, reader, neo_block, file_ext = self.load_file_with_neo(vc_one_channel_edr, cfgs)

        assert len(channels) == 1, "wrong num channels, " + file_ext
        assert channels[0][4].strip() == "pA", "wrong channel type, " + file_ext
        assert type(neo_block) == neo.core.block.Block, "block neo file load failed, : " + file_ext

    def test_cc_one_channel_axgd(self, cc_one_channel_axgd, cfgs):
        channels, reader, neo_block, file_ext = self.load_file_with_neo(cc_one_channel_axgd, cfgs)

        assert len(channels) == 1, "wrong num channels, " + file_ext
        assert channels[0][4].strip() == "V", "wrong channel type, " + file_ext
        assert type(neo_block) == neo.core.block.Block, "block neo file load failed, : " + file_ext

    def test_vc_multi_channel_axgx(self, vc_multi_channel_axgx, cfgs):
        channels, reader, neo_block, file_ext = self.load_file_with_neo(vc_multi_channel_axgx, cfgs)

        assert len(channels) == 6, "wrong num channels, " + file_ext
        assert channels[0][4].strip() == "A", "wrong channel type, " + file_ext
        assert channels[1][4].strip() == "", "wrong channel type, " + file_ext
        assert channels[2][4].strip() == "", "wrong channel type, " + file_ext
        assert channels[3][4].strip() == "", "wrong channel type, " + file_ext
        assert channels[4][4].strip() == "", "wrong channel type, " + file_ext
        assert channels[5][4].strip() == "", "wrong channel type, " + file_ext
        assert type(neo_block) == neo.core.block.Block, "block neo file load failed, : " + file_ext

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test get channel info
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_cc_two_channels_get_channel_info(self, cc_two_channels_path, cfgs):
        """
        Only test 1-2 channel cases as need to test user input separately
        """
        channels, reader, __, file_ext, import_data = self.load_file_with_neo(cc_two_channels_path, cfgs, return_class=True)
        channel_1, channel_1_idx, channel_2, channel_2_idx = import_data.get_channel_data_info()

        assert channel_1[4].strip() == "mV", "wrong channel 1 type, " + file_ext
        assert channel_2[4].strip() == "pA", "wrong channel 2 type, " + file_ext
        assert channel_1_idx == 0, "wrong channel 1 idx, " + file_ext
        assert channel_2_idx == 1, "wrong channel 2 idx, " + file_ext

    def test_cc_one_channel_get_channel_info(self, cc_one_channel_path, cfgs):
        channels, reader, __, file_ext, import_data = self.load_file_with_neo(cc_one_channel_path, cfgs, return_class=True)
        channel_1, channel_1_idx, channel_2, channel_2_idx = import_data.get_channel_data_info()

        assert channel_1[4].strip() == "mV",  "wrong channel type, " + file_ext
        assert channel_1_idx == 0, "wrong channel idx, " + file_ext
        assert channel_2 is None, "channel 2 is not none, " + file_ext
        assert channel_2_idx is None, "channel 2 idx is not none, " + file_ext

    def test_vc_two_channels_get_channel_info(self, vc_two_channels_path, cfgs):

        channels, reader, __, file_ext, import_data = self.load_file_with_neo(vc_two_channels_path, cfgs, return_class=True)
        channel_1, channel_1_idx, channel_2, channel_2_idx = import_data.get_channel_data_info()

        assert channel_1[4].strip() == "pA", "wrong channel type, " + file_ext
        assert channel_1_idx == 0, "wrong channel idx, " + file_ext
        assert channel_2[4].strip() == "mV", "wrong channel type, " + file_ext
        assert channel_2_idx == 1, "wrong channel idx, " + file_ext

    def test_vc_one_channel_edr_get_channel_info(self, vc_one_channel_edr, cfgs):

        channels, reader, __, file_ext, import_data = self.load_file_with_neo(vc_one_channel_edr, cfgs, return_class=True)
        channel_1, channel_1_idx, channel_2, channel_2_idx = import_data.get_channel_data_info()

        assert channel_1[4].strip() == "pA", "wrong channel type, " + file_ext
        assert channel_1_idx == 0, "wrong channel idx, " + file_ext
        assert channel_2 is None, "channel 2 is not none, " + file_ext
        assert channel_2_idx is None, "channel 2 idx is not none, " + file_ext

    def test_cc_one_channel_axgd_get_channel_info(self, cc_one_channel_axgd, cfgs):
        channels, reader, __, file_ext, import_data = self.load_file_with_neo(cc_one_channel_axgd, cfgs, return_class=True)
        channel_1, channel_1_idx, channel_2, channel_2_idx = import_data.get_channel_data_info()

        assert channel_1[4].strip() == "V", "wrong channel type, " + file_ext
        assert channel_1_idx == 0, "wrong channel idx, " + file_ext
        assert channel_2 is None, "channel 2 is not none, " + file_ext
        assert channel_2_idx is None, "channel 2 idx is not none, " + file_ext

    def test_vc_multi_channel_axgx_get_channel_info(self, vc_multi_channel_axgx, cfgs):

        cfgs.file_load_options["select_channels_to_load"]["on"] = True
        cfgs.file_load_options["select_channels_to_load"]["channel_1_idx"] = 0
        cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"] = None

        channels, reader, __, file_ext, import_data = self.load_file_with_neo(vc_multi_channel_axgx, cfgs, return_class=True)
        channel_1, channel_1_idx, channel_2, channel_2_idx = import_data.get_channel_data_info()

        assert channel_1[0].strip() == "Current", "wrong channel type, " + file_ext
        assert channel_1_idx == 0, "wrong channel idx, " + file_ext
        assert channel_2 is None, "channel 2 is not none, " + file_ext
        assert channel_2_idx is None, "channel 2 idx is not none, " + file_ext

    def test_process_non_default_channel_units(self, vc_multi_channel_axgx, cfgs):

        channels, reader, __, file_ext, import_data = self.load_file_with_neo(vc_multi_channel_axgx, cfgs, return_class=True)  # TODO: check !

        import_data.cfgs.file_load_options["default_im_units"] = {"on": True}
        for im_test in ["fA", "nA", "uA", "mA", "A"]:

            import_data.cfgs.file_load_options["default_im_units"]["assume_pa"] = True

            channel_units, channel_type = import_data.process_non_default_channel_units(im_test,  None)
            assert channel_units == "pA"
            assert channel_type == "Im"

            import_data.cfgs.file_load_options["default_im_units"]["assume_pa"] = False
            import_data.cfgs.file_load_options["default_im_units"]["pa_unit_to_convert"] = im_test

            channel_units, channel_type = import_data.process_non_default_channel_units(im_test,  None)
            assert channel_units == im_test
            assert channel_type == "Im"

        import_data.cfgs.file_load_options["default_vm_units"] = {"on": True}
        for im_test in ["pV", "nV", "uV", "V"]:

            import_data.cfgs.file_load_options["default_vm_units"]["assume_mv"] = True

            channel_units, channel_type = import_data.process_non_default_channel_units(im_test, None)
            assert channel_units == "mV"
            assert channel_type == "Vm"

            import_data.cfgs.file_load_options["default_vm_units"]["assume_mv"] = False
            import_data.cfgs.file_load_options["default_vm_units"]["mv_unit_to_convert"] = im_test

            channel_units, channel_type = import_data.process_non_default_channel_units(im_test, None)
            assert channel_units == im_test
            assert channel_type == "Vm"

    def test_conversion_tables(self):
        """
        Double checked based on the conversions at
        https://www.convert-measurement-units.com/convert+Picoampere+to+Femtoampere.php
        """
        conversion_to_pa_table = core_analysis_methods.get_conversion_to_pa_table()
        test_units_pa = {
            "fA": 0.001,
            "nA": 1000,
            "uA": 1000000,
            "mA": 1000000000,
            "A": 1000000000000,
        }
        for unit in ["fA", "nA", "uA", "mA", "A"]:
            assert conversion_to_pa_table[unit] == test_units_pa[unit]

        conversion_to_mv_table = core_analysis_methods.get_conversion_to_mv_table()
        test_units_mv = {
            "pV": 0.000000001,
            "nV": 0.000001,
            "uV": 0.001,
            "V": 1000,
        }
        for unit in ["pV", "nV", "uV", "V"]:
            assert conversion_to_mv_table[unit] == test_units_mv[unit]

    def test_unit_conversions(self, artificial_neo_block, cfgs):
        """
        Load a file as if it was a different unit. This will be converted 'back' to pA.
        Copy this expected conversion here and check.
        Also change time to ms and check result is orig time / 1000
        """
        reader = SimpleNamespace(header={"signal_channels": None})
        conversion_to_pa_table = core_analysis_methods.get_conversion_to_pa_table()
        for unit in ["fA", "nA", "uA", "mA", "A"]:
            data = importdata.RawData(artificial_neo_block, 1, cfgs, None, None, None, "Im", unit, 0, reader, "None")
            assert (data.im_array == artificial_neo_block.test_vm_array * conversion_to_pa_table[unit]).all()
            assert data.im_units == "pA"

        conversion_to_mv_table = core_analysis_methods.get_conversion_to_mv_table()
        for unit in ["pV", "nV", "uV", "V"]:
            data = importdata.RawData(artificial_neo_block, 1, cfgs, "Vm", unit, 0, None, None, None, reader, "None")
            assert (data.vm_array == artificial_neo_block.test_vm_array * conversion_to_mv_table[unit]).all()
            assert data.vm_units == "mV"

        artificial_neo_block.segments[0].t_start = artificial_neo_block.segments[0].t_start.magnitude * quantities.ms
        data = importdata.RawData(artificial_neo_block, 1, cfgs, "Vm", unit, 0, None, None, None, reader, "None")
        assert (np.round(data.time_array, 10) == np.round((artificial_neo_block.test_time_array / 1000), 10)).all()
        assert (np.round(data.time_array, 10) == np.round((artificial_neo_block.test_time_array / 1000), 10)).all()
        assert data.time_units == "s"

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test get channel type
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_get_channel_type(self, cc_two_channels_path, cc_one_channel_path, vc_two_channels_path, vc_one_channel_edr, cfgs):
        """
        Test the funtion which returns channel type and units. This may differ based on the vendor. Here each test file is used
        not because of their num channels but because they are from a range of vendors. Iterate through all to check.

        Note: due to the way fixtures work this will iterate num inputs * num inputs times. Doesn't really matter as so fast.
        """

        for full_filepath, type_ in zip([cc_two_channels_path,           cc_one_channel_path,   vc_two_channels_path,            vc_one_channel_edr],
                                        [[["Vm", "mV"], ["Im", "pA"]],   [["Vm", "mV"]],        [["Im", "pA"], ["Vm", "mV"]],    [["Im", "pA"]]]):

            filename = os.path.split(full_filepath[0])[-1]
            channels, reader, __, file_ext, import_data = self.load_file_with_neo(full_filepath, cfgs, return_class=True)
            channel_1, channel_1_idx, channel_2, channel_2_idx = import_data.get_channel_data_info()

            channel_units, channel_type = import_data.get_channel_type(channel_1, channel_1_idx)
            assert channel_type == type_[0][0], "failed channel 1 type for " + filename
            assert channel_units == type_[0][1], "failed channel 1 units for " + filename

            if channel_2:
                channel_units, channel_type = import_data.get_channel_type(channel_2, channel_2_idx)
                assert channel_type == type_[1][0], "failed channel 2 type for " + filename
                assert channel_units == type_[1][1], "failed channel 2 units for " + filename

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Test Importdata Class
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    @pytest.mark.parametrize("file_info", [["cc_one_channel",     ".abf"],
                                           ["cc_two_channels_cumu",    ".abf"],
                                           ["cc_one_channel",     ".wcp"],
                                           ["cc_two_channels",    ".wcp"],
                                           ["vc_one_channel_edr", ".edr"],
                                           ["vc_two_channels",     ".wcp"],
                                           ["cc_one_channel_axograph", ".axgd"],
                                           ["vc_multi_channel_axograph", ".axgx"]])
    def test_importdata_all_files(self, file_info, cc_one_channel_abf_info_dict, cc_two_channels_abf_info_dict,
                                  cc_one_channel_wcp_info_dict, cc_two_channels_wcp_info_dict, vc_one_channel_edr_info_dict, vc_two_channels_info_dict,
                                  cc_one_channel_axgd_info_dict, vc_multi_channel_axograph_info_dict, cfgs):
        """
        Note the units are V, A for the axograph files. The configs here must be setup correctly to automatically convert on load
        """
        filename = file_info[0]
        file_ext = file_info[1]
        info_dict = self.get_info_dict_from_fileinfo(filename, file_ext,
                                                     cc_one_channel_abf_info_dict, cc_two_channels_abf_info_dict,
                                                     cc_one_channel_wcp_info_dict, cc_two_channels_wcp_info_dict,
                                                     vc_one_channel_edr_info_dict, vc_two_channels_info_dict,
                                                     cc_one_channel_axgd_info_dict, vc_multi_channel_axograph_info_dict)

        if filename == "vc_multi_channel_axograph":
            cfgs.file_load_options["select_channels_to_load"]["on"] = True
            cfgs.file_load_options["select_channels_to_load"]["channel_1_idx"] = 0
            cfgs.file_load_options["select_channels_to_load"]["channel_2_idx"] = None

        full_filepath = make_path(file_ext, filename)
        channels, reader, __, file_ext, import_data = self.load_file_with_neo([full_filepath, file_ext], cfgs, return_class=True)
        imported_data = import_data.load_data()

        assert imported_data.num_recs == info_dict["num_recs"],                       "num recs" + filename + file_ext
        assert imported_data.num_samples == info_dict["num_samples"],                 "num samples: " + filename + file_ext
        assert imported_data.num_data_channels == info_dict["num_data_channels"],     "data channels: " + filename + file_ext
        assert imported_data.t_start == info_dict["t_start"],                         "t start: " + filename + file_ext
        assert imported_data.t_stop == info_dict["t_stop"],                           "t stop: " + filename + file_ext
        assert imported_data.vm_units == info_dict["vm_units"],                       "vm units: " + filename + file_ext
        assert imported_data.im_units == info_dict["im_units"],                       "im units: " + filename + file_ext
        assert imported_data.time_units == info_dict["time_units"],                   "time units: " + filename + file_ext
        assert imported_data.time_offset == info_dict["time_offset"],                 "time offset: " + filename + file_ext
        assert imported_data.channel_1_type == info_dict["channel_1_type"],           "channel 1 type: " + filename + file_ext
        assert imported_data.channel_2_type == info_dict["channel_2_type"],           "channel 2 type: " + filename + file_ext
        assert imported_data.recording_type == info_dict["analysis_type"],             "analysis type: " + filename + file_ext

    def test_extract_data_vm_from_array(self, artificial_neo_block):

        importdata_vm_array = importdata.RawData.extract_data_from_array(self=artificial_neo_block,
                                                                         data_idx=artificial_neo_block.vm_idx,
                                                                         neo_block=artificial_neo_block)
        assert np.array_equal(importdata_vm_array,
                              artificial_neo_block.test_vm_array), "extract data vm"

    def test_extract_data_im_from_array(self, artificial_neo_block):
        importdata_im_array = importdata.RawData.extract_data_from_array(self=artificial_neo_block,
                                                                         data_idx=artificial_neo_block.im_idx,
                                                                         neo_block=artificial_neo_block)
        assert np.array_equal(importdata_im_array,
                              artificial_neo_block.test_im_array),  "extract data vm"

    def test_extract_time_array(self, artificial_neo_block):

        importdata_time_array, t_start, t_stop = importdata.RawData.extract_time_array(self=artificial_neo_block,
                                                                                       neo_block=artificial_neo_block)

        assert utils.allclose(artificial_neo_block.test_time_array.magnitude,
                              importdata_time_array,
                              1e-8),  "extract data time"

# ----------------------------------------------------------------------------------------------------------------------------------------------------
# Misc (move)
# ----------------------------------------------------------------------------------------------------------------------------------------------------

    def test_file_tags(self):
        """
        Load sample tag file make a multi-tag version of Neo's reader output and check it is formatted correctly
        Finally load a file without any tags and check the fields is hidden from file details
        """
        tgui = GuiTestSetup("test_tags")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.test_load_norm_time_file()

        class TestReader:
            def __init__(self):

                self._axon_info = {"listTag": [
                                               {"sComment": bytes("test_tag_1", "utf-8")},
                                               {"sComment": bytes("test_tag_2", "utf-8")},
                                               {"sComment": bytes("test_tag_3", "utf-8")},
                ]}
        test_tags = "test_tag_1 Tag 2: test_tag_2 Tag 3: test_tag_3 "
        file_ext = ".ABF"
        reader = TestReader()

        assert tgui.mw.loaded_file.data.tags == "sample tag "

        tgui.mw.loaded_file.data.update_tags(reader, file_ext)
        assert tgui.mw.loaded_file.data.tags == test_tags

        tgui.mw.mw.actionFile_Details.trigger()
        file_details_tags = tgui.mw.dialogs["file_details_dialog"].items[7]

        assert len(tgui.mw.dialogs["file_details_dialog"].items) == 8

        assert file_details_tags == "Tags: " + test_tags
        tgui.mw.dialogs["file_details_dialog"].close()

        tgui.load_a_filetype("current_clamp")
        assert tgui.mw.loaded_file.data.tags == ""

        tgui.mw.mw.actionFile_Details.trigger()
        assert len(tgui.mw.dialogs["file_details_dialog"].items) == 7

        tgui.shutdown()

    def test_time_on_tagged_file(self):
        """
        """
        tgui = GuiTestSetup("test_tags")
        tgui.setup_mainwindow(show=True)
        tgui.test_update_fileinfo()
        tgui.test_load_norm_time_file()  # TODO: this is assumed to be tagged file, checked against clampfit

        for rec in range(tgui.mw.loaded_file.data.num_recs):
            rec_time = tgui.mw.loaded_file.data.time_array[rec]
            assert np.isclose(rec_time[-1] - rec_time[0],
                              0.4999,
                              atol=1e-10, rtol=0)
        assert tgui.mw.loaded_file.data.fs == 10000
