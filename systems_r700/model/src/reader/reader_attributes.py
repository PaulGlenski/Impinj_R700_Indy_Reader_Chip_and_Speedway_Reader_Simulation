##
#  These classes encapsulate all the static attributes of the reader hardware.
##

import numpy as np
import yaml
import pandas as pd
from os.path import join, dirname
from systems_r700.model.src.tag.tag_waveforms import GenerateRevWaveform
from collections import namedtuple

#wvfms = GenerateRevWaveform.generate_waveform(GenerateRevWaveform.generate_waveform.__init__())

#Attribute dictionary wrapper from below -- added support for nested dictionaries to make it more opaque to user
#https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        self.convert_nested_attr_dict(self.__dict__)

    #Converts any nested dictionaries in an attribute dictionary into attribute dictionaries
    def convert_nested_attr_dict(self, attr_dict):
        for a in attr_dict:
            if type(attr_dict[a]) is dict:
                attr_dict[a]= AttrDict(attr_dict[a])


#Attributes are stored in a Yaml file and then placed in an attribute dictionary to simulate class structure
class ReaderAttributes(object):
    _FILE_DIRECTORY = dirname(__file__)
    _DEFAULT_SETTINGS_FILE_PATH = join(_FILE_DIRECTORY, "..//bom//reader_attributes.yaml")
    _VGA_CHARACTERISTICS_PATH = join(_FILE_DIRECTORY, "..//bom//VGA_Characteristics.csv")
    _CC_FCC_I_ATTEN_BENCH_PATH = join(_FILE_DIRECTORY, "..//bom//cc_FCC_I_atten_bench_range.csv")
    _CC_FCC_Q_ATTEN_BENCH_PATH = join(_FILE_DIRECTORY, "..//bom//cc_FCC_Q_atten_bench_range.csv")
    _CC_FCC_I_ATTEN_IDEAL_PATH = join(_FILE_DIRECTORY, "..//bom//cc_FCC_I_atten_ideal_range.csv")
    _CC_FCC_Q_ATTEN_IDEAL_PATH = join(_FILE_DIRECTORY, "..//bom//cc_FCC_Q_atten_ideal_range.csv")
    #_VM_IDEAL_ATTEN_PATH = 'systems_r700/Analysis/carrier_cancellation/generated_data/gains_ideal_interpolated_original.npy'
    _VM_IDEAL_ATTEN_PATH = 'C:/Users/pglenski/Git Repos/systems/systems_r700/Analysis/carrier_cancellation/generated_data/gains_ideal_interpolated.npy'
    #Could not get it to call with systems_r700/Analysis/.../gains_ideal_interpolated_original.npy
    #TODO: FIX THE ABOVE VM IDEAL ATTAN PATH
    
    # constructor
    def __init__(self, num_antennas=1, fs=20.48e6, fc=2.048e6, settings_path=_DEFAULT_SETTINGS_FILE_PATH):
        self.load_settings(settings_path)
        self.load_settings_simulation_control()
        self.load_settings_rx()
        self.load_settings_tx()
        self.load_settings_common()
    
    # Load Yaml File into one big dictionary
    def load_settings(self, settings_path):
        with open(settings_path, "r") as default_settings_file_stream:
            self.attr_dict_ = AttrDict(yaml.load(default_settings_file_stream))

    # Loading simulation control settings
    def load_settings_simulation_control(self):
        self.sim_ctrl_ = self.attr_dict_.reader_simulation_control_attributes

    # Loading rx settings
    def load_settings_rx(self):
        self.rx_ = self.attr_dict_.reader_rx_attributes
        self.rx_.return_loss_db_ = []
        self.rx_.reflection_phase_ = []
        self.rx_.i_atten_bench_ = pd.read_csv(self._CC_FCC_I_ATTEN_BENCH_PATH)
        self.rx_.q_atten_bench_ = pd.read_csv(self._CC_FCC_Q_ATTEN_BENCH_PATH)
        self.rx_.i_atten_ideal_ = pd.read_csv(self._CC_FCC_I_ATTEN_IDEAL_PATH)
        self.rx_.q_atten_ideal_ = pd.read_csv(self._CC_FCC_Q_ATTEN_IDEAL_PATH)

        if self.rx_.use_measured_gains_:
            self.rx_.vm_attenuations_ = np.load(self.rx_.measured_gains_data_file_)
        else:
            self.rx_.vm_attenuations_ = np.load(self._VM_IDEAL_ATTEN_PATH)

        for i in range(self.rx_.num_antennas_):
            #np.append(-10.0,  self.rx_.return_loss_db_)
            self.rx_.return_loss_db_.append(-10.0)
            # self.rx_.return_loss_db_.append(-10.0625) #worst case for 0deg
            # self.rx_.return_loss_db_.append(-10.049066945) #worst case for 45deg
            #np.append(np.random.uniform(0, 2 * np.pi), self.rx_.reflection_phase_)
            self.rx_.reflection_phase_.append(np.random.uniform(0, 2 * np.pi))

    # Load Tx Settings
    def load_settings_tx(self):
        self.tx_ = self.attr_dict_.reader_tx_attributes
        self.tx_.vga_characteristics = pd.read_csv(self._VGA_CHARACTERISTICS_PATH)
    
    # Load Common Settings
    def load_settings_common(self):
        self.common_ = self.attr_dict_.reader_common_attributes
