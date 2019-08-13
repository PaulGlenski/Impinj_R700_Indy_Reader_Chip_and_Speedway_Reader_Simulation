##
#  This class models the reader FPGA hardware accelerators.
##

from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.common.adc import Adc
from systems_r700.model.src.reader.reader_modem import ReaderModem

import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt


class ReaderFpga(object):
    # constructor
    def __init__(self, attrib=ReaderAttributes()):
        #Initializes Modem
        self.modem_ = ReaderModem(attrib)
        
        #Initializes CCI and CCQ values
        self.cc_adc_i_ = Adc(attrib.rx_.cc_adc_v_analog_range_, attrib.rx_.cc_adc_num_bits_, attrib.rx_.cc_adc_bipolar_)
        self.cc_adc_q_ = Adc(attrib.rx_.cc_adc_v_analog_range_, attrib.rx_.cc_adc_num_bits_, attrib.rx_.cc_adc_bipolar_)
        self.rx_adc_i_ = Adc(attrib.rx_.rx_adc_v_analog_range_, attrib.rx_.rx_adc_num_bits_, attrib.rx_.rx_adc_bipolar_)
        self.rx_adc_q_ = Adc(attrib.rx_.rx_adc_v_analog_range_, attrib.rx_.rx_adc_num_bits_, attrib.rx_.rx_adc_bipolar_)
        
        #Initializes Tx Rampup params
        self.tx_adc = Adc(attrib.tx_.aux_adc_fs_v, attrib.tx_.aux_adc_num_bits, bipolar=False)
        self.dec_filt_i = attrib.tx_.input_digital_level_I
        self.dec_filt_q = attrib.tx_.input_digital_level_Q
        self.tx_aux_dac_level = 0 #initial input to the aux DAC from FPGA stored
        self.tx_adc_error = attrib.tx_.bb_initial_err_val #inital bb error stored
        self.fs_ = attrib.rx_.fs_

    def cc_process(self, v_in):
        cc_sample_i = self.cc_adc_i_.process(v_in.real)
        cc_sample_q = self.cc_adc_q_.process(v_in.imag)
        cc_sample_i_array = np.asarray(cc_sample_i)
        cc_sample_q_array = np.asarray(cc_sample_q)
        output = self.modem_.cc_process(cc_sample_i_array + 1j * cc_sample_q_array)
        return output

    def rx_batch_process(self, v_in, link, config):
        rx_sample_i = self.rx_adc_i_.process(v_in.real)
        rx_sample_q = self.rx_adc_q_.process(v_in.imag)
        rx_sample_i_array = np.asarray(rx_sample_i)
        rx_sample_q_array = np.asarray(rx_sample_q)
        rx = rx_sample_i_array + 1j * rx_sample_q_array

        # print("The length of the wvfm before the modem is ", len(rx))
        # plt.plot(np.real(rx), ".-")
        # plt.plot(np.imag(rx), ".-")
        # plt.title("wvfm before the modem")
        # plt.figure()
        # plt.show()

        output = self.modem_.rx_batch_process(rx, link, config)
        return output

    #Output to the Tx Dac
    def tx_bb_out_process(self):
        mod_gain_out_I = self.modem_.tx_bb_.get_mod_gain_output(self.dec_filt_i)
        mod_gain_out_Q = self.modem_.tx_bb_.get_mod_gain_output(self.dec_filt_i) #full scale voltage sum
        print("mod_gain_I: {}, mod_gain_Q: {}".format(mod_gain_out_I, mod_gain_out_Q))
        return mod_gain_out_I, mod_gain_out_Q
    
    #Rampup process
    def tx_ramp_process(self, adc_in, ramp_down_flag):
        adc_out = self.tx_adc.process(adc_in)
        if ramp_down_flag:
            aux_dac_in = self.modem_.tx_ramp_down_process(adc_out)
        else:
            aux_dac_in = self.modem_.tx_ramp_up_process(adc_out)
        self.tx_adc_error = self.modem_.tx_bb_.err
        print("reader_fpga tx_adc_error: {}, aux_dac_in: {}".format(self.tx_adc_error, aux_dac_in))
        self.tx_aux_dac_level = aux_dac_in
        print(adc_in, adc_out)
        return aux_dac_in