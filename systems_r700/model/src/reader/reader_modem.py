##
#  This class models the reader Modem (Rx and Tx baseband processing).
##

import numpy as np
import matplotlib.pyplot as plt
import systems_r700.model.src.common.utils as ut
import pickle
import os

from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.reader.reader_rx_baseband import ReaderRxBaseband
from systems_r700.model.src.reader.reader_tx_baseband import ReaderTxBaseband
from systems_r700.model.src.common.system_config import ConfigClass
from systems_r700.model.src.simulation_engine.sim_loop_parameters import SimulationParameters

from systems_r700.model.src.tag.python_tools.systems_utils.ddc import DdcClass
from systems_r700.model.src.tag.python_tools.systems_utils.rate_conversion import TrlClass
from systems_r700.model.src.tag.python_tools.systems_utils.dmf import DmfClass
from systems_r700.model.src.tag.python_tools.systems_utils.pmf import PmfClass
from systems_r700.model.src.tag.python_tools.systems_utils.decode_bits import DecodeBitsClass
from systems_r700.model.src.tag.python_tools.systems_utils.decode_symbols import DecodeSymbolClass

import numpy as np
from collections import namedtuple

class ReaderModem(object):
    Link = namedtuple('link', 'mod_type mval tr_extend blf_hz fsample_hz')

    def __init__(self, attrib=ReaderAttributes()):
        self.rx_bb_ = ReaderRxBaseband(attrib)
        self.tx_bb_ = ReaderTxBaseband(attrib)
        self.parameters = SimulationParameters()

    def cc_process(self, sample):
        return self.rx_bb_.cc_process(sample)
    
    def tx_ramp_up_process(self, adc_out):
        return self.tx_bb_.rampup_process(adc_out)
    
    def tx_ramp_down_process(self, adc_out):
        return self.tx_bb_.rampdown_process(adc_out)

    def rx_batch_process(self, rx_adc, link, config):

        self.link = link
        self.config = config

        ddc = DdcClass(x=rx_adc,
                       config=self.config,
                       plot_fig=False)
        ddc.execute()

        # Simulation points
        num_in_pts = ddc.n_out_smpls
        num_out_pts = int(num_in_pts / self.config.Ti_Ts)
        max_pts = np.max([num_in_pts, num_out_pts])

        # Create the timing recovery loop object
        loop_obj_dict = self.parameters.get_loop_components(max_pts)

        # print("The length of the wvfm before the TRL is ", len(rx_adc))
        # plt.plot(np.real(rx_adc), ".-")
        # plt.title("wvfm before the TRL")
        # plt.figure()

        # Execute the timing recovery loop
        trl = TrlClass(num_in_pts,
                       num_out_pts,
                       config=self.config,
                       **loop_obj_dict)

        if SimulationParameters.get_float_calc:
            x_up = ddc.ddc_up_decimate
            x_dn = ddc.ddc_dn_decimate
        else:
            x_up = np.round(ddc.ddc_up_decimate * 2 ** 7) * 2 ** -7 * 0.95
            x_dn = np.round(ddc.ddc_dn_decimate * 2 ** 7) * 2 ** -7 * 0.95

        if self.config.trl_dict['trl_bypass']:
            dmf_input = np.real(ddc.ddc_up_decimate)
        else:
            # DMF
            trl.execute(x_up, x_dn)
            dmf_input = np.real(trl.y_lf)

        # print("The length of the TRL output is ", len(np.real(trl.y_lf)))
        # plt.plot(np.real(trl.y_lf), ".-")
        # plt.title("TRL output")
        # plt.figure()

        dmf = DmfClass(dmf_input,
                       plot_fig=False)
        dmf.execute()

        # PMF
        pmf = PmfClass(x=dmf_input,
                       config=self.config,
                       plot_fig=False)
        pmf.execute()

        # print("The output of the dmf is ", len(dmf.y))
        # plt.plot(np.real(dmf.y), ".-")
        # plt.title("DMF outputs")
        # plt.figure()

        # Relevant PMF output added to config
        self.config.set_pmf_config(pmf_obj=pmf)

        decode_bits = DecodeBitsClass(x=dmf.y,
                                      config=self.config,
                                      plot_fig=False)
        decode_bits.execute()

        output_bits = decode_bits.y

        decoded_symbols = DecodeSymbolClass(x=output_bits[self.config.preamble_len:],
                                            orig=0,
                                            config=self.config,
                                            last_word=decode_bits.last_word)
        decoded_symbols.execute()

        return decoded_symbols.y






