##
#  This class models the simulation engine.
##

from systems_r700.reader.rtp_lib.sim_data_control import DataControl
from systems_r700.model.src.reader.reader import Reader
from systems_r700.model.src.common.system_config import ConfigClass
from systems_r700.model.src.tag.python_tools.systems_utils.revlink_modes import revlink
from systems_r700.reader.rtp_lib.ber_manager import BERManager
from systems_r700.model.src.tag.tag_waveforms import GenerateRevWaveform
from systems_r700.model.src.reader.reader_impairments import ReaderImpairments
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes

import numpy as np
import matplotlib.pyplot as plt
import systems_r700.model.src.common.utils as ut
import pickle
import os
from numpy.random import rand, randn

class SimulationEngine:

    def __init__(self, link='fm0_640kHz_8LF',
                       max_bits=30,
                       packet_size=30,
                       min_error_bits=0,
                       max_error_bits=30,
                       plot=True,
                       save_data=True,
                       attrib=ReaderAttributes(),
                       impairments=ReaderImpairments()):

        self.link = revlink[link]

        if '3b4b' in self.link.mod_type:
            self.last_word = 3
        else:
            self.last_word = 2
        self.adjusted_packet_size = packet_size - self.last_word

        self.config = ConfigClass(link=self.link, num_bits=max_bits, packet_size=packet_size, adjusted_packet_size=self.adjusted_packet_size)
        self.num_packets = np.ceil(max_bits / packet_size)
        self.max_bits = self.adjusted_packet_size * self.num_packets
        self.packet_size = packet_size
        self.min_error_bits = min_error_bits
        self.max_error_bits = max_error_bits
        self.plot = plot
        self.save_data = save_data

        self.ber = []
        self.flag = True
        self.print_data = True
        self.plot_bpsk_model_curve = True
        self.Eb_No_array = np.arange(5, 6)

        self.data_control = DataControl(self.packet_size)
        self.tag_ = GenerateRevWaveform(config=ConfigClass(link=self.link),
                                                drc_in_smpls_per_blf=8,
                                                EbNo_dB=self.config.EbNo_dB_vec[self.config.ebno_cnt],
                                                n_pad_zero_bits=0,
                                                )
        attrib.rx_.fs_ = self.link.fsample_hz
        self.fs = attrib.rx_.fs_
        self.reader_ = Reader(self.link, attrib, impairments)
        self.ber_manager = BERManager(self.max_bits, min_error_bits, max_error_bits)

    def rx_ber(self):

        for Eb_No in self.Eb_No_array:
            wvfm_power_dB = -174 + 22 + Eb_No + (10 * np.log10(self.link.data_rate))
            Ps = 10 ** (wvfm_power_dB / 10)

            while not self.ber_manager.satisfied():

                randomized_bits = self.data_control.get_bits()

                adjusted_last_word_bits = randomized_bits[:self.adjusted_packet_size]

                backscatter_wvfm = self.tag_.generate_waveform(bits=randomized_bits,
                                                               plot=False,
                                                               phase_rad=0.0,
                                                               phase_bits=None,
                                                               lpf_type='FIR',
                                                               num_bits=self.packet_size,)

                wvfm_variance = np.var(backscatter_wvfm)
                wvfm_unity = backscatter_wvfm / (np.sqrt(wvfm_variance))
                backscatter_upscaled_wvfm = (np.sqrt(Ps)) * wvfm_unity

                # print("The length of the upscaled wvfm is ", len(backscatter_upscaled_wvfm))
                # plt.plot(np.real(backscatter_upscaled_wvfm), ".-")
                # plt.title("upscaled wvfm")
                # plt.figure()

                if self.flag:
                    cc_process = self.reader_.cc_process()
                    self.flag = False
                    self.config.cc_i_sign = cc_process[0]
                    self.config.cc_i = cc_process[1]
                    self.config.cc_q_sign = cc_process[2]
                    self.config.cc_q = cc_process[3]
                decoded_symbols = self.reader_.rx_batch_process(backscatter_upscaled_wvfm, self.config)

                self.ber_manager.update_new(decoded_symbols, adjusted_last_word_bits)

            final_ber = self.ber_manager.get_ber()
            self.ber.append(final_ber)
            self.ber_manager.reset()
            self.flag = True

            print("The EbNo just finished is ", Eb_No, " and the BER is ", final_ber)
            if self.save_data:
                self.save_ber_pickle()
            if self.print_data:
                if self.save_data:
                    self.print_ber_pickle()


        print("The different BER for each power/noise ratio are ", self.ber)

        if self.plot_bpsk_model_curve:
            if self.plot:
                self.plot_bpsk = True
        if self.plot:
            self.plot_curve()

    def process_tx_calibration(self, tx_gain):
        self.reader_.rfa_.set_pa_bias(100)
        return self.reader_.tx_calibration_process(tx_gain)

    def process_tx_rampdown(self):
        err = 500
        error_threshold = 100
        num_iterations = 0
        #TODO: Setting PA bias to placeholder value for now
        while num_iterations in range(0,100) and np.abs(err) > error_threshold:
            err = self.reader_.tx_ramp_process(True)
            num_iterations = num_iterations + 1
        self.reader_.rfa_.set_pa_bias(0)
        return num_iterations

    def process_tx_rampup(self):
        err = 500
        error_threshold = 4
        num_iterations = 0
        num_iterations_good = 0
        max_num_loops = 500
        max_num_good_loops = 5
        #TODO: Setting PA bias to placeholder value for now
        self.reader_.rfa_.set_pa_bias(190)
        while num_iterations in range(0,max_num_loops):
            err = self.reader_.tx_ramp_process(False)
            num_iterations = num_iterations + 1
            if np.abs(err) <= error_threshold:
                num_iterations_good += 1
            else:
                num_iterations_good = 0
            if num_iterations_good >= max_num_good_loops:
                break
        return num_iterations

    def calculate_cc_ideal(self):
        ideal_i, ideal_q, dc_residual = self.reader_.calculate_ideal()
        return ideal_i, ideal_q, dc_residual

    def process_cc_all_search(self):
        cc_error_dict = self.reader_.cc_all_search()
        return cc_error_dict

    def process_cc_lms_search(self, quantized=True):
        cc_coarse, cc_fine, cc_lms, cc_final = self.reader_.cc_process(quantized)
        return cc_coarse, cc_fine, cc_lms, cc_final

    # ------------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------------

    def plot_curve(self):

        #Print the BER Curve
        curve_values = [(self.Eb_No_array[i], self.ber[i]) for i in range(0, len(self.Eb_No_array))]
        ut.create_figure_doc(w=4, h=1.5, rect_cones=[0.18, 0.21, 0.80, 0.77])
        plt.semilogy(*zip(*curve_values), '.-', linewidth=1, markersize=1, label='Bit Error Rate')

        if self.plot_bpsk:
            if self.max_bits >= 10000000:
                #Simulate a new BPSK Model Curve
                self.bpsk_curve_setup()
                bpsk_ebno = self.Eb_No_array
                bpsk_ber = self.bpsk_ber
                bpsk_values = [(bpsk_ebno[i], bpsk_ber[i]) for i in range(0, len(bpsk_ebno))]
                plt.semilogy(*zip(*bpsk_values), '.-', linewidth=1, markersize=1, label='BPSK Ideal')
            else:
                #Prints previoulsy-simulated BPSK Model Curve to e-06
                bpsk_ebno = np.arange(0, 11.5, .5)
                bpsk_ber = [.078655, .067113, .056214, .046432, .037592, .029609, .022808,
                            .017181, .012489, .008791, .005946, .003893, .002382, .001404,
                            .000776, .000388, .000193, 8.037e-05, 3.375e-05, 1.05e-05, 2.65e-06, 8.75e-07, 2.5e-07]
                bpsk_values = [(bpsk_ebno[i], bpsk_ber[i]) for i in range(0, len(bpsk_ebno))]
                plt.semilogy(*zip(*bpsk_values), '.-', linewidth=1, markersize=1, label='BPSK Ideal')

        ut.annotate_figure(title=None,
                           xlabel='Signal To Noise Ratios',
                           ylabel='Bit Error Rate',
                           xlim=None,
                           ylim=None,
                           legend=True,
                           leg_loc=0,
                           grid=True,
                           titlesize=4,
                           labelsize=5,
                           ticklabelsize=5,
                           legendsize=5)

    def save_ber_pickle(self):
        self.filename = os.path.join(ut.get_root_directory(), 'systems_r700', 'model', 'src', 'simulation_engine',
                                     'saved_ber_data.pkl')
        with open(self.filename, 'wb') as f:
            pickle.dump([self.ber], f, pickle.HIGHEST_PROTOCOL)

    def print_ber_pickle(self):
        pickle_in = open(self.filename, "rb")
        pickle_ber = pickle.load(pickle_in)
        print("The pickle's saved ber data is ", pickle_ber)

    def bpsk_curve_setup(self):
        N = self.max_bits
        EbNodB_range = self.Eb_No_array
        itr = len(EbNodB_range)
        ber = [None] * itr

        for n in range(0, itr):
            EbNodB = EbNodB_range[n]
            EbNo = 10.0 ** (EbNodB / 10.0)
            x = 2 * (rand(int(N)) >= 0.5) - 1
            noise_std = 1 / np.sqrt(2 * EbNo)
            y = x + noise_std * randn(int(N))
            y_d = 2 * (y >= 0) - 1
            errors = (x != y_d).sum()
            ber[n] = 1.0 * errors / int(N)
        self.bpsk_ber = ber
        print("The bpsk values are ", self.bpsk_ber)




