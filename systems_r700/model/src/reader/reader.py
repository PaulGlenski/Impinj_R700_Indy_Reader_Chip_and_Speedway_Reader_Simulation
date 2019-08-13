##
#  This class models the full reader.
##

from systems_r700.model.src.reader.reader_impairments import ReaderImpairments
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.reader.reader_antenna_array import ReaderAntennaArray
from systems_r700.model.src.reader.reader_rf_analog import ReaderRfAnalog
from systems_r700.model.src.reader.reader_fpga import ReaderFpga
import systems_r700.model.src.common.utils as ut

import numpy as np
import matplotlib.pyplot as plt


class Reader(object):
    # constructor
    def __init__(self, link, attrib=ReaderAttributes(), impairments=ReaderImpairments()):
        self.link = link
        self.ant_array_ = ReaderAntennaArray(attrib)
        self.ant_array_.select_antenna(attrib.common_.selected_antenna_)
        self.rfa_ = ReaderRfAnalog(attrib, impairments)
        self.fpga_ = ReaderFpga(attrib)

        self.sim_ctrl_ = attrib.sim_ctrl_

        self.vm_attenuations_ = attrib.rx_.vm_attenuations_
        self.vm_precision_db_ = attrib.rx_.vm_precision_db_

        # quantized gains
        cci = np.arange(0,511)
        self.quantized_gains = (10 ** ((cci-511) * self.vm_precision_db_ / 20.0))

        self.attenuator_i_state_coarse = np.arange(64, 511, 128)
        self.attenuator_q_state_coarse = np.arange(64, 511, 128)

        self.i_sign = attrib.rx_.i_sign
        self.q_sign = attrib.rx_.q_sign

        self.max_lms_steps = attrib.rx_.lms_steps
        self.filter1_settle_samples = attrib.rx_.filter1_settle_samples
        self.mu = attrib.rx_.mu

        self._lms_residual_dc = np.array([])
        self._lms_residual_dc_fpga = np.array([])
        self._lms_grad_i = np.array([])
        self._lms_grad_q = np.array([])
        self._lms_weighted_mu = np.array([])
        self._lms_cci = np.array([])
        self._lms_ccq = np.array([])
        self._lms_gain = np.array([])
        self._lms_gain_quantized = np.array([])

        self._cc_i_sign = 0
        self._cc_q_sign = 0
        self._cc_i = 0
        self._cc_q = 0

    def quantize_gains(self, gain):
        diff = np.abs(gain) - self.quantized_gains
        cc_sign = np.sign(gain)
        cc = np.argmin(np.abs(diff))
        quant_gain = cc_sign * (10 ** ((cc-511) * self.vm_precision_db_ / 20.0))

        return cc_sign, cc, quant_gain


    def tx_ramp_process(self, ramp_down_flag):
        mod_gain_out_I, mod_gain_out_Q = self.fpga_.tx_bb_out_process()
        tx_dac_out = self.rfa_.tx_dac_out_process(mod_gain_out_I, mod_gain_out_Q)
        self.rfa_.tx_ramp_process(tx_dac_out, self.fpga_.tx_aux_dac_level)
        self.fpga_.tx_ramp_process(self.rfa_.pdet_out, ramp_down_flag)
        return self.fpga_.tx_adc_error
    
    def tx_calibration_process(self, tx_gain):
        print("Tx Pwr = {}".format(self.rfa_.tx_attrib_.tx_pwr_dbm_))
        #self.rfa_.analog_settling.settling_reset()
        mod_gain_out_I, mod_gain_out_Q = self.fpga_.tx_bb_out_process()
        tx_dac_out = self.rfa_.tx_dac_out_process(mod_gain_out_I, mod_gain_out_Q)
        pdet_out = self.rfa_.tx_ramp_process(tx_dac_out, tx_gain)
        adc_out = self.fpga_.tx_adc.process(pdet_out)
        return adc_out

    def cc_process(self, quantized=True):
        error_dict_coarse = self.cc_coarse_search()
        error_dict_fine = self.cc_fine_search(error_dict_coarse)
        error_dict_lms = self.gradient_start_search(error_dict_fine, quantized)
        error_dict_final = self.cc_final_search(error_dict_lms)

        error_dict_final_min = min(error_dict_final.keys(), key=(lambda k: error_dict_final[k]))

        self._cc_i_sign = error_dict_final_min[0]
        self._cc_q_sign = error_dict_final_min[2]
        self._cc_i = error_dict_final_min[1]
        self._cc_q = error_dict_final_min[3]
        return error_dict_final_min
        #return error_dict_coarse, error_dict_fine, error_dict_lms, error_dict_final

    def cc_all_search(self):
        error_dict = {}
        for i_sign in self.i_sign:
            for q_sign in self.q_sign:
                for atten_i in range(511):
                    for atten_q in range(511):
                        for samples in range(self.filter1_settle_samples):
                            cw = self.rfa_.cc_fwd_process()
                            refl = self.ant_array_.process(cw)
                            rfa_cc_out = self.rfa_.cc_process(refl, i_sign, q_sign, atten_i, atten_q)
                        cc_out = self.fpga_.cc_process(rfa_cc_out)
                        error_dict[(np.sign(i_sign), atten_i, np.sign(q_sign), atten_q)] = np.abs(self.rfa_.combined_[-1])

        error_dict_final_min = min(error_dict.keys(), key=(lambda k: error_dict[k]))

        self._cc_i_sign = error_dict_final_min[0]
        self._cc_q_sign = error_dict_final_min[2]
        self._cc_i = error_dict_final_min[1]
        self._cc_q = error_dict_final_min[3]
        return error_dict

    def cc_coarse_search(self):
        error_dict_coarse = {}
        for cci_sign in self.i_sign:
            for ccq_sign in self.q_sign:
                for cci in self.attenuator_i_state_coarse:
                    for ccq in self.attenuator_q_state_coarse:
                        for samples in range(self.filter1_settle_samples):
                            cw = self.rfa_.cc_fwd_process()
                            refl = self.ant_array_.process(cw)
                            rfa_cc_out = self.rfa_.cc_process(refl, cci_sign, ccq_sign, cci, ccq)
                        cc_out = self.fpga_.cc_process(rfa_cc_out)

                        error_dict_coarse[(cci_sign, cci, ccq_sign, ccq)] = np.abs(self.rfa_.combined_[-1])

        return error_dict_coarse

    def cc_fine_search(self, coarse_dict):
        coarse_min = min(coarse_dict.items(), key=lambda x: x[1])
        coarse_min_i = coarse_min[0][1]
        coarse_min_q = coarse_min[0][3]
        cci_sign = coarse_min[0][0]
        ccq_sign = coarse_min[0][2]
        fine_step = 60
        error_dict_fine = {}
        for cci in np.arange(coarse_min_i - fine_step, coarse_min_i + (2 * fine_step), fine_step):
            for ccq in np.arange(coarse_min_q - fine_step, coarse_min_q + (2 * fine_step), fine_step):
                for samples in range(self.filter1_settle_samples):
                    cw = self.rfa_.cc_fwd_process()
                    refl = self.ant_array_.process(cw)
                    rfa_cc_out = self.rfa_.cc_process(refl, cci_sign, ccq_sign, cci, ccq)
                cc_out = self.fpga_.cc_process(rfa_cc_out)
                error_dict_fine[(cci_sign, cci, ccq_sign, ccq)] = (np.abs(self.rfa_.combined_[-1]), self.rfa_.combined_[-1])
        return error_dict_fine

    def lms_update(self, curr_gain, curr_out, grad_i, grad_q, sign_change):
        if sign_change:
            # floor mu to allow oscillation of the final solution
            if self.mu >= -9:
                self.mu -= 1

        # mu = (2 ** self.mu)
        mu = (2 ** (-14))
        self._lms_weighted_mu = np.append(self._lms_weighted_mu, mu)

        updi = (curr_out.real * grad_i.real) + (curr_out.imag * grad_i.imag)
        updq = (curr_out.real * grad_q.real) + (curr_out.imag * grad_q.imag)
        upd = np.complex(mu * updi, mu * updq)

        new_gain = curr_gain + upd

        return new_gain

    def gradient_start_search(self, fine_dict, quantized):
        fine_min = min(fine_dict.items(), key=lambda x: x[1][0])
        cci = fine_min[0][1]
        ccq = fine_min[0][3]
        cci_sign = fine_min[0][0]
        ccq_sign = fine_min[0][2]
        error_lms_dict = {}
        step_counter = 1

        # opposite sign based on residual dc for lms' first step
        sign_i = -cci_sign * np.sign(fine_min[1][1].real)
        sign_q = -ccq_sign * np.sign(fine_min[1][1].imag)
        sign_change = False

        cc_out = np.complex(1, 1)
        while step_counter <= self.max_lms_steps:
            curr_gain = np.complex(cci_sign * (10 ** ((cci - 511) * self.vm_precision_db_ / 20.0)),
                                   ccq_sign * (10 ** ((ccq - 511) * self.vm_precision_db_ / 20.0)))

            self._lms_cci = np.append(self._lms_cci, cci)
            self._lms_ccq = np.append(self._lms_ccq, ccq)
            self._lms_gain_quantized = np.append(self._lms_gain_quantized, curr_gain)

            prev_cc_out = cc_out

            # get current residual-DC
            for samples in range(self.filter1_settle_samples):
                cw = self.rfa_.cc_fwd_process()
                refl = self.ant_array_.process(cw)
                rfa_cc_out = self.rfa_.cc_process(refl, cci_sign, ccq_sign, cci, ccq)
            cc_out = self.fpga_.cc_process(rfa_cc_out)
            self._lms_residual_dc = np.append(self._lms_residual_dc, np.abs(self.rfa_.combined_[-1]))
            self._lms_residual_dc_fpga = np.append(self._lms_residual_dc_fpga, cc_out)

            error_lms_dict[cci_sign, cci, ccq_sign, ccq] = (step_counter, np.abs(self.rfa_.combined_[-1]))

            if((np.sign(prev_cc_out.real) != np.sign(cc_out.real)) or (np.sign(prev_cc_out.imag) != np.sign(cc_out.imag))):
                sign_change = True
            else:
                sign_change = False

            # determine the direction in which to measure gradient (the direction that takes residual DC towards zero)
            # the logic below assumes that the combiner output is computed as (cc - rx) and not the other way around
            if cci_sign > 0:
                sign_i = -1
            else:
                sign_i = 1

            if ccq_sign > 0:
                sign_q = -1
            else:
                sign_q = 1

            # get residual-DC when taking one step in I direction
            cci_update = int(cci + sign_i)
            for samples in range(self.filter1_settle_samples):
                cw = self.rfa_.cc_fwd_process()
                refl = self.ant_array_.process(cw)
                rfa_cc_out = self.rfa_.cc_process(refl, cci_sign, ccq_sign, cci_update, ccq)
            cc_out_i_step = self.fpga_.cc_process(rfa_cc_out)

            # get residual-DC when taking one step in the Q direction
            ccq_update = int(ccq + sign_q)
            for samples in range(self.filter1_settle_samples):
                cw = self.rfa_.cc_fwd_process()
                refl = self.ant_array_.process(cw)
                rfa_cc_out = self.rfa_.cc_process(refl, cci_sign, ccq_sign, cci, ccq_update)
            cc_out_q_step = self.fpga_.cc_process(rfa_cc_out)

            grad_i = cc_out_i_step - cc_out
            grad_q = cc_out_q_step - cc_out
            self._lms_grad_i = np.append(self._lms_grad_i, grad_i)
            self._lms_grad_q = np.append(self._lms_grad_q, grad_q)

            new_gain = self.lms_update(curr_gain, cc_out, grad_i, grad_q, sign_change)
            self._lms_gain = np.append(self._lms_gain, new_gain)

            # quantize to nearest cci and ccq
            cci_sign, cci, gain_i_quant = self.quantize_gains(new_gain.real)
            ccq_sign, ccq, gain_q_quant = self.quantize_gains(new_gain.imag)

            step_counter += 1

        # ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        # plt.subplot(211)
        # plt.plot(self._lms_residual_dc_fpga.real, '.-')
        # ut.annotate_figure(xlabel='', ylabel='Residual DC (real)')
        # plt.subplot(212)
        # plt.plot(self._lms_residual_dc_fpga.imag, '.-')
        # ut.annotate_figure(xlabel='LMS Iterations', ylabel='Residual DC (imag.)')
        #
        # ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        # plt.subplot(211)
        # plt.plot(self._lms_gain.real, '.-')
        # plt.plot(self._lms_gain_quantized.real, 'r.-')
        # ut.annotate_figure(xlabel='', ylabel='Gain (real)')
        # plt.subplot(212)
        # plt.plot(self._lms_gain.imag, '.-')
        # plt.plot(self._lms_gain_quantized.imag, 'r.-')
        # ut.annotate_figure(xlabel='LMS Iterations', ylabel='Gain (imag.)')
        #
        # ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        # plt.subplot(211)
        # plt.plot(self._lms_cci, '.-')
        # ut.annotate_figure(xlabel='', ylabel='CC Control (real)')
        # plt.subplot(212)
        # plt.plot(self._lms_ccq, '.-')
        # ut.annotate_figure(xlabel='LMS Iterations', ylabel='CC Control (imag.)')
        #
        # ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        # plt.subplot(211)
        # plt.plot(self._lms_grad_i.real, '.-')
        # ut.annotate_figure(xlabel='', ylabel='I Gradient (real)')
        # plt.subplot(212)
        # plt.plot(self._lms_grad_i.imag, '.-')
        # ut.annotate_figure(xlabel='LMS Iterations', ylabel='I Gradient (imag.)')
        #
        # ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        # plt.subplot(211)
        # plt.plot(self._lms_grad_q.real, '.-')
        # ut.annotate_figure(xlabel='', ylabel='Q Gradient (real)')
        # plt.subplot(212)
        # plt.plot(self._lms_grad_q.imag, '.-')
        # ut.annotate_figure(xlabel='LMS Iterations', ylabel='Q Gradient (imag.)')
        # plt.show()

        return error_lms_dict

    def gradient_search(self, fine_dict, antenna_in, quantized):
        fine_min = min(fine_dict.items(), key=lambda x: x[1][0])
        cci = fine_min[0][1]
        ccq = fine_min[0][3]
        cci_sign = fine_min[0][0]
        ccq_sign = fine_min[0][2]
        lms_dict = {}
        step_counter = 1

        # opposite sign based on residual dc for lms' first step
        sign_i = -cci_sign * np.sign(fine_min[1][1].real)
        sign_q = -ccq_sign * np.sign(fine_min[1][1].imag)
        sign_change = False

        cc_out = np.complex(1, 1)
        while step_counter <= self.max_lms_steps:
            curr_gain = np.complex(cci_sign * (10 ** ((cci - 511) * self.vm_precision_db_ / 20.0)),
                                   ccq_sign * (10 ** ((ccq - 511) * self.vm_precision_db_ / 20.0)))

            self._lms_cci = np.append(self._lms_cci, cci)
            self._lms_ccq = np.append(self._lms_ccq, ccq)
            self._lms_gain_quantized = np.append(self._lms_gain_quantized, curr_gain)

            prev_cc_out = cc_out

            # get current residual-DC
            for samples in range(self.filter1_settle_samples):
                cw = self.rfa_.cc_fwd_process()
                refl = self.ant_array_.process(cw, antenna_in)
                rfa_cc_out = self.rfa_.cc_process(refl, cci_sign, ccq_sign, cci, ccq)
            cc_out = self.fpga_.cc_process(rfa_cc_out)
            self._lms_residual_dc = np.append(self._lms_residual_dc, np.abs(self.rfa_.combined_[-1]))
            self._lms_residual_dc_fpga = np.append(self._lms_residual_dc_fpga, cc_out)

            lms_dict[cci_sign, cci, ccq_sign, ccq] = (step_counter, np.abs(self.rfa_.combined_[-1]))

            #determine if sign of residual DC has changed to decide whether step-size needs to be reduced
            if((np.sign(prev_cc_out.real) != np.sign(cc_out.real)) or (np.sign(prev_cc_out.imag) != np.sign(cc_out.imag))):
                sign_change = True
            else:
                sign_change = False

            # determine the direction in which to measure gradient (the direction that takes residual DC towards zero)
            # the logic below assumes that the combiner output is computed as (cc - rx) and not the other way around
            if cci_sign > 0:
                sign_i = -1
            else:
                sign_i = 1

            if ccq_sign > 0:
                sign_q = -1
            else:
                sign_q = 1

            # get residual-DC when taking one step in I direction
            cci_update = int(cci + sign_i)
            for samples in range(self.filter1_settle_samples):
                cw = self.rfa_.cc_fwd_process()
                refl = self.ant_array_.process(cw, antenna_in)
                rfa_cc_out = self.rfa_.cc_process(refl, cci_sign, ccq_sign, cci_update, ccq)
            cc_out_i_step = self.fpga_.cc_process(rfa_cc_out)

            # get residual-DC when taking one step in the Q direction
            ccq_update = int(ccq + sign_q)
            for samples in range(self.filter1_settle_samples):
                cw = self.rfa_.cc_fwd_process()
                refl = self.ant_array_.process(cw, antenna_in)
                rfa_cc_out = self.rfa_.cc_process(refl, cci_sign, ccq_sign, cci, ccq_update)
            cc_out_q_step = self.fpga_.cc_process(rfa_cc_out)

            grad_i = cc_out_i_step - cc_out
            grad_q = cc_out_q_step - cc_out
            self._lms_grad_i = np.append(self._lms_grad_i, grad_i)
            self._lms_grad_q = np.append(self._lms_grad_q, grad_q)

            new_gain = self.lms_update(curr_gain, cc_out, grad_i, grad_q, sign_change)
            self._lms_gain = np.append(self._lms_gain, new_gain)

            # quantize to nearest cci and ccq
            cci_sign, cci, gain_i_quant = self.quantize_gains(new_gain.real)
            ccq_sign, ccq, gain_q_quant = self.quantize_gains(new_gain.imag)

            step_counter += 1

        ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        plt.subplot(211)
        plt.plot(self._lms_residual_dc_fpga.real, '.-')
        ut.annotate_figure(xlabel='', ylabel='Residual DC (real)')
        plt.subplot(212)
        plt.plot(self._lms_residual_dc_fpga.imag, '.-')
        ut.annotate_figure(xlabel='LMS Iterations', ylabel='Residual DC (imag.)')

        ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        plt.subplot(211)
        plt.plot(self._lms_gain.real, '.-')
        plt.plot(self._lms_gain_quantized.real, 'r.-')
        ut.annotate_figure(xlabel='', ylabel='Gain (real)')
        plt.subplot(212)
        plt.plot(self._lms_gain.imag, '.-')
        plt.plot(self._lms_gain_quantized.imag, 'r.-')
        ut.annotate_figure(xlabel='LMS Iterations', ylabel='Gain (imag.)')

        ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        plt.subplot(211)
        plt.plot(self._lms_cci, '.-')
        ut.annotate_figure(xlabel='', ylabel='CC Control (real)')
        plt.subplot(212)
        plt.plot(self._lms_ccq, '.-')
        ut.annotate_figure(xlabel='LMS Iterations', ylabel='CC Control (imag.)')

        ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        plt.subplot(211)
        plt.plot(self._lms_grad_i.real, '.-')
        ut.annotate_figure(xlabel='', ylabel='I Gradient (real)')
        plt.subplot(212)
        plt.plot(self._lms_grad_i.imag, '.-')
        ut.annotate_figure(xlabel='LMS Iterations', ylabel='I Gradient (imag.)')

        ut.create_figure_doc(h=3, w=6, nrows=2, ncols=1, sharex=True)
        plt.subplot(211)
        plt.plot(self._lms_grad_q.real, '.-')
        ut.annotate_figure(xlabel='', ylabel='Q Gradient (real)')
        plt.subplot(212)
        plt.plot(self._lms_grad_q.imag, '.-')
        ut.annotate_figure(xlabel='LMS Iterations', ylabel='Q Gradient (imag.)')
        plt.show()

        return lms_dict

    def cc_final_search(self, lms_dict):
        lms_last_step = max(lms_dict.items(), key=lambda x: x[1][0])
        # print('Result of LMS')
        # print(lms_last_step)
        lms_last_i = lms_last_step[0][1]
        lms_last_q = lms_last_step[0][3]
        sign_lms_last_i = lms_last_step[0][0]
        sign_lms_last_q = lms_last_step[0][2]
        super_fine_step = 1
        lms_i_min = np.clip(np.abs(lms_last_i - (super_fine_step)), 0, None)
        lms_i_max = np.clip(np.abs(lms_last_i + (2 * super_fine_step)), None, 511)
        lms_q_min = np.clip(np.abs(lms_last_q - (super_fine_step)), 0, None)
        lms_q_max = np.clip(np.abs(lms_last_q + (2 * super_fine_step)), None, 511)
        if lms_last_i == 0:
            lms_i_min = 0
        if lms_last_q == 0:
            lms_q_min = 0
        error_dict_final = {}
        for atten_i in np.arange(lms_i_min, lms_i_max, super_fine_step):
            for atten_q in np.arange(lms_q_min, lms_q_max, super_fine_step):
                for samples in range(self.filter1_settle_samples):
                    cw = self.rfa_.cc_fwd_process()
                    refl = self.ant_array_.process(cw)
                    rfa_cc_out = self.rfa_.cc_process(refl, sign_lms_last_i, sign_lms_last_q, atten_i, atten_q)
                cc_out = self.fpga_.cc_process(rfa_cc_out)
                error_dict_final[(sign_lms_last_i, atten_i, sign_lms_last_q, atten_q)] = np.abs(self.rfa_.combined_[-1])
        return error_dict_final

    def calculate_ideal(self):
        tx_phase = 0
        carrier_phasor = np.exp(1j * tx_phase)
        cw = self.rfa_.tx_modulator_.process(self.rfa_.tx_amp_, carrier_phasor)
        ant_in = (cw * self.rfa_.forward_path_gain_)
        refl = self.ant_array_.process(ant_in)
        ideal_i, ideal_q, ideal_residual = self.rfa_.calculate_ideal(refl, cw)
        return ideal_i, ideal_q, ideal_residual

    def rx_process(self, antenna_in):
        tx = self.rfa_.cc_fwd_process()
        rx_process_data = self.ant_array_.process(tx, antenna_in)
        plt.show()
        return rx_process_data

    def rx_batch_process(self, backscatter_wvfm, config):

        num_samples = np.size(backscatter_wvfm)
        cw = self.rfa_.cc_fwd_batch_process(num_samples)

        antenna_receive = self.ant_array_.batch_process(cw, backscatter_wvfm)

        # print("The length of the wvfm after the ant_array is ", len(backscatter_wvfm))
        # plt.plot(np.real(antenna_receive), ".-")
        # plt.title("wvfm after the ant_array")
        # plt.figure()

        rfa_cc_out = self.rfa_.cc_batch_process(antenna_receive, config.cc_i_sign, config.cc_q_sign, config.cc_i, config.cc_q)

        # print("The length of the wvfm after the rfa is ", len(backscatter_wvfm))
        # plt.plot(np.real(rfa_cc_out), ".-")
        # plt.plot(np.imag(rfa_cc_out), ".-")
        # plt.title("wvfm after the rfa")
        # plt.figure()

        decoded_bits = self.fpga_.rx_batch_process(rfa_cc_out, self.link, config)

        return decoded_bits




