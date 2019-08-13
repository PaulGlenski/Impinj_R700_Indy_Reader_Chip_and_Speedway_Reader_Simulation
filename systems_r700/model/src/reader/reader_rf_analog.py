##
#  This class models the RF and analog circuitry of the reader hardware.
#  Currently this is modeled in baseband under the assumption that all operations
#  are linear.  When we determine that certain non-linearities are important to
#  model to determine and characterize performance, then we will need to convert
#  this model to RF.
##

import math
import numpy as np
import scipy.constants as consts
import matplotlib.pyplot as plt

import systems_r700.model.src.common.utils as ut
from systems_r700.model.src.common.awgn import Awgn
from systems_r700.model.src.common.analog_filter import AnalogFilter
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.reader.reader_impairments import ReaderImpairments
from systems_r700.model.src.reader.vga import Vga
from systems_r700.model.src.common.pdet import Pdet
from systems_r700.model.src.common.local_oscillator import LocalOscillator
from systems_r700.model.src.common.modulator import Modulator
from systems_r700.model.src.common.dac import Dac
from systems_r700.model.src.common.pa import PowerAmplifier
from systems_r700.model.src.reader.reader_rf_analog_settling import ReaderRfAnalogLoopSettling

class ReaderRfAnalog(object):

    # constructor
    def __init__(self, attrib=ReaderAttributes(), impairments=ReaderImpairments()):
        # Save off incoming attribute list
        self.sim_ctrl_ = attrib.sim_ctrl_
        self.common_attrib_ = attrib.common_
        self.rx_attrib_ = attrib.rx_
        self.tx_attrib_ = attrib.tx_

        # instantiate sub-modules
        #TODO: SETS UP CARRIER PHASOR, LO
        self.carrier_phasor = np.complex(1.0, 0.0)
        self.lo = LocalOscillator(self.common_attrib_.randomize_lo_initial_phase, self.rx_attrib_.phase_noise_freq_hz, self.rx_attrib_.phase_noise_dbc,
                                  self.rx_attrib_.fs_)
        self.tx_modulator_ = Modulator(1.0)

        rf_demod_gain = ut.db2lin(self.rx_attrib_.rf_demod_gain_db_)
        self.rx_demodulator_ = Modulator(rf_demod_gain)

        self.filter1_input_thermal_noise = Awgn()
        self.rx_adc_thermal_noise = Awgn()

        self.filter1 = AnalogFilter(self.rx_attrib_.filter1_order, self.rx_attrib_.filter1_passband_ripple,
                                    self.rx_attrib_.fs_, self.rx_attrib_.filter1_bw)

        # Set Tx amplitude
        self.tx_amp_ = ut.dbm2amp(attrib.tx_.tx_pwr_dbm_) * math.sqrt(attrib.common_.impedance_)

        # Compute combined path gains
        self.forward_path_gain_ = 1.0
        self.pre_cc_return_path_gain_ = 1.0
        self.cc_path_gain_ = 1.0
        self.cc_combiner_gain_ = 1.0
        self.post_cc_rf_gain_ = 1.0
        self.post_demod_aux_adc_gain_ = 1.0
        self.post_demod_rx_adc_gain_ = 1.0
        self.tx_post_modulator_gain = 1.0
        self.tx_pdet_path_gain = 1.0
        self.tx_coupler_gain = 1.0
        self.compute_combined_gains()

        self.filter1_input_noise_spectral_density = 0.0
        self.rx_adc_noise_spectral_density = 0.0
        self.compute_thermal_noise_densities(impairments.rx.temperature)
        self.filter1_input_thermal_noise.set_noise_spectral_density(self.filter1_input_noise_spectral_density, self.rx_attrib_.fs_)
        self.rx_adc_thermal_noise.set_noise_spectral_density(self.rx_adc_noise_spectral_density, self.rx_attrib_.fs_)

        # Carrier frequency
        self.fc_ = attrib.rx_.fc_/attrib.rx_.fs_

        # time
        self.cw_sample_ = 0
        self.time_step_ = 1/attrib.rx_.fs_
        self.time_ = 0.0

        # signals saved for debugging and analysis
        self.bb_tx_cw = []
        self.cw_ = []
        self.rx_ = []
        self.cc_ = []
        self.combined_ = []
        self.demod_out_ = []
        self.filter1_in_ = []
        self.filter1_out_ = []
        self.aux_adc_out_ = []
        self.aux_adc_out_with_noise_ = []
        self.rx_adc_out_ = []
        self.rx_adc_out_with_noise_ = []

        self.bb_ = []
        self.tx_pwr = []
        self.tx_pwr_sampled = []
        self.pa_voltage_target = -1

        #Tx Path Components
        self.tx_dac_i = Dac(bits=attrib.tx_.tx_dac_num_bits, fs_output_V=attrib.tx_.tx_dac_voltage_range)
        self.tx_dac_q = Dac(bits=attrib.tx_.tx_dac_num_bits, fs_output_V=attrib.tx_.tx_dac_voltage_range)
        self.vga = Vga(attrib.tx_)
        self.pa = PowerAmplifier()
        self.pdet = Pdet()
        self.tx_aux_dac_lo = Dac(bits=attrib.tx_.aux_dac_num_bits, fs_output_V=attrib.tx_.aux_dac_voltage_range)
        self.tx_aux_dac_hi = Dac(bits=attrib.tx_.aux_dac_num_bits, fs_output_V=attrib.tx_.aux_dac_voltage_range)
        self.pdet_out = 0
        self.output_impedance = attrib.common_.impedance_
        self.analog_settling = ReaderRfAnalogLoopSettling(tx_settling_time_us=attrib.tx_.tx_settling_time_us)
        self.tx_ts = 1.0/self.rx_attrib_.fs_
        self.curr_step_time_s = 0
    
    #Resistor Divider network at the output of the Aux DAC
    def _aux_dac_resistor_divider_network(self, txg_h_out=0, txg_l_out=0, r1=280, r2=510e3, r3=10e3, r4=560, tolerance=.01):
        r1 = r1*(1+np.random.uniform(-1, 1)*tolerance)
        r2 = r2*(1+np.random.uniform(-1, 1)*tolerance) 
        r3 = r3*(1+np.random.uniform(-1, 1)*tolerance)
        r4 = r4*(1+np.random.uniform(-1, 1)*tolerance)
        denom = (r1*r2*(r3+r4) + r3*r4*(r1+r2))/(r1*r2*r3*r4)
        numer = (txg_h_out*r2 + txg_l_out*r1)/(r1*r2)
        return float(numer)/float(denom)

    # compute the aggregated gains of various relevant paths and save off the combined linear gain value
    def compute_combined_gains(self):
        forward_path_gain_db_ = self.common_attrib_.coupler_il_gain_db_ + self.common_attrib_.rf_mux_gain_db_
        self.forward_path_gain_ = ut.db2lin(forward_path_gain_db_)

        pre_cc_return_path_gain_db = self.common_attrib_.rf_mux_gain_db_ + self.common_attrib_.coupler_gain_db_ + \
                                    self.rx_attrib_.pre_cc_pad_gain_db_ + self.rx_attrib_.pre_cc_spdt1_gain_db_ + \
                                    self.rx_attrib_.rx_phase_shift_gain_db_ + self.rx_attrib_.pre_cc_spdt2_gain_db_
        self.pre_cc_return_path_gain_ = ut.db2lin(pre_cc_return_path_gain_db)

        cc_path_gain_db = self.common_attrib_.coupler_gain_db_ + self.rx_attrib_.quadrature_hybrid1_gain_db_ + \
                                self.rx_attrib_.coarse_cc_attenuation_gain_db_ + \
                                self.rx_attrib_.fine_cc_attenuation_gain_db_ + \
                                self.rx_attrib_.quadrant_select_switch_gain_db_ + \
                                self.rx_attrib_.quadrature_hybrid2_gain_db_
        self.cc_path_gain_ = ut.db2lin(cc_path_gain_db)

        self.cc_combiner_gain_ = ut.db2lin(self.rx_attrib_.cc_combiner_gain_db_)

        post_cc_rf_gain_db = self.rx_attrib_.post_cc_spdt1_gain_db_ + self.rx_attrib_.post_cc_spdt2_gain_db_ + \
                             self.rx_attrib_.pre_lna_hpf_gain_db_ + self.rx_attrib_.lna_gain_db_ + \
                             self.rx_attrib_.post_lna_pad_gain_db_ + self.rx_attrib_.rf_demod_gain_db_
        if self.rx_attrib_.add_atten_pad_:
            post_cc_rf_gain_db += self.rx_attrib_.post_cc_pad_gain_db_

        post_demod_aux_adc_gain_db = self.rx_attrib_.filter1_gain_db_
        post_demod_rx_adc_gain_db = self.rx_attrib_.filter1_gain_db_ + self.rx_attrib_.preamp_gain_db_ + \
                                    self.rx_attrib_.filter2_gain_db_ + self.rx_attrib_.driver_gain_db_ + \
                                    self.rx_attrib_.adc_gain_db_

        self.post_cc_rf_gain_ = ut.db2lin(post_cc_rf_gain_db)
        self.post_demod_aux_adc_gain_ = ut.db2lin(post_demod_aux_adc_gain_db)
        self.post_demod_rx_adc_gain_ = ut.db2lin(post_demod_rx_adc_gain_db)
        
        #Tx Path Gains
        post_modulator_gain_db = self.tx_attrib_.pa_driver_amp_gain_db + self.tx_attrib_.saw_filter_gain_db
        self.coupler_gain = ut.db2lin(self.tx_attrib_.bi_d_coupler_gain_db)
        self.tx_post_modulator_gain = ut.db2lin(post_modulator_gain_db)
        self.tx_pdet_path_gain = self.tx_attrib_.rf_pdet_out_gain
        self.tx_antenna_gain_dB = self.tx_attrib_.antenna_gain_db

    def compute_thermal_noise_densities(self, temp):
        n0_incident = 10*np.log10((consts.Boltzmann * consts.convert_temperature(float(temp), 'C', 'K')) / 1e-3)
        print(n0_incident)
        n0_rf = n0_incident + self.rx_attrib_.nf_rf_path
        print(n0_rf)
        n0_rf_gain = n0_rf + 20*np.log10(self.pre_cc_return_path_gain_ * self.post_cc_rf_gain_)
        print(n0_rf_gain)
        n0_rf_lin = (10 ** (n0_rf_gain/10)) * 1e-3
        print(n0_rf_lin)
        spectral_density_rf = np.sqrt(n0_rf_lin * self.common_attrib_.impedance_) * 1e9
        print(spectral_density_rf)
        spectral_density_demod = spectral_density_rf * ut.db2lin(self.rx_attrib_.rf_demod_gain_db_)
        print(spectral_density_demod)

        spectral_density_filter1_input = np.sqrt((spectral_density_demod ** 2) + (self.rx_attrib_.filter1_psd_ ** 2))

        spectral_density_preamp = self.rx_attrib_.preamp_psd_  * ut.db2lin(self.rx_attrib_.preamp_gain_db_)

        spectral_density_filter2 = np.sqrt((spectral_density_preamp ** 2) + (self.rx_attrib_.filter2_psd_ ** 2)) * \
                                   ut.db2lin(self.rx_attrib_.filter2_gain_db_)
        spectral_density_driver = np.sqrt((spectral_density_filter2 ** 2) + (self.rx_attrib_.driver_psd_ ** 2)) * \
                                   ut.db2lin(self.rx_attrib_.driver_gain_db_)
        spectral_density_rx_adc = np.sqrt((spectral_density_driver ** 2) + (self.rx_attrib_.adc_psd_ ** 2)) * \
                                   ut.db2lin(self.rx_attrib_.adc_gain_db_)

        self.filter1_input_noise_spectral_density = spectral_density_filter1_input
        self.rx_adc_noise_spectral_density = spectral_density_rx_adc

        print('n0_incident: ' + str(n0_incident))
        print('n0_rf: ' + str(n0_rf))
        print('RF Gain: ' + str(20 * np.log10(self.pre_cc_return_path_gain_ * self.post_cc_rf_gain_)))
        print('n0_rf_gain: ' + str(n0_rf_gain))
        print('Spectral Density at RF: ' + str(spectral_density_rf))
        print('Spectral Density at Demod: ' + str(spectral_density_demod))
        print('Spectral Density at Filter1 Input: ' + str(spectral_density_filter1_input))
        print('Spectral Density at Preamp: ' + str(spectral_density_preamp))
        print('Spectral Density at Filter2: ' + str(spectral_density_filter2))
        print('Spectral Density at Driver: ' + str(spectral_density_driver))
        print('Spectral Density at ADC: ' + str(spectral_density_rx_adc))

    def tx_dac_out_process(self, mod_gain_out_I, mod_gain_out_Q):
        tx_dac_out_i = self.tx_dac_i.process(mod_gain_out_I)
        tx_dac_out_q = self.tx_dac_q.process(mod_gain_out_Q)
        return tx_dac_out_i + tx_dac_out_q*1j
    
    def set_pa_bias(self, bias):
        self.pa.set_pa_bias(bias)
    
    def _get_adc_voltage(self, pa_voltage):
        self.pdet_out = self.pdet.process(np.abs(pa_voltage)*self.coupler_gain)*self.tx_pdet_path_gain
        return self.pdet_out
    
    def _get_target_pa_voltage(self, tx_dac_out, aux_dac_in):
#         print " aux dac: ", aux_dac_in
#         print " aux upper 10 bits:  ", aux_dac_in >> 6
#         print " aux lower 6 bits:  ", aux_dac_in & 0x3F
        aux_dac_out_hi = self.tx_aux_dac_hi.process( (aux_dac_in >> 6)) #upper 10 bits
        aux_dac_out_lo = self.tx_aux_dac_lo.process( (aux_dac_in & 0x3F)) #lower 6 bits
        aux_dac_out = self._aux_dac_resistor_divider_network(txg_h_out=aux_dac_out_hi, txg_l_out=aux_dac_out_lo)
        vga_output = self.vga.process(tx_dac_out, aux_dac_out)
        pa_in_voltage = vga_output*self.tx_post_modulator_gain
        pa_voltage_target = np.abs(self.pa.process(pa_in_voltage))
        if(pa_voltage_target > 11):
            pa_voltage_target = 11
        return pa_voltage_target
    
    def _get_tx_pwr_dbm(self, pa_voltage):
        return (10.0*np.log10(np.abs(pa_voltage)**2/self.output_impedance) + 30)

    def tx_ramp_process(self, tx_dac_out, aux_dac_in):
        '''
        Simulates analog rampup behavior using ticks of 1/fs:
        @Params:
        - tx_dac_out: The tx baseband signal to be amplified
        - aux_dac_in: The tx gain control signal
        '''
        
        #Initial State Variables
        tx_gain_d2a_delay_s = self.tx_attrib_.d2a_delay_us/1.0e6
        tx_gain_a2d_delay_s = self.tx_attrib_.a2d_delay_us/1.0e6
        tx_gain_code_delay_s = self.tx_attrib_.code_delay_us/1.0e6
        
        #Wait for the DAC write to go out
        self._tx_ramp_step_dac_wait(self.curr_step_time_s, tx_gain_d2a_delay_s)

        #New waveform generated from DAC write
        pa_voltage_target = self._get_target_pa_voltage(tx_dac_out, aux_dac_in)
        self._tx_ramp_step_start(pa_voltage_target)
        self.curr_step_time_s = 0
        
        #Wait in the analog domain (code running)
        self.curr_step_time_s = self._tx_ramp_step_code_wait(self.curr_step_time_s, tx_gain_code_delay_s)
        
        #Wait for the ADC Read to Come in
        self.curr_step_time_s = self._tx_ramp_step_adc_wait(self.curr_step_time_s, tx_gain_a2d_delay_s)
        pa_voltage = self.analog_settling.get_tx_settling_time_voltage(self.curr_step_time_s )
        self.tx_pwr_sampled.append(self._get_tx_pwr_dbm(pa_voltage))
        
        #print "aux_dac_in {}, pa_voltage: {}, pa_voltage_target: {}".format(aux_dac_in, pa_voltage, pa_voltage_target)
        return self._get_adc_voltage(pa_voltage)

    def _tx_wait_ts(self, curr_step_time_s, time_to_wait_s):
        '''
        Waits in increments of 1/fs where fs = 20.48e6 
        '''
        t = curr_step_time_s
        while t < curr_step_time_s + time_to_wait_s:
            pa_voltage = self.analog_settling.get_tx_settling_time_voltage(t)
            self.tx_pwr.append(self._get_tx_pwr_dbm(pa_voltage))
            t += self.tx_ts
        return t

    def _tx_ramp_step_dac_wait(self, curr_step_time_s, d2a_delay_sec):
        '''
        Apply a wait while the dac is still settling
        @Params:
        - curr_step_time_s: current time elapsed in the current step
        - d2a_delay_sec: The amount of digital latency for d2a conversion
        '''
        t = self._tx_wait_ts(curr_step_time_s, d2a_delay_sec)
        pa_voltage = self.analog_settling.get_tx_settling_time_voltage(t)
        self.analog_settling.update_curr_settled_voltage(pa_voltage)

    def _tx_ramp_step_start(self, pa_voltage_target):
        '''
        DAC has already settled, precompute PA rise for the new DAC
        @Params:
        - pa_voltage_target: Target voltage that needs to be computed
        '''
        self.analog_settling.generate_tx_settling_time(pa_voltage_target, self.tx_ts)

    def _tx_ramp_step_code_wait(self, curr_step_time_sec, wait_time_sec):
        '''
        Adds a wait in the rise-time to simulate a delay in the code
        @param:
        - curr_step_time_sec: current time elapsed in the current step
        - wait_time_sec: amount of time the code is running for
        '''
        return self._tx_wait_ts(curr_step_time_sec, wait_time_sec)

    def _tx_ramp_step_adc_wait(self, curr_step_time_s, a2d_delay_sec):
        '''
        Adds a wait in the rise-time to simulate a delay in the code
        @param:
        - curr_step_time_sec: current time elapsed in the current step
        - a2d_delay_sec: latency in a2d conversion
        '''
        return self._tx_wait_ts(curr_step_time_s, a2d_delay_sec)
        
        
    # forward CW path for carrier-cancellation
    # CW -> Antenna
    def cc_fwd_process(self):
        self.bb_tx_cw.append(self.tx_amp_)
        self.carrier_phasor = self.lo.process()
        if self.sim_ctrl_.disable_all_noise_ or self.sim_ctrl_.disable_phase_noise_:
            cw = self.tx_amp_
        else:
            cw = self.tx_modulator_.process(self.tx_amp_, self.carrier_phasor)
        self.cw_.append(cw)
        ant_in = (self.forward_path_gain_ * cw)
        return ant_in

    def cc_fwd_batch_process(self, num_samples):
        self.bb_tx_cw.append(self.tx_amp_)
        self.carrier_phasor = self.lo.batch_process(num_samples)
        if self.sim_ctrl_.disable_all_noise_ or self.sim_ctrl_.disable_phase_noise_:
            cw = self.tx_amp_ * np.ones(num_samples)
        else:
            cw = self.tx_modulator_.process(self.tx_amp_, self.carrier_phasor)
        self.cw_batch = cw
        ant_in = (self.forward_path_gain_ * cw)
        return ant_in
    #ant_in is tx in the form of num_samples x 1 array

    # carrier-cancellation path
    # CW      -> cancellation-combiner
    # Antenna -> cancellation-combiner
    def cc_process(self, refl, cci_sign, ccq_sign, cci, ccq):
        rx = refl * self.pre_cc_return_path_gain_
        self.rx_.append(rx)

        atten_i_lin = cci_sign * np.real(self.rx_attrib_.vm_attenuations_[cci, ccq])
        atten_q_lin = ccq_sign * np.imag(self.rx_attrib_.vm_attenuations_[cci, ccq])

        cc = ((np.abs(self.cw_[-1]) * atten_i_lin) +
              (1j * (np.abs(self.cw_[-1]) * atten_q_lin))) * self.cc_path_gain_
        self.cc_.append(cc)

        combined = (cc + rx) * self.cc_combiner_gain_
        self.combined_.append(combined)

        self.cw_sample_ += 1

        # RF Demodulator
        if self.sim_ctrl_.disable_all_noise_ or self.sim_ctrl_.disable_phase_noise_:
            demod_out = combined
        else:
            demod_out = self.rx_demodulator_.process(combined, np.conj(self.carrier_phasor))
        self.demod_out_.append(demod_out)

        # First thermal noise source before analog LPF
        if self.sim_ctrl_.disable_all_noise_ or self.sim_ctrl_.disable_thermal_noise_:
            filter1_in = demod_out
        else:
            filter1_in = self.filter1_input_thermal_noise.add_noise(demod_out)
        self.filter1_in_.append(filter1_in)

        # Analog LPF (Filter-1)
        if self.sim_ctrl_.bypass_rx_baseband_filter_:
            filter1_out = filter1_in
        else:
            filter1_out = self.filter1.process(filter1_in)
        self.filter1_out_.append(filter1_out)

        # post RF-Demod path to auxiliary ADC
        aux_adc_out = filter1_out * self.post_demod_aux_adc_gain_
        self.aux_adc_out_.append(aux_adc_out)

        # post RF-Demod path to main RX ADC
        rx_adc_out = filter1_out * self.post_demod_rx_adc_gain_
        if self.sim_ctrl_.disable_all_noise_ or self.sim_ctrl_.disable_thermal_noise_:
            rx_adc_out_with_noise = rx_adc_out
        else:
            rx_adc_out_with_noise = self.rx_adc_thermal_noise.add_noise(rx_adc_out)
        self.rx_adc_out_.append(rx_adc_out)

        # self.bb_.append(rx_adc_out_with_noise)
        self.bb_.append(rx_adc_out_with_noise)
        return aux_adc_out

    def cc_batch_process(self, refl, cci_sign, ccq_sign, cci, ccq):
        rx = refl * self.pre_cc_return_path_gain_

        atten_i_lin = cci_sign * np.real(self.rx_attrib_.vm_attenuations_[cci, ccq])
        atten_q_lin = ccq_sign * np.imag(self.rx_attrib_.vm_attenuations_[cci, ccq])

        cc = ((np.abs(self.cw_batch) * atten_i_lin) +
              (1j * (np.abs(self.cw_batch) * atten_q_lin))) * self.cc_path_gain_

        self.cc_.append(cc)
        #self.cc_.append(cc)

        combined = (cc + rx) * self.cc_combiner_gain_
        #self.combined_.append(combined)
        #Combined is cc-rx, being passed into fpga

        # RF Demodulator
        if self.sim_ctrl_.disable_all_noise_ or self.sim_ctrl_.disable_phase_noise_:
            demod_out = combined
        else:
            demod_out = self.rx_demodulator_.process(combined, np.conj(self.carrier_phasor))
        #self.demod_out_.append(demod_out)

        # First thermal noise source before analog LPF
        if self.sim_ctrl_.disable_all_noise_ or self.sim_ctrl_.disable_thermal_noise_:
            filter1_in = demod_out
        else:
            noise = self.filter1_input_thermal_noise.gen_noise(len(demod_out))
            filter1_in = demod_out + noise
        #self.filter1_in_.append(filter1_in)

        # Analog LPF (Filter-1)
        if self.sim_ctrl_.bypass_rx_baseband_filter_:
            filter1_out_list = filter1_in
        else:
            filter1_out_list = self.filter1.process(filter1_in)
        filter1_out = np.asarray(filter1_out_list)
        #self.filter1_out_.append(filter1_out)

        # post RF-Demod path to auxiliary ADC
        aux_adc_out = filter1_out * self.post_demod_aux_adc_gain_
        #self.aux_adc_out_.append(aux_adc_out)

        # post RF-Demod path to main RX ADC
        rx_adc_out = filter1_out * self.post_demod_rx_adc_gain_
        if self.sim_ctrl_.disable_all_noise_ or self.sim_ctrl_.disable_thermal_noise_:
            rx_adc_out_with_noise = rx_adc_out
        else:
            noise = self.rx_adc_thermal_noise.gen_noise(len(demod_out))
            rx_adc_out_with_noise = noise + rx_adc_out
        #self.rx_adc_out_.append(rx_adc_out)

        #self.bb_.append(rx_adc_out_with_noise)
        return aux_adc_out

    def calculate_ideal(self, refl, cw):
        rx = refl * self.pre_cc_return_path_gain_
        self.rx_.append(rx)

        db_prec = 0.0625
        db_clip_max = 31.9375

        solution_i = abs(rx.real) / (np.abs(self.cw_[-1]) * self.cc_path_gain_)
        solution_q = abs(rx.imag) / (np.abs(self.cw_[-1]) * self.cc_path_gain_)

        if np.sign(rx.real) == 0.0:
            sign_i = 1.0
        else:
            sign_i = np.sign(rx.real)

        if np.sign(rx.imag) == 0.0:
            sign_q = 1.0
        else:
            sign_q = np.sign(rx.imag)

        if solution_i <= 1e-15:
            solution_i_db = db_clip_max
            solution_q_db = np.clip(-1 * ut.lin2db(solution_q), db_prec, db_clip_max)
            ideal_i_db = np.round(solution_i_db / db_prec) * db_prec
            ideal_q_db = np.round(solution_q_db / db_prec) * db_prec
            atten_i_lin = sign_i * ut.db2lin(-ideal_i_db)
            atten_q_lin = sign_q * ut.db2lin(-ideal_q_db)
        elif solution_q <= 1e-15:
            solution_i_db = np.clip(-1 * ut.lin2db(solution_i), db_prec, db_clip_max)
            solution_q_db = db_clip_max
            ideal_i_db = np.round(solution_i_db / db_prec) * db_prec
            ideal_q_db = np.round(solution_q_db / db_prec) * db_prec
            atten_i_lin = sign_i * ut.db2lin(-ideal_i_db)
            atten_q_lin = sign_q * ut.db2lin(-ideal_q_db)
        else:
            solution_i_db = np.clip(-1 * ut.lin2db(solution_i), db_prec, db_clip_max)
            solution_q_db = np.clip(-1 * ut.lin2db(solution_q), db_prec, db_clip_max)
            ideal_i_db = np.round(solution_i_db / db_prec) * db_prec
            ideal_q_db = np.round(solution_q_db / db_prec) * db_prec
            atten_i_lin = sign_i * ut.db2lin(-ideal_i_db)
            atten_q_lin = sign_q * ut.db2lin(-ideal_q_db)

        cc = ((np.abs(cw) * atten_i_lin) +
              (1j * (np.abs(cw) * atten_q_lin))) * self.cc_path_gain_
        combined = (cc - rx) * self.cc_combiner_gain_

        out = combined

        return sign_i * ideal_i_db, sign_q * ideal_q_db, out