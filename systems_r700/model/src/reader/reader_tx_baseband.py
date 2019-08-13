#Models Tx BB

import numpy as np
from scipy.interpolate import interp1d
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
from systems_r700.model.src.reader.reader_mod_gain import Mod_Gain

class ReaderTxBaseband(object):
    """ TxBB Class. Handles the digital Tx block in this simulation.
        Class Input Params:
        - dec_filt_rate Picks the output of the CIC filter block
        - step_size: Default step size is .25dB in picking output power
        - initial_err_val: Inital error value in error integrator
        - Ki: Error integrator constant
    """
    
    def __init__(self, attrib=ReaderAttributes()):
        self.step_size = float(attrib.tx_.pwr_step_size_db);
        self.dec_filt_rate = attrib.tx_.dec_filt_rate
        self.accum_feedback_val = attrib.tx_.bb_initial_err_val #initial error value in error integrator
        self.Ki = attrib.tx_.tx_ramp_up_loop_gain
        self.err_threshold = 4
        self.err = attrib.tx_.bb_initial_err_val #arbitrary error value > 1
        self.power_adc_func = self._generate_cal_table_function(attrib.tx_.tx_cal_data)
        self.mod_gain_offsets = attrib.tx_.mod_gain_offsets
        self.mod_gain_controller = Mod_Gain(unity_pwr_dbm=30, max_pwr_dbm=32.5, unity_gain=0x800, 
                 pwr_dbm=attrib.tx_.tx_pwr_dbm_, min_pwr_dbm=25, step_size=.25)
        self.pwr = attrib.tx_.tx_pwr_dbm_
    
    def _generate_cal_table_function(self, tx_cal_data):
        """
        Generates a function of adc vs tx_pwr given cal data
        @params:
        - tx_cal_data:
        """
        tx_pwrs_dbm = list(tx_cal_data.keys())
        adc_vals = [tx_cal_data[tx_pwr] for tx_pwr in tx_pwrs_dbm]
        return interp1d(tx_pwrs_dbm, adc_vals, assume_sorted=False, fill_value='extrapolate')
                
    def get_mod_gain_output(self, mod_gain_input):
        """ 
        Given an input pwr and the output of the CIC filters, gets the mod_gain_output 
        @params:
        - mod_gain_input: output of the CIC filters
        Returns:
        - mod_gain_output for the give power
        """
        return self.mod_gain_controller.get_mod_gain_output(mod_gain_input)

    def _get_mod_gain_offset(self, pwr):
        """ 
        Given an input pwr and the output of the CIC filters, gets the mod_gain_output_offset
        for the error correction loop
        @params:
        - pwr: power to set
        Returns:
        - Mod gain offset for the given power
        """
        pwr = self.mod_gain_controller._round_pwr_to_nearest_quarterdb(pwr)
        if pwr > 32.5:
            pwr = 32.5
        if pwr < 24.75:
            pwr = 0
        return self.mod_gain_offsets[pwr]
    
    #Returns signed value for fixed point simulation
    def _get_signed_value(self, val, num_bits):
        """
        Takes a value and the position of the sign bit, 
        and returns a num_bits signed value
        @params:
        - val: raw bits of number to convert
        - num_bits: number of bits of value
        """
        sign_bit = (1 << (num_bits - 1))
        sign = val & sign_bit
        return (sign*-1) + ((sign_bit-1) & val)
    
    def get_rampup_error(self):
        return self.err
    
    def power_ctl_loop(self, adc_pwr, target_pwr, rampdown_flag):
        """
        Microblaze implementation of power control
        @params:
        - adc_pwr: forward power read in from the fp detector
        - target_pwr: target_pwr 
        """
        err = target_pwr - adc_pwr
        
        #19 bit signed stored accum out
        accum_fb_val_s = self._get_signed_value(self.accum_feedback_val, 19)
        accum_output = (err + accum_fb_val_s) & 0x7FFFF
        
        #Store the accumulated output for the next iteration
        self.accum_feedback_val = accum_output

        #Round to 17 bits signed
        accum_output = (accum_output + (1 << 1)) >> 2
        accum_output = self._get_signed_value(accum_output, 17)

        #Round down and print debug messages
        intermediate_value = ((self.Ki * accum_output)&(0xFFFFFFFF)) >> 6
        intermediate_value = self._get_signed_value(intermediate_value, 26)
        Ki_Output = intermediate_value + self._get_signed_value(self._get_mod_gain_offset(self.pwr), 14)
        
        #Prevent accumulator wrapping
        if rampdown_flag:
            if Ki_Output < 0:
                Ki_Output = 0
        else:
            if Ki_Output > 0xFFFF:
                Ki_Output = 0xFFFF
        
        print("accum_fb_val_s: {}, accum_feedback_val: {}, ki_output: {}")\
        .format(accum_fb_val_s, self.accum_feedback_val, Ki_Output)
        
        #Get lower 16bits
        self.err = err
        return Ki_Output
    
    def rampdown_process(self, adc_pwr):
        ramp_down_target_pwr = 0
        return self.power_ctl_loop(adc_pwr, ramp_down_target_pwr, True)
        
    
    def rampup_process(self, adc_pwr):
        """ Simulates the fixed point error feedback loop in the FPGA
            Params:
            - adc_pwr: adc reading from the pdet
            Returns:
            - input to the Aux DAC to drive the VGA in the modulator
        """
        #Get the Error (signed 14 bit)
        target_pwr = int(self.power_adc_func(self.pwr))
        return self.power_ctl_loop(adc_pwr, target_pwr, False)
        
        
