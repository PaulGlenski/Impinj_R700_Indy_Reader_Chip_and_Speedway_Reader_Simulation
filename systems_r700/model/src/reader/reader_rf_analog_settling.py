'''
Created on Nov 23, 2017

@author: zchen
'''
from systems_r700.model.src.reader.reader_attributes import ReaderAttributes
import numpy as np
from scipy.interpolate import interp1d

class ReaderRfAnalogLoopSettling(object):
    '''
    Models time-dependent RF analog parameters
    '''

    def __init__(self, max_loop_wait_us=10000, tx_settling_time_us=10, ts_sec=(1.0/(20.48e6))):
        '''
        Params:
            time_domain: time range to model analog behavior. Is a vector of increasing time values
        '''
        self.tx_settling_time_us = tx_settling_time_us
        self.curr_voltage = 0 
        self.max_loop_wait_us = max_loop_wait_us
        self.v_settling_function_us = None
        self.settling_time_init(ts_sec)
        print("reader tx settling time: ", tx_settling_time_us)
        
    def settling_time_init(self, ts_sec):
        ts_us = ts_sec*(10**6)
        t = np.arange(0, self.max_loop_wait_us, ts_us)
        self.v_settling_function_us = interp1d(t, np.zeros(len(t)), fill_value = 'extrapolate')
    
    def settling_reset(self):
        self.curr_voltage = 0 
    
    #Precomputes a settling time voltage
    def generate_tx_settling_time(self, voltage_target, ts_seconds):
        '''
        Precomputes settling time voltage for a given voltage target 
        and an initialized settling time parameter
        Params:
            voltage_target: target voltage of the settling function
            ts: time step in seconds
        
        Returns:
        '''
        ts_us = ts_seconds*(10**6)
        dv = voltage_target - self.curr_voltage
        t = np.arange(0, self.max_loop_wait_us, ts_us)

#             print "dv = {}, target_voltage = {}, curr_voltage = {}".format(dv, voltage_target, self.curr_voltage)
        v_settling_function_us = self.curr_voltage + dv*(1 - np.exp(-t/self.tx_settling_time_us))
        self.v_settling_function_us = interp1d(t, v_settling_function_us, fill_value = 'extrapolate')
            
    def get_tx_settling_time_voltage(self, t_sec):
        t = t_sec * 10**6
        curr_voltage = self.v_settling_function_us(t)
        return curr_voltage
    
    def update_curr_settled_voltage(self, settled_voltage):
        self.curr_voltage = settled_voltage
    
    
        
        
        