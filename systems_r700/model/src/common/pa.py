'''
Created on Nov 13, 2017

@author: zchen
'''
import systems_r700.model.src.common.utils as ut
import numpy as np
class PowerAmplifier(object):
    '''
    Base Power Amplifier class. Distortion to be added later
    '''
    
    def __init__(self, gain_db=29):
        '''
        Initialize PA with 29dB of default gain
        '''
        self.pa_bias = 0
        self.gain = ut.db2lin(gain_db)
        
    def set_pa_bias(self, pa_bias=0):
        '''
        Currently, any non-zero bias will enable the PA
        '''
        self.pa_bias = pa_bias
        
    def process(self, pa_in):
        '''
        Complexity to be added later once more HW characterization is done
        '''
        output = pa_in
        if self.pa_bias != 0:
            #Make the complex values explicit
            output = np.real(pa_in)*self.gain + 1j*np.imag(pa_in)*self.gain
        return output