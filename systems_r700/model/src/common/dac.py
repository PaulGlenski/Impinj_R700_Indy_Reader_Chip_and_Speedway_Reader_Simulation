import numpy as np

class Dac(object):
    """ DAC Component class. Takes in the following parameters:
        - r_load: Output load for a current DAC
        - r_tolerance: Tolerance of r_load
        - fs_output_I: Full scale output current in Amps for current DAC
        - bits: Number of input bits
        - fs_output_V: Full scale output voltage in Volts for voltage DAC
        - is_complex: Whether the output signal is complex
    """
    def __init__(self, r_tolerance=0.01, bits=12, fs_output_V=2.0):
        self.r_tolerance = r_tolerance
        self.fs_output_V = fs_output_V*(1 + np.random.uniform(-1, 1)*self.r_tolerance)
        self.bits = bits
        self.out = 0
    
    def process(self, dac_in):
        out = (float(dac_in)/(2**self.bits-1)) * self.fs_output_V
        self.out = out
        return out
    
    @property
    def dac_output(self):
        return self.out