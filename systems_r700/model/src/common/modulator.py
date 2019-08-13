##
#  This class models nn RF modulator.  It may be used for either up-conversion (positive frequency) or
#  down-conversion (negative frequency)
##
import numpy as np

class Modulator(object):

    # constructor
    def __init__(self, gain=1.0):
        self.gain = gain

    # main processing
    def process(self, x, carrier_phasor):
        y = self.gain * x * carrier_phasor
        return y

    # def batch_process(self, x, carrier_phasor, num_samples):
    #     y_array = np.ones(num_samples)
    #     y = self.gain * x * carrier_phasor * num_samples
    #     y_array = y_array * y
    #     return y_array
