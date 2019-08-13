import numpy as np
import scipy.signal as sp
from systems_r700.model.src.common.filter import Filter


class AnalogFilter(Filter):
    """
    This class implements an analog filter as a digital filter.
    """

    # constructor
    def __init__(self, order, ripple, Fs, Fc):
        """

        :param order: int, Order of the analog filter
        :param ripple: float, Pass-band ripple in dB
        :param Fs: float, Sampling frequency of filter implementation in Hz
        :param Fc: float, 3dB edge frequency of low-pass filter in Hz
        """

        B, A = sp.ellip(order, ripple, 40, 2*np.pi*Fc, analog=True)
        b, a = sp.bilinear(B, A, Fs)
        super(AnalogFilter, self).__init__(b, a)
