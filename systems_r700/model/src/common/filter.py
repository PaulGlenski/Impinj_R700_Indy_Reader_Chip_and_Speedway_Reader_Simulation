import numpy as np
import scipy.signal as sp

class Filter:
    """
    This class implements a general purpose digital filter.
    """

    # constructor
    def __init__(self, b, a):
        """

        :param b: float, List of numerator coefficients of H(z)
        :param a: float, List of denominator coefficients of H(z)
        """

        self.b = np.array(b)
        """
        Internal storage of numerator coefficients of the filter 
        """

        self.a = np.array(a)
        """
        Internal storage of the denominator coefficients of the filter
        """

        self.in_taps = np.zeros(self.b.size)
        """
        Tapped delay line of input samples
        """

        self.out_taps = np.zeros(self.a.size - 1)
        """
        Tapped delay line of output samples
        """

    def filter_signal(self, in_sig):
        """

        :param in_sig: float/complex, List of samples representing the full input signal
        :return: float, List of samples representing the full output signal
        """
        out_sig = sp.lfilter(self.b, self.a, in_sig)
        return out_sig

    def process(self, in_samp):
        """
        :param in_samp: float/complex, Input sample
        :return: float/complex, Output sample
        """
        if isinstance(in_samp, np.ndarray) == True:
            output_list = []
            for value in in_samp:
                self.in_taps = np.append(value, self.in_taps[0:-1])
                out_samp = (sum(self.in_taps[:] * self.b[:]) - sum(self.out_taps[:] * self.a[1:])) / self.a[0]
                self.out_taps = np.append(out_samp, self.out_taps[0:-1])
                output_list.append(out_samp)
            out_samp = output_list
        else:
            self.in_taps = np.append(in_samp, self.in_taps[0:-1])
            out_samp = (sum(self.in_taps[:] * self.b[:]) - sum(self.out_taps[:] * self.a[1:])) / self.a[0]
            self.out_taps = np.append(out_samp, self.out_taps[0:-1])

        # self.in_taps = np.append(in_samp, self.in_taps[0:-1])
        # out_samp = (sum(self.in_taps[:] * self.b[:]) - sum(self.out_taps[:] * self.a[1:])) / self.a[0]
        # self.out_taps = np.append(out_samp, self.out_taps[0:-1])

        return out_samp
