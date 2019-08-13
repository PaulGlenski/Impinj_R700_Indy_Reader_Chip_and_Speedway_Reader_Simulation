import math
import cmath
import scipy.constants as consts
import numpy as np


class Awgn(object):
    """
    This class models a thermal noise source at any point in the analog circuit.
    """

    # constructor
    def __init__(self, temperature=20, nf=0.0, bw=0.5, load=50, is_complex=True):

        """
        It requires five parameters to construct:

        :param temperature: float, Ambient temperature in :math:`^{\circ}C` (defaults to :math:`20^{\circ}C` if not provided)
        :param          nf: float, Noise figure in dB at the point of injection (defaults to 0dB if not provided)
        :param          bw: float, Bandwidth of signal in Hz (defaults to 0.5 Hz if not provided)
        :param        load: float, Load impedance in :math:`\Omega` (defaults to 50 :math:`\Omega` if not provided)
        :param  is_complex: bool, Selects complex vs real noise source (defaults to True if not selected)

        """

        incident_n0 = consts.Boltzmann * consts.convert_temperature(float(temperature), 'C', 'K')
        incident_n = incident_n0 * float(2*bw)
        n = incident_n * (10.0 ** (float(nf)/10.0))

        self.v_rms = math.sqrt(n * float(load))
        """
        float
        
        RMS value of noise voltage as if the signal was a real RF signal.  The RMS value of each component of a 
        complex baseband noise sample is :math:`\\frac{1}{\\sqrt{2}}` times RMS value.
        """

        self.is_complex = is_complex
        """
        bool
        
        State that determines real vs complex noise samples
        """

    def set_noise_spectral_density(self, spectral_density, fs, is_complex=True):
        """

        :param spectral_density: float, Noise spectral density expressed in :math:`\frac{nV}{\sqrt{Hz}}`
        :param fs: float, Sampling rate in Hz at which the noise samples are generated
        :param is_complex: boolean, Indicates whether the noise samples should be complex or real
        """

        self.v_rms = spectral_density * 1e-9 * math.sqrt(fs)
        self.is_complex = is_complex

    def add_noise(self, v_in=0.0):
        """
        Add noise to one input sample.

        :param v_in: float/complex, Input sample (defaults to 0 if not provided)
        :return: real/complex -- Output sample with noise added
        """
        if(self.is_complex):
            v_out = complex(v_in) + \
                    (np.random.normal(0.0, self.v_rms/math.sqrt(2)) + 1j * np.random.normal(0.0, self.v_rms/math.sqrt(2)))
        else:
            v_out = v_in + np.random.normal(0.0, self.v_rms/math.sqrt(2))

        return v_out

    def gen_noise(self, num_samples):
        """
        Generate an array of noise samples.

        :param num_samples: int, Number of noise samples to be generated
        :return: numpy.array -- A 1D array of length num_samples containing the noise samples
        """

        if (self.is_complex):
            noise = np.random.normal(0.0, self.v_rms / math.sqrt(2), num_samples) + \
                    1j * np.random.normal(0.0, self.v_rms / math.sqrt(2), num_samples)
        else:
            noise = np.random.normal(0.0, self.v_rms, num_samples)

        return noise
