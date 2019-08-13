import math
import cmath
import scipy.constants as consts
import numpy as np
import matplotlib.pyplot as plt


class PhaseNoise(object):
    """
    This class models a phase-noise source for any part of the analog circuit that uses a local-oscillator.
    """

    # constructor
    def __init__(self, freq=None, phase_noise=None, fs=1, delta_f=None):

        """
        It requires four parameters to construct:

        :param        freq: float,  A list of frequency offsets from carrier (specified in Hz) at which the strength of the
                                    phase-noise is specified.
                                    If not provided, it defaults to the range :math:`10^{-7}`
                                    Hz to 0.5 Hz in steps of a decade

        :param phase_noise: float,  A list of values representing the strength of phase noise at the specified
                                    frequency offsets in :math:`\\frac{dBc}{Hz}`.
                                    If not provided, it defaults to the
                                    value of :math:`-50 \\frac{dBc}{Hz}` at :math:`10^{-7}` Hz and drops at a rate of
                                    20 dB/decade till 0.5 Hz

        :param          fs: float,  Sampling frequency in Hz.  If not provided, it defaults to 1 Hz.
        :param     delta_f: float,  Frequency resolution in Hz at which to generate phase-noise.  If not provided, it
                                    defaults to a value less than the first non-zero frequency offset that was specified.

        """
        self.freq = freq
        self.n_fft = int(4096)
        """
        int
        
        Number of frequency points at which phase-noise is generated in frequency domain.
        """

        self.delta_freq = 1.0 / 1024.0
        """
        float
        
        Frequency resolution at which phase noise is defined.
        """

        # number of samples of phase-noise left
        self._num_samples_left = self.n_fft

        if (freq is None) and (phase_noise is None):
            freq = [0, 1e-3, 1e-2, 1e-1, 5e-1]
            phase_noise = [-50, -50, -70, -90, -103.9794]

        # Convert frequency from Hz to digital frequency
        self.Fs = fs
        self.FFT = np.array([])
        self.IFFT = np.array([])
        """
        float
        
        Sampling frequency
        """

        freq_dig = []
        for f in freq:
            freq_dig.append(f / fs)

        # Interpolate phase-noise profile to linearly & equally spaced frequency points
        freq_resp = self._interpolate_to_linear(freq_dig, phase_noise, delta_f)

        self.freq = freq_resp['freq']
        self.delta_f = freq_resp['delta_f']
        """
        list of floats
        
        Frequency offsets from carrier where phase-noise is defined.  This list of frequencies is equally spaced from 0 to :math:`\\frac{Fs}{2}`.
        
        """

        self.phase_noise = freq_resp['phase_noise']
        """
        list of floats

        Phase noise at above frequency-offsets specified in :math:`\\frac{dBc}{Hz}`.

        """

        self.noise_samples = np.zeros(self.n_fft)
        """
        list of complex floats
        
        Current set of phase noise samples.
        """

        # generate the first set of noise samples and keep it ready
        self.gen_new_sample_set()

    def _interpolate_to_linear(self, freq, phase_noise, delta_f=None):
        """
        Interpolate
        """

        if freq[0] == 0:
            freq[0] = np.spacing(1)

        # Determine frequency spacing such that it is no greater than delta_freq (if delta_freq provided) or no greater
        # than the difference between the first two specified frequency points (if delta_freq is not provided) with the
        # additional constraint that Nfft is a power to 2.
        if delta_f is None:
            delta_f = freq[1] - freq[0]

        # Quantize to a value such that Nfft becomes a power of two
        n_fft = (1.0 / delta_f) + 1
        n_fft_log = math.ceil(math.log(n_fft, 2))
        self.n_fft = int(2 ** n_fft_log)
        if self.n_fft < 1024:
            self.n_fft = 1024
        self.delta_freq = 1.0 / self.n_fft

        lin_freq = np.linspace(0, 0.5 - self.delta_freq, self.n_fft / 2)
        lin_freq[0] = np.spacing(1)

        lin_phase_noise = np.empty(self.n_fft // 2)
        spec_idx = 0
        cur_slope = 0.0
        idx = 0
        while idx < self.n_fft / 2:
            if idx == 0:
                lin_phase_noise[idx] = phase_noise[0]
                idx += 1
                continue

            # update slope if we have gone past the next specified frequency
            # retain final slope if we have gone past the last specified frequency
            if (lin_freq[idx] > freq[spec_idx]) and (spec_idx < (len(freq) - 1)):
                spec_idx += 1
                cur_slope = (phase_noise[spec_idx] - phase_noise[spec_idx - 1]) / (
                    math.log10(freq[spec_idx]) - math.log10(freq[spec_idx - 1]))

            # compute new phase-noise sample based on the slope of the specified phase-noise
            if cur_slope == -0.0:
                lin_phase_noise[idx] = phase_noise[spec_idx - 1]
            else:
                lin_phase_noise[idx] = phase_noise[spec_idx - 1] + cur_slope * \
                                                               (math.log10(lin_freq[idx]) -
                                                                math.log10(freq[spec_idx - 1]))

            # increment linear frequency index
            idx += 1

        lin_freq[0] = 1.0/self.Fs
        lin_freq_full = np.append(lin_freq, lin_freq + 0.5)
        lin_phase_noise_full = np.append(lin_phase_noise, np.flipud(lin_phase_noise))
        return {'freq': lin_freq, 'phase_noise': lin_phase_noise, 'delta_f': delta_f}

    def gen_new_sample_set(self):
        # sigma = math.sqrt(float(self.Fs / self.n_fft))
        # sigma = math.sqrt(float(self.Fs / self.n_fft))
        sigma = float(self.n_fft / 2)
        white_noise_freq_samples = np.random.normal(0, sigma, self.n_fft // 2) + 1j * np.random.normal(0, sigma,
                                                                                                      self.n_fft // 2)
        colored_noise_freq_samples = (10 ** ((self.phase_noise + 0.0) / 20.0)) * white_noise_freq_samples
        colored_noise_freq_samples[0] = 1.0
        # colored_noise_freq_samples_full = np.append(colored_noise_freq_samples, np.flipud(np.conj(colored_noise_freq_samples)))
        colored_noise_freq_samples_full = np.append(colored_noise_freq_samples, 1.0)

        self.FFT = colored_noise_freq_samples_full
        #colored_noise_freq_samples_full_real = colored_noise_freq_samples_full.real
        colored_noise_time_samples = np.fft.irfft(colored_noise_freq_samples_full)
        self.IFFT = colored_noise_time_samples
        # plt.figure()
        # plt.plot(20*np.log10(np.abs(colored_noise_freq_samples_full)))
        # plt.show()
        # plt.waitforbuttonpress
        colored_phase_time_samples = np.real(colored_noise_time_samples)
        # plt.figure()
        # plt.plot(colored_phase_time_samples)
        self.noise_samples = np.exp(1j * colored_phase_time_samples)
        # for i in range(len(self.noise_samples)):
        #    print 'Phase: ' + str(colored_noise_time_samples[i]) + ', Phase Noise: ' + str(self.noise_samples[i])
        self._num_samples_left = self.n_fft

    def add_noise(self, v_in=np.complex(1.0, 0.0)):
        """
        Add noise to one input sample.

        :param v_in: complex, Input sample (defaults to 0 if not provided)
        :return: complex -- Output sample with noise added
        """

        v_out = v_in * self.noise_samples[-self._num_samples_left]

        self._num_samples_left -= 1
        if self._num_samples_left == 0:
            self.gen_new_sample_set()

        return v_out

    def gen_noise(self, num_samples):
        """
        Generate an array of noise samples.

        :param num_samples: int, Number of noise samples to be generated
        :return: numpy.array -- A 1D array of length num_samples containing the noise samples
        """

        noise = np.empty(0)
        num_times = math.ceil(num_samples / self.n_fft)
        last_samples = num_samples % self.n_fft
        # num_times = num_samples
        # last_samples = num_samples
        for t in range(num_times - 1):
            self.gen_new_sample_set()
            noise = np.append(noise, self.noise_samples)

        self.gen_new_sample_set()
        noise = np.append(noise, self.noise_samples[0:last_samples])
        self._num_samples_left -= last_samples

        return noise