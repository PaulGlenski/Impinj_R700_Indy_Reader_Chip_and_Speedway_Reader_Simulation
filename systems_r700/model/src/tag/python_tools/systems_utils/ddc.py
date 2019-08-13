import numpy as np
from python_tools.systems_utils.rx_waveforms import get_unity_norm_scaler
from scipy import signal as ss
import matplotlib.pyplot as plt


class DdcClass(object):
    """
    Dual downconverter class
    """

    def __init__(self,
                 x=0,
                 config=0,
                 # link=None,
                 # phase_rad=0,
                 plot_fig=False
                 ):
        self.mval = (1 if config.link_dict['mod_type'] == '3b4b_baseband' else
                     config.link_dict['m'])
        self.mod_type = config.link_dict['mod_type']
        #TODO:
        self.x = x
        self.lpf_fcut = 2.0 / (4 * self.mval)
        self.blf_hz = config.link_dict['blf_hz']
        self.fsample_hz = config.link_dict['fsample_hz']
        self.fblf_norm =self.blf_hz / self.fsample_hz
        self.phase_rad = config.link_dict['theta_rf_rad']
        self.phase_bits = None
        #TODO:
        self.pts = len(x)
        self.h = np.zeros(64)
        self.ddc_up = np.zeros(self.pts)
        self.ddc_dn = np.zeros(self.pts)
        self.N = 64  # Number of DDC image filter taps
        self.plot_fig = plot_fig
        self.bypass_ddc = False
        #TODO:
        # if self.fsample_hz == 20.48e6:
        #     self.x = x[::4]
        #     self.pts = len(x[::4])
        #     self.ddc_up = np.zeros(self.pts)
        #     self.ddc_dn = np.zeros(self.pts)

    def execute(self):
        """Main body of DDC execution
        Mval of 1 is bypass mode in DDC"""
        self._find_bypass_state()
        #if self.mval == 1 and '3b4b_subcarrier' not in self.mod_type:
        if self.bypass_ddc:
            self.ddc_up = self.x
            self.ddc_dn = np.zeros(len(self.x))
        else:
            self.lo_up = self._create_blf_lo(pol=1)
            self.lo_dn = self._create_blf_lo(pol=-1)
            self._dual_mixer()
            self._ddc_filter()

        # Decimate
        self.ddc_up_decimate = self.ddc_up[0::self.mval].copy()
        self.ddc_dn_decimate = self.ddc_dn[0::self.mval].copy()

        # Normalize
        norm_up = get_unity_norm_scaler(self.ddc_up_decimate)
        norm_dn = get_unity_norm_scaler(self.ddc_dn_decimate)
        norm = np.min([norm_up, norm_dn])
        self.ddc_up_decimate *= norm
        self.ddc_dn_decimate *= norm
        self.n_out_smpls = len(self.ddc_up_decimate)

        if self.plot_fig and bool(self.bypass_ddc) is False:
            self._plot_freq_response()

    def _find_bypass_state(self):
        if self.mval == 1:
            if '3b4b_subcarrier' not in self.mod_type:
                self.bypass_ddc = True
            if 'bpsk' in self.mod_type:
                self.bypass_ddc = True

        #TODO: Original _find_bypass_state is below
            #if '3b4b_subcarrier' not in self.mod_type or 'bpsk' not in self.mod_type:
                #self.bypass_ddc = True


    def _create_blf_lo(self, pol):
        """
        Return a complex exponential x(t) = exp( j*2*pi*f_blf*t)

            where t = n(Ti)
                    = n/fs  where fs=drc output sample rate

                  Therefore f_blf*t = n*(f_blf / fs)
                                    = n*fblf_norm
                                    = n / drc_out_smpls_per_blf

            x(n) = exp(j*2*pi*n / drc_out_smpls_per_blf)

        Args:
            pts: Length of array to return
            pol: 1=mix up, -1=mix dn
            mxr_phs_rad: mixer phase offset in radians (default=0)
            mxr_phs_bits: phase wordsize

        """
        n = np.arange(self.pts)
        if self.phase_bits is not None:
            # subtracting 3 bits from mixer_phs_bits accounts for the
            # divide by 8 applied to the nom_fcw in yk_rx_wvfm.py
            fxd_shift = 2 ** (self.phase_bits - 3)
            self.fblf_norm = np.round(self.fblf_norm * fxd_shift) / fxd_shift
        lo_phs = (2 * np.pi * n * self.fblf_norm) + self.phase_rad
        return np.exp(pol * 1j * lo_phs)

    def filter_data(self, x):
        return np.convolve(x, self.h)[int(self.N / 2) - 1:int(-self.N / 2)]

    def _get_filter_coeff(self):
        """Get the filter pass and stop bands for the DDC image filter"""

        ft = 0.15  # Frequency tolerance
        fc = self.blf_hz*(1+ft)/self.mval
        if '3b4b_subcarrier' in self.mod_type and self.mval == 1:
            fc = 0.7*fc  # BW of Baseband 3b4b signal
        tag = self.blf_hz*(1-ft)
        lo_freq = self.blf_hz + tag*(1-1./self.mval)
        hi_freq = self.blf_hz + tag*(1+1./self.mval)

        bands = np.array([0., fc, lo_freq, hi_freq, hi_freq + 2,
                          self.fsample_hz / 2.]) / self.fsample_hz
        gain = [1, 0, 0]
        weight = [1, 1, 1]

        self.h = ss.remez(self.N, bands=bands, desired=gain, weight=weight)

    def _dual_mixer(self):
        """
        Create y_up and y_dn complex signals
        """
        self.y_mix_up = np.multiply(self.lo_up, self.x)
        self.y_mix_dn = np.multiply(self.lo_dn, self.x)

    def _ddc_filter(self):
        # The sample rate is 8*M
        # fcut = 2.0 * fscale / (2 * self.mval)

        # self.ddc_filt = FirFilterClass(ntaps=79,
        #                                fcut=self.lpf_fcut,
        #                                pass_zero=True)

        self._get_filter_coeff()
        # self.ddc_up = self.ddc_filt.filter_data(self.y_mix_up)
        # self.ddc_dn = self.ddc_filt.filter_data(self.y_mix_dn)
        self.ddc_up = self.filter_data(self.y_mix_up)
        self.ddc_dn = self.filter_data(self.y_mix_dn)

    def _plot_freq_response(self):
        fs = self.fsample_hz
        [w, h] = ss.freqz(b=self.h, a=1, worN=1024, whole=True)
        fig, ax = plt.subplots()
        w_khz = w * fs / (2 * np.pi) / 1e3
        habs = np.abs(h)
        hmax = np.max(habs)
        hmag = 20. * np.log10(habs*1.0/hmax)
        wf = np.fft.fftshift(w_khz)
        wf[wf >= fs/2.e3] = wf[wf >= fs/2.e3] - fs / 1.e3
        ax.plot(wf, hmag, linewidth=2)
        ax.set_xlabel("frequency in khz")
        ax.set_ylabel('Magnitude in dB')
        ax.set_title('DDC Image filter frequency response')
        ax.grid(True)
        plt.show()
